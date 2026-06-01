import torch
from torch import nn


class SimpleCNNBackbone(nn.Module):
    def __init__(self, in_channels=3, img_size=32, cnn_channels=None):
        super().__init__()
        _ = img_size
        cnn_channels = [32, 64, 128] if cnn_channels is None else list(cnn_channels)
        if len(cnn_channels) != 3:
            raise ValueError("cnn_channels must contain exactly three channel sizes")

        blocks = []
        current_channels = in_channels
        for out_channels in cnn_channels:
            blocks.append(
                nn.Sequential(
                    nn.Conv2d(
                        current_channels,
                        out_channels,
                        kernel_size=3,
                        padding=1,
                        bias=False,
                    ),
                    nn.GroupNorm(1, out_channels),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),
                )
            )
            current_channels = out_channels

        self.blocks = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.feature_dim = cnn_channels[-1]

    def forward(self, x):
        x = self.blocks(x)
        x = self.pool(x)
        return x.flatten(1)


class ExpertFFN(nn.Module):
    def __init__(self, feature_dim, expert_hidden_dim, num_classes, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(feature_dim, expert_hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(expert_hidden_dim, num_classes)

    def forward(self, x):
        return self.fc2(self.dropout(self.relu(self.fc1(x))))


class SimpleMoEHead(nn.Module):
    def __init__(
        self,
        feature_dim,
        num_classes,
        num_experts,
        top_k=1,
        expert_hidden_dim=256,
        dropout=0.0,
    ):
        super().__init__()
        if top_k != 1:
            raise ValueError("SimpleCNNMoEHead only supports top_k=1 for now")

        self.num_classes = num_classes
        self.num_experts = num_experts
        self.router = nn.Linear(feature_dim, num_experts, bias=False)
        self.experts = nn.ModuleList([
            ExpertFFN(feature_dim, expert_hidden_dim, num_classes, dropout)
            for _ in range(num_experts)
        ])

    def forward(self, features):
        router_logits = self.router(features)
        router_probs = torch.softmax(router_logits.float(), dim=-1)
        top1_probs, top1_indices = router_probs.max(dim=-1)

        logits = features.new_zeros((features.size(0), self.num_classes))
        for expert_id, expert in enumerate(self.experts):
            sample_mask = top1_indices == expert_id
            if not sample_mask.any():
                continue

            selected_prob = top1_probs[sample_mask]
            hard_gate = selected_prob + (1.0 - selected_prob).detach()
            logits[sample_mask] = expert(features[sample_mask]) * hard_gate.unsqueeze(-1)

        selected_counts = torch.bincount(top1_indices, minlength=self.num_experts)
        expert_activations = selected_counts
        overflow_counts = torch.zeros_like(selected_counts)
        avg_router_probs = router_probs.detach().mean(dim=0)
        zero_tensor = features.new_tensor(0.0)
        layer_stats = {
            "expert_activations": expert_activations,
            "selected_counts": selected_counts,
            "overflow_counts": overflow_counts,
            "avg_router_probs": avg_router_probs,
            "capacity": 0,
        }

        return {
            "logits": logits,
            "aux_loss": zero_tensor,
            "router_aux_loss": zero_tensor,
            "router_z_loss": zero_tensor,
            "total_router_loss": zero_tensor,
            "expert_activations": expert_activations,
            "expert_activations_summary": expert_activations,
            "selected_counts_summary": selected_counts,
            "overflow_counts_summary": overflow_counts,
            "avg_router_probs": avg_router_probs,
            "expert_stats_by_layer": {"moe_head": layer_stats},
            "expert_activations_by_layer": {"moe_head": expert_activations},
            "selected_counts_by_layer": {"moe_head": selected_counts},
            "overflow_counts_by_layer": {"moe_head": overflow_counts},
            "avg_router_probs_by_layer": {"moe_head": avg_router_probs},
            "capacity_by_layer": {"moe_head": 0},
        }


class SimpleCNNMoEHead(nn.Module):
    def __init__(
        self,
        in_channels=3,
        num_classes=10,
        img_size=32,
        num_experts=4,
        top_k=1,
        cnn_channels=None,
        expert_hidden_dim=256,
        dropout=0.0,
    ):
        super().__init__()
        self.backbone = SimpleCNNBackbone(
            in_channels=in_channels,
            img_size=img_size,
            cnn_channels=cnn_channels,
        )
        self.moe_head = SimpleMoEHead(
            feature_dim=self.backbone.feature_dim,
            num_classes=num_classes,
            num_experts=num_experts,
            top_k=top_k,
            expert_hidden_dim=expert_hidden_dim,
            dropout=dropout,
        )

    def forward(self, x):
        features = self.backbone(x)
        result = self.moe_head(features)
        result["feature"] = features
        return result
