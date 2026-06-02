import torch
from torch import nn


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return self.relu(out)


class ResNetBackbone(nn.Module):
    def __init__(self, in_channels, img_size):
        super().__init__()
        stem_stride = 1 if img_size <= 32 else 2
        self.stem = nn.Sequential(
            nn.Conv2d(
                in_channels,
                64,
                kernel_size=3,
                stride=stem_stride,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.layer1 = self._make_layer(64, 64, stride=1)
        self.layer2 = self._make_layer(64, 128, stride=2)
        self.layer3 = self._make_layer(128, 256, stride=2)
        self.layer4 = self._make_layer(256, 512, stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.feat_dim = 512

    @staticmethod
    def _make_layer(in_channels, out_channels, stride):
        return nn.Sequential(
            BasicBlock(in_channels, out_channels, stride=stride),
            BasicBlock(out_channels, out_channels, stride=1),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        return x.flatten(1)


class ExpertFFN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


class TopKGating(nn.Module):
    def __init__(self, in_dim, num_experts, top_k):
        super().__init__()
        if top_k <= 0 or top_k > num_experts:
            raise ValueError("top_k must be between 1 and num_experts")

        self.top_k = top_k
        self.gate = nn.Linear(in_dim, num_experts, bias=False)

    def forward(self, x):
        probs = torch.softmax(self.gate(x).float(), dim=-1)
        topk_values, topk_indices = probs.topk(self.top_k, dim=-1)

        weights = torch.zeros_like(probs)
        weights.scatter_(1, topk_indices, topk_values)
        return weights.to(x.dtype), topk_indices


class MoELayer(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_experts, top_k):
        super().__init__()
        self.gating = TopKGating(in_dim, num_experts, top_k)
        self.experts = nn.ModuleList([
            ExpertFFN(in_dim, hidden_dim, out_dim)
            for _ in range(num_experts)
        ])
        self.out_dim = out_dim

    def forward(self, x):
        weights, topk_indices = self.gating(x)
        out = x.new_zeros((x.size(0), self.out_dim))

        # 一次性取回活跃 expert，避免逐 expert 的 .any() 触发多次 GPU 同步。
        selected_counts = torch.bincount(
            topk_indices.reshape(-1),
            minlength=len(self.experts),
        )
        active_expert_ids = torch.nonzero(selected_counts, as_tuple=False).flatten().tolist()
        for expert_id in active_expert_ids:
            expert = self.experts[expert_id]
            token_mask = (topk_indices == expert_id).any(dim=-1)
            expert_out = expert(x[token_mask])
            selected_weights = weights[token_mask, expert_id]
            out[token_mask] += expert_out * selected_weights.unsqueeze(-1)

        return out


class ResNetSparseMoEHead(nn.Module):
    def __init__(
        self,
        in_channels=3,
        num_classes=10,
        img_size=32,
        num_experts=4,
        top_k=2,
    ):
        super().__init__()
        self.backbone = ResNetBackbone(in_channels, img_size)
        self.moe_head = MoELayer(
            in_dim=self.backbone.feat_dim,
            hidden_dim=512,
            out_dim=num_classes,
            num_experts=num_experts,
            top_k=top_k,
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.moe_head(features)
