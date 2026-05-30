import math

import torch
import torch.nn.functional as F
from torch import nn


class DenseFFN(nn.Module):
    # 普通 Transformer FFN，不使用 MoE。
    # 结构是 Linear -> GELU -> Dropout -> Linear -> Dropout。
    def __init__(self, embed_dim, mlp_ratio=4.0, dropout_rate=0.1):
        super(DenseFFN, self).__init__()

        # FFN 中间层维度，通常是 embed_dim 的 4 倍。
        hidden_dim = int(embed_dim * mlp_ratio)

        # 标准两层 MLP 前馈网络。
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x):
        # 输入输出形状保持一致：[batch_size, num_tokens, embed_dim]。
        return self.net(x)


class SwitchFFNExpert(nn.Module):
    # Switch Transformer 里的单个 expert。
    # 每个 expert 本质上也是一个 FFN，只是不同 token 会被 router 分配到不同 expert。
    def __init__(self, embed_dim, hidden_dim, dropout_rate=0.1):
        super(SwitchFFNExpert, self).__init__()

        # 单个 expert 的前馈网络结构。
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x):
        # x 是被分配到当前 expert 的 token。
        return self.net(x)


class TokenSwitchFFN(nn.Module):
    # Token 级别的 Switch FFN。
    # 每个 token 经过 router，选择 top-1 expert 进行处理。
    def __init__(
        self,
        embed_dim,
        num_experts,
        mlp_ratio=4.0,
        dropout_rate=0.1,
        router_jitter_noise=0.0,
        capacity_factor=1.25,
        min_capacity=4,
        drop_tokens=True,
        top_k=1,
    ):
        super(TokenSwitchFFN, self).__init__()

        # 当前实现只支持 top-1 routing。
        if top_k != 1:
            raise ValueError("TokenSwitchFFN currently supports top_k=1 only")

        self.embed_dim = embed_dim
        self.num_experts = num_experts

        # router_jitter_noise 用于训练时给 router 输入加轻微扰动，增加 routing 随机性。
        self.router_jitter_noise = router_jitter_noise

        # capacity_factor 控制每个 expert 最多接收多少 token。
        self.capacity_factor = capacity_factor

        # 每个 expert 的最小容量。
        self.min_capacity = min_capacity

        # 如果某个 expert 分到的 token 超过容量，是否丢弃超出的 token。
        self.drop_tokens = drop_tokens

        # expert 内部 FFN 的 hidden 维度。
        hidden_dim = int(embed_dim * mlp_ratio)

        # router 根据 token embedding 输出每个 expert 的概率。
        self.router = nn.Linear(embed_dim, num_experts)

        # 多个 expert，每个 expert 都是一个独立的 FFN。
        self.experts = nn.ModuleList([
            SwitchFFNExpert(embed_dim, hidden_dim, dropout_rate)
            for _ in range(num_experts)
        ])

    def forward(self, x):
        # x 形状：[batch_size, num_tokens, embed_dim]。
        batch_size, num_tokens, embed_dim = x.shape

        # 默认 router 输入就是 token 本身。
        router_input = x

        # 训练阶段可选地加入 jitter noise，增强 router 的探索性。
        if self.training and self.router_jitter_noise > 0:
            noise = torch.empty_like(router_input).uniform_(
                1.0 - self.router_jitter_noise,
                1.0 + self.router_jitter_noise,
            )
            router_input = router_input * noise

        # router_logits 形状：[batch_size, num_tokens, num_experts]。
        router_logits = self.router(router_input)

        # 将 logits 转成 expert 概率。
        router_probs = F.softmax(router_logits.float(), dim=-1).to(x.dtype)

        # top-1 routing：每个 token 选择概率最大的 expert。
        top1_probs, top1_indices = torch.max(router_probs, dim=-1)

        # 将 token 展平成二维，方便按 expert 收集 token。
        flat_x = x.reshape(batch_size * num_tokens, embed_dim)

        # flat_output 用来保存每个 token 经过 expert 后的输出。
        flat_output = torch.zeros_like(flat_x)

        # flat_indices 表示每个 token 被分配到哪个 expert。
        flat_indices = top1_indices.reshape(-1)

        # flat_top1_probs 表示每个 token 被分配到目标 expert 的 router 概率。
        flat_top1_probs = top1_probs.reshape(-1)

        # 当前 batch 的 token 总数。
        total_tokens = max(batch_size * num_tokens, 1)

        # 计算每个 expert 的容量上限。
        # capacity_factor 越大，每个 expert 能接收的 token 越多。
        capacity = max(
            self.min_capacity,
            math.ceil(self.capacity_factor * total_tokens / self.num_experts),
        )

        # selected_counts 表示每个 expert 被 router 选中的 token 数。
        # 注意：这是被选中的数量，不一定等于最终实际处理的数量，因为可能发生 overflow。
        selected_counts = torch.bincount(
            flat_indices,
            minlength=self.num_experts,
        ).to(x.device)

        # expert_activations 记录每个 expert 实际处理的 token 数。
        expert_activations = torch.zeros(self.num_experts, device=x.device, dtype=torch.long)

        # overflow_counts 记录每个 expert 超出 capacity 的 token 数。
        overflow_counts = torch.zeros(self.num_experts, device=x.device, dtype=torch.long)

        # 逐个 expert 处理分配给自己的 token。
        for expert_id, expert in enumerate(self.experts):
            # 清理上一轮 forward 临时挂载的 sample id，避免 stale state。
            if hasattr(expert, "_fedwolf_accepted_sample_ids"):
                delattr(expert, "_fedwolf_accepted_sample_ids")

            # 找到当前 expert 被分配到的 token 位置。
            token_positions = torch.nonzero(flat_indices == expert_id, as_tuple=False).flatten()

            # 当前 expert 没有 token，则跳过。
            if token_positions.numel() == 0:
                continue

            # 计算当前 expert 超出容量的 token 数。
            overflow_count = max(token_positions.numel() - capacity, 0)
            overflow_counts[expert_id] = overflow_count

            # 如果 drop_tokens=True，则只保留 capacity 范围内的 token。
            # 否则当前 expert 处理所有分配到的 token。
            if self.drop_tokens:
                accepted_positions = token_positions[:capacity]
            else:
                accepted_positions = token_positions

            # 记录当前 expert 实际处理的 token 数。
            expert_activations[expert_id] = accepted_positions.numel()

            # 当前 expert 对分配给它的 token 做 FFN 计算。
            if accepted_positions.numel() > 0:
                expert._fedwolf_accepted_sample_ids = (accepted_positions // num_tokens).detach()
                expert_output = expert(flat_x[accepted_positions])

                # Switch Transformer 中通常会用 router probability 对 expert 输出做缩放。
                flat_output[accepted_positions] = (
                    expert_output * flat_top1_probs[accepted_positions].unsqueeze(-1)
                )

        # 恢复成原始 token 形状。
        output = flat_output.reshape(batch_size, num_tokens, embed_dim)

        # 每个 expert 被选中的 token 比例，用于 router auxiliary loss。
        usage_fraction = selected_counts.float() / float(total_tokens)

        # 每个 expert 的平均 router 概率。
        avg_router_probs = router_probs.float().mean(dim=(0, 1))

        # router_aux_loss 用于鼓励不同 expert 使用更均衡。
        router_aux_loss = self.num_experts * torch.sum(usage_fraction.detach() * avg_router_probs)

        # router_z_loss 用于约束 router logits，避免 logits 过大。
        router_z_loss = torch.mean(torch.logsumexp(router_logits.float(), dim=-1) ** 2)

        # 返回 hidden 和各种 expert/router 统计信息。
        return {
            "hidden": output,
            "router_aux_loss": router_aux_loss,
            "router_z_loss": router_z_loss,
            "expert_activations": expert_activations,
            "selected_counts": selected_counts,
            "overflow_counts": overflow_counts,
            "capacity": capacity,
            "avg_router_probs": avg_router_probs,
        }


class TransformerBlock(nn.Module):
    # Transformer Block：
    # 包含 Multi-Head Attention + FFN。
    # FFN 可以是普通 DenseFFN，也可以是 TokenSwitchFFN。
    def __init__(
        self,
        embed_dim,
        num_heads,
        mlp_ratio=4.0,
        dropout_rate=0.1,
        use_switch_ffn=False,
        num_experts=8,
        router_jitter_noise=0.0,
        capacity_factor=1.25,
        min_capacity=4,
        drop_tokens=True,
        top_k=1,
        layer_id=0,
    ):
        super(TransformerBlock, self).__init__()

        # 当前 block 的层号，用于按层记录 expert 统计。
        self.layer_id = layer_id

        # 是否在当前 block 使用 Switch FFN。
        self.use_switch_ffn = use_switch_ffn

        # Attention 前的 LayerNorm。
        self.norm1 = nn.LayerNorm(embed_dim)

        # 多头自注意力，batch_first=True 表示输入形状是 [B, N, C]。
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True,
        )

        self.dropout = nn.Dropout(dropout_rate)

        # FFN 前的 LayerNorm。
        self.norm2 = nn.LayerNorm(embed_dim)

        if use_switch_ffn:
            # MoE / Switch FFN 分支。
            self.ffn = TokenSwitchFFN(
                embed_dim=embed_dim,
                num_experts=num_experts,
                mlp_ratio=mlp_ratio,
                dropout_rate=dropout_rate,
                router_jitter_noise=router_jitter_noise,
                capacity_factor=capacity_factor,
                min_capacity=min_capacity,
                drop_tokens=drop_tokens,
                top_k=top_k,
            )
        else:
            # 普通 dense FFN 分支。
            self.ffn = DenseFFN(
                embed_dim=embed_dim,
                mlp_ratio=mlp_ratio,
                dropout_rate=dropout_rate,
            )

    def forward(self, x):
        # Pre-LN Transformer：先 norm，再 attention。
        norm_x = self.norm1(x)

        # 自注意力：Q/K/V 都来自 norm_x。
        attention_out, _ = self.attention(norm_x, norm_x, norm_x, need_weights=False)

        # Attention 残差连接。
        x = x + self.dropout(attention_out)

        # FFN 前再做一次 LayerNorm。
        ffn_input = self.norm2(x)

        if self.use_switch_ffn:
            # 如果当前层是 MoE 层，则 FFN 返回 hidden 和 router/expert 统计。
            switch_result = self.ffn(ffn_input)

            # Switch FFN 残差连接。
            x = x + self.dropout(switch_result["hidden"])

            # 返回当前层输出，以及当前层的 expert 统计。
            return x, {
                "layer_id": self.layer_id,
                "router_aux_loss": switch_result["router_aux_loss"],
                "router_z_loss": switch_result["router_z_loss"],
                "expert_activations": switch_result["expert_activations"],
                "selected_counts": switch_result["selected_counts"],
                "overflow_counts": switch_result["overflow_counts"],
                "capacity": switch_result["capacity"],
                "avg_router_probs": switch_result["avg_router_probs"],
            }

        # 普通 FFN 层只返回 hidden，不返回 expert 统计。
        x = x + self.dropout(self.ffn(ffn_input))
        return x, None


def make_resnet18_group_norm(num_channels, max_groups=32):
    # 为 ResNet block 创建 GroupNorm。
    # 从 max_groups 往下找，选择能整除 num_channels 的最大 group 数。
    for num_groups in range(min(max_groups, num_channels), 0, -1):
        if num_channels % num_groups == 0:
            return nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)

    # 理论上一定会找到 num_groups=1，这里作为兜底。
    return nn.GroupNorm(num_groups=1, num_channels=num_channels)


class ResNet18BasicBlock(nn.Module):
    # ResNet-18 的基础残差块。
    # 包含两个 3x3 卷积和一条 shortcut 分支。
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNet18BasicBlock, self).__init__()

        # 第一个 3x3 卷积，可能负责下采样。
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.norm1 = make_resnet18_group_norm(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # 第二个 3x3 卷积，stride 固定为 1。
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.norm2 = make_resnet18_group_norm(out_channels)

        # 如果通道数或空间尺寸发生变化，需要用 1x1 卷积调整 shortcut。
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                make_resnet18_group_norm(out_channels),
            )
        else:
            # 如果输入输出形状一致，shortcut 直接恒等映射。
            self.shortcut = nn.Identity()

    def forward(self, x):
        # 主分支：conv -> norm -> relu -> conv -> norm。
        out = self.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))

        # 残差连接。
        out = out + self.shortcut(x)

        # 残差相加后再经过 ReLU。
        return self.relu(out)


class ResNet18Backbone(nn.Module):
    """CIFAR-sized ResNet-18 feature extractor without a classification head."""

    def __init__(self):
        super(ResNet18Backbone, self).__init__()

        # CIFAR 图像较小，所以 stem 使用 3x3 conv，stride=1，不使用 ImageNet ResNet 的 7x7 conv + maxpool。
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            make_resnet18_group_norm(64),
            nn.ReLU(inplace=True),
        )

        # ResNet-18 四个 stage。
        # stage1 不下采样，后面每个 stage 的第一个 block 用 stride=2 下采样。
        self.stage1 = self._make_stage(64, 64, num_blocks=2, stride=1)
        self.stage2 = self._make_stage(64, 128, num_blocks=2, stride=2)
        self.stage3 = self._make_stage(128, 256, num_blocks=2, stride=2)
        self.stage4 = self._make_stage(256, 512, num_blocks=2, stride=2)

        # ResNet backbone 最终输出通道数。
        self.out_channels = 512

    def _make_stage(self, in_channels, out_channels, num_blocks, stride):
        # 每个 stage 的第一个 block 负责可能的通道变化和下采样。
        blocks = [ResNet18BasicBlock(in_channels, out_channels, stride=stride)]

        # 后续 block 保持通道数和空间尺寸不变。
        for _ in range(1, num_blocks):
            blocks.append(ResNet18BasicBlock(out_channels, out_channels, stride=1))

        return nn.Sequential(*blocks)

    def forward(self, x):
        # 输入图像先经过 ResNet stem。
        x = self.stem(x)

        # 依次经过四个 stage，得到最终 feature map。
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        return x


class ResNet18SwitchTransformer(nn.Module):
    """ResNet-18 backbone followed by standard Switch Transformer blocks."""

    def __init__(
        self,
        num_classes=100,
        embed_dim=128,
        depth=4,
        num_heads=4,
        mlp_ratio=4.0,
        num_experts=8,
        moe_layers=None,
        dropout_rate=0.1,
        router_jitter_noise=0.0,
        capacity_factor=1.25,
        min_capacity=4,
        drop_tokens=True,
        top_k=1,
        stem_channels=None,
        token_grid_size=8,
        use_cls_token=False,
        router_aux_loss_coef=0.01,
        router_z_loss_coef=0.001,
    ):
        super(ResNet18SwitchTransformer, self).__init__()

        # Multi-head attention 要求 embed_dim 能被 num_heads 整除。
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        # 当前参数没有实际使用，保留是为了兼容配置。
        _ = stem_channels

        self.num_experts = num_experts
        self.embed_dim = embed_dim
        self.depth = depth

        # moe_layers 指定哪些 Transformer layer 使用 Switch FFN。
        # 例如 moe_layers=[1, 3] 表示第 1 和第 3 层用 MoE。
        self.moe_layers = set(moe_layers or [])

        # ResNet feature map 会被池化成 token_grid_size x token_grid_size 的 token 网格。
        self.token_grid_size = token_grid_size

        # 是否使用 cls token 做分类。
        # 如果 False，则使用所有 token 的平均池化结果做分类。
        self.use_cls_token = use_cls_token

        # router 辅助损失和 z-loss 的系数。
        self.router_aux_loss_coef = router_aux_loss_coef
        self.router_z_loss_coef = router_z_loss_coef

        # 图像特征提取 backbone。
        self.backbone = ResNet18Backbone()

        # 将 backbone 输出 feature map 池化成固定大小 token 网格。
        self.token_pool = nn.AdaptiveAvgPool2d((token_grid_size, token_grid_size))

        # 用 1x1 卷积把 ResNet 输出通道投影到 Transformer 的 embed_dim。
        self.token_projection = nn.Conv2d(self.backbone.out_channels, embed_dim, kernel_size=1)

        # 位置编码 token 数 = 网格 token 数 + 可选 cls token。
        num_position_tokens = token_grid_size * token_grid_size + (1 if use_cls_token else 0)
        self.position_embedding = nn.Parameter(torch.zeros(1, num_position_tokens, embed_dim))

        # 可选 cls token。
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        else:
            self.cls_token = None

        self.position_dropout = nn.Dropout(dropout_rate)

        # 构建 Transformer blocks。
        # 每一层是否使用 Switch FFN 由 layer_id in self.moe_layers 决定。
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout_rate=dropout_rate,
                use_switch_ffn=layer_id in self.moe_layers,
                num_experts=num_experts,
                router_jitter_noise=router_jitter_noise,
                capacity_factor=capacity_factor,
                min_capacity=min_capacity,
                drop_tokens=drop_tokens,
                top_k=top_k,
                layer_id=layer_id,
            )
            for layer_id in range(depth)
        ])

        # Transformer 输出前的 LayerNorm。
        self.norm = nn.LayerNorm(embed_dim)

        # 最终分类头。
        self.classifier = nn.Linear(embed_dim, num_classes)

        # 初始化模型权重。
        self._init_weights()

    def _init_weights(self):
        # 初始化位置编码。
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

        # 如果使用 cls token，也初始化 cls token。
        if self.cls_token is not None:
            nn.init.trunc_normal_(self.cls_token, std=0.02)

        # 遍历所有子模块，分别初始化 Conv2d 和 Linear。
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                # 卷积层使用 Kaiming 初始化，适合 ReLU 网络。
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.Linear):
                # 线性层使用截断正态初始化。
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def get_expert_state_dict_by_layer(self):
        # 按层返回每个 expert 的 state_dict。
        # 主要用于检查、保存或分析 MoE expert 参数。
        expert_states = {}

        for layer_id, block in enumerate(self.blocks):
            # 非 MoE 层没有 experts，直接跳过。
            if not block.use_switch_ffn:
                continue

            # 当前层每个 expert 单独保存。
            expert_states[str(layer_id)] = {
                str(expert_id): expert.state_dict()
                for expert_id, expert in enumerate(block.ffn.experts)
            }

        return expert_states

    def get_router_state_dict_by_layer(self):
        # 按层返回 router 的 state_dict。
        # 只有 MoE 层才有 router。
        router_states = {}

        for layer_id, block in enumerate(self.blocks):
            if block.use_switch_ffn:
                router_states[str(layer_id)] = block.ffn.router.state_dict()

        return router_states

    def get_moe_parameter_groups(self):
        # 返回 MoE 相关参数组。
        # 每个参数组包含类型、层号、expert_id 以及对应参数。
        parameter_groups = []

        for layer_id, block in enumerate(self.blocks):
            # 非 MoE 层没有 router 和 experts，直接跳过。
            if not block.use_switch_ffn:
                continue

            # 当前 MoE 层的 router 参数组。
            parameter_groups.append({
                "type": "router",
                "layer_id": str(layer_id),
                "params": block.ffn.router.parameters(),
            })

            # 当前 MoE 层的每个 expert 参数组。
            for expert_id, expert in enumerate(block.ffn.experts):
                parameter_groups.append({
                    "type": "expert",
                    "layer_id": str(layer_id),
                    "expert_id": str(expert_id),
                    "params": expert.parameters(),
                })

        return parameter_groups

    def forward(self, x):
        # 先用 ResNet-18 backbone 提取图像 feature map。
        feature_map = self.backbone(x)

        # 将 feature map 池化成固定大小 token 网格。
        feature_map = self.token_pool(feature_map)

        # 用 1x1 卷积投影到 Transformer embedding 维度。
        tokens = self.token_projection(feature_map)

        # 将 [B, C, H, W] 转成 [B, H*W, C]，作为 Transformer token 序列。
        tokens = tokens.flatten(2).transpose(1, 2)

        # 如果使用 cls token，则在 token 序列最前面拼接 cls token。
        if self.cls_token is not None:
            cls_tokens = self.cls_token.expand(tokens.size(0), -1, -1)
            tokens = torch.cat([cls_tokens, tokens], dim=1)

        # 加位置编码并做 dropout。
        tokens = self.position_dropout(tokens + self.position_embedding)

        # 初始化 router 相关损失。
        router_aux_loss = tokens.new_tensor(0.0)
        router_z_loss = tokens.new_tensor(0.0)

        # 初始化全模型层面的 expert/router 统计。
        expert_activations = torch.zeros(self.num_experts, device=tokens.device)
        selected_counts = torch.zeros(self.num_experts, device=tokens.device)
        overflow_counts = torch.zeros(self.num_experts, device=tokens.device)
        avg_router_probs = torch.zeros(self.num_experts, device=tokens.device)

        # 初始化按层保存的 expert/router 统计。
        expert_stats_by_layer = {}
        expert_activations_by_layer = {}
        selected_counts_by_layer = {}
        overflow_counts_by_layer = {}
        avg_router_probs_by_layer = {}
        capacity_by_layer = {}

        # 记录当前 forward 中实际经过了多少个 Switch FFN 层。
        switch_layer_count = 0

        # 依次经过每个 Transformer block。
        for block in self.blocks:
            tokens, switch_stats = block(tokens)

            # 普通 DenseFFN 层不会返回 switch_stats。
            if switch_stats is None:
                continue

            # 当前 MoE 层的 layer id。
            layer_key = str(switch_stats["layer_id"])

            # 累加 router auxiliary loss 和 z-loss。
            router_aux_loss = router_aux_loss + switch_stats["router_aux_loss"]
            router_z_loss = router_z_loss + switch_stats["router_z_loss"]

            # 累加所有 MoE 层的 expert 使用统计。
            expert_activations = expert_activations + switch_stats["expert_activations"]
            selected_counts = selected_counts + switch_stats["selected_counts"]
            overflow_counts = overflow_counts + switch_stats["overflow_counts"]
            avg_router_probs = avg_router_probs + switch_stats["avg_router_probs"]

            # 保存当前层的完整 expert/router 统计。
            layer_stats = {
                "expert_activations": switch_stats["expert_activations"],
                "selected_counts": switch_stats["selected_counts"],
                "overflow_counts": switch_stats["overflow_counts"],
                "capacity": switch_stats["capacity"],
                "avg_router_probs": switch_stats["avg_router_probs"],
            }

            # 按层记录统计信息，方便服务端做 expert 级别聚合或日志分析。
            expert_stats_by_layer[layer_key] = layer_stats
            expert_activations_by_layer[layer_key] = layer_stats["expert_activations"]
            selected_counts_by_layer[layer_key] = layer_stats["selected_counts"]
            overflow_counts_by_layer[layer_key] = layer_stats["overflow_counts"]
            avg_router_probs_by_layer[layer_key] = layer_stats["avg_router_probs"]
            capacity_by_layer[layer_key] = layer_stats["capacity"]

            switch_layer_count += 1

        # 如果存在 MoE 层，则对 router loss 和平均 router 概率按 MoE 层数取平均。
        if switch_layer_count > 0:
            router_aux_loss = router_aux_loss / switch_layer_count
            router_z_loss = router_z_loss / switch_layer_count
            avg_router_probs = avg_router_probs / switch_layer_count

        # 总 router loss 会在客户端训练时加到主分类 loss 上。
        total_router_loss = (
            self.router_aux_loss_coef * router_aux_loss
            + self.router_z_loss_coef * router_z_loss
        )

        # Transformer 输出做 LayerNorm。
        tokens = self.norm(tokens)

        # 分类特征：
        # - 如果有 cls token，则取第 0 个 token；
        # - 否则对所有 token 做 mean pooling。
        pooled = tokens[:, 0] if self.cls_token is not None else tokens.mean(dim=1)

        # 分类 logits。
        logits = self.classifier(pooled)

        # 返回分类输出、特征、router loss 和 expert 统计。
        return {
            "logits": logits,
            "feature": pooled,
            "aux_loss": router_aux_loss,
            "router_aux_loss": router_aux_loss,
            "router_z_loss": router_z_loss,
            "total_router_loss": total_router_loss,
            "expert_activations": expert_activations,
            "expert_activations_summary": expert_activations,
            "selected_counts_summary": selected_counts,
            "overflow_counts_summary": overflow_counts,
            "avg_router_probs": avg_router_probs,
            "expert_stats_by_layer": expert_stats_by_layer,
            "expert_activations_by_layer": expert_activations_by_layer,
            "selected_counts_by_layer": selected_counts_by_layer,
            "overflow_counts_by_layer": overflow_counts_by_layer,
            "avg_router_probs_by_layer": avg_router_probs_by_layer,
            "capacity_by_layer": capacity_by_layer,
        }