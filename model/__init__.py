"""Model package."""


_DATASET_MODEL_CONFIGS = {
    "cifar10": {"num_classes": 10, "in_channels": 3, "img_size": 32},
    "cifar100": {"num_classes": 100, "in_channels": 3, "img_size": 32},
    "tinyimagenet": {"num_classes": 200, "in_channels": 3, "img_size": 64},
    "femnist": {"num_classes": 62, "in_channels": 1, "img_size": 28},
}


def get_num_classes(data_name):
    # 根据数据集名称返回分类类别数。
    dataset_config = _DATASET_MODEL_CONFIGS.get(data_name)
    if dataset_config is not None:
        return dataset_config["num_classes"]

    # 如果传入的数据集名称不支持，直接报错，避免后续模型分类头维度错误。
    raise ValueError(f"Unsupported dataset: {data_name}")


def parse_moe_layers(moe_layers, depth):
    # 解析配置里的 moe_layers。
    # moe_layers 通常是类似 "1,3" 的字符串，表示第 1 层和第 3 层使用 MoE。
    # 如果为空，说明所有 Transformer block 都使用普通 FFN，不启用 MoE 层。
    if moe_layers is None or str(moe_layers).strip() == "":
        return []

    layer_ids = []

    # 按逗号分隔每一个 MoE 层编号。
    for item in str(moe_layers).split(","):
        item = item.strip()

        # 跳过空字符串，兼容类似 "1, 3," 这种写法。
        if item == "":
            continue

        # 将层编号从字符串转成整数。
        layer_id = int(item)

        # 检查 MoE 层编号是否合法。
        # Transformer block 的编号范围是 [0, depth - 1]。
        if layer_id < 0 or layer_id >= depth:
            raise ValueError(f"moe layer index {layer_id} is outside depth {depth}")

        layer_ids.append(layer_id)

    return layer_ids


def build_model_from_args(args):
    """Build the project model from a single shared args-based code path."""

    dataset_config = _DATASET_MODEL_CONFIGS.get(args.data_name)
    if dataset_config is None:
        raise ValueError(f"Unsupported dataset: {args.data_name}")

    if args.model_type == "resnet_sparse_moe_head":
        from model.ResNetSparseMoEHead import ResNetSparseMoEHead

        return ResNetSparseMoEHead(
            in_channels=dataset_config["in_channels"],
            num_classes=dataset_config["num_classes"],
            img_size=dataset_config["img_size"],
            num_experts=args.num_experts,
            top_k=args.top_k,
        )

    # Transformer block 的总层数。
    depth = args.depth

    # 不同模型可能共享的一组构造参数。
    # 这里统一从 args 中读取，保证训练、客户端、服务端构建模型时走同一套配置。
    common_kwargs = dict(
        # 根据数据集名称自动确定分类头输出维度。
        num_classes=get_num_classes(args.data_name),

        # Transformer token embedding 维度。
        embed_dim=args.embed_dim,

        # Transformer block 层数。
        depth=depth,

        # 多头注意力的 head 数。
        num_heads=args.num_heads,

        # FFN hidden_dim 相对 embed_dim 的放大倍数。
        mlp_ratio=args.mlp_ratio,

        # MoE expert 数量。
        num_experts=args.num_experts,

        # 解析哪些层使用 MoE / Switch FFN。
        moe_layers=parse_moe_layers(args.moe_layers, depth),

        # dropout 概率。
        dropout_rate=args.dropout,

        # router 选择 expert 的 top-k 数量。
        top_k=args.top_k,

        # ResNet feature map 会被池化成 token_grid_size x token_grid_size 的 token 网格。
        token_grid_size=args.token_grid_size,

        # 是否使用 cls token 做分类。
        # false 时通常使用 token mean pooling。
        use_cls_token=args.use_cls_token,

        # router auxiliary loss 系数，旧配置缺省时默认关闭。
        router_aux_loss_coef=getattr(args, "router_aux_loss_coef", 0.0),

        # router z-loss 系数，旧配置缺省时默认关闭。
        router_z_loss_coef=getattr(args, "router_z_loss_coef", 0.0),
    )

    # 当前项目支持的模型类型：ResNet18 backbone + Switch Transformer。
    if args.model_type == "resnet18_switch_transformer":
        # 延迟导入模型类，避免 model 包初始化时就加载所有模型文件。
        from model.ResNet18SwitchTransformer import ResNet18SwitchTransformer

        # 使用统一参数构建模型实例。
        return ResNet18SwitchTransformer(**common_kwargs)

    # 如果配置中的 model_type 没有注册，直接报错。
    raise ValueError(f"Unsupported model_type: {args.model_type}")