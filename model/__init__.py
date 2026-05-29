"""Model package."""


def get_num_classes(data_name):
    if data_name == "cifar10":
        return 10
    if data_name == "cifar100":
        return 100
    raise ValueError(f"Unsupported dataset: {data_name}")


def parse_moe_layers(moe_layers, depth):
    if moe_layers is None or str(moe_layers).strip() == "":
        return []

    layer_ids = []
    for item in str(moe_layers).split(","):
        item = item.strip()
        if item == "":
            continue
        layer_id = int(item)
        if layer_id < 0 or layer_id >= depth:
            raise ValueError(f"moe layer index {layer_id} is outside depth {depth}")
        layer_ids.append(layer_id)
    return layer_ids


def build_model_from_args(args):
    """Build the project model from a single shared args-based code path."""

    depth = args.depth
    common_kwargs = dict(
        num_classes=get_num_classes(args.data_name),
        embed_dim=args.embed_dim,
        depth=depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        num_experts=args.num_experts,
        moe_layers=parse_moe_layers(args.moe_layers, depth),
        dropout_rate=args.dropout,
        top_k=args.top_k,
        token_grid_size=args.token_grid_size,
        use_cls_token=args.use_cls_token,
    )

    if args.model_type == "resnet18_switch_transformer":
        from model.ResNet18SwitchTransformer import ResNet18SwitchTransformer

        return ResNet18SwitchTransformer(**common_kwargs)
    raise ValueError(f"Unsupported model_type: {args.model_type}")
