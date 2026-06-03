"""Model package for the slim moefedavg baseline."""


_DATASET_MODEL_CONFIGS = {
    "cifar10": {"num_classes": 10, "in_channels": 3, "img_size": 32},
    "cifar100": {"num_classes": 100, "in_channels": 3, "img_size": 32},
}


def build_model_from_args(args):
    """Build the only model supported by the slim baseline."""

    dataset_config = _DATASET_MODEL_CONFIGS.get(args.data_name)
    if dataset_config is None:
        raise ValueError(f"Unsupported dataset: {args.data_name}")

    if args.model_type != "resnet_sparse_moe_head":
        raise ValueError(
            "The slim baseline only supports model_type='resnet_sparse_moe_head'. "
            f"Got {args.model_type!r}."
        )

    from model.ResNetSparseMoEHead import ResNetSparseMoEHead

    return ResNetSparseMoEHead(
        in_channels=dataset_config["in_channels"],
        num_classes=dataset_config["num_classes"],
        img_size=dataset_config["img_size"],
        num_experts=args.num_experts,
        top_k=args.top_k,
    )
