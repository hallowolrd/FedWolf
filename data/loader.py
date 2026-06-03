import os

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100


# 当前项目期望的数据划分协议名称。
# 用来确认 partition_meta.pt 是由本项目的数据划分脚本生成的。
EXPECTED_PROTOCOL = "client_train_global_test_index_partition"

# 当前项目期望的数据划分文件版本。
# 如果 meta 文件版本不一致，说明可能是旧格式或不兼容格式。
EXPECTED_VERSION = 3


def get_cifar_stats(data_name, stats_impl=None):
    """根据数据集名字，返回：
    1. 数据集类 CIFAR10 或 CIFAR100
    2. 归一化均值 mean
    3. 归一化标准差 std
    4. 类别数 num_classes"""

    # 保存不同 CIFAR 数据集对应的数据集类、归一化参数和类别数。
    data_dict = {
        "cifar10": (
            CIFAR10,
            (0.4914, 0.4822, 0.4465),
            (0.2470, 0.2435, 0.2616),
            10,
        ),
        "cifar100": (
            CIFAR100,
            (0.5071, 0.4867, 0.4408),
            (0.2675, 0.2565, 0.2761),
            100,
        ),
    }

    # 严格 baseline 复用 moefedavg.py 的 CIFAR10 normalize std。
    if str(stats_impl).strip().lower() == "moefedavg" and data_name == "cifar10":
        data_dict["cifar10"] = (
            CIFAR10,
            (0.4914, 0.4822, 0.4465),
            (0.2023, 0.1994, 0.2010),
            10,
        )

    # 如果配置的数据集名称不支持，直接报错。
    if data_name not in data_dict:
        raise ValueError(f"Unsupported dataset: {data_name}")

    return data_dict[data_name]


def build_transforms(data_name, transform_impl=None):
    """构造图像预处理流程：
    - client_train:使用数据增强
    - eval(global_test)：使用确定性预处理，不做增强"""

    # 读取当前数据集对应的归一化均值、标准差。
    _, mean, std, _ = get_cifar_stats(data_name, stats_impl=transform_impl)

    # 训练阶段 transform：
    # 随机裁剪 + 随机水平翻转 + 转 Tensor + 归一化。
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    # 测试 / evidence 阶段 transform：
    # 不使用随机增强，保证评估结果稳定。
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    return train_transform, eval_transform


def load_partition_meta(args):
    """ 从磁盘加载 partition_meta.pt。
    这个文件是上一段 data.py 划分数据后保存的元信息文件。 """

    # 根据配置拼出 partition_meta.pt 路径。
    meta_path = os.path.join(args.data_save_path, args.partition_meta_name)

    # 如果划分文件不存在，提示用户先运行数据划分脚本。
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"Missing partition metadata: {meta_path}. "
            "Update config.yaml if needed, then run "
            "`python -m data.data --config <path/to/config.yaml>` before training."
        )

    # partition_meta.pt 保存的是普通 Python dict，不是模型权重。
    meta = torch.load(meta_path, weights_only=False)

    # 加载后立即检查 meta 是否和当前配置匹配。
    validate_partition_meta(meta, args)

    return meta


def validate_partition_meta(meta, args):
    """检查当前加载的 partition_meta.pt 是否和当前配置一致。
    目的：防止你改了 yaml 配置，却还在用旧的 partition_meta.pt。"""

    # 先检查 meta 的结构是否完整。
    validate_partition_structure(meta, args)

    # 逐项检查 meta 中保存的配置是否和当前 args 一致。
    checks = [
        ("protocol", meta.get("protocol"), EXPECTED_PROTOCOL, "str"),
        ("version", meta.get("version"), EXPECTED_VERSION, "int"),
        ("dataset", meta.get("dataset"), args.data_name, "str"),
        ("num_clients", meta.get("num_clients"), args.num_clients, "int"),
        ("alpha", meta.get("alpha"), args.alpha, "float"),
        ("seed", meta.get("seed"), args.seed, "int"),
        ("data_path", meta.get("data_path"), args.data_path, "path"),
    ]
    if hasattr(args, "partition_impl"):
        checks.append(("partition_impl", meta.get("partition_impl"), args.partition_impl, "str"))

    # 任意一个字段不匹配，就说明当前配置和已有 partition_meta.pt 不一致。
    for field, actual, expected, value_type in checks:
        if not metadata_value_matches(actual, expected, value_type):
            raise_partition_mismatch(field, actual, expected)


def validate_partition_structure(meta, args):
    """ 检查 partition_meta 的结构是否完整。
    这里主要检查：
    - 有没有 splits
    - splits 里关键字段齐不齐
    - client_train_indices 的键是不是完整"""

    # splits 是数据划分文件中最核心的字段。
    splits = meta.get("splits")
    if not isinstance(splits, dict):
        raise ValueError(
            "partition_meta is incomplete: missing a valid `splits` dictionary. "
            "Please regenerate partition_meta.pt and partition_stats.json."
        )

    # 当前协议要求 splits 至少包含客户端训练索引和全局测试索引。
    required_split_keys = {
        "client_train_indices",
        "global_test_indices",
    }

    # 检查 splits 中是否缺少必要字段。
    missing = required_split_keys - set(splits.keys())
    if missing:
        raise ValueError(
            f"partition_meta is incomplete: missing split keys {sorted(missing)}. "
            "Please regenerate partition_meta.pt and partition_stats.json."
        )

    # client_train_indices 应该是 dict：
    # key 是客户端编号，value 是该客户端拥有的训练样本索引。
    if not isinstance(splits["client_train_indices"], dict):
        raise ValueError(
            "`splits['client_train_indices']` must be a dict. "
            "Please regenerate partition_meta.pt and partition_stats.json."
        )

    # 根据当前 num_clients，构造期望的客户端编号集合。
    expected_client_keys = {str(i) for i in range(1, args.num_clients + 1)}

    # 读取 meta 中实际保存的客户端编号集合。
    actual_client_keys = set(splits["client_train_indices"].keys())

    # 检查客户端编号是否完整、是否和当前配置一致。
    if actual_client_keys != expected_client_keys:
        raise ValueError(
            "partition_meta has incomplete client_train_indices keys: "
            f"expected {sorted(expected_client_keys)}, found {sorted(actual_client_keys)}. "
            "Please regenerate partition_meta.pt and partition_stats.json."
        )

    # 每个客户端的训练索引都必须是 list 或 tuple。
    for client_id, indices in splits["client_train_indices"].items():
        if not isinstance(indices, (list, tuple)):
            raise ValueError(
                f"`splits['client_train_indices']['{client_id}']` must be a list or tuple. "
                "Please regenerate partition_meta.pt and partition_stats.json."
            )

    # global_test_indices 也必须是 list 或 tuple。
    for key in ["global_test_indices"]:
        if not isinstance(splits[key], (list, tuple)):
            raise ValueError(
                f"`splits['{key}']` must be a list or tuple. "
                "Please regenerate partition_meta.pt and partition_stats.json."
            )


def metadata_value_matches(actual, expected, value_type):
    """ 比较 meta 中读到的值 actual 和当前期望值 expected 是否匹配。
    不同类型采用不同比较方式。 """

    # 缺失值直接认为不匹配。
    if actual is None:
        return False

    # float 类型用一个很小的容差比较，避免浮点表示误差。
    if value_type == "float":
        return abs(float(actual) - float(expected)) <= 1e-12

    # int 类型统一转成 int 后比较。
    if value_type == "int":
        return int(actual) == int(expected)

    # path 类型先做路径归一化，再转成绝对路径比较。
    if value_type == "path":
        return os.path.abspath(os.path.normpath(str(actual))) == os.path.abspath(os.path.normpath(str(expected)))

    # 其他类型直接比较。
    return actual == expected


def raise_partition_mismatch(field, actual, expected):
    """ 当 partition_meta 中某个字段和当前配置不一致时，抛出统一格式的报错。 """

    # 统一报错信息，告诉用户哪个字段不匹配，以及如何重新生成 partition_meta。
    raise ValueError(
        f"partition_meta mismatch for `{field}`: found {actual!r}, expected {expected!r}. "
        "Please update config.yaml and re-run "
        "`python -m data.data --config <path/to/config.yaml>` "
        "to regenerate partition_meta.pt."
    )


def build_raw_cifar_dataset(args, train, transform):
    """ 构造原始 CIFAR 数据集对象。
    注意：
    这里只是加载“完整官方数据集”，
    后面再通过 Subset + 索引切出某个 split。 """

    # 根据数据集名称拿到 CIFAR10 或 CIFAR100 类。
    dataset_cls, _, _, _ = get_cifar_stats(args.data_name)

    try:
        # 训练阶段不自动下载数据。
        # 原始数据应该已经在运行 data.data 构建划分时下载好了。
        return dataset_cls(
            root=args.data_path,
            train=train,
            download=False,
            transform=transform,
        )
    except FileNotFoundError as e:
        # 原始 CIFAR 文件不存在时，给出更明确的提示。
        raise FileNotFoundError(
            f"Could not load raw CIFAR data from `{args.data_path}` with download=False. "
            "This project uses index-based partition metadata, so training still requires "
            "the original CIFAR files. Please make sure the raw dataset exists under data_path, "
            "or re-run: `python -m data.data --config <path/to/config.yaml>` "
            "to download the dataset and regenerate partition files."
        ) from e
    except RuntimeError as e:
        # torchvision 加载数据失败时，很多情况会抛 RuntimeError。
        error_text = str(e).lower()

        # 用关键词判断是否可能是数据缺失。
        missing_keywords = ["not found", "dataset not found", "no such file", "download"]

        # 用关键词判断是否可能是数据损坏。
        corrupt_keywords = ["corrupt", "corrupted", "truncate", "truncated", "invalid", "pickle", "unpickling"]

        # 数据缺失时，提示重新下载并生成划分。
        if any(keyword in error_text for keyword in missing_keywords):
            raise FileNotFoundError(
                f"Could not load raw CIFAR data from `{args.data_path}` with download=False. "
                "This project uses index-based partition metadata, so training still requires "
                "the original CIFAR files. Please make sure the raw dataset exists under data_path, "
                "or re-run: `python -m data.data --config <path/to/config.yaml>` "
                "to download the dataset and regenerate partition files."
            ) from e

        # 数据损坏时，提示检查数据文件或重新下载。
        if any(keyword in error_text for keyword in corrupt_keywords):
            raise RuntimeError(
                f"Raw CIFAR files were found under `{args.data_path}`, but loading failed and "
                "the dataset may be corrupted or incomplete. This project uses index-based "
                "partition metadata, so training still requires the original CIFAR files. "
                "Please check the dataset files or re-run: "
                "`python -m data.data --config <path/to/config.yaml>` "
                "to re-download and regenerate partition files."
            ) from e

        # 其他 RuntimeError 统一包装成更有上下文的报错。
        raise RuntimeError(
            f"Failed to load raw CIFAR data from `{args.data_path}` with download=False. "
            "This project uses index-based partition metadata, so training still requires "
            "the original CIFAR files. Please check data_path or re-run: "
            "`python -m data.data --config <path/to/config.yaml>` "
            "to regenerate partition files after ensuring the dataset can be read."
        ) from e
    except Exception as e:
        # 捕获其他未知异常，补充项目上下文信息后继续抛出。
        raise RuntimeError(
            f"Unexpected error while loading raw CIFAR data from `{args.data_path}` with download=False. "
            "This project uses index-based partition metadata, so training still requires "
            "the original CIFAR files. Please check data_path or re-run: "
            "`python -m data.data --config <path/to/config.yaml>`."
        ) from e


def build_index_dataset(args, split, client_id=None, meta=None):
    """ 根据 split 类型，构造一个“按索引切好”的数据集 Subset。"""

    # 如果调用方已经传入 meta，就直接复用；否则从磁盘加载 partition_meta.pt。
    meta = meta or load_partition_meta(args)

    # 构造训练 transform 和评估 transform。
    train_transform, eval_transform = build_transforms(
        args.data_name,
        transform_impl=getattr(args, "partition_impl", None),
    )

    # 取出 meta 中保存的划分索引。
    splits = meta["splits"]

    if split == "client_train":
        # 构造客户端训练集时必须指定 client_id。
        if client_id is None:
            raise ValueError("client_id is required for client_train split")

        # 取出当前客户端对应的训练样本索引。
        indices = splits["client_train_indices"][str(client_id)]

        # 客户端训练集来自官方 train set，使用训练增强 transform。
        dataset = build_raw_cifar_dataset(args, train=True, transform=train_transform)
    elif split == "global_test":
        # 全局测试集来自官方 test set。
        indices = splits["global_test_indices"]

        # 测试集使用确定性的 eval_transform。
        dataset = build_raw_cifar_dataset(args, train=False, transform=eval_transform)
    else:
        # 当前只支持 client_train 和 global_test 两种 split。
        raise ValueError(f"Unknown split: {split}")

    # 使用 Subset 按索引切出对应数据集。
    return Subset(dataset, indices)


def build_client_train_loader(args, client_id, meta=None):
    """ 构造某个客户端的训练 DataLoader。 """

    # 先构造当前客户端的训练数据集。
    dataset = build_index_dataset(
        args=args,
        split="client_train",
        client_id=client_id,
        meta=meta,
    )

    # 训练 DataLoader 使用 shuffle=True，让每个 epoch 的样本顺序随机。
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )


def build_client_evidence_loader(args, client_id, meta=None):
    """构造某个客户端用于 Fisher evidence 的确定性 DataLoader。"""

    # 如果调用方没传 meta，则从磁盘加载。
    meta = meta or load_partition_meta(args)

    # Fisher evidence 使用 eval_transform，不使用随机增强，保证估计更稳定。
    _, eval_transform = build_transforms(
        args.data_name,
        transform_impl=getattr(args, "partition_impl", None),
    )

    # evidence 数据仍然来自该客户端自己的训练样本。
    indices = meta["splits"]["client_train_indices"][str(client_id)]

    # 加载官方训练集，但 transform 使用确定性的 eval_transform。
    dataset = build_raw_cifar_dataset(
        args=args,
        train=True,
        transform=eval_transform,
    )

    # 按当前客户端训练索引切出子集。
    dataset = Subset(dataset, indices)

    # Fisher evidence loader 使用 shuffle=False，避免样本顺序随机变化。
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )


def build_global_eval_loader(args, split, meta=None):
    """ 构造全局测试 DataLoader。 """

    # 当前函数只允许构造 global_test。
    if split != "global_test":
        raise ValueError("split must be global_test")

    # 构造全局测试数据集。
    dataset = build_index_dataset(args=args, split=split, meta=meta)

    eval_batch_size = getattr(args, "eval_batch_size", None)
    if eval_batch_size is None:
        eval_batch_size = args.batch_size
    eval_batch_size = int(eval_batch_size)

    # 测试 DataLoader 不打乱样本顺序。
    return DataLoader(
        dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )


def get_client_train_size(args, client_id, meta=None):
    """ 获取某个客户端训练集的样本数量。
    这个函数不真正构造 Dataset,只是直接看 meta 里的索引长度。 """

    # 如果调用方没传 meta，则从磁盘加载 partition_meta.pt。
    meta = meta or load_partition_meta(args)

    # 返回当前客户端训练样本索引数量。
    # FedAvg 会使用这个值作为客户端聚合权重。
    return len(meta["splits"]["client_train_indices"][str(client_id)])