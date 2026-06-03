import argparse
import json
import os
from collections import Counter
from types import SimpleNamespace

import numpy as np
import torch

from configs import add_config_path_arguments, load_args, validate_output_paths
from data.loader import get_cifar_stats
from utils.utils import set_seed


class CIFARPartitionBuilder:
    """为 CIFAR10 / CIFAR100 构建联邦学习的数据划分。
    Protocol:
    1. 官方训练集全部通过 Dirichlet non-IID 划分给客户端。
    2. 官方测试集保留为 global_test，客户端从不使用。
    3. 最后保存“索引”和“统计信息”,真正的数据增强和transform会在loader.py里动态做。
    """

    def __init__(self, args: SimpleNamespace):
        """初始化函数：读取配置参数，加载数据集，准备好后面要用到的信息。 """

        # 保存完整配置，后面构建数据划分、保存文件都会用到。
        self.args = args

        # 数据划分文件保存路径。
        self.data_save_path = self.args.data_save_path

        # 联邦学习客户端数量。
        self.num_clients = self.args.num_clients

        # 数据集名称，例如 cifar10 / cifar100。
        self.data_name = self.args.data_name

        # 原始 CIFAR 数据下载 / 存放路径。
        self.data_path = self.args.data_path

        # Dirichlet 分布参数 alpha。
        # alpha 越小，客户端之间的数据越 non-IID；alpha 越大，类别分布越接近均匀。
        self.alpha = self.args.alpha

        # 数据划分实现。default 保持旧逻辑，moefedavg 复现单文件 baseline。
        self.partition_impl = str(getattr(self.args, "partition_impl", "default")).strip().lower()

        # 先加载 torchvision 里的原始训练集和测试集。
        self.train_dataset,self.test_dataset,self.num_classes = self.load_dataset()

        # 官方 train set 全部划给客户端；官方 test set 保持统一，不参与客户端划分。
        self.seed = self.args.seed

        # 使用 NumPy 的随机数生成器，保证 Dirichlet 划分可复现。
        self.rng = np.random.default_rng(self.seed)

        # 取出训练集和测试集的标签，后续按类别划分样本会用到。
        self.train_targets = np.array(self.train_dataset.targets)
        self.test_targets = np.array(self.test_dataset.targets)

    def load_dataset(self):
        """根据 data_name 加载 CIFAR10 或 CIFAR100。"""

        # get_cifar_stats 根据数据集名称返回数据集类、transform 信息和类别数。
        dataset_cls, _, _, num_classes = get_cifar_stats(self.data_name)

        # download=True 表示如果 ./data 下没有数据，会自动下载。
        # transform=None 表示这里只保存原始数据索引，真正的数据增强在 loader.py 中处理。
        train_dataset = dataset_cls(root=self.args.data_path, train=True, download=True, transform=None)
        test_dataset = dataset_cls(root=self.args.data_path, train=False, download=True, transform=None)
        return train_dataset, test_dataset, num_classes

    def build(self):
        """整个数据划分流程的主函数,创建partition_meta.pt和partition_stats.json
        partition_meta.pt：训练真正要用的“划分依据”
        partition_stats.json：给你检查划分结果的“统计报告” """

        # 先检查配置是否合法。
        self.validate_args()

        # 官方训练集的所有样本索引。
        official_train_indices = list(range(len(self.train_dataset)))

        # 按 Dirichlet non-IID 方式把训练集索引切给各客户端。
        client_train_indices = self.dirichlet_client_split(official_train_indices)

        # meta 保存真正训练时要用的数据划分信息。
        # 注意这里只保存索引，不保存真实图片数据。
        meta = {
            "protocol": "client_train_global_test_index_partition",
            "version": 3,
            "dataset": self.data_name,
            "data_path": self.data_path,
            "num_classes": self.num_classes,
            "num_clients": self.num_clients,
            "alpha": self.alpha,
            "seed": self.seed,
            "partition_impl": self.partition_impl,
            "index_space": {
                "client_train": "official_train",
                "global_test": "official_test",
            },
            "splits": {
                "client_train_indices": {
                    str(client_id): indices
                    for client_id, indices in client_train_indices.items()
                },
                "global_test_indices": list(range(len(self.test_dataset))),
            },
        }

        # 根据 meta 统计每个客户端的数据量和类别分布。
        stats = self.build_stats(meta)

        # 保存 meta 和 stats 到磁盘。
        self.save(meta, stats)

        return meta, stats

    def validate_args(self):
        """ 检查配置参数是否合法。
        如果不合法，就直接抛出错误。 """

        # 客户端数量必须大于 0。
        if self.num_clients <= 0:
            raise ValueError("num_clients must be positive")

        # Dirichlet alpha 必须大于 0。
        if self.alpha <= 0:
            raise ValueError("alpha must be positive")

        if self.partition_impl not in {"default", "moefedavg"}:
            raise ValueError("partition_impl must be either 'default' or 'moefedavg'")

    def dirichlet_client_split(self, pool_indices):
        """ 把官方训练集按 Dirichlet 分布切给多个客户端，构造 non-IID 的联邦训练数据。"""

        if self.partition_impl == "moefedavg":
            return self.moefedavg_dirichlet_client_split(pool_indices)

        # 将输入索引转成 NumPy 数组，方便后面按类别筛选。
        pool_indices = np.array(pool_indices)

        # 取出这些索引对应的标签。
        pool_targets = self.train_targets[pool_indices]

        # class_indices[class_id] 保存该类别下所有样本索引。
        class_indices = [
            pool_indices[np.where(pool_targets == class_id)[0]]
            for class_id in range(self.num_classes)
        ]

        # 初始化每个客户端的索引列表。
        # 客户端编号从 1 开始。
        client_indices = {client_id: [] for client_id in range(1, self.num_clients + 1)}

        # 为每个类别采样一个长度为 num_clients 的 Dirichlet 分布。
        # label_distribution[class_id][client_id] 表示该类别分给某客户端的比例。
        label_distribution = self.rng.dirichlet(
            [self.alpha] * self.num_clients,
            self.num_classes,
        )

        # 对每个类别分别按照 Dirichlet 比例切分给客户端。
        for class_id, class_idcs in enumerate(class_indices):
            # 先打乱该类别的样本索引。
            shuffled_idcs = self.rng.permutation(class_idcs)

            # 根据 Dirichlet 比例计算切分点。
            split_points = (
                np.cumsum(label_distribution[class_id])[:-1] * len(shuffled_idcs)
            ).astype(int)

            # 按切分点切成 num_clients 份，并分给各客户端。
            for client_id, idcs in enumerate(np.split(shuffled_idcs, split_points), start=1):
                client_indices[client_id].extend(idcs.tolist())

        # 每个客户端内部再打乱一次样本顺序。
        for idcs in client_indices.values():
            self.rng.shuffle(idcs)

        return client_indices

    def moefedavg_dirichlet_client_split(self, pool_indices):
        """复现 moefedavg.py 的 per-class dirichlet + multinomial 划分。"""

        pool_indices = np.array(pool_indices)
        pool_targets = self.train_targets[pool_indices]
        client_indices = {client_id: [] for client_id in range(1, self.num_clients + 1)}

        for class_id in range(self.num_classes):
            class_idcs = pool_indices[np.where(pool_targets == class_id)[0]].copy()
            self.rng.shuffle(class_idcs)

            proportions = self.rng.dirichlet(np.full(self.num_clients, self.alpha))
            counts = self.rng.multinomial(len(class_idcs), proportions)

            offset = 0
            for client_id, count in enumerate(counts, start=1):
                next_offset = offset + int(count)
                client_indices[client_id].extend(class_idcs[offset:next_offset].tolist())
                offset = next_offset

        for idcs in client_indices.values():
            self.rng.shuffle(idcs)

        return client_indices

    def build_stats(self, meta):
        """ 根据 meta 中的划分结果，生成统计信息 stats。
        stats 主要用于：
        - 看每个集合有多少样本
        - 看每个集合的类别分布
        - 检查 non-IID 是否符合预期 """

        # 取出 meta 中保存的划分索引。
        splits = meta["splits"]

        # 统计每个客户端训练集里的类别分布。
        client_class_counts = {
            client_id: self.class_counts(indices, self.train_targets)
            for client_id, indices in splits["client_train_indices"].items()
        }

        # 返回整体统计报告。
        return {
            "protocol": meta["protocol"],
            "dataset": self.data_name,
            "num_classes": self.num_classes,
            "num_clients": self.num_clients,
            "alpha": self.alpha,
            "seed": self.seed,
            "partition_impl": self.partition_impl,
            "sizes": {
                # 官方训练集总样本数。
                "official_train": len(self.train_dataset),

                # 全局测试集样本数。
                "global_test": len(splits["global_test_indices"]),

                # 每个客户端训练集样本数。
                "client_train": {
                    client_id: len(indices)
                    for client_id, indices in splits["client_train_indices"].items()
                },
            },
            "class_counts": {
                # 官方训练集类别分布。
                "official_train": self.class_counts(list(range(len(self.train_dataset))), self.train_targets),

                # 全局测试集类别分布。
                "global_test": self.class_counts(splits["global_test_indices"], self.test_targets),

                # 每个客户端训练集类别分布。
                "client_train": client_class_counts,
            },
        }

    def class_counts(self, indices, targets):
        """ 统计给定索引集合中，每个类别各有多少个样本。 """

        # Counter 统计当前索引集合中每个标签出现次数。
        counts = Counter(int(targets[index]) for index in indices)

        # 保证每个类别都会出现在结果中；如果某类别没有样本，则计数为 0。
        return {str(class_id): int(counts.get(class_id, 0)) for class_id in range(self.num_classes)}

    def save(self, meta, stats):
        """ 把划分结果和统计信息保存到文件。 """

        # 确保保存目录存在。
        os.makedirs(self.data_save_path, exist_ok=True)

        # partition_meta.pt 保存训练真正使用的数据划分索引。
        meta_path = os.path.join(self.data_save_path, self.args.partition_meta_name)

        # partition_stats.json 保存可读性更强的统计报告。
        stats_path = os.path.join(self.data_save_path, self.args.partition_stats_name)

        # meta 用 torch.save 保存，后续 loader.py 可以直接加载。
        torch.save(meta, meta_path)

        # stats 用 JSON 保存，方便人工查看每个客户端的数据分布。
        with open(stats_path, "w", encoding="utf-8") as stats_file:
            json.dump(stats, stats_file, ensure_ascii=False, indent=2)

        print(f"Saved partition meta to {meta_path}")
        print(f"Saved partition stats to {stats_path}")


if __name__ == '__main__':
    # 单独运行该文件时，作为数据划分脚本使用。
    cli_parser = argparse.ArgumentParser(description="Build CIFAR partitions from one nested YAML config file.")

    # 添加 --config 参数，用于指定 YAML 配置文件路径。
    add_config_path_arguments(cli_parser)

    # 解析命令行参数。
    cli_args = cli_parser.parse_args()

    # Load partition settings from the config.yaml passed by --config.
    # 从配置文件读取数据划分相关参数。
    args = load_args(config_path=cli_args.config)

    # 检查输出路径是否合法，避免数据划分文件写到错误位置。
    validate_output_paths(args, stage="data")

    # 固定随机种子，保证 Dirichlet 划分可复现。
    set_seed(args.seed)

    # 构建并保存 CIFAR 联邦数据划分。
    CIFARPartitionBuilder(args=args).build()