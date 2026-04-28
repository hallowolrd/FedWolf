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

        self.args = args
        self.data_save_path = self.args.data_save_path
        self.num_clients = self.args.num_clients
        self.data_name = self.args.data_name
        self.data_path = self.args.data_path
        self.alpha = self.args.alpha

        # 先加载 torchvision 里的原始训练集和测试集。
        self.train_dataset,self.test_dataset,self.num_classes = self.load_dataset()

        # 官方 train set 全部划给客户端；官方 test set 保持统一，不参与客户端划分。
        self.min_datasize = self.args.min_datasize
        self.seed = self.args.seed
        self.rng = np.random.default_rng(self.seed)
        self.train_targets = np.array(self.train_dataset.targets)
        self.test_targets = np.array(self.test_dataset.targets)

    def load_dataset(self):
        """根据 data_name 加载 CIFAR10 或 CIFAR100。"""

        dataset_cls, _, _, num_classes = get_cifar_stats(self.data_name)

        # download=True 表示如果 ./data 下没有数据，会自动下载。
        train_dataset = dataset_cls(root=self.args.data_path, train=True, download=True, transform=None)
        test_dataset = dataset_cls(root=self.args.data_path, train=False, download=True, transform=None)
        return train_dataset, test_dataset, num_classes

    def build(self):
        """整个数据划分流程的主函数,创建partition_meta.pt和partition_stats.json
        partition_meta.pt：训练真正要用的“划分依据”
        partition_stats.json：给你检查划分结果的“统计报告” """

        self.validate_args()
        official_train_indices = list(range(len(self.train_dataset)))
        client_train_indices = self.dirichlet_client_split(official_train_indices)

        meta = {
            "protocol": "client_train_global_test_index_partition",
            "version": 2,
            "dataset": self.data_name,
            "data_path": self.data_path,
            "num_classes": self.num_classes,
            "num_clients": self.num_clients,
            "alpha": self.alpha,
            "seed": self.seed,
            "min_datasize": self.min_datasize,
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
        stats = self.build_stats(meta)
        self.save(meta, stats)
        return meta, stats

    def validate_args(self):
        """ 检查配置参数是否合法。
        如果不合法，就直接抛出错误。 """

        if self.num_clients <= 0:
            raise ValueError("num_clients must be positive")
        if self.alpha <= 0:
            raise ValueError("alpha must be positive")
        if self.min_datasize <= 0:
            raise ValueError("min_datasize must be positive")

    def dirichlet_client_split(self, pool_indices, max_attempts=100):
        """ 把官方训练集按 Dirichlet 分布切给多个客户端，构造 non-IID 的联邦训练数据。"""

        pool_indices = np.array(pool_indices)
        pool_targets = self.train_targets[pool_indices]
        class_indices = [
            pool_indices[np.where(pool_targets == class_id)[0]]
            for class_id in range(self.num_classes)
        ]

        for _ in range(max_attempts):
            client_indices = {client_id: [] for client_id in range(1, self.num_clients + 1)}
            label_distribution = self.rng.dirichlet(
                [self.alpha] * self.num_clients,
                self.num_classes,
            )

            for class_id, class_idcs in enumerate(class_indices):
                shuffled_idcs = self.rng.permutation(class_idcs)
                split_points = (
                    np.cumsum(label_distribution[class_id])[:-1] * len(shuffled_idcs)
                ).astype(int)
                for client_id, idcs in enumerate(np.split(shuffled_idcs, split_points), start=1):
                    client_indices[client_id].extend(idcs.tolist())

            for idcs in client_indices.values():
                self.rng.shuffle(idcs)

            if min(len(idcs) for idcs in client_indices.values()) >= self.min_datasize:
                return client_indices

        raise ValueError(
            "Unable to split data with the requested min_datasize. "
            "Try increasing alpha, reducing num_clients, or lowering min_datasize in config.yaml."
        )

    def build_stats(self, meta):
        """ 根据 meta 中的划分结果，生成统计信息 stats。
        stats 主要用于：
        - 看每个集合有多少样本
        - 看每个集合的类别分布
        - 检查 non-IID 是否符合预期 """

        splits = meta["splits"]
        client_class_counts = {
            client_id: self.class_counts(indices, self.train_targets)
            for client_id, indices in splits["client_train_indices"].items()
        }

        return {
            "protocol": meta["protocol"],
            "dataset": self.data_name,
            "num_classes": self.num_classes,
            "num_clients": self.num_clients,
            "alpha": self.alpha,
            "seed": self.seed,
            "sizes": {
                "official_train": len(self.train_dataset),
                "global_test": len(splits["global_test_indices"]),
                "client_train": {
                    client_id: len(indices)
                    for client_id, indices in splits["client_train_indices"].items()
                },
            },
            "class_counts": {
                "official_train": self.class_counts(list(range(len(self.train_dataset))), self.train_targets),
                "global_test": self.class_counts(splits["global_test_indices"], self.test_targets),
                "client_train": client_class_counts,
            },
        }

    def class_counts(self, indices, targets):
        """ 统计给定索引集合中，每个类别各有多少个样本。 """
        counts = Counter(int(targets[index]) for index in indices)
        return {str(class_id): int(counts.get(class_id, 0)) for class_id in range(self.num_classes)}

    def save(self, meta, stats):
        """ 把划分结果和统计信息保存到文件。 """
        os.makedirs(self.data_save_path, exist_ok=True)
        meta_path = os.path.join(self.data_save_path, self.args.partition_meta_name)
        stats_path = os.path.join(self.data_save_path, self.args.partition_stats_name)

        torch.save(meta, meta_path)
        with open(stats_path, "w", encoding="utf-8") as stats_file:
            json.dump(stats, stats_file, ensure_ascii=False, indent=2)

        print(f"Saved partition meta to {meta_path}")
        print(f"Saved partition stats to {stats_path}")

if __name__ == '__main__':
    cli_parser = argparse.ArgumentParser(description="Build CIFAR partitions from one nested YAML config file.")
    add_config_path_arguments(cli_parser)
    cli_args = cli_parser.parse_args()

    # Load partition settings from the config.yaml passed by --config.
    args = load_args(config_path=cli_args.config)
    validate_output_paths(args, stage="data")
    set_seed(args.seed)
    CIFARPartitionBuilder(args=args).build()
