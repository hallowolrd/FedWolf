import csv 
import os 
import random
import re

import numpy as np
import torch


def set_seed(seed:int):
    """Set Python, NumPy, Torch, and CUDA seeds from one project-level value."""

    # 固定 Python 哈希种子，避免部分依赖哈希顺序的操作出现随机性。
    os.environ["PYTHONHASHSEED"] = str(seed)

    # 固定 Python 内置 random 模块的随机种子。
    random.seed(seed)

    # 固定 NumPy 的随机种子。
    np.random.seed(seed)

    # 固定 PyTorch CPU 随机种子。
    torch.manual_seed(seed)

    # 如果使用 CUDA，则同时固定所有 GPU 的随机种子。
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 关闭 cudnn 自动寻找最快算法，减少非确定性。
    torch.backends.cudnn.benchmark = False

    # 强制 cudnn 使用确定性算法，增强实验可复现性。
    torch.backends.cudnn.deterministic = True


def get_experiment_stem(args):
    # 对 run_name 做清洗：
    # 只保留字母、数字、下划线、点和横线；
    # 其他字符统一替换成下划线，避免文件名非法或混乱。
    run_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(args.run_name).strip())

    # 根据关键实验配置拼接实验名。
    # 这个 stem 后续会用于日志文件名、CSV 文件名等。
    stem = (
        f"data_{args.data_name}_"
        f"clients_{args.num_clients}_"
        f"alpha_{args.alpha}_"
        f"seed_{args.seed}_"
        f"agg_{args.agg_method}_"
        f"model_{args.model_type}"
    )

    # 把用户自定义的 run_name 也拼到实验名里，方便区分同配置下的不同实验。
    stem += f"_run_{run_name}"

    return stem


def get_csv_path(args):
    # 读取参数后，拼出本次实验的 CSV 结果文件路径。
    # CSV 文件会记录每一轮、每个客户端的 loss/acc。
    detail_dir = os.path.join(args.save_result, "detail")
    filename = f"{get_experiment_stem(args)}.csv"
    return os.path.join(detail_dir, filename)


def get_server_csv_path(args):
    # 拼出服务端结果 CSV 的保存路径。
    # 该文件主要记录每轮 / 最终的 global_test 结果。
    server_dir = os.path.join(args.save_result, "server")
    filename = f"{get_experiment_stem(args)}.csv"
    return os.path.join(server_dir, filename)


def _keep_existing_csv_for_resume(args, csv_path):
    return (
        bool(getattr(args, "resume", False))
        and os.path.exists(csv_path)
        and os.path.getsize(csv_path) > 0
    )


def _read_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _rewrite_csv(csv_path, fieldnames, keep_row):
    if not os.path.exists(csv_path):
        return

    with open(csv_path, "r", newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        rows = [row for row in reader if keep_row(row)]

    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def trim_result_csv_for_resume(args, completed_round, final_evaluated=False):
    """裁掉 checkpoint 之后的 CSV 记录，避免续训重复统计未完成轮。"""

    completed_round = int(completed_round)

    client_fieldnames = [
        "T",
        "client_epoch",
        "client_id",
        "train_loss",
        "train_acc",
        "router_aux_loss",
        "router_z_loss",
    ]
    _rewrite_csv(
        get_csv_path(args),
        client_fieldnames,
        lambda row: (
            _read_int(row.get("T")) is not None
            and _read_int(row.get("T")) < completed_round
        ),
    )

    server_fieldnames = [
        "phase",
        "round",
        "test_loss",
        "test_acc",
        "selected_round",
    ]

    def keep_server_row(row):
        phase = row.get("phase")
        round_id = _read_int(row.get("round"))
        if phase == "round_test":
            return round_id is not None and round_id <= completed_round
        if phase == "final_test":
            return bool(final_evaluated) and round_id == getattr(args, "server_epochs", None)
        return False

    _rewrite_csv(
        get_server_csv_path(args),
        server_fieldnames,
        keep_server_row,
    )


def init_result_csv(args):
    """初始化结果 CSV，写入表头。

    Server 初始化时会调用一次；断点续训时会保留已有 CSV 并继续追加。
    """

    # 获取客户端训练详细结果 CSV 路径。
    csv_path = get_csv_path(args)

    # 确保 CSV 所在目录存在。
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    if _keep_existing_csv_for_resume(args, csv_path):
        return

    # 以写入模式打开文件。
    # mode='w' 会覆盖同名旧文件，因此每次新训练都会重新写表头。
    with open(csv_path, 'w', newline='') as csvfile:
        # 客户端训练结果 CSV 的字段。
        fieldnames = ['T', 'client_epoch', 'client_id',"train_loss","train_acc","router_aux_loss","router_z_loss"]

        # 使用 DictWriter 按字段名写入字典格式记录。
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # 写入表头。
        writer.writeheader()


def init_server_result_csv(args):
    """初始化服务端结果 CSV。

    当前协议记录每轮 / 最终 global_test 评估；断点续训时继续追加。"""

    # 获取服务端结果 CSV 路径。
    csv_path = get_server_csv_path(args)

    # 确保 CSV 所在目录存在。
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    if _keep_existing_csv_for_resume(args, csv_path):
        return

    # 以写入模式打开文件，重新运行训练时会覆盖同名旧文件。
    with open(csv_path, 'w', newline='') as csvfile:
        # 服务端结果 CSV 的字段。
        fieldnames = [
            'phase',
            'round',
            'test_loss',
            'test_acc',
            'selected_round',
        ]

        # 创建 CSV 字典写入器。
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # 写入表头。
        writer.writeheader()


def record_result(record_dic:dict, args):
    """追加写入一条客户端训练记录。"""

    # 获取客户端训练详细结果 CSV 路径。
    csv_path = get_csv_path(args)

    # 以追加模式打开文件，不覆盖已有记录。
    with open(csv_path, 'a', newline='') as csvfile:
        # 字段顺序需要和 init_result_csv 中保持一致。
        fieldnames = ['T', 'client_epoch', 'client_id', "train_loss", "train_acc", "router_aux_loss", "router_z_loss"]

        # 创建 CSV 字典写入器。
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # 追加写入一条客户端训练记录。
        writer.writerow(record_dic)


def record_server_result(record_dic:dict, args):
    """追加写入一条服务端结果记录。"""

    # 获取服务端结果 CSV 路径。
    csv_path = get_server_csv_path(args)

    # 以追加模式打开文件，不覆盖已有记录。
    with open(csv_path, 'a', newline='') as csvfile:
        # 字段顺序需要和 init_server_result_csv 中保持一致。
        fieldnames = [
            'phase',
            'round',
            'test_loss',
            'test_acc',
            'selected_round',
        ]

        # 创建 CSV 字典写入器。
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # 追加写入一条服务端测试结果记录。
        writer.writerow(record_dic)