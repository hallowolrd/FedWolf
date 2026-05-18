import csv
import os
import random
import re

import numpy as np
import torch


def set_seed(seed:int):
    """根据项目级 seed 设置 Python、NumPy、Torch 和 CUDA 随机种子。"""

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_experiment_stem(args):
    run_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(args.run_name).strip())
    stem = (
        f"data_{args.data_name}_"
        f"clients_{args.num_clients}_"
        f"alpha_{args.alpha}_"
        f"seed_{args.seed}_"
        f"agg_{args.agg_method}_"
        f"model_{args.model_type}"
    )
    if args.model_type == "switch_transformer":
        patch_size = getattr(args, "patch_size", None)
        patch_tag = "auto" if patch_size is None else str(patch_size)
        stem += f"_patch_{patch_tag}"
    stem += f"_run_{run_name}"
    return stem


def get_csv_path(args):
    # 读取参数后，拼出本次实验的 CSV 结果文件路径。
    # CSV 文件会记录每一轮、每个客户端的 loss/acc。
    detail_dir = os.path.join(args.save_result, "detail")
    filename = f"{get_experiment_stem(args)}.csv"
    return os.path.join(detail_dir, filename)


def get_server_csv_path(args):
    server_dir = os.path.join(args.save_result, "server")
    filename = f"{get_experiment_stem(args)}.csv"
    return os.path.join(server_dir, filename)


def get_timing_csv_path(args):
    timing_dir = os.path.join(args.save_result, "timing")
    filename = f"{get_experiment_stem(args)}.csv"
    return os.path.join(timing_dir, filename)

def init_result_csv(args):
    """初始化结果 CSV，写入表头。

    Server 初始化时会调用一次，所以每次重新运行训练会覆盖同名 CSV。
    """

    csv_path = get_csv_path(args)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = [
            'T',
            'client_epoch',
            'client_id',
            'learning_rate',
            "train_loss",
            "train_acc",
            "router_aux_loss",
            "router_z_loss",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()


def init_server_result_csv(args):
    """初始化服务端结果 CSV。

    记录每轮 round_test global_test 监控结果，以及训练结束后的 final_test。
    """

    csv_path = get_server_csv_path(args)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = [
            'phase',
            'round',
            'test_loss',
            'test_acc',
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

def init_timing_csv(args):
    """初始化每轮耗时 CSV，写入表头。

    记录每轮 client loop、aggregation、server save、round eval 和总耗时。
    """

    csv_path = get_timing_csv_path(args)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = [
            'phase',
            'round',
            'num_clients',
            'client_loop_sec',
            'aggregation_sec',
            'save_server_sec',
            'round_eval_sec',
            'final_eval_sec',
            'round_total_sec',
            'cumulative_train_sec',
            'avg_round_sec',
            'eta_sec',
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()


def record_timing_result(record_dic: dict, args):
    """追加写入一条 timing 记录。"""

    csv_path = get_timing_csv_path(args)
    with open(csv_path, 'a', newline='') as csvfile:
        fieldnames = [
            'phase',
            'round',
            'num_clients',
            'client_loop_sec',
            'aggregation_sec',
            'save_server_sec',
            'round_eval_sec',
            'final_eval_sec',
            'round_total_sec',
            'cumulative_train_sec',
            'avg_round_sec',
            'eta_sec',
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow(record_dic)


def record_result(record_dic:dict, args):
    """追加写入一条客户端训练记录。"""

    csv_path = get_csv_path(args)
    with open(csv_path, 'a', newline='') as csvfile:
        fieldnames = [
            'T',
            'client_epoch',
            'client_id',
            'learning_rate',
            "train_loss",
            "train_acc",
            "router_aux_loss",
            "router_z_loss",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow(record_dic)


def record_server_result(record_dic:dict, args):
    """追加写入一条服务端结果记录。"""

    csv_path = get_server_csv_path(args)
    with open(csv_path, 'a', newline='') as csvfile:
        fieldnames = [
            'phase',
            'round',
            'test_loss',
            'test_acc',
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow(record_dic)
