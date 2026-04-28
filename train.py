import argparse
import logging
import os
import warnings

from configs import add_config_path_arguments, load_args, validate_output_paths
from data.data import CIFARPartitionBuilder
from data.loader import load_partition_meta
from fl.server import Server
from utils.utils import get_experiment_stem, set_seed

warnings.filterwarnings("ignore")


def build_logger(args):
    """ 构造日志器 logger。
    日志会同时输出到：
    1. 控制台
    2. 日志文件 """

    log_dir = os.path.join(args.save_result, "logs")
    os.makedirs(log_dir, exist_ok=True)

    # 日志文件名中加入关键实验配置，方便区分不同实验结果。
    logger_name = get_experiment_stem(args)

    # Python 标准 logging 用法：
    # logger 负责统一接收日志，handler 决定日志输出到哪里。
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    if logger.handlers:
        logger.handlers.clear()

    # 控制台日志：训练时直接在终端输出。
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))

    # 文件日志：同时把训练过程保存到 save/result/logs/*.log。
    file_handler = logging.FileHandler(os.path.join(log_dir, f"{logger_name}.log"), mode="w")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger


def log_or_print(logger, message):
    """logger 还没创建时用 print；logger 创建后用 logger.info。"""

    if logger is None:
        print(message)
    else:
        logger.info(message)


def need_prepare_data(args, force_repartition=False):
    """判断训练前是否需要生成或重新生成数据划分文件。"""

    meta_path = os.path.join(args.data_save_path, args.partition_meta_name)
    stats_path = os.path.join(args.data_save_path, args.partition_stats_name)

    if force_repartition:
        return True, "force_repartition=True"

    if not os.path.exists(meta_path):
        return True, f"Missing partition metadata file: {meta_path}"

    if not os.path.exists(stats_path):
        return True, f"Missing partition stats file: {stats_path}"

    try:
        load_partition_meta(args)
    except Exception as exc:
        return True, (
            "Existing partition files are invalid or mismatched with current config: "
            f"{exc}"
        )

    return False, "partition files exist and match current config"


def prepare_data_if_needed(args, logger=None, force_repartition=False):
    """如果数据划分不存在、损坏或与当前配置不匹配，则自动重新生成。"""

    should_prepare, reason = need_prepare_data(
        args=args,
        force_repartition=force_repartition,
    )

    if not should_prepare:
        log_or_print(logger, f"[Data Prepare] Skip partition generation: {reason}")
        return

    log_or_print(logger, f"[Data Prepare] Generate partition files: {reason}")

    os.makedirs(args.data_save_path, exist_ok=True)
    CIFARPartitionBuilder(args=args).build()
    load_partition_meta(args)

    log_or_print(
        logger,
        "[Data Prepare] Partition files generated and validated successfully.",
    )


def main():
    """ 程序主函数。
    主要流程：
    1. 解析命令行参数
    2. 读取 YAML 配置
    3. 设置随机种子
    4. 自动检查 / 生成数据划分文件
    5. 创建日志器
    6. 创建 Server 并启动联邦训练 """

    cli_parser = argparse.ArgumentParser(description="Train with one nested YAML config file.")
    add_config_path_arguments(cli_parser)
    cli_parser.add_argument(
        "--force_repartition",
        action="store_true",
        help="Force regeneration of partition_meta.pt and partition_stats.json before training.",
    )
    cli_parser.add_argument(
        "--no_auto_prepare_data",
        action="store_true",
        help="Disable automatic data partition checking and generation before training.",
    )
    cli_args = cli_parser.parse_args()

    # Read experiment settings from the config.yaml passed by --config.
    args = load_args(config_path=cli_args.config)
    validate_output_paths(args, stage="train")
    set_seed(args.seed)

    if not cli_args.no_auto_prepare_data:
        prepare_data_if_needed(
            args=args,
            logger=None,
            force_repartition=cli_args.force_repartition,
        )

    logger = build_logger(args)

    # 项目主入口：创建服务端对象，然后启动联邦训练流程。
    Server(args=args, logger=logger).train()


if __name__ == "__main__":
    main()
