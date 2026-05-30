import argparse 
import logging 
import os
import warnings

from configs import add_config_path_arguments, load_args, validate_output_paths
from data.data import CIFARPartitionBuilder
from data.loader import load_partition_meta
from fl.server import Server
from utils.utils import get_experiment_stem, set_seed


# 忽略 warning 输出，避免训练日志被大量无关警告刷屏。
warnings.filterwarnings("ignore")


def build_logger(args):
    """ 构造日志器 logger。
    日志会同时输出到：
    1. 控制台
    2. 日志文件 """

    # 日志目录：save_result/logs
    log_dir = os.path.join(args.save_result, "logs")
    os.makedirs(log_dir, exist_ok=True)

    # 日志文件名中加入关键实验配置，方便区分不同实验结果。
    logger_name = get_experiment_stem(args)

    # Python 标准 logging 用法：
    # logger 负责统一接收日志，handler 决定日志输出到哪里。
    logger = logging.getLogger(logger_name)

    # DEBUG 级别会记录最详细的日志。
    logger.setLevel(logging.DEBUG)

    # 禁止日志继续向父 logger 传播，避免重复打印。
    logger.propagate = False

    # 如果同名 logger 已经存在 handler，先清空。
    # 这样可以避免多次运行 main 时重复输出同一条日志。
    if logger.handlers:
        logger.handlers.clear()

    # 控制台日志：训练时直接在终端输出。
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))

    # 文件日志：同时把训练过程保存到 save/result/logs/*.log。
    # mode="w" 表示每次运行都会覆盖同名旧日志。
    file_handler = logging.FileHandler(os.path.join(log_dir, f"{logger_name}.log"), mode="w")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))

    # logger 同时挂载控制台 handler 和文件 handler。
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


def log_or_print(logger, message):
    """logger 还没创建时用 print；logger 创建后用 logger.info。"""

    # 数据划分准备阶段可能发生在 logger 创建之前，
    # 所以这里做一个兼容：没有 logger 就直接 print。
    if logger is None:
        print(message)
    else:
        logger.info(message)


def need_prepare_data(args, force_repartition=False):
    """判断训练前是否需要生成或重新生成数据划分文件。"""

    # 数据划分元信息文件路径。
    meta_path = os.path.join(args.data_save_path, args.partition_meta_name)

    # 数据划分统计文件路径。
    stats_path = os.path.join(args.data_save_path, args.partition_stats_name)

    # 如果命令行指定强制重新划分，则直接返回需要重新生成。
    if force_repartition:
        return True, "force_repartition=True"

    # 如果 partition metadata 文件不存在，需要重新生成。
    if not os.path.exists(meta_path):
        return True, f"Missing partition metadata file: {meta_path}"

    # 如果 partition stats 文件不存在，也需要重新生成。
    if not os.path.exists(stats_path):
        return True, f"Missing partition stats file: {stats_path}"

    try:
        # 尝试加载已有的数据划分文件。
        # 如果文件损坏、字段不匹配或和当前配置不兼容，会在这里抛异常。
        load_partition_meta(args)
    except Exception as exc:
        return True, (
            "Existing partition files are invalid or mismatched with current config: "
            f"{exc}"
        )

    # 走到这里说明数据划分文件存在，并且能被当前配置正常读取。
    return False, "partition files exist and match current config"


def prepare_data_if_needed(args, logger=None, force_repartition=False):
    """如果数据划分不存在、损坏或与当前配置不匹配，则自动重新生成。"""

    # 先判断是否需要准备数据划分文件，并记录原因。
    should_prepare, reason = need_prepare_data(
        args=args,
        force_repartition=force_repartition,
    )

    # 如果已有数据划分文件可用，就跳过生成过程。
    if not should_prepare:
        log_or_print(logger, f"[Data Prepare] Skip partition generation: {reason}")
        return

    # 需要重新生成数据划分文件时，打印原因。
    log_or_print(logger, f"[Data Prepare] Generate partition files: {reason}")

    # 确保数据保存目录存在。
    os.makedirs(args.data_save_path, exist_ok=True)

    # 根据当前配置重新构建 CIFAR 客户端数据划分。
    CIFARPartitionBuilder(args=args).build()

    # 生成后立刻重新加载一次，用于验证划分文件是否可读、是否和当前配置匹配。
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

    # 创建命令行参数解析器。
    cli_parser = argparse.ArgumentParser(description="Train with one nested YAML config file.")

    # 添加配置文件路径参数，例如 --config configs/xxx/config.yaml。
    add_config_path_arguments(cli_parser)

    # 强制重新生成数据划分文件。
    # 即使已有 partition_meta.pt 和 partition_stats.json，也会重新划分。
    cli_parser.add_argument(
        "--force_repartition",
        action="store_true",
        help="Force regeneration of partition_meta.pt and partition_stats.json before training.",
    )

    # 关闭自动数据准备。
    # 如果指定该参数，程序不会自动检查或生成数据划分文件。
    cli_parser.add_argument(
        "--no_auto_prepare_data",
        action="store_true",
        help="Disable automatic data partition checking and generation before training.",
    )

    # 解析命令行参数。
    cli_args = cli_parser.parse_args()

    # Read experiment settings from the config.yaml passed by --config.
    # 从 YAML 配置文件中读取完整实验参数。
    args = load_args(config_path=cli_args.config)

    # 检查输出路径配置是否合法，避免训练结果被写到错误位置。
    validate_output_paths(args, stage="train")

    # 固定随机种子，尽量保证实验可复现。
    set_seed(args.seed)

    # 默认会自动检查数据划分文件。
    # 如果缺失、损坏或配置不匹配，就重新生成。
    if not cli_args.no_auto_prepare_data:
        prepare_data_if_needed(
            args=args,
            logger=None,
            force_repartition=cli_args.force_repartition,
        )

    # 创建日志器。
    # 注意：logger 在数据划分准备之后创建，所以 prepare_data_if_needed 里 logger=None。
    logger = build_logger(args)

    # 项目主入口：创建服务端对象，然后启动联邦训练流程。
    Server(args=args, logger=logger).train()


# Python 脚本入口。
# 只有直接运行该文件时才会执行 main()；
# 如果该文件被 import，则不会自动启动训练。
if __name__ == "__main__":
    main()