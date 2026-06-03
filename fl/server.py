import torch
import os
from types import SimpleNamespace
from torch import nn
from tqdm import tqdm

from data.loader import build_global_eval_loader, get_client_train_size, load_partition_meta
from fl.aggregators import build_aggregator
from fl.client import Client
from model import build_model_from_args
from utils.utils import (
    init_result_csv,
    init_server_result_csv,
    record_server_result,
    trim_result_csv_for_resume,
)


class Server:
    """ Server 表示联邦学习中的服务端。
    它不直接拿全部训练数据训练，而是：
    1. 初始化全局模型
    2. 让多个客户端各自本地训练
    3. 聚合客户端模型
    4. 最后用全局测试集评估最终模型 """

    def __init__(self, args: SimpleNamespace, logger):
        """ 初始化服务端。
        参数：
        - args: 全部配置参数
        - logger: 日志器 """

        self.args = args

        # 根据非专家参数和专家参数的聚合策略构建聚合器。
        self.aggregator = build_aggregator(self.args)

        # 基础联邦训练配置。
        self.num_clients = self.args.num_clients
        self.server_epochs = self.args.server_epochs

        # 客户端编号从 1 开始，例如 num_clients=4 时为 [1, 2, 3, 4]。
        self.clientsID_list = [i+1 for i in range(self.num_clients)]

        # 服务端使用的设备，例如 cuda 或 cpu。
        self.device = self.args.device

        # 日志器，用于记录训练过程、聚合方式、测试结果等。
        self.logger = logger

        # 每次实验启动时只打印一次关键配置，便于对照日志分析结果。
        self.log_experiment_config()

        # 确保模型保存目录存在。
        os.makedirs(self.args.model_save_path, exist_ok=True)

        # 断点续训 checkpoint 路径。
        self.checkpoint_path = os.path.join(self.args.model_save_path, "checkpoint.pth")
        self.resume = bool(getattr(self.args, "resume", False))
        self.start_round = 0
        self.final_evaluated = False

        # 加载数据划分元信息，后续客户端和服务端都复用这份划分信息。
        self.partition_meta = load_partition_meta(self.args)

        # 构造全局测试集 DataLoader。
        # 服务端用它评估每轮聚合后的全局模型性能。
        self.global_test_loader = build_global_eval_loader(
            args=self.args,
            split="global_test",
            meta=self.partition_meta,
        )

        # 初始化全局模型；断点续训时会随后加载 checkpoint 参数。
        self.init_global_model(save_initial=not self.resume)
        if self.resume:
            self.load_training_checkpoint()

        # 服务端评估分类任务时使用交叉熵损失。
        self.criterion = nn.CrossEntropyLoss()

        # 初始化 CSV 结果文件，后续客户端训练会不断追加记录。
        if self.resume:
            trim_result_csv_for_resume(
                self.args,
                completed_round=self.start_round,
                final_evaluated=self.final_evaluated,
            )
        init_result_csv(self.args)
        init_server_result_csv(self.args)


    def log_experiment_config(self):
        """打印一次实验关键配置。"""

        experiment_config = {
            "data_name": self.args.data_name,
            "batch_size": self.args.batch_size,
            "alpha": self.args.alpha,
            "seed": self.args.seed,
            "partition_impl": getattr(self.args, "partition_impl", "default"),
            "num_clients": self.args.num_clients,
            "server_epochs": self.args.server_epochs,
            "client_epochs": self.args.client_epochs,
            "model_type": self.args.model_type,
            "num_experts": self.args.num_experts,
            "top_k": self.args.top_k,
            "cnn_channels": getattr(self.args, "cnn_channels", None),
            "expert_hidden_dim": getattr(self.args, "expert_hidden_dim", None),
            "dropout": self.args.dropout,
            "learning_rate": self.args.learning_rate,
            "optimizer": getattr(self.args, "optimizer", "adam"),
            "momentum": getattr(self.args, "momentum", 0.0),
            "weight_decay": getattr(self.args, "weight_decay", 0.0),
            "grad_clip_norm": getattr(self.args, "grad_clip_norm", 0.0),
            "label_smooth": getattr(self.args, "label_smooth", 0.0),
            "aggregation_mode": getattr(self.args, "aggregation_mode", "split"),
            "non_expert_agg_method": self.args.non_expert_agg_method,
            "expert_agg_method": self.args.expert_agg_method,
            "router_aux_loss_coef": getattr(self.args, "router_aux_loss_coef", 0.0),
            "router_z_loss_coef": getattr(self.args, "router_z_loss_coef", 0.0),
            "router_balance_loss_coef": getattr(self.args, "router_balance_loss_coef", 0.0),
            "save_root": self.args.save_root,
            "run_name": self.args.run_name,
        }
        self.logger.info(f"--experiment_config : {experiment_config}\n")


    def init_global_model(self, save_initial=True):
        """ 初始化服务端全局模型。 """

        # 根据 model_type 初始化全局模型。
        self.model = build_model_from_args(self.args)

        # 普通训练初始化完成后立即保存，客户端 renew_model 可从磁盘读取。
        if save_initial:
            self.save_server_model()

    def get_server_state_dict(self):
        """将当前服务端模型参数拷贝到 CPU。"""

        return {
            key: value.detach().cpu().clone()
            for key, value in self.model.state_dict().items()
        }

    def save_server_model(self):
        """ 保存当前服务端模型参数到 server.pth。 """

        # 保存为 server.pth，供客户端下一轮同步全局模型。
        torch.save(self.get_server_state_dict(), self.args.model_save_path + f"/server.pth")

    def save_training_checkpoint(self, completed_round, final_evaluated=False):
        """保存 server round 级别的断点续训 checkpoint。"""

        checkpoint = {
            "completed_round": int(completed_round),
            "final_evaluated": bool(final_evaluated),
            "server_state_dict": self.get_server_state_dict(),
            "aggregator_state": self.aggregator.state_dict()
            if hasattr(self.aggregator, "state_dict")
            else {},
        }
        torch.save(checkpoint, self.checkpoint_path)

    def load_training_checkpoint(self):
        """从 checkpoint 恢复服务端模型和已完成轮数。"""

        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(
                "resume=True requires checkpoint file: "
                f"{self.checkpoint_path}"
            )

        checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
        if "server_state_dict" not in checkpoint or "completed_round" not in checkpoint:
            raise ValueError(
                "Invalid checkpoint: expected server_state_dict and completed_round "
                f"in {self.checkpoint_path}"
            )

        completed_round = int(checkpoint["completed_round"])
        if completed_round < 0 or completed_round > self.server_epochs:
            raise ValueError(
                "Invalid checkpoint completed_round: "
                f"{completed_round}, server_epochs={self.server_epochs}"
            )

        self.model.load_state_dict(checkpoint["server_state_dict"])
        aggregator_state = checkpoint.get("aggregator_state", {})
        if aggregator_state and hasattr(self.aggregator, "load_state_dict"):
            self.aggregator.load_state_dict(aggregator_state)

        self.start_round = completed_round
        self.final_evaluated = bool(checkpoint.get("final_evaluated", False))
        self.save_server_model()
        self.logger.info(
            f"--resume_checkpoint : {self.checkpoint_path} "
            f"--completed_round : {self.start_round} "
            f"--server_epochs : {self.server_epochs}\n"
        )


    def train(self):
        """ 服务端训练主流程。
        每一轮(global round)大致做：
        1. 依次调度每个客户端本地训练
        2. 收集客户端返回的 expert 统计
        3. 聚合客户端模型
        4. 保存 server.pth,供下一轮客户端同步
        5. 所有轮次结束后，在 global_test 上评估最终模型 """

        if self.start_round >= self.server_epochs:
            if self.final_evaluated:
                self.logger.info(
                    f"--resume_status : already completed "
                    f"--completed_round : {self.start_round}\n"
                )
                return

            self.evaluate_final_on_global_test()
            self.save_training_checkpoint(
                completed_round=self.server_epochs,
                final_evaluated=True,
            )
            return

        total_client_steps = self.server_epochs * self.num_clients
        completed_client_steps = self.start_round * self.num_clients

        # 进度条覆盖全部 server round 中的 client 训练，ETA 对应整段训练剩余时间。
        with tqdm(
            total=total_client_steps,
            initial=completed_client_steps,
            desc="training clients",
            unit="client",
            dynamic_ncols=True,
        ) as progress_bar:
            # 外层循环是一轮轮服务端通信，也就是联邦学习中的 global round。
            for c_T in range(self.start_round, self.server_epochs):
                self.logger.info(f"============================== T:{c_T+1} start !!! ===============================\n")

                # 当前轮开始前，把服务端模型参数拷贝出来。
                # 之后传给客户端，客户端本地训练前会加载这份全局参数。
                server_state_dict = {
                    key: value.detach().cpu().clone()
                    for key, value in self.model.state_dict().items()
                }

                # 统计本轮所有客户端总体 expert 使用次数。
                round_expert_usage_summary = torch.zeros(self.args.num_experts)

                # 统计本轮所有客户端按层的 expert 使用信息。
                round_layer_stats = {}

                # 保存每个客户端返回的 expert usage / Fisher score 等统计信息。
                # 聚合器会用这些信息做 expert 级别加权。
                round_client_expert_usages = []

                # 保存每个客户端本地训练后的模型参数。
                round_client_states = []

                # 保存每个客户端训练集大小，FedAvg 会用它作为聚合权重。
                round_client_sizes = []

                progress_bar.set_description(f"T:{c_T + 1}/{self.server_epochs} clients")
                for id in self.clientsID_list:
                    # 每个客户端执行本地训练，并返回本轮信息。
                    client_stats = Client(
                        args=self.args,
                        client_id=id,
                        logger=self.logger,
                        c_T=c_T,
                        partition_meta=self.partition_meta,
                        server_state_dict=server_state_dict,
                    ).train()

                    # local_state_dict 是客户端训练后的模型参数。
                    # pop 后 client_stats 中剩下的就是 expert usage / Fisher score 等统计信息。
                    client_state_dict = client_stats.pop("local_state_dict")

                    # 收集客户端模型参数，用于后续服务端聚合。
                    round_client_states.append(client_state_dict)

                    # 收集客户端训练样本数，用于 FedAvg 权重。
                    round_client_sizes.append(self.get_client_train_size(id))

                    # 当前客户端总 expert 激活次数。
                    client_expert_usage = client_stats["expert_activations"].float().cpu()

                    # 保存当前客户端 expert 统计信息，供 usage / Fisher 等专家聚合策略使用。
                    round_client_expert_usages.append(client_stats)

                    # 累加到本轮总 expert usage。
                    round_expert_usage_summary += client_expert_usage

                    # 累加按层 expert 统计信息。
                    for layer_id, stats in client_stats.get("expert_stats_by_layer", {}).items():
                        # 第一次遇到该 layer 时，初始化本轮该层统计容器。
                        if layer_id not in round_layer_stats:
                            round_layer_stats[layer_id] = {
                                "expert_activations": torch.zeros(self.args.num_experts),
                                "overflow_counts": torch.zeros(self.args.num_experts),
                                "capacity": stats.get("capacity", 0),
                            }

                        # 累加该层每个 expert 的激活次数。
                        round_layer_stats[layer_id]["expert_activations"] += stats["expert_activations"].float().cpu()

                        # 累加该层每个 expert 的 overflow 次数。
                        round_layer_stats[layer_id]["overflow_counts"] += stats["overflow_counts"].float().cpu()

                        # capacity 表示该层 expert 容量配置，不是累计量，直接保留即可。
                        round_layer_stats[layer_id]["capacity"] = stats.get("capacity", round_layer_stats[layer_id]["capacity"])

                    progress_bar.update(1)

                # 将本轮总 expert usage 转成 int list，方便日志阅读。
                usage_list = [int(v) for v in round_expert_usage_summary.tolist()]
                self.logger.info(f"--round_expert_usage_summary : {usage_list}\n")

                usage_sum = round_expert_usage_summary.sum().item()
                if usage_sum > 0:
                    usage_fraction = round_expert_usage_summary / usage_sum
                    router_usage_imbalance = self.args.num_experts * torch.sum(usage_fraction ** 2).item()
                else:
                    router_usage_imbalance = 0.0
                self.logger.info(f"--router_usage_imbalance : {router_usage_imbalance:.4f}\n")

                # 保存最近一轮所有客户端的 expert 统计。
                # aggregation_by_method 中会通过 self.last_client_expert_usages 传给聚合器。
                self.last_client_expert_usages = round_client_expert_usages

                # 每个客户端各自的 expert usage，便于检查 expert 是否极度不均衡。
                client_usage_list = [
                    [int(v) for v in stats["expert_activations"].tolist()]
                    for stats in round_client_expert_usages
                ]

                # 简单 CNN MoE 没有 overflow / capacity，日志里只展示 expert 激活。
                if self.args.model_type == "simple_cnn_moe_head":
                    layer_stats_log = {
                        layer_id: {
                            "expert_activations": [int(v) for v in stats["expert_activations"].tolist()],
                        }
                        for layer_id, stats in round_layer_stats.items()
                    }
                else:
                    # 其他模型保留完整按层 expert 统计日志。
                    layer_stats_log = {
                        layer_id: {
                            "expert_activations": [int(v) for v in stats["expert_activations"].tolist()],
                            "overflow_counts": [int(v) for v in stats["overflow_counts"].tolist()],
                            "capacity": int(stats["capacity"]),
                        }
                        for layer_id, stats in round_layer_stats.items()
                    }

                # 打印客户端级别和层级别的 expert 使用情况。
                self.logger.info(f"--client_expert_usage_summary : {client_usage_list}\n")
                self.logger.info(f"--round_expert_stats_by_layer : {layer_stats_log}\n")

                # 所有客户端本地训练完成后，服务端通过聚合器更新全局模型。
                self.aggregation(
                    client_states=round_client_states,
                    client_sizes=round_client_sizes,
                )

                # 每轮结束保存当前服务端模型，供下一轮客户端同步。
                self.save_server_model()

                # 每轮聚合后在 global_test 上评估一次，用来观察训练曲线。
                self.evaluate_round_on_global_test(round_id=c_T + 1)

                # checkpoint 记录已经完整完成的 server round。
                self.save_training_checkpoint(completed_round=c_T + 1)

        # 所有 server round 结束后，再做一次最终测试。
        self.evaluate_final_on_global_test()
        self.save_training_checkpoint(
            completed_round=self.server_epochs,
            final_evaluated=True,
        )

    def evaluate_global_model(self, data_loader):
        """ 用给定的数据集(global_test)评估当前服务端模型。
        返回：
        - eval_loss
        - eval_acc """

        # 将服务端模型移动到指定设备并切换到 eval 模式。
        self.model.to(self.device)
        self.model.eval()

        # 评估统计留在 GPU 累计，循环结束后统一拷回 CPU。
        eval_metric_sums = torch.zeros(2, dtype=torch.float64, device=self.device)

        # 评估阶段不需要反向传播，因此关闭梯度计算。
        with torch.no_grad():
            for inputs, labels in data_loader:
                # 将输入和标签移动到评估设备。
                inputs = inputs.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                # 前向推理，兼容字典结果和直接返回的 logits。
                result = self.model(inputs)
                outputs = result["logits"] if isinstance(result, dict) else result

                # 计算当前 batch 的分类损失。
                loss = self.criterion(outputs, labels)

                # 取 logits 最大值对应类别作为预测类别。
                _, preds = torch.max(outputs, 1)

                # 累加总损失和正确样本数，不在 batch 循环内同步 GPU。
                eval_metric_sums += torch.stack(
                    (
                        loss.detach() * inputs.size(0),
                        torch.sum(preds == labels.data),
                    )
                ).to(dtype=torch.float64)

        # 计算整个评估集上的平均 loss 和准确率。
        running_loss, running_corrects = eval_metric_sums.detach().cpu().tolist()
        eval_loss = running_loss / len(data_loader.dataset)
        eval_acc = running_corrects / len(data_loader.dataset)

        # 评估结束后将模型移回 CPU，减少显存占用。
        self.model.to("cpu")

        return eval_loss, eval_acc

    def evaluate_round_on_global_test(self, round_id):
        """每轮聚合后在 global_test 上评估一次，仅用于监控训练曲线。"""

        # 用当前全局模型评估 global_test。
        test_loss, test_acc = self.evaluate_global_model(self.global_test_loader)

        # 写入日志，方便观察每一轮的全局测试性能。
        self.logger.info(
            f"--round_global_test_loss : {test_loss:.4f} "
            f"--round_global_test_acc : {test_acc:.4f} "
            f"--round : {round_id}\n"
        )

        # 写入服务端结果文件。
        record_server_result(
            {
                "phase": "round_test",
                "round": round_id,
                "test_loss": test_loss,
                "test_acc": test_acc,
                "selected_round": round_id,
            },
            self.args,
        )

    def evaluate_final_on_global_test(self):
        """ 在所有训练轮次结束后，用最终服务端模型在 global_test 上评估一次。 """

        # 用最终全局模型评估 global_test。
        test_loss, test_acc = self.evaluate_global_model(self.global_test_loader)

        # 打印最终测试结果。
        self.logger.info(
            f"--final_global_test_loss : {test_loss:.4f} "
            f"--final_global_test_acc : {test_acc:.4f} "
            f"--selected_round : {self.server_epochs}\n"
        )

        # 写入最终测试结果。
        record_server_result(
            {
                "phase": "final_test",
                "round": self.server_epochs,
                "test_loss": test_loss,
                "test_acc": test_acc,
                "selected_round": self.server_epochs,
            },
            self.args,
        )

    def get_client_train_size(self,client_id):
        """ 获取某个客户端训练样本数。
        FedAvg 会把这个作为聚合权重。 """

        # FedAvg 使用客户端训练样本数作为聚合权重。
        return get_client_train_size(self.args, client_id, meta=self.partition_meta)

    def aggregation_by_method(self, client_states=None, client_sizes=None):
        """聚合器接口：分别按非专家参数策略和专家参数策略执行参数聚合。"""

        if client_states is None:
            # 如果没有从内存传入客户端模型参数，就从磁盘读取每个客户端保存的 .pth。
            self.logger.info("--client_state_transport : disk\n")
            client_states = []
            for id in self.clientsID_list:
                client_state_dict = torch.load(
                    self.args.model_save_path + f"/{id}.pth",
                    map_location="cpu",
                )
                client_states.append(client_state_dict)
        else:
            # 当前代码主流程会直接从内存传入客户端参数，避免反复读写磁盘。
            self.logger.info("--client_state_transport : memory\n")

        if client_sizes is None:
            # 如果没有传入客户端样本数，就现场读取每个客户端训练集大小。
            client_sizes = [
                self.get_client_train_size(id)
                for id in self.clientsID_list
            ]

        # 所有客户端训练样本数之和，样本数加权聚合会使用它。
        total_size = sum(client_sizes)
        if total_size <= 0:
            raise ValueError("Aggregation requires at least one training sample across clients")

        # 调用具体聚合器完成聚合。
        # client_updates：客户端本地训练后的模型参数；
        # client_weights：客户端训练样本数；
        # global_model：当前服务端模型，用于某些聚合方法在无有效更新时保留旧参数；
        # client_stats：客户端 expert usage / Fisher score 等统计信息。
        client_stats = getattr(self, "last_client_expert_usages", None)
        aggregated_state = self.aggregator.aggregate(
            client_updates=client_states,
            client_weights=client_sizes,
            global_model=self.model,
            client_stats=client_stats,
            expert_weights=client_stats,
        )

        # 将聚合后的参数加载回服务端模型，完成本轮全局模型更新。
        self.model.load_state_dict(aggregated_state)

        history_summary = getattr(self.aggregator, "last_history_wolf_summary", None)
        if (
            self.args.expert_agg_method
            in {"history_wolf_filter", "fisher_history_wolf"}
            and history_summary
        ):
            compact_keys = (
                "num_experts",
                "num_clients",
                "valid_contrib_count",
                "skipped_zero_delta_count",
                "history_wolf_quality_variant",
                "mean_usage_conf",
                "mean_q",
                "mean_mu_eff",
                "mean_direction",
                "mean_direction_cosine",
                "mean_direction_has_reference",
                "mean_positive_cosine_rate",
                "mean_negative_cosine_rate",
                "mean_quality_direction_component",
                "mean_quality_magnitude_component",
                "mean_quality_usage_component",
                "mean_conflict_direction",
                "mean_magnitude",
                "mean_filter_raw",
                "mean_final_raw",
                "weight_ref_count",
                "weight_all_zero_ref_count",
                "mean_nonzero_clients_per_ref",
                "mean_top1_share",
                "mean_ess_ratio",
                "mean_raw_weight_cv",
            )
            compact_summary = {
                key: history_summary.get(key, 0.0)
                for key in compact_keys
            }
            quadrants = history_summary.get("quadrants")
            if quadrants is not None:
                compact_summary["quadrant_counts"] = {
                    name: values.get("count", 0)
                    for name, values in quadrants.items()
                }
                compact_summary["quadrant_mean_final_raw"] = {
                    name: values.get("mean_final_raw", 0.0)
                    for name, values in quadrants.items()
                }
                compact_summary["quadrant_mean_q"] = {
                    name: values.get("mean_q", 0.0)
                    for name, values in quadrants.items()
                }
                compact_summary["quadrant_mean_mu_eff"] = {
                    name: values.get("mean_mu_eff", 0.0)
                    for name, values in quadrants.items()
                }
                compact_summary["quadrant_mean_positive_cosine_rate"] = {
                    name: values.get("mean_positive_cosine_rate", 0.0)
                    for name, values in quadrants.items()
                }
                compact_summary["quadrant_mean_negative_cosine_rate"] = {
                    name: values.get("mean_negative_cosine_rate", 0.0)
                    for name, values in quadrants.items()
                }
            if history_summary.get("use_fisher"):
                compact_summary["mean_fisher_multiplier"] = history_summary.get(
                    "mean_fisher_multiplier",
                    0.0,
                )
                if quadrants is not None:
                    compact_summary["quadrant_mean_fisher_multiplier"] = {
                        name: values.get("mean_fisher_multiplier", 0.0)
                        for name, values in quadrants.items()
                    }

            # 判读：quadrant_counts 过度集中，说明四象限没有真正发挥区分作用。
            # 各象限 mean_final_raw 接近，说明分类后最终权重仍未明显拉开。
            # mean_ess_ratio 接近 1 且 mean_raw_weight_cv 很低，说明权重接近 uniform。
            # weight_all_zero_ref_count 很高，说明可能存在 router collapse / dead expert。
            # current_bad_history_good 高于 current_bad_history_bad，说明历史会保留可靠客户端。
            # current_good_history_bad 高于 current_bad_history_bad，说明当前质量会给历史差的客户端重新机会。
            self.logger.info(f"--history_wolf_filter_summary : {compact_summary}\n")

        # 打印当前聚合方法和客户端样本数，方便检查实验配置。
        self.logger.info(f"--aggregation_mode : {getattr(self.args, 'aggregation_mode', 'split')}\n")
        self.logger.info(f"--aggregation_method : {self.args.agg_method}\n")
        self.logger.info(f"--non_expert_agg_method : {self.args.non_expert_agg_method}\n")
        self.logger.info(f"--expert_agg_method : {self.args.expert_agg_method}\n")
        self.logger.info(f"--client_train_sizes : {client_sizes}\n")

    def aggregation(self, client_states=None, client_sizes=None):
        """ 聚合入口函数。
        现在只是简单调用 aggregation_by_method()，
        后续如果想扩展多种聚合流程，可以在这里继续封装。 """

        # 当前版本没有额外封装，直接调用按方法聚合的实现。
        self.aggregation_by_method(
            client_states=client_states,
            client_sizes=client_sizes,
        )