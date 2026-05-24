import os
import sys
import time
import math
from types import SimpleNamespace

import torch
from torch import nn

try:
    from tqdm.auto import tqdm
except ModuleNotFoundError:
    class tqdm:
        def __init__(
            self,
            total=None,
            desc=None,
            disable=False,
            leave=True,
            dynamic_ncols=True,
            **kwargs,
        ):
            self.total = int(total or 0)
            self.desc = desc or ""
            self.disable = bool(disable)
            self.leave = bool(leave)
            self.n = 0
            self.postfix = {}
            self.start_time = time.time()
            if not self.disable:
                self._render()

        def update(self, n=1):
            self.n += int(n)
            if not self.disable:
                self._render()

        def set_postfix(self, ordered_dict=None, **kwargs):
            self.postfix = ordered_dict or kwargs
            if not self.disable:
                self._render()

        def close(self):
            if self.disable:
                return
            if self.leave:
                sys.stderr.write("\n")
            else:
                sys.stderr.write("\r\033[K")
            sys.stderr.flush()

        def _render(self):
            elapsed = max(time.time() - self.start_time, 1e-12)
            if self.total > 0:
                fraction = min(max(self.n / self.total, 0.0), 1.0)
                percent = 100.0 * fraction
                eta = elapsed * (self.total - self.n) / max(self.n, 1) if self.n > 0 else 0.0
                progress = f"{self.n}/{self.total} {percent:6.2f}% ETA {eta:6.1f}s"
            else:
                progress = f"{self.n} steps"
            postfix = ""
            if self.postfix:
                postfix = " | " + ", ".join(f"{key}={value}" for key, value in self.postfix.items())
            sys.stderr.write(f"\r{self.desc}: {progress}{postfix}")
            sys.stderr.flush()

from data.loader import build_global_eval_loader, get_client_train_size, load_partition_meta
from fl.aggregators import build_aggregator, parse_expert_ref_from_key
from fl.client import Client
from model import build_model_from_args
from utils.checkpoint import load_training_checkpoint, save_training_checkpoint
from utils.utils import (
    init_result_csv,
    init_server_result_csv,
    init_timing_csv,
    record_server_result,
    record_timing_result,
)


def _format_fedwolf_summary_value(value, integer=False):
    if value is None:
        return "None"
    if isinstance(value, str):
        return value
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if integer:
        return str(int(round(numeric_value)))
    return f"{numeric_value:.12e}"


SERVER_UPDATE_NORM_SUMMARY_KEYS = [
    "expert_update_norm",
    "nonexpert_update_norm",
    "total_update_norm",
    "expert_update_norm_ratio",
    "expert_update_energy_ratio",
    "expert_tensor_count",
    "nonexpert_tensor_count",
    "expert_numel",
    "nonexpert_numel",
    "skipped_nonfloat_count",
    "skipped_missing_count",
    "skipped_shape_mismatch_count",
]

SERVER_UPDATE_NORM_INTEGER_FIELDS = {
    "expert_tensor_count",
    "nonexpert_tensor_count",
    "expert_numel",
    "nonexpert_numel",
    "skipped_nonfloat_count",
    "skipped_missing_count",
    "skipped_shape_mismatch_count",
}


def compute_update_norm_summary(old_state, new_state, eps=1e-12):
    expert_update_norm_sq = 0.0
    nonexpert_update_norm_sq = 0.0
    expert_tensor_count = 0
    nonexpert_tensor_count = 0
    expert_numel = 0
    nonexpert_numel = 0
    skipped_nonfloat_count = 0
    skipped_missing_count = 0
    skipped_shape_mismatch_count = 0

    keys = list(old_state.keys())
    for key in new_state.keys():
        if key not in old_state:
            keys.append(key)

    for key in keys:
        if key not in old_state or key not in new_state:
            skipped_missing_count += 1
            continue

        old_tensor = old_state[key]
        new_tensor = new_state[key]
        if not isinstance(old_tensor, torch.Tensor) or not isinstance(new_tensor, torch.Tensor):
            continue
        if not torch.is_floating_point(old_tensor) or not torch.is_floating_point(new_tensor):
            skipped_nonfloat_count += 1
            continue
        if old_tensor.shape != new_tensor.shape:
            skipped_shape_mismatch_count += 1
            continue

        old_f = old_tensor.detach().to(device="cpu", dtype=torch.float32)
        new_f = new_tensor.detach().to(device="cpu", dtype=torch.float32)
        diff = new_f - old_f
        update_norm_sq = float(diff.pow(2).sum().item())
        numel = int(new_tensor.numel())

        if parse_expert_ref_from_key(key) is not None:
            expert_update_norm_sq += update_norm_sq
            expert_tensor_count += 1
            expert_numel += numel
        else:
            nonexpert_update_norm_sq += update_norm_sq
            nonexpert_tensor_count += 1
            nonexpert_numel += numel

    total_update_norm_sq = expert_update_norm_sq + nonexpert_update_norm_sq
    expert_update_norm = math.sqrt(expert_update_norm_sq)
    nonexpert_update_norm = math.sqrt(nonexpert_update_norm_sq)
    total_update_norm = math.sqrt(total_update_norm_sq)

    return {
        "expert_update_norm": expert_update_norm,
        "nonexpert_update_norm": nonexpert_update_norm,
        "total_update_norm": total_update_norm,
        "expert_update_norm_ratio": expert_update_norm / (total_update_norm + eps),
        "expert_update_energy_ratio": expert_update_norm_sq / (total_update_norm_sq + eps),
        "expert_tensor_count": expert_tensor_count,
        "nonexpert_tensor_count": nonexpert_tensor_count,
        "expert_numel": expert_numel,
        "nonexpert_numel": nonexpert_numel,
        "skipped_nonfloat_count": skipped_nonfloat_count,
        "skipped_missing_count": skipped_missing_count,
        "skipped_shape_mismatch_count": skipped_shape_mismatch_count,
    }


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
        self.aggregator = build_aggregator(self.args)
        # 基础联邦训练配置。
        self.num_clients = self.args.num_clients
        self.server_epochs = self.args.server_epochs
        self.start_round = 0
        # 客户端编号从 1 开始，例如 num_clients=4 时为 [1, 2, 3, 4]。
        self.clientsID_list = [i+1 for i in range(self.num_clients)]
        self.device = self.args.device
        self.logger = logger
        self.logger.info(f"--aggregation_device : {getattr(self.aggregator, 'aggregation_device', 'cpu')}\n")
        self.logger.info(f"--dataloader_seed_mode : {getattr(self.args, 'dataloader_seed_mode', 'legacy')}\n")
        self.logger.info(f"--dataloader_base_seed : {getattr(self.args, 'seed', None)}\n")
        os.makedirs(self.args.model_save_path, exist_ok=True)
        self.partition_meta = load_partition_meta(self.args)
        self.client_cache = {}
        self.global_test_loader = build_global_eval_loader(
            args=self.args,
            split="global_test",
            meta=self.partition_meta,
        )
        # 初始化全局模型，并保存到 server.pth。
        self.init_global_model()
        if bool(getattr(self.args, "resume", False)):
            completed_round = load_training_checkpoint(
                self.args,
                self.model,
                logger=self.logger,
            )
            self.start_round = int(completed_round)
            self.save_server_model()
        self.criterion = nn.CrossEntropyLoss()
        # 初始化 CSV 结果文件，后续客户端训练会不断追加记录。
        append_existing = bool(getattr(self.args, "resume", False))
        init_result_csv(self.args, append_existing=append_existing)
        init_server_result_csv(self.args, append_existing=append_existing)
        init_timing_csv(self.args, append_existing=append_existing)


    def get_or_create_client(self, client_id):
        if client_id not in self.client_cache:
            self.client_cache[client_id] = Client(
                args=self.args,
                client_id=client_id,
                logger=self.logger,
                c_T=0,
                partition_meta=self.partition_meta,
                server_state_dict=None,
            )
        return self.client_cache[client_id]

    def init_global_model(self):
        """ 初始化服务端全局模型。 """

        # 根据 model_type 初始化全局模型。
        self.model = build_model_from_args(self.args)
        # 初始化完成后立即保存，客户端 renew_model 时会读取这个文件。
        self.save_server_model()

    def save_server_model(self):
        """ 保存当前服务端模型参数到 server.pth。 """

        cpu_state_dict = {
            key: value.detach().cpu().clone()
            for key, value in self.model.state_dict().items()
        }
        torch.save(cpu_state_dict, self.args.model_save_path + f"/server.pth")

    def _progress_enabled(self):
        return bool(getattr(self.args, "show_progress", True))

    def _progress_leave(self):
        return bool(getattr(self.args, "progress_leave", True))

    def train(self):
        """ 服务端训练主流程。
        每一轮(global round)大致做：
        1. 依次调度每个客户端本地训练
        2. 收集客户端返回的 expert 统计
        3. 聚合客户端模型
        4. 保存 server.pth,供下一轮客户端同步
        5. 所有轮次结束后，在 global_test 上评估最终模型 """

        train_start_time = time.perf_counter()
        round_total_sec_acc = 0.0
        completed_this_run_rounds = 0

        num_clients = len(self.clientsID_list)
        if self.start_round >= self.server_epochs:
            self.logger.info(
                f"--resume_already_complete : completed_round={self.start_round} "
                f"target_rounds={self.server_epochs}"
            )
            return

        steps_per_round = num_clients + 2
        remaining_rounds = self.server_epochs - self.start_round
        total_steps = remaining_rounds * steps_per_round + 1
        progress_bar = tqdm(
            total=total_steps,
            desc="Total training progress",
            disable=not self._progress_enabled(),
            leave=self._progress_leave(),
            dynamic_ncols=True,
        )

        try:
            # 外层循环是一轮轮服务端通信，也就是联邦学习中的 global round。
            for c_T in range(self.start_round, self.server_epochs):
                round_id = c_T + 1
                round_start_time = time.perf_counter()
                self.logger.info(f"============================== T:{round_id} start !!! ===============================\n")
                server_state_dict = {
                    key: value.detach().cpu().clone()
                    for key, value in self.model.state_dict().items()
                }
                round_expert_usage_summary = torch.zeros(self.args.num_experts)
                round_layer_stats = {}
                round_client_expert_usages = []
                round_client_states = []
                round_client_sizes = []
                client_loop_start = time.perf_counter()
                for id in self.clientsID_list:
                    # 每个客户端执行本地训练，并返回本轮信息。
                    client = self.get_or_create_client(id)
                    client_stats = client.train(
                        c_T=c_T,
                        server_state_dict=server_state_dict,
                    )
                    client_state_dict = client_stats.pop("local_state_dict")
                    round_client_states.append(client_state_dict)
                    round_client_sizes.append(self.get_client_train_size(id))
                    client_expert_usage = client_stats["expert_activations"].float().cpu()
                    round_client_expert_usages.append(client_stats)
                    round_expert_usage_summary += client_expert_usage
                    for layer_id, stats in client_stats.get("expert_stats_by_layer", {}).items():
                        if layer_id not in round_layer_stats:
                            round_layer_stats[layer_id] = {
                                "expert_activations": torch.zeros(self.args.num_experts),
                                "overflow_counts": torch.zeros(self.args.num_experts),
                                "capacity": stats.get("capacity", 0),
                            }
                        round_layer_stats[layer_id]["expert_activations"] += stats["expert_activations"].float().cpu()
                        round_layer_stats[layer_id]["overflow_counts"] += stats["overflow_counts"].float().cpu()
                        round_layer_stats[layer_id]["capacity"] = stats.get("capacity", round_layer_stats[layer_id]["capacity"])
                    progress_bar.update(1)
                    progress_bar.set_postfix({
                        "round": f"{c_T + 1}/{self.server_epochs}",
                        "stage": "client",
                        "client": id,
                    })

                usage_list = [int(v) for v in round_expert_usage_summary.tolist()]
                self.logger.info(f"--round_expert_usage_summary : {usage_list}\n")
                self.last_client_expert_usages = round_client_expert_usages
                client_usage_list = [
                    [int(v) for v in stats["expert_activations"].tolist()]
                    for stats in round_client_expert_usages
                ]
                layer_stats_log = {
                    layer_id: {
                        "expert_activations": [int(v) for v in stats["expert_activations"].tolist()],
                        "overflow_counts": [int(v) for v in stats["overflow_counts"].tolist()],
                        "capacity": int(stats["capacity"]),
                    }
                    for layer_id, stats in round_layer_stats.items()
                }
                self.logger.info(f"--client_expert_usage_summary : {client_usage_list}\n")
                self.logger.info(f"--round_expert_stats_by_layer : {layer_stats_log}\n")
                client_loop_sec = time.perf_counter() - client_loop_start
                # 所有客户端本地训练完成后，服务端通过聚合器更新全局模型。
                aggregation_start = time.perf_counter()
                self.aggregation(
                    client_states=round_client_states,
                    client_sizes=round_client_sizes,
                    old_server_state=server_state_dict,
                    round_id=round_id,
                )
                aggregation_sec = time.perf_counter() - aggregation_start

                # 每轮结束保存当前服务端模型，供下一轮客户端同步。
                save_server_start = time.perf_counter()
                self.save_server_model()
                save_server_sec = time.perf_counter() - save_server_start
                progress_bar.update(1)
                progress_bar.set_postfix({
                    "round": f"{round_id}/{self.server_epochs}",
                    "stage": "aggregation",
                })

                round_eval_start = time.perf_counter()
                self.evaluate_round_on_global_test(round_id=round_id)
                round_eval_sec = time.perf_counter() - round_eval_start
                progress_bar.update(1)
                progress_bar.set_postfix({
                    "round": f"{round_id}/{self.server_epochs}",
                    "stage": "round_eval",
                })
                torch.cuda.empty_cache()
                checkpoint_every = int(getattr(self.args, "checkpoint_every", 1))
                if round_id % checkpoint_every == 0:
                    save_training_checkpoint(
                        self.args,
                        self.model,
                        completed_round=round_id,
                        logger=self.logger,
                    )

                round_total_sec = time.perf_counter() - round_start_time
                round_total_sec_acc += round_total_sec
                completed_this_run_rounds += 1
                cumulative_train_sec = time.perf_counter() - train_start_time
                avg_round_sec = (
                    round_total_sec_acc / completed_this_run_rounds
                    if completed_this_run_rounds > 0
                    else 0.0
                )
                eta_sec = avg_round_sec * max(self.server_epochs - round_id, 0)
                record_timing_result(
                    {
                        "phase": "round",
                        "round": round_id,
                        "num_clients": num_clients,
                        "client_loop_sec": client_loop_sec,
                        "aggregation_sec": aggregation_sec,
                        "save_server_sec": save_server_sec,
                        "round_eval_sec": round_eval_sec,
                        "final_eval_sec": 0.0,
                        "round_total_sec": round_total_sec,
                        "cumulative_train_sec": cumulative_train_sec,
                        "avg_round_sec": avg_round_sec,
                        "eta_sec": eta_sec,
                    },
                    self.args,
                )
                self.logger.info(
                    f"--round_timing : round={round_id} "
                    f"client_loop_sec={client_loop_sec:.2f} "
                    f"aggregation_sec={aggregation_sec:.2f} "
                    f"save_server_sec={save_server_sec:.2f} "
                    f"round_eval_sec={round_eval_sec:.2f} "
                    f"round_total_sec={round_total_sec:.2f} "
                    f"eta_sec={eta_sec:.2f}\n"
                )

            save_training_checkpoint(
                self.args,
                self.model,
                completed_round=self.server_epochs,
                logger=self.logger,
            )
            final_eval_start = time.perf_counter()
            self.evaluate_final_on_global_test()
            final_eval_sec = time.perf_counter() - final_eval_start
            cumulative_train_sec = time.perf_counter() - train_start_time
            avg_round_sec = (
                round_total_sec_acc / completed_this_run_rounds
                if completed_this_run_rounds > 0
                else 0.0
            )
            record_timing_result(
                {
                    "phase": "final_eval",
                    "round": self.server_epochs,
                    "num_clients": num_clients,
                    "client_loop_sec": 0.0,
                    "aggregation_sec": 0.0,
                    "save_server_sec": 0.0,
                    "round_eval_sec": 0.0,
                    "final_eval_sec": final_eval_sec,
                    "round_total_sec": final_eval_sec,
                    "cumulative_train_sec": cumulative_train_sec,
                    "avg_round_sec": avg_round_sec,
                    "eta_sec": 0.0,
                },
                self.args,
            )
            progress_bar.update(1)
            progress_bar.set_postfix({
                "round": f"{self.server_epochs}/{self.server_epochs}",
                "stage": "final_eval",
            })
        finally:
            progress_bar.close()

    def evaluate_global_model(self, data_loader):
        """ 用给定的数据集(global_test)评估当前服务端模型。
        返回：
        - eval_loss
        - eval_acc """

        self.model.to(self.device)
        self.model.eval()
        running_loss = torch.zeros((), device=self.device)
        running_corrects = torch.zeros((), device=self.device, dtype=torch.long)
        total_samples = 0

        with torch.inference_mode():
            for inputs, labels in data_loader:
                inputs, labels = self._move_batch_to_device(inputs, labels)
                result = self.model(inputs)
                outputs = result["logits"] if isinstance(result, dict) else result
                loss = self.criterion(outputs, labels)

                batch_size = labels.size(0)
                running_loss += loss.detach() * batch_size
                running_corrects += (outputs.argmax(dim=1) == labels).sum()
                total_samples += batch_size

        if total_samples <= 0:
            raise ValueError("Evaluation data_loader has no samples")

        eval_loss = (running_loss / total_samples).item()
        eval_acc = (running_corrects.float() / total_samples).item()
        return eval_loss, eval_acc

    def _move_batch_to_device(self, inputs, labels):
        non_blocking = (
            bool(getattr(self.args, "pin_memory", False))
            and str(self.device).startswith("cuda")
        )
        return (
            inputs.to(self.device, non_blocking=non_blocking),
            labels.to(self.device, non_blocking=non_blocking),
        )

    def evaluate_round_on_global_test(self, round_id):
        """每轮聚合后在 global_test 上评估一次，仅用于监控训练曲线。"""

        test_loss, test_acc = self.evaluate_global_model(self.global_test_loader)
        self.logger.info(
            f"--round_global_test_loss : {test_loss:.4f} "
            f"--round_global_test_acc : {test_acc:.4f} "
            f"--round : {round_id}\n"
        )
        record_server_result(
            {
                "phase": "round_test",
                "round": round_id,
                "test_loss": test_loss,
                "test_acc": test_acc,
            },
            self.args,
        )

    def evaluate_final_on_global_test(self):
        """ 在所有训练轮次结束后，用最终服务端模型在 global_test 上评估一次。 """

        test_loss, test_acc = self.evaluate_global_model(self.global_test_loader)
        self.logger.info(
            f"--final_global_test_loss : {test_loss:.4f} "
            f"--final_global_test_acc : {test_acc:.4f}\n"
        )
        record_server_result(
            {
                "phase": "final_test",
                "round": self.server_epochs,
                "test_loss": test_loss,
                "test_acc": test_acc,
            },
            self.args,
        )

    def get_client_train_size(self,client_id):
        """ 获取某个客户端训练样本数。
        FedAvg 会把这个作为聚合权重。 """

        # FedAvg 使用客户端训练样本数作为聚合权重。
        return get_client_train_size(self.args, client_id, meta=self.partition_meta)

    def aggregation_by_method(
        self,
        client_states=None,
        client_sizes=None,
        old_server_state=None,
        round_id=None,
    ):
        """聚合器接口：按当前配置的聚合方法执行参数聚合。

        - fedavg: 对完整 state_dict 按客户端训练样本数加权平均。
        - expert_fedavg: 非 expert 参数按客户端权重聚合，expert 参数按 expert usage 聚合。
        - fedwolf: 当前只支持 fedwolf_fusion_mode=robust_update_fusion；
          expert 参数由 fedwolf_update_fusion_variant 选择
          uniform_update / fisher_only / robust_only / fisher_wolf。
        """

        if client_states is None:
            self.logger.info("--client_state_transport : disk\n")
            client_states = []
            for id in self.clientsID_list:
                client_state_dict = torch.load(
                    self.args.model_save_path + f"/{id}.pth",
                    map_location="cpu",
                )
                client_states.append(client_state_dict)
        else:
            self.logger.info("--client_state_transport : memory\n")

        if client_sizes is None:
            client_sizes = [
                self.get_client_train_size(id)
                for id in self.clientsID_list
            ]

        total_size = sum(client_sizes)
        if total_size <= 0:
            raise ValueError("FedAvg requires at least one training sample across clients")

        aggregate_kwargs = {
            "client_updates": client_states,
            "client_weights": client_sizes,
            "global_model": self.model,
        }
        if self.args.agg_method == "fedwolf":
            aggregate_kwargs["client_stats"] = getattr(self, "last_client_expert_usages", None)
        else:
            aggregate_kwargs["expert_weights"] = getattr(self, "last_client_expert_usages", None)

        fedavg_state = self.aggregator.aggregate(**aggregate_kwargs)
        enable_update_norm_diag = bool(getattr(self.args, "server_update_norm_diag", True))
        update_norm_diag_interval = int(getattr(self.args, "server_update_norm_diag_interval", 1))
        if update_norm_diag_interval <= 0:
            update_norm_diag_interval = 1
        should_log_update_norm = (
            enable_update_norm_diag
            and old_server_state is not None
            and (round_id is None or round_id % update_norm_diag_interval == 0)
        )
        if should_log_update_norm:
            update_norm_summary = compute_update_norm_summary(old_server_state, fedavg_state)
            summary_text = " ".join(
                f"{key}={_format_fedwolf_summary_value(update_norm_summary.get(key), integer=key in SERVER_UPDATE_NORM_INTEGER_FIELDS)}"
                for key in SERVER_UPDATE_NORM_SUMMARY_KEYS
            )
            round_text = "None" if round_id is None else str(round_id)
            self.logger.info(f"--server_update_norm_summary : round={round_text} {summary_text}\n")
        self.model.load_state_dict(fedavg_state)
        self.logger.info(f"--aggregation_method : {self.args.agg_method}\n")
        self.logger.info(f"--client_train_sizes : {client_sizes}\n")
        robust_update_summary = getattr(self.aggregator, "last_robust_update_summary", None)
        if (
            isinstance(robust_update_summary, dict)
            and "fedwolf_update_fusion_variant" in robust_update_summary
        ):
            summary_text = " ".join(
                f"{key}={_format_fedwolf_summary_value(robust_update_summary.get(key), integer=key.endswith('_count') or key.endswith('_params') or key.endswith('_contribs') or key.endswith('_steps'))}"
                for key in sorted(robust_update_summary.keys())
            )
            self.logger.info(f"--fedwolf_robust_update_summary : {summary_text}\n")

    def aggregation(
        self,
        client_states=None,
        client_sizes=None,
        old_server_state=None,
        round_id=None,
    ):
        """ 聚合入口函数。
        现在只是简单调用 aggregation_by_method()，
        后续如果想扩展多种聚合流程，可以在这里继续封装。 """

        self.aggregation_by_method(
            client_states=client_states,
            client_sizes=client_sizes,
            old_server_state=old_server_state,
            round_id=round_id,
        )
