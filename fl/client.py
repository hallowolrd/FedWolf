import math

import torch
import torch.optim as optim
from types import SimpleNamespace
from torch import nn

from data.loader import build_client_evidence_loader, build_client_train_loader
from fl.expert_evidence import compute_expert_fisher_evidence
from model import build_model_from_args
from utils.utils import record_result

FISHER_EVIDENCE_AGG_METHODS = {"fedwolf", "fedwolf_fisher_only"}


class Client:
    """ Client 表示联邦学习中的一个客户端。
    每个客户端：
    1. 有自己的本地数据
    2. 有自己的本地模型副本
    3. 每一轮会先同步服务端模型，再做本地训练
    4. 训练完保存本地模型，并返回一些统计信息给服务端 """

    def __init__(
        self,
        args: SimpleNamespace,
        client_id: int,
        logger,
        c_T: int,
        partition_meta=None,
        server_state_dict=None,
    ):
        """ 初始化一个客户端对象。
        参数：
        - args: 所有配置参数
        - client_id: 当前客户端编号
        - logger: 日志记录器
        - c_T: 当前联邦通信轮次
        - partition_meta: 可选，已经加载好的数据划分信息
        - server_state_dict: 可选，服务端在内存中传入的全局模型参数 """

        self.args = args
        self.client_id = client_id
        self.server_state_dict = server_state_dict
        self.model_path = self.args.model_save_path + f"/{self.client_id}.pth"
        self.model = build_model_from_args(self.args)
        self.device = self.args.device
        self.model.to(self.device)
        # c_T 表示当前是第几轮服务端通信轮次，主要用于记录日志。
        self.c_T =  c_T
        self.client_epochs = self.args.client_epochs
        # 分类任务常用交叉熵损失。
        self.criterion = nn.CrossEntropyLoss()
        self.current_lr = self.get_current_learning_rate()
        self.optimizer = self.build_optimizer()

        self.batch_size = self.args.batch_size
        self.partition_meta = partition_meta
        self.train_loader = None
        self.evidence_loader = None
        # 加载当前客户端的训练索引，并动态封装成 DataLoader。
        self.get_dataloader()

        self.logger = logger
        self.router_aux_loss_coef = self.args.router_aux_loss_coef
        self.router_z_loss_coef = self.args.router_z_loss_coef

    def _cosine_learning_rate(self, base_lr: float, min_lr: float, progress: float) -> float:
        progress = min(max(float(progress), 0.0), 1.0)
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr + (base_lr - min_lr) * cosine_factor

    def build_weight_decay_param_groups(self, weight_decay: float):
        if weight_decay <= 0.0:
            return [{"params": [p for p in self.model.parameters() if p.requires_grad], "weight_decay": 0.0}]

        decay_params = []
        no_decay_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            name_lower = name.lower()
            if (
                name_lower.endswith(".bias")
                or "bn" in name_lower
                or "norm" in name_lower
                or "layernorm" in name_lower
            ):
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        param_groups = []
        if decay_params:
            param_groups.append({"params": decay_params, "weight_decay": weight_decay})
        if no_decay_params:
            param_groups.append({"params": no_decay_params, "weight_decay": 0.0})

        return param_groups

    def build_optimizer(self):
        optimizer_name = str(getattr(self.args, "optimizer", "adam")).strip().lower()
        weight_decay = float(getattr(self.args, "weight_decay", 0.0))
        momentum = float(getattr(self.args, "momentum", 0.9))

        if weight_decay < 0.0:
            raise ValueError("weight_decay must be non-negative.")
        if momentum < 0.0:
            raise ValueError("momentum must be non-negative.")

        if optimizer_name == "adam":
            return optim.Adam(
                self.model.parameters(),
                lr=self.current_lr,
                weight_decay=weight_decay,
            )

        if optimizer_name == "adamw":
            param_groups = self.build_weight_decay_param_groups(weight_decay)
            return optim.AdamW(
                param_groups,
                lr=self.current_lr,
            )

        if optimizer_name == "sgd":
            return optim.SGD(
                self.model.parameters(),
                lr=self.current_lr,
                momentum=momentum,
                weight_decay=weight_decay,
            )

        raise ValueError(
            f"Unsupported optimizer: {optimizer_name!r}. "
            "Expected 'adam', 'adamw', or 'sgd'."
        )

    def get_current_learning_rate(self) -> float:
        """返回当前 global communication round 的学习率。

        该调度基于 server round c_T，而不是 local epoch 或 batch。
        Client 对象和 Adam optimizer 会在每个 server round 重新创建，
        所以在这里使用 PyTorch local scheduler 会每轮重启。
        """

        schedule = str(getattr(self.args, "lr_schedule", "constant")).strip().lower()
        base_lr = float(self.args.learning_rate)

        if base_lr <= 0.0:
            raise ValueError("learning_rate must be positive.")

        if schedule in {"constant", "none"}:
            return base_lr

        min_lr = float(getattr(self.args, "min_learning_rate", 0.0))
        if min_lr < 0.0:
            raise ValueError("min_learning_rate must be non-negative.")
        if min_lr > base_lr:
            raise ValueError("min_learning_rate must be <= learning_rate.")

        total_rounds = int(getattr(self.args, "server_epochs", 1))
        if total_rounds <= 0:
            raise ValueError("server_epochs must be positive.")

        if schedule == "cosine":
            denominator = max(total_rounds - 1, 1)
            progress = float(self.c_T) / float(denominator)
            return self._cosine_learning_rate(base_lr, min_lr, progress)

        if schedule in {"warmup_cosine", "cosine_warmup", "linear_warmup_cosine"}:
            warmup_rounds = int(getattr(self.args, "warmup_rounds", 0))
            if warmup_rounds < 0:
                raise ValueError("warmup_rounds must be non-negative.")

            warmup_start_lr = getattr(self.args, "warmup_start_learning_rate", None)
            if warmup_start_lr is None:
                warmup_start_lr = min_lr if min_lr > 0.0 else base_lr * 0.2
            warmup_start_lr = float(warmup_start_lr)
            if warmup_start_lr < 0.0:
                raise ValueError("warmup_start_learning_rate must be non-negative.")
            if warmup_start_lr > base_lr:
                raise ValueError("warmup_start_learning_rate must be <= learning_rate.")

            if warmup_rounds <= 0:
                denominator = max(total_rounds - 1, 1)
                progress = float(self.c_T) / float(denominator)
                return self._cosine_learning_rate(base_lr, min_lr, progress)

            if warmup_rounds >= total_rounds:
                raise ValueError(
                    "warmup_rounds must be smaller than server_epochs for warmup_cosine."
                )

            if self.c_T < warmup_rounds:
                progress = float(self.c_T) / float(max(warmup_rounds - 1, 1))
                return warmup_start_lr + (base_lr - warmup_start_lr) * min(max(progress, 0.0), 1.0)

            cosine_round = self.c_T - warmup_rounds
            cosine_total = max(total_rounds - warmup_rounds - 1, 1)
            progress = float(cosine_round) / float(cosine_total)
            return self._cosine_learning_rate(base_lr, min_lr, progress)

        raise ValueError(
            f"Unsupported lr_schedule: {schedule!r}. "
            "Expected 'constant', 'none', 'cosine', 'warmup_cosine', 'cosine_warmup', or 'linear_warmup_cosine'."
        )

    def should_compute_fisher_evidence(self):
        return getattr(self.args, "agg_method", None) in FISHER_EVIDENCE_AGG_METHODS

    def get_fisher_data_loader(self):
        if not self.should_compute_fisher_evidence():
            raise RuntimeError(
                "get_fisher_data_loader() was called although current agg_method "
                f"does not require Fisher evidence: {getattr(self.args, 'agg_method', None)!r}"
            )

        evidence_loader_mode = str(
            getattr(self.args, "fedwolf_evidence_loader_mode", "deterministic")
        ).strip().lower()
        if evidence_loader_mode == "deterministic":
            if self.evidence_loader is None:
                self.evidence_loader = build_client_evidence_loader(
                    args=self.args,
                    client_id=self.client_id,
                    meta=self.partition_meta,
                    round_id=self.c_T,
                )
            return self.evidence_loader
        if evidence_loader_mode == "train_loader":
            return self.train_loader
        raise ValueError(
            "fedwolf_evidence_loader_mode must be either 'deterministic' or "
            f"'train_loader', got {evidence_loader_mode!r}."
        )

    def summarize_fisher_diagnostics(self, diagnostics):
        if diagnostics is None:
            return None

        summary_keys = [
            "matched_param_name_count",
            "total_samples",
            "num_batches",
            "fisher_estimator",
            "fisher_score_mode",
            "fisher_score_mode_raw",
            "normalization",
            "model_mode",
            "debug_batches",
            "max_samples",
            "max_batches",
            "limit_reached",
            "stop_reason",
            "effective_total_samples",
            "effective_num_batches",
            "auxiliary_loss_used_for_fisher",
            "zero_score_reason",
            "expert_block_fisher_matched_block_count",
            "expert_block_fisher_positive_block_count",
            "expert_block_fisher_mean_positive",
            "expert_block_fisher_max_positive",
            "unmatched_block_name_count",
            "num_samples_with_grad_by_layer",
            "score_scientific_by_layer",
            "score_mean_diag_by_layer",
            "score_mean_diag_active_by_layer",
            "score_trace_per_sample_by_layer",
            "score_trace_per_active_sample_by_layer",
            "score_trace_raw_by_layer",
            "param_count_by_layer",
            "evidence_expert_stats_by_layer",
            "evidence_expert_activations_by_layer",
            "evidence_selected_counts_by_layer",
            "evidence_overflow_counts_by_layer",
        ]
        return {
            key: diagnostics.get(key)
            for key in summary_keys
            if key in diagnostics
        }

    def save_client_model(self):
        """ 本地训练结束后，把当前客户端模型保存回原来的路径。 """
        torch.save(self.get_cpu_state_dict(), self.model_path)

    def get_cpu_state_dict(self):
        return {
            key: value.detach().cpu().clone()
            for key, value in self.model.state_dict().items()
        }

    def get_dataloader(self):
        """构造当前客户端自己的训练 DataLoader。

        train_loader 对所有聚合方法都需要；
        evidence_loader 只在 FedWoLF Fisher evidence 需要 deterministic loader 时构造。
        """
    
        self.train_loader = build_client_train_loader(
            args=self.args,
            client_id=self.client_id,
            meta=self.partition_meta,
            round_id=self.c_T,
        )
        self.evidence_loader = None

        if not self.should_compute_fisher_evidence():
            return

        evidence_loader_mode = str(
            getattr(self.args, "fedwolf_evidence_loader_mode", "deterministic")
        ).strip().lower()

        if evidence_loader_mode == "deterministic":
            self.evidence_loader = build_client_evidence_loader(
                args=self.args,
                client_id=self.client_id,
                meta=self.partition_meta,
                round_id=self.c_T,
            )
            return

        if evidence_loader_mode == "train_loader":
            return

        raise ValueError(
            "fedwolf_evidence_loader_mode must be either 'deterministic' or "
            f"'train_loader', got {evidence_loader_mode!r}."
        )


    def renew_model(self, server_state_dict=None):
        """ 在每一轮本地训练开始前，
        客户端先从服务端同步最新的全局模型参数。 """

        if server_state_dict is None:
            server_state_dict = self.server_state_dict

        if server_state_dict is not None:
            self.model.load_state_dict(server_state_dict)
            return

        server_state_dict = torch.load(
            self.args.model_save_path + f"/server.pth",
            map_location="cpu",
        )
        self.model.load_state_dict(server_state_dict)

    def get_auxiliary_losses(self, result):
        """ 从模型 forward 的结果字典中，取出额外损失项。 """

        zero = torch.tensor(0.0, device=self.device)
        router_aux_loss = result.get("router_aux_loss", result.get("aux_loss", zero))
        router_z_loss = result.get("router_z_loss", zero)
        extra_loss = (
            self.router_aux_loss_coef * router_aux_loss
            + self.router_z_loss_coef * router_z_loss
        )
        return extra_loss, router_aux_loss, router_z_loss

    def _move_batch_to_device(self, inputs, labels):
        non_blocking = (
            bool(getattr(self.args, "pin_memory", False))
            and str(self.device).startswith("cuda")
        )
        return (
            inputs.to(self.device, non_blocking=non_blocking),
            labels.to(self.device, non_blocking=non_blocking),
        )

    def get_expert_activations(self, result):
        """ 从模型输出结果里拿到 expert 的激活/使用统计。
        如果模型没有返回这个字段，就用全零向量代替。 """
        
        usage = result.get("expert_activations")
        if usage is None:
            usage = torch.zeros(self.args.num_experts, device=self.device)
        return usage.to(self.device)

    def get_avg_router_probs(self, result):
        """ 从模型输出结果里读取平均 router 概率。
        如果没有，就返回全零向量。 """

        probs = result.get("avg_router_probs")
        if probs is None:
            probs = torch.zeros(self.args.num_experts, device=self.device)
        return probs.to(self.device)

    def get_layer_expert_stats(self, result):
        """ 获取“按层统计”的 expert 使用信息。
        优先读取：
        - expert_stats_by_layer
        如果没有，就尝试从：
        - expert_activations_by_layer
        构造一个简化版本。 """

        layer_stats = result.get("expert_stats_by_layer")
        if layer_stats is not None:
            return layer_stats

        return {
            layer_id: {"expert_activations": usage}
            for layer_id, usage in result.get("expert_activations_by_layer", {}).items()
        }

    def add_layer_stats(self, total_stats, batch_stats):
        """ 把一个 batch 的按层统计 batch_stats,累加到 total_stats 中。 """

        for layer_id, stats in batch_stats.items():
            layer_key = str(layer_id)
            if layer_key not in total_stats:
                total_stats[layer_key] = {
                    "expert_activations": torch.zeros(self.args.num_experts, device=self.device),
                    "selected_counts": torch.zeros(self.args.num_experts, device=self.device),
                    "overflow_counts": torch.zeros(self.args.num_experts, device=self.device),
                    "avg_router_probs": torch.zeros(self.args.num_experts, device=self.device),
                    "capacity": stats.get("capacity", 0),
                }

            for stat_key in ["expert_activations", "selected_counts", "overflow_counts", "avg_router_probs"]:
                value = stats.get(stat_key)
                if value is not None:
                    total_stats[layer_key][stat_key] += value.to(self.device)
            total_stats[layer_key]["capacity"] = stats.get("capacity", total_stats[layer_key]["capacity"])

    def train(self):
        """ 执行客户端本地训练。
        流程：
        1. 先同步服务端最新模型
        2. 在本地训练若干 epoch
        3. 记录 loss / acc / expert 使用情况
        4. 训练结束后保存客户端模型
        5. 返回专家统计信息给服务端 """

        self.renew_model()

        last_avg_router_probs = torch.zeros(self.args.num_experts, device=self.device)
        local_usage_total = torch.zeros(self.args.num_experts, device=self.device)
        local_layer_usage_total = {}

        for epoch in range(self.client_epochs):
            self.model.train()
            running_loss = torch.zeros((), device=self.device)
            running_aux_loss = torch.zeros((), device=self.device)
            running_z_loss = torch.zeros((), device=self.device)
            running_corrects = torch.zeros((), device=self.device)
            total_samples = 0
            usage_total = torch.zeros(self.args.num_experts, device=self.device)
            layer_usage_total = {}
            router_prob_sum = torch.zeros(self.args.num_experts, device=self.device)

            for inputs, labels in self.train_loader:
                inputs, labels = self._move_batch_to_device(inputs, labels)
                self.optimizer.zero_grad()

                result = self.model(inputs)
                outputs = result["logits"]
                extra_loss, router_aux_loss, router_z_loss = self.get_auxiliary_losses(result)
                loss = self.criterion(outputs, labels) + extra_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1)
                self.optimizer.step()

                batch_size = inputs.size(0)
                running_loss += loss.detach() * batch_size
                running_aux_loss += router_aux_loss.detach() * batch_size
                running_z_loss += router_z_loss.detach() * batch_size
                total_samples += batch_size
                _, preds = torch.max(outputs, 1)
                running_corrects += torch.sum(preds == labels).detach()

                usage_total += self.get_expert_activations(result)
                self.add_layer_stats(layer_usage_total, self.get_layer_expert_stats(result))
                router_prob_sum += self.get_avg_router_probs(result) * batch_size

            denominator = max(total_samples, 1)
            train_loss_tensor = running_loss / denominator
            train_acc_tensor = running_corrects.double() / denominator
            avg_aux_loss_tensor = running_aux_loss / denominator
            avg_z_loss_tensor = running_z_loss / denominator

            train_loss = float(train_loss_tensor.detach().cpu().item())
            train_acc = float(train_acc_tensor.detach().cpu().item())
            avg_aux_loss = float(avg_aux_loss_tensor.detach().cpu().item())
            avg_z_loss = float(avg_z_loss_tensor.detach().cpu().item())
            local_usage_total += usage_total.detach()
            self.add_layer_stats(local_layer_usage_total, layer_usage_total)
            last_avg_router_probs = router_prob_sum / max(total_samples, 1)

            usage_list = [int(v) for v in usage_total.detach().cpu().tolist()]
            router_prob_list = [round(float(v), 4) for v in last_avg_router_probs.detach().cpu().tolist()]
            self.logger.info(
                f"--client: {self.client_id} --epoch:{epoch+1}/{self.client_epochs} "
                f"--train_loss :{train_loss:.4f} --train_acc :{train_acc:.4f} "
                f"--lr : {self.current_lr:.6e} "
                f"--router_aux_loss : {avg_aux_loss:.4f} "
                f"--router_z_loss : {avg_z_loss:.4f} "
                f"--expert_usage : {usage_list} --avg_router_probs : {router_prob_list}"
            )
            if layer_usage_total:
                layer_usage_log = {
                    layer_id: {
                        "expert_activations": [int(v) for v in stats["expert_activations"].detach().cpu().tolist()],
                        "overflow_counts": [int(v) for v in stats["overflow_counts"].detach().cpu().tolist()],
                        "capacity": int(stats["capacity"]),
                    }
                    for layer_id, stats in layer_usage_total.items()
                }
                self.logger.info(f"--client: {self.client_id} --layer_expert_stats : {layer_usage_log}")

            record_dic = {
                'T': self.c_T + 1,
                'client_epoch': epoch + 1,
                'client_id': self.client_id,
                "learning_rate": self.current_lr,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "router_aux_loss": avg_aux_loss,
                "router_z_loss": avg_z_loss,
            }
            record_result(record_dic=record_dic, args=self.args)

        fisher_score_by_layer = {}
        fisher_log_score_by_layer = {}
        fisher_block_score_by_layer = {}
        fisher_diagnostics = None

        if self.should_compute_fisher_evidence():
            evidence_loader_mode = getattr(self.args, "fedwolf_evidence_loader_mode", "deterministic")
            evidence_model_mode = getattr(self.args, "fedwolf_evidence_model_mode", "eval")
            fisher_score_mode = getattr(self.args, "fedwolf_fisher_score_mode", "mean_diag")
            fisher_debug_batches = getattr(self.args, "fedwolf_fisher_debug_batches", 0)
            fisher_max_samples = getattr(self.args, "fedwolf_fisher_max_samples", None)
            fisher_max_batches = getattr(self.args, "fedwolf_fisher_max_batches", None)
            fisher_data_loader = self.get_fisher_data_loader()
            self.logger.info(
                f"--client: {self.client_id} "
                f"--fedwolf_evidence_loader_mode : {evidence_loader_mode} "
                f"--fedwolf_evidence_model_mode : {evidence_model_mode} "
                f"--fedwolf_fisher_score_mode : {fisher_score_mode} "
                f"--fedwolf_fisher_max_samples : {fisher_max_samples} "
                f"--fedwolf_fisher_max_batches : {fisher_max_batches}"
            )
            fisher_score_by_layer, fisher_log_score_by_layer, fisher_diagnostics = compute_expert_fisher_evidence(
                model=self.model,
                data_loader=fisher_data_loader,
                criterion=self.criterion,
                device=self.device,
                num_experts=self.args.num_experts,
                get_auxiliary_losses=self.get_auxiliary_losses,
                return_diagnostics=True,
                model_mode=evidence_model_mode,
                score_mode=fisher_score_mode,
                debug_batches=fisher_debug_batches,
                max_samples=fisher_max_samples,
                max_batches=fisher_max_batches,
                pin_memory=getattr(self.args, "pin_memory", False),
            )
            fisher_score_log = {
                layer_id: [f"{float(v):.12e}" for v in scores.tolist()]
                for layer_id, scores in fisher_score_by_layer.items()
            }
            fisher_log_score_log = {
                layer_id: [f"{float(v):.12e}" for v in scores.tolist()]
                for layer_id, scores in fisher_log_score_by_layer.items()
            }
            self.logger.info(
                f"--client: {self.client_id} "
                f"--expert_fisher_score_by_layer : {fisher_score_log} "
                f"--expert_fisher_log_score_by_layer : {fisher_log_score_log}"
            )
            fisher_block_score_by_layer = fisher_diagnostics.get(
                "expert_block_fisher_score_by_layer",
                {},
            )
            fisher_debug = bool(getattr(self.args, "fedwolf_fisher_debug", False))
            fisher_diagnostics_summary = self.summarize_fisher_diagnostics(fisher_diagnostics)
            self.logger.info(
                f"--client: {self.client_id} "
                f"--expert_fisher_diagnostics_summary : {fisher_diagnostics_summary}"
            )
            if fisher_debug:
                fisher_diagnostics_for_log = dict(fisher_diagnostics)
                if "expert_block_fisher_score_by_layer" in fisher_diagnostics_for_log:
                    fisher_diagnostics_for_log["expert_block_fisher_score_by_layer"] = (
                        "<omitted from log; returned in client_stats>"
                    )
                self.logger.info(
                    f"--client: {self.client_id} "
                    f"--expert_fisher_diagnostics_full : {fisher_diagnostics_for_log}"
                )
            else:
                self.logger.info(
                    f"--client: {self.client_id} "
                    f"--expert_fisher_diagnostics_full_skipped : fedwolf_fisher_debug=False"
                )
        else:
            self.logger.info(
                f"--client: {self.client_id} "
                f"--skip_expert_fisher_evidence : agg_method={getattr(self.args, 'agg_method', None)}"
            )

        local_state_dict = self.get_cpu_state_dict()
        if bool(getattr(self.args, "save_client_models", False)):
            torch.save(local_state_dict, self.model_path)

        layer_stats_cpu = {
            layer_id: {
                stat_key: (value.detach().cpu() if torch.is_tensor(value) else value)
                for stat_key, value in stats.items()
            }
            for layer_id, stats in local_layer_usage_total.items()
        }
        return {
            "expert_activations": local_usage_total.detach().cpu(),
            "expert_stats_by_layer": layer_stats_cpu,
            "expert_activations_by_layer": {
                layer_id: stats["expert_activations"]
                for layer_id, stats in layer_stats_cpu.items()
            },
            "expert_fisher_score_by_layer": fisher_score_by_layer,
            "expert_fisher_log_score_by_layer": fisher_log_score_by_layer,
            "expert_block_fisher_score_by_layer": fisher_block_score_by_layer,
            "evidence_expert_stats_by_layer": (
                fisher_diagnostics.get("evidence_expert_stats_by_layer", {})
                if fisher_diagnostics else {}
            ),
            "evidence_expert_activations_by_layer": (
                fisher_diagnostics.get("evidence_expert_activations_by_layer", {})
                if fisher_diagnostics else {}
            ),
            "evidence_selected_counts_by_layer": (
                fisher_diagnostics.get("evidence_selected_counts_by_layer", {})
                if fisher_diagnostics else {}
            ),
            "evidence_overflow_counts_by_layer": (
                fisher_diagnostics.get("evidence_overflow_counts_by_layer", {})
                if fisher_diagnostics else {}
            ),
            "local_state_dict": local_state_dict,
        }
