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

    def __init__(self, args: SimpleNamespace, client_id: int, logger, c_T: int, partition_meta=None):
        """ 初始化一个客户端对象。
        参数：
        - args: 所有配置参数
        - client_id: 当前客户端编号
        - logger: 日志记录器
        - c_T: 当前联邦通信轮次
        - partition_meta: 可选，已经加载好的数据划分信息 """

        self.args = args
        self.client_id = client_id
        self.model_path = self.args.model_save_path + f"/{self.client_id}.pth"
        self.model = build_model_from_args(self.args)
        self.device = self.args.device
        self.model.to(self.device)
        # c_T 表示当前是第几轮服务端通信轮次，主要用于记录日志。
        self.c_T =  c_T
        self.client_epochs = self.args.client_epochs
        # 分类任务常用交叉熵损失。
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)

        self.batch_size = self.args.batch_size
        self.partition_meta = partition_meta
        self.train_loader = None
        self.evidence_loader = None
        # 加载当前客户端的训练索引，并动态封装成 DataLoader。
        self.get_dataloader()

        self.logger = logger
        self.router_aux_loss_coef = self.args.router_aux_loss_coef
        self.router_z_loss_coef = self.args.router_z_loss_coef

    def should_compute_fisher_evidence(self):
        return getattr(self.args, "agg_method", None) in FISHER_EVIDENCE_AGG_METHODS

    def get_fisher_data_loader(self):
        evidence_loader_mode = getattr(self.args, "fedwolf_evidence_loader_mode", "deterministic")
        if evidence_loader_mode == "deterministic":
            return self.evidence_loader
        if evidence_loader_mode == "train_loader":
            return self.train_loader
        raise ValueError(
            "fedwolf_evidence_loader_mode must be either 'deterministic' or "
            f"'train_loader', got {evidence_loader_mode!r}."
        )

    def save_client_model(self):
        """ 本地训练结束后，把当前客户端模型保存回原来的路径。 """
        cpu_state_dict = {
            key: value.detach().cpu().clone()
            for key, value in self.model.state_dict().items()
        }
        torch.save(cpu_state_dict, self.model_path)

    def get_dataloader(self):
        """ 构造当前客户端自己的训练 DataLoader。
        注意：
        客户端只拥有自己的训练数据；
        验证集和测试集由服务端统一评估。 """
    
        self.train_loader = build_client_train_loader(
            args=self.args,
            client_id=self.client_id,
            meta=self.partition_meta,
        )
        self.evidence_loader = build_client_evidence_loader(
            args=self.args,
            client_id=self.client_id,
            meta=self.partition_meta,
        )


    def renew_model(self):
        """ 在每一轮本地训练开始前，
        客户端先从服务端同步最新的全局模型参数。 """

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
            running_loss = 0.0
            running_aux_loss = 0.0
            running_z_loss = 0.0
            running_corrects = 0
            total_samples = 0
            usage_total = torch.zeros(self.args.num_experts, device=self.device)
            layer_usage_total = {}
            router_prob_sum = torch.zeros(self.args.num_experts, device=self.device)

            for inputs, labels in self.train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()

                result = self.model(inputs)
                outputs = result["logits"]
                extra_loss, router_aux_loss, router_z_loss = self.get_auxiliary_losses(result)
                loss = self.criterion(outputs, labels) + extra_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1)
                self.optimizer.step()

                batch_size = inputs.size(0)
                running_loss += loss.item() * batch_size
                running_aux_loss += router_aux_loss.item() * batch_size
                running_z_loss += router_z_loss.item() * batch_size
                total_samples += batch_size
                _, preds = torch.max(outputs, 1)
                running_corrects += torch.sum(preds == labels.data)

                usage_total += self.get_expert_activations(result)
                self.add_layer_stats(layer_usage_total, self.get_layer_expert_stats(result))
                router_prob_sum += self.get_avg_router_probs(result) * batch_size

            train_loss = running_loss / len(self.train_loader.dataset)
            train_acc = running_corrects.double() / len(self.train_loader.dataset)
            avg_aux_loss = running_aux_loss / max(total_samples, 1)
            avg_z_loss = running_z_loss / max(total_samples, 1)
            local_usage_total += usage_total.detach()
            self.add_layer_stats(local_layer_usage_total, layer_usage_total)
            last_avg_router_probs = router_prob_sum / max(total_samples, 1)

            usage_list = [int(v) for v in usage_total.detach().cpu().tolist()]
            router_prob_list = [round(float(v), 4) for v in last_avg_router_probs.detach().cpu().tolist()]
            self.logger.info(
                f"--client: {self.client_id} --epoch:{epoch+1}/{self.client_epochs} "
                f"--train_loss :{train_loss:.4f} --train_acc :{train_acc:.4f} "
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
                'T': self.c_T,
                'client_epoch': epoch+1,
                'client_id': self.client_id,
                "train_loss": train_loss,
                "train_acc": train_acc.item(),
                "router_aux_loss": avg_aux_loss,
                "router_z_loss": avg_z_loss,
            }
            record_result(record_dic=record_dic, args=self.args)

        fisher_score_by_layer = {}
        fisher_log_score_by_layer = {}
        fisher_diagnostics = None

        if self.should_compute_fisher_evidence():
            evidence_loader_mode = getattr(self.args, "fedwolf_evidence_loader_mode", "deterministic")
            evidence_model_mode = getattr(self.args, "fedwolf_evidence_model_mode", "eval")
            fisher_data_loader = self.get_fisher_data_loader()
            self.logger.info(
                f"--client: {self.client_id} "
                f"--fedwolf_evidence_loader_mode : {evidence_loader_mode} "
                f"--fedwolf_evidence_model_mode : {evidence_model_mode}"
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
            self.logger.info(
                f"--client: {self.client_id} "
                f"--expert_fisher_diagnostics : {fisher_diagnostics}"
            )
        else:
            self.logger.info(
                f"--client: {self.client_id} "
                f"--skip_expert_fisher_evidence : agg_method={getattr(self.args, 'agg_method', None)}"
            )

        self.save_client_model()
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
        }
