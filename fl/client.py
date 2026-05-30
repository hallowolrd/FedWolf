import torch
import torch.optim as optim
from types import SimpleNamespace
from torch import nn

from data.loader import build_client_evidence_loader, build_client_train_loader
from fl.expert_evidence import compute_expert_fisher_evidence
from model import build_model_from_args
from utils.utils import record_result


# 需要计算 Fisher evidence 的聚合方法集合。
# 当前只有 fedwolf_fisher_only 会在客户端训练结束后额外计算 expert Fisher score。
FISHER_EVIDENCE_AGG_METHODS = {"fedwolf_fisher_only"}


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

        # 服务端传入的全局模型参数。
        # 如果不为空，客户端 renew_model 时会直接从内存加载，不再从 server.pth 读取。
        self.server_state_dict = server_state_dict

        # 当前客户端本地模型保存路径。
        self.model_path = self.args.model_save_path + f"/{self.client_id}.pth"

        # 根据配置构建客户端本地模型副本。
        self.model = build_model_from_args(self.args)
        self.device = self.args.device
        self.model.to(self.device)

        # c_T 表示当前是第几轮服务端通信轮次，主要用于记录日志。
        self.c_T =  c_T

        # 客户端本地训练 epoch 数。
        self.client_epochs = self.args.client_epochs

        # 分类任务常用交叉熵损失。
        self.criterion = nn.CrossEntropyLoss()

        # 客户端本地优化器。
        # 当前使用 SGD，学习率来自配置 args.learning_rate。
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.args.learning_rate)

        self.batch_size = self.args.batch_size

        # 数据划分元信息，避免每个客户端重复加载划分文件。
        self.partition_meta = partition_meta

        # train_loader 用于正常本地训练。
        self.train_loader = None

        # evidence_loader 用于计算 Fisher evidence。
        # 通常可以是确定性采样的数据加载器，避免 Fisher 估计波动太大。
        self.evidence_loader = None

        # 加载当前客户端的训练索引，并动态封装成 DataLoader。
        self.get_dataloader()

        self.logger = logger

    def should_compute_fisher_evidence(self):
        # 判断当前聚合方法是否需要客户端额外计算 Fisher evidence。
        return getattr(self.args, "agg_method", None) in FISHER_EVIDENCE_AGG_METHODS

    def get_fisher_data_loader(self):
        # 选择用于 Fisher evidence 估计的数据加载器。
        # deterministic：使用单独构造的 evidence_loader，更稳定；
        # train_loader：直接复用训练 loader，更简单但可能受 shuffle 影响。
        evidence_loader_mode = getattr(self.args, "fedwolf_evidence_loader_mode", "deterministic")

        if evidence_loader_mode == "deterministic":
            return self.evidence_loader

        if evidence_loader_mode == "train_loader":
            return self.train_loader

        # 如果配置写错，直接报错，避免静默使用错误的数据来源。
        raise ValueError(
            "fedwolf_evidence_loader_mode must be either 'deterministic' or "
            f"'train_loader', got {evidence_loader_mode!r}."
        )

    def summarize_fisher_diagnostics(self, diagnostics):
        # 从完整 Fisher diagnostics 中筛选出适合写入普通日志的关键字段。
        # 这样可以避免日志过长，同时保留排查 Fisher 质量所需的信息。
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
            "auxiliary_loss_used_for_fisher",
            "zero_score_reason",
            "num_samples_with_grad_by_layer",
            "score_scientific_by_layer",
            "score_mean_diag_by_layer",
            "score_mean_diag_active_by_layer",
            "score_trace_per_sample_by_layer",
            "score_trace_per_active_sample_by_layer",
            "score_trace_raw_by_layer",
            "param_count_by_layer",
        ]

        # 只返回 diagnostics 中实际存在的字段，兼容不同版本的 diagnostics 格式。
        return {
            key: diagnostics.get(key)
            for key in summary_keys
            if key in diagnostics
        }

    def save_client_model(self):
        """ 本地训练结束后，把当前客户端模型保存回原来的路径。 """
        torch.save(self.get_cpu_state_dict(), self.model_path)

    def get_cpu_state_dict(self):
        # 把模型 state_dict 中所有参数 / buffer 拷贝到 CPU。
        # clone 可以避免后续模型继续训练时影响已返回给服务端的参数快照。
        return {
            key: value.detach().cpu().clone()
            for key, value in self.model.state_dict().items()
        }

    def get_dataloader(self):
        """ 构造当前客户端自己的训练 DataLoader。
        注意：
        客户端只拥有自己的训练数据；
        验证集和测试集由服务端统一评估。 """
    
        # 构造本地训练 loader。
        self.train_loader = build_client_train_loader(
            args=self.args,
            client_id=self.client_id,
            meta=self.partition_meta,
        )

        # 构造 Fisher evidence 估计使用的 loader。
        self.evidence_loader = build_client_evidence_loader(
            args=self.args,
            client_id=self.client_id,
            meta=self.partition_meta,
        )


    def renew_model(self, server_state_dict=None):
        """ 在每一轮本地训练开始前，
        客户端先从服务端同步最新的全局模型参数。 """

        # 如果调用时没有显式传入，就使用初始化时保存的 server_state_dict。
        if server_state_dict is None:
            server_state_dict = self.server_state_dict

        # 优先从内存中的 server_state_dict 加载，减少磁盘读写。
        if server_state_dict is not None:
            self.model.load_state_dict(server_state_dict)
            return

        # 如果没有内存参数，则回退到磁盘中的 server.pth。
        server_state_dict = torch.load(
            self.args.model_save_path + f"/server.pth",
            map_location="cpu",
        )
        self.model.load_state_dict(server_state_dict)

    def get_auxiliary_losses(self, result):
        """ 从模型 forward 的结果字典中，取出额外损失项。 """

        # 如果模型没有返回某些辅助损失，就用 0 代替，保证后续计算不会报错。
        zero = torch.tensor(0.0, device=self.device)

        # router_aux_loss 兼容旧字段 aux_loss。
        router_aux_loss = result.get("router_aux_loss", result.get("aux_loss", zero))

        # router_z_loss 是 router 的 z-loss，如果模型没有返回则为 0。
        router_z_loss = result.get("router_z_loss", zero)

        # total_router_loss 通常是训练时真正加到主损失上的额外 router loss。
        extra_loss = result.get("total_router_loss", zero)

        return extra_loss, router_aux_loss, router_z_loss

    def get_expert_activations(self, result):
        """ 从模型输出结果里拿到 expert 的激活/使用统计。
        如果模型没有返回这个字段，就用全零向量代替。 """
        
        usage = result.get("expert_activations")
        if usage is None:
            usage = torch.zeros(self.args.num_experts, device=self.device)

        # 保证返回值在当前训练设备上，方便后续累加。
        return usage.to(self.device)

    def get_avg_router_probs(self, result):
        """ 从模型输出结果里读取平均 router 概率。
        如果没有，就返回全零向量。 """

        probs = result.get("avg_router_probs")
        if probs is None:
            probs = torch.zeros(self.args.num_experts, device=self.device)

        # 保证返回值在当前训练设备上。
        return probs.to(self.device)

    def get_layer_expert_stats(self, result):
        """ 获取“按层统计”的 expert 使用信息。
        优先读取：
        - expert_stats_by_layer
        如果没有，就尝试从：
        - expert_activations_by_layer
        构造一个简化版本。 """

        # 新格式：每一层都有完整 expert 统计信息。
        layer_stats = result.get("expert_stats_by_layer")
        if layer_stats is not None:
            return layer_stats

        # 旧格式：只有每层 expert_activations，则包装成统一结构。
        return {
            layer_id: {"expert_activations": usage}
            for layer_id, usage in result.get("expert_activations_by_layer", {}).items()
        }

    def add_layer_stats(self, total_stats, batch_stats):
        """ 把一个 batch 的按层统计 batch_stats,累加到 total_stats 中。 """

        for layer_id, stats in batch_stats.items():
            layer_key = str(layer_id)

            # 第一次遇到该层时，先初始化该层的累计统计容器。
            if layer_key not in total_stats:
                total_stats[layer_key] = {
                    "expert_activations": torch.zeros(self.args.num_experts, device=self.device),
                    "selected_counts": torch.zeros(self.args.num_experts, device=self.device),
                    "overflow_counts": torch.zeros(self.args.num_experts, device=self.device),
                    "avg_router_probs": torch.zeros(self.args.num_experts, device=self.device),
                    "capacity": stats.get("capacity", 0),
                }

            # 对每个可累加的统计项做逐 expert 累加。
            for stat_key in ["expert_activations", "selected_counts", "overflow_counts", "avg_router_probs"]:
                value = stats.get(stat_key)
                if value is not None:
                    total_stats[layer_key][stat_key] += value.to(self.device)

            # capacity 不是逐 batch 累加量，而是当前层容量配置，直接更新保存即可。
            total_stats[layer_key]["capacity"] = stats.get("capacity", total_stats[layer_key]["capacity"])

    def train(self):
        """ 执行客户端本地训练。
        流程：
        1. 先同步服务端最新模型
        2. 在本地训练若干 epoch
        3. 记录 loss / acc / expert 使用情况
        4. 训练结束后保存客户端模型
        5. 返回专家统计信息给服务端 """

        # 每轮本地训练开始前，先同步服务端全局模型。
        self.renew_model()

        # 保存最后一个 epoch 的平均 router 概率。
        last_avg_router_probs = torch.zeros(self.args.num_experts, device=self.device)

        # 累计整个本地训练过程中的 expert 使用次数。
        local_usage_total = torch.zeros(self.args.num_experts, device=self.device)

        # 累计整个本地训练过程中的按层 expert 使用统计。
        local_layer_usage_total = {}

        for epoch in range(self.client_epochs):
            self.model.train()

            # 当前 epoch 的累计损失。
            running_loss = 0.0

            # 当前 epoch 的 router auxiliary loss 累计值。
            running_aux_loss = 0.0

            # 当前 epoch 的 router z-loss 累计值。
            running_z_loss = 0.0

            # 当前 epoch 的正确预测数量。
            running_corrects = 0

            # 当前 epoch 实际处理的样本数。
            total_samples = 0

            # 当前 epoch 的 expert 使用次数统计。
            usage_total = torch.zeros(self.args.num_experts, device=self.device)

            # 当前 epoch 的按层 expert 使用统计。
            layer_usage_total = {}

            # 当前 epoch 的 router 概率加权和，用于后面计算平均 router 概率。
            router_prob_sum = torch.zeros(self.args.num_experts, device=self.device)

            for inputs, labels in self.train_loader:
                # 把输入和标签移动到当前训练设备。
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # 清空上一轮 batch 的梯度。
                self.optimizer.zero_grad()

                # 前向传播，result 是模型返回的字典。
                result = self.model(inputs)

                # logits 用于分类损失和准确率计算。
                outputs = result["logits"]

                # 取出 router 相关辅助损失。
                extra_loss, router_aux_loss, router_z_loss = self.get_auxiliary_losses(result)

                # 总训练损失 = 主分类损失 + router 额外损失。
                loss = self.criterion(outputs, labels) + extra_loss

                # 反向传播。
                loss.backward()

                # 梯度裁剪，避免梯度爆炸。
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1)

                # 更新本地模型参数。
                self.optimizer.step()

                batch_size = inputs.size(0)

                # loss.item() 是 batch 平均 loss，这里乘 batch_size 后累加，方便后面算 epoch 平均。
                running_loss += loss.item() * batch_size
                running_aux_loss += router_aux_loss.item() * batch_size
                running_z_loss += router_z_loss.item() * batch_size
                total_samples += batch_size

                # 计算当前 batch 的预测类别。
                _, preds = torch.max(outputs, 1)

                # 累加预测正确的样本数。
                running_corrects += torch.sum(preds == labels.data)

                # 累加当前 batch 的 expert 激活次数。
                usage_total += self.get_expert_activations(result)

                # 累加当前 batch 的按层 expert 统计。
                self.add_layer_stats(layer_usage_total, self.get_layer_expert_stats(result))

                # 按 batch_size 加权累计 router 概率，用于计算样本级平均。
                router_prob_sum += self.get_avg_router_probs(result) * batch_size

            # 当前 epoch 平均训练损失。
            train_loss = running_loss / len(self.train_loader.dataset)

            # 当前 epoch 训练准确率。
            train_acc = running_corrects.double() / len(self.train_loader.dataset)

            # 当前 epoch 平均 router aux loss。
            avg_aux_loss = running_aux_loss / max(total_samples, 1)

            # 当前 epoch 平均 router z-loss。
            avg_z_loss = running_z_loss / max(total_samples, 1)

            # 累加到整个本地训练过程的 expert usage。
            local_usage_total += usage_total.detach()

            # 累加到整个本地训练过程的按层 usage。
            self.add_layer_stats(local_layer_usage_total, layer_usage_total)

            # 当前 epoch 的平均 router 概率。
            last_avg_router_probs = router_prob_sum / max(total_samples, 1)

            # 将 expert usage 转成普通 Python list，方便日志打印。
            usage_list = [int(v) for v in usage_total.detach().cpu().tolist()]

            # 将平均 router 概率转成保留 4 位小数的 list，方便日志查看。
            router_prob_list = [round(float(v), 4) for v in last_avg_router_probs.detach().cpu().tolist()]

            # 打印当前客户端当前 epoch 的训练指标和 expert 使用情况。
            self.logger.info(
                f"--client: {self.client_id} --epoch:{epoch+1}/{self.client_epochs} "
                f"--train_loss :{train_loss:.4f} --train_acc :{train_acc:.4f} "
                f"--router_aux_loss : {avg_aux_loss:.4f} "
                f"--router_z_loss : {avg_z_loss:.4f} "
                f"--expert_usage : {usage_list} --avg_router_probs : {router_prob_list}"
            )

            # 如果当前模型返回了按层 expert 统计，就额外打印每层 expert 的使用情况。
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

            # 记录当前客户端当前 epoch 的训练结果，通常会写入 csv/json 日志文件。
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

        # Fisher score 字典，key 通常是 layer_id，value 是该层每个 expert 的 Fisher score。
        fisher_score_by_layer = {}

        # Fisher log score 字典，通常用于日志分析或数值稳定性诊断。
        fisher_log_score_by_layer = {}

        # Fisher 诊断信息，包含样本数、梯度匹配情况、score 分布等。
        fisher_diagnostics = None

        if self.should_compute_fisher_evidence():
            # 读取 Fisher evidence 相关配置。
            evidence_loader_mode = getattr(self.args, "fedwolf_evidence_loader_mode", "deterministic")
            evidence_model_mode = getattr(self.args, "fedwolf_evidence_model_mode", "eval")
            fisher_score_mode = getattr(self.args, "fedwolf_fisher_score_mode", "mean_diag")
            fisher_debug_batches = getattr(self.args, "fedwolf_fisher_debug_batches", 0)

            # 根据配置选择 Fisher evidence 使用的数据加载器。
            fisher_data_loader = self.get_fisher_data_loader()

            # 打印 Fisher evidence 的关键配置，方便之后对照实验日志。
            self.logger.info(
                f"--client: {self.client_id} "
                f"--fedwolf_evidence_loader_mode : {evidence_loader_mode} "
                f"--fedwolf_evidence_model_mode : {evidence_model_mode} "
                f"--fedwolf_fisher_score_mode : {fisher_score_mode}"
            )

            # 计算当前客户端本地模型中每层每个 expert 的 Fisher evidence。
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
            )

            # 将 Fisher score 转成科学计数法字符串，避免日志中小数过小看不清。
            fisher_score_log = {
                layer_id: [f"{float(v):.12e}" for v in scores.tolist()]
                for layer_id, scores in fisher_score_by_layer.items()
            }

            # 将 Fisher log score 也转成科学计数法字符串。
            fisher_log_score_log = {
                layer_id: [f"{float(v):.12e}" for v in scores.tolist()]
                for layer_id, scores in fisher_log_score_by_layer.items()
            }

            # 打印每层每个 expert 的 Fisher score 和 log score。
            self.logger.info(
                f"--client: {self.client_id} "
                f"--expert_fisher_score_by_layer : {fisher_score_log} "
                f"--expert_fisher_log_score_by_layer : {fisher_log_score_log}"
            )

            # fedwolf_fisher_debug 控制是否打印完整 diagnostics。
            fisher_debug = bool(getattr(self.args, "fedwolf_fisher_debug", False))

            # 默认只打印摘要版 diagnostics，避免日志过长。
            fisher_diagnostics_summary = self.summarize_fisher_diagnostics(fisher_diagnostics)
            self.logger.info(
                f"--client: {self.client_id} "
                f"--expert_fisher_diagnostics_summary : {fisher_diagnostics_summary}"
            )

            # 如果开启 debug，则打印完整 Fisher diagnostics。
            if fisher_debug:
                self.logger.info(
                    f"--client: {self.client_id} "
                    f"--expert_fisher_diagnostics_full : {fisher_diagnostics}"
                )
        else:
            # 当前聚合方法不需要 Fisher evidence 时，只打印跳过信息。
            self.logger.info(
                f"--client: {self.client_id} "
                f"--skip_expert_fisher_evidence : agg_method={getattr(self.args, 'agg_method', None)}"
            )

        # 获取训练结束后的本地模型参数快照。
        local_state_dict = self.get_cpu_state_dict()

        # 如果配置要求保存客户端模型，则把本地模型参数写入磁盘。
        if bool(getattr(self.args, "save_client_models", False)):
            torch.save(local_state_dict, self.model_path)

        # 将按层 expert 统计转到 CPU，方便服务端聚合和序列化保存。
        layer_stats_cpu = {
            layer_id: {
                stat_key: (value.detach().cpu() if torch.is_tensor(value) else value)
                for stat_key, value in stats.items()
            }
            for layer_id, stats in local_layer_usage_total.items()
        }

        # 返回给服务端的客户端统计信息和本地模型参数。
        # 服务端后续会用这些信息做 FedAvg / ExpertFedAvg / Fisher-only 等聚合。
        return {
            "expert_activations": local_usage_total.detach().cpu(),
            "expert_stats_by_layer": layer_stats_cpu,
            "expert_activations_by_layer": {
                layer_id: stats["expert_activations"]
                for layer_id, stats in layer_stats_cpu.items()
            },
            "expert_fisher_score_by_layer": fisher_score_by_layer,
            "expert_fisher_log_score_by_layer": fisher_log_score_by_layer,
            "local_state_dict": local_state_dict,
        }