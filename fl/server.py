import torch
import os
from types import SimpleNamespace
from torch import nn

from data.loader import build_global_eval_loader, get_client_train_size, load_partition_meta
from fl.aggregators import build_aggregator
from fl.client import Client
from model import build_model_from_args
from utils.utils import init_result_csv, init_server_result_csv, record_server_result

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
        # 客户端编号从 1 开始，例如 num_clients=4 时为 [1, 2, 3, 4]。
        self.clientsID_list = [i+1 for i in range(self.num_clients)]
        self.device = self.args.device
        self.logger = logger
        os.makedirs(self.args.model_save_path, exist_ok=True)
        self.partition_meta = load_partition_meta(self.args)
        self.global_test_loader = build_global_eval_loader(
            args=self.args,
            split="global_test",
            meta=self.partition_meta,
        )
        # 初始化全局模型，并保存到 server.pth。
        self.init_global_model()
        self.criterion = nn.CrossEntropyLoss()
        # 初始化 CSV 结果文件，后续客户端训练会不断追加记录。
        init_result_csv(self.args)
        init_server_result_csv(self.args)


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


    def train(self):
        """ 服务端训练主流程。
        每一轮(global round)大致做：
        1. 依次调度每个客户端本地训练
        2. 收集客户端返回的 expert 统计
        3. 聚合客户端模型
        4. 保存 server.pth,供下一轮客户端同步
        5. 所有轮次结束后，在 global_test 上评估最终模型 """

        # 外层循环是一轮轮服务端通信，也就是联邦学习中的 global round。
        for c_T in range(self.server_epochs):
            self.logger.info(f"============================== T:{c_T+1} start !!! ===============================\n")
            server_state_dict = {
                key: value.detach().cpu().clone()
                for key, value in self.model.state_dict().items()
            }
            round_expert_usage_summary = torch.zeros(self.args.num_experts)
            round_layer_stats = {}
            round_client_expert_usages = []
            round_client_states = []
            round_client_sizes = []
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
            # 所有客户端本地训练完成后，服务端通过聚合器更新全局模型。
            self.aggregation(
                client_states=round_client_states,
                client_sizes=round_client_sizes,
            )

            # 每轮结束保存当前服务端模型，供下一轮客户端同步。
            self.save_server_model()
            self.evaluate_round_on_global_test(round_id=c_T + 1)
            torch.cuda.empty_cache()

        self.evaluate_final_on_global_test()

    def evaluate_global_model(self, data_loader):
        """ 用给定的数据集(global_test)评估当前服务端模型。
        返回：
        - eval_loss
        - eval_acc """

        self.model.to(self.device)
        self.model.eval()
        running_loss = 0.0
        running_corrects = 0

        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                result = self.model(inputs)
                outputs = result["logits"]
                loss = self.criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                running_corrects += torch.sum(preds == labels.data)

        eval_loss = running_loss / len(data_loader.dataset)
        eval_acc = running_corrects.double() / len(data_loader.dataset)
        self.model.to("cpu")
        return eval_loss, eval_acc.item()

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
                "selected_round": round_id,
            },
            self.args,
        )

    def evaluate_final_on_global_test(self):
        """ 在所有训练轮次结束后，用最终服务端模型在 global_test 上评估一次。 """

        test_loss, test_acc = self.evaluate_global_model(self.global_test_loader)
        self.logger.info(
            f"--final_global_test_loss : {test_loss:.4f} "
            f"--final_global_test_acc : {test_acc:.4f} "
            f"--selected_round : {self.server_epochs}\n"
        )
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
        """ 聚合器接口：按当前配置的聚合方法执行参数聚合
        - fedavg:对完整 state_dict 按客户端训练样本数加权平均；
        - expert_fedavg:普通层按客户端样本数聚合,专家层按每个 expert 实际处理样本数聚合；
        - fedwolf_fisher_only:普通层按客户端样本数聚合,专家层按 Fisher score 聚合；
        - fedwolf:在 Fisher score 聚合基础上加入 WoLF-IMQ 状态更新和 gamma 插值。 """

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

        fedavg_state = self.aggregator.aggregate(
            client_updates=client_states,
            client_weights=client_sizes,
            global_model=self.model,
            expert_weights=getattr(self, "last_client_expert_usages", None),
        )
        self.model.load_state_dict(fedavg_state)
        self.logger.info(f"--aggregation_method : {self.args.agg_method}\n")
        self.logger.info(f"--client_train_sizes : {client_sizes}\n")
        filter_summary = getattr(self.aggregator, "last_filter_summary", None)
        if filter_summary:
            self.logger.info(f"--fedwolf_filter_state_summary : {filter_summary}\n")

    def aggregation(self, client_states=None, client_sizes=None):
        """ 聚合入口函数。
        现在只是简单调用 aggregation_by_method()，
        后续如果想扩展多种聚合流程，可以在这里继续封装。 """

        self.aggregation_by_method(
            client_states=client_states,
            client_sizes=client_sizes,
        )
