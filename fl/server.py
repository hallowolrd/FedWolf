import os
from types import SimpleNamespace

import torch
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
    def __init__(self, args: SimpleNamespace, logger):
        self.args = args
        self.logger = logger
        self.aggregator = build_aggregator(self.args)
        self.num_clients = self.args.num_clients
        self.server_epochs = self.args.server_epochs
        self.clientsID_list = [index + 1 for index in range(self.num_clients)]
        self.device = self.args.device
        self.best_acc = 0.0

        self.log_experiment_config()
        os.makedirs(self.args.model_save_path, exist_ok=True)

        self.checkpoint_path = os.path.join(self.args.model_save_path, "checkpoint.pth")
        self.resume = bool(getattr(self.args, "resume", False))
        self.start_round = 0
        self.final_evaluated = False

        self.partition_meta = load_partition_meta(self.args)
        self.global_test_loader = build_global_eval_loader(
            args=self.args,
            split="global_test",
            meta=self.partition_meta,
        )

        self.init_global_model(save_initial=not self.resume)
        if self.resume:
            self.load_training_checkpoint()

        self.criterion = nn.CrossEntropyLoss()
        if self.resume:
            trim_result_csv_for_resume(
                self.args,
                completed_round=self.start_round,
                final_evaluated=self.final_evaluated,
            )
        init_result_csv(self.args)
        init_server_result_csv(self.args)

    def log_experiment_config(self):
        run_keys = (
            "data_name",
            "alpha",
            "seed",
            "partition_impl",
            "num_clients",
            "server_epochs",
            "client_epochs",
            "batch_size",
        )
        model_keys = (
            "model_type",
            "num_experts",
            "top_k",
            "learning_rate",
            "router_aux_loss_coef",
            "router_z_loss_coef",
            "router_balance_loss_coef",
        )
        train_keys = (
            "aggregation_mode",
            "non_expert_agg_method",
            "expert_agg_method",
            "optimizer",
            "momentum",
            "weight_decay",
            "grad_clip_norm",
            "label_smooth",
        )

        self.logger.info(self._format_config_block("RunConfig", run_keys))
        self.logger.info(self._format_config_block("ModelConfig", model_keys))
        self.logger.info(self._format_config_block("TrainConfig", train_keys))

    def _format_config_block(self, title, keys):
        lines = [f"[{title}]"]
        for key in keys:
            lines.append(f"{key}={getattr(self.args, key)}")
        return "\n".join(lines) + "\n"

    def init_global_model(self, save_initial=True):
        self.model = build_model_from_args(self.args)
        if save_initial:
            self.save_server_model()

    def get_server_state_dict(self):
        return {
            key: value.detach().cpu().clone()
            for key, value in self.model.state_dict().items()
        }

    def save_server_model(self):
        torch.save(self.get_server_state_dict(), self.args.model_save_path + "/server.pth")

    def save_training_checkpoint(self, completed_round, final_evaluated=False):
        checkpoint = {
            "completed_round": int(completed_round),
            "final_evaluated": bool(final_evaluated),
            "best_acc": float(self.best_acc),
            "server_state_dict": self.get_server_state_dict(),
            "aggregator_state": self.aggregator.state_dict()
            if hasattr(self.aggregator, "state_dict")
            else {},
        }
        torch.save(checkpoint, self.checkpoint_path)

    def load_training_checkpoint(self):
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
        self.best_acc = float(checkpoint.get("best_acc", 0.0))
        self.save_server_model()
        self.logger.info(
            f"--resume_checkpoint : {self.checkpoint_path} "
            f"--completed_round : {self.start_round} "
            f"--server_epochs : {self.server_epochs}\n"
        )

    def train(self):
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
        with tqdm(
            total=total_client_steps,
            initial=completed_client_steps,
            desc="training clients",
            unit="client",
            dynamic_ncols=True,
        ) as progress_bar:
            for round_index in range(self.start_round, self.server_epochs):
                round_id = round_index + 1
                self.logger.info(
                    f"============================== round:{round_id} start ==============================\n"
                )
                server_state_dict = self.get_server_state_dict()
                round_client_states = []
                round_client_sizes = []
                round_client_stats = []
                round_train_losses = []

                progress_bar.set_description(f"round {round_id}/{self.server_epochs} clients")
                for client_id in self.clientsID_list:
                    client_stats = Client(
                        args=self.args,
                        client_id=client_id,
                        logger=self.logger,
                        c_T=round_index,
                        partition_meta=self.partition_meta,
                        server_state_dict=server_state_dict,
                    ).train()
                    client_state_dict = client_stats.pop("local_state_dict")
                    round_client_states.append(client_state_dict)
                    round_client_sizes.append(self.get_client_train_size(client_id))
                    round_train_losses.append(float(client_stats.get("train_loss", 0.0)))
                    round_client_stats.append(client_stats)
                    progress_bar.update(1)

                self.last_client_expert_usages = round_client_stats
                avg_train_loss = sum(round_train_losses) / max(len(round_train_losses), 1)

                self.aggregation(
                    client_states=round_client_states,
                    client_sizes=round_client_sizes,
                )
                self.save_server_model()

                test_loss, test_acc = self.evaluate_global_model(self.global_test_loader)
                self.best_acc = max(self.best_acc, test_acc)
                self.log_round_summary(
                    round_id=round_id,
                    avg_train_loss=avg_train_loss,
                    test_acc=test_acc,
                )
                self.record_round_result(round_id, test_loss, test_acc)
                self.save_training_checkpoint(completed_round=round_id)

        self.evaluate_final_on_global_test()
        self.save_training_checkpoint(
            completed_round=self.server_epochs,
            final_evaluated=True,
        )

    def log_round_summary(self, round_id, avg_train_loss, test_acc):
        message = (
            f"[RoundSummary] round={round_id} "
            f"avg_train_loss={avg_train_loss:.4f} "
            f"global_test_acc={test_acc:.4f} "
            f"best_acc={self.best_acc:.4f} "
            f"aggregation_mode={self.args.aggregation_mode}"
        )
        if self.args.aggregation_mode == "split_expert":
            message += f" expert_agg_method={self.args.expert_agg_method}"
        self.logger.info(message + "\n")

    def record_round_result(self, round_id, test_loss, test_acc):
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

    def evaluate_global_model(self, data_loader):
        self.model.to(self.device)
        self.model.eval()
        eval_metric_sums = torch.zeros(2, dtype=torch.float64, device=self.device)

        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs = inputs.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                result = self.model(inputs)
                outputs = result["logits"] if isinstance(result, dict) else result
                loss = self.criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                eval_metric_sums += torch.stack(
                    (
                        loss.detach() * inputs.size(0),
                        torch.sum(preds == labels.data),
                    )
                ).to(dtype=torch.float64)

        running_loss, running_corrects = eval_metric_sums.detach().cpu().tolist()
        eval_loss = running_loss / max(len(data_loader.dataset), 1)
        eval_acc = running_corrects / max(len(data_loader.dataset), 1)
        self.model.to("cpu")
        return eval_loss, eval_acc

    def evaluate_final_on_global_test(self):
        test_loss, test_acc = self.evaluate_global_model(self.global_test_loader)
        self.best_acc = max(self.best_acc, test_acc)
        self.logger.info(
            f"[FinalSummary] final_global_test_loss={test_loss:.4f} "
            f"final_global_test_acc={test_acc:.4f} "
            f"best_acc={self.best_acc:.4f} "
            f"selected_round={self.server_epochs}\n"
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

    def get_client_train_size(self, client_id):
        return get_client_train_size(self.args, client_id, meta=self.partition_meta)

    def aggregation_by_method(self, client_states=None, client_sizes=None):
        if client_states is None:
            client_states = [
                torch.load(
                    self.args.model_save_path + f"/{client_id}.pth",
                    map_location="cpu",
                )
                for client_id in self.clientsID_list
            ]
        if client_sizes is None:
            client_sizes = [
                self.get_client_train_size(client_id)
                for client_id in self.clientsID_list
            ]

        client_stats = getattr(self, "last_client_expert_usages", None)
        if self.args.aggregation_mode == "whole_model_uniform_avg":
            self.logger.info(
                "[Aggregator] whole_model_uniform_avg active: "
                "averaging full state_dict uniformly\n"
            )
        else:
            self.logger.info(
                "[Aggregator] split_expert active: "
                f"non_expert_agg_method={self.args.non_expert_agg_method}, "
                f"expert_agg_method={self.args.expert_agg_method}\n"
            )

        aggregated_state = self.aggregator.aggregate(
            client_updates=client_states,
            client_weights=client_sizes,
            global_model=self.model,
            client_stats=client_stats,
        )
        self.model.load_state_dict(aggregated_state)

        if self.args.aggregation_mode == "split_expert":
            if self.args.expert_agg_method == "fisher_raw_score":
                summary = getattr(self.aggregator, "last_fisher_raw_score_summary", None)
                if summary:
                    self.logger.info(f"--fisher_raw_score_summary : {summary}\n")
            if self.args.expert_agg_method == "history_wolf_filter":
                summary = getattr(self.aggregator, "last_history_wolf_summary", None)
                if summary:
                    self.logger.info(f"--history_wolf_filter_summary : {summary}\n")

        self.logger.info(f"--client_train_sizes : {client_sizes}\n")

    def aggregation(self, client_states=None, client_sizes=None):
        self.aggregation_by_method(
            client_states=client_states,
            client_sizes=client_sizes,
        )
