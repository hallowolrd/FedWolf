import torch
import torch.optim as optim
from torch import nn
from types import SimpleNamespace

from data.loader import build_client_evidence_loader, build_client_train_loader
from fl.expert_evidence import compute_expert_fisher_evidence
from model import build_model_from_args
from utils.utils import record_result


class Client:
    def __init__(
        self,
        args: SimpleNamespace,
        client_id: int,
        logger,
        c_T: int,
        partition_meta=None,
        server_state_dict=None,
    ):
        self.args = args
        self.client_id = client_id
        self.logger = logger
        self.c_T = c_T
        self.server_state_dict = server_state_dict
        self.device = self.args.device
        self.client_epochs = self.args.client_epochs
        self.batch_size = self.args.batch_size
        self.partition_meta = partition_meta
        self.model_path = self.args.model_save_path + f"/{self.client_id}.pth"

        self.model = build_model_from_args(self.args)
        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=float(getattr(self.args, "label_smooth", 0.0))
        )
        self.optimizer = self._build_optimizer()

        self.train_loader = None
        self.evidence_loader = None
        self.get_dataloader()

    def _build_optimizer(self):
        optimizer_name = str(getattr(self.args, "optimizer", "sgd")).strip().lower()
        if optimizer_name == "sgd":
            return optim.SGD(
                self.model.parameters(),
                lr=float(self.args.learning_rate),
                momentum=float(getattr(self.args, "momentum", 0.0)),
                weight_decay=float(getattr(self.args, "weight_decay", 0.0)),
            )
        if optimizer_name == "adam":
            return optim.Adam(
                self.model.parameters(),
                lr=float(self.args.learning_rate),
                weight_decay=float(getattr(self.args, "weight_decay", 0.0)),
            )
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    def should_compute_fisher_evidence(self):
        return (
            getattr(self.args, "aggregation_mode", None) == "split_expert"
            and getattr(self.args, "expert_agg_method", None) == "fisher_raw_score"
        )

    def get_dataloader(self):
        self.train_loader = build_client_train_loader(
            args=self.args,
            client_id=self.client_id,
            meta=self.partition_meta,
        )
        if (
            self.should_compute_fisher_evidence()
            and getattr(self.args, "fedwolf_evidence_loader_mode", "deterministic")
            == "deterministic"
        ):
            self.evidence_loader = build_client_evidence_loader(
                args=self.args,
                client_id=self.client_id,
                meta=self.partition_meta,
            )

    def get_fisher_data_loader(self):
        mode = getattr(self.args, "fedwolf_evidence_loader_mode", "deterministic")
        if mode == "deterministic":
            return self.evidence_loader
        if mode == "train_loader":
            return self.train_loader
        raise ValueError(
            "fedwolf_evidence_loader_mode must be either 'deterministic' or "
            f"'train_loader', got {mode!r}."
        )

    def renew_model(self, server_state_dict=None):
        if server_state_dict is None:
            server_state_dict = self.server_state_dict
        if server_state_dict is not None:
            self.model.load_state_dict(server_state_dict)
            return

        server_state_dict = torch.load(
            self.args.model_save_path + "/server.pth",
            map_location="cpu",
        )
        self.model.load_state_dict(server_state_dict)

    def get_cpu_state_dict(self):
        return {
            key: value.detach().cpu().clone()
            for key, value in self.model.state_dict().items()
        }

    def get_auxiliary_losses(self, result):
        zero = torch.tensor(0.0, device=self.device)
        return zero, zero, zero

    def _extract_logits(self, result):
        if isinstance(result, dict):
            return result["logits"]
        return result

    def _get_routing_stats(self, result):
        if isinstance(result, dict):
            if "expert_stats_by_layer" in result or "expert_activations" in result:
                return result
            return None

        get_stats = getattr(self.model, "get_moe_stats", None)
        if get_stats is None:
            return None
        return get_stats()

    def _add_routing_stats(self, totals, stats, batch_size):
        if not stats:
            return False

        usage = stats.get("expert_activations")
        avg_router_probs = stats.get("avg_router_probs")
        if usage is None and avg_router_probs is None and not stats.get("expert_stats_by_layer"):
            return False

        if usage is not None:
            totals["expert_activations"] += usage.detach().to(self.device).float()
        if avg_router_probs is not None:
            totals["router_prob_sum"] += (
                avg_router_probs.detach().to(self.device).float() * batch_size
            )
            totals["router_prob_samples"] += batch_size

        for layer_id, layer_stats in stats.get("expert_stats_by_layer", {}).items():
            layer_key = str(layer_id)
            if layer_key not in totals["expert_stats_by_layer"]:
                totals["expert_stats_by_layer"][layer_key] = {
                    "expert_activations": torch.zeros(self.args.num_experts, device=self.device),
                    "selected_counts": torch.zeros(self.args.num_experts, device=self.device),
                    "router_prob_sum": torch.zeros(self.args.num_experts, device=self.device),
                    "router_prob_samples": 0,
                }

            target = totals["expert_stats_by_layer"][layer_key]
            layer_usage = layer_stats.get("expert_activations")
            selected_counts = layer_stats.get("selected_counts", layer_usage)
            layer_probs = layer_stats.get("avg_router_probs")
            if layer_usage is not None:
                target["expert_activations"] += layer_usage.detach().to(self.device).float()
            if selected_counts is not None:
                target["selected_counts"] += selected_counts.detach().to(self.device).float()
            if layer_probs is not None:
                target["router_prob_sum"] += layer_probs.detach().to(self.device).float() * batch_size
                target["router_prob_samples"] += batch_size

        return True

    def _new_routing_totals(self):
        return {
            "expert_activations": torch.zeros(self.args.num_experts, device=self.device),
            "router_prob_sum": torch.zeros(self.args.num_experts, device=self.device),
            "router_prob_samples": 0,
            "expert_stats_by_layer": {},
        }

    def _finalize_layer_stats(self, totals):
        finalized = {}
        for layer_id, stats in totals["expert_stats_by_layer"].items():
            samples = max(int(stats["router_prob_samples"]), 1)
            finalized[layer_id] = {
                "expert_activations": stats["expert_activations"].detach().cpu(),
                "selected_counts": stats["selected_counts"].detach().cpu(),
                "avg_router_probs": (stats["router_prob_sum"] / samples).detach().cpu(),
            }
        return finalized

    def train(self):
        self.renew_model()
        local_totals = self._new_routing_totals()
        local_has_routing_stats = False
        last_train_loss = 0.0
        last_train_acc = 0.0

        for epoch in range(self.client_epochs):
            self.model.train()
            epoch_metric_sums = torch.zeros(2, dtype=torch.float64, device=self.device)
            running_corrects = torch.zeros((), dtype=torch.long, device=self.device)
            total_samples = 0
            epoch_totals = self._new_routing_totals()
            epoch_has_routing_stats = False

            for inputs, labels in self.train_loader:
                inputs = inputs.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                self.optimizer.zero_grad()
                result = self.model(inputs)
                outputs = self._extract_logits(result)
                loss = self.criterion(outputs, labels)
                loss.backward()

                grad_clip_norm = float(getattr(self.args, "grad_clip_norm", 0.0))
                if grad_clip_norm > 0.0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip_norm)

                self.optimizer.step()

                batch_size = inputs.size(0)
                total_samples += batch_size
                _, preds = torch.max(outputs, 1)
                running_corrects += torch.sum(preds == labels.data)
                epoch_metric_sums += torch.stack(
                    (loss.detach() * batch_size, loss.detach() * batch_size)
                ).to(dtype=torch.float64)

                routing_stats = self._get_routing_stats(result)
                if self._add_routing_stats(epoch_totals, routing_stats, batch_size):
                    epoch_has_routing_stats = True
                if self._add_routing_stats(local_totals, routing_stats, batch_size):
                    local_has_routing_stats = True

            running_loss, running_ce_loss = epoch_metric_sums.detach().cpu().tolist()
            running_correct_count = running_corrects.detach().cpu().item()
            dataset_size = len(self.train_loader.dataset)
            last_train_loss = running_loss / max(dataset_size, 1)
            last_train_acc = running_correct_count / max(dataset_size, 1)
            avg_ce_loss = running_ce_loss / max(total_samples, 1)

            routing_log = "--routing_stats: unavailable_for_logits_only_forward"
            if epoch_has_routing_stats:
                usage_list = [
                    int(v)
                    for v in epoch_totals["expert_activations"].detach().cpu().tolist()
                ]
                prob_samples = max(int(epoch_totals["router_prob_samples"]), 1)
                router_prob_list = [
                    round(float(v), 4)
                    for v in (epoch_totals["router_prob_sum"] / prob_samples).detach().cpu().tolist()
                ]
                routing_log = (
                    f"--expert_usage : {usage_list} "
                    f"--avg_router_probs : {router_prob_list}"
                )

            self.logger.info(
                f"--client: {self.client_id} --epoch:{epoch + 1}/{self.client_epochs} "
                f"--train_loss :{last_train_loss:.4f} --train_acc :{last_train_acc:.4f} "
                f"--ce_loss : {avg_ce_loss:.4f} {routing_log}"
            )

            record_result(
                record_dic={
                    "T": self.c_T,
                    "client_epoch": epoch + 1,
                    "client_id": self.client_id,
                    "train_loss": last_train_loss,
                    "train_acc": last_train_acc,
                    "ce_loss": avg_ce_loss,
                    "router_balance_loss": 0.0,
                    "router_balance_loss_coef": 0.0,
                    "router_aux_loss": 0.0,
                    "router_z_loss": 0.0,
                },
                args=self.args,
            )

        fisher_score_by_layer = {}
        fisher_log_score_by_layer = {}
        if self.should_compute_fisher_evidence():
            fisher_score_by_layer, fisher_log_score_by_layer = compute_expert_fisher_evidence(
                model=self.model,
                data_loader=self.get_fisher_data_loader(),
                criterion=self.criterion,
                device=self.device,
                num_experts=self.args.num_experts,
                get_auxiliary_losses=self.get_auxiliary_losses,
                return_diagnostics=False,
                model_mode=getattr(self.args, "fedwolf_evidence_model_mode", "eval"),
                score_mode=getattr(self.args, "fedwolf_fisher_score_mode", "trace_per_active_sample"),
                debug_batches=int(getattr(self.args, "fedwolf_fisher_debug_batches", 0)),
                fisher_estimator=getattr(self.args, "fedwolf_fisher_estimator", "linear_hook_token_fast"),
            )

        local_state_dict = self.get_cpu_state_dict()
        if bool(getattr(self.args, "save_client_models", False)):
            torch.save(local_state_dict, self.model_path)

        layer_stats_cpu = self._finalize_layer_stats(local_totals)
        if local_has_routing_stats:
            prob_samples = max(int(local_totals["router_prob_samples"]), 1)
            expert_activations = local_totals["expert_activations"].detach().cpu()
            avg_router_probs = (local_totals["router_prob_sum"] / prob_samples).detach().cpu()
        else:
            expert_activations = None
            avg_router_probs = None

        return {
            "routing_stats_available": bool(local_has_routing_stats),
            "expert_activations": expert_activations,
            "avg_router_probs": avg_router_probs,
            "expert_stats_by_layer": layer_stats_cpu,
            "expert_activations_by_layer": {
                layer_id: stats["expert_activations"]
                for layer_id, stats in layer_stats_cpu.items()
            },
            "expert_fisher_score_by_layer": fisher_score_by_layer,
            "expert_fisher_log_score_by_layer": fisher_log_score_by_layer,
            "train_loss": float(last_train_loss),
            "train_acc": float(last_train_acc),
            "local_state_dict": local_state_dict,
        }
