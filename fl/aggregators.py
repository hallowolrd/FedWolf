import collections
import math

import torch

from fl.history_wolf_filter import HistoryWolfExpertFilter


def parse_expert_ref_from_key(key):
    parts = key.split(".")
    if "experts" not in parts:
        return None

    experts_idx = parts.index("experts")
    if experts_idx + 1 >= len(parts) or not parts[experts_idx + 1].isdigit():
        return None

    if parts[:experts_idx] == ["moe_head"]:
        layer_id = "moe_head"
    elif "blocks" in parts:
        blocks_idx = parts.index("blocks")
        if blocks_idx + 1 >= len(parts) or not parts[blocks_idx + 1].isdigit():
            return None
        layer_id = parts[blocks_idx + 1]
    else:
        layer_id = ".".join(parts[:experts_idx]) or "experts"

    return layer_id, int(parts[experts_idx + 1])


class WholeModelUniformAggregator:
    def aggregate(self, client_updates, client_weights=None, global_model=None, **kwargs):
        if not client_updates:
            raise ValueError("WholeModelUniformAggregator requires at least one client update")

        num_clients = len(client_updates)
        reference_state = (
            global_model.state_dict()
            if global_model is not None
            else client_updates[0]
        )

        aggregated_state = collections.OrderedDict()
        for key, reference_value in reference_state.items():
            avg_value = torch.zeros_like(
                reference_value.detach().cpu(),
                dtype=torch.float32,
                device="cpu",
            )
            for state in client_updates:
                avg_value += state[key].detach().cpu().float() / num_clients
            aggregated_state[key] = avg_value.to(dtype=reference_value.dtype)

        return aggregated_state

    def state_dict(self):
        return {}

    def load_state_dict(self, state):
        return


class SplitExpertAggregator:
    NON_EXPERT_METHODS = {"equal_avg", "sample_weighted_avg"}
    EXPERT_METHODS = {
        "equal_avg",
        "sample_weighted_avg",
        "fisher_raw_score",
        "history_wolf_filter",
    }

    def __init__(self, args):
        self.non_expert_agg_method = str(args.non_expert_agg_method).strip().lower()
        self.expert_agg_method = str(args.expert_agg_method).strip().lower()
        if self.non_expert_agg_method not in self.NON_EXPERT_METHODS:
            raise ValueError(
                "non_expert_agg_method must be one of "
                f"{sorted(self.NON_EXPERT_METHODS)}, got {self.non_expert_agg_method!r}"
            )
        if self.expert_agg_method not in self.EXPERT_METHODS:
            raise ValueError(
                "expert_agg_method must be one of "
                f"{sorted(self.EXPERT_METHODS)}, got {self.expert_agg_method!r}"
            )

        self.eps = float(getattr(args, "fedwolf_eps", 1e-8))
        self.history_wolf_filter = HistoryWolfExpertFilter(args, eps=self.eps)
        self._history_wolf_weights_cache = None
        self._fisher_weights_cache = {}
        self.last_fisher_raw_score_summary = []

    def aggregate(self, client_updates, client_weights, global_model=None, **kwargs):
        self._validate_inputs(client_updates, client_weights)
        self._history_wolf_weights_cache = None
        self._fisher_weights_cache = {}
        self.last_fisher_raw_score_summary = []

        client_stats = kwargs.get("client_stats")
        global_state = global_model.state_dict() if global_model is not None else None

        if self.expert_agg_method in {"fisher_raw_score", "history_wolf_filter"}:
            if client_stats is None:
                raise ValueError(
                    f"expert_agg_method={self.expert_agg_method!r} requires client_stats"
                )
            if len(client_stats) != len(client_updates):
                raise ValueError("client_stats and client_updates must have the same length")

        if self.expert_agg_method == "history_wolf_filter":
            if global_state is None:
                raise ValueError("history_wolf_filter requires global_model")
            expert_keys_by_ref = self._collect_expert_keys_by_ref(client_updates[0].keys())
            self._history_wolf_weights_cache = self.history_wolf_filter.compute_weights(
                client_updates=client_updates,
                client_stats=client_stats,
                global_state=global_state,
                expert_keys_by_ref=expert_keys_by_ref,
            )

        aggregated_state = collections.OrderedDict()
        for key in client_updates[0].keys():
            first_value = client_updates[0][key].detach().cpu()
            if not torch.is_floating_point(first_value):
                aggregated_state[key] = first_value.clone()
                continue

            expert_ref = parse_expert_ref_from_key(key)
            if expert_ref is None:
                weights, add_eps = self._get_non_expert_weights(client_weights)
            else:
                weights, add_eps = self._get_expert_weights(
                    client_weights=client_weights,
                    client_stats=client_stats,
                    expert_ref=expert_ref,
                )

            total_weight = sum(weights)
            if total_weight <= 0.0:
                if global_state is not None:
                    aggregated_state[key] = global_state[key].detach().cpu().clone()
                else:
                    aggregated_state[key] = first_value.clone()
                continue

            denominator = total_weight + (self.eps if add_eps else 0.0)
            avg_value = torch.zeros_like(first_value, dtype=torch.float32, device="cpu")
            for update, weight in zip(client_updates, weights):
                avg_value += update[key].detach().cpu().float() * (float(weight) / denominator)
            aggregated_state[key] = avg_value.to(dtype=first_value.dtype)

        return aggregated_state

    @property
    def last_history_wolf_summary(self):
        return getattr(self.history_wolf_filter, "last_summary", None)

    def state_dict(self):
        return {"history_wolf_filter": self.history_wolf_filter.state_dict()}

    def load_state_dict(self, state):
        if state:
            self.history_wolf_filter.load_state_dict(state.get("history_wolf_filter", {}))

    def _validate_inputs(self, client_updates, client_weights):
        if not client_updates:
            raise ValueError("SplitExpertAggregator requires at least one client update")
        if len(client_updates) != len(client_weights):
            raise ValueError("client_updates and client_weights must have the same length")
        if any(float(weight) < 0.0 for weight in client_weights):
            raise ValueError("client_weights must be non-negative")
        if sum(float(weight) for weight in client_weights) <= 0.0:
            raise ValueError("client_weights must contain at least one positive value")

    def _collect_expert_keys_by_ref(self, keys):
        expert_keys_by_ref = collections.defaultdict(list)
        for key in keys:
            expert_ref = parse_expert_ref_from_key(key)
            if expert_ref is not None:
                expert_keys_by_ref[expert_ref].append(key)
        return dict(expert_keys_by_ref)

    def _get_non_expert_weights(self, client_weights):
        if self.non_expert_agg_method == "equal_avg":
            return [1.0 for _ in client_weights], False
        if self.non_expert_agg_method == "sample_weighted_avg":
            return [float(weight) for weight in client_weights], False
        raise ValueError(f"Unknown non_expert_agg_method: {self.non_expert_agg_method}")

    def _get_expert_weights(self, client_weights, client_stats, expert_ref):
        if self.expert_agg_method == "equal_avg":
            return [1.0 for _ in client_weights], False
        if self.expert_agg_method == "sample_weighted_avg":
            return [float(weight) for weight in client_weights], False
        if self.expert_agg_method == "fisher_raw_score":
            return self._get_fisher_weights(client_stats, expert_ref), True
        if self.expert_agg_method == "history_wolf_filter":
            if self._history_wolf_weights_cache is None:
                raise RuntimeError("history_wolf_filter weights were not prepared")
            return self._history_wolf_weights_cache.get(
                expert_ref,
                [0.0 for _ in client_weights],
            ), False
        raise ValueError(f"Unknown expert_agg_method: {self.expert_agg_method}")

    def _get_fisher_weights(self, client_stats, expert_ref):
        if expert_ref in self._fisher_weights_cache:
            return self._fisher_weights_cache[expert_ref]

        layer_id, expert_id = expert_ref
        fisher_scores = [
            self._get_fisher_score(client_stat, layer_id, expert_id)
            for client_stat in client_stats
        ]
        weights = [max(score, 0.0) for score in fisher_scores]
        valid_clients = [
            index + 1
            for index, weight in enumerate(weights)
            if weight > 0.0
        ]
        skipped_reason = None if valid_clients else "all_fisher_scores_non_positive"
        self.last_fisher_raw_score_summary.append(
            {
                "expert_id": int(expert_id),
                "valid_clients": valid_clients,
                "weights": self._normalize_for_log(weights),
                "fisher_scores": [float(score) for score in fisher_scores],
                "skipped_reason": skipped_reason,
            }
        )
        self._fisher_weights_cache[expert_ref] = weights
        return weights

    def _get_fisher_score(self, client_stats, layer_id, expert_id):
        score = self._get_layer_expert_value(
            client_stats=client_stats,
            field_name="expert_fisher_score_by_layer",
            layer_id=layer_id,
            expert_id=expert_id,
        )
        if score is None:
            return 0.0
        return max(score, 0.0)

    def _get_layer_expert_value(self, client_stats, field_name, layer_id, expert_id):
        if not isinstance(client_stats, dict):
            return None

        value_by_layer = client_stats.get(field_name, {})
        layer_values = value_by_layer.get(str(layer_id))
        if layer_values is None:
            return None

        if torch.is_tensor(layer_values):
            flat_values = layer_values.detach().cpu().flatten()
            if expert_id >= flat_values.numel():
                return None
            value = flat_values[expert_id].item()
        else:
            if expert_id >= len(layer_values):
                return None
            value = layer_values[expert_id]

        value = float(value)
        if not math.isfinite(value):
            return None
        return value

    def _normalize_for_log(self, weights):
        total = sum(float(weight) for weight in weights)
        if total <= 0.0:
            return [0.0 for _ in weights]
        return [round(float(weight) / total, 6) for weight in weights]


def build_aggregator(args):
    aggregation_mode = str(getattr(args, "aggregation_mode", "whole_model_uniform_avg")).strip().lower()
    if aggregation_mode == "whole_model_uniform_avg":
        return WholeModelUniformAggregator()
    if aggregation_mode == "split_expert":
        return SplitExpertAggregator(args)
    raise ValueError(
        "aggregation_mode must be 'whole_model_uniform_avg' or 'split_expert', "
        f"got {aggregation_mode!r}"
    )
