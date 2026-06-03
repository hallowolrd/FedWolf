import math

import torch


class HistoryWolfExpertFilter:
    def __init__(self, args, eps=1e-8):
        self.eps = float(eps)
        self.min_usage = int(getattr(args, "history_wolf_min_usage", 1))
        self.score_momentum = float(getattr(args, "history_wolf_score_momentum", 0.5))
        self.history_scores = {}
        self.last_summary = []

    def compute_weights(self, client_updates, client_stats, global_state, expert_keys_by_ref):
        self.last_summary = []
        weights_by_ref = {}

        for expert_ref, keys in sorted(expert_keys_by_ref.items(), key=lambda item: (str(item[0][0]), item[0][1])):
            layer_id, expert_id = expert_ref
            valid = []
            deltas = []

            for client_index, stats in enumerate(client_stats):
                usage = self._get_expert_usage(stats, layer_id, expert_id)
                if usage < self.min_usage:
                    continue

                delta = self._flatten_delta(client_updates[client_index], global_state, keys)
                if delta is None or torch.linalg.vector_norm(delta).item() <= self.eps:
                    continue

                valid.append(client_index)
                deltas.append(delta)

            if not valid:
                weights = [0.0 for _ in client_updates]
                skipped_reason = "no_valid_clients"
                history_values = [
                    self.history_scores.get(self._state_key(layer_id, expert_id, index), 0.0)
                    for index in range(len(client_updates))
                ]
            else:
                reference = torch.stack(deltas, dim=0).mean(dim=0)
                reference_norm = torch.linalg.vector_norm(reference).item()
                if reference_norm <= self.eps:
                    weights = [0.0 for _ in client_updates]
                    skipped_reason = "zero_reference_delta"
                    history_values = [
                        self.history_scores.get(self._state_key(layer_id, expert_id, index), 0.0)
                        for index in range(len(client_updates))
                    ]
                else:
                    weights = [0.0 for _ in client_updates]
                    skipped_reason = None
                    for client_index, delta in zip(valid, deltas):
                        score = self._cosine_score(delta, reference)
                        state_key = self._state_key(layer_id, expert_id, client_index)
                        previous = self.history_scores.get(state_key, score)
                        history_score = (
                            self.score_momentum * previous
                            + (1.0 - self.score_momentum) * score
                        )
                        self.history_scores[state_key] = float(history_score)
                        weights[client_index] = max(float(history_score), 0.0)

                    history_values = [
                        self.history_scores.get(self._state_key(layer_id, expert_id, index), 0.0)
                        for index in range(len(client_updates))
                    ]

            weights_by_ref[expert_ref] = weights
            self.last_summary.append(
                {
                    "expert_id": int(expert_id),
                    "valid_clients": [index + 1 for index in valid],
                    "weights": self._normalize_for_log(weights),
                    "history_scores": [round(float(value), 6) for value in history_values],
                    "skipped_reason": skipped_reason,
                }
            )

        return weights_by_ref

    def state_dict(self):
        return {"history_scores": dict(self.history_scores)}

    def load_state_dict(self, state):
        scores = state.get("history_scores", {}) if isinstance(state, dict) else {}
        self.history_scores = {
            str(key): float(value)
            for key, value in scores.items()
            if self._is_finite_number(value)
        }

    def _flatten_delta(self, client_state, global_state, keys):
        chunks = []
        for key in keys:
            if key not in client_state or key not in global_state:
                continue
            client_value = client_state[key].detach().cpu().float().reshape(-1)
            global_value = global_state[key].detach().cpu().float().reshape(-1)
            chunks.append(client_value - global_value)

        if not chunks:
            return None
        return torch.cat(chunks)

    def _get_expert_usage(self, stats, layer_id, expert_id):
        if not isinstance(stats, dict):
            return 0.0

        layer_stats = stats.get("expert_stats_by_layer", {}).get(str(layer_id), {})
        usage = layer_stats.get("expert_activations")
        if usage is None:
            usage = stats.get("expert_activations_by_layer", {}).get(str(layer_id))
        if usage is None:
            usage = stats.get("expert_activations")
        if usage is None:
            return 0.0

        if torch.is_tensor(usage):
            flat_usage = usage.detach().cpu().flatten()
            if expert_id >= flat_usage.numel():
                return 0.0
            return float(flat_usage[expert_id].item())

        if expert_id >= len(usage):
            return 0.0
        return float(usage[expert_id])

    def _cosine_score(self, delta, reference):
        denominator = torch.linalg.vector_norm(delta) * torch.linalg.vector_norm(reference)
        if denominator.item() <= self.eps:
            return 0.0
        cosine = torch.dot(delta, reference) / denominator
        if not torch.isfinite(cosine):
            return 0.0
        return max(float(cosine.item()), 0.0)

    def _normalize_for_log(self, weights):
        total = sum(float(weight) for weight in weights)
        if total <= 0.0:
            return [0.0 for _ in weights]
        return [round(float(weight) / total, 6) for weight in weights]

    def _state_key(self, layer_id, expert_id, client_index):
        return f"{layer_id}|{expert_id}|{client_index}"

    def _is_finite_number(self, value):
        try:
            value = float(value)
        except (TypeError, ValueError):
            return False
        return math.isfinite(value)
