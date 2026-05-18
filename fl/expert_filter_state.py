import math

import torch


def safe_float(value, default=0.0):
    """
    将 value 转成有限 Python float。

    如果 value 为 None、不能转换或不是有限值，则返回 default。
    """

    if value is None:
        return default

    if torch.is_tensor(value):
        if value.numel() != 1:
            return default
        try:
            value = value.detach().float().item()
        except (RuntimeError, TypeError, ValueError):
            return default

    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(value):
        return default
    return value


def predict_expert_state(mu_prev, p_prev, q, eps=1e-12):
    """
    单个 expert evidence state 的 random-walk predict step。

    mu_pred = mu_prev
    P_pred = P_prev + q
    """

    eps = max(safe_float(eps, default=1e-12), 1e-12)
    mu_pred = safe_float(mu_prev, default=0.0)
    p_prev = safe_float(p_prev, default=1.0)
    q = safe_float(q, default=0.0)

    p_prev = max(p_prev, eps)
    q = max(q, 0.0)
    p_pred = max(p_prev + q, eps)
    return mu_pred, p_pred


def compute_support_noise(n_active, mean_positive_n, sigma_e2, eps=1e-12):
    """
    计算由 activation support 得到的 observation noise。

    n_active: client m 和 expert k 的 activation support。
    mean_positive_n: 该 expert 已选 client 中正 activation support 的均值。
    sigma_e2: 基础 observation noise。
    """

    eps = max(safe_float(eps, default=1e-12), 1e-12)
    n_active = safe_float(n_active, default=None)
    mean_positive_n = safe_float(mean_positive_n, default=0.0)
    sigma_e2 = max(safe_float(sigma_e2, default=1.0), eps)

    if n_active is None or n_active <= 0.0 or mean_positive_n <= 0.0:
        n_rel = 0.0
    else:
        n_rel = n_active / (mean_positive_n + eps)

    support_reliability = math.sqrt(max(n_rel, 0.0) + eps)
    r = sigma_e2 / (support_reliability + eps)
    if not math.isfinite(r) or r <= 0.0:
        r = sigma_e2 / eps
    return r, n_rel, support_reliability


def compute_standardized_residual(observation, mu_pred, p_pred, R, eps=1e-12):
    """
    计算 robust filtering 的 standardized residual。

    nu = (z - mu_pred) / sqrt(P_pred + R + eps)
    """

    eps = max(safe_float(eps, default=1e-12), 1e-12)
    z = safe_float(observation, default=0.0)
    mu_pred = safe_float(mu_pred, default=0.0)
    p_pred = max(safe_float(p_pred, default=1.0), eps)
    R = max(safe_float(R, default=1.0), eps)

    denom = math.sqrt(p_pred + R + eps)
    if not math.isfinite(denom) or denom <= 0.0:
        return 0.0
    nu = (z - mu_pred) / denom
    if not math.isfinite(nu):
        return 0.0
    return nu


def compute_imq_weight(standardized_residual, imq_c, eps=1e-12):
    """
    根据 standardized residual 计算 IMQ robust weight。

    rho = (1 + nu^2 / c^2)^(-1/2)
    """

    eps = max(safe_float(eps, default=1e-12), 1e-12)
    nu = safe_float(standardized_residual, default=0.0)
    c = max(safe_float(imq_c, default=1.0), eps)
    rho = (1.0 + (nu * nu) / (c * c + eps)) ** -0.5
    if not math.isfinite(rho):
        return 1.0
    return min(max(rho, 0.0), 1.0)


def compute_filter_precision(rho, R, eps=1e-12):
    """
    计算 evidence reliability precision。

    lambda_filter = rho^2 / (R + eps)
    """

    eps = max(safe_float(eps, default=1e-12), 1e-12)
    rho = safe_float(rho, default=1.0)
    rho = min(max(rho, 0.0), 1.0)
    R = max(safe_float(R, default=1.0), eps)

    lambda_filter = (rho * rho) / (R + eps)
    if not math.isfinite(lambda_filter) or lambda_filter < 0.0:
        return 0.0
    return lambda_filter


def batch_update_filter_state(mu_pred, p_pred, observations, lambda_filters, eps=1e-12):
    """
    单个 expert evidence state 的 batch 形式 precision update。

    P_new = 1 / (1 / P_pred + sum_m lambda_filter_m)
    mu_new = P_new * (mu_pred / P_pred + sum_m lambda_filter_m * z_m)
    """

    eps = max(safe_float(eps, default=1e-12), 1e-12)
    mu_pred = safe_float(mu_pred, default=0.0)
    p_pred = max(safe_float(p_pred, default=1.0), eps)

    valid_pairs = []
    for observation, lambda_filter in zip(observations or [], lambda_filters or []):
        z = safe_float(observation, default=None)
        lam = safe_float(lambda_filter, default=0.0)
        if z is None or lam <= 0.0:
            continue
        valid_pairs.append((z, lam))

    if not valid_pairs:
        return mu_pred, p_pred

    prior_precision = 1.0 / (p_pred + eps)
    posterior_precision = prior_precision + sum(lam for _, lam in valid_pairs)
    if not math.isfinite(posterior_precision) or posterior_precision <= 0.0:
        return mu_pred, p_pred

    p_new = max(1.0 / (posterior_precision + eps), eps)
    mu_numer = mu_pred / (p_pred + eps)
    for z, lam in valid_pairs:
        mu_numer += lam * z

    mu_new = p_new * mu_numer
    if not math.isfinite(mu_new) or not math.isfinite(p_new):
        return mu_pred, p_pred
    return mu_new, p_new
