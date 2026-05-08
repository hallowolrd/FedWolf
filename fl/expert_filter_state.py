import math


def _safe_float(value, default=0.0):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(value):
        return default
    return value


def predict_expert_state(mu_prev, p_prev, q):
    mu_pred = _safe_float(mu_prev)
    p_pred = max(_safe_float(p_prev, default=1.0), 1e-12) + max(_safe_float(q), 0.0)
    return mu_pred, p_pred


def compute_observation_noise(score, sigma_e2, eps):
    score = max(_safe_float(score), 0.0)
    sigma_e2 = max(_safe_float(sigma_e2, default=1.0), 1e-12)
    eps = max(_safe_float(eps, default=1e-8), 1e-12)
    return sigma_e2 / (score + eps)


def compute_imq_weight(residual, imq_c):
    residual = _safe_float(residual)
    imq_c = max(_safe_float(imq_c, default=1.0), 1e-12)
    return (1.0 + (residual * residual) / (imq_c * imq_c)) ** -0.5


def wolf_scalar_update(
    mu,
    p,
    observation,
    score,
    sigma_e2,
    eps,
    imq_c,
    observation_reliability=None,
):
    """One-dimensional WoLF-IMQ update for a scalar expert evidence state."""

    mu = _safe_float(mu)
    p = max(_safe_float(p, default=1.0), 1e-12)
    observation = _safe_float(observation)
    noise_score = score if observation_reliability is None else observation_reliability
    noise_score = max(_safe_float(noise_score), 0.0)
    r = compute_observation_noise(noise_score, sigma_e2, eps)
    residual = observation - mu
    weight = compute_imq_weight(residual, imq_c)

    obs_precision = (weight * weight) / r
    precision_new = 1.0 / p + obs_precision
    p_new = 1.0 / precision_new
    kalman_gain = p_new * obs_precision
    mu_new = mu + kalman_gain * residual

    return mu_new, max(p_new, 1e-12), {
        "R": r,
        "residual": residual,
        "weight": weight,
        "kalman_gain": kalman_gain,
        "obs_precision": obs_precision,
        "noise_score": noise_score,
        "observation_reliability": noise_score,
    }
