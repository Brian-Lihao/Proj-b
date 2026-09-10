import torch
import torch.nn.functional as F


SUPPORTED_METHODS = (
    "spo",
    "ppr_linear",
    "ppr_mid",
    "ppr_hard",
    "direct_lpips",
    "snr_lpips",
)
PPR_METHODS = frozenset(("ppr_linear", "ppr_mid", "ppr_hard"))
REFERENCE_LPIPS_METHODS = frozenset(("direct_lpips", "snr_lpips"))
LPIPS_METHODS = PPR_METHODS | REFERENCE_LPIPS_METHODS


def validate_method(method):
    method = str(method).lower()
    if method not in SUPPORTED_METHODS:
        choices = ", ".join(SUPPORTED_METHODS)
        raise ValueError(f"Unknown training method {method!r}. Choose one of: {choices}.")
    return method


def preference_loss(log_ratio_diff, beta):
    """Per-sample reference-normalized preference loss."""
    return F.softplus(-float(beta) * log_ratio_diff)


def ppr_weight(
    distance,
    method,
    scale,
    max_distance=0.0,
    mid_mu=0.3,
    mid_sigma=0.15,
    hard_gamma=5.0,
):
    """Build a detached perceptual weight for a PPR method."""
    if method not in PPR_METHODS:
        raise ValueError(f"{method!r} is not a PPR reweighting method")

    distance = distance.detach().reshape(-1).float()
    distance = torch.nan_to_num(
        distance, nan=0.0, posinf=0.0, neginf=0.0
    ).clamp_min(0.0)
    if max_distance > 0:
        distance = distance.clamp_max(float(max_distance))

    scale = float(scale)
    if method == "ppr_linear":
        return 1.0 + scale * distance
    if method == "ppr_mid":
        mid_sigma = float(mid_sigma)
        if mid_sigma <= 0:
            raise ValueError("train.ppr_mid_sigma must be greater than zero")
        exponent = -((distance - float(mid_mu)) ** 2) / (2.0 * mid_sigma**2)
        return 1.0 + scale * torch.exp(exponent)
    return 1.0 + scale * torch.exp(-float(hard_gamma) * distance)
