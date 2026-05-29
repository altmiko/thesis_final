"""
ELBO loss for MixedInputBetaVAE (Path C).

The 4 derived binary columns (TCP/UDP/ICMP/IGMP) appear nowhere in the loss.
Binary targets and protocol targets must already be precomputed in raw space
by the dataset (PerClassDataset) — do not inverse-transform here.
"""

from __future__ import annotations

import logging
import math

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

_LOG_2PI: float = math.log(2 * math.pi)
_LOG_2: float = math.log(2.0)


def _constraint_pos(partition: dict, full_feature_idx: int) -> int:
    return partition["continuous_idx"].index(full_feature_idx)


def _compute_constraint_loss(
    continuous_mu_raw: torch.Tensor,
    partition: dict,
) -> dict[str, torch.Tensor]:
    """Differentiable raw-space penalties for validator-style consistency rules."""
    ttl_idx = _constraint_pos(partition, 2)
    min_idx = _constraint_pos(partition, 31)
    max_idx = _constraint_pos(partition, 32)
    avg_idx = _constraint_pos(partition, 33)
    std_idx = _constraint_pos(partition, 34)
    number_idx = _constraint_pos(partition, 37)
    variance_idx = _constraint_pos(partition, 38)

    nonneg_penalty = torch.relu(-continuous_mu_raw).mean()
    ttl_upper_penalty = torch.relu(continuous_mu_raw[:, ttl_idx] - 255.0).mean()

    min_raw = continuous_mu_raw[:, min_idx]
    max_raw = continuous_mu_raw[:, max_idx]
    avg_raw = continuous_mu_raw[:, avg_idx]
    std_raw = continuous_mu_raw[:, std_idx]
    number_raw = continuous_mu_raw[:, number_idx]
    variance_raw = continuous_mu_raw[:, variance_idx]

    ordering_penalty = (
        torch.relu(min_raw - max_raw)
        + torch.relu(min_raw - avg_raw)
        + torch.relu(avg_raw - max_raw)
    ).mean()

    std_sq = std_raw.square()
    variance_penalty = (
        torch.abs(variance_raw - std_sq) / (std_sq.abs() + 1.0)
    ).mean()

    packet_positive_penalty = torch.relu(1.0 - number_raw).mean()
    # Squared fractional distance to the nearest integer. The previous
    # ``sin(pi * N)**2`` form loses all precision in float32 once N is large
    # (argument reduction error), so it produced meaningless gradients for
    # exactly the high-count rows it was meant to constrain. ``round()`` carries
    # no gradient, so the gradient here is the well-conditioned 2*(N - round(N)).
    packet_integer_penalty = (number_raw - number_raw.round()).square().mean()

    total = (
        nonneg_penalty
        + ttl_upper_penalty
        + ordering_penalty
        + variance_penalty
        + packet_positive_penalty
        + packet_integer_penalty
    )

    return {
        "total": total,
        "nonneg": nonneg_penalty,
        "ttl": ttl_upper_penalty,
        "ordering": ordering_penalty,
        "variance": variance_penalty,
        "packet_positive": packet_positive_penalty,
        "packet_integer": packet_integer_penalty,
    }


def _compute_physics_constraint_loss(
    continuous_mu_raw: torch.Tensor,
    partition: dict,
) -> dict[str, torch.Tensor]:
    """Differentiable raw-space penalties for post-hoc physics rules P2/P4/P5."""
    tot_sum_idx = _constraint_pos(partition, 30)
    min_idx = _constraint_pos(partition, 31)
    max_idx = _constraint_pos(partition, 32)
    avg_idx = _constraint_pos(partition, 33)
    std_idx = _constraint_pos(partition, 34)
    number_idx = _constraint_pos(partition, 37)
    variance_idx = _constraint_pos(partition, 38)

    total_sum_raw = continuous_mu_raw[:, tot_sum_idx]
    min_raw = continuous_mu_raw[:, min_idx]
    max_raw = continuous_mu_raw[:, max_idx]
    avg_raw = continuous_mu_raw[:, avg_idx]
    std_raw = continuous_mu_raw[:, std_idx]
    number_raw = continuous_mu_raw[:, number_idx]
    variance_raw = continuous_mu_raw[:, variance_idx]

    p2_total_bytes = (
        torch.abs(total_sum_raw - (number_raw * avg_raw)) / (total_sum_raw.abs() + 1.0)
    ).mean()

    std_cap = 0.5 * (max_raw - min_raw)
    p4_std_bound = torch.relu(std_raw - std_cap).mean()

    singleton_mask = number_raw <= 1.5
    singleton_count = singleton_mask.float().sum().clamp_min(1.0)
    singleton_size_match = (
        (
            torch.abs(min_raw - avg_raw)
            + torch.abs(max_raw - avg_raw)
        ) * singleton_mask.float()
    ).sum() / singleton_count
    singleton_var_zero = (
        (std_raw.abs() + variance_raw.abs()) * singleton_mask.float()
    ).sum() / singleton_count
    p5_singleton = singleton_size_match + singleton_var_zero

    total = p2_total_bytes + p4_std_bound + p5_singleton
    return {
        "total": total,
        "p2_total_bytes": p2_total_bytes,
        "p4_std_bound": p4_std_bound,
        "p5_singleton": p5_singleton,
    }


def compute_elbo(
    batch_x_39: torch.Tensor,
    model_out: dict,
    partition: dict,
    beta: float,
    target_ind_binary: torch.Tensor | None = None,
    target_protocol_idx: torch.Tensor | None = None,
    protocol_class_weights: torch.Tensor | None = None,
    protocol_loss_weight: float = 1.0,
    constraint_loss_weight: float = 0.0,
    physics_constraint_loss_weight: float = 0.0,
    continuous_feature_weights: torch.Tensor | None = None,
    binary_feature_weights: torch.Tensor | None = None,
    continuous_logvar_floor: float = -4.0,
    continuous_logvar_ceiling: float = 2.0,
    continuous_likelihood: str = "gaussian",
    continuous_nll_per_sample_cap: float | None = None,
    free_bits_lambda: float = 0.0,
    continuous_target_raw: torch.Tensor | None = None,
    raw_relative_continuous_loss_weight: float = 0.0,
    raw_relative_feature_weights: torch.Tensor | None = None,
    raw_relative_epsilon: float = 1.0,
    raw_relative_tail_focus_quantile: float | None = None,
    raw_relative_tail_focus_weight: float = 0.0,
) -> dict:
    """Compute the β-VAE ELBO for a MixedInputBetaVAE forward pass.

    Parameters
    ----------
    batch_x_39:
        (N, 39) scaled feature tensor — continuous targets are read from here.
    model_out:
        Dict returned by ``MixedInputBetaVAE.forward()``.
    partition:
        Feature partition dict from ``schema.get_partition()``.
    beta:
        Current β weight applied to the KL term.
    target_ind_binary:
        (N, 11) raw {0, 1} float tensor for independent binary features.
        Must be precomputed by the dataset — not derived here.
    target_protocol_idx:
        (N,) long tensor of protocol indices (0..5).
        Must be precomputed by the dataset — not derived here.
    protocol_class_weights:
        Optional class weights for the protocol cross-entropy term.
    protocol_loss_weight:
        Multiplier applied to the protocol reconstruction loss.
    constraint_loss_weight:
        Multiplier applied to the differentiable raw-space consistency penalty.
    continuous_feature_weights:
        Optional per-continuous-feature weights applied inside the Gaussian NLL.
    binary_feature_weights:
        Optional per-binary-feature weights applied inside BCE.
    continuous_logvar_floor:
        Lower clamp used in the continuous Gaussian NLL for stability.
    continuous_logvar_ceiling:
        Upper clamp used in the continuous Gaussian NLL for stability.
    continuous_nll_per_sample_cap:
        Optional cap on each sample's summed continuous NLL before batch averaging.
    continuous_target_raw:
        Optional raw-space continuous target tensor aligned with
        ``model_out['continuous_mu_raw']``. When provided together with a positive
        ``raw_relative_continuous_loss_weight``, an auxiliary raw relative error
        term is added to the loss.

    Returns
    -------
    dict with keys: recon_continuous, recon_independent_binary,
                    recon_pseudo_binary, recon_protocol, kl, loss.
    """
    if target_ind_binary is None:
        raise ValueError("target_ind_binary must be provided (precomputed by dataset)")
    if target_protocol_idx is None:
        raise ValueError("target_protocol_idx must be provided (precomputed by dataset)")

    mu: torch.Tensor = model_out["mu"]
    logvar: torch.Tensor = model_out["logvar"]
    continuous_mu: torch.Tensor = model_out["continuous_mu"]
    continuous_logvar: torch.Tensor = model_out["continuous_logvar"]
    continuous_mu_raw: torch.Tensor | None = model_out.get("continuous_mu_raw")
    binary_logits: torch.Tensor = model_out["binary_logits"]
    protocol_logits: torch.Tensor = model_out["protocol_logits"]
    pseudo_binary_sigmoid: torch.Tensor | None = model_out.get("pseudo_binary_sigmoid")

    # Continuous reconstruction: Gaussian NLL
    # sum over feature dim, mean over batch
    x_cont_target = batch_x_39[:, partition["continuous_idx"]]
    effective_logvar = continuous_logvar.clamp(
        min=continuous_logvar_floor,
        max=continuous_logvar_ceiling,
    )
    if continuous_likelihood == "gaussian":
        per_feature_continuous_nll = 0.5 * (
            effective_logvar
            + (x_cont_target - continuous_mu) ** 2 / effective_logvar.exp()
            + _LOG_2PI
        )
    elif continuous_likelihood == "laplace":
        # Heteroscedastic Laplace NLL: the second continuous head is reinterpreted
        # as the log-scale (log b), so NLL = log(2b) + |x - mu| / b. The L1 error
        # term is far more robust to the heavy right tails of flow statistics
        # (Rate, IAT, Tot sum) than the Gaussian L2 term, which is dominated by a
        # handful of large-magnitude rows. The decoder mean (continuous_mu) is the
        # Laplace location, so reconstruction/decoding is unaffected by this choice.
        log_scale = effective_logvar
        per_feature_continuous_nll = (
            _LOG_2 + log_scale + (x_cont_target - continuous_mu).abs() / log_scale.exp()
        )
    else:
        raise ValueError(
            f"Unknown continuous_likelihood={continuous_likelihood!r} "
            "(expected 'gaussian' or 'laplace')"
        )
    if continuous_feature_weights is not None:
        per_feature_continuous_nll = (
            per_feature_continuous_nll
            * continuous_feature_weights.to(per_feature_continuous_nll.device).unsqueeze(0)
        )
    per_sample_continuous_nll = per_feature_continuous_nll.sum(dim=1)
    if continuous_nll_per_sample_cap is not None:
        per_sample_continuous_nll = per_sample_continuous_nll.clamp(
            max=float(continuous_nll_per_sample_cap)
        )
    recon_continuous = per_sample_continuous_nll.mean()

    if (
        continuous_mu_raw is not None
        and continuous_target_raw is not None
        and raw_relative_continuous_loss_weight > 0.0
    ):
        denom = continuous_target_raw.abs() + float(raw_relative_epsilon)
        per_feature_raw_relative = torch.abs(continuous_mu_raw - continuous_target_raw) / denom
        if raw_relative_feature_weights is not None:
            per_feature_raw_relative = (
                per_feature_raw_relative
                * raw_relative_feature_weights.to(per_feature_raw_relative.device).unsqueeze(0)
            )
        recon_continuous_raw_relative = per_feature_raw_relative.sum(dim=1).mean()
        if raw_relative_tail_focus_weight > 0.0 and raw_relative_tail_focus_quantile is not None:
            per_sample_raw_relative = per_feature_raw_relative.sum(dim=1)
            threshold = torch.quantile(
                per_sample_raw_relative.detach(),
                float(raw_relative_tail_focus_quantile),
            )
            tail_mask = per_sample_raw_relative >= threshold
            recon_continuous_raw_relative_tail = per_sample_raw_relative[tail_mask].mean()
        else:
            recon_continuous_raw_relative_tail = torch.tensor(0.0, device=mu.device)
    else:
        recon_continuous_raw_relative = torch.tensor(0.0, device=mu.device)
        recon_continuous_raw_relative_tail = torch.tensor(0.0, device=mu.device)

    # Independent binary reconstruction: BCE with logits
    # reduction='none' → (N, 11), sum over features, mean over batch
    per_feature_binary_bce = F.binary_cross_entropy_with_logits(
        binary_logits, target_ind_binary, reduction="none"
    )
    if binary_feature_weights is not None:
        per_feature_binary_bce = (
            per_feature_binary_bce
            * binary_feature_weights.to(per_feature_binary_bce.device).unsqueeze(0)
        )
    recon_independent_binary = per_feature_binary_bce.sum(dim=1).mean()

    # Pseudo-binary: MSE (no pseudo columns in this project, kept for completeness)
    if pseudo_binary_sigmoid is not None and pseudo_binary_sigmoid.numel() > 0:
        pseudo_idx = partition.get("pseudo_binary_idx", [])
        x_pseudo_target = batch_x_39[:, pseudo_idx]
        recon_pseudo_binary = F.mse_loss(pseudo_binary_sigmoid, x_pseudo_target)
    else:
        recon_pseudo_binary = torch.tensor(0.0, device=mu.device)

    # Protocol reconstruction: cross-entropy over 6 classes, mean over batch
    # Scaled by 1.0 to match units — single categorical, but architecturally important
    recon_protocol = F.cross_entropy(
        protocol_logits,
        target_protocol_idx,
        reduction="mean",
        weight=protocol_class_weights,
    )

    # KL divergence to N(0, I): -0.5 * sum_latent(1 + logvar - mu^2 - exp(logvar))
    # sum over latent dim, mean over batch
    per_dim_kl = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())
    if free_bits_lambda > 0.0:
        kl = per_dim_kl.clamp(min=float(free_bits_lambda)).sum(dim=1).mean()
    else:
        kl = per_dim_kl.sum(dim=1).mean()

    if continuous_mu_raw is not None and constraint_loss_weight > 0.0:
        constraint_terms = _compute_constraint_loss(continuous_mu_raw, partition)
        constraint_loss = constraint_terms["total"]
    else:
        zero = torch.tensor(0.0, device=mu.device)
        constraint_terms = {
            "total": zero,
            "nonneg": zero,
            "ttl": zero,
            "ordering": zero,
            "variance": zero,
            "packet_positive": zero,
            "packet_integer": zero,
        }
        constraint_loss = zero

    if continuous_mu_raw is not None and physics_constraint_loss_weight > 0.0:
        physics_constraint_terms = _compute_physics_constraint_loss(continuous_mu_raw, partition)
        physics_constraint_loss = physics_constraint_terms["total"]
    else:
        zero = torch.tensor(0.0, device=mu.device)
        physics_constraint_terms = {
            "total": zero,
            "p2_total_bytes": zero,
            "p4_std_bound": zero,
            "p5_singleton": zero,
        }
        physics_constraint_loss = zero

    loss = (
        recon_continuous
        + recon_independent_binary
        + protocol_loss_weight * recon_protocol
        + beta * kl
        + constraint_loss_weight * constraint_loss
        + physics_constraint_loss_weight * physics_constraint_loss
        + raw_relative_continuous_loss_weight * recon_continuous_raw_relative
        + raw_relative_tail_focus_weight * recon_continuous_raw_relative_tail
    )
    if pseudo_binary_sigmoid is not None and pseudo_binary_sigmoid.numel() > 0:
        loss = loss + recon_pseudo_binary

    return {
        "recon_continuous": recon_continuous,
        "recon_independent_binary": recon_independent_binary,
        "recon_continuous_raw_relative": recon_continuous_raw_relative,
        "recon_continuous_raw_relative_tail": recon_continuous_raw_relative_tail,
        "recon_pseudo_binary": recon_pseudo_binary,
        "recon_protocol": recon_protocol,
        "kl": kl,
        "per_dim_kl_mean": per_dim_kl.mean(dim=0),
        "constraint_loss": constraint_loss,
        "constraint_nonneg": constraint_terms["nonneg"],
        "constraint_ttl": constraint_terms["ttl"],
        "constraint_ordering": constraint_terms["ordering"],
        "constraint_variance": constraint_terms["variance"],
        "constraint_packet_positive": constraint_terms["packet_positive"],
        "constraint_packet_integer": constraint_terms["packet_integer"],
        "physics_constraint_loss": physics_constraint_loss,
        "physics_p2_total_bytes": physics_constraint_terms["p2_total_bytes"],
        "physics_p4_std_bound": physics_constraint_terms["p4_std_bound"],
        "physics_p5_singleton": physics_constraint_terms["p5_singleton"],
        "loss": loss,
    }


class BetaScheduler:
    """Linear warmup β scheduler.

    β increases linearly from 0 to ``beta_target`` over the first
    ``warmup_frac * total_steps`` steps, then stays constant.
    """

    def __init__(
        self,
        beta_target: float,
        total_steps: int,
        warmup_frac: float = 0.3,
        warmup_steps: int | None = None,
    ) -> None:
        self.beta_target = beta_target
        self.total_steps = total_steps
        # ``warmup_steps`` takes precedence when provided. Deriving warmup from
        # ``warmup_frac * total_steps`` ties it to ``max_epochs`` (e.g. 200), so
        # with early stopping β often never reaches its target before the run
        # ends. Callers should pass an epoch-based ``warmup_steps`` instead.
        if warmup_steps is not None:
            self.warmup_steps = int(warmup_steps)
        else:
            self.warmup_steps = int(warmup_frac * total_steps)
        self._step_count: int = 0
        self._current_beta: float = 0.0

    def step(self) -> float:
        """Advance one step and return current beta."""
        if self.warmup_steps > 0 and self._step_count < self.warmup_steps:
            # Linear interpolation: step 0 → first nonzero increment, step warmup_steps-1 → beta_target
            self._current_beta = self.beta_target * (self._step_count + 1) / self.warmup_steps
        else:
            self._current_beta = self.beta_target
        self._step_count += 1
        return self._current_beta

    @property
    def current_beta(self) -> float:
        return self._current_beta
