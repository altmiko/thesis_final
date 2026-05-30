from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from attack.latent_infra import (
    PerturbationMask,
    apply_decoder_residual,
    reimpose_protocol_features,
)


MetadataValue = torch.Tensor | float | int | bool | str | list[int] | list[float]


def classifier_logits(classifier: Any, x_batch: torch.Tensor, *, device: str) -> torch.Tensor:
    if not isinstance(classifier, torch.nn.Module):
        raise TypeError(f"Latent gradient attacks require a torch classifier, got {type(classifier)!r}")

    classifier = classifier.to(device)
    classifier.eval()
    # Disable CuDNN for attack-time classifier passes so recurrent baselines can
    # backpropagate in eval mode on CUDA without tripping the CuDNN RNN guard.
    with torch.backends.cudnn.flags(enabled=False):
        logits = classifier(x_batch.to(device))
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits


def _target_tensor(y_true: torch.Tensor, target_class: int) -> torch.Tensor:
    return torch.full_like(y_true, int(target_class), dtype=torch.long)


def _objective_loss(
    logits: torch.Tensor,
    y_true: torch.Tensor,
    *,
    targeted: bool,
    target_class: int,
) -> torch.Tensor:
    if targeted:
        return -F.cross_entropy(logits, _target_tensor(y_true, target_class))
    return F.cross_entropy(logits, y_true)


def _per_sample_objective(
    logits: torch.Tensor,
    y_true: torch.Tensor,
    *,
    targeted: bool,
    target_class: int,
) -> torch.Tensor:
    if targeted:
        return -F.cross_entropy(
            logits,
            _target_tensor(y_true, target_class),
            reduction="none",
        )
    return F.cross_entropy(logits, y_true, reduction="none")


def _attack_success_mask(
    logits: torch.Tensor,
    y_true: torch.Tensor,
    *,
    targeted: bool,
    target_class: int,
) -> torch.Tensor:
    pred = torch.argmax(logits, dim=1)
    if targeted:
        return pred == int(target_class)
    return pred != y_true


def _project_z_linf(
    z_candidate: torch.Tensor,
    z_orig: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    return torch.max(torch.min(z_candidate, z_orig + float(epsilon)), z_orig - float(epsilon))


def _normalise_z_initializers(
    *,
    z_orig: torch.Tensor,
    epsilon: float,
    random_start: bool,
    num_restarts: int,
    z_initializers: torch.Tensor | None,
    project: bool,
) -> list[torch.Tensor]:
    restarts = max(1, int(num_restarts))

    if z_initializers is not None:
        z_init = z_initializers.detach().to(device=z_orig.device, dtype=z_orig.dtype)
        if z_init.ndim == 2:
            z_init = z_init.unsqueeze(0)
        if z_init.ndim != 3:
            raise ValueError(
                "z_initializers must have shape (batch, latent_dim) or "
                f"(num_restarts, batch, latent_dim), got {tuple(z_init.shape)}"
            )
        if z_init.shape[1:] != z_orig.shape:
            raise ValueError(
                "z_initializers batch/latent shape mismatch: "
                f"{tuple(z_init.shape[1:])} vs {tuple(z_orig.shape)}"
            )

        limit = z_init.shape[0] if restarts <= 1 else min(restarts, z_init.shape[0])
        candidates = [z_init[i].clone() for i in range(limit)]
    else:
        candidates = []
        for restart_idx in range(restarts):
            z_start = z_orig.clone()
            if random_start or restart_idx > 0:
                z_start = z_start + torch.empty_like(z_start).uniform_(
                    -float(epsilon),
                    float(epsilon),
                )
            candidates.append(z_start)

    if project:
        candidates = [_project_z_linf(z_start, z_orig, epsilon) for z_start in candidates]
    return candidates


def latent_pgd_attack(
    *,
    vae: torch.nn.Module,
    classifier: torch.nn.Module,
    mask: PerturbationMask,
    x_original: torch.Tensor,
    y_true: torch.Tensor,
    scaler: Any,
    epsilon: float,
    alpha: float,
    num_steps: int,
    random_start: bool,
    device: str,
    targeted: bool = False,
    target_class: int = 0,
    num_restarts: int = 1,
    restart_strategy: str = "encoded",
    z_initializers: torch.Tensor | None = None,
    adaptive_pgd: bool = True,
    checkpoint_interval: int = 10,
    rho: float = 0.75,
    min_alpha: float = 1e-4,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, MetadataValue]]:
    x_original = x_original.to(device=device, dtype=torch.float32)
    y_true = y_true.to(device=device, dtype=torch.long)

    vae = vae.to(device)
    vae.eval()
    classifier = classifier.to(device)
    classifier.eval()

    with torch.no_grad():
        z_orig, _ = vae.encode(x_original)
        x_orig_dec_soft, _ = vae.decode_to_39(z_orig, scaler, mode="soft")
        x_orig_dec_hard, _ = vae.decode_to_39(z_orig, scaler, mode="hard")

    if float(epsilon) <= 0.0:
        x_passthrough = reimpose_protocol_features(mask.apply(x_original, x_original), x_original)
        metadata: dict[str, MetadataValue] = {
            "loss_final": 0.0,
            "num_steps": int(num_steps),
            "random_start": bool(random_start),
            "epsilon": float(epsilon),
            "alpha": float(alpha),
            "targeted": bool(targeted),
            "target_class": int(target_class),
            "num_restarts": int(num_restarts),
            "restart_strategy": str(restart_strategy),
            "adaptive_pgd": bool(adaptive_pgd),
            "checkpoint_interval": int(checkpoint_interval),
            "rho": float(rho),
            "min_alpha": float(min_alpha),
            "alpha_final": float(alpha),
            "zero_budget_passthrough": True,
            "anchored_decoder_residual": True,
            "z_orig": z_orig.detach(),
            "z_adv": z_orig.detach(),
            "selected_restart": torch.zeros(x_original.shape[0], dtype=torch.long, device=device),
            "best_success_mask": torch.zeros(x_original.shape[0], dtype=torch.bool, device=device),
        }
        return x_passthrough.detach(), z_orig.detach(), metadata

    z_starts = _normalise_z_initializers(
        z_orig=z_orig,
        epsilon=float(epsilon),
        random_start=bool(random_start),
        num_restarts=int(num_restarts),
        z_initializers=z_initializers,
        project=True,
    )

    best_x: torch.Tensor | None = None
    best_z: torch.Tensor | None = None
    best_success = torch.zeros(x_original.shape[0], dtype=torch.bool, device=device)
    best_l2 = torch.full((x_original.shape[0],), float("inf"), device=device)
    best_objective = torch.full((x_original.shape[0],), float("-inf"), device=device)
    selected_restart = torch.zeros(x_original.shape[0], dtype=torch.long, device=device)
    restart_success_counts: list[int] = []
    restart_final_alphas: list[float] = []
    loss_value = 0.0

    for restart_idx, z_start in enumerate(z_starts):
        z_adv = z_start.clone()
        step_alpha = float(alpha)
        best_loss_seen = float("-inf")
        steps_since_improvement = 0
        effective_checkpoint_interval = max(1, int(checkpoint_interval))
        improvement_patience = max(1, int(float(rho) * effective_checkpoint_interval))

        for step_idx in range(num_steps):
            z_adv = z_adv.detach().requires_grad_(True)
            x_soft_dec, _ = vae.decode_to_39(z_adv, scaler, mode="soft")
            x_soft = apply_decoder_residual(
                x_decoded=x_soft_dec,
                x_anchor_decoded=x_orig_dec_soft,
                x_original=x_original,
                mask=mask,
            )

            logits = classifier_logits(classifier, x_soft, device=device)
            loss = _objective_loss(
                logits,
                y_true,
                targeted=bool(targeted),
                target_class=int(target_class),
            )
            grad = torch.autograd.grad(loss, z_adv, retain_graph=False, create_graph=False)[0]
            loss_scalar = float(loss.item())

            if loss_scalar > best_loss_seen + 1e-7:
                best_loss_seen = loss_scalar
                steps_since_improvement = 0
            else:
                steps_since_improvement += 1

            with torch.no_grad():
                z_adv = z_adv + step_alpha * grad.sign()
                z_adv = _project_z_linf(z_adv, z_orig, float(epsilon))
                loss_value = loss_scalar

                if (
                    bool(adaptive_pgd)
                    and (step_idx + 1) % effective_checkpoint_interval == 0
                    and steps_since_improvement > improvement_patience
                ):
                    step_alpha = max(step_alpha * 0.5, float(min_alpha))
                    steps_since_improvement = 0

        with torch.no_grad():
            x_adv_dec_final, _ = vae.decode_to_39(z_adv, scaler, mode="hard")
            x_adv_final = apply_decoder_residual(
                x_decoded=x_adv_dec_final,
                x_anchor_decoded=x_orig_dec_hard,
                x_original=x_original,
                mask=mask,
            )
            logits_final = classifier_logits(classifier, x_adv_final, device=device)
            success = _attack_success_mask(
                logits_final,
                y_true,
                targeted=bool(targeted),
                target_class=int(target_class),
            )
            objective = _per_sample_objective(
                logits_final,
                y_true,
                targeted=bool(targeted),
                target_class=int(target_class),
            )
            input_l2 = torch.linalg.norm(x_adv_final - x_original, dim=1)
            restart_success_counts.append(int(success.float().sum().item()))
            restart_final_alphas.append(float(step_alpha))

            if best_x is None or best_z is None:
                best_x = x_adv_final.detach().clone()
                best_z = z_adv.detach().clone()
                best_success = success.detach().clone()
                best_l2 = input_l2.detach().clone()
                best_objective = objective.detach().clone()
                selected_restart = torch.full(
                    (x_original.shape[0],),
                    int(restart_idx),
                    dtype=torch.long,
                    device=device,
                )
                continue

            improved = success & (~best_success)
            improved |= (success == best_success) & success & (input_l2 < best_l2)
            improved |= (success == best_success) & (~success) & (objective > best_objective)

            if improved.any():
                best_x[improved] = x_adv_final.detach()[improved]
                best_z[improved] = z_adv.detach()[improved]
                best_success[improved] = success.detach()[improved]
                best_l2[improved] = input_l2.detach()[improved]
                best_objective[improved] = objective.detach()[improved]
                selected_restart[improved] = int(restart_idx)

    if best_x is None or best_z is None:
        raise RuntimeError("No PGD restarts were executed")

    metadata = {
        "loss_final": loss_value,
        "num_steps": int(num_steps),
        "random_start": bool(random_start),
        "epsilon": float(epsilon),
        "alpha": float(alpha),
        "targeted": bool(targeted),
        "target_class": int(target_class),
        "num_restarts": int(len(z_starts)),
        "restart_strategy": str(restart_strategy),
        "adaptive_pgd": bool(adaptive_pgd),
        "checkpoint_interval": int(checkpoint_interval),
        "rho": float(rho),
        "min_alpha": float(min_alpha),
        "alpha_final": float(min(restart_final_alphas)) if restart_final_alphas else float(alpha),
        "restart_final_alphas": restart_final_alphas,
        "zero_budget_passthrough": False,
        "anchored_decoder_residual": True,
        "z_orig": z_orig.detach(),
        "z_adv": best_z.detach(),
        "selected_restart": selected_restart.detach(),
        "best_success_mask": best_success.detach(),
        "best_objective": best_objective.detach(),
        "restart_success_counts": restart_success_counts,
    }
    return best_x.detach(), best_z.detach(), metadata
