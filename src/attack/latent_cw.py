from __future__ import annotations

from typing import Any

import torch

from attack.latent_infra import (
    PerturbationMask,
    apply_decoder_residual,
    reimpose_protocol_features,
)
from attack.latent_pgd import (
    MetadataValue,
    _attack_success_mask,
    _normalise_z_initializers,
    _per_sample_objective,
    classifier_logits,
    target_logit_margin,
)


def latent_cw_attack(
    *,
    vae: torch.nn.Module,
    classifier: torch.nn.Module,
    mask: PerturbationMask,
    x_original: torch.Tensor,
    y_true: torch.Tensor,
    scaler: Any,
    lambda_conf: float,
    kappa: float,
    num_iterations: int,
    learning_rate: float,
    convergence_threshold: float,
    device: str,
    targeted: bool = False,
    target_class: int = 0,
    num_restarts: int = 1,
    restart_strategy: str = "encoded",
    z_initializers: torch.Tensor | None = None,
    restart_jitter: float = 0.05,
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

    if float(lambda_conf) <= 0.0:
        x_passthrough = reimpose_protocol_features(mask.apply(x_original, x_original), x_original)
        metadata: dict[str, MetadataValue] = {
            "loss_final": 0.0,
            "num_iterations": int(num_iterations),
            "learning_rate": float(learning_rate),
            "lambda_conf": float(lambda_conf),
            "kappa": float(kappa),
            "convergence_threshold": float(convergence_threshold),
            "targeted": bool(targeted),
            "target_class": int(target_class),
            "num_restarts": int(num_restarts),
            "restart_strategy": str(restart_strategy),
            "restart_jitter": float(restart_jitter),
            "zero_budget_passthrough": True,
            "anchored_decoder_residual": True,
            "iterations_run": 0,
            "converged_early": False,
            "z_orig": z_orig.detach(),
            "z_adv": z_orig.detach(),
            "best_success_mask": torch.zeros(x_original.shape[0], dtype=torch.bool, device=device),
            "selected_restart": torch.zeros(x_original.shape[0], dtype=torch.long, device=device),
        }
        return x_passthrough.detach(), z_orig.detach(), metadata

    # A non-zero jitter is required for the internal multi-restart loop to do
    # anything when no explicit z_initializers are supplied: with epsilon=0 every
    # restart would start at exactly z_orig and Adam is deterministic, so all
    # restarts would be identical (fix.md #14). When z_initializers are provided
    # (e.g. GMM seeds) this jitter is ignored by _normalise_z_initializers.
    z_starts = _normalise_z_initializers(
        z_orig=z_orig,
        epsilon=float(restart_jitter),
        random_start=False,
        num_restarts=int(num_restarts),
        z_initializers=z_initializers,
        project=False,
    )

    best_x: torch.Tensor | None = None
    best_z: torch.Tensor | None = None
    best_success = torch.zeros(x_original.shape[0], dtype=torch.bool, device=device)
    best_l2 = torch.full((x_original.shape[0],), float("inf"), device=device)
    best_objective = torch.full((x_original.shape[0],), float("-inf"), device=device)
    selected_restart = torch.zeros(x_original.shape[0], dtype=torch.long, device=device)
    restart_success_counts: list[int] = []
    loss_value = 0.0
    max_iterations_run = 0
    any_converged_early = False

    for restart_idx, z_start in enumerate(z_starts):
        delta = (z_start - z_orig).detach().clone().requires_grad_(True)
        optimizer = torch.optim.Adam([delta], lr=learning_rate)

        best_delta = delta.detach().clone()
        best_delta_l2 = torch.full((z_orig.shape[0],), float("inf"), device=device)
        best_restart_success = torch.zeros(z_orig.shape[0], dtype=torch.bool, device=device)

        prev_delta = delta.detach().clone()
        converged_early = False
        iterations_run = 0

        for iteration in range(num_iterations):
            optimizer.zero_grad(set_to_none=True)

            z_current = z_orig + delta
            x_soft_dec, _ = vae.decode_to_39(z_current, scaler, mode="soft")
            x_soft = apply_decoder_residual(
                x_decoded=x_soft_dec,
                x_anchor_decoded=x_orig_dec_soft,
                x_original=x_original,
                mask=mask,
            )

            logits = classifier_logits(classifier, x_soft, device=device)
            true_logits = logits.gather(1, y_true.unsqueeze(1)).squeeze(1)

            if targeted:
                # Targeted CW must drive the target logit above the best
                # competing class (max over all others excluding the target),
                # not merely above the true class. Reuse latent_pgd's cw-margin
                # formulation so the objective matches the argmax==target success
                # test (fix.md #13).
                _t_margin, target_logits, max_other_logits = target_logit_margin(
                    logits, target_class=int(target_class)
                )
                margin = max_other_logits - target_logits
            else:
                masked_logits = logits.clone()
                masked_logits.scatter_(1, y_true.unsqueeze(1), float("-inf"))
                other_logits = masked_logits.max(dim=1).values
                margin = true_logits - other_logits

            conf_term = torch.clamp(margin + float(kappa), min=0.0)
            delta_l2_sq = delta.pow(2).sum(dim=1)
            loss_per_sample = float(lambda_conf) * conf_term + delta_l2_sq
            loss = loss_per_sample.mean()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                z_post = z_orig + delta.detach()
                # Track the best per-restart delta using the same hard-projection
                # decode and argmax-based success criterion as the final
                # cross-restart selector (fix.md #13, #16), so the low-L2 delta
                # chosen inside a restart is consistent with the reported success
                # (and correct for the targeted case).
                x_post_dec, _ = vae.decode_to_39(z_post, scaler, mode="hard")
                x_post = apply_decoder_residual(
                    x_decoded=x_post_dec,
                    x_anchor_decoded=x_orig_dec_hard,
                    x_original=x_original,
                    mask=mask,
                )
                logits_post = classifier_logits(classifier, x_post, device=device)
                success_mask = _attack_success_mask(
                    logits_post,
                    y_true,
                    targeted=bool(targeted),
                    target_class=int(target_class),
                )

                delta_l2 = torch.linalg.norm(delta.detach(), dim=1)
                improved = success_mask & ((~best_restart_success) | (delta_l2 < best_delta_l2))
                if improved.any():
                    best_delta[improved] = delta.detach()[improved]
                    best_delta_l2[improved] = delta_l2[improved]
                    best_restart_success[improved] = True

                delta_shift = torch.linalg.norm(
                    (delta.detach() - prev_delta).reshape(delta.shape[0], -1),
                    dim=1,
                )
                prev_delta = delta.detach().clone()
                loss_value = float(loss.item())
                iterations_run = iteration + 1

                if torch.max(delta_shift).item() < float(convergence_threshold):
                    converged_early = True
                    break

        with torch.no_grad():
            final_delta = delta.detach()
            chosen_delta = torch.where(best_restart_success.unsqueeze(1), best_delta, final_delta)
            z_adv_final = z_orig + chosen_delta
            x_adv_dec_final, _ = vae.decode_to_39(z_adv_final, scaler, mode="hard")
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

            if best_x is None or best_z is None:
                best_x = x_adv_final.detach().clone()
                best_z = z_adv_final.detach().clone()
                best_success = success.detach().clone()
                best_l2 = input_l2.detach().clone()
                best_objective = objective.detach().clone()
                selected_restart = torch.full(
                    (x_original.shape[0],),
                    int(restart_idx),
                    dtype=torch.long,
                    device=device,
                )
            else:
                improved = success & (~best_success)
                improved |= (success == best_success) & success & (input_l2 < best_l2)
                improved |= (success == best_success) & (~success) & (objective > best_objective)

                if improved.any():
                    best_x[improved] = x_adv_final.detach()[improved]
                    best_z[improved] = z_adv_final.detach()[improved]
                    best_success[improved] = success.detach()[improved]
                    best_l2[improved] = input_l2.detach()[improved]
                    best_objective[improved] = objective.detach()[improved]
                    selected_restart[improved] = int(restart_idx)

        max_iterations_run = max(max_iterations_run, int(iterations_run))
        any_converged_early = any_converged_early or bool(converged_early)

    if best_x is None or best_z is None:
        raise RuntimeError("No C&W restarts were executed")

    metadata = {
        "loss_final": loss_value,
        "num_iterations": int(num_iterations),
        "learning_rate": float(learning_rate),
        "lambda_conf": float(lambda_conf),
        "kappa": float(kappa),
        "convergence_threshold": float(convergence_threshold),
        "targeted": bool(targeted),
        "target_class": int(target_class),
        "num_restarts": int(len(z_starts)),
        "restart_strategy": str(restart_strategy),
        "restart_jitter": float(restart_jitter),
        "zero_budget_passthrough": False,
        "anchored_decoder_residual": True,
        "iterations_run": int(max_iterations_run),
        "converged_early": bool(any_converged_early),
        "z_orig": z_orig.detach(),
        "z_adv": best_z.detach(),
        "best_success_mask": best_success.detach(),
        "best_objective": best_objective.detach(),
        "selected_restart": selected_restart.detach(),
        "restart_success_counts": restart_success_counts,
    }
    return best_x.detach(), best_z.detach(), metadata
