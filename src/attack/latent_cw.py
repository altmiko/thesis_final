from __future__ import annotations

from typing import Any

import torch

from attack.latent_infra import PerturbationMask, reimpose_protocol_features
from attack.latent_pgd import classifier_logits


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
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor | float | int | bool]]:
    x_original = x_original.to(device=device, dtype=torch.float32)
    y_true = y_true.to(device=device, dtype=torch.long)

    vae = vae.to(device)
    vae.eval()
    classifier = classifier.to(device)
    classifier.eval()

    with torch.no_grad():
        z_orig, _ = vae.encode(x_original)

    if float(lambda_conf) <= 0.0:
        x_passthrough = reimpose_protocol_features(mask.apply(x_original, x_original), x_original)
        metadata: dict[str, torch.Tensor | float | int | bool] = {
            "loss_final": 0.0,
            "num_iterations": int(num_iterations),
            "learning_rate": float(learning_rate),
            "lambda_conf": float(lambda_conf),
            "kappa": float(kappa),
            "convergence_threshold": float(convergence_threshold),
            "zero_budget_passthrough": True,
            "iterations_run": 0,
            "converged_early": False,
            "z_orig": z_orig.detach(),
            "z_adv": z_orig.detach(),
            "best_success_mask": torch.zeros(x_original.shape[0], dtype=torch.bool, device=device),
        }
        return x_passthrough.detach(), z_orig.detach(), metadata

    delta = torch.zeros_like(z_orig, requires_grad=True)
    optimizer = torch.optim.Adam([delta], lr=learning_rate)

    best_delta = torch.zeros_like(z_orig)
    best_delta_l2 = torch.full((z_orig.shape[0],), float("inf"), device=device)
    best_success_mask = torch.zeros(z_orig.shape[0], dtype=torch.bool, device=device)

    prev_delta = delta.detach().clone()
    converged_early = False
    loss_value = 0.0
    iterations_run = 0

    for iteration in range(num_iterations):
        optimizer.zero_grad(set_to_none=True)

        z_current = z_orig + delta
        x_soft, _ = vae.decode_to_39(z_current, scaler, mode="soft")
        x_soft = mask.apply(x_soft, x_original)
        x_soft = reimpose_protocol_features(x_soft, x_original)

        logits = classifier_logits(classifier, x_soft, device=device)
        true_logits = logits.gather(1, y_true.unsqueeze(1)).squeeze(1)
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
            x_post, _ = vae.decode_to_39(z_post, scaler, mode="soft")
            x_post = mask.apply(x_post, x_original)
            x_post = reimpose_protocol_features(x_post, x_original)
            logits_post = classifier_logits(classifier, x_post, device=device)
            true_logits_post = logits_post.gather(1, y_true.unsqueeze(1)).squeeze(1)
            masked_logits_post = logits_post.clone()
            masked_logits_post.scatter_(1, y_true.unsqueeze(1), float("-inf"))
            other_logits_post = masked_logits_post.max(dim=1).values
            success_mask = (true_logits_post - other_logits_post) <= 0.0

            delta_l2 = torch.linalg.norm(delta.detach(), dim=1)
            improved = success_mask & (delta_l2 < best_delta_l2)
            if improved.any():
                best_delta[improved] = delta.detach()[improved]
                best_delta_l2[improved] = delta_l2[improved]
                best_success_mask[improved] = True

            delta_shift = torch.linalg.norm((delta.detach() - prev_delta).reshape(delta.shape[0], -1), dim=1)
            prev_delta = delta.detach().clone()
            loss_value = float(loss.item())
            iterations_run = iteration + 1

            if torch.max(delta_shift).item() < float(convergence_threshold):
                converged_early = True
                break

    with torch.no_grad():
        final_delta = delta.detach()
        final_success = best_success_mask
        chosen_delta = torch.where(final_success.unsqueeze(1), best_delta, final_delta)
        z_adv_final = z_orig + chosen_delta
        x_adv_final, _ = vae.decode_to_39(z_adv_final, scaler, mode="hard")
        x_adv_final = mask.apply(x_adv_final, x_original)
        x_adv_final = reimpose_protocol_features(x_adv_final, x_original)

    metadata = {
        "loss_final": loss_value,
        "num_iterations": int(num_iterations),
        "learning_rate": float(learning_rate),
        "lambda_conf": float(lambda_conf),
        "kappa": float(kappa),
        "convergence_threshold": float(convergence_threshold),
        "zero_budget_passthrough": False,
        "iterations_run": int(iterations_run),
        "converged_early": bool(converged_early),
        "z_orig": z_orig.detach(),
        "z_adv": z_adv_final.detach(),
        "best_success_mask": best_success_mask.detach(),
    }
    return x_adv_final.detach(), z_adv_final.detach(), metadata
