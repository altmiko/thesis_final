from __future__ import annotations

import torch

from attack.latent_pgd import classifier_logits


def input_pgd_attack(
    *,
    classifier: torch.nn.Module,
    x_original: torch.Tensor,
    y_true: torch.Tensor,
    epsilon: float,
    alpha: float,
    num_steps: int,
    random_start: bool,
    device: str,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | float | int | bool]]:
    x_original = x_original.to(device=device, dtype=torch.float32)
    y_true = y_true.to(device=device, dtype=torch.long)

    classifier = classifier.to(device)
    classifier.eval()

    if float(epsilon) <= 0.0:
        metadata: dict[str, torch.Tensor | float | int | bool] = {
            "loss_final": 0.0,
            "num_steps": int(num_steps),
            "random_start": bool(random_start),
            "epsilon": float(epsilon),
            "alpha": float(alpha),
            "zero_budget_passthrough": True,
            "x_orig": x_original.detach(),
            "x_adv": x_original.detach(),
        }
        return x_original.detach(), metadata

    x_adv = x_original.clone()
    if random_start:
        x_adv = x_adv + torch.empty_like(x_adv).uniform_(-epsilon, epsilon)
        x_adv = torch.max(torch.min(x_adv, x_original + epsilon), x_original - epsilon)

    loss_value = 0.0
    for _ in range(num_steps):
        x_adv = x_adv.detach().requires_grad_(True)
        logits = classifier_logits(classifier, x_adv, device=device)
        loss = torch.nn.functional.cross_entropy(logits, y_true)
        grad = torch.autograd.grad(loss, x_adv, retain_graph=False, create_graph=False)[0]

        with torch.no_grad():
            x_adv = x_adv + alpha * grad.sign()
            x_adv = torch.max(torch.min(x_adv, x_original + epsilon), x_original - epsilon)
            loss_value = float(loss.item())

    metadata = {
        "loss_final": loss_value,
        "num_steps": int(num_steps),
        "random_start": bool(random_start),
        "epsilon": float(epsilon),
        "alpha": float(alpha),
        "zero_budget_passthrough": False,
        "x_orig": x_original.detach(),
        "x_adv": x_adv.detach(),
    }
    return x_adv.detach(), metadata


def input_cw_attack(
    *,
    classifier: torch.nn.Module,
    x_original: torch.Tensor,
    y_true: torch.Tensor,
    lambda_conf: float,
    kappa: float,
    num_iterations: int,
    learning_rate: float,
    convergence_threshold: float,
    device: str,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | float | int | bool]]:
    x_original = x_original.to(device=device, dtype=torch.float32)
    y_true = y_true.to(device=device, dtype=torch.long)

    classifier = classifier.to(device)
    classifier.eval()

    if float(lambda_conf) <= 0.0:
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
            "x_orig": x_original.detach(),
            "x_adv": x_original.detach(),
            "best_success_mask": torch.zeros(x_original.shape[0], dtype=torch.bool, device=device),
        }
        return x_original.detach(), metadata

    delta = torch.zeros_like(x_original, requires_grad=True)
    optimizer = torch.optim.Adam([delta], lr=learning_rate)

    best_delta = torch.zeros_like(x_original)
    best_delta_l2 = torch.full((x_original.shape[0],), float("inf"), device=device)
    best_success_mask = torch.zeros(x_original.shape[0], dtype=torch.bool, device=device)

    prev_delta = delta.detach().clone()
    converged_early = False
    loss_value = 0.0
    iterations_run = 0

    for iteration in range(num_iterations):
        optimizer.zero_grad(set_to_none=True)

        x_current = x_original + delta
        logits = classifier_logits(classifier, x_current, device=device)
        true_logits = logits.gather(1, y_true.unsqueeze(1)).squeeze(1)
        masked_logits = logits.clone()
        masked_logits.scatter_(1, y_true.unsqueeze(1), float("-inf"))
        other_logits = masked_logits.max(dim=1).values
        margin = true_logits - other_logits

        conf_term = torch.clamp(margin + float(kappa), min=0.0)
        delta_l2_sq = delta.reshape(delta.shape[0], -1).pow(2).sum(dim=1)
        loss_per_sample = float(lambda_conf) * conf_term + delta_l2_sq
        loss = loss_per_sample.mean()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            x_post = x_original + delta.detach()
            logits_post = classifier_logits(classifier, x_post, device=device)
            true_logits_post = logits_post.gather(1, y_true.unsqueeze(1)).squeeze(1)
            masked_logits_post = logits_post.clone()
            masked_logits_post.scatter_(1, y_true.unsqueeze(1), float("-inf"))
            other_logits_post = masked_logits_post.max(dim=1).values
            success_mask = (true_logits_post - other_logits_post) <= 0.0

            delta_l2 = torch.linalg.norm(delta.detach().reshape(delta.shape[0], -1), dim=1)
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
        chosen_delta = torch.where(best_success_mask.unsqueeze(1), best_delta, final_delta)
        x_adv_final = x_original + chosen_delta

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
        "x_orig": x_original.detach(),
        "x_adv": x_adv_final.detach(),
        "best_success_mask": best_success_mask.detach(),
    }
    return x_adv_final.detach(), metadata
