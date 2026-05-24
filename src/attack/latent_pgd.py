from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from attack.latent_infra import PerturbationMask, reimpose_protocol_features


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
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor | float | int | bool]]:
    x_original = x_original.to(device=device, dtype=torch.float32)
    y_true = y_true.to(device=device, dtype=torch.long)

    vae = vae.to(device)
    vae.eval()
    classifier = classifier.to(device)
    classifier.eval()

    with torch.no_grad():
        z_orig, _ = vae.encode(x_original)

    if float(epsilon) <= 0.0:
        x_passthrough = reimpose_protocol_features(mask.apply(x_original, x_original), x_original)
        metadata: dict[str, torch.Tensor | float | int | bool] = {
            "loss_final": 0.0,
            "num_steps": int(num_steps),
            "random_start": bool(random_start),
            "epsilon": float(epsilon),
            "alpha": float(alpha),
            "zero_budget_passthrough": True,
            "z_orig": z_orig.detach(),
            "z_adv": z_orig.detach(),
        }
        return x_passthrough.detach(), z_orig.detach(), metadata

    z_adv = z_orig.clone()
    if random_start:
        z_adv = z_adv + torch.empty_like(z_adv).uniform_(-epsilon, epsilon)
        z_adv = torch.max(torch.min(z_adv, z_orig + epsilon), z_orig - epsilon)

    loss_value = 0.0
    for _ in range(num_steps):
        z_adv = z_adv.detach().requires_grad_(True)
        x_soft, _ = vae.decode_to_39(z_adv, scaler, mode="soft")
        x_soft = mask.apply(x_soft, x_original)
        x_soft = reimpose_protocol_features(x_soft, x_original)

        logits = classifier_logits(classifier, x_soft, device=device)
        loss = F.cross_entropy(logits, y_true)
        grad = torch.autograd.grad(loss, z_adv, retain_graph=False, create_graph=False)[0]

        with torch.no_grad():
            z_adv = z_adv + alpha * grad.sign()
            z_adv = torch.max(torch.min(z_adv, z_orig + epsilon), z_orig - epsilon)
            loss_value = float(loss.item())

    with torch.no_grad():
        x_adv_final, _ = vae.decode_to_39(z_adv, scaler, mode="hard")
        x_adv_final = mask.apply(x_adv_final, x_original)
        x_adv_final = reimpose_protocol_features(x_adv_final, x_original)

    metadata = {
        "loss_final": loss_value,
        "num_steps": int(num_steps),
        "random_start": bool(random_start),
        "epsilon": float(epsilon),
        "alpha": float(alpha),
        "zero_budget_passthrough": False,
        "z_orig": z_orig.detach(),
        "z_adv": z_adv.detach(),
    }
    return x_adv_final.detach(), z_adv.detach(), metadata
