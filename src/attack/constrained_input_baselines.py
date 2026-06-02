from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from attack.latent_infra import PerturbationMask, reimpose_protocol_features
from attack.latent_pgd import classifier_logits
from preprocessing.feature_groups import FEATURE_NAMES
from vae.schema import get_partition


_PARTITION = get_partition()
_CONTINUOUS_IDX = _PARTITION["continuous_idx"]
_BINARY_IDX = _PARTITION["independent_binary_idx"] + _PARTITION["derived_binary_idx"]


def _target_tensor(y_true: torch.Tensor, target_class: int) -> torch.Tensor:
    return torch.full_like(y_true, int(target_class), dtype=torch.long)


def _validate_target_class(logits: torch.Tensor, target_class: int) -> int:
    target = int(target_class)
    if target < 0 or target >= int(logits.shape[1]):
        raise ValueError(f"target_class={target} is outside logits width {logits.shape[1]}")
    return target


def _target_logit_margin(
    logits: torch.Tensor,
    *,
    target_class: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    target = _validate_target_class(logits, target_class)
    target_logits = logits[:, target]
    masked_logits = logits.clone()
    masked_logits[:, target] = float("-inf")
    max_other_logits = masked_logits.max(dim=1).values
    margin = target_logits - max_other_logits
    return margin, target_logits, max_other_logits


def _success_mask(
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


class VAEConstraintProjection:
    """VAE structured-decoder constraints applied directly to scaled inputs."""

    def __init__(
        self,
        scaler: Any,
        mask: PerturbationMask,
        *,
        enable_physics: bool = True,
        structured_std_floor: float = 0.0,
        device: str = "cpu",
    ) -> None:
        center = np.asarray(scaler.center_, dtype=np.float32)
        scale = np.asarray(scaler.scale_, dtype=np.float32)
        if center.shape != (len(FEATURE_NAMES),) or scale.shape != (len(FEATURE_NAMES),):
            raise ValueError(
                f"Expected a {len(FEATURE_NAMES)}-feature scaler, got "
                f"center={center.shape}, scale={scale.shape}"
            )
        if np.any(scale == 0.0):
            raise ValueError("RobustScaler scale_ contains zeros; affine unscale is undefined")

        self.mask = mask
        self.enable_physics = bool(enable_physics)
        self.structured_std_floor = float(structured_std_floor)
        self.device = str(device)

        self.center = torch.tensor(center, dtype=torch.float32, device=device)
        self.scale = torch.tensor(scale, dtype=torch.float32, device=device)
        self.continuous_idx = torch.tensor(_CONTINUOUS_IDX, dtype=torch.long, device=device)
        self.binary_idx = torch.tensor(_BINARY_IDX, dtype=torch.long, device=device)

        self.ttl_idx = FEATURE_NAMES.index("Time_To_Live")
        self.tot_sum_idx = FEATURE_NAMES.index("Tot sum")
        self.min_idx = FEATURE_NAMES.index("Min")
        self.max_idx = FEATURE_NAMES.index("Max")
        self.avg_idx = FEATURE_NAMES.index("AVG")
        self.std_idx = FEATURE_NAMES.index("Std")
        self.tot_size_idx = FEATURE_NAMES.index("Tot size")
        self.number_idx = FEATURE_NAMES.index("Number")
        self.variance_idx = FEATURE_NAMES.index("Variance")

    def to(self, device: str) -> "VAEConstraintProjection":
        self.device = str(device)
        self.center = self.center.to(device)
        self.scale = self.scale.to(device)
        self.continuous_idx = self.continuous_idx.to(device)
        self.binary_idx = self.binary_idx.to(device)
        return self

    def _unscale(self, x_scaled: torch.Tensor) -> torch.Tensor:
        return x_scaled * self.scale.to(x_scaled.device).unsqueeze(0) + self.center.to(x_scaled.device).unsqueeze(0)

    def _rescale(self, x_raw: torch.Tensor) -> torch.Tensor:
        return (x_raw - self.center.to(x_raw.device).unsqueeze(0)) / self.scale.to(x_raw.device).unsqueeze(0)

    def _structure_raw(self, x_raw: torch.Tensor, mode: str = "soft") -> torch.Tensor:
        if mode not in {"soft", "hard"}:
            raise ValueError(f"mode must be 'soft' or 'hard', got {mode!r}")

        structured = x_raw.clone()
        structured[:, self.continuous_idx.to(x_raw.device)] = x_raw[
            :,
            self.continuous_idx.to(x_raw.device),
        ].clamp_min(0.0)

        min_base = F.softplus(x_raw[:, self.min_idx])
        avg_gap = F.softplus(x_raw[:, self.avg_idx] - x_raw[:, self.min_idx])
        max_gap = F.softplus(x_raw[:, self.max_idx] - x_raw[:, self.avg_idx])
        avg_val = min_base + avg_gap
        max_val = avg_val + max_gap

        number_pos = 1.0 + F.softplus(x_raw[:, self.number_idx])
        number_val = number_pos + (torch.round(number_pos) - number_pos).detach()
        ttl_val = torch.sigmoid(x_raw[:, self.ttl_idx] / 32.0) * 255.0

        std_candidate = self.structured_std_floor + F.softplus(x_raw[:, self.std_idx])
        if self.enable_physics:
            std_cap = 0.5 * (max_val - min_base)
            std_val = torch.minimum(std_candidate, std_cap)
        else:
            std_val = std_candidate

        if self.enable_physics:
            singleton_mask = number_val <= 1.0
            singleton_size = F.softplus(x_raw[:, self.avg_idx])
            min_base = torch.where(singleton_mask, singleton_size, min_base)
            avg_val = torch.where(singleton_mask, singleton_size, avg_val)
            max_val = torch.where(singleton_mask, singleton_size, max_val)
            std_val = torch.where(singleton_mask, torch.zeros_like(std_val), std_val)

        variance_val = std_val.square()

        if self.enable_physics:
            tot_size_val = avg_val
            tot_sum_val = number_val * avg_val
        else:
            tot_size_val = structured[:, self.tot_size_idx]
            tot_sum_val = structured[:, self.tot_sum_idx]

        structured[:, self.ttl_idx] = ttl_val
        structured[:, self.tot_sum_idx] = tot_sum_val
        structured[:, self.min_idx] = min_base
        structured[:, self.avg_idx] = avg_val
        structured[:, self.max_idx] = max_val
        structured[:, self.std_idx] = std_val
        structured[:, self.tot_size_idx] = tot_size_val
        structured[:, self.variance_idx] = variance_val
        structured[:, self.number_idx] = number_val

        binary_raw = x_raw[:, self.binary_idx.to(x_raw.device)]
        binary_soft = torch.sigmoid(binary_raw)
        if mode == "hard":
            binary_hard = torch.round(binary_soft)
            structured[:, self.binary_idx.to(x_raw.device)] = (
                binary_soft + (binary_hard - binary_soft).detach()
            )
        else:
            structured[:, self.binary_idx.to(x_raw.device)] = binary_soft
        return structured

    def project(
        self,
        x_scaled: torch.Tensor,
        x_original_scaled: torch.Tensor,
        mode: str = "soft",
    ) -> torch.Tensor:
        if x_scaled.shape != x_original_scaled.shape:
            raise ValueError(f"Shape mismatch: {x_scaled.shape} vs {x_original_scaled.shape}")

        x_raw = self._unscale(x_scaled)
        x_structured = self._rescale(self._structure_raw(x_raw, mode=mode))
        x_masked = self.mask.apply(x_structured, x_original_scaled)
        return reimpose_protocol_features(x_masked, x_original_scaled)


def constrained_input_pgd_attack(
    *,
    classifier: torch.nn.Module,
    projection: VAEConstraintProjection,
    x_original: torch.Tensor,
    y_true: torch.Tensor,
    epsilon: float,
    alpha: float,
    num_steps: int,
    random_start: bool,
    device: str,
    targeted: bool = False,
    target_class: int = 0,
) -> tuple[torch.Tensor, dict[str, Any]]:
    x_original = x_original.to(device=device, dtype=torch.float32)
    y_true = y_true.to(device=device, dtype=torch.long)
    classifier = classifier.to(device)
    classifier.eval()
    projection = projection.to(device)

    if float(epsilon) <= 0.0:
        x_pass = projection.project(x_original, x_original, mode="hard")
        return x_pass.detach(), {
            "loss_final": 0.0,
            "num_steps": int(num_steps),
            "random_start": bool(random_start),
            "epsilon": float(epsilon),
            "alpha": float(alpha),
            "targeted": bool(targeted),
            "target_class": int(target_class),
            "zero_budget_passthrough": True,
            "constraint_projection": True,
            "x_orig": x_original.detach(),
            "x_adv": x_pass.detach(),
        }

    x_adv = x_original.clone()
    if random_start:
        x_adv = x_adv + torch.empty_like(x_adv).uniform_(-float(epsilon), float(epsilon))
        x_adv = torch.max(torch.min(x_adv, x_original + float(epsilon)), x_original - float(epsilon))

    loss_value = 0.0
    for _ in range(int(num_steps)):
        x_adv = x_adv.detach().requires_grad_(True)
        x_projected = projection.project(x_adv, x_original, mode="soft")
        logits = classifier_logits(classifier, x_projected, device=device)
        if targeted:
            loss = -F.cross_entropy(logits, _target_tensor(y_true, target_class))
        else:
            loss = F.cross_entropy(logits, y_true)
        grad = torch.autograd.grad(loss, x_adv, retain_graph=False, create_graph=False)[0]

        with torch.no_grad():
            x_adv = x_adv + float(alpha) * grad.sign()
            x_adv = torch.max(
                torch.min(x_adv, x_original + float(epsilon)),
                x_original - float(epsilon),
            )
            loss_value = float(loss.item())

    x_adv_final = projection.project(x_adv.detach(), x_original, mode="hard")
    return x_adv_final.detach(), {
        "loss_final": loss_value,
        "num_steps": int(num_steps),
        "random_start": bool(random_start),
        "epsilon": float(epsilon),
        "alpha": float(alpha),
        "targeted": bool(targeted),
        "target_class": int(target_class),
        "zero_budget_passthrough": False,
        "constraint_projection": True,
        "x_orig": x_original.detach(),
        "x_adv": x_adv_final.detach(),
    }


def constrained_input_cw_attack(
    *,
    classifier: torch.nn.Module,
    projection: VAEConstraintProjection,
    x_original: torch.Tensor,
    y_true: torch.Tensor,
    lambda_conf: float,
    kappa: float,
    num_iterations: int,
    learning_rate: float,
    convergence_threshold: float,
    device: str,
    targeted: bool = False,
    target_class: int = 0,
) -> tuple[torch.Tensor, dict[str, Any]]:
    x_original = x_original.to(device=device, dtype=torch.float32)
    y_true = y_true.to(device=device, dtype=torch.long)
    classifier = classifier.to(device)
    classifier.eval()
    projection = projection.to(device)

    if float(lambda_conf) <= 0.0:
        x_pass = projection.project(x_original, x_original, mode="hard")
        return x_pass.detach(), {
            "loss_final": 0.0,
            "num_iterations": int(num_iterations),
            "learning_rate": float(learning_rate),
            "lambda_conf": float(lambda_conf),
            "kappa": float(kappa),
            "convergence_threshold": float(convergence_threshold),
            "targeted": bool(targeted),
            "target_class": int(target_class),
            "zero_budget_passthrough": True,
            "constraint_projection": True,
            "iterations_run": 0,
            "converged_early": False,
            "x_orig": x_original.detach(),
            "x_adv": x_pass.detach(),
            "best_success_mask": torch.zeros(x_original.shape[0], dtype=torch.bool, device=device),
        }

    delta = torch.zeros_like(x_original, requires_grad=True)
    optimizer = torch.optim.Adam([delta], lr=float(learning_rate))

    best_delta = torch.zeros_like(x_original)
    best_delta_l2 = torch.full((x_original.shape[0],), float("inf"), device=device)
    best_success_mask = torch.zeros(x_original.shape[0], dtype=torch.bool, device=device)

    prev_delta = delta.detach().clone()
    converged_early = False
    loss_value = 0.0
    iterations_run = 0

    for iteration in range(int(num_iterations)):
        optimizer.zero_grad(set_to_none=True)

        x_current = projection.project(x_original + delta, x_original, mode="soft")
        logits = classifier_logits(classifier, x_current, device=device)
        if targeted:
            _margin, target_logits, max_other_logits = _target_logit_margin(
                logits,
                target_class=int(target_class),
            )
            margin = max_other_logits - target_logits
        else:
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
            x_hard = projection.project(x_original + delta.detach(), x_original, mode="hard")
            logits_post = classifier_logits(classifier, x_hard, device=device)
            success_mask = _success_mask(
                logits_post,
                y_true,
                targeted=bool(targeted),
                target_class=int(target_class),
            )

            delta_l2 = torch.linalg.norm(delta.detach().reshape(delta.shape[0], -1), dim=1)
            improved = success_mask & (delta_l2 < best_delta_l2)
            if improved.any():
                best_delta[improved] = delta.detach()[improved]
                best_delta_l2[improved] = delta_l2[improved]
                best_success_mask[improved] = True

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
        chosen_delta = torch.where(best_success_mask.unsqueeze(1), best_delta, final_delta)
        x_adv_final = projection.project(x_original + chosen_delta, x_original, mode="hard")

    return x_adv_final.detach(), {
        "loss_final": loss_value,
        "num_iterations": int(num_iterations),
        "learning_rate": float(learning_rate),
        "lambda_conf": float(lambda_conf),
        "kappa": float(kappa),
        "convergence_threshold": float(convergence_threshold),
        "targeted": bool(targeted),
        "target_class": int(target_class),
        "zero_budget_passthrough": False,
        "constraint_projection": True,
        "iterations_run": int(iterations_run),
        "converged_early": bool(converged_early),
        "x_orig": x_original.detach(),
        "x_adv": x_adv_final.detach(),
        "best_success_mask": best_success_mask.detach(),
    }
