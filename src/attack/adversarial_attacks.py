"""
Reusable adversarial attack helpers for CICIoT2023 baselines.

This module intentionally runs unconstrained attacks for thesis analysis.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchattacks

try:
    from src.classifiers.models import get_model
except ModuleNotFoundError:
    from classifiers.models import get_model


class TabularFGSM(torchattacks.FGSM):
    """FGSM variant without image-space [0,1] clamping."""

    def forward(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self.targeted:
            target_labels = self.get_target_label(images, labels)

        loss = nn.CrossEntropyLoss()
        images.requires_grad = True
        outputs = self.get_logits(images)

        if self.targeted:
            cost = -loss(outputs, target_labels)
        else:
            cost = loss(outputs, labels)

        grad = torch.autograd.grad(cost, images, retain_graph=False, create_graph=False)[0]
        adv_images = (images + self.eps * grad.sign()).detach()
        return adv_images


class TabularPGD(torchattacks.PGD):
    """PGD variant that keeps epsilon-ball projection but removes [0,1] clamping."""

    def forward(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self.targeted:
            target_labels = self.get_target_label(images, labels)

        loss = nn.CrossEntropyLoss()
        adv_images = images.clone().detach()

        if self.random_start:
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(-self.eps, self.eps)

        for _ in range(self.steps):
            adv_images.requires_grad = True
            outputs = self.get_logits(adv_images)

            if self.targeted:
                cost = -loss(outputs, target_labels)
            else:
                cost = loss(outputs, labels)

            grad = torch.autograd.grad(cost, adv_images, retain_graph=False, create_graph=False)[0]

            adv_images = adv_images.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = (images + delta).detach()

        return adv_images


class TabularCW(torchattacks.CW):
    """CW variant optimized directly in feature space without tanh [0,1] restriction."""

    def forward(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self.targeted:
            target_labels = self.get_target_label(images, labels)

        adv_images = nn.Parameter(images.clone().detach())
        best_adv_images = images.clone().detach()
        best_l2 = 1e10 * torch.ones((len(images)), device=self.device)

        mse_loss = nn.MSELoss(reduction="none")
        flatten = nn.Flatten()
        optimizer = optim.Adam([adv_images], lr=self.lr)
        prev_cost = 1e10
        dim = len(images.shape)

        for step in range(self.steps):
            current_l2 = mse_loss(flatten(adv_images), flatten(images)).sum(dim=1)
            l2_loss = current_l2.sum()

            outputs = self.get_logits(adv_images)
            if self.targeted:
                f_loss = self.f(outputs, target_labels).sum()
            else:
                f_loss = self.f(outputs, labels).sum()

            cost = l2_loss + self.c * f_loss

            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

            with torch.no_grad():
                preds = torch.argmax(outputs, dim=1)
                if self.targeted:
                    condition = (preds == target_labels)
                else:
                    condition = (preds != labels)

                improved = condition & (best_l2 > current_l2.detach())
                best_l2 = torch.where(improved, current_l2.detach(), best_l2)

                mask = improved.view([-1] + [1] * (dim - 1))
                best_adv_images = torch.where(mask, adv_images.detach(), best_adv_images)

            if step % max(self.steps // 10, 1) == 0:
                if cost.item() > prev_cost:
                    return best_adv_images
                prev_cost = cost.item()

        return best_adv_images


def _resolve_device(device: str) -> torch.device:
    if device == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device)


def _infer_model_type(model_path: str) -> str:
    name = Path(model_path).name.lower()
    if "mlp" in name:
        return "mlp"
    if "dualpath" in name:
        return "dualpath"
    if "attn_serial" in name:
        return "attn_serial"
    if "serial" in name:
        return "serial"
    if "lstm" in name:
        return "lstm"
    if "cnn" in name:
        return "cnn"
    raise ValueError(
        f"Could not infer model type from filename '{name}'. "
        "Expected one of: mlp, cnn, lstm, serial, dualpath, attn_serial."
    )


def _extract_logits(output: Any) -> torch.Tensor:
    if isinstance(output, tuple):
        return output[0]
    return output


def load_model(
    model_path: str,
    num_features: int,
    num_classes: int,
    device: str = "cuda",
) -> nn.Module:
    """Load a trained baseline checkpoint and return eval-ready model."""
    model_type = _infer_model_type(model_path)

    checkpoint = torch.load(model_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    model_kwargs: Dict[str, Any] = {}
    if model_type == "mlp":
        hidden_dims = []
        layer_idx = 0
        while f"features.{layer_idx}.weight" in state_dict:
            hidden_dims.append(int(state_dict[f"features.{layer_idx}.weight"].shape[0]))
            layer_idx += 3
        if hidden_dims:
            model_kwargs["hidden_dims"] = tuple(hidden_dims)

    model = get_model(
        model_type,
        num_features=num_features,
        num_classes=num_classes,
        **model_kwargs,
    )

    model.load_state_dict(state_dict, strict=True)
    device_obj = _resolve_device(device)
    model = model.to(device_obj)
    model.eval()
    return model


def _build_attack(
    model: nn.Module,
    attack_name: str,
    eps: float,
    attack_kwargs: Dict[str, Any],
):
    name = attack_name.lower()
    if name == "fgsm":
        fgsm_kwargs = dict(attack_kwargs)
        fgsm_kwargs.setdefault("eps", eps)
        return TabularFGSM(model, **fgsm_kwargs)
    if name == "pgd":
        pgd_kwargs = dict(attack_kwargs)
        pgd_kwargs.setdefault("eps", eps)
        pgd_kwargs.setdefault("alpha", eps / 4.0)
        pgd_kwargs.setdefault("steps", 40)
        pgd_kwargs.setdefault("random_start", True)
        return TabularPGD(model, **pgd_kwargs)
    if name == "cw":
        cw_kwargs = dict(attack_kwargs)
        cw_kwargs.setdefault("c", 1.0)
        cw_kwargs.setdefault("kappa", 0)
        cw_kwargs.setdefault("steps", 100)
        cw_kwargs.setdefault("lr", 0.01)
        return TabularCW(model, **cw_kwargs)
    raise ValueError("attack_name must be one of: 'fgsm', 'pgd', 'cw'")


def run_attack(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    attack_name: str,
    perturbation_mask: Optional[np.ndarray] = None,
    eps: float = 0.3,
    batch_size: int = 1024,
    device: str = "cuda",
    **attack_kwargs,
) -> dict:
    """Run an unconstrained attack in batches and return raw attack artifacts."""
    if X.ndim != 2:
        raise ValueError(f"Expected X to be 2D, got shape {X.shape}")
    if y.ndim != 1:
        raise ValueError(f"Expected y to be 1D, got shape {y.shape}")
    if len(X) != len(y):
        raise ValueError(f"X/y length mismatch: {len(X)} vs {len(y)}")
    if perturbation_mask is not None and perturbation_mask.shape[0] != X.shape[1]:
        raise ValueError("perturbation_mask must have same feature size as X")

    device_obj = _resolve_device(device)
    model = model.to(device_obj)
    model.eval()

    attack = _build_attack(model, attack_name=attack_name, eps=eps, attack_kwargs=attack_kwargs)

    x_adv_all: List[np.ndarray] = []
    x_clean_all: List[np.ndarray] = []
    y_true_all: List[np.ndarray] = []
    y_pred_clean_all: List[np.ndarray] = []
    y_pred_adv_all: List[np.ndarray] = []

    n_samples = len(X)
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        x_batch_np = np.array(X[start:end], dtype=np.float32, copy=True)
        y_batch_np = np.array(y[start:end], dtype=np.int64, copy=True)

        x_batch = torch.from_numpy(x_batch_np).to(device_obj)
        y_batch = torch.from_numpy(y_batch_np).to(device_obj)

        with torch.no_grad():
            logits_clean = _extract_logits(model(x_batch))
            pred_clean = torch.argmax(logits_clean, dim=1)

        x_adv_batch = attack(x_batch, y_batch)

        # TODO: Apply perturbation_mask here for constrained attacks.
        # X_adv_batch = X_batch + (X_adv_batch - X_batch) * mask_tensor

        with torch.no_grad():
            logits_adv = _extract_logits(model(x_adv_batch))
            pred_adv = torch.argmax(logits_adv, dim=1)

        x_clean_all.append(x_batch.detach().cpu().numpy())
        x_adv_all.append(x_adv_batch.detach().cpu().numpy())
        y_true_all.append(y_batch.detach().cpu().numpy())
        y_pred_clean_all.append(pred_clean.detach().cpu().numpy())
        y_pred_adv_all.append(pred_adv.detach().cpu().numpy())

    result = {
        "X_adv": np.concatenate(x_adv_all, axis=0),
        "X_clean": np.concatenate(x_clean_all, axis=0),
        "y_true": np.concatenate(y_true_all, axis=0),
        "y_pred_clean": np.concatenate(y_pred_clean_all, axis=0),
        "y_pred_adv": np.concatenate(y_pred_adv_all, axis=0),
        "attack_name": attack_name.lower(),
        "eps": float(eps),
    }
    return result


def run_attack_with_restarts(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    attack_name: str,
    num_restarts: int,
    perturbation_mask: Optional[np.ndarray] = None,
    eps: float = 0.3,
    batch_size: int = 1024,
    device: str = "cuda",
    base_seed: Optional[int] = None,
    **attack_kwargs,
) -> dict:
    """
    Run an attack with multiple restarts and select best adversarial sample per row.

    Selection policy:
    1) Successful flips (clean-correct and adv-wrong) always beat unsuccessful restarts.
    2) Among successful restarts, prefer lower L2 perturbation in scaled space.

    FGSM has no random restart mechanism in this implementation; for FGSM or
    num_restarts <= 1, this function falls back to a single run.
    """
    if num_restarts < 1:
        raise ValueError(f"num_restarts must be >= 1, got {num_restarts}")

    attack_name_l = attack_name.lower()
    if attack_name_l == "fgsm" or num_restarts == 1:
        single = run_attack(
            model=model,
            X=X,
            y=y,
            attack_name=attack_name,
            perturbation_mask=perturbation_mask,
            eps=eps,
            batch_size=batch_size,
            device=device,
            **attack_kwargs,
        )
        single["num_restarts"] = 1
        single["selected_restart"] = np.zeros(len(X), dtype=np.int32)
        single["restart_samples_flipped"] = np.array([int(((single["y_pred_clean"] == single["y_true"]) & (single["y_pred_adv"] != single["y_true"])).sum())], dtype=np.int64)
        return single

    # Initialize with first restart.
    if base_seed is not None:
        torch.manual_seed(int(base_seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(base_seed))

    best = run_attack(
        model=model,
        X=X,
        y=y,
        attack_name=attack_name,
        perturbation_mask=perturbation_mask,
        eps=eps,
        batch_size=batch_size,
        device=device,
        **attack_kwargs,
    )

    clean_correct = best["y_pred_clean"] == best["y_true"]
    best_success = clean_correct & (best["y_pred_adv"] != best["y_true"])
    best_l2 = np.linalg.norm(best["X_adv"] - best["X_clean"], ord=2, axis=1)
    selected_restart = np.zeros(len(X), dtype=np.int32)
    restart_flipped_counts: List[int] = [int(best_success.sum())]

    for restart_idx in range(1, num_restarts):
        if base_seed is not None:
            seed = int(base_seed) + restart_idx
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        cand = run_attack(
            model=model,
            X=X,
            y=y,
            attack_name=attack_name,
            perturbation_mask=perturbation_mask,
            eps=eps,
            batch_size=batch_size,
            device=device,
            **attack_kwargs,
        )

        cand_success = clean_correct & (cand["y_pred_adv"] != cand["y_true"])
        cand_l2 = np.linalg.norm(cand["X_adv"] - cand["X_clean"], ord=2, axis=1)
        restart_flipped_counts.append(int(cand_success.sum()))

        improve_mask = cand_success & ((~best_success) | (cand_l2 < best_l2))
        if np.any(improve_mask):
            best["X_adv"][improve_mask] = cand["X_adv"][improve_mask]
            best["y_pred_adv"][improve_mask] = cand["y_pred_adv"][improve_mask]
            best_success[improve_mask] = cand_success[improve_mask]
            best_l2[improve_mask] = cand_l2[improve_mask]
            selected_restart[improve_mask] = restart_idx

    best["num_restarts"] = int(num_restarts)
    best["selected_restart"] = selected_restart
    best["restart_samples_flipped"] = np.array(restart_flipped_counts, dtype=np.int64)
    return best


def compute_attack_metrics(result: dict, class_names: Optional[List[str]] = None) -> dict:
    """Compute aggregate and per-class adversarial effectiveness metrics."""
    x_adv = np.asarray(result["X_adv"], dtype=np.float32)
    x_clean = np.asarray(result["X_clean"], dtype=np.float32)
    y_true = np.asarray(result["y_true"], dtype=np.int64)
    y_pred_clean = np.asarray(result["y_pred_clean"], dtype=np.int64)
    y_pred_adv = np.asarray(result["y_pred_adv"], dtype=np.int64)

    clean_correct = y_pred_clean == y_true
    wrong_adv = y_pred_adv != y_true
    flipped = clean_correct & wrong_adv

    clean_correct_count = int(clean_correct.sum())
    samples_flipped = int(flipped.sum())

    clean_accuracy = float(clean_correct.mean())
    asr_raw = float(samples_flipped / clean_correct_count) if clean_correct_count > 0 else 0.0

    deltas = x_adv - x_clean
    linf_per_sample = np.max(np.abs(deltas), axis=1)
    l2_per_sample = np.linalg.norm(deltas, ord=2, axis=1)

    per_class_asr = []
    for cls in sorted(np.unique(y_true).tolist()):
        cls_mask = y_true == cls
        cls_correct_mask = clean_correct & cls_mask
        cls_correct_count = int(cls_correct_mask.sum())
        cls_flipped = int((flipped & cls_mask).sum())
        cls_asr = float(cls_flipped / cls_correct_count) if cls_correct_count > 0 else 0.0
        if class_names is not None and cls < len(class_names):
            cls_name = class_names[cls]
        else:
            cls_name = str(cls)

        per_class_asr.append(
            {
                "class_index": int(cls),
                "class_name": cls_name,
                "samples_originally_correct": cls_correct_count,
                "samples_flipped": cls_flipped,
                "asr_raw": cls_asr,
            }
        )

    return {
        "attack_name": result.get("attack_name", "unknown"),
        "eps": float(result.get("eps", 0.0)),
        "total_samples": int(len(y_true)),
        "samples_originally_correct": clean_correct_count,
        "samples_flipped": samples_flipped,
        "clean_accuracy": clean_accuracy,
        "attack_success_rate_raw": asr_raw,
        "mean_l_inf": float(linf_per_sample.mean()),
        "mean_l2": float(l2_per_sample.mean()),
        "per_class_asr": per_class_asr,
    }
