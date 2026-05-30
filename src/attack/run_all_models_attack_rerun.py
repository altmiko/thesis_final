from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = str(_REPO_ROOT / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from attack.adversarial_attacks import load_model  # noqa: E402
from attack.input_baselines import input_cw_attack, input_pgd_attack  # noqa: E402
from attack.latent_cw import latent_cw_attack  # noqa: E402
from attack.latent_infra import (  # noqa: E402
    AttackRouter,
    AttackRunLogger,
    MahalanobisOutlierDetector,
    PerturbationMask,
    ProtocolValidator,
    build_per_class_dataset,
    encode_dataset_mu,
    load_collapsed_dims,
    load_split,
    phase0_config_snapshot,
    predict_labels,
    set_global_seed,
)
from attack.latent_pgd import classifier_logits, latent_pgd_attack  # noqa: E402
from attack.latent_pgd import _per_sample_objective  # noqa: E402
from attack.validator import validate_batch  # noqa: E402
from preprocessing.feature_groups import FEATURE_NAMES  # noqa: E402
from vae.config import CLASS_TO_ID, CLASSES  # noqa: E402

SAMPLES_PER_SOURCE_CLASS = 100
SELECTION_BATCH_SIZE = 1024
SOURCE_CLASSES = [name for name in CLASSES if name != "Benign"]
MODEL_SPECS = [
    {"tag": "mlp", "label": "MLP", "checkpoint": "mlp_8class.pt"},
    {"tag": "cnn", "label": "CNN", "checkpoint": "cnn_8class.pt"},
    {"tag": "lstm", "label": "LSTM", "checkpoint": "lstm_8class.pt"},
    {"tag": "serial", "label": "CNN-LSTM", "checkpoint": "serial_8class.pt"},
    {"tag": "dualpath", "label": "DualPath", "checkpoint": "dualpath_8class.pt"},
]
ATTACK_ORDER = ["latent-pgd", "latent-cw", "input-pgd", "input-cw"]
SUMMARY_COLUMNS = [
    "model",
    "attack",
    "n",
    "asr_overall",
    "asr_valid_only",
    "asr_invalid_only",
    "protocol_validity_rate",
    "mask_compliance_rate",
    "joint_validity_rate",
    "idsr",
    "mean_l2_input",
    "mean_l2_latent",
    "raw_g1g8_validity_rate",
    "successful_joint_valid_count",
    "selected_restart_mean",
    "restart_success_counts",
    "restart_labels",
]


def _config_snapshot(seed: int, device: str) -> dict[str, Any]:
    snapshot = phase0_config_snapshot(seed, device)
    snapshot.update(
        {
            "phase": "all_models_rerun",
            "split": "test",
            "models": MODEL_SPECS,
            "sampling": {
                "source_classes": SOURCE_CLASSES,
                "samples_per_source_class": SAMPLES_PER_SOURCE_CLASS,
                "selection_batch_size": SELECTION_BATCH_SIZE,
                "selection_rule": "up to 100 correctly classified test samples per non-benign source class; skip zero-available cells",
                "correctly_classified_only": True,
            },
            "attacks": {
                "latent-pgd": {
                    "epsilon": 0.5,
                    "alpha": 0.05,
                    "num_steps": 40,
                    "random_start": True,
                },
                "latent-cw": {
                    "lambda_conf": 1.0,
                    "kappa": 0.0,
                    "num_iterations": 200,
                    "learning_rate": 0.01,
                    "convergence_threshold": 1e-5,
                },
                "input-pgd": {
                    "epsilon": 0.5,
                    "alpha": 0.05,
                    "num_steps": 40,
                    "random_start": True,
                },
                "input-cw": {
                    "lambda_conf": 1.0,
                    "kappa": 0.0,
                    "num_iterations": 200,
                    "learning_rate": 0.01,
                    "convergence_threshold": 1e-5,
                },
            },
        }
    )
    return snapshot


def _fit_detector(router: AttackRouter, device: str) -> MahalanobisOutlierDetector:
    split_val = load_split("val")
    collapsed_by_class = load_collapsed_dims()
    detector = MahalanobisOutlierDetector(latent_dim=16)
    for class_id, _class_name in enumerate(CLASSES):
        ds = build_per_class_dataset(
            class_id,
            X_val=split_val["X"],
            y_val_8=split_val["y_8"],
            scaler=split_val["scaler"],
            partition=split_val["partition"],
        )
        vae = router.get_vae(class_id)
        z_mu = encode_dataset_mu(vae, ds, device=device)
        detector.fit(class_id, z_mu.cpu(), collapsed_dims=collapsed_by_class[class_id])
    return detector


def _select_correct_source_samples(
    *,
    x_test: np.ndarray,
    y_test: np.ndarray,
    classifier: torch.nn.Module,
    device: str,
    class_id: int,
    max_samples: int,
    selection_batch_size: int = SELECTION_BATCH_SIZE,
) -> np.ndarray:
    class_indices = np.where(y_test == class_id)[0]
    if len(class_indices) == 0:
        raise RuntimeError(f"No test samples found for class_id={class_id}")

    pred_batches: list[np.ndarray] = []
    for start in range(0, len(class_indices), selection_batch_size):
        batch_indices = class_indices[start : start + selection_batch_size]
        preds = predict_labels(
            classifier,
            torch.from_numpy(x_test[batch_indices].astype(np.float32)),
            device=device,
        ).numpy()
        pred_batches.append(preds)
    preds = np.concatenate(pred_batches, axis=0)
    keep = class_indices[preds == class_id]
    return keep[:max_samples]


def _conditional_rate(numerator_mask: torch.Tensor, denominator_mask: torch.Tensor) -> float:
    denom = int(denominator_mask.sum().item())
    if denom == 0:
        return 0.0
    return float((numerator_mask & denominator_mask).float().sum().item() / denom)


def _masked_rate(mask: torch.Tensor | None) -> float | None:
    if mask is None:
        return None
    if mask.numel() == 0:
        return 0.0
    return float(mask.float().mean().item())


def _raw_g1g8_validity(x_adv: torch.Tensor, scaler: Any) -> torch.Tensor:
    x_np = x_adv.detach().cpu().numpy().astype(np.float64)
    x_raw = scaler.inverse_transform(x_np)
    result = validate_batch(x_raw, FEATURE_NAMES)
    return torch.from_numpy(result.overall_valid.astype(np.bool_))


def _sum_restart_counts(rows: list[dict[str, Any]]) -> list[int] | None:
    counts_by_row = [
        row.get("restart_success_counts")
        for row in rows
        if isinstance(row.get("restart_success_counts"), list)
    ]
    if not counts_by_row:
        return None
    max_len = max(len(counts) for counts in counts_by_row)
    summed = [0 for _ in range(max_len)]
    for counts in counts_by_row:
        for idx, value in enumerate(counts):
            summed[idx] += int(value)
    return summed


def _candidate_metrics(
    *,
    class_id: int,
    x_batch: torch.Tensor,
    y_batch: torch.Tensor,
    x_adv: torch.Tensor,
    z_adv: torch.Tensor,
    z_orig: torch.Tensor,
    vae: torch.nn.Module,
    classifier: torch.nn.Module,
    mask: PerturbationMask,
    protocol_validator: ProtocolValidator,
    detector: MahalanobisOutlierDetector,
    scaler: Any,
    device: str,
) -> dict[str, torch.Tensor]:
    with torch.no_grad():
        logits_after = classifier_logits(classifier, x_adv.to(device), device=device).cpu()
        pred_after = torch.argmax(logits_after, dim=1)
        success_mask = pred_after != y_batch.cpu()
        protocol_valid = protocol_validator.validate(x_adv, already_scaled=True).cpu()
        mask_compliance = mask.verify(x_adv.cpu(), x_batch.cpu())["all_compliant"].cpu()
        raw_valid = _raw_g1g8_validity(x_adv.cpu(), scaler)
        joint_valid = protocol_valid & mask_compliance & raw_valid
        z_reencoded, _ = vae.encode(x_adv.to(device))
        outlier_mask = detector.outlier_mask(class_id, z_reencoded.cpu()).cpu()
        input_l2 = torch.linalg.norm(x_adv.cpu() - x_batch.cpu(), dim=1)
        latent_l2 = torch.linalg.norm(z_adv.cpu() - z_orig.cpu(), dim=1)
        objective = _per_sample_objective(
            logits_after,
            y_batch.cpu(),
            targeted=False,
            target_class=0,
        )

    return {
        "pred_after": pred_after,
        "success_mask": success_mask,
        "protocol_valid": protocol_valid,
        "mask_compliance": mask_compliance,
        "raw_valid": raw_valid,
        "joint_valid": joint_valid,
        "outlier_mask": outlier_mask,
        "input_l2": input_l2,
        "latent_l2": latent_l2,
        "objective": objective,
    }


def _empty_best(batch_size: int) -> dict[str, Any]:
    return {
        "x_adv": None,
        "z_adv": None,
        "selected_restart": torch.zeros(batch_size, dtype=torch.long),
        "pred_after": torch.zeros(batch_size, dtype=torch.long),
        "success_mask": torch.zeros(batch_size, dtype=torch.bool),
        "protocol_valid": torch.zeros(batch_size, dtype=torch.bool),
        "mask_compliance": torch.zeros(batch_size, dtype=torch.bool),
        "raw_valid": torch.zeros(batch_size, dtype=torch.bool),
        "joint_valid": torch.zeros(batch_size, dtype=torch.bool),
        "outlier_mask": torch.ones(batch_size, dtype=torch.bool),
        "input_l2": torch.full((batch_size,), float("inf")),
        "latent_l2": torch.full((batch_size,), float("inf")),
        "objective": torch.full((batch_size,), float("-inf")),
    }


def _merge_candidate(
    *,
    best: dict[str, Any],
    candidate_x: torch.Tensor,
    candidate_z: torch.Tensor,
    candidate_metrics: dict[str, torch.Tensor],
    restart_idx: int,
) -> None:
    if best["x_adv"] is None or best["z_adv"] is None:
        best["x_adv"] = candidate_x.cpu().clone()
        best["z_adv"] = candidate_z.cpu().clone()
        best["selected_restart"] = torch.full(
            (candidate_x.shape[0],),
            int(restart_idx),
            dtype=torch.long,
        )
        for key, value in candidate_metrics.items():
            best[key] = value.cpu().clone()
        return

    current_success = best["success_mask"]
    current_joint = best["joint_valid"]
    current_l2 = best["input_l2"]
    current_objective = best["objective"]

    candidate_success = candidate_metrics["success_mask"]
    candidate_joint = candidate_metrics["joint_valid"]
    candidate_l2 = candidate_metrics["input_l2"]
    candidate_objective = candidate_metrics["objective"]

    improved = candidate_success & (~current_success)
    improved |= (candidate_success == current_success) & candidate_joint & (~current_joint)
    improved |= (
        (candidate_success == current_success)
        & (candidate_joint == current_joint)
        & (candidate_l2 < current_l2)
    )
    improved |= (
        (candidate_success == current_success)
        & (candidate_joint == current_joint)
        & (candidate_l2 == current_l2)
        & (candidate_objective > current_objective)
    )

    if improved.any():
        assert best["x_adv"] is not None
        assert best["z_adv"] is not None
        best["x_adv"][improved] = candidate_x.cpu()[improved]
        best["z_adv"][improved] = candidate_z.cpu()[improved]
        best["selected_restart"][improved] = int(restart_idx)
        for key, value in candidate_metrics.items():
            best[key][improved] = value.cpu()[improved]


def _row_from_best(best: dict[str, Any], *, class_id: int, n: int) -> dict[str, Any]:
    success_mask = best["success_mask"]
    protocol_valid = best["protocol_valid"]
    mask_compliance = best["mask_compliance"]
    raw_valid = best["raw_valid"]
    joint_valid = best["joint_valid"]
    invalid_mask = ~joint_valid
    outlier_mask = best["outlier_mask"]

    successful_joint = int((success_mask & joint_valid).sum().item())
    return {
        "class_name": CLASSES[class_id],
        "pred_after": best["pred_after"].numpy().tolist(),
        "success_mask": success_mask.numpy().tolist(),
        "protocol_valid_mask": protocol_valid.numpy().tolist(),
        "mask_compliance_mask": mask_compliance.numpy().tolist(),
        "raw_g1g8_valid_mask": raw_valid.numpy().tolist(),
        "joint_valid_mask": joint_valid.numpy().tolist(),
        "selected_restart": best["selected_restart"].numpy().tolist(),
        "n": int(n),
        "asr_overall": _masked_rate(success_mask),
        "asr_valid_only": _conditional_rate(success_mask, joint_valid),
        "asr_invalid_only": _conditional_rate(success_mask, invalid_mask),
        "protocol_validity_rate": _masked_rate(protocol_valid),
        "mask_compliance_rate": _masked_rate(mask_compliance),
        "raw_g1g8_validity_rate": _masked_rate(raw_valid),
        "joint_validity_rate": _masked_rate(joint_valid),
        "idsr": _masked_rate(~outlier_mask),
        "mean_l2_input": float(best["input_l2"].mean().item()),
        "mean_l2_latent": float(best["latent_l2"].mean().item()),
        "valid_count": int(joint_valid.sum().item()),
        "invalid_count": int(invalid_mask.sum().item()),
        "successful_total_count": int(success_mask.sum().item()),
        "successful_valid_count": successful_joint,
        "successful_invalid_count": int((success_mask & invalid_mask).sum().item()),
        "successful_joint_valid_count": successful_joint,
        "selected_restart_mean": float(best["selected_restart"].float().mean().item()),
    }


def _evaluate_latent_attack(
    *,
    attack_name: str,
    class_id: int,
    x_batch: torch.Tensor,
    y_batch: torch.Tensor,
    vae: torch.nn.Module,
    classifier: torch.nn.Module,
    mask: PerturbationMask,
    protocol_validator: ProtocolValidator,
    detector: MahalanobisOutlierDetector,
    scaler: Any,
    device: str,
    epsilon: float,
    alpha: float,
    num_steps: int,
    random_start: bool,
    lambda_conf: float,
    kappa: float,
    num_iterations: int,
    learning_rate: float,
    convergence_threshold: float,
    num_restarts: int = 1,
    restart_strategy: str = "encoded",
    z_initializers: torch.Tensor | None = None,
    restart_labels: list[str] | None = None,
    adaptive_pgd: bool = True,
    checkpoint_interval: int = 10,
    rho: float = 0.75,
    min_alpha: float = 1e-4,
) -> dict[str, Any]:
    def run_one_restart(
        restart_z: torch.Tensor | None,
        label: str,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        init = restart_z.unsqueeze(0) if restart_z is not None else None
        if attack_name == "latent-pgd":
            return latent_pgd_attack(
                vae=vae,
                classifier=classifier,
                mask=mask,
                x_original=x_batch,
                y_true=y_batch,
                scaler=scaler,
                epsilon=epsilon,
                alpha=alpha,
                num_steps=num_steps,
                random_start=random_start if init is None else False,
                device=device,
                num_restarts=1 if init is not None else int(num_restarts),
                restart_strategy=label,
                z_initializers=init,
                adaptive_pgd=adaptive_pgd,
                checkpoint_interval=checkpoint_interval,
                rho=rho,
                min_alpha=min_alpha,
            )
        if attack_name == "latent-cw":
            return latent_cw_attack(
                vae=vae,
                classifier=classifier,
                mask=mask,
                x_original=x_batch,
                y_true=y_batch,
                scaler=scaler,
                lambda_conf=lambda_conf,
                kappa=kappa,
                num_iterations=num_iterations,
                learning_rate=learning_rate,
                convergence_threshold=convergence_threshold,
                device=device,
                num_restarts=1 if init is not None else int(num_restarts),
                restart_strategy=label,
                z_initializers=init,
            )
        raise KeyError(f"Unsupported latent attack: {attack_name}")

    best = _empty_best(int(x_batch.shape[0]))
    restart_success_counts: list[int] = []
    labels_used: list[str] = []

    if z_initializers is not None:
        restart_tensor = z_initializers.detach().to(device=device, dtype=torch.float32)
        if restart_tensor.ndim == 2:
            restart_tensor = restart_tensor.unsqueeze(0)
        if restart_tensor.ndim != 3:
            raise ValueError(
                "z_initializers must have shape (batch, latent_dim) or "
                f"(num_restarts, batch, latent_dim), got {tuple(restart_tensor.shape)}"
            )
        labels = restart_labels or [f"restart_{idx}" for idx in range(restart_tensor.shape[0])]
        if len(labels) != restart_tensor.shape[0]:
            raise ValueError(
                f"restart_labels length {len(labels)} does not match "
                f"z_initializers count {restart_tensor.shape[0]}"
            )

        for restart_idx, restart_z in enumerate(restart_tensor):
            x_adv, z_adv, metadata = run_one_restart(restart_z, labels[restart_idx])
            z_orig = metadata["z_orig"]
            if not isinstance(z_orig, torch.Tensor):
                raise TypeError("latent attack metadata['z_orig'] must be a tensor")
            metrics = _candidate_metrics(
                class_id=class_id,
                x_batch=x_batch.cpu(),
                y_batch=y_batch.cpu(),
                x_adv=x_adv.cpu(),
                z_adv=z_adv.cpu(),
                z_orig=z_orig.cpu(),
                vae=vae,
                classifier=classifier,
                mask=mask,
                protocol_validator=protocol_validator,
                detector=detector,
                scaler=scaler,
                device=device,
            )
            restart_success_counts.append(int(metrics["success_mask"].sum().item()))
            labels_used.append(str(labels[restart_idx]))
            _merge_candidate(
                best=best,
                candidate_x=x_adv.cpu(),
                candidate_z=z_adv.cpu(),
                candidate_metrics=metrics,
                restart_idx=restart_idx,
            )
    else:
        x_adv, z_adv, metadata = run_one_restart(None, str(restart_strategy))
        z_orig = metadata["z_orig"]
        if not isinstance(z_orig, torch.Tensor):
            raise TypeError("latent attack metadata['z_orig'] must be a tensor")
        metrics = _candidate_metrics(
            class_id=class_id,
            x_batch=x_batch.cpu(),
            y_batch=y_batch.cpu(),
            x_adv=x_adv.cpu(),
            z_adv=z_adv.cpu(),
            z_orig=z_orig.cpu(),
            vae=vae,
            classifier=classifier,
            mask=mask,
            protocol_validator=protocol_validator,
            detector=detector,
            scaler=scaler,
            device=device,
        )
        _merge_candidate(
            best=best,
            candidate_x=x_adv.cpu(),
            candidate_z=z_adv.cpu(),
            candidate_metrics=metrics,
            restart_idx=0,
        )
        selected_restart = metadata.get("selected_restart")
        if isinstance(selected_restart, torch.Tensor):
            best["selected_restart"] = selected_restart.detach().cpu().long()
        restart_counts = metadata.get("restart_success_counts", [])
        restart_success_counts = (
            [int(value) for value in restart_counts]
            if isinstance(restart_counts, list)
            else [int(metrics["success_mask"].sum().item())]
        )
        labels_used = [str(restart_strategy)]

    row = _row_from_best(best, class_id=class_id, n=int(x_batch.shape[0]))
    row["restart_success_counts"] = restart_success_counts
    row["restart_labels"] = labels_used
    return row


def _evaluate_input_attack(
    *,
    attack_name: str,
    class_id: int,
    x_batch: torch.Tensor,
    y_batch: torch.Tensor,
    classifier: torch.nn.Module,
    mask: PerturbationMask,
    protocol_validator: ProtocolValidator,
    device: str,
    epsilon: float,
    alpha: float,
    num_steps: int,
    random_start: bool,
    lambda_conf: float,
    kappa: float,
    num_iterations: int,
    learning_rate: float,
    convergence_threshold: float,
) -> dict[str, Any]:
    if attack_name == "input-pgd":
        x_adv, _metadata = input_pgd_attack(
            classifier=classifier,
            x_original=x_batch,
            y_true=y_batch,
            epsilon=epsilon,
            alpha=alpha,
            num_steps=num_steps,
            random_start=random_start,
            device=device,
        )
    elif attack_name == "input-cw":
        x_adv, _metadata = input_cw_attack(
            classifier=classifier,
            x_original=x_batch,
            y_true=y_batch,
            lambda_conf=lambda_conf,
            kappa=kappa,
            num_iterations=num_iterations,
            learning_rate=learning_rate,
            convergence_threshold=convergence_threshold,
            device=device,
        )
    else:
        raise KeyError(f"Unsupported input attack: {attack_name}")

    with torch.no_grad():
        logits_after = classifier_logits(classifier, x_adv.to(device), device=device).cpu()
        pred_after = torch.argmax(logits_after, dim=1)
        success_mask = pred_after != y_batch.cpu()
        protocol_valid = protocol_validator.validate(x_adv, already_scaled=True).cpu()
        mask_compliance = mask.verify(x_adv.cpu(), x_batch.cpu())["all_compliant"].cpu()
        joint_valid = protocol_valid & mask_compliance
        invalid_mask = ~joint_valid
        input_l2 = torch.linalg.norm(x_adv.cpu() - x_batch.cpu(), dim=1)

    return {
        "class_name": CLASSES[class_id],
        "pred_after": pred_after.numpy().tolist(),
        "success_mask": success_mask.numpy().tolist(),
        "protocol_valid_mask": protocol_valid.numpy().tolist(),
        "mask_compliance_mask": mask_compliance.numpy().tolist(),
        "joint_valid_mask": joint_valid.numpy().tolist(),
        "n": int(x_batch.shape[0]),
        "asr_overall": float(success_mask.float().mean().item()),
        "asr_valid_only": _conditional_rate(success_mask, joint_valid),
        "asr_invalid_only": _conditional_rate(success_mask, invalid_mask),
        "protocol_validity_rate": float(protocol_valid.float().mean().item()),
        "mask_compliance_rate": float(mask_compliance.float().mean().item()),
        "joint_validity_rate": float(joint_valid.float().mean().item()),
        "idsr": None,
        "mean_l2_input": float(input_l2.mean().item()),
        "mean_l2_latent": None,
        "valid_count": int(joint_valid.sum().item()),
        "invalid_count": int(invalid_mask.sum().item()),
        "successful_total_count": int(success_mask.sum().item()),
        "successful_valid_count": int((success_mask & joint_valid).sum().item()),
        "successful_invalid_count": int((success_mask & invalid_mask).sum().item()),
        "successful_joint_valid_count": int((success_mask & joint_valid).sum().item()),
        "raw_g1g8_validity_rate": None,
        "selected_restart_mean": None,
        "restart_success_counts": None,
        "restart_labels": None,
    }


def _aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total_n = sum(row["n"] for row in rows)
    total_valid = sum(row["valid_count"] for row in rows)
    total_invalid = sum(row["invalid_count"] for row in rows)

    latent_l2_rows = [row for row in rows if row["mean_l2_latent"] is not None]
    idsr_rows = [row for row in rows if row["idsr"] is not None]
    raw_rows = [row for row in rows if row.get("raw_g1g8_validity_rate") is not None]
    restart_rows = [row for row in rows if row.get("selected_restart_mean") is not None]
    successful_joint = sum(int(row.get("successful_joint_valid_count", 0)) for row in rows)

    return {
        "n": int(total_n),
        "asr_overall": float(sum(row["successful_total_count"] for row in rows) / total_n),
        "asr_valid_only": float(sum(row["successful_valid_count"] for row in rows) / total_valid) if total_valid > 0 else 0.0,
        "asr_invalid_only": float(sum(row["successful_invalid_count"] for row in rows) / total_invalid) if total_invalid > 0 else 0.0,
        "protocol_validity_rate": float(sum(row["protocol_validity_rate"] * row["n"] for row in rows) / total_n),
        "mask_compliance_rate": float(sum(row["mask_compliance_rate"] * row["n"] for row in rows) / total_n),
        "joint_validity_rate": float(sum(row["joint_validity_rate"] * row["n"] for row in rows) / total_n),
        "idsr": float(sum(row["idsr"] * row["n"] for row in idsr_rows) / total_n) if idsr_rows else None,
        "mean_l2_input": float(sum(row["mean_l2_input"] * row["n"] for row in rows) / total_n),
        "mean_l2_latent": float(sum(row["mean_l2_latent"] * row["n"] for row in latent_l2_rows) / total_n) if latent_l2_rows else None,
        "raw_g1g8_validity_rate": (
            float(sum(row["raw_g1g8_validity_rate"] * row["n"] for row in raw_rows) / total_n)
            if raw_rows
            else None
        ),
        "successful_joint_valid_count": successful_joint,
        "selected_restart_mean": (
            float(sum(row["selected_restart_mean"] * row["n"] for row in restart_rows) / total_n)
            if restart_rows
            else None
        ),
        "restart_success_counts": _sum_restart_counts(rows),
        "restart_labels": rows[0].get("restart_labels") if rows else None,
    }


def _fmt_pct(value: Any) -> str:
    if value is None:
        return "NA"
    return f"{float(value) * 100.0:.2f}%"


def _fmt_float(value: Any) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.4f}"


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    def csv_cell(value: Any) -> Any:
        if isinstance(value, (list, dict)):
            return json.dumps(value)
        return value

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_cell(row.get(key, "")) for key in columns})


def _write_summary_md(path: Path, rows: list[dict[str, Any]]) -> None:
    headers = [
        "Model",
        "Attack",
        "N",
        "ASR",
        "ASR Valid",
        "ASR Invalid",
        "Protocol",
        "Mask",
        "Raw G1-G8",
        "Joint Valid",
        "IDSR",
        "L2 Input",
        "L2 Latent",
        "Sel Restart",
    ]
    lines = [
        "# Attack Rerun Summary",
        "",
        f"Sample setting: {SAMPLES_PER_SOURCE_CLASS} correctly classified test samples per non-benign source class.",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["model"]),
                    str(row["attack"]),
                    str(row["n"]),
                    _fmt_pct(row["asr_overall"]),
                    _fmt_pct(row["asr_valid_only"]),
                    _fmt_pct(row["asr_invalid_only"]),
                    _fmt_pct(row["protocol_validity_rate"]),
                    _fmt_pct(row["mask_compliance_rate"]),
                    _fmt_pct(row.get("raw_g1g8_validity_rate")),
                    _fmt_pct(row["joint_validity_rate"]),
                    _fmt_pct(row["idsr"]),
                    _fmt_float(row["mean_l2_input"]),
                    _fmt_float(row["mean_l2_latent"]),
                    _fmt_float(row.get("selected_restart_mean")),
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_per_sample_rows(
    *,
    sample_indices: np.ndarray,
    y_true: torch.Tensor,
    metrics: dict[str, Any],
    model_name: str,
    model_tag: str,
    attack_name: str,
) -> list[dict[str, Any]]:
    y_true_np = y_true.cpu().numpy()
    pred_after = np.asarray(metrics["pred_after"], dtype=np.int64)
    success_mask = np.asarray(metrics["success_mask"], dtype=np.bool_)
    protocol_valid_mask = np.asarray(metrics["protocol_valid_mask"], dtype=np.bool_)
    mask_compliance_mask = np.asarray(metrics["mask_compliance_mask"], dtype=np.bool_)
    raw_valid_mask = np.asarray(
        metrics.get("raw_g1g8_valid_mask", np.ones_like(success_mask)),
        dtype=np.bool_,
    )
    joint_valid_mask = np.asarray(metrics["joint_valid_mask"], dtype=np.bool_)
    selected_restart = np.asarray(
        metrics.get("selected_restart", np.zeros_like(y_true_np)),
        dtype=np.int64,
    )

    rows: list[dict[str, Any]] = []
    for pos, sample_id in enumerate(sample_indices.tolist()):
        rows.append(
            {
                "sample_id": int(sample_id),
                "true_label": CLASSES[int(y_true_np[pos])],
                "predicted_label": CLASSES[int(pred_after[pos])],
                "protocol_valid": bool(protocol_valid_mask[pos]),
                "mask_valid": bool(mask_compliance_mask[pos]),
                "raw_g1g8_valid": bool(raw_valid_mask[pos]),
                "joint_valid": bool(joint_valid_mask[pos]),
                "success": bool(success_mask[pos]),
                "selected_restart": int(selected_restart[pos]),
                "attack_type": attack_name,
                "model_name": model_name,
                "model_tag": model_tag,
                "source_class": metrics["class_name"],
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Rerun latent/input PGD and C&W across all 8-class neural baselines.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epsilon", type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--num-steps", type=int, default=40)
    parser.add_argument("--random-start", action="store_true", default=True)
    parser.add_argument("--no-random-start", dest="random_start", action="store_false")
    parser.add_argument("--lambda-conf", type=float, default=1.0)
    parser.add_argument("--kappa", type=float, default=0.0)
    parser.add_argument("--num-iterations", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--convergence-threshold", type=float, default=1e-5)
    args = parser.parse_args()

    set_global_seed(args.seed)
    run_logger = AttackRunLogger.create(
        phase_name="all_models_rerun",
        seed=args.seed,
        config_snapshot=_config_snapshot(args.seed, args.device),
    )

    router = AttackRouter(device=args.device)
    mask = PerturbationMask.from_preprocessing_artifacts()
    protocol_validator = ProtocolValidator(router.scaler)
    split_test = load_split("test")
    detector = _fit_detector(router, args.device)

    classifiers: dict[str, torch.nn.Module] = {}
    for spec in MODEL_SPECS:
        ckpt_path = _REPO_ROOT / "models" / spec["checkpoint"]
        classifiers[spec["tag"]] = load_model(
            model_path=str(ckpt_path),
            num_features=len(FEATURE_NAMES),
            num_classes=len(CLASSES),
            device=args.device,
        )

    per_class_rows: list[dict[str, Any]] = []
    per_sample_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    skipped_cells: list[dict[str, Any]] = []

    for spec in MODEL_SPECS:
        classifier = classifiers[spec["tag"]]
        selected_indices_by_class: dict[int, np.ndarray] = {}
        for class_name in SOURCE_CLASSES:
            class_id = CLASS_TO_ID[class_name]
            selected_indices_by_class[class_id] = _select_correct_source_samples(
                x_test=split_test["X"],
                y_test=split_test["y_8"],
                classifier=classifier,
                device=args.device,
                class_id=class_id,
                max_samples=SAMPLES_PER_SOURCE_CLASS,
            )
            if selected_indices_by_class[class_id].size == 0:
                skipped_cells.append(
                    {
                        "model": spec["label"],
                        "model_tag": spec["tag"],
                        "class_name": class_name,
                        "reason": "no_correctly_classified_samples",
                    }
                )

        for attack_name in ATTACK_ORDER:
            attack_rows: list[dict[str, Any]] = []
            for class_name in SOURCE_CLASSES:
                class_id = CLASS_TO_ID[class_name]
                sample_idx = selected_indices_by_class[class_id]
                if sample_idx.size == 0:
                    continue
                x_batch = torch.from_numpy(split_test["X"][sample_idx].astype(np.float32))
                y_batch = torch.from_numpy(split_test["y_8"][sample_idx].astype(np.int64))

                if attack_name.startswith("latent-"):
                    vae = router.get_vae(class_id)
                    metrics = _evaluate_latent_attack(
                        attack_name=attack_name,
                        class_id=class_id,
                        x_batch=x_batch,
                        y_batch=y_batch,
                        vae=vae,
                        classifier=classifier,
                        mask=mask,
                        protocol_validator=protocol_validator,
                        detector=detector,
                        scaler=router.scaler,
                        device=args.device,
                        epsilon=args.epsilon,
                        alpha=args.alpha,
                        num_steps=args.num_steps,
                        random_start=args.random_start,
                        lambda_conf=args.lambda_conf,
                        kappa=args.kappa,
                        num_iterations=args.num_iterations,
                        learning_rate=args.learning_rate,
                        convergence_threshold=args.convergence_threshold,
                    )
                else:
                    metrics = _evaluate_input_attack(
                        attack_name=attack_name,
                        class_id=class_id,
                        x_batch=x_batch,
                        y_batch=y_batch,
                        classifier=classifier,
                        mask=mask,
                        protocol_validator=protocol_validator,
                        device=args.device,
                        epsilon=args.epsilon,
                        alpha=args.alpha,
                        num_steps=args.num_steps,
                        random_start=args.random_start,
                        lambda_conf=args.lambda_conf,
                        kappa=args.kappa,
                        num_iterations=args.num_iterations,
                        learning_rate=args.learning_rate,
                        convergence_threshold=args.convergence_threshold,
                    )

                metrics["model"] = spec["label"]
                metrics["model_tag"] = spec["tag"]
                metrics["attack"] = attack_name
                attack_rows.append(metrics)
                per_class_rows.append(metrics)
                per_sample_rows.extend(
                    _build_per_sample_rows(
                        sample_indices=sample_idx,
                        y_true=y_batch,
                        metrics=metrics,
                        model_name=spec["label"],
                        model_tag=spec["tag"],
                        attack_name=attack_name,
                    )
                )
                run_logger.log(json.dumps(metrics))

            if not attack_rows:
                summary_rows.append(
                    {
                        "model": spec["label"],
                        "model_tag": spec["tag"],
                        "attack": attack_name,
                        "n": 0,
                        "asr_overall": None,
                        "asr_valid_only": None,
                        "asr_invalid_only": None,
                        "protocol_validity_rate": None,
                        "mask_compliance_rate": None,
                        "joint_validity_rate": None,
                        "idsr": None,
                        "mean_l2_input": None,
                        "mean_l2_latent": None,
                    }
                )
                continue

            summary = _aggregate_rows(attack_rows)
            summary["model"] = spec["label"]
            summary["model_tag"] = spec["tag"]
            summary["attack"] = attack_name
            summary_rows.append(summary)

    summary_rows.sort(key=lambda row: (row["model"], ATTACK_ORDER.index(row["attack"])))
    per_class_rows.sort(key=lambda row: (row["model"], row["attack"], row["class_name"]))

    payload = {
        "summary": summary_rows,
        "per_class": per_class_rows,
        "per_sample": per_sample_rows,
        "skipped_cells": skipped_cells,
    }
    with open(run_logger.run_dir / "all_results.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    _write_csv(run_logger.run_dir / "summary.csv", summary_rows, ["model_tag"] + SUMMARY_COLUMNS)
    _write_csv(
        run_logger.run_dir / "per_class.csv",
        per_class_rows,
        [
            "model",
            "model_tag",
            "attack",
            "class_name",
            "n",
            "asr_overall",
            "asr_valid_only",
            "asr_invalid_only",
            "protocol_validity_rate",
            "mask_compliance_rate",
            "joint_validity_rate",
            "idsr",
            "mean_l2_input",
            "mean_l2_latent",
            "raw_g1g8_validity_rate",
            "successful_joint_valid_count",
            "selected_restart_mean",
            "restart_success_counts",
            "restart_labels",
        ],
    )
    _write_csv(
        run_logger.run_dir / "per_sample_results.csv",
        per_sample_rows,
        [
            "sample_id",
            "true_label",
            "predicted_label",
            "protocol_valid",
            "mask_valid",
            "raw_g1g8_valid",
            "joint_valid",
            "success",
            "selected_restart",
            "attack_type",
            "model_name",
            "model_tag",
            "source_class",
        ],
    )
    _write_csv(
        run_logger.run_dir / "skipped_cells.csv",
        skipped_cells,
        ["model", "model_tag", "class_name", "reason"],
    )
    _write_summary_md(run_logger.run_dir / "summary.md", summary_rows)

    print("=== All-Model Attack Rerun Summary ===")
    print(f"Output directory: {run_logger.run_dir}")
    print()
    for row in summary_rows:
        print(
            f"{row['model']:9s} {row['attack']:10s} "
            f"ASR={_fmt_pct(row['asr_overall']):>8s} "
            f"ASR_valid={_fmt_pct(row['asr_valid_only']):>8s} "
            f"Proto={_fmt_pct(row['protocol_validity_rate']):>8s} "
            f"Mask={_fmt_pct(row['mask_compliance_rate']):>8s} "
            f"Raw={_fmt_pct(row.get('raw_g1g8_validity_rate')):>8s} "
            f"Joint={_fmt_pct(row['joint_validity_rate']):>8s} "
            f"IDSR={_fmt_pct(row['idsr']) if row['idsr'] is not None else 'NA':>7s} "
            f"L2_in={_fmt_float(row['mean_l2_input']):>8s} "
            f"L2_z={_fmt_float(row['mean_l2_latent']):>8s}"
        )


if __name__ == "__main__":
    main()
