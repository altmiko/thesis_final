from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from torch.utils.data import Subset

warnings.filterwarnings(
    "ignore",
    message=".*adaptive_max_pool2d_backward_cuda does not have a deterministic implementation.*",
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = str(_REPO_ROOT / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from attack.adversarial_attacks import load_model  # noqa: E402
from attack.latent_gmm import LatentGMMPrior, fit_or_load_latent_gmm  # noqa: E402
from attack.latent_infra import (  # noqa: E402
    AttackRouter,
    AttackRunLogger,
    MahalanobisOutlierDetector,
    PerturbationMask,
    ProtocolValidator,
    build_per_class_dataset,
    encode_dataset_mu,
    load_split,
    phase0_config_snapshot,
    predict_labels,
    set_global_seed,
)
from attack.latent_pgd import classifier_logits, latent_pgd_attack  # noqa: E402
from attack.validator import validate_batch  # noqa: E402
from preprocessing.feature_groups import FEATURE_NAMES  # noqa: E402
from vae.config import CLASS_TO_ID, CLASSES  # noqa: E402


TARGET_CLASS_ID = CLASS_TO_ID["Benign"]
TARGET_CLASS_NAME = "Benign"
SOURCE_CLASSES = [name for name in CLASSES if name != TARGET_CLASS_NAME]
MODEL_SPECS = [
    {"tag": "mlp", "label": "MLP", "checkpoint": "mlp_8class.pt"},
    {"tag": "cnn", "label": "CNN", "checkpoint": "cnn_8class.pt"},
    {"tag": "lstm", "label": "LSTM", "checkpoint": "lstm_8class.pt"},
    {"tag": "serial", "label": "CNN-LSTM", "checkpoint": "serial_8class.pt"},
    {"tag": "dualpath", "label": "DualPath", "checkpoint": "dualpath_8class.pt"},
]
SUMMARY_COLUMNS = [
    "vae_run_tag",
    "model",
    "model_tag",
    "attack",
    "n",
    "benign_target_success_rate",
    "benign_target_success_joint_rate",
    "benign_target_success_given_joint_valid",
    "protocol_validity_rate",
    "mask_compliance_rate",
    "raw_g1g8_validity_rate",
    "joint_validity_rate",
    "idsr",
    "mean_l2_input",
    "mean_l2_latent",
    "mean_selected_restart",
]
PER_CLASS_COLUMNS = SUMMARY_COLUMNS + ["class_name", "restart_target_success_counts"]
PER_SAMPLE_COLUMNS = [
    "sample_id",
    "vae_run_tag",
    "model",
    "model_tag",
    "source_class",
    "true_label",
    "pred_before",
    "pred_after",
    "target_class",
    "target_success",
    "protocol_valid",
    "mask_valid",
    "raw_g1g8_valid",
    "joint_valid",
    "selected_restart",
]


def _load_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _sanitize_phase_name(value: str) -> str:
    keep = []
    for char in value:
        if char.isalnum() or char in {"_", "-"}:
            keep.append(char)
        else:
            keep.append("_")
    return "".join(keep)


def _resolve_vae_paths(args: argparse.Namespace) -> tuple[str, Path, Path]:
    run_tag = str(args.vae_run_tag or "").strip()
    manifest_path = Path(args.vae_manifest).resolve() if args.vae_manifest else None
    diagnostics_dir = Path(args.vae_diagnostics_dir).resolve() if args.vae_diagnostics_dir else None

    if manifest_path is None:
        if run_tag:
            manifest_path = _REPO_ROOT / "results" / "vae" / run_tag / "vae_run_manifest.json"
        else:
            manifest_path = _REPO_ROOT / "vae_run_manifest.json"

    if not manifest_path.exists():
        raise FileNotFoundError(f"VAE manifest not found: {manifest_path}")

    if diagnostics_dir is None:
        diagnostics_dir = _REPO_ROOT / "results" / "vae" / run_tag if run_tag else _REPO_ROOT / "results" / "vae"

    if not diagnostics_dir.exists():
        raise FileNotFoundError(f"VAE diagnostics directory not found: {diagnostics_dir}")

    if not run_tag:
        run_tag = "root_manifest" if manifest_path.parent == _REPO_ROOT else manifest_path.parent.name

    return run_tag, manifest_path, diagnostics_dir


def _load_collapsed_dims_for_run(
    *,
    diagnostics_dir: Path,
    manifest: dict[str, Any],
) -> dict[int, list[int]]:
    collapsed: dict[int, list[int]] = {}
    manifest_diagnostics = manifest.get("diagnostics", {})

    for class_name in CLASSES:
        class_id = CLASS_TO_ID[class_name]
        diag_path = diagnostics_dir / f"diagnostics_{class_name}.json"
        manifest_diag = manifest_diagnostics.get(class_name, {})
        if not diag_path.exists() and manifest_diag.get("path"):
            diag_path = Path(str(manifest_diag["path"]))
        if not diag_path.exists():
            raise FileNotFoundError(f"Diagnostics not found for {class_name}: {diag_path}")

        diag = _load_json(diag_path)
        collapsed[class_id] = list(diag["posterior_collapse"]["collapsed_dim_indices"])

    return collapsed


def _config_snapshot(
    args: argparse.Namespace,
    *,
    run_tag: str,
    manifest_path: Path,
    diagnostics_dir: Path,
    manifest: dict[str, Any],
    model_specs: list[dict[str, str]],
) -> dict[str, Any]:
    snapshot = phase0_config_snapshot(args.seed, args.device)
    snapshot.update(
        {
            "phase": "targeted_benign_pgd",
            "split": "test",
            "vae_run": {
                "run_tag": run_tag,
                "manifest_path": str(manifest_path),
                "diagnostics_dir": str(diagnostics_dir),
                "checkpoint_paths": {
                    class_name: manifest["checkpoints"][class_name]["path"]
                    for class_name in CLASSES
                },
            },
            "target": {"class_id": TARGET_CLASS_ID, "class_name": TARGET_CLASS_NAME},
            "models": model_specs,
            "sampling": {
                "source_classes": SOURCE_CLASSES,
                "samples_per_source_class": int(args.samples_per_class),
                "selection_rule": "up to N correctly classified test samples per non-benign source class",
                "correctly_classified_only": True,
            },
            "attack": {
                "name": "targeted-benign-latent-pgd",
                "epsilon": float(args.epsilon),
                "alpha": float(args.alpha),
                "num_steps": int(args.num_steps),
                "num_restarts": int(args.num_restarts),
                "restart_strategy": str(args.restart_strategy),
                "targeted": True,
                "target_class": TARGET_CLASS_ID,
                "selection_policy": "target success, then joint validity, then lower input L2",
            },
            "gmm": {
                "split": str(args.gmm_split),
                "n_components": int(args.gmm_components),
                "max_fit_samples": args.gmm_fit_max_samples,
                "force_refit": bool(args.force_refit_gmm),
            },
            "idsr_detector": {
                "max_samples_per_class": args.detector_max_samples,
            },
            "selection_batch_size": int(args.selection_batch_size),
        }
    )
    return snapshot


def _resolve_model_specs(models_arg: str) -> list[dict[str, str]]:
    requested = [item.strip().lower() for item in models_arg.split(",") if item.strip()]
    if not requested or requested == ["all"]:
        return list(MODEL_SPECS)

    by_tag = {spec["tag"]: spec for spec in MODEL_SPECS}
    unknown = [tag for tag in requested if tag not in by_tag]
    if unknown:
        raise ValueError(f"Unknown model tags {unknown}; valid tags: {sorted(by_tag)} or all")
    return [by_tag[tag] for tag in requested]


def _fit_detector(
    router: AttackRouter,
    device: str,
    *,
    collapsed_by_class: dict[int, list[int]],
    max_samples_per_class: int | None = None,
    seed: int = 42,
) -> MahalanobisOutlierDetector:
    split_val = load_split("val")
    detector = MahalanobisOutlierDetector(latent_dim=16)
    for class_id, _class_name in enumerate(CLASSES):
        ds = build_per_class_dataset(
            class_id,
            X_val=split_val["X"],
            y_val_8=split_val["y_8"],
            scaler=split_val["scaler"],
            partition=split_val["partition"],
        )
        ds_for_fit: Any = ds
        if max_samples_per_class is not None and len(ds) > int(max_samples_per_class):
            rng = np.random.default_rng(int(seed) + int(class_id))
            indices = np.sort(
                rng.choice(len(ds), size=int(max_samples_per_class), replace=False)
            ).tolist()
            ds_for_fit = Subset(ds, indices)
        vae = router.get_vae(class_id)
        z_mu = encode_dataset_mu(vae, ds_for_fit, device=device)
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
    prediction_batch_size: int,
) -> np.ndarray:
    class_indices = np.where(y_test == class_id)[0]
    if len(class_indices) == 0:
        return np.array([], dtype=np.int64)

    pred_batches: list[np.ndarray] = []
    for start in range(0, len(class_indices), int(prediction_batch_size)):
        end = min(start + int(prediction_batch_size), len(class_indices))
        batch_indices = class_indices[start:end]
        preds_batch = predict_labels(
            classifier,
            torch.from_numpy(x_test[batch_indices].astype(np.float32)),
            device=device,
        ).numpy()
        pred_batches.append(preds_batch)
    preds = np.concatenate(pred_batches, axis=0)
    keep = class_indices[preds == class_id]
    return keep[: int(max_samples)]


def _build_restart_initializers(
    *,
    vae: torch.nn.Module,
    x_batch: torch.Tensor,
    gmm_prior: LatentGMMPrior | None,
    epsilon: float,
    num_restarts: int,
    restart_strategy: str,
    seed: int,
    class_id: int,
    device: str,
) -> tuple[list[torch.Tensor], list[str]]:
    with torch.no_grad():
        z_orig, _ = vae.encode(x_batch.to(device=device, dtype=torch.float32))

    strategy_parts = {
        part.strip().lower()
        for part in restart_strategy.replace(",", "+").split("+")
        if part.strip()
    }
    if "gmm" in strategy_parts and gmm_prior is None:
        raise ValueError("restart_strategy includes gmm, but no gmm_prior was provided")

    z_starts = [z_orig.detach().clone()]
    labels = ["encoded"]
    restart_kinds: list[str] = []
    if "jitter" in strategy_parts:
        restart_kinds.append("jitter")
    if "gmm" in strategy_parts:
        restart_kinds.append("gmm")
    if not restart_kinds:
        restart_kinds.append("encoded")

    for restart_idx in range(1, max(1, int(num_restarts))):
        kind = restart_kinds[(restart_idx - 1) % len(restart_kinds)]
        if kind == "jitter":
            generator = torch.Generator(device=z_orig.device)
            generator.manual_seed(int(seed) + int(class_id) * 1009 + int(restart_idx))
            if float(epsilon) > 0.0:
                noise = torch.empty_like(z_orig).uniform_(
                    -float(epsilon),
                    float(epsilon),
                    generator=generator,
                )
                z_start = z_orig + noise
            else:
                z_start = z_orig.clone()
        elif kind == "gmm":
            assert gmm_prior is not None
            z_start = gmm_prior.sample(
                int(z_orig.shape[0]),
                seed=int(seed) + int(class_id) * 2003 + int(restart_idx),
                device=device,
                dtype=z_orig.dtype,
            )
            if float(epsilon) > 0.0:
                z_start = torch.max(
                    torch.min(z_start, z_orig + float(epsilon)),
                    z_orig - float(epsilon),
                )
        else:
            z_start = z_orig.clone()
        z_starts.append(z_start.detach().clone())
        labels.append(kind)

    return z_starts, labels


def _raw_g1g8_validity(x_adv: torch.Tensor, scaler: Any) -> torch.Tensor:
    x_np = x_adv.detach().cpu().numpy().astype(np.float64)
    x_raw = scaler.inverse_transform(x_np)
    result = validate_batch(x_raw, FEATURE_NAMES)
    return torch.from_numpy(result.overall_valid.astype(np.bool_))


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
        target_success = pred_after == TARGET_CLASS_ID
        protocol_valid = protocol_validator.validate(x_adv, already_scaled=True).cpu()
        mask_valid = mask.verify(x_adv.cpu(), x_batch.cpu())["all_compliant"].cpu()
        raw_valid = _raw_g1g8_validity(x_adv.cpu(), scaler)
        joint_valid = protocol_valid & mask_valid & raw_valid
        z_reencoded, _ = vae.encode(x_adv.to(device))
        outlier_mask = detector.outlier_mask(class_id, z_reencoded.cpu()).cpu()
        input_l2 = torch.linalg.norm(x_adv.cpu() - x_batch.cpu(), dim=1)
        latent_l2 = torch.linalg.norm(z_adv.cpu() - z_orig.cpu(), dim=1)

    return {
        "pred_after": pred_after,
        "target_success": target_success,
        "protocol_valid": protocol_valid,
        "mask_valid": mask_valid,
        "raw_valid": raw_valid,
        "joint_valid": joint_valid,
        "outlier_mask": outlier_mask,
        "input_l2": input_l2,
        "latent_l2": latent_l2,
    }


def _run_targeted_pgd_restart_pool(
    *,
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
    z_starts: list[torch.Tensor],
    restart_labels: list[str],
) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    restart_target_success_counts: list[int] = []
    z_orig_ref: torch.Tensor | None = None

    for restart_idx, z_start in enumerate(z_starts):
        x_adv, z_adv, metadata = latent_pgd_attack(
            vae=vae,
            classifier=classifier,
            mask=mask,
            x_original=x_batch,
            y_true=y_batch,
            scaler=scaler,
            epsilon=epsilon,
            alpha=alpha,
            num_steps=num_steps,
            random_start=False,
            device=device,
            targeted=True,
            target_class=TARGET_CLASS_ID,
            num_restarts=1,
            restart_strategy=restart_labels[restart_idx],
            z_initializers=z_start.unsqueeze(0),
        )
        z_orig = metadata["z_orig"]
        if not isinstance(z_orig, torch.Tensor):
            raise TypeError("latent_pgd_attack metadata['z_orig'] must be a tensor")
        z_orig_ref = z_orig.detach().cpu()

        metrics = _candidate_metrics(
            class_id=class_id,
            x_batch=x_batch.cpu(),
            y_batch=y_batch.cpu(),
            x_adv=x_adv.cpu(),
            z_adv=z_adv.cpu(),
            z_orig=z_orig_ref,
            vae=vae,
            classifier=classifier,
            mask=mask,
            protocol_validator=protocol_validator,
            detector=detector,
            scaler=scaler,
            device=device,
        )
        restart_target_success_counts.append(int(metrics["target_success"].sum().item()))

        if best is None:
            best = {
                "x_adv": x_adv.cpu().clone(),
                "z_adv": z_adv.cpu().clone(),
                "selected_restart": torch.full(
                    (x_batch.shape[0],),
                    int(restart_idx),
                    dtype=torch.long,
                ),
                **metrics,
            }
            continue

        current_success = best["target_success"]
        current_joint = best["joint_valid"]
        current_l2 = best["input_l2"]

        candidate_success = metrics["target_success"]
        candidate_joint = metrics["joint_valid"]
        candidate_l2 = metrics["input_l2"]

        improved = candidate_success & (~current_success)
        improved |= (candidate_success == current_success) & candidate_joint & (~current_joint)
        improved |= (
            (candidate_success == current_success)
            & (candidate_joint == current_joint)
            & (candidate_l2 < current_l2)
        )

        if improved.any():
            best["x_adv"][improved] = x_adv.cpu()[improved]
            best["z_adv"][improved] = z_adv.cpu()[improved]
            best["selected_restart"][improved] = int(restart_idx)
            for key, value in metrics.items():
                best[key][improved] = value[improved]

    if best is None or z_orig_ref is None:
        raise RuntimeError("No targeted PGD restart candidates were evaluated")

    best["z_orig"] = z_orig_ref
    best["restart_target_success_counts"] = restart_target_success_counts
    return best


def _rate(mask: torch.Tensor) -> float:
    if mask.numel() == 0:
        return 0.0
    return float(mask.float().mean().item())


def _conditional_rate(numerator: torch.Tensor, denominator: torch.Tensor) -> float:
    denom = int(denominator.sum().item())
    if denom == 0:
        return 0.0
    return float((numerator & denominator).float().sum().item() / denom)


def _row_from_best(
    *,
    vae_run_tag: str,
    model_label: str,
    model_tag: str,
    class_name: str,
    best: dict[str, Any],
) -> dict[str, Any]:
    target_success = best["target_success"]
    joint_valid = best["joint_valid"]
    in_distribution = ~best["outlier_mask"]

    return {
        "vae_run_tag": vae_run_tag,
        "model": model_label,
        "model_tag": model_tag,
        "attack": "targeted_benign_latent_pgd",
        "class_name": class_name,
        "n": int(target_success.numel()),
        "benign_target_success_rate": _rate(target_success),
        "benign_target_success_joint_rate": _rate(target_success & joint_valid),
        "benign_target_success_given_joint_valid": _conditional_rate(target_success, joint_valid),
        "protocol_validity_rate": _rate(best["protocol_valid"]),
        "mask_compliance_rate": _rate(best["mask_valid"]),
        "raw_g1g8_validity_rate": _rate(best["raw_valid"]),
        "joint_validity_rate": _rate(joint_valid),
        "idsr": _rate(in_distribution),
        "mean_l2_input": float(best["input_l2"].float().mean().item()),
        "mean_l2_latent": float(best["latent_l2"].float().mean().item()),
        "mean_selected_restart": float(best["selected_restart"].float().mean().item()),
        "restart_target_success_counts": json.dumps(best["restart_target_success_counts"]),
    }


def _aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total_n = sum(int(row["n"]) for row in rows)
    if total_n == 0:
        raise ValueError("Cannot aggregate empty rows")

    def weighted(key: str) -> float:
        return float(sum(float(row[key]) * int(row["n"]) for row in rows) / total_n)

    total_joint = sum(float(row["joint_validity_rate"]) * int(row["n"]) for row in rows)
    total_success_joint = sum(
        float(row["benign_target_success_joint_rate"]) * int(row["n"])
        for row in rows
    )
    return {
        "vae_run_tag": rows[0].get("vae_run_tag", ""),
        "model": rows[0]["model"],
        "model_tag": rows[0]["model_tag"],
        "attack": "targeted_benign_latent_pgd",
        "n": int(total_n),
        "benign_target_success_rate": weighted("benign_target_success_rate"),
        "benign_target_success_joint_rate": weighted("benign_target_success_joint_rate"),
        "benign_target_success_given_joint_valid": (
            float(total_success_joint / total_joint) if total_joint > 0 else 0.0
        ),
        "protocol_validity_rate": weighted("protocol_validity_rate"),
        "mask_compliance_rate": weighted("mask_compliance_rate"),
        "raw_g1g8_validity_rate": weighted("raw_g1g8_validity_rate"),
        "joint_validity_rate": weighted("joint_validity_rate"),
        "idsr": weighted("idsr"),
        "mean_l2_input": weighted("mean_l2_input"),
        "mean_l2_latent": weighted("mean_l2_latent"),
        "mean_selected_restart": weighted("mean_selected_restart"),
    }


def _build_per_sample_rows(
    *,
    sample_indices: np.ndarray,
    vae_run_tag: str,
    model_label: str,
    model_tag: str,
    class_name: str,
    y_batch: torch.Tensor,
    pred_before: torch.Tensor,
    best: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    y_np = y_batch.cpu().numpy()
    pred_before_np = pred_before.cpu().numpy()
    pred_after_np = best["pred_after"].cpu().numpy()
    target_success_np = best["target_success"].cpu().numpy()
    protocol_np = best["protocol_valid"].cpu().numpy()
    mask_np = best["mask_valid"].cpu().numpy()
    raw_np = best["raw_valid"].cpu().numpy()
    joint_np = best["joint_valid"].cpu().numpy()
    selected_restart_np = best["selected_restart"].cpu().numpy()

    for pos, sample_id in enumerate(sample_indices.tolist()):
        rows.append(
            {
                "sample_id": int(sample_id),
                "vae_run_tag": vae_run_tag,
                "model": model_label,
                "model_tag": model_tag,
                "source_class": class_name,
                "true_label": CLASSES[int(y_np[pos])],
                "pred_before": CLASSES[int(pred_before_np[pos])],
                "pred_after": CLASSES[int(pred_after_np[pos])],
                "target_class": TARGET_CLASS_NAME,
                "target_success": bool(target_success_np[pos]),
                "protocol_valid": bool(protocol_np[pos]),
                "mask_valid": bool(mask_np[pos]),
                "raw_g1g8_valid": bool(raw_np[pos]),
                "joint_valid": bool(joint_np[pos]),
                "selected_restart": int(selected_restart_np[pos]),
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _fmt_pct(value: Any) -> str:
    return f"{float(value) * 100.0:.2f}%"


def _fmt_float(value: Any) -> str:
    return f"{float(value):.4f}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run targeted-to-Benign latent PGD with GMM-seeded restarts."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--vae-run-tag", default=None)
    parser.add_argument("--vae-manifest", default=None)
    parser.add_argument("--vae-diagnostics-dir", default=None)
    parser.add_argument("--models", default="mlp", help="Comma-separated model tags or all")
    parser.add_argument("--samples-per-class", type=int, default=100)
    parser.add_argument("--epsilon", type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--num-steps", type=int, default=40)
    parser.add_argument("--num-restarts", type=int, default=5)
    parser.add_argument("--restart-strategy", default="encoded+jitter+gmm")
    parser.add_argument("--gmm-split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--gmm-components", type=int, default=5)
    parser.add_argument("--gmm-fit-max-samples", type=int, default=50000)
    parser.add_argument("--force-refit-gmm", action="store_true")
    parser.add_argument(
        "--detector-max-samples",
        type=int,
        default=None,
        help="Optional per-class cap for IDSR detector fitting; useful for smoke runs.",
    )
    parser.add_argument(
        "--selection-batch-size",
        type=int,
        default=8192,
        help="Batch size for selecting correctly classified samples on GPU.",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Optional output root; defaults to outputs/latent_attacks.",
    )
    args = parser.parse_args()

    set_global_seed(args.seed)
    run_tag, manifest_path, diagnostics_dir = _resolve_vae_paths(args)
    manifest = _load_json(manifest_path)
    model_specs = _resolve_model_specs(args.models)
    collapsed_by_class = _load_collapsed_dims_for_run(
        diagnostics_dir=diagnostics_dir,
        manifest=manifest,
    )
    phase_name = "targeted_benign_pgd"
    if run_tag != "root_manifest":
        phase_name = f"targeted_benign_pgd_{_sanitize_phase_name(run_tag)}"

    run_logger = AttackRunLogger.create(
        phase_name=phase_name,
        seed=args.seed,
        config_snapshot=_config_snapshot(
            args,
            run_tag=run_tag,
            manifest_path=manifest_path,
            diagnostics_dir=diagnostics_dir,
            manifest=manifest,
            model_specs=model_specs,
        ),
        output_root=Path(args.output_root) if args.output_root else None,
    )

    router = AttackRouter(device=args.device)
    router.manifest = manifest
    mask = PerturbationMask.from_preprocessing_artifacts()
    protocol_validator = ProtocolValidator(router.scaler)
    split_test = load_split("test")
    detector = _fit_detector(
        router,
        args.device,
        collapsed_by_class=collapsed_by_class,
        max_samples_per_class=args.detector_max_samples,
        seed=args.seed,
    )

    uses_gmm = "gmm" in {
        part.strip().lower()
        for part in args.restart_strategy.replace(",", "+").split("+")
        if part.strip()
    }
    gmm_priors: dict[int, LatentGMMPrior] = {}
    if uses_gmm:
        for class_name in SOURCE_CLASSES:
            class_id = CLASS_TO_ID[class_name]
            print(f"[GMM] {class_name}: fitting/loading latent prior")
            gmm_priors[class_id] = fit_or_load_latent_gmm(
                router=router,
                class_id=class_id,
                split_name=args.gmm_split,
                n_components=args.gmm_components,
                max_fit_samples=args.gmm_fit_max_samples,
                seed=args.seed,
                device=args.device,
                force_refit=args.force_refit_gmm,
            )

    classifiers: dict[str, torch.nn.Module] = {}
    for spec in model_specs:
        ckpt_path = _REPO_ROOT / "models" / spec["checkpoint"]
        classifiers[spec["tag"]] = load_model(
            model_path=str(ckpt_path),
            num_features=len(FEATURE_NAMES),
            num_classes=len(CLASSES),
            device=args.device,
        )

    summary_rows: list[dict[str, Any]] = []
    per_class_rows: list[dict[str, Any]] = []
    per_sample_rows: list[dict[str, Any]] = []
    skipped_cells: list[dict[str, Any]] = []

    for spec in model_specs:
        classifier = classifiers[spec["tag"]]
        model_rows: list[dict[str, Any]] = []

        for class_name in SOURCE_CLASSES:
            class_id = CLASS_TO_ID[class_name]
            sample_idx = _select_correct_source_samples(
                x_test=split_test["X"],
                y_test=split_test["y_8"],
                classifier=classifier,
                device=args.device,
                class_id=class_id,
                max_samples=args.samples_per_class,
                prediction_batch_size=args.selection_batch_size,
            )
            if sample_idx.size == 0:
                skipped_cells.append(
                    {
                        "model": spec["label"],
                        "model_tag": spec["tag"],
                        "vae_run_tag": run_tag,
                        "class_name": class_name,
                        "reason": "no_correctly_classified_samples",
                    }
                )
                continue

            x_batch = torch.from_numpy(split_test["X"][sample_idx].astype(np.float32))
            y_batch = torch.from_numpy(split_test["y_8"][sample_idx].astype(np.int64))
            pred_before = predict_labels(classifier, x_batch, device=args.device)
            vae = router.get_vae(class_id)
            z_starts, restart_labels = _build_restart_initializers(
                vae=vae,
                x_batch=x_batch,
                gmm_prior=gmm_priors.get(class_id),
                epsilon=args.epsilon,
                num_restarts=args.num_restarts,
                restart_strategy=args.restart_strategy,
                seed=args.seed,
                class_id=class_id,
                device=args.device,
            )
            best = _run_targeted_pgd_restart_pool(
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
                z_starts=z_starts,
                restart_labels=restart_labels,
            )

            row = _row_from_best(
                vae_run_tag=run_tag,
                model_label=spec["label"],
                model_tag=spec["tag"],
                class_name=class_name,
                best=best,
            )
            model_rows.append(row)
            per_class_rows.append(row)
            per_sample_rows.extend(
                _build_per_sample_rows(
                    sample_indices=sample_idx,
                    vae_run_tag=run_tag,
                    model_label=spec["label"],
                    model_tag=spec["tag"],
                    class_name=class_name,
                    y_batch=y_batch,
                    pred_before=pred_before,
                    best=best,
                )
            )

            np.savez_compressed(
                run_logger.run_dir / f"{spec['tag']}_targeted_benign_pgd_{class_name}.npz",
                x_orig=x_batch.cpu().numpy(),
                x_adv=best["x_adv"].cpu().numpy(),
                y_true=y_batch.cpu().numpy(),
                y_pred_before=pred_before.cpu().numpy(),
                y_pred_after=best["pred_after"].cpu().numpy(),
                target_success=best["target_success"].cpu().numpy(),
                protocol_valid=best["protocol_valid"].cpu().numpy(),
                mask_valid=best["mask_valid"].cpu().numpy(),
                raw_g1g8_valid=best["raw_valid"].cpu().numpy(),
                joint_valid=best["joint_valid"].cpu().numpy(),
                z_orig=best["z_orig"].cpu().numpy(),
                z_adv=best["z_adv"].cpu().numpy(),
                selected_restart=best["selected_restart"].cpu().numpy(),
                restart_labels=np.asarray(restart_labels),
            )
            run_logger.log(json.dumps(row))

            print(
                f"{spec['label']:9s} {class_name:10s} "
                f"Target={_fmt_pct(row['benign_target_success_rate']):>8s} "
                f"JointTarget={_fmt_pct(row['benign_target_success_joint_rate']):>8s} "
                f"Joint={_fmt_pct(row['joint_validity_rate']):>8s} "
                f"L2={_fmt_float(row['mean_l2_input']):>8s}"
            )

        if model_rows:
            summary_rows.append(_aggregate_rows(model_rows))

    summary_rows.sort(key=lambda row: row["model"])
    per_class_rows.sort(key=lambda row: (row["model"], row["class_name"]))

    payload = {
        "vae_run_tag": run_tag,
        "vae_manifest_path": str(manifest_path),
        "vae_diagnostics_dir": str(diagnostics_dir),
        "summary": summary_rows,
        "per_class": per_class_rows,
        "per_sample": per_sample_rows,
        "skipped_cells": skipped_cells,
        "target_class": TARGET_CLASS_NAME,
        "restart_strategy": args.restart_strategy,
    }
    with open(run_logger.run_dir / "targeted_benign_results.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    _write_csv(run_logger.run_dir / "summary.csv", summary_rows, SUMMARY_COLUMNS)
    _write_csv(run_logger.run_dir / "per_class.csv", per_class_rows, PER_CLASS_COLUMNS)
    _write_csv(run_logger.run_dir / "per_sample_results.csv", per_sample_rows, PER_SAMPLE_COLUMNS)
    _write_csv(
        run_logger.run_dir / "skipped_cells.csv",
        skipped_cells,
        ["vae_run_tag", "model", "model_tag", "class_name", "reason"],
    )

    print()
    print("=== Targeted Benign Latent-PGD Summary ===")
    print(f"VAE run tag: {run_tag}")
    print(f"Output directory: {run_logger.run_dir}")
    for row in summary_rows:
        print(
            f"{row['model']:9s} "
            f"Target={_fmt_pct(row['benign_target_success_rate']):>8s} "
            f"JointTarget={_fmt_pct(row['benign_target_success_joint_rate']):>8s} "
            f"Joint={_fmt_pct(row['joint_validity_rate']):>8s} "
            f"IDSR={_fmt_pct(row['idsr']):>8s} "
            f"L2={_fmt_float(row['mean_l2_input']):>8s} "
            f"L2z={_fmt_float(row['mean_l2_latent']):>8s}"
        )


if __name__ == "__main__":
    main()
