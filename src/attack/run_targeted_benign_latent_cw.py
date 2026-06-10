from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from attack.adversarial_attacks import load_model  # noqa: E402
from attack.latent_cw import latent_cw_attack  # noqa: E402
from attack.latent_gmm import LatentGMMPrior, fit_or_load_latent_gmm  # noqa: E402
from attack.latent_infra import (  # noqa: E402
    AttackRouter,
    AttackRunLogger,
    PerturbationMask,
    ProtocolValidator,
    load_split,
    phase0_config_snapshot,
    predict_labels,
    set_global_seed,
)
from attack import run_targeted_benign_latent_pgd as targeted_pgd  # noqa: E402
from preprocessing.feature_groups import FEATURE_NAMES  # noqa: E402
from vae.config import CLASS_TO_ID, CLASSES  # noqa: E402

GAUSSIAN_RUN_TAG = "gaussian_anticollapse_beta05_freebits01_20260529_173512"
TARGET_CLASS_ID = CLASS_TO_ID["Benign"]
TARGET_CLASS_NAME = "Benign"
SOURCE_CLASSES = [name for name in CLASSES if name != TARGET_CLASS_NAME]
MODEL_SPECS = targeted_pgd.MODEL_SPECS

SUMMARY_COLUMNS = [
    "vae_run_tag",
    "model",
    "model_tag",
    "attack",
    "target_class",
    "n",
    "ASR_raw",
    "ASR_valid",
    "mahalanobis_id_rate",
    "he_idsr",
    "protocol_validity_rate",
    "mask_compliance_rate",
    "raw_g1g8_validity_rate",
    "joint_validity_rate",
    "mean_l2_input",
    "mean_l2_latent",
    "mean_selected_restart",
    "benign_margin_mean",
    "target_confidence_mean",
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
    "mahalanobis_outlier",
    "selected_restart",
    "benign_logit",
    "max_other_logit",
    "benign_margin",
    "target_confidence",
]


def _run_restart_pool(
    *,
    class_id: int,
    x_batch: torch.Tensor,
    y_batch: torch.Tensor,
    vae: torch.nn.Module,
    classifier: torch.nn.Module,
    mask: PerturbationMask,
    protocol_validator: ProtocolValidator,
    detector: Any,
    scaler: Any,
    device: str,
    lambda_conf: float,
    kappa: float,
    num_iterations: int,
    learning_rate: float,
    convergence_threshold: float,
    z_starts: list[torch.Tensor],
    restart_labels: list[str],
) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    restart_target_success_counts: list[int] = []
    z_orig_ref: torch.Tensor | None = None

    for restart_idx, z_start in enumerate(z_starts):
        x_adv, z_adv, metadata = latent_cw_attack(
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
            targeted=True,
            target_class=TARGET_CLASS_ID,
            num_restarts=1,
            restart_strategy=restart_labels[restart_idx],
            z_initializers=z_start.unsqueeze(0),
        )
        z_orig = metadata["z_orig"]
        if not isinstance(z_orig, torch.Tensor):
            raise TypeError("latent_cw_attack metadata['z_orig'] must be a tensor")
        z_orig_ref = z_orig.detach().cpu()

        metrics = targeted_pgd._candidate_metrics(
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
        restart_target_success_counts.append(
            int(metrics["target_success"].sum().item())
        )

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
        improved |= (
            (candidate_success == current_success)
            & candidate_joint
            & (~current_joint)
        )
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
        raise RuntimeError("No targeted CW restart candidates were evaluated")

    best["z_orig"] = z_orig_ref
    best["restart_target_success_counts"] = restart_target_success_counts
    return best


def _rate(mask: torch.Tensor) -> float:
    return float(mask.float().mean().item()) if mask.numel() else 0.0


def _row_from_best(
    *,
    run_tag: str,
    model_label: str,
    model_tag: str,
    class_name: str,
    best: dict[str, Any],
) -> dict[str, Any]:
    success = best["target_success"]
    joint_valid = best["joint_valid"]
    in_distribution = ~best["outlier_mask"]
    return {
        "vae_run_tag": run_tag,
        "model": model_label,
        "model_tag": model_tag,
        "attack": "targeted-benign-latent-cw",
        "target_class": TARGET_CLASS_NAME,
        "class_name": class_name,
        "n": int(success.numel()),
        "ASR_raw": _rate(success),
        "ASR_valid": _rate(success & joint_valid),
        "mahalanobis_id_rate": _rate(in_distribution),
        "he_idsr": _rate(success & in_distribution),
        "protocol_validity_rate": _rate(best["protocol_valid"]),
        "mask_compliance_rate": _rate(best["mask_valid"]),
        "raw_g1g8_validity_rate": _rate(best["raw_valid"]),
        "joint_validity_rate": _rate(joint_valid),
        "mean_l2_input": float(best["input_l2"].float().mean().item()),
        "mean_l2_latent": float(best["latent_l2"].float().mean().item()),
        "mean_selected_restart": float(
            best["selected_restart"].float().mean().item()
        ),
        "benign_margin_mean": float(best["benign_margin"].float().mean().item()),
        "target_confidence_mean": float(
            best["target_confidence"].float().mean().item()
        ),
        "restart_target_success_counts": json.dumps(
            best["restart_target_success_counts"]
        ),
    }


def _aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total_n = sum(int(row["n"]) for row in rows)
    if total_n == 0:
        raise ValueError("Cannot aggregate empty rows")

    def weighted(key: str) -> float:
        return float(
            sum(float(row[key]) * int(row["n"]) for row in rows) / total_n
        )

    return {
        "vae_run_tag": rows[0]["vae_run_tag"],
        "model": rows[0]["model"],
        "model_tag": rows[0]["model_tag"],
        "attack": "targeted-benign-latent-cw",
        "target_class": TARGET_CLASS_NAME,
        "n": total_n,
        **{
            key: weighted(key)
            for key in SUMMARY_COLUMNS
            if key
            not in {
                "vae_run_tag",
                "model",
                "model_tag",
                "attack",
                "target_class",
                "n",
            }
        },
    }


def _per_sample_rows(
    *,
    sample_indices: np.ndarray,
    run_tag: str,
    model_label: str,
    model_tag: str,
    class_name: str,
    y_batch: torch.Tensor,
    pred_before: torch.Tensor,
    best: dict[str, Any],
) -> list[dict[str, Any]]:
    arrays = {
        "y_true": y_batch.cpu().numpy(),
        "pred_before": pred_before.cpu().numpy(),
        "pred_after": best["pred_after"].cpu().numpy(),
        "target_success": best["target_success"].cpu().numpy(),
        "protocol_valid": best["protocol_valid"].cpu().numpy(),
        "mask_valid": best["mask_valid"].cpu().numpy(),
        "raw_valid": best["raw_valid"].cpu().numpy(),
        "joint_valid": best["joint_valid"].cpu().numpy(),
        "outlier": best["outlier_mask"].cpu().numpy(),
        "selected_restart": best["selected_restart"].cpu().numpy(),
        "benign_logit": best["benign_logit"].cpu().numpy(),
        "max_other_logit": best["max_other_logit"].cpu().numpy(),
        "benign_margin": best["benign_margin"].cpu().numpy(),
        "target_confidence": best["target_confidence"].cpu().numpy(),
    }

    rows: list[dict[str, Any]] = []
    for pos, sample_id in enumerate(sample_indices.tolist()):
        rows.append(
            {
                "sample_id": int(sample_id),
                "vae_run_tag": run_tag,
                "model": model_label,
                "model_tag": model_tag,
                "source_class": class_name,
                "true_label": CLASSES[int(arrays["y_true"][pos])],
                "pred_before": CLASSES[int(arrays["pred_before"][pos])],
                "pred_after": CLASSES[int(arrays["pred_after"][pos])],
                "target_class": TARGET_CLASS_NAME,
                "target_success": bool(arrays["target_success"][pos]),
                "protocol_valid": bool(arrays["protocol_valid"][pos]),
                "mask_valid": bool(arrays["mask_valid"][pos]),
                "raw_g1g8_valid": bool(arrays["raw_valid"][pos]),
                "joint_valid": bool(arrays["joint_valid"][pos]),
                "mahalanobis_outlier": bool(arrays["outlier"][pos]),
                "selected_restart": int(arrays["selected_restart"][pos]),
                "benign_logit": float(arrays["benign_logit"][pos]),
                "max_other_logit": float(arrays["max_other_logit"][pos]),
                "benign_margin": float(arrays["benign_margin"][pos]),
                "target_confidence": float(arrays["target_confidence"][pos]),
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


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
            "phase": "targeted_benign_latent_cw",
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
            "target": {
                "class_id": TARGET_CLASS_ID,
                "class_name": TARGET_CLASS_NAME,
            },
            "models": model_specs,
            "sampling": {
                "source_classes": SOURCE_CLASSES,
                "samples_per_source_class": int(args.samples_per_class),
                "correctly_classified_only": True,
            },
            "attack": {
                "name": "targeted-benign-latent-cw",
                "lambda_conf": float(args.lambda_conf),
                "kappa": float(args.kappa),
                "num_iterations": int(args.num_iterations),
                "learning_rate": float(args.learning_rate),
                "convergence_threshold": float(args.convergence_threshold),
                "restart_radius": float(args.restart_radius),
                "num_restarts": int(args.num_restarts),
                "restart_strategy": str(args.restart_strategy),
                "targeted": True,
                "target_class": TARGET_CLASS_ID,
                "selection_policy": (
                    "target success, then joint validity, then lower input L2"
                ),
            },
            "gmm": {
                "split": str(args.gmm_split),
                "n_components": int(args.gmm_components),
                "max_fit_samples": int(args.gmm_fit_max_samples),
            },
            "idsr_detector": {
                "max_samples_per_class": args.detector_max_samples,
            },
            "selection_batch_size": int(args.selection_batch_size),
        }
    )
    return snapshot


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run targeted-to-Benign latent CW with GMM-seeded restarts."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--vae-run-tag", default=GAUSSIAN_RUN_TAG)
    parser.add_argument("--vae-manifest", default=None)
    parser.add_argument("--vae-diagnostics-dir", default=None)
    parser.add_argument("--models", default="all")
    parser.add_argument("--samples-per-class", type=int, default=100)
    parser.add_argument("--lambda-conf", type=float, default=1.0)
    parser.add_argument("--kappa", type=float, default=0.0)
    parser.add_argument("--num-iterations", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--convergence-threshold", type=float, default=1e-5)
    parser.add_argument("--restart-radius", type=float, default=0.5)
    parser.add_argument("--num-restarts", type=int, default=5)
    parser.add_argument("--restart-strategy", default="encoded+jitter+gmm")
    parser.add_argument("--gmm-split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--gmm-components", type=int, default=5)
    parser.add_argument("--gmm-fit-max-samples", type=int, default=50000)
    parser.add_argument("--force-refit-gmm", action="store_true")
    parser.add_argument("--detector-max-samples", type=int, default=None)
    parser.add_argument("--selection-batch-size", type=int, default=8192)
    parser.add_argument("--output-root", default=None)
    args = parser.parse_args()

    set_global_seed(args.seed)
    run_tag, manifest_path, diagnostics_dir = targeted_pgd._resolve_vae_paths(args)
    manifest = targeted_pgd._load_json(manifest_path)
    model_specs = targeted_pgd._resolve_model_specs(args.models)
    collapsed_by_class = targeted_pgd._load_collapsed_dims_for_run(
        diagnostics_dir=diagnostics_dir,
        manifest=manifest,
    )

    run_logger = AttackRunLogger.create(
        phase_name=f"targeted_benign_cw_{targeted_pgd._sanitize_phase_name(run_tag)}",
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
    detector = targeted_pgd._fit_detector(
        router,
        args.device,
        collapsed_by_class=collapsed_by_class,
        max_samples_per_class=args.detector_max_samples,
        seed=args.seed,
    )

    gmm_priors: dict[int, LatentGMMPrior] = {}
    if "gmm" in args.restart_strategy.lower():
        for class_name in SOURCE_CLASSES:
            class_id = CLASS_TO_ID[class_name]
            print(f"[GMM] {class_name}: fitting/loading latent prior", flush=True)
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

    classifiers = {
        spec["tag"]: load_model(
            model_path=str(REPO_ROOT / "models" / spec["checkpoint"]),
            num_features=len(FEATURE_NAMES),
            num_classes=len(CLASSES),
            device=args.device,
        )
        for spec in model_specs
    }

    summary_rows: list[dict[str, Any]] = []
    per_class_rows: list[dict[str, Any]] = []
    per_sample_rows: list[dict[str, Any]] = []
    skipped_cells: list[dict[str, Any]] = []

    for spec in model_specs:
        classifier = classifiers[spec["tag"]]
        model_rows: list[dict[str, Any]] = []
        for class_name in SOURCE_CLASSES:
            class_id = CLASS_TO_ID[class_name]
            sample_idx = targeted_pgd._select_correct_source_samples(
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
                        "vae_run_tag": run_tag,
                        "model": spec["label"],
                        "model_tag": spec["tag"],
                        "class_name": class_name,
                        "reason": "no_correctly_classified_samples",
                    }
                )
                continue

            print(
                f"[CW] {spec['label']} {class_name} N={len(sample_idx)}",
                flush=True,
            )
            x_batch = torch.from_numpy(
                split_test["X"][sample_idx].astype(np.float32)
            )
            y_batch = torch.from_numpy(
                split_test["y_8"][sample_idx].astype(np.int64)
            )
            pred_before = predict_labels(classifier, x_batch, device=args.device)
            vae = router.get_vae(class_id)
            z_starts, restart_labels = targeted_pgd._build_restart_initializers(
                vae=vae,
                x_batch=x_batch,
                gmm_prior=gmm_priors.get(class_id),
                epsilon=args.restart_radius,
                num_restarts=args.num_restarts,
                restart_strategy=args.restart_strategy,
                seed=args.seed,
                class_id=class_id,
                device=args.device,
            )
            best = _run_restart_pool(
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
                lambda_conf=args.lambda_conf,
                kappa=args.kappa,
                num_iterations=args.num_iterations,
                learning_rate=args.learning_rate,
                convergence_threshold=args.convergence_threshold,
                z_starts=z_starts,
                restart_labels=restart_labels,
            )
            row = _row_from_best(
                run_tag=run_tag,
                model_label=spec["label"],
                model_tag=spec["tag"],
                class_name=class_name,
                best=best,
            )
            model_rows.append(row)
            per_class_rows.append(row)
            per_sample_rows.extend(
                _per_sample_rows(
                    sample_indices=sample_idx,
                    run_tag=run_tag,
                    model_label=spec["label"],
                    model_tag=spec["tag"],
                    class_name=class_name,
                    y_batch=y_batch,
                    pred_before=pred_before,
                    best=best,
                )
            )

            np.savez_compressed(
                run_logger.run_dir
                / f"{spec['tag']}_targeted_benign_cw_{class_name}.npz",
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
                mahalanobis_outlier=best["outlier_mask"].cpu().numpy(),
                z_orig=best["z_orig"].cpu().numpy(),
                z_adv=best["z_adv"].cpu().numpy(),
                selected_restart=best["selected_restart"].cpu().numpy(),
                restart_labels=np.asarray(restart_labels),
            )
            run_logger.log(json.dumps(row))
            print(
                f"  ASR={row['ASR_raw'] * 100:6.2f}% "
                f"ASR_valid={row['ASR_valid'] * 100:6.2f}% "
                f"He-IDSR={row['he_idsr'] * 100:6.2f}%",
                flush=True,
            )

        if model_rows:
            summary_rows.append(_aggregate_rows(model_rows))

    summary_rows.sort(key=lambda row: row["model"])
    per_class_rows.sort(key=lambda row: (row["model"], row["class_name"]))

    payload = {
        "vae_run_tag": run_tag,
        "summary": summary_rows,
        "per_class": per_class_rows,
        "per_sample": per_sample_rows,
        "skipped_cells": skipped_cells,
    }
    with open(
        run_logger.run_dir / "targeted_benign_cw_results.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(payload, f, indent=2)
    _write_csv(run_logger.run_dir / "summary.csv", summary_rows, SUMMARY_COLUMNS)
    _write_csv(
        run_logger.run_dir / "per_class.csv",
        per_class_rows,
        PER_CLASS_COLUMNS,
    )
    _write_csv(
        run_logger.run_dir / "per_sample_results.csv",
        per_sample_rows,
        PER_SAMPLE_COLUMNS,
    )
    _write_csv(
        run_logger.run_dir / "skipped_cells.csv",
        skipped_cells,
        ["vae_run_tag", "model", "model_tag", "class_name", "reason"],
    )

    print()
    print("=== Targeted Benign Latent-CW Summary ===")
    print(f"Output directory: {run_logger.run_dir}")
    for row in summary_rows:
        print(
            f"{row['model']:9s} "
            f"ASR={row['ASR_raw'] * 100:6.2f}% "
            f"ASR_valid={row['ASR_valid'] * 100:6.2f}% "
            f"ID={row['mahalanobis_id_rate'] * 100:6.2f}% "
            f"He-IDSR={row['he_idsr'] * 100:6.2f}%"
        )


if __name__ == "__main__":
    main()
