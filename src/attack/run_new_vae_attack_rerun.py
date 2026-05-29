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

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = str(_REPO_ROOT / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from attack.adversarial_attacks import load_model  # noqa: E402
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
from attack.run_all_models_attack_rerun import (  # noqa: E402
    ATTACK_ORDER as BASE_ATTACK_ORDER,
    MODEL_SPECS,
    SOURCE_CLASSES,
    SUMMARY_COLUMNS,
    _aggregate_rows,
    _build_per_sample_rows,
    _evaluate_latent_attack,
    _fmt_float,
    _fmt_pct,
    _write_csv,
)
from preprocessing.feature_groups import FEATURE_NAMES  # noqa: E402
from vae.config import CLASS_TO_ID, CLASSES  # noqa: E402

DEFAULT_VAE_RUN_TAG = "gaussian_anticollapse_beta05_freebits01_20260529_173512"
DEFAULT_ATTACKS = ["latent-pgd", "latent-cw"]


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
    run_tag = str(args.vae_run_tag).strip()
    manifest_path = Path(args.vae_manifest) if args.vae_manifest else None
    diagnostics_dir = Path(args.vae_diagnostics_dir) if args.vae_diagnostics_dir else None

    if manifest_path is None:
        if not run_tag:
            raise ValueError("--vae-run-tag is required when --vae-manifest is not supplied")
        manifest_path = _REPO_ROOT / "results" / "vae" / run_tag / "vae_run_manifest.json"

    manifest_path = manifest_path.resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(f"VAE manifest not found: {manifest_path}")

    if diagnostics_dir is None:
        if run_tag:
            diagnostics_dir = _REPO_ROOT / "results" / "vae" / run_tag
        else:
            diagnostics_dir = manifest_path.parent

    diagnostics_dir = diagnostics_dir.resolve()
    if not diagnostics_dir.exists():
        raise FileNotFoundError(f"VAE diagnostics directory not found: {diagnostics_dir}")

    if not run_tag:
        run_tag = manifest_path.parent.name

    return run_tag, manifest_path, diagnostics_dir


def _resolve_model_specs(models_arg: str) -> list[dict[str, str]]:
    requested = [item.strip().lower() for item in models_arg.split(",") if item.strip()]
    if not requested or requested == ["all"]:
        return list(MODEL_SPECS)

    by_tag = {str(spec["tag"]): spec for spec in MODEL_SPECS}
    unknown = [tag for tag in requested if tag not in by_tag]
    if unknown:
        raise ValueError(f"Unknown model tags {unknown}; valid tags: {sorted(by_tag)} or all")
    return [by_tag[tag] for tag in requested]


def _resolve_attacks(attacks_arg: str) -> list[str]:
    requested = [item.strip().lower() for item in attacks_arg.split(",") if item.strip()]
    if not requested or requested == ["latent"]:
        return list(DEFAULT_ATTACKS)

    valid = [attack for attack in BASE_ATTACK_ORDER if attack.startswith("latent-")]
    unknown = [attack for attack in requested if attack not in valid]
    if unknown:
        raise ValueError(f"Unknown latent attacks {unknown}; valid attacks: {valid}")
    return requested


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


def _fit_detector(
    *,
    router: AttackRouter,
    device: str,
    collapsed_by_class: dict[int, list[int]],
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
        vae = router.get_vae(class_id)
        z_mu = encode_dataset_mu(vae, ds, device=device)
        detector.fit(class_id, z_mu.cpu(), collapsed_dims=collapsed_by_class[class_id])
    return detector


def _select_correct_source_samples_batched(
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


def _config_snapshot(
    *,
    args: argparse.Namespace,
    run_tag: str,
    manifest_path: Path,
    diagnostics_dir: Path,
    model_specs: list[dict[str, str]],
    attack_order: list[str],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    snapshot = phase0_config_snapshot(args.seed, args.device)
    snapshot.update(
        {
            "phase": "new_vae_attack_rerun",
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
            "models": model_specs,
            "sampling": {
                "source_classes": SOURCE_CLASSES,
                "samples_per_source_class": int(args.samples_per_class),
                "selection_rule": "up to N correctly classified test samples per non-benign source class; skip zero-available cells",
                "correctly_classified_only": True,
                "prediction_batch_size": int(args.selection_batch_size),
            },
            "attacks": {
                "selected": attack_order,
                "latent-pgd": {
                    "epsilon": float(args.epsilon),
                    "alpha": float(args.alpha),
                    "num_steps": int(args.num_steps),
                    "random_start": bool(args.random_start),
                },
                "latent-cw": {
                    "lambda_conf": float(args.lambda_conf),
                    "kappa": float(args.kappa),
                    "num_iterations": int(args.num_iterations),
                    "learning_rate": float(args.learning_rate),
                    "convergence_threshold": float(args.convergence_threshold),
                },
            },
        }
    )
    return snapshot


def _write_summary_md(path: Path, rows: list[dict[str, Any]], run_tag: str, samples_per_class: int) -> None:
    headers = [
        "Model",
        "Attack",
        "N",
        "ASR",
        "ASR Valid",
        "ASR Invalid",
        "Protocol",
        "Mask",
        "Joint Valid",
        "IDSR",
        "L2 Input",
        "L2 Latent",
    ]
    lines = [
        "# New VAE Attack Rerun Summary",
        "",
        f"VAE run tag: `{run_tag}`",
        f"Sample setting: {samples_per_class} correctly classified test samples per non-benign source class.",
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
                    _fmt_pct(row["joint_validity_rate"]),
                    _fmt_pct(row["idsr"]),
                    _fmt_float(row["mean_l2_input"]),
                    _fmt_float(row["mean_l2_latent"]),
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rerun latent attacks using a tagged VAE rerun manifest."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--vae-run-tag", default=DEFAULT_VAE_RUN_TAG)
    parser.add_argument("--vae-manifest", default=None)
    parser.add_argument("--vae-diagnostics-dir", default=None)
    parser.add_argument("--models", default="all", help="Comma-separated model tags or all")
    parser.add_argument("--attacks", default=",".join(DEFAULT_ATTACKS))
    parser.add_argument("--samples-per-class", type=int, default=100)
    parser.add_argument("--selection-batch-size", type=int, default=8192)
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
    run_tag, manifest_path, diagnostics_dir = _resolve_vae_paths(args)
    manifest = _load_json(manifest_path)
    model_specs = _resolve_model_specs(args.models)
    attack_order = _resolve_attacks(args.attacks)
    collapsed_by_class = _load_collapsed_dims_for_run(
        diagnostics_dir=diagnostics_dir,
        manifest=manifest,
    )

    phase_name = f"new_vae_attacks_{_sanitize_phase_name(run_tag)}"
    run_logger = AttackRunLogger.create(
        phase_name=phase_name,
        seed=args.seed,
        config_snapshot=_config_snapshot(
            args=args,
            run_tag=run_tag,
            manifest_path=manifest_path,
            diagnostics_dir=diagnostics_dir,
            model_specs=model_specs,
            attack_order=attack_order,
            manifest=manifest,
        ),
    )

    router = AttackRouter(device=args.device)
    router.manifest = manifest
    mask = PerturbationMask.from_preprocessing_artifacts()
    protocol_validator = ProtocolValidator(router.scaler)
    split_test = load_split("test")
    detector = _fit_detector(
        router=router,
        device=args.device,
        collapsed_by_class=collapsed_by_class,
    )

    per_class_rows: list[dict[str, Any]] = []
    per_sample_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    skipped_cells: list[dict[str, Any]] = []

    for spec in model_specs:
        ckpt_path = _REPO_ROOT / "models" / str(spec["checkpoint"])
        classifier = load_model(
            model_path=str(ckpt_path),
            num_features=len(FEATURE_NAMES),
            num_classes=len(CLASSES),
            device=args.device,
        )
        selected_indices_by_class: dict[int, np.ndarray] = {}
        for class_name in SOURCE_CLASSES:
            class_id = CLASS_TO_ID[class_name]
            selected_indices_by_class[class_id] = _select_correct_source_samples_batched(
                x_test=split_test["X"],
                y_test=split_test["y_8"],
                classifier=classifier,
                device=args.device,
                class_id=class_id,
                max_samples=args.samples_per_class,
                prediction_batch_size=args.selection_batch_size,
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

        for attack_name in attack_order:
            attack_rows: list[dict[str, Any]] = []
            for class_name in SOURCE_CLASSES:
                class_id = CLASS_TO_ID[class_name]
                sample_idx = selected_indices_by_class[class_id]
                if sample_idx.size == 0:
                    continue

                x_batch = torch.from_numpy(split_test["X"][sample_idx].astype(np.float32))
                y_batch = torch.from_numpy(split_test["y_8"][sample_idx].astype(np.int64))
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

                metrics["vae_run_tag"] = run_tag
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
                        model_name=str(spec["label"]),
                        model_tag=str(spec["tag"]),
                        attack_name=attack_name,
                    )
                )
                run_logger.log(json.dumps(metrics))

            if not attack_rows:
                summary_rows.append(
                    {
                        "vae_run_tag": run_tag,
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
            summary["vae_run_tag"] = run_tag
            summary["model"] = spec["label"]
            summary["model_tag"] = spec["tag"]
            summary["attack"] = attack_name
            summary_rows.append(summary)

        del classifier
        if torch.cuda.is_available() and str(args.device).startswith("cuda"):
            torch.cuda.empty_cache()

    summary_rows.sort(key=lambda row: (row["model"], attack_order.index(row["attack"])))
    per_class_rows.sort(key=lambda row: (row["model"], row["attack"], row["class_name"]))

    payload = {
        "vae_run_tag": run_tag,
        "summary": summary_rows,
        "per_class": per_class_rows,
        "per_sample": per_sample_rows,
        "skipped_cells": skipped_cells,
    }
    with open(run_logger.run_dir / "all_results.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    _write_csv(run_logger.run_dir / "summary.csv", summary_rows, ["vae_run_tag", "model_tag"] + SUMMARY_COLUMNS)
    _write_csv(
        run_logger.run_dir / "per_class.csv",
        per_class_rows,
        [
            "vae_run_tag",
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
            "success",
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
    _write_summary_md(
        run_logger.run_dir / "summary.md",
        summary_rows,
        run_tag,
        int(args.samples_per_class),
    )

    print("=== New VAE Attack Rerun Summary ===")
    print(f"VAE run tag: {run_tag}")
    print(f"Output directory: {run_logger.run_dir}")
    print()
    for row in summary_rows:
        print(
            f"{row['model']:9s} {row['attack']:10s} "
            f"ASR={_fmt_pct(row['asr_overall']):>8s} "
            f"ASR_valid={_fmt_pct(row['asr_valid_only']):>8s} "
            f"Proto={_fmt_pct(row['protocol_validity_rate']):>8s} "
            f"Mask={_fmt_pct(row['mask_compliance_rate']):>8s} "
            f"Joint={_fmt_pct(row['joint_validity_rate']):>8s} "
            f"IDSR={_fmt_pct(row['idsr']):>8s} "
            f"L2_in={_fmt_float(row['mean_l2_input']):>8s} "
            f"L2_z={_fmt_float(row['mean_l2_latent']):>8s}"
        )


if __name__ == "__main__":
    main()
