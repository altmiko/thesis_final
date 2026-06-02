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
from attack.constrained_input_baselines import (  # noqa: E402
    VAEConstraintProjection,
    constrained_input_cw_attack,
    constrained_input_pgd_attack,
)
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
from attack.latent_pgd import classifier_logits  # noqa: E402
from attack.validator import validate_batch  # noqa: E402
from preprocessing.feature_groups import FEATURE_NAMES  # noqa: E402
from vae.config import CLASS_TO_ID, CLASSES  # noqa: E402


DEFAULT_VAE_RUN_TAG = "gaussian_anticollapse_beta05_freebits01_20260529_173512"
SAMPLES_PER_SOURCE_CLASS = 100
SELECTION_BATCH_SIZE = 8192
SOURCE_CLASSES = [name for name in CLASSES if name != "Benign"]
BENIGN_TARGET_CLASS = CLASS_TO_ID["Benign"]
MODEL_SPECS = [
    {"tag": "mlp", "label": "MLP", "checkpoint": "mlp_8class.pt"},
    {"tag": "cnn", "label": "CNN", "checkpoint": "cnn_8class.pt"},
    {"tag": "lstm", "label": "LSTM", "checkpoint": "lstm_8class.pt"},
    {"tag": "serial", "label": "CNN-LSTM", "checkpoint": "serial_8class.pt"},
    {"tag": "dualpath", "label": "DualPath", "checkpoint": "dualpath_8class.pt"},
]
ATTACK_ORDER = [
    "cinput-pgd",
    "cinput-cw",
    "cinput-pgd-target-benign",
    "cinput-cw-target-benign",
]
SUMMARY_COLUMNS = [
    "model",
    "attack",
    "attack_goal",
    "target_class",
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


def _attack_goal(attack_name: str) -> str:
    return "target-benign" if attack_name.endswith("-target-benign") else "untargeted"


def _is_targeted_benign_attack(attack_name: str) -> bool:
    return _attack_goal(attack_name) == "target-benign"


def _load_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


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
        diagnostics_dir = (_REPO_ROOT / "results" / "vae" / run_tag) if run_tag else manifest_path.parent
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
    *,
    args: argparse.Namespace,
    run_tag: str,
    manifest_path: Path,
    diagnostics_dir: Path,
    manifest: dict[str, Any],
    model_specs: list[dict[str, str]],
) -> dict[str, Any]:
    snapshot = phase0_config_snapshot(args.seed, args.device)
    snapshot.update(
        {
            "phase": "constrained_input_baselines",
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
                "selection_batch_size": int(args.selection_batch_size),
                "selection_rule": "up to N correctly classified test samples per non-benign source class; skip zero-available cells",
                "correctly_classified_only": True,
            },
            "attacks": {
                "selected": ATTACK_ORDER,
                "cinput-pgd": {
                    "epsilon": float(args.input_epsilon),
                    "alpha": float(args.input_alpha),
                    "num_steps": int(args.num_steps),
                    "random_start": bool(args.random_start),
                    "targeted": False,
                    "target_class": None,
                    "constraint_projection": "full+physics",
                },
                "cinput-pgd-target-benign": {
                    "epsilon": float(args.input_epsilon),
                    "alpha": float(args.input_alpha),
                    "num_steps": int(args.num_steps),
                    "random_start": bool(args.random_start),
                    "targeted": True,
                    "target_class": "Benign",
                    "constraint_projection": "full+physics",
                },
                "cinput-cw": {
                    "lambda_conf": float(args.lambda_conf),
                    "kappa": float(args.kappa),
                    "num_iterations": int(args.num_iterations),
                    "learning_rate": float(args.learning_rate),
                    "convergence_threshold": float(args.convergence_threshold),
                    "targeted": False,
                    "target_class": None,
                    "constraint_projection": "full+physics",
                },
                "cinput-cw-target-benign": {
                    "lambda_conf": float(args.lambda_conf),
                    "kappa": float(args.kappa),
                    "num_iterations": int(args.num_iterations),
                    "learning_rate": float(args.learning_rate),
                    "convergence_threshold": float(args.convergence_threshold),
                    "targeted": True,
                    "target_class": "Benign",
                    "constraint_projection": "full+physics",
                },
            },
            "evaluation_note": "Final samples are hard-projected by the differentiable constraint mechanic, not raw_postprocess-forced.",
        }
    )
    return snapshot


def _fit_detector(
    router: AttackRouter,
    device: str,
    *,
    collapsed_by_class: dict[int, list[int]],
) -> MahalanobisOutlierDetector:
    split_val = load_split("val")
    detector = MahalanobisOutlierDetector(latent_dim=16)
    for class_id, class_name in enumerate(CLASSES):
        print(f"[IDSR] fitting Mahalanobis detector for {class_name}")
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


def _masked_rate(mask: torch.Tensor) -> float:
    if mask.numel() == 0:
        return 0.0
    return float(mask.float().mean().item())


def _raw_g1g8_validity(x_adv: torch.Tensor, scaler: Any) -> torch.Tensor:
    x_np = x_adv.detach().cpu().numpy().astype(np.float64)
    x_raw = scaler.inverse_transform(x_np)
    result = validate_batch(x_raw, FEATURE_NAMES)
    return torch.from_numpy(result.overall_valid.astype(np.bool_))


def _evaluate(
    *,
    attack_name: str,
    class_id: int,
    x_batch: torch.Tensor,
    y_batch: torch.Tensor,
    vae: torch.nn.Module,
    classifier: torch.nn.Module,
    projection: VAEConstraintProjection,
    mask: PerturbationMask,
    protocol_validator: ProtocolValidator,
    detector: MahalanobisOutlierDetector,
    scaler: Any,
    device: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    targeted = _is_targeted_benign_attack(attack_name)
    target_class = BENIGN_TARGET_CLASS if targeted else -1
    target_class_name = CLASSES[target_class] if targeted else None

    if attack_name in {"cinput-pgd", "cinput-pgd-target-benign"}:
        x_adv, _metadata = constrained_input_pgd_attack(
            classifier=classifier,
            projection=projection,
            x_original=x_batch,
            y_true=y_batch,
            epsilon=args.input_epsilon,
            alpha=args.input_alpha,
            num_steps=args.num_steps,
            random_start=args.random_start,
            device=device,
            targeted=targeted,
            target_class=BENIGN_TARGET_CLASS,
        )
    elif attack_name in {"cinput-cw", "cinput-cw-target-benign"}:
        x_adv, _metadata = constrained_input_cw_attack(
            classifier=classifier,
            projection=projection,
            x_original=x_batch,
            y_true=y_batch,
            lambda_conf=args.lambda_conf,
            kappa=args.kappa,
            num_iterations=args.num_iterations,
            learning_rate=args.learning_rate,
            convergence_threshold=args.convergence_threshold,
            device=device,
            targeted=targeted,
            target_class=BENIGN_TARGET_CLASS,
        )
    else:
        raise KeyError(f"Unsupported attack: {attack_name}")

    with torch.no_grad():
        logits_after = classifier_logits(classifier, x_adv.to(device), device=device).cpu()
        pred_after = torch.argmax(logits_after, dim=1)
        if targeted:
            success_mask = pred_after == BENIGN_TARGET_CLASS
        else:
            success_mask = pred_after != y_batch.cpu()
        protocol_valid = protocol_validator.validate(x_adv, already_scaled=True).cpu()
        mask_compliance = mask.verify(x_adv.cpu(), x_batch.cpu())["all_compliant"].cpu()
        raw_valid = _raw_g1g8_validity(x_adv.cpu(), scaler)
        joint_valid = protocol_valid & mask_compliance & raw_valid
        invalid_mask = ~joint_valid
        z_reencoded, _ = vae.encode(x_adv.to(device))
        outlier_mask = detector.outlier_mask(class_id, z_reencoded.cpu()).cpu()
        input_l2 = torch.linalg.norm(x_adv.cpu() - x_batch.cpu(), dim=1)

    successful_joint = int((success_mask & joint_valid).sum().item())
    return {
        "class_name": CLASSES[class_id],
        "attack_goal": _attack_goal(attack_name),
        "target_class": target_class_name,
        "pred_after": pred_after.numpy().tolist(),
        "success_mask": success_mask.numpy().tolist(),
        "protocol_valid_mask": protocol_valid.numpy().tolist(),
        "mask_compliance_mask": mask_compliance.numpy().tolist(),
        "raw_g1g8_valid_mask": raw_valid.numpy().tolist(),
        "joint_valid_mask": joint_valid.numpy().tolist(),
        "selected_restart": [0 for _ in range(int(x_batch.shape[0]))],
        "n": int(x_batch.shape[0]),
        "asr_overall": _masked_rate(success_mask),
        "asr_valid_only": _conditional_rate(success_mask, joint_valid),
        "asr_invalid_only": _conditional_rate(success_mask, invalid_mask),
        "protocol_validity_rate": _masked_rate(protocol_valid),
        "mask_compliance_rate": _masked_rate(mask_compliance),
        "raw_g1g8_validity_rate": _masked_rate(raw_valid),
        "joint_validity_rate": _masked_rate(joint_valid),
        "idsr": _masked_rate(~outlier_mask),
        "mean_l2_input": float(input_l2.mean().item()),
        "mean_l2_latent": None,
        "valid_count": int(joint_valid.sum().item()),
        "invalid_count": int(invalid_mask.sum().item()),
        "successful_total_count": int(success_mask.sum().item()),
        "successful_valid_count": successful_joint,
        "successful_invalid_count": int((success_mask & invalid_mask).sum().item()),
        "successful_joint_valid_count": successful_joint,
        "selected_restart_mean": None,
        "restart_success_counts": None,
        "restart_labels": None,
    }


def _aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total_n = sum(row["n"] for row in rows)
    total_valid = sum(row["valid_count"] for row in rows)
    total_invalid = sum(row["invalid_count"] for row in rows)
    return {
        "attack_goal": rows[0].get("attack_goal") if rows else None,
        "target_class": rows[0].get("target_class") if rows else None,
        "n": int(total_n),
        "asr_overall": float(sum(row["successful_total_count"] for row in rows) / total_n),
        "asr_valid_only": (
            float(sum(row["successful_valid_count"] for row in rows) / total_valid)
            if total_valid > 0
            else 0.0
        ),
        "asr_invalid_only": (
            float(sum(row["successful_invalid_count"] for row in rows) / total_invalid)
            if total_invalid > 0
            else 0.0
        ),
        "protocol_validity_rate": float(
            sum(row["protocol_validity_rate"] * row["n"] for row in rows) / total_n
        ),
        "mask_compliance_rate": float(
            sum(row["mask_compliance_rate"] * row["n"] for row in rows) / total_n
        ),
        "joint_validity_rate": float(
            sum(row["joint_validity_rate"] * row["n"] for row in rows) / total_n
        ),
        "idsr": float(sum(row["idsr"] * row["n"] for row in rows) / total_n),
        "mean_l2_input": float(sum(row["mean_l2_input"] * row["n"] for row in rows) / total_n),
        "mean_l2_latent": None,
        "raw_g1g8_validity_rate": float(
            sum(row["raw_g1g8_validity_rate"] * row["n"] for row in rows) / total_n
        ),
        "successful_joint_valid_count": sum(
            int(row["successful_joint_valid_count"]) for row in rows
        ),
        "selected_restart_mean": None,
        "restart_success_counts": None,
        "restart_labels": None,
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


def _write_summary_md(path: Path, rows: list[dict[str, Any]], samples_per_class: int) -> None:
    headers = [
        "Model",
        "Attack",
        "Goal",
        "Target",
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
    ]
    lines = [
        "# Constraint-Augmented Input Attack Summary",
        "",
        f"Sample setting: {samples_per_class} correctly classified test samples per non-benign source class.",
        "Constraint set: full+physics. Final sample validity is measured, not raw_postprocess-forced.",
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
                    str(row.get("attack_goal") or ""),
                    str(row.get("target_class") or ""),
                    str(row["n"]),
                    _fmt_pct(row["asr_overall"]),
                    _fmt_pct(row["asr_valid_only"]),
                    _fmt_pct(row["asr_invalid_only"]),
                    _fmt_pct(row["protocol_validity_rate"]),
                    _fmt_pct(row["mask_compliance_rate"]),
                    _fmt_pct(row["raw_g1g8_validity_rate"]),
                    _fmt_pct(row["joint_validity_rate"]),
                    _fmt_pct(row["idsr"]),
                    _fmt_float(row["mean_l2_input"]),
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
    raw_valid_mask = np.asarray(metrics["raw_g1g8_valid_mask"], dtype=np.bool_)
    joint_valid_mask = np.asarray(metrics["joint_valid_mask"], dtype=np.bool_)
    attack_goal = str(metrics.get("attack_goal") or _attack_goal(attack_name))
    target_class = metrics.get("target_class")
    is_targeted = attack_goal == "target-benign"

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
                "target_success": bool(success_mask[pos]) if is_targeted else None,
                "selected_restart": 0,
                "attack_type": attack_name,
                "attack_goal": attack_goal,
                "target_class": target_class,
                "model_name": model_name,
                "model_tag": model_tag,
                "source_class": metrics["class_name"],
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Constraint-augmented input PGD/CW and target-Benign variants across 8-class models."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--vae-run-tag", default=DEFAULT_VAE_RUN_TAG)
    parser.add_argument("--vae-manifest", default=None)
    parser.add_argument("--vae-diagnostics-dir", default=None)
    parser.add_argument("--models", default="all", help="Comma-separated model tags or all")
    parser.add_argument("--samples-per-class", type=int, default=SAMPLES_PER_SOURCE_CLASS)
    parser.add_argument("--selection-batch-size", type=int, default=SELECTION_BATCH_SIZE)
    parser.add_argument("--input-epsilon", type=float, default=0.5)
    parser.add_argument("--input-alpha", type=float, default=0.05)
    parser.add_argument("--num-steps", type=int, default=40)
    parser.add_argument("--random-start", action="store_true", default=True)
    parser.add_argument("--no-random-start", dest="random_start", action="store_false")
    parser.add_argument("--lambda-conf", type=float, default=1.0)
    parser.add_argument("--kappa", type=float, default=0.0)
    parser.add_argument("--num-iterations", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--convergence-threshold", type=float, default=1e-5)
    args = parser.parse_args()

    if int(args.samples_per_class) < 1:
        raise ValueError("--samples-per-class must be >= 1")
    if int(args.selection_batch_size) < 1:
        raise ValueError("--selection-batch-size must be >= 1")

    model_specs = _resolve_model_specs(args.models)
    run_tag, manifest_path, diagnostics_dir = _resolve_vae_paths(args)
    manifest = _load_json(manifest_path)
    collapsed_by_class = _load_collapsed_dims_for_run(
        diagnostics_dir=diagnostics_dir,
        manifest=manifest,
    )

    set_global_seed(args.seed)
    run_logger = AttackRunLogger.create(
        phase_name="constrained_input_baselines",
        seed=args.seed,
        config_snapshot=_config_snapshot(
            args=args,
            run_tag=run_tag,
            manifest_path=manifest_path,
            diagnostics_dir=diagnostics_dir,
            manifest=manifest,
            model_specs=model_specs,
        ),
    )

    router = AttackRouter(device=args.device)
    router.manifest = manifest
    mask = PerturbationMask.from_preprocessing_artifacts()
    protocol_validator = ProtocolValidator(router.scaler)
    projection = VAEConstraintProjection(router.scaler, mask, enable_physics=True, device=args.device)
    split_test = load_split("test")
    detector = _fit_detector(router, args.device, collapsed_by_class=collapsed_by_class)

    classifiers: dict[str, torch.nn.Module] = {}
    for spec in model_specs:
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

    for spec in model_specs:
        print(f"[Model] {spec['label']}: selecting correctly classified samples")
        classifier = classifiers[spec["tag"]]
        selected_indices_by_class: dict[int, np.ndarray] = {}
        for class_name in SOURCE_CLASSES:
            class_id = CLASS_TO_ID[class_name]
            selected = _select_correct_source_samples(
                x_test=split_test["X"],
                y_test=split_test["y_8"],
                classifier=classifier,
                device=args.device,
                class_id=class_id,
                max_samples=args.samples_per_class,
                selection_batch_size=args.selection_batch_size,
            )
            selected_indices_by_class[class_id] = selected
            if selected.size == 0:
                skipped_cells.append(
                    {
                        "vae_run_tag": run_tag,
                        "model": spec["label"],
                        "model_tag": spec["tag"],
                        "class_name": class_name,
                        "reason": "no_correctly_classified_samples",
                    }
                )

        for attack_name in ATTACK_ORDER:
            attack_rows: list[dict[str, Any]] = []
            print(f"[Attack] {spec['label']} {attack_name}")
            for class_name in SOURCE_CLASSES:
                class_id = CLASS_TO_ID[class_name]
                sample_idx = selected_indices_by_class[class_id]
                if sample_idx.size == 0:
                    continue

                x_batch = torch.from_numpy(split_test["X"][sample_idx].astype(np.float32))
                y_batch = torch.from_numpy(split_test["y_8"][sample_idx].astype(np.int64))
                metrics = _evaluate(
                    attack_name=attack_name,
                    class_id=class_id,
                    x_batch=x_batch,
                    y_batch=y_batch,
                    vae=router.get_vae(class_id),
                    classifier=classifier,
                    projection=projection,
                    mask=mask,
                    protocol_validator=protocol_validator,
                    detector=detector,
                    scaler=router.scaler,
                    device=args.device,
                    args=args,
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
                        model_name=spec["label"],
                        model_tag=spec["tag"],
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
                        "attack_goal": _attack_goal(attack_name),
                        "target_class": "Benign" if _is_targeted_benign_attack(attack_name) else None,
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
                        "raw_g1g8_validity_rate": None,
                        "successful_joint_valid_count": None,
                        "selected_restart_mean": None,
                        "restart_success_counts": None,
                        "restart_labels": None,
                    }
                )
                continue

            summary = _aggregate_rows(attack_rows)
            summary["vae_run_tag"] = run_tag
            summary["model"] = spec["label"]
            summary["model_tag"] = spec["tag"]
            summary["attack"] = attack_name
            summary_rows.append(summary)

    summary_rows.sort(key=lambda row: (row["model"], ATTACK_ORDER.index(row["attack"])))
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
            "attack_goal",
            "target_class",
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
            "target_success",
            "selected_restart",
            "attack_type",
            "attack_goal",
            "target_class",
            "model_name",
            "model_tag",
            "source_class",
        ],
    )
    _write_csv(
        run_logger.run_dir / "skipped_cells.csv",
        skipped_cells,
        ["vae_run_tag", "model", "model_tag", "class_name", "reason"],
    )
    _write_summary_md(run_logger.run_dir / "summary.md", summary_rows, int(args.samples_per_class))

    print("=== Constraint-Augmented Input Attack Summary ===")
    print(f"VAE run tag: {run_tag}")
    print(f"Output directory: {run_logger.run_dir}")
    print()
    for row in summary_rows:
        print(
            f"{row['model']:9s} {row['attack']:11s} "
            f"ASR={_fmt_pct(row['asr_overall']):>8s} "
            f"ASR_valid={_fmt_pct(row['asr_valid_only']):>8s} "
            f"Proto={_fmt_pct(row['protocol_validity_rate']):>8s} "
            f"Mask={_fmt_pct(row['mask_compliance_rate']):>8s} "
            f"Raw={_fmt_pct(row['raw_g1g8_validity_rate']):>8s} "
            f"Joint={_fmt_pct(row['joint_validity_rate']):>8s} "
            f"IDSR={_fmt_pct(row['idsr']):>8s} "
            f"L2_in={_fmt_float(row['mean_l2_input']):>8s}"
        )


if __name__ == "__main__":
    main()
