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
from preprocessing.feature_groups import FEATURE_NAMES  # noqa: E402
from vae.config import CLASS_TO_ID, CLASSES  # noqa: E402

SAMPLES_PER_SOURCE_CLASS = 100
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
) -> np.ndarray:
    class_indices = np.where(y_test == class_id)[0]
    if len(class_indices) == 0:
        raise RuntimeError(f"No test samples found for class_id={class_id}")

    preds = predict_labels(
        classifier,
        torch.from_numpy(x_test[class_indices].astype(np.float32)),
        device=device,
    ).numpy()
    keep = class_indices[preds == class_id]
    return keep[:max_samples]


def _conditional_rate(numerator_mask: torch.Tensor, denominator_mask: torch.Tensor) -> float:
    denom = int(denominator_mask.sum().item())
    if denom == 0:
        return 0.0
    return float((numerator_mask & denominator_mask).float().sum().item() / denom)


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
) -> dict[str, Any]:
    if attack_name == "latent-pgd":
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
            random_start=random_start,
            device=device,
        )
    elif attack_name == "latent-cw":
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
        )
    else:
        raise KeyError(f"Unsupported latent attack: {attack_name}")

    with torch.no_grad():
        logits_after = classifier_logits(classifier, x_adv.to(device), device=device).cpu()
        pred_after = torch.argmax(logits_after, dim=1)
        success_mask = pred_after != y_batch.cpu()
        protocol_valid = protocol_validator.validate(x_adv, already_scaled=True).cpu()
        mask_compliance = mask.verify(x_adv.cpu(), x_batch.cpu())["all_compliant"].cpu()
        joint_valid = protocol_valid & mask_compliance
        invalid_mask = ~joint_valid
        z_reencoded, _ = vae.encode(x_adv.to(device))
        outlier_mask = detector.outlier_mask(class_id, z_reencoded.cpu()).cpu()
        input_l2 = torch.linalg.norm(x_adv.cpu() - x_batch.cpu(), dim=1)
        latent_l2 = torch.linalg.norm(z_adv.cpu() - metadata["z_orig"].cpu(), dim=1)

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
        "idsr": float((~outlier_mask).float().mean().item()),
        "mean_l2_input": float(input_l2.mean().item()),
        "mean_l2_latent": float(latent_l2.mean().item()),
        "valid_count": int(joint_valid.sum().item()),
        "invalid_count": int(invalid_mask.sum().item()),
        "successful_total_count": int(success_mask.sum().item()),
        "successful_valid_count": int((success_mask & joint_valid).sum().item()),
        "successful_invalid_count": int((success_mask & invalid_mask).sum().item()),
    }


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
    }


def _aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total_n = sum(row["n"] for row in rows)
    total_valid = sum(row["valid_count"] for row in rows)
    total_invalid = sum(row["invalid_count"] for row in rows)

    latent_l2_rows = [row for row in rows if row["mean_l2_latent"] is not None]
    idsr_rows = [row for row in rows if row["idsr"] is not None]

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
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        if rows:
            writer.writerows(rows)


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
        "Joint Valid",
        "IDSR",
        "L2 Input",
        "L2 Latent",
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
                    _fmt_pct(row["joint_validity_rate"]),
                    _fmt_pct(row["idsr"]),
                    _fmt_float(row["mean_l2_input"]),
                    _fmt_float(row["mean_l2_latent"]),
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

    rows: list[dict[str, Any]] = []
    for pos, sample_id in enumerate(sample_indices.tolist()):
        rows.append(
            {
                "sample_id": int(sample_id),
                "true_label": CLASSES[int(y_true_np[pos])],
                "predicted_label": CLASSES[int(pred_after[pos])],
                "protocol_valid": bool(protocol_valid_mask[pos]),
                "mask_valid": bool(mask_compliance_mask[pos]),
                "success": bool(success_mask[pos]),
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
            f"IDSR={_fmt_pct(row['idsr']) if row['idsr'] is not None else 'NA':>7s} "
            f"L2_in={_fmt_float(row['mean_l2_input']):>8s} "
            f"L2_z={_fmt_float(row['mean_l2_latent']):>8s}"
        )


if __name__ == "__main__":
    main()
