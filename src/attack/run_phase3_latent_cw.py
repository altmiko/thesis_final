from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = str(_REPO_ROOT / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

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
from attack.latent_pgd import classifier_logits  # noqa: E402
from vae.config import CLASS_TO_ID, CLASSES  # noqa: E402

SAMPLES_PER_SOURCE_CLASS = 100
SOURCE_CLASSES = [name for name in CLASSES if name != "Benign"]


def _phase3_config_snapshot(seed: int, device: str) -> dict:
    snapshot = phase0_config_snapshot(seed, device)
    snapshot.update(
        {
            "phase": "phase3",
            "split": "test",
            "sampling": {
                "source_classes": SOURCE_CLASSES,
                "samples_per_source_class": SAMPLES_PER_SOURCE_CLASS,
                "correctly_classified_only": True,
            },
            "classifier": "mlp-3l",
            "attack": {
                "name": "latent-cw",
                "lambda_conf": 1.0,
                "kappa": 0.0,
                "num_iterations": 200,
                "learning_rate": 0.01,
                "convergence_threshold": 1e-5,
            },
        }
    )
    return snapshot


def _to_float(value: torch.Tensor | float) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)


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
    if len(keep) < max_samples:
        raise RuntimeError(
            f"Need {max_samples} correctly classified samples for {CLASSES[class_id]}, found {len(keep)}"
        )
    return keep[:max_samples]


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


def _run_zero_budget_check(
    *,
    router: AttackRouter,
    mask: PerturbationMask,
    x_test: np.ndarray,
    y_test: np.ndarray,
    classifier: torch.nn.Module,
    device: str,
) -> dict:
    class_id = CLASS_TO_ID[SOURCE_CLASSES[0]]
    selected = _select_correct_source_samples(
        x_test=x_test,
        y_test=y_test,
        classifier=classifier,
        device=device,
        class_id=class_id,
        max_samples=1,
    )

    x_batch = torch.from_numpy(x_test[selected].astype(np.float32))
    y_batch = torch.from_numpy(y_test[selected].astype(np.int64))
    vae = router.get_vae(class_id)

    x_adv, z_adv, metadata = latent_cw_attack(
        vae=vae,
        classifier=classifier,
        mask=mask,
        x_original=x_batch,
        y_true=y_batch,
        scaler=router.scaler,
        lambda_conf=0.0,
        kappa=0.0,
        num_iterations=200,
        learning_rate=0.01,
        convergence_threshold=1e-5,
        device=device,
    )
    max_abs_diff = float(torch.max(torch.abs(x_adv.cpu() - x_batch)).item())
    return {
        "n_checked": 1,
        "exact_equal": bool(torch.equal(x_adv.cpu(), x_batch)),
        "max_abs_diff": max_abs_diff,
        "zero_budget_passthrough": bool(metadata["zero_budget_passthrough"]),
        "latent_l2": float(torch.linalg.norm(z_adv.cpu() - metadata["z_orig"].cpu(), dim=1).max().item()),
    }


def _evaluate_source_class(
    *,
    class_id: int,
    x_batch: torch.Tensor,
    y_batch: torch.Tensor,
    vae: torch.nn.Module,
    classifier: torch.nn.Module,
    mask: PerturbationMask,
    protocol_validator: ProtocolValidator,
    detector: MahalanobisOutlierDetector,
    scaler,
    device: str,
    lambda_conf: float,
    kappa: float,
    num_iterations: int,
    learning_rate: float,
    convergence_threshold: float,
) -> tuple[dict, dict]:
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

    with torch.no_grad():
        logits_before = classifier_logits(classifier, x_batch.to(device), device=device).cpu()
        logits_after = classifier_logits(classifier, x_adv.to(device), device=device).cpu()
        pred_before = torch.argmax(logits_before, dim=1)
        pred_after = torch.argmax(logits_after, dim=1)
        success_mask = pred_after != y_batch.cpu()

        protocol_valid = protocol_validator.validate(x_adv, already_scaled=True).cpu()
        mask_compliance = mask.verify(x_adv.cpu(), x_batch.cpu())["all_compliant"].cpu()
        z_reencoded, _ = vae.encode(x_adv.to(device))
        outlier_mask = detector.outlier_mask(class_id, z_reencoded.cpu()).cpu()

        input_l2 = torch.linalg.norm(x_adv.cpu() - x_batch.cpu(), dim=1)
        latent_l2 = torch.linalg.norm(z_adv.cpu() - metadata["z_orig"].cpu(), dim=1)

    row = {
        "class_name": CLASSES[class_id],
        "n": int(x_batch.shape[0]),
        "asr": float(success_mask.float().mean().item()),
        "mean_l2_input": float(input_l2.mean().item()),
        "mean_l2_latent": float(latent_l2.mean().item()),
        "protocol_validity_rate": float(protocol_valid.float().mean().item()),
        "mask_compliance_rate": float(mask_compliance.float().mean().item()),
        "idsr": float((~outlier_mask).float().mean().item()),
        "outlier_rate": float(outlier_mask.float().mean().item()),
        "loss_final": _to_float(metadata["loss_final"]),
        "iterations_run": int(metadata["iterations_run"]),
        "converged_early": bool(metadata["converged_early"]),
    }
    artifacts = {
        "x_orig": x_batch.cpu().numpy(),
        "x_adv": x_adv.cpu().numpy(),
        "y_true": y_batch.cpu().numpy(),
        "y_pred_before": pred_before.numpy(),
        "y_pred_after": pred_after.numpy(),
        "success_mask": success_mask.numpy(),
        "z_orig": metadata["z_orig"].cpu().numpy(),
        "z_adv": z_adv.cpu().numpy(),
    }
    return row, artifacts


def _latest_phase2_results(output_root: Path) -> dict | None:
    candidates = sorted(output_root.glob("phase2_*_seed*/phase2_results.json"))
    if not candidates:
        return None
    with open(candidates[-1], encoding="utf-8") as f:
        payload = json.load(f)
    payload["_path"] = str(candidates[-1])
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3 Latent-C&W checkpoint runner.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--lambda-conf", type=float, default=1.0)
    parser.add_argument("--kappa", type=float, default=0.0)
    parser.add_argument("--num-iterations", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--convergence-threshold", type=float, default=1e-5)
    args = parser.parse_args()

    set_global_seed(args.seed)
    run_logger = AttackRunLogger.create(
        phase_name="phase3",
        seed=args.seed,
        config_snapshot=_phase3_config_snapshot(args.seed, args.device),
    )

    router = AttackRouter(device=args.device)
    mask = PerturbationMask.from_preprocessing_artifacts()
    protocol_validator = ProtocolValidator(router.scaler)
    classifier = router.get_classifier("mlp-3l")
    split_test = load_split("test")
    detector = _fit_detector(router, args.device)

    zero_budget = _run_zero_budget_check(
        router=router,
        mask=mask,
        x_test=split_test["X"],
        y_test=split_test["y_8"],
        classifier=classifier,
        device=args.device,
    )

    rows: list[dict] = []
    per_class_artifacts: dict[str, dict] = {}
    for class_name in SOURCE_CLASSES:
        class_id = CLASS_TO_ID[class_name]
        sample_idx = _select_correct_source_samples(
            x_test=split_test["X"],
            y_test=split_test["y_8"],
            classifier=classifier,
            device=args.device,
            class_id=class_id,
            max_samples=SAMPLES_PER_SOURCE_CLASS,
        )
        x_batch = torch.from_numpy(split_test["X"][sample_idx].astype(np.float32))
        y_batch = torch.from_numpy(split_test["y_8"][sample_idx].astype(np.int64))
        vae = router.get_vae(class_id)

        row, artifacts = _evaluate_source_class(
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
        )
        rows.append(row)
        per_class_artifacts[class_name] = artifacts
        run_logger.log(json.dumps(row))

    overall_n = sum(row["n"] for row in rows)
    summary = {
        "n": overall_n,
        "asr": float(sum(row["asr"] * row["n"] for row in rows) / overall_n),
        "mean_l2_input": float(sum(row["mean_l2_input"] * row["n"] for row in rows) / overall_n),
        "mean_l2_latent": float(sum(row["mean_l2_latent"] * row["n"] for row in rows) / overall_n),
        "protocol_validity_rate": float(sum(row["protocol_validity_rate"] * row["n"] for row in rows) / overall_n),
        "mask_compliance_rate": float(sum(row["mask_compliance_rate"] * row["n"] for row in rows) / overall_n),
        "idsr": float(sum(row["idsr"] * row["n"] for row in rows) / overall_n),
    }

    phase2_payload = _latest_phase2_results(run_logger.run_dir.parent)
    payload = {
        "zero_budget": zero_budget,
        "per_class": rows,
        "overall": summary,
        "phase2_comparison_source": phase2_payload["_path"] if phase2_payload is not None else None,
        "phase2_overall": phase2_payload["overall"] if phase2_payload is not None else None,
        "phase2_per_class": phase2_payload["per_class"] if phase2_payload is not None else None,
    }
    out_json = run_logger.run_dir / "phase3_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    for class_name, artifacts in per_class_artifacts.items():
        np.savez_compressed(run_logger.run_dir / f"phase3_latent_cw_{class_name}.npz", **artifacts)

    print("=== Phase 3 Checkpoint ===")
    print(f"Output directory: {run_logger.run_dir}")
    print()
    print("Zero-budget test")
    print(f"  exact_equal={zero_budget['exact_equal']}")
    print(f"  max_abs_diff={zero_budget['max_abs_diff']:.8f}")
    print(f"  latent_l2={zero_budget['latent_l2']:.8f}")
    print()
    print("Latent-C&W vs MLP-3L")
    for row in rows:
        print(
            f"  {row['class_name']:10s} "
            f"ASR={row['asr'] * 100.0:6.2f}% "
            f"L2_in={row['mean_l2_input']:.4f} "
            f"L2_z={row['mean_l2_latent']:.4f} "
            f"Proto={row['protocol_validity_rate'] * 100.0:6.2f}% "
            f"Mask={row['mask_compliance_rate'] * 100.0:6.2f}% "
            f"IDSR={row['idsr'] * 100.0:6.2f}% "
            f"(n={row['n']})"
        )
    print()
    print(
        "Overall "
        f"ASR={summary['asr'] * 100.0:6.2f}% "
        f"L2_in={summary['mean_l2_input']:.4f} "
        f"L2_z={summary['mean_l2_latent']:.4f} "
        f"Proto={summary['protocol_validity_rate'] * 100.0:6.2f}% "
        f"Mask={summary['mask_compliance_rate'] * 100.0:6.2f}% "
        f"IDSR={summary['idsr'] * 100.0:6.2f}% "
        f"(n={summary['n']})"
    )

    if phase2_payload is not None:
        print()
        print(f"Phase 2 comparison source: {phase2_payload['_path']}")
        print(
            "Phase 2 Overall "
            f"ASR={phase2_payload['overall']['asr'] * 100.0:6.2f}% "
            f"L2_in={phase2_payload['overall']['mean_l2_input']:.4f} "
            f"L2_z={phase2_payload['overall']['mean_l2_latent']:.4f} "
            f"Proto={phase2_payload['overall']['protocol_validity_rate'] * 100.0:6.2f}% "
            f"Mask={phase2_payload['overall']['mask_compliance_rate'] * 100.0:6.2f}% "
            f"IDSR={phase2_payload['overall']['idsr'] * 100.0:6.2f}% "
            f"(n={phase2_payload['overall']['n']})"
        )


if __name__ == "__main__":
    main()
