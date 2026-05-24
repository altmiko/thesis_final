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
from vae.config import CLASS_TO_ID, CLASSES  # noqa: E402

SAMPLES_PER_SOURCE_CLASS = 100
SOURCE_CLASSES = [name for name in CLASSES if name != "Benign"]


def _phase2_config_snapshot(seed: int, device: str) -> dict:
    snapshot = phase0_config_snapshot(seed, device)
    snapshot.update(
        {
            "phase": "phase2",
            "split": "test",
            "sampling": {
                "source_classes": SOURCE_CLASSES,
                "samples_per_source_class": SAMPLES_PER_SOURCE_CLASS,
                "correctly_classified_only": True,
            },
            "classifier": "mlp-3l",
            "attack": {
                "name": "latent-pgd",
                "epsilon": 0.5,
                "alpha": 0.05,
                "num_steps": 40,
                "random_start": True,
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
    sample_indices: list[int] = []
    for class_name in SOURCE_CLASSES:
        class_id = CLASS_TO_ID[class_name]
        selected = _select_correct_source_samples(
            x_test=x_test,
            y_test=y_test,
            classifier=classifier,
            device=device,
            class_id=class_id,
            max_samples=1,
        )
        sample_indices.extend(selected.tolist())

    x_batch = torch.from_numpy(x_test[np.asarray(sample_indices)].astype(np.float32))
    y_batch = torch.from_numpy(y_test[np.asarray(sample_indices)].astype(np.int64))
    vae = router.get_vae(int(y_batch[0].item()))

    x_adv, z_adv, metadata = latent_pgd_attack(
        vae=vae,
        classifier=classifier,
        mask=mask,
        x_original=x_batch[:1],
        y_true=y_batch[:1],
        scaler=router.scaler,
        epsilon=0.0,
        alpha=0.05,
        num_steps=40,
        random_start=True,
        device=device,
    )
    max_abs_diff = float(torch.max(torch.abs(x_adv.cpu() - x_batch[:1])).item())
    return {
        "n_checked": 1,
        "exact_equal": bool(torch.equal(x_adv.cpu(), x_batch[:1])),
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
    epsilon: float,
    alpha: float,
    num_steps: int,
    random_start: bool,
) -> tuple[dict, dict]:
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2 Latent-PGD checkpoint runner.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epsilon", type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--num-steps", type=int, default=40)
    parser.add_argument("--random-start", action="store_true", default=True)
    parser.add_argument("--no-random-start", dest="random_start", action="store_false")
    args = parser.parse_args()

    set_global_seed(args.seed)
    run_logger = AttackRunLogger.create(
        phase_name="phase2",
        seed=args.seed,
        config_snapshot=_phase2_config_snapshot(args.seed, args.device),
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
            epsilon=args.epsilon,
            alpha=args.alpha,
            num_steps=args.num_steps,
            random_start=args.random_start,
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

    payload = {
        "zero_budget": zero_budget,
        "per_class": rows,
        "overall": summary,
    }
    out_json = run_logger.run_dir / "phase2_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    for class_name, artifacts in per_class_artifacts.items():
        np.savez_compressed(run_logger.run_dir / f"phase2_latent_pgd_{class_name}.npz", **artifacts)

    print("=== Phase 2 Checkpoint ===")
    print(f"Output directory: {run_logger.run_dir}")
    print()
    print("Zero-budget test")
    print(f"  exact_equal={zero_budget['exact_equal']}")
    print(f"  max_abs_diff={zero_budget['max_abs_diff']:.8f}")
    print(f"  latent_l2={zero_budget['latent_l2']:.8f}")
    print()
    print("Latent-PGD vs MLP-3L")
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


if __name__ == "__main__":
    main()
