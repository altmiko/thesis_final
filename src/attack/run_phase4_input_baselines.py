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

from attack.input_baselines import input_cw_attack, input_pgd_attack  # noqa: E402
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
from attack.latent_pgd import classifier_logits  # noqa: E402
from vae.config import CLASS_TO_ID, CLASSES  # noqa: E402

SAMPLES_PER_SOURCE_CLASS = 100
SOURCE_CLASSES = [name for name in CLASSES if name != "Benign"]


def _phase4_config_snapshot(seed: int, device: str) -> dict:
    snapshot = phase0_config_snapshot(seed, device)
    snapshot.update(
        {
            "phase": "phase4",
            "split": "test",
            "sampling": {
                "source_classes": SOURCE_CLASSES,
                "samples_per_source_class": SAMPLES_PER_SOURCE_CLASS,
                "correctly_classified_only": True,
            },
            "classifier": "mlp-3l",
            "attacks": [
                {
                    "name": "input-pgd",
                    "epsilon": 0.5,
                    "alpha": 0.05,
                    "num_steps": 40,
                    "random_start": True,
                },
                {
                    "name": "input-cw",
                    "lambda_conf": 1.0,
                    "kappa": 0.0,
                    "num_iterations": 200,
                    "learning_rate": 0.01,
                    "convergence_threshold": 1e-5,
                },
            ],
            "evaluation_note": "Mask and protocol validity are applied post-hoc only, not during optimization.",
        }
    )
    return snapshot


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


def _masked_rate(mask_tensor: torch.Tensor) -> float:
    if mask_tensor.numel() == 0:
        return 0.0
    return float(mask_tensor.float().mean().item())


def _conditional_asr(success_mask: torch.Tensor, subset_mask: torch.Tensor) -> float:
    denom = int(subset_mask.sum().item())
    if denom == 0:
        return 0.0
    return float((success_mask & subset_mask).float().sum().item() / denom)


def _evaluate_attack(
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
) -> tuple[dict, dict]:
    if attack_name == "input-pgd":
        x_adv, metadata = input_pgd_attack(
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
        x_adv, metadata = input_cw_attack(
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
        raise KeyError(f"Unsupported attack_name={attack_name}")

    with torch.no_grad():
        logits_before = classifier_logits(classifier, x_batch.to(device), device=device).cpu()
        logits_after = classifier_logits(classifier, x_adv.to(device), device=device).cpu()
        pred_before = torch.argmax(logits_before, dim=1)
        pred_after = torch.argmax(logits_after, dim=1)
        success_mask = pred_after != y_batch.cpu()

        protocol_valid = protocol_validator.validate(x_adv, already_scaled=True).cpu()
        mask_compliance = mask.verify(x_adv.cpu(), x_batch.cpu())["all_compliant"].cpu()
        valid_mask = protocol_valid & mask_compliance
        invalid_mask = ~valid_mask

        input_l2 = torch.linalg.norm(x_adv.cpu() - x_batch.cpu(), dim=1)

    row = {
        "attack_name": attack_name,
        "class_name": CLASSES[class_id],
        "n": int(x_batch.shape[0]),
        "asr_overall": float(success_mask.float().mean().item()),
        "asr_valid_only": _conditional_asr(success_mask, valid_mask),
        "asr_invalid_only": _conditional_asr(success_mask, invalid_mask),
        "protocol_validity_rate": _masked_rate(protocol_valid),
        "mask_compliance_rate": _masked_rate(mask_compliance),
        "joint_validity_rate": _masked_rate(valid_mask),
        "mean_l2_input": float(input_l2.mean().item()),
        "valid_count": int(valid_mask.sum().item()),
        "invalid_count": int(invalid_mask.sum().item()),
        "successful_valid_count": int((success_mask & valid_mask).sum().item()),
        "successful_invalid_count": int((success_mask & invalid_mask).sum().item()),
        "successful_total_count": int(success_mask.sum().item()),
        "loss_final": float(metadata["loss_final"]),
    }
    artifacts = {
        "x_orig": x_batch.cpu().numpy(),
        "x_adv": x_adv.cpu().numpy(),
        "y_true": y_batch.cpu().numpy(),
        "y_pred_before": pred_before.numpy(),
        "y_pred_after": pred_after.numpy(),
        "success_mask": success_mask.numpy(),
        "protocol_valid": protocol_valid.numpy(),
        "mask_compliance": mask_compliance.numpy(),
        "joint_valid": valid_mask.numpy(),
    }
    return row, artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4 input-space baseline checkpoint runner.")
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
        phase_name="phase4",
        seed=args.seed,
        config_snapshot=_phase4_config_snapshot(args.seed, args.device),
    )

    router = AttackRouter(device=args.device)
    mask = PerturbationMask.from_preprocessing_artifacts()
    protocol_validator = ProtocolValidator(router.scaler)
    classifier = router.get_classifier("mlp-3l")
    split_test = load_split("test")

    attack_rows: dict[str, list[dict]] = {"input-pgd": [], "input-cw": []}
    artifacts_by_attack: dict[str, dict[str, dict]] = {"input-pgd": {}, "input-cw": {}}

    for attack_name in ("input-pgd", "input-cw"):
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

            row, artifacts = _evaluate_attack(
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
            attack_rows[attack_name].append(row)
            artifacts_by_attack[attack_name][class_name] = artifacts
            run_logger.log(json.dumps(row))

    overall: dict[str, dict] = {}
    for attack_name, rows in attack_rows.items():
        overall_n = sum(row["n"] for row in rows)
        total_valid = sum(row["valid_count"] for row in rows)
        total_invalid = sum(row["invalid_count"] for row in rows)
        overall[attack_name] = {
            "n": overall_n,
            "asr_overall": float(sum(row["successful_total_count"] for row in rows) / overall_n),
            "asr_valid_only": float(sum(row["successful_valid_count"] for row in rows) / total_valid) if total_valid > 0 else 0.0,
            "asr_invalid_only": float(sum(row["successful_invalid_count"] for row in rows) / total_invalid) if total_invalid > 0 else 0.0,
            "protocol_validity_rate": float(sum(row["protocol_validity_rate"] * row["n"] for row in rows) / overall_n),
            "mask_compliance_rate": float(sum(row["mask_compliance_rate"] * row["n"] for row in rows) / overall_n),
            "joint_validity_rate": float(sum(row["joint_validity_rate"] * row["n"] for row in rows) / overall_n),
            "mean_l2_input": float(sum(row["mean_l2_input"] * row["n"] for row in rows) / overall_n),
            "valid_count": int(total_valid),
            "invalid_count": int(total_invalid),
        }

    payload = {
        "per_attack": attack_rows,
        "overall": overall,
    }
    out_json = run_logger.run_dir / "phase4_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    for attack_name, by_class in artifacts_by_attack.items():
        for class_name, artifacts in by_class.items():
            np.savez_compressed(run_logger.run_dir / f"phase4_{attack_name}_{class_name}.npz", **artifacts)

    print("=== Phase 4 Checkpoint ===")
    print(f"Output directory: {run_logger.run_dir}")
    print()
    for attack_name in ("input-pgd", "input-cw"):
        pretty_name = "Input-PGD" if attack_name == "input-pgd" else "Input-C&W"
        print(pretty_name + " vs MLP-3L")
        for row in attack_rows[attack_name]:
            print(
                f"  {row['class_name']:10s} "
                f"ASR={row['asr_overall'] * 100.0:6.2f}% "
                f"ASR_valid={row['asr_valid_only'] * 100.0:6.2f}% "
                f"ASR_invalid={row['asr_invalid_only'] * 100.0:6.2f}% "
                f"Proto={row['protocol_validity_rate'] * 100.0:6.2f}% "
                f"Mask={row['mask_compliance_rate'] * 100.0:6.2f}% "
                f"JointValid={row['joint_validity_rate'] * 100.0:6.2f}% "
                f"(n={row['n']})"
            )
        summary = overall[attack_name]
        print()
        print(
            f"Overall {pretty_name} "
            f"ASR={summary['asr_overall'] * 100.0:6.2f}% "
            f"ASR_valid={summary['asr_valid_only'] * 100.0:6.2f}% "
            f"ASR_invalid={summary['asr_invalid_only'] * 100.0:6.2f}% "
            f"Proto={summary['protocol_validity_rate'] * 100.0:6.2f}% "
            f"Mask={summary['mask_compliance_rate'] * 100.0:6.2f}% "
            f"JointValid={summary['joint_validity_rate'] * 100.0:6.2f}% "
            f"(n={summary['n']})"
        )
        print()


if __name__ == "__main__":
    main()
