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
    PerturbationMask,
    inverse_transform_scaled,
    load_split,
    phase0_config_snapshot,
    predict_labels,
    reimpose_protocol_features,
    set_global_seed,
)
from attack.validator import validate_batch  # noqa: E402
from preprocessing.feature_groups import FEATURE_NAMES  # noqa: E402
from vae.config import CLASS_TO_ID, CLASSES  # noqa: E402
from vae.schema import raw_protocol_to_scaled  # noqa: E402

RECON_SAMPLES_PER_CLASS = 100
MASK_SAMPLES_PER_CLASS = 50


def _phase1_config_snapshot(seed: int, device: str) -> dict:
    snapshot = phase0_config_snapshot(seed, device)
    snapshot.update(
        {
            "phase": "phase1",
            "split": "test",
            "reconstruction_samples_per_class": RECON_SAMPLES_PER_CLASS,
            "mask_samples_per_class": MASK_SAMPLES_PER_CLASS,
            "trace_classifier": "mlp-3l",
        }
    )
    return snapshot


def _sample_class_indices(y: np.ndarray, class_id: int, n: int) -> np.ndarray:
    class_idx = np.where(y == class_id)[0]
    return class_idx[: min(n, len(class_idx))]


def _normalized_l2(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    num = torch.linalg.norm(x - y, dim=1)
    denom = torch.linalg.norm(x, dim=1).clamp_min(1e-12)
    return num / denom


def _to_python_floats(arr: torch.Tensor | np.ndarray) -> list[float]:
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu().numpy()
    return [float(x) for x in np.asarray(arr).reshape(-1)]


def _protocol_freezing_check(
    router: AttackRouter,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    class_name = "DoS"
    class_id = CLASS_TO_ID[class_name]
    vae = router.get_vae(class_id)

    class_indices = np.where(y_test == class_id)[0]
    tcp_index = FEATURE_NAMES.index("TCP")
    udp_index = FEATURE_NAMES.index("UDP")

    tcp_sample_idx = None
    for idx in class_indices:
        x_raw = inverse_transform_scaled(x_test[idx : idx + 1], router.scaler)
        if round(float(x_raw[0, tcp_index])) == 1:
            tcp_sample_idx = int(idx)
            break

    if tcp_sample_idx is None:
        raise RuntimeError("Could not find a TCP DoS sample for protocol freezing check")

    x_original = torch.from_numpy(x_test[tcp_sample_idx : tcp_sample_idx + 1].astype(np.float32))
    mu, _ = vae.encode(x_original.to(router.device))
    x_decoded, _ = vae.decode_to_39(mu, router.scaler, mode="hard")
    x_decoded = x_decoded.cpu()

    protocol_idx = FEATURE_NAMES.index("Protocol Type")
    tampered = x_decoded.clone()
    udp_protocol_scaled = float(
        raw_protocol_to_scaled(
            np.array([17.0], dtype=np.float64),
            router.scaler,
            protocol_idx=protocol_idx,
            n_features=len(FEATURE_NAMES),
        )[0]
    )
    tampered[:, protocol_idx] = udp_protocol_scaled
    tampered[:, tcp_index] = x_original[:, tcp_index]
    tampered[:, udp_index] = x_original[:, udp_index]

    tampered_raw = inverse_transform_scaled(tampered, router.scaler)
    tampered_validity = validate_batch(tampered_raw, FEATURE_NAMES)

    repaired = reimpose_protocol_features(tampered, x_original)
    repaired_raw = inverse_transform_scaled(repaired, router.scaler)
    repaired_validity = validate_batch(repaired_raw, FEATURE_NAMES)

    original_raw = inverse_transform_scaled(x_original, router.scaler)
    repaired_protocol = float(repaired_raw[0, protocol_idx])
    original_protocol = float(original_raw[0, protocol_idx])

    return {
        "sample_index": tcp_sample_idx,
        "tampered_valid": bool(tampered_validity.overall_valid[0]),
        "tampered_violation_rates": tampered_validity.per_rule_violation_rate(),
        "repaired_valid": bool(repaired_validity.overall_valid[0]),
        "original_protocol_raw": original_protocol,
        "repaired_protocol_raw": repaired_protocol,
        "protocol_restored_exactly": abs(repaired_protocol - original_protocol) < 1e-9,
    }


def _find_reasonable_dos_trace(
    router: AttackRouter,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    class_name = "DoS"
    class_id = CLASS_TO_ID[class_name]
    vae = router.get_vae(class_id)
    classifier = router.get_classifier("mlp-3l")
    class_indices = np.where(y_test == class_id)[0]

    for idx in class_indices:
        x = torch.from_numpy(x_test[idx : idx + 1].astype(np.float32))
        mu, _ = vae.encode(x.to(router.device))
        x_recon, _ = vae.decode_to_39(mu, router.scaler, mode="hard")
        x_recon = x_recon.cpu()

        pred_x = int(predict_labels(classifier, x, device=router.device)[0].item())
        pred_recon = int(predict_labels(classifier, x_recon, device=router.device)[0].item())
        if pred_x == class_id and pred_recon == class_id:
            return {
                "sample_index": int(idx),
                "x_scaled": _to_python_floats(x[0]),
                "z_mu": _to_python_floats(mu.cpu()[0]),
                "x_reconstructed_scaled": _to_python_floats(x_recon[0]),
                "pred_original": CLASSES[pred_x],
                "pred_reconstructed": CLASSES[pred_recon],
                "l2_x_to_recon": float(torch.linalg.norm(x - x_recon, dim=1)[0].item()),
                "l2_z_norm": float(torch.linalg.norm(mu.cpu(), dim=1)[0].item()),
            }

    raise RuntimeError("Could not find a DoS sample classified as DoS before and after reconstruction")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1 sanity checks for latent attack pipeline.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    set_global_seed(args.seed)
    run_logger = AttackRunLogger.create(
        phase_name="phase1",
        seed=args.seed,
        config_snapshot=_phase1_config_snapshot(args.seed, args.device),
    )

    router = AttackRouter(device=args.device)
    mask = PerturbationMask.from_preprocessing_artifacts()
    split = load_split("test")
    x_test = split["X"]
    y_test = split["y_8"]

    reconstruction_rows: list[dict] = []
    round_trip_pass = True
    noisy_mask_pass = True

    for class_id, class_name in enumerate(CLASSES):
        vae = router.get_vae(class_id)
        class_indices = _sample_class_indices(y_test, class_id, RECON_SAMPLES_PER_CLASS)
        x_batch = torch.from_numpy(x_test[class_indices].astype(np.float32)).to(router.device)

        with torch.no_grad():
            mu, _ = vae.encode(x_batch)
            x_recon, _ = vae.decode_to_39(mu, router.scaler, mode="hard")
            recon_error = _normalized_l2(x_batch, x_recon)

        reconstruction_rows.append(
            {
                "class_name": class_name,
                "n_samples": int(len(class_indices)),
                "median": float(torch.quantile(recon_error.cpu(), 0.5).item()),
                "p95": float(torch.quantile(recon_error.cpu(), 0.95).item()),
                "max": float(recon_error.max().item()),
            }
        )

        mask_indices = _sample_class_indices(y_test, class_id, MASK_SAMPLES_PER_CLASS)
        x_mask = torch.from_numpy(x_test[mask_indices].astype(np.float32))
        x_round_trip = mask.apply(x_mask, x_mask)
        round_trip_pass = round_trip_pass and bool(torch.equal(x_round_trip, x_mask))

        noise = torch.randn_like(x_mask) * 0.75
        x_noisy = x_mask + noise
        x_masked = mask.apply(x_noisy, x_mask)
        compliance = mask.verify(x_masked, x_mask)
        noisy_mask_pass = noisy_mask_pass and bool(compliance["all_compliant"].all().item())

    protocol_check = _protocol_freezing_check(router, x_test, y_test)
    trace = _find_reasonable_dos_trace(router, x_test, y_test)

    out_json = run_logger.run_dir / "phase1_results.json"
    payload = {
        "reconstruction_rows": reconstruction_rows,
        "mask_round_trip_pass": round_trip_pass,
        "mask_noisy_compliance_pass": noisy_mask_pass,
        "protocol_check": protocol_check,
        "trace": trace,
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("=== Phase 1 Checkpoint ===")
    print(f"Output directory: {run_logger.run_dir}")
    print()
    print("Reconstruction fidelity")
    for row in reconstruction_rows:
        print(
            f"  {row['class_name']:10s} "
            f"median={row['median']:.4f} "
            f"p95={row['p95']:.4f} "
            f"max={row['max']:.4f} "
            f"(n={row['n_samples']})"
        )
    print()
    print(f"Mask round-trip exact pass: {round_trip_pass}")
    print(f"Mask noisy compliance pass: {noisy_mask_pass}")
    print()
    print("Protocol freezing check")
    print(f"  tampered_valid={protocol_check['tampered_valid']}")
    print(f"  repaired_valid={protocol_check['repaired_valid']}")
    print(f"  protocol_restored_exactly={protocol_check['protocol_restored_exactly']}")
    print()
    print("DoS pipeline trace")
    print(f"  sample_index={trace['sample_index']}")
    print(f"  classifier(original)={trace['pred_original']}")
    print(f"  classifier(reconstructed)={trace['pred_reconstructed']}")
    print(f"  L2(x, x_reconstructed)={trace['l2_x_to_recon']:.6f}")
    print(f"  ||z||_2={trace['l2_z_norm']:.6f}")


if __name__ == "__main__":
    main()
