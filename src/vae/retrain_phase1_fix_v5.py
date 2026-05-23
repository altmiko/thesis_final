from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import sys
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = str(_REPO_ROOT / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from attack.latent_infra import (  # noqa: E402
    AttackRunLogger,
    AttackRouter,
    MahalanobisOutlierDetector,
    encode_dataset_mu,
    load_collapsed_dims,
    load_split,
    phase0_config_snapshot,
    predict_labels,
    set_global_seed,
)
from vae.config import CLASS_TO_ID, CLASSES, DEFAULT_CONFIG  # noqa: E402
from vae.dataset import PerClassDataset  # noqa: E402
from vae.train import train_one_vae  # noqa: E402

LOGGER = logging.getLogger(__name__)
TARGET_CLASSES = ("BruteForce", "DoS")


def _snapshot(seed: int, device: str) -> dict:
    snap = phase0_config_snapshot(seed, device)
    snap.update({"phase": "phaseBCD_v5", "target_classes": list(TARGET_CLASSES)})
    return snap


def _load_shared_arrays() -> dict:
    root = _REPO_ROOT
    with open(str(root / "data" / "processed" / "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    return {
        "X_train": np.load(str(root / "data" / "processed" / "X_train.npy")),
        "y_train_34": np.load(str(root / "data" / "processed" / "y_train.npy")),
        "X_val": np.load(str(root / "data" / "processed" / "X_val.npy")),
        "y_val_34": np.load(str(root / "data" / "processed" / "y_val.npy")),
        "y_train_8": np.load(str(root / "data" / "processed" / "y_train_cat.npy")),
        "y_val_8": np.load(str(root / "data" / "processed" / "y_val_cat.npy")),
        "scaler": scaler,
    }


def _config_for(class_name: str) -> dict:
    cfg = dict(DEFAULT_CONFIG)
    cfg["latent_dim"] = dict(DEFAULT_CONFIG["latent_dim"])
    cfg["beta_target"] = dict(DEFAULT_CONFIG["beta_target"])
    cfg["max_epochs"] = 300
    cfg["early_stop_patience"] = 25
    cfg["early_stop_metric"] = "recon_metric"
    cfg["recon_metric_every_n_epochs"] = 1
    cfg["recon_metric_quantile"] = 0.95
    cfg["free_bits_lambda"] = 0.5
    cfg["weight_decay"] = 1e-4
    cfg["normalize_feature_loss_weights"] = True
    cfg["raw_relative_epsilon"] = 1.0

    if class_name == "BruteForce":
        cfg["latent_dim"]["BruteForce"] = 8
        cfg["beta_target"]["BruteForce"] = 0.35
        cfg["checkpoint_name_override"] = "vae_bruteforce_v5.pt"
        cfg["curves_name_override"] = "curves_BruteForce_v5.png"
        cfg["raw_relative_continuous_loss_weight"] = 1.0
        cfg["raw_relative_tail_focus_quantile"] = 0.80
        cfg["raw_relative_tail_focus_weight"] = 5.0
        cfg["continuous_feature_loss_weights"] = {
            "Rate": 6.0,
            "Variance": 4.0,
            "Tot sum": 2.0,
            "Header_Length": 1.5,
            "ack_count": 1.5,
            "IAT": 1.5,
        }
        cfg["raw_relative_feature_loss_weights"] = {
            "Rate": 14.0,
            "Variance": 8.0,
            "Tot sum": 3.0,
            "Header_Length": 2.0,
            "ack_count": 2.0,
            "IAT": 2.0,
            "Number": 2.0,
        }
    elif class_name == "DoS":
        cfg["latent_dim"]["DoS"] = 16
        cfg["beta_target"]["DoS"] = 0.5
        cfg["checkpoint_name_override"] = "vae_dos_v5.pt"
        cfg["curves_name_override"] = "curves_DoS_v5.png"
        cfg["raw_relative_continuous_loss_weight"] = 2.0
        cfg["raw_relative_tail_focus_quantile"] = 0.90
        cfg["raw_relative_tail_focus_weight"] = 6.0
        cfg["continuous_feature_loss_weights"] = {
            "Variance": 18.0,
            "Tot sum": 10.0,
            "Max": 4.0,
            "Std": 4.0,
            "AVG": 3.0,
            "Tot size": 3.0,
            "Rate": 3.0,
            "Min": 2.0,
        }
        cfg["raw_relative_feature_loss_weights"] = {
            "Variance": 22.0,
            "Tot sum": 12.0,
            "Max": 5.0,
            "Std": 5.0,
            "AVG": 4.0,
            "Tot size": 4.0,
            "Rate": 4.0,
            "Min": 3.0,
        }
    else:
        raise ValueError(class_name)
    return cfg


def _checkpoint_path(class_name: str, version: str) -> Path:
    filename = f"vae_{class_name.lower()}_{version}.pt"
    return _REPO_ROOT / "models" / "vae" / filename


def _get_vae(router: AttackRouter, class_id: int, override_path: Path | None):
    if override_path is None:
        return router.get_vae(class_id)
    class_name = CLASSES[class_id]
    custom = AttackRouter(device=router.device, classifier_paths=router.classifier_paths)
    custom.manifest["checkpoints"][class_name]["path"] = str(override_path)
    return custom.get_vae(class_id)


def _recon_summary(router: AttackRouter, class_id: int, x_np: np.ndarray, override_path: Path | None) -> dict:
    vae = _get_vae(router, class_id, override_path)
    batch_size = 2048
    errs: list[np.ndarray] = []
    for start in range(0, len(x_np), batch_size):
        end = min(start + batch_size, len(x_np))
        x = torch.from_numpy(x_np[start:end].astype(np.float32)).to(router.device)
        with torch.no_grad():
            mu, _ = vae.encode(x)
            x_recon, _ = vae.decode_to_39(mu, router.scaler, mode="hard")
        err = torch.linalg.norm(x - x_recon, dim=1) / torch.linalg.norm(x, dim=1).clamp_min(1e-12)
        errs.append(err.cpu().numpy())
    all_err = np.concatenate(errs, axis=0)
    return {
        "median": float(np.quantile(all_err, 0.5)),
        "p90": float(np.quantile(all_err, 0.9)),
        "p95": float(np.quantile(all_err, 0.95)),
        "p99": float(np.quantile(all_err, 0.99)),
        "max": float(all_err.max()),
    }


def _mahalanobis_summary(
    router: AttackRouter,
    class_id: int,
    x_val: np.ndarray,
    y_val: np.ndarray,
    override_path: Path | None,
) -> dict:
    vae = _get_vae(router, class_id, override_path)
    dataset = PerClassDataset(x_val, y_val, class_id, router.scaler, router.partition)
    z_mu = encode_dataset_mu(vae, dataset, device=router.device)
    collapsed = load_collapsed_dims()[class_id]
    detector = MahalanobisOutlierDetector(latent_dim=z_mu.shape[1])
    stats = detector.fit(class_id, z_mu, collapsed_dims=collapsed)
    return {
        "outlier_rate": detector.outlier_rate(class_id, z_mu),
        "effective_dimensionality": stats["effective_dimensionality"],
        "ridge_lambda": stats["ridge_lambda"],
        "collapsed_dims_used": collapsed,
    }


def _pipeline_trace(router: AttackRouter, class_id: int, x_np: np.ndarray, override_path: Path | None) -> dict:
    vae = _get_vae(router, class_id, override_path)
    clf = router.get_classifier("mlp-3l")
    for idx in range(len(x_np)):
        x = torch.from_numpy(x_np[idx : idx + 1].astype(np.float32))
        with torch.no_grad():
            mu, _ = vae.encode(x.to(router.device))
            x_recon, _ = vae.decode_to_39(mu, router.scaler, mode="hard")
        pred_x = int(predict_labels(clf, x, device=router.device)[0].item())
        pred_recon = int(predict_labels(clf, x_recon.cpu(), device=router.device)[0].item())
        if pred_x == class_id and pred_recon == class_id:
            return {
                "sample_index": idx,
                "pred_original": CLASSES[pred_x],
                "pred_reconstructed": CLASSES[pred_recon],
                "l2_x_to_recon": float(torch.linalg.norm(x - x_recon.cpu(), dim=1)[0].item()),
                "z_norm": float(torch.linalg.norm(mu.cpu(), dim=1)[0].item()),
            }
    return {
        "sample_index": None,
        "pred_original": None,
        "pred_reconstructed": None,
        "l2_x_to_recon": None,
        "z_norm": None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Tail-focused raw-relative VAE retraining for BruteForce and DoS.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
    set_global_seed(args.seed)
    run_logger = AttackRunLogger.create(
        phase_name="phaseBCD_v5",
        seed=args.seed,
        config_snapshot=_snapshot(args.seed, args.device),
    )
    shared = _load_shared_arrays()

    training_results: dict[str, dict] = {}
    for class_name in TARGET_CLASSES:
        class_id = CLASS_TO_ID[class_name]
        metrics = train_one_vae(
            class_id=class_id,
            class_name=class_name,
            config=_config_for(class_name),
            device=args.device,
            shared_arrays=shared,
        )
        training_results[class_name] = {
            "checkpoint_path": metrics["checkpoint_path"],
            "best_val_loss": metrics["best_val_loss"],
            "best_monitor_value": metrics["best_monitor_value"],
            "best_epoch": metrics["best_epoch"],
            "epochs_trained": metrics["epochs_trained"],
            "early_stop_metric": metrics["early_stop_metric"],
        }

    router = AttackRouter(device=args.device)
    test_split = load_split("test")
    val_split = load_split("val")
    phase_d: dict[str, dict] = {}
    for class_name in TARGET_CLASSES:
        class_id = CLASS_TO_ID[class_name]
        ckpt_v2 = _REPO_ROOT / "models" / "vae" / (
            "vae_bruteforce_v2.pt" if class_name == "BruteForce" else "vae_dos_v2.pt"
        )
        ckpt_v5 = _checkpoint_path(class_name, "v5")
        idx = np.where(test_split["y_8"] == class_id)[0][:1000]
        x_test = test_split["X"][idx].astype(np.float32)
        phase_d[class_name] = {
            "v2_reconstruction": _recon_summary(router, class_id, x_test, ckpt_v2),
            "v5_reconstruction": _recon_summary(router, class_id, x_test, ckpt_v5),
            "mahalanobis_v5": _mahalanobis_summary(
                router,
                class_id,
                val_split["X"],
                val_split["y_8"],
                ckpt_v5,
            ),
            "pipeline_trace_v2": _pipeline_trace(router, class_id, x_test, ckpt_v2),
            "pipeline_trace_v5": _pipeline_trace(router, class_id, x_test, ckpt_v5),
        }

    results = {
        "seed": args.seed,
        "device": args.device,
        "training_results": training_results,
        "phase_d": phase_d,
    }
    out_path = run_logger.run_dir / "phaseBCD_v5_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("=== Phase B/C/D v5 Results ===")
    print(f"Output directory: {run_logger.run_dir}")
    print(f"Results JSON: {out_path}")
    print()
    for class_name in TARGET_CLASSES:
        row = phase_d[class_name]
        print(
            f"{class_name:10s} v2 median/p95={row['v2_reconstruction']['median']:.4f}/{row['v2_reconstruction']['p95']:.4f} "
            f"v5 median/p95={row['v5_reconstruction']['median']:.4f}/{row['v5_reconstruction']['p95']:.4f} "
            f"MD self-rate={row['mahalanobis_v5']['outlier_rate'] * 100.0:.2f}%"
        )
        print(
            f"  trace v2={row['pipeline_trace_v2']['pred_original']}->{row['pipeline_trace_v2']['pred_reconstructed']} "
            f"v5={row['pipeline_trace_v5']['pred_original']}->{row['pipeline_trace_v5']['pred_reconstructed']}"
        )


if __name__ == "__main__":
    main()
