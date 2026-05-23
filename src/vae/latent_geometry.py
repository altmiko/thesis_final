"""
Latent-space geometry diagnostics for all 8 per-class β-VAEs.

For each class VAE:
  1. Encode the class's own val split → fit latent center (mu_bar) and
     precision matrix (Lambda = Sigma^{-1}) via empirical estimates of
     the posterior means.
  2. Encode the Benign val split through that same VAE and compute
     Mahalanobis distances from the class center.
  3. Report the chi^2(0.95, k) outlier threshold and the fraction of
     Benign samples that fall inside / outside it.

Results saved to results/vae/latent_geometry.json.

Usage:
    python src/vae/latent_geometry.py --device cuda
    python src/vae/latent_geometry.py --device cpu --classes Benign,DDoS
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.stats import chi2
from torch.utils.data import DataLoader

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = str(_REPO_ROOT / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from vae.config import CLASS_TO_ID, CLASSES, DEFAULT_CONFIG  # noqa: E402
from vae.dataset import PerClassDataset  # noqa: E402
from vae.model import MixedInputBetaVAE  # noqa: E402
from vae.schema import get_partition  # noqa: E402
from vae.train import _load_8class_labels  # noqa: E402
from vae.train_all import _resolve_model_hparams  # noqa: E402

logger = logging.getLogger(__name__)

BENIGN_CLASS_ID: int = CLASS_TO_ID["Benign"]


def _setup_logging() -> None:
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    if not root.handlers:
        sh = logging.StreamHandler(sys.stdout)
        sh.setLevel(logging.INFO)
        sh.setFormatter(fmt)
        root.addHandler(sh)


def _encode_dataset(
    model: MixedInputBetaVAE,
    ds: PerClassDataset,
    device: str,
    batch_size: int = 2048,
) -> np.ndarray:
    """Return posterior means mu for all samples in ds, shape (N, latent_dim)."""
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    mu_list: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            x = batch["x_scaled"].to(device)
            mu, _ = model.encode(x)
            mu_list.append(mu.cpu().numpy())
    return np.concatenate(mu_list, axis=0)  # (N, latent_dim)


def _fit_latent_stats(mu_class: np.ndarray) -> dict[str, Any]:
    """Fit empirical mean and covariance from posterior means of the class's own val split."""
    mu_bar = mu_class.mean(axis=0)  # (k,)
    centered = mu_class - mu_bar     # (N, k)
    N, k = centered.shape

    # Unbiased sample covariance with regularisation for numerical stability
    sigma = np.cov(centered, rowvar=False)  # (k, k)
    if sigma.ndim == 0:
        sigma = np.array([[float(sigma)]])

    # Tikhonov regularisation: add 1e-4 * I to avoid singular matrices for
    # classes where a latent dim has near-zero variance (collapsed dims).
    sigma_reg = sigma + 1e-4 * np.eye(k)

    try:
        lambda_mat = np.linalg.inv(sigma_reg)  # (k, k) precision matrix
        inv_ok = True
    except np.linalg.LinAlgError:
        lambda_mat = np.eye(k)
        inv_ok = False
        logger.warning("Sigma inversion failed — falling back to identity precision")

    return {
        "mu_bar": mu_bar,
        "sigma": sigma,
        "sigma_reg": sigma_reg,
        "lambda_mat": lambda_mat,
        "inv_ok": inv_ok,
        "n_samples_fit": int(N),
        "latent_dim": int(k),
    }


def _mahalanobis_sq(mu_query: np.ndarray, mu_bar: np.ndarray, lambda_mat: np.ndarray) -> np.ndarray:
    """Return squared Mahalanobis distances for each row of mu_query, shape (M,)."""
    diff = mu_query - mu_bar[None, :]          # (M, k)
    d_sq = np.einsum("mi,ij,mj->m", diff, lambda_mat, diff)  # (M,)
    return d_sq


def run_latent_geometry(
    classes: list[str],
    device: str,
    max_benign_samples: int = 5000,
) -> dict[str, Any]:
    root = _REPO_ROOT

    # ------------------------------------------------------------------
    # Load shared data
    # ------------------------------------------------------------------
    logger.info("Loading shared val arrays ...")
    X_val = np.load(str(root / "data" / "processed" / "X_val.npy"))
    y_val_34 = np.load(str(root / "data" / "processed" / "y_val.npy"))
    with open(str(root / "data" / "processed" / "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    y_val_8 = _load_8class_labels(root, y_val_34, "val")
    partition = get_partition()

    # Benign val dataset (for query encoding across all VAEs)
    benign_ds = PerClassDataset(X_val, y_val_8, BENIGN_CLASS_ID, scaler, partition)
    logger.info("Benign val samples: %d (will use up to %d)", len(benign_ds), max_benign_samples)

    # Subsample Benign to cap query cost
    n_benign = min(len(benign_ds), max_benign_samples)
    benign_loader = DataLoader(benign_ds, batch_size=2048, shuffle=False, num_workers=0)
    benign_x_list: list[torch.Tensor] = []
    n_collected = 0
    for batch in benign_loader:
        benign_x_list.append(batch["x_scaled"])
        n_collected += batch["x_scaled"].shape[0]
        if n_collected >= n_benign:
            break
    benign_x_tensor = torch.cat(benign_x_list, dim=0)[:n_benign]  # (n_benign, 39)

    # ------------------------------------------------------------------
    # Per-class pass
    # ------------------------------------------------------------------
    results: dict[str, Any] = {}

    for class_name in classes:
        class_id = CLASS_TO_ID[class_name]
        logger.info("=== Latent geometry for class %d (%s) ===", class_id, class_name)

        ckpt_path = root / "models" / "vae" / f"vae_class_{class_id}_{class_name}.pt"
        if not ckpt_path.exists():
            logger.warning("Checkpoint not found for %s — skipping", class_name)
            results[class_name] = {"error": f"checkpoint not found: {ckpt_path}"}
            continue

        ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
        ckpt_config = ckpt.get("config", dict(DEFAULT_CONFIG))

        (
            latent_dim,
            encoder_hidden,
            decoder_hidden,
            protocol_embed_dim,
            use_structured_continuous_decoder,
            structured_continuous_mode,
            structured_std_floor,
            latent_logvar_bounds,
        ) = _resolve_model_hparams(ckpt_config, class_name)

        model = MixedInputBetaVAE(
            partition=partition,
            latent_dim=latent_dim,
            protocol_embed_dim=protocol_embed_dim,
            encoder_hidden=encoder_hidden,
            decoder_hidden=decoder_hidden,
            n_pseudo_binary=0,
            use_structured_continuous_decoder=use_structured_continuous_decoder,
            structured_continuous_mode=structured_continuous_mode,
            structured_std_floor=structured_std_floor,
            latent_logvar_bounds=latent_logvar_bounds,
        )
        model.load_state_dict(ckpt["state_dict"], strict=False)
        model.register_protocol_references(scaler)
        model = model.to(device)
        model.eval()

        # 1. Fit latent stats from class's own val data
        class_ds = PerClassDataset(X_val, y_val_8, class_id, scaler, partition)
        logger.info("  Encoding own val split (%d samples) ...", len(class_ds))
        mu_class = _encode_dataset(model, class_ds, device)  # (N_class, k)
        stats = _fit_latent_stats(mu_class)
        mu_bar = stats["mu_bar"]
        lambda_mat = stats["lambda_mat"]
        k = stats["latent_dim"]

        # chi^2(0.95, k) threshold for outlier detection (IDSR boundary)
        chi2_threshold_95 = float(chi2.ppf(0.95, df=k))
        chi2_threshold_99 = float(chi2.ppf(0.99, df=k))

        # Self-consistency: fraction of class's own samples inside the 95% ellipsoid
        d_sq_own = _mahalanobis_sq(mu_class, mu_bar, lambda_mat)
        frac_inside_own_95 = float((d_sq_own <= chi2_threshold_95).mean())

        # 2. Encode Benign samples through THIS class's VAE
        logger.info("  Encoding %d Benign samples ...", n_benign)
        with torch.no_grad():
            mu_benign_t, _ = model.encode(benign_x_tensor.to(device))
        mu_benign = mu_benign_t.cpu().numpy()  # (n_benign, k)

        d_sq_benign = _mahalanobis_sq(mu_benign, mu_bar, lambda_mat)
        d_benign = np.sqrt(d_sq_benign.clip(min=0.0))

        frac_benign_inside_95 = float((d_sq_benign <= chi2_threshold_95).mean())
        frac_benign_inside_99 = float((d_sq_benign <= chi2_threshold_99).mean())

        # Percentile summary of Mahalanobis distances for Benign samples
        percentiles = [5, 25, 50, 75, 95, 99]
        d_percentiles = {
            str(p): float(np.percentile(d_benign, p)) for p in percentiles
        }

        logger.info(
            "  [%s] chi2_95=%.2f | own_in95=%.3f | benign_in95=%.3f | "
            "benign_median_maha=%.3f",
            class_name,
            chi2_threshold_95,
            frac_inside_own_95,
            frac_benign_inside_95,
            float(np.median(d_benign)),
        )

        results[class_name] = {
            "class_id": int(class_id),
            "latent_dim": int(k),
            "n_own_val_samples": int(len(class_ds)),
            "n_benign_query_samples": int(n_benign),
            # Latent center and precision (for downstream IDSR module)
            "mu_bar": mu_bar.tolist(),
            "sigma_diagonal": np.diag(stats["sigma"]).tolist(),
            "sigma_reg_diagonal": np.diag(stats["sigma_reg"]).tolist(),
            # Precision matrix stored for fast Mahalanobis computation at attack time
            "lambda_mat": lambda_mat.tolist(),
            "inversion_ok": bool(stats["inv_ok"]),
            # IDSR outlier thresholds
            "chi2_threshold_95": chi2_threshold_95,
            "chi2_threshold_99": chi2_threshold_99,
            # Self-consistency
            "frac_own_val_inside_95": frac_inside_own_95,
            # Benign-through-this-VAE statistics
            "benign_mahalanobis_percentiles": d_percentiles,
            "frac_benign_inside_95": frac_benign_inside_95,
            "frac_benign_inside_99": frac_benign_inside_99,
            "benign_mean_mahalanobis": float(d_benign.mean()),
            "benign_std_mahalanobis": float(d_benign.std()),
        }

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    out_dir = root / "results" / "vae"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "latent_geometry.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info("Latent geometry saved to %s", out_path)

    return results


def _print_summary(results: dict[str, Any]) -> None:
    print("\n=== Latent Geometry Summary ===")
    header = (
        f"{'Class':12s}  {'k':>3}  {'chi2_95':>8}  "
        f"{'own_in95':>9}  {'ben_in95':>9}  {'ben_in99':>9}  "
        f"{'ben_med_d':>10}  {'inv_ok':>7}"
    )
    print(header)
    print("-" * len(header))
    for class_name in CLASSES:
        r = results.get(class_name)
        if r is None or "error" in r:
            print(f"  {class_name:12s}  MISSING")
            continue
        print(
            f"  {class_name:12s}  {r['latent_dim']:3d}  "
            f"{r['chi2_threshold_95']:8.2f}  "
            f"{r['frac_own_val_inside_95']:9.3f}  "
            f"{r['frac_benign_inside_95']:9.3f}  "
            f"{r['frac_benign_inside_99']:9.3f}  "
            f"{r['benign_mahalanobis_percentiles']['50']:10.3f}  "
            f"{'YES' if r.get('inversion_ok') else 'NO':>7}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Latent-space geometry diagnostics.")
    parser.add_argument("--device", default="cpu", help="PyTorch device (default: cpu)")
    parser.add_argument(
        "--classes",
        default=None,
        help="Comma-separated class names (default: all 8)",
    )
    parser.add_argument(
        "--max-benign-samples",
        type=int,
        default=5000,
        help="Max Benign val samples to encode per class VAE (default: 5000)",
    )
    args = parser.parse_args()

    _setup_logging()

    classes = (
        [c.strip() for c in args.classes.split(",")]
        if args.classes
        else list(CLASSES)
    )
    results = run_latent_geometry(classes, args.device, args.max_benign_samples)
    _print_summary(results)
