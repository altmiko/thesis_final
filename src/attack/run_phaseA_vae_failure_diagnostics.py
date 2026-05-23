from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = str(_REPO_ROOT / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from attack.latent_infra import (  # noqa: E402
    AttackRouter,
    AttackRunLogger,
    inverse_transform_scaled,
    load_split,
    set_global_seed,
)
from preprocessing.feature_groups import FEATURE_NAMES  # noqa: E402
from vae.config import CLASS_TO_ID  # noqa: E402

TARGET_CLASSES = ("BruteForce", "DoS")
HIST_BINS = 50
MU_HIST_BINS = 40


def _phaseA_config_snapshot(seed: int, device: str) -> dict:
    return {
        "phase": "phaseA",
        "seed": seed,
        "device": device,
        "split": "val",
        "target_classes": list(TARGET_CLASSES),
        "reconstruction_mode": "hard",
        "hist_bins": HIST_BINS,
        "mu_hist_bins": MU_HIST_BINS,
    }


def _batched_reconstructions(
    router: AttackRouter,
    class_id: int,
    x_scaled_np: np.ndarray,
    *,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    vae = router.get_vae(class_id)
    x_recon_batches: list[np.ndarray] = []
    mu_batches: list[np.ndarray] = []

    for start in range(0, len(x_scaled_np), batch_size):
        end = min(start + batch_size, len(x_scaled_np))
        x_batch = torch.from_numpy(x_scaled_np[start:end].astype(np.float32)).to(router.device)
        with torch.no_grad():
            mu, _ = vae.encode(x_batch)
            x_recon, _ = vae.decode_to_39(mu, router.scaler, mode="hard")
        x_recon_batches.append(x_recon.cpu().numpy())
        mu_batches.append(mu.cpu().numpy())

    return np.concatenate(x_recon_batches, axis=0), np.concatenate(mu_batches, axis=0)


def _error_metrics(x_raw: np.ndarray, x_recon_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sq_error = (x_raw - x_recon_raw) ** 2
    recon_error = np.linalg.norm(x_raw - x_recon_raw, axis=1) / np.clip(
        np.linalg.norm(x_raw, axis=1), 1e-12, None
    )
    return recon_error, sq_error


def _histogram(values: np.ndarray, bins: int) -> dict:
    counts, edges = np.histogram(values, bins=bins)
    return {
        "counts": counts.astype(int).tolist(),
        "bin_edges": edges.astype(float).tolist(),
    }


def _quantiles(values: np.ndarray, probs: tuple[float, ...] = (0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99)) -> dict:
    return {f"q{int(p * 100):02d}": float(np.quantile(values, p)) for p in probs}


def _feature_error_table(sq_error: np.ndarray) -> list[dict]:
    rows: list[dict] = []
    for feat_idx, feat_name in enumerate(FEATURE_NAMES):
        feat_errors = sq_error[:, feat_idx]
        rows.append(
            {
                "feature_index": feat_idx,
                "feature_name": feat_name,
                "mean_squared_error": float(feat_errors.mean()),
                "p95_squared_error": float(np.quantile(feat_errors, 0.95)),
            }
        )
    rows.sort(key=lambda row: row["mean_squared_error"], reverse=True)
    return rows


def _top_error_slice(recon_error: np.ndarray) -> np.ndarray:
    threshold = np.quantile(recon_error, 0.9)
    return recon_error >= threshold


def _protocol_distribution(x_raw: np.ndarray) -> dict[str, float]:
    proto_idx = FEATURE_NAMES.index("Protocol Type")
    values = np.round(x_raw[:, proto_idx]).astype(int)
    unique, counts = np.unique(values, return_counts=True)
    total = max(len(values), 1)
    return {str(int(v)): float(c / total) for v, c in zip(unique, counts)}


def _distribution_shift_summary(x_raw: np.ndarray, top_mask: np.ndarray) -> dict:
    overall = x_raw
    top = x_raw[top_mask]
    ttl_idx = FEATURE_NAMES.index("Time_To_Live")
    iat_idx = FEATURE_NAMES.index("IAT")

    shifts: list[dict] = []
    for feat_idx, feat_name in enumerate(FEATURE_NAMES):
        all_vals = overall[:, feat_idx]
        top_vals = top[:, feat_idx]
        all_mean = float(all_vals.mean())
        top_mean = float(top_vals.mean())
        all_std = float(all_vals.std())
        shift = (top_mean - all_mean) / (all_std + 1e-12)
        shifts.append(
            {
                "feature_index": feat_idx,
                "feature_name": feat_name,
                "overall_mean": all_mean,
                "top10_mean": top_mean,
                "standardized_mean_shift": float(shift),
            }
        )

    shifts.sort(key=lambda row: abs(row["standardized_mean_shift"]), reverse=True)
    return {
        "top10_fraction": float(top_mask.mean()),
        "protocol_distribution_overall": _protocol_distribution(overall),
        "protocol_distribution_top10": _protocol_distribution(top),
        "ttl_quantiles_overall": _quantiles(overall[:, ttl_idx]),
        "ttl_quantiles_top10": _quantiles(top[:, ttl_idx]),
        "iat_quantiles_overall": _quantiles(overall[:, iat_idx]),
        "iat_quantiles_top10": _quantiles(top[:, iat_idx]),
        "largest_distribution_shifts": shifts[:10],
    }


def _dominant_error_features(sq_error: np.ndarray, top_mask: np.ndarray) -> list[dict]:
    top_sq = sq_error[top_mask]
    mean_error = top_sq.mean(axis=0)
    total = float(mean_error.sum()) + 1e-12
    rows: list[dict] = []
    for feat_idx, feat_name in enumerate(FEATURE_NAMES):
        rows.append(
            {
                "feature_index": feat_idx,
                "feature_name": feat_name,
                "top10_mean_squared_error": float(mean_error[feat_idx]),
                "top10_error_share": float(mean_error[feat_idx] / total),
            }
        )
    rows.sort(key=lambda row: row["top10_mean_squared_error"], reverse=True)
    return rows


def _count_histogram_peaks(counts: np.ndarray) -> int:
    if counts.size < 3:
        return 0
    smoothed = np.convolve(counts.astype(np.float64), np.array([0.25, 0.5, 0.25]), mode="same")
    threshold = 0.1 * max(smoothed.max(), 1.0)
    peaks = 0
    for i in range(1, len(smoothed) - 1):
        if smoothed[i] >= threshold and smoothed[i] > smoothed[i - 1] and smoothed[i] >= smoothed[i + 1]:
            peaks += 1
    return peaks


def _save_bruteforce_mu_histograms(mu: np.ndarray, out_path: Path) -> list[dict]:
    n_dims = mu.shape[1]
    ncols = 4
    nrows = math.ceil(n_dims / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 3.5 * nrows))
    axes = np.array(axes).reshape(-1)

    rows: list[dict] = []
    for dim in range(n_dims):
        ax = axes[dim]
        counts, edges, _ = ax.hist(mu[:, dim], bins=MU_HIST_BINS, color="#3a78a1", alpha=0.85)
        peak_count = _count_histogram_peaks(counts)
        z = (mu[:, dim] - mu[:, dim].mean()) / (mu[:, dim].std() + 1e-12)
        outlier_frac = float((np.abs(z) > 3.0).mean())
        ax.set_title(f"dim {dim} | peaks={peak_count}")
        rows.append(
            {
                "dim": dim,
                "mean": float(mu[:, dim].mean()),
                "std": float(mu[:, dim].std()),
                "min": float(mu[:, dim].min()),
                "max": float(mu[:, dim].max()),
                "hist_counts": counts.astype(int).tolist(),
                "hist_bin_edges": edges.astype(float).tolist(),
                "peak_count": int(peak_count),
                "outlier_fraction_abs_z_gt_3": outlier_frac,
            }
        )

    for dim in range(n_dims, len(axes)):
        axes[dim].axis("off")

    fig.suptitle("BruteForce validation-set mu histograms", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return rows


def _summarize_findings(
    class_name: str,
    recon_error: np.ndarray,
    dominant_features: list[dict],
    shift_summary: dict,
) -> dict:
    protocol_overall = shift_summary["protocol_distribution_overall"]
    protocol_top10 = shift_summary["protocol_distribution_top10"]
    protocol_lines: list[dict] = []
    for proto, top_frac in sorted(protocol_top10.items(), key=lambda item: item[1], reverse=True):
        overall_frac = protocol_overall.get(proto, 0.0)
        enrich = top_frac / (overall_frac + 1e-12)
        protocol_lines.append(
            {
                "protocol_raw": int(proto),
                "overall_fraction": float(overall_frac),
                "top10_fraction": float(top_frac),
                "enrichment_ratio": float(enrich),
            }
        )

    return {
        "class_name": class_name,
        "n_samples": int(len(recon_error)),
        "reconstruction_error": {
            "median": float(np.quantile(recon_error, 0.5)),
            "p90": float(np.quantile(recon_error, 0.9)),
            "p95": float(np.quantile(recon_error, 0.95)),
            "p99": float(np.quantile(recon_error, 0.99)),
            "max": float(recon_error.max()),
        },
        "dominant_error_features_top10": dominant_features[:5],
        "most_shifted_features_top10": shift_summary["largest_distribution_shifts"][:5],
        "protocol_enrichment_top10": protocol_lines[:5],
        "ttl_top10_vs_overall": {
            "overall": shift_summary["ttl_quantiles_overall"],
            "top10": shift_summary["ttl_quantiles_top10"],
        },
        "iat_top10_vs_overall": {
            "overall": shift_summary["iat_quantiles_overall"],
            "top10": shift_summary["iat_quantiles_top10"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase A diagnostics for BruteForce and DoS VAE failures.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=2048)
    args = parser.parse_args()

    set_global_seed(args.seed)
    run_logger = AttackRunLogger.create(
        phase_name="phaseA",
        seed=args.seed,
        config_snapshot=_phaseA_config_snapshot(args.seed, args.device),
    )

    router = AttackRouter(device=args.device)
    split = load_split("val")
    diagnostics: dict[str, object] = {
        "seed": args.seed,
        "device": args.device,
        "split": split["split_name"],
        "classes": {},
    }

    for class_name in TARGET_CLASSES:
        class_id = CLASS_TO_ID[class_name]
        class_mask = split["y_8"] == class_id
        x_scaled = split["X"][class_mask].astype(np.float32)
        x_recon_scaled, mu = _batched_reconstructions(router, class_id, x_scaled, batch_size=args.batch_size)
        x_raw = inverse_transform_scaled(x_scaled, router.scaler).astype(np.float32)
        x_recon_raw = inverse_transform_scaled(x_recon_scaled, router.scaler).astype(np.float32)

        recon_error, sq_error = _error_metrics(x_raw, x_recon_raw)
        top_mask = _top_error_slice(recon_error)
        feature_error_rows = _feature_error_table(sq_error)
        dominant_features = _dominant_error_features(sq_error, top_mask)
        shift_summary = _distribution_shift_summary(x_raw, top_mask)

        class_result: dict[str, object] = {
            "n_samples": int(len(x_scaled)),
            "reconstruction_error_histogram": _histogram(recon_error, HIST_BINS),
            "reconstruction_error_summary": {
                "median": float(np.quantile(recon_error, 0.5)),
                "p90": float(np.quantile(recon_error, 0.9)),
                "p95": float(np.quantile(recon_error, 0.95)),
                "p99": float(np.quantile(recon_error, 0.99)),
                "max": float(recon_error.max()),
                "mean": float(recon_error.mean()),
            },
            "per_feature_squared_error_raw": feature_error_rows,
            "dominant_error_features_top10": dominant_features,
            "top10_error_distribution_shift": shift_summary,
        }

        if class_name == "BruteForce":
            hist_path = run_logger.run_dir / "bruteforce_mu_histograms.png"
            class_result["mu_histogram_figure"] = str(hist_path)
            class_result["mu_histograms"] = _save_bruteforce_mu_histograms(mu, hist_path)
        if class_name == "DoS":
            diag_json = json.load(open(_REPO_ROOT / "results" / "vae" / "diagnostics_DoS.json", encoding="utf-8"))
            class_result["collapsed_dims_confirmation"] = {
                "collapsed_dim_indices": diag_json["posterior_collapse"]["collapsed_dim_indices"],
                "per_dim_kl": diag_json["posterior_collapse"]["per_dim_kl"],
                "collapse_threshold": diag_json["posterior_collapse"]["collapse_threshold"],
            }

        diagnostics["classes"][class_name] = class_result
        run_logger.log(json.dumps(_summarize_findings(class_name, recon_error, dominant_features, shift_summary)))

    out_path = run_logger.run_dir / "phaseA_diagnostics.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(diagnostics, f, indent=2)

    print("=== Phase A Diagnostics ===")
    print(f"Output directory: {run_logger.run_dir}")
    print(f"Diagnostics JSON: {out_path}")
    print()

    for class_name in TARGET_CLASSES:
        info = diagnostics["classes"][class_name]
        summary = info["reconstruction_error_summary"]
        print(
            f"{class_name:10s} n={info['n_samples']} "
            f"median={summary['median']:.4f} p95={summary['p95']:.4f} "
            f"p99={summary['p99']:.4f} max={summary['max']:.4f}"
        )
        top_features = info["dominant_error_features_top10"][:5]
        print(
            "  dominant top-10% error features: "
            + ", ".join(
                f"{row['feature_name']} ({row['top10_error_share'] * 100.0:.1f}%)"
                for row in top_features
            )
        )
        shifts = info["top10_error_distribution_shift"]["largest_distribution_shifts"][:5]
        print(
            "  largest top-10% distribution shifts: "
            + ", ".join(
                f"{row['feature_name']} ({row['standardized_mean_shift']:+.2f} sd)"
                for row in shifts
            )
        )
        proto = info["top10_error_distribution_shift"]["protocol_distribution_top10"]
        print(f"  top-10% protocol mix: {proto}")
        if class_name == "DoS":
            collapsed = info["collapsed_dims_confirmation"]["collapsed_dim_indices"]
            print(f"  collapsed dims confirmed: {collapsed}")
        if class_name == "BruteForce":
            mu_rows = info["mu_histograms"]
            multimodal = [row["dim"] for row in mu_rows if row["peak_count"] >= 2]
            print(f"  BruteForce mu dims with >=2 histogram peaks: {multimodal}")
        print()


if __name__ == "__main__":
    main()
