from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.signal import argrelextrema
from scipy.spatial.distance import jensenshannon
from scipy.stats import gaussian_kde
from scipy.stats import wasserstein_distance
from sklearn.decomposition import PCA

from attack.latent_infra import PerturbationMask, load_split
from vae.config import CLASSES


SEED = 42
RUG_MAX_POINTS = 500
METHOD_SPECS = [
    ("PGD", "X_adv_input_pgd", "#d62728"),
    ("C&W", "X_adv_input_cw", "#ff8c00"),
    ("VAE", "X_adv_latent_cw", "#1f77b4"),
]


def load_bundle(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def js_divergence_1d(a: np.ndarray, b: np.ndarray, bins: np.ndarray) -> float:
    hist_a, _ = np.histogram(a, bins=bins, density=True)
    hist_b, _ = np.histogram(b, bins=bins, density=True)
    hist_a = hist_a.astype(np.float64) + 1e-12
    hist_b = hist_b.astype(np.float64) + 1e-12
    hist_a /= hist_a.sum()
    hist_b /= hist_b.sum()
    return float(jensenshannon(hist_a, hist_b, base=2.0) ** 2)


def median_heuristic_bandwidth(x: np.ndarray, y: np.ndarray) -> float:
    stacked = np.concatenate([x, y]).reshape(-1, 1)
    diffs = np.abs(stacked - stacked.T)
    tri = diffs[np.triu_indices_from(diffs, k=1)]
    tri = tri[tri > 0]
    if tri.size == 0:
        return 1.0
    return float(np.median(tri))


def rbf_kernel_1d(x: np.ndarray, y: np.ndarray, gamma: float) -> np.ndarray:
    sqdist = (x[:, None] - y[None, :]) ** 2
    return np.exp(-gamma * sqdist)


def mmd_rbf_unbiased(x: np.ndarray, y: np.ndarray, sigma: float) -> float:
    if sigma <= 0:
        sigma = 1.0
    gamma = 1.0 / (2.0 * sigma * sigma)
    k_xx = rbf_kernel_1d(x, x, gamma)
    k_yy = rbf_kernel_1d(y, y, gamma)
    k_xy = rbf_kernel_1d(x, y, gamma)

    n = x.shape[0]
    m = y.shape[0]
    if n < 2 or m < 2:
        return float("nan")

    term_xx = (k_xx.sum() - np.trace(k_xx)) / (n * (n - 1))
    term_yy = (k_yy.sum() - np.trace(k_yy)) / (m * (m - 1))
    term_xy = 2.0 * k_xy.mean()
    return float(term_xx + term_yy - term_xy)


def find_kde_troughs(grid: np.ndarray, density: np.ndarray) -> np.ndarray:
    minima_idx = argrelextrema(density, np.less)[0]
    if minima_idx.size > 0:
        return grid[minima_idx]

    lo = np.quantile(grid, 0.10)
    hi = np.quantile(grid, 0.90)
    interior = np.where((grid >= lo) & (grid <= hi))[0]
    if interior.size == 0:
        return np.array([], dtype=np.float64)
    fallback_idx = interior[np.argmin(density[interior])]
    return np.array([grid[fallback_idx]], dtype=np.float64)


def correlation_difference(real: np.ndarray, adv: np.ndarray) -> tuple[np.ndarray, float]:
    with np.errstate(invalid="ignore", divide="ignore"):
        corr_real = np.corrcoef(real, rowvar=False)
        corr_adv = np.corrcoef(adv, rowvar=False)
    corr_real = np.nan_to_num(corr_real, nan=0.0, posinf=0.0, neginf=0.0)
    corr_adv = np.nan_to_num(corr_adv, nan=0.0, posinf=0.0, neginf=0.0)
    diff = np.abs(corr_real - corr_adv)
    return diff, float(np.nanmean(diff))


def add_rug(ax: plt.Axes, values: np.ndarray, color: str, y_level: float, rng: np.random.Generator) -> None:
    vals = values
    if vals.size > RUG_MAX_POINTS:
        vals = rng.choice(vals, size=RUG_MAX_POINTS, replace=False)
    ax.scatter(
        vals,
        np.full(vals.shape[0], y_level),
        marker="|",
        s=140,
        linewidths=1.2,
        color=color,
        alpha=0.85,
        clip_on=False,
        zorder=4,
    )


def sample_1d(values: np.ndarray, max_points: int, rng: np.random.Generator) -> np.ndarray:
    if max_points <= 0 or values.shape[0] <= max_points:
        return values
    idx = rng.choice(values.shape[0], size=max_points, replace=False)
    return values[idx]


def main() -> None:
    parser = argparse.ArgumentParser(description="Benign-vs-adversarial PCA KDE overlays with fidelity metrics.")
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=Path("results/attacks/thesis_bundle.npz"),
        help="Path to thesis bundle NPZ.",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=[name for name in CLASSES if name != "Benign"],
        help="Attack categories to render.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("thesis_figures/benign_overlay_fidelity"),
        help="Directory for per-category figures and summary CSV.",
    )
    parser.add_argument(
        "--vae-key",
        choices=["X_adv_latent_pgd", "X_adv_latent_cw"],
        default="X_adv_latent_cw",
        help="Bundle key for the VAE adversarial distribution.",
    )
    parser.add_argument(
        "--benign-sample-size",
        type=int,
        default=5000,
        help="Maximum benign samples used for KDE/summary overlays.",
    )
    parser.add_argument(
        "--metric-sample-size",
        type=int,
        default=2000,
        help="Maximum samples per side used by O(n^2) MMD/bandwidth metrics.",
    )
    args = parser.parse_args()

    rng = np.random.default_rng(SEED)

    sns.set_theme(style="white")
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "axes.labelsize": 13,
            "axes.titlesize": 16,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
            "savefig.dpi": 300,
            "figure.dpi": 120,
        }
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)

    bundle = load_bundle(args.artifacts)
    y_true = np.asarray(bundle["y_true"]).astype(str)

    split_test = load_split("test")
    benign_id = CLASSES.index("Benign")
    benign_mask = split_test["y_8"] == benign_id
    x_benign_full = np.asarray(split_test["X"][benign_mask], dtype=np.float32)

    perturb_mask = PerturbationMask.from_preprocessing_artifacts()
    mutable_idx = perturb_mask.full_indices + perturb_mask.partial_indices

    x_benign_mut_full = x_benign_full[:, mutable_idx]
    pca = PCA(n_components=1, random_state=SEED)
    benign_proj_full = pca.fit_transform(x_benign_mut_full).ravel()

    if args.benign_sample_size > 0 and benign_proj_full.shape[0] > args.benign_sample_size:
        benign_idx = rng.choice(
            benign_proj_full.shape[0],
            size=args.benign_sample_size,
            replace=False,
        )
        benign_proj = benign_proj_full[benign_idx]
        x_benign_mut = x_benign_mut_full[benign_idx]
    else:
        benign_proj = benign_proj_full
        x_benign_mut = x_benign_mut_full

    summary_rows: list[dict[str, object]] = []

    method_specs = [
        ("PGD", "X_adv_input_pgd", "#d62728"),
        ("C&W", "X_adv_input_cw", "#ff8c00"),
        ("VAE", args.vae_key, "#1f77b4"),
    ]

    for category in args.categories:
        category_mask = y_true == category
        if not np.any(category_mask):
            print(f"[WARN] Skipping {category}: not present in bundle.")
            continue

        projected_methods: dict[str, np.ndarray] = {}
        mutable_methods: dict[str, np.ndarray] = {}
        for label, key, _color in method_specs:
            x_adv = np.asarray(bundle[key][category_mask], dtype=np.float32)
            x_adv_mut = x_adv[:, mutable_idx]
            mutable_methods[label] = x_adv_mut
            projected_methods[label] = pca.transform(x_adv_mut).ravel()

        combined = [benign_proj]
        combined.extend(projected_methods.values())
        all_values = np.concatenate(combined)
        grid_pad = 0.08 * max(all_values.max() - all_values.min(), 1e-6)
        grid = np.linspace(all_values.min() - grid_pad, all_values.max() + grid_pad, 1024)

        benign_kde = gaussian_kde(benign_proj, bw_method="scott")
        benign_density = benign_kde(grid)
        troughs = find_kde_troughs(grid, benign_density)

        fig, ax = plt.subplots(figsize=(9.5, 5.6))
        sns.kdeplot(
            x=benign_proj,
            ax=ax,
            color="#808080",
            fill=True,
            alpha=0.35,
            linewidth=2.0,
            label="Benign",
            warn_singular=False,
        )

        y_min, y_max = ax.get_ylim()
        rug_levels = np.linspace(y_min - 0.02 * (y_max - y_min), y_min - 0.11 * (y_max - y_min), len(method_specs) + 1)
        add_rug(ax, benign_proj, "#808080", rug_levels[0], rng)

        metric_lines: list[str] = []
        for rug_idx, (label, _key, color) in enumerate(method_specs, start=1):
            values = projected_methods[label]
            x_adv_mut = mutable_methods[label]
            sns.kdeplot(
                x=values,
                ax=ax,
                color=color,
                fill=False,
                linewidth=2.2,
                label=label,
                warn_singular=False,
            )
            add_rug(ax, values, color, rug_levels[rug_idx], rng)

            bins = np.histogram_bin_edges(np.concatenate([benign_proj, values]), bins="fd")
            if len(bins) < 10:
                bins = np.linspace(min(benign_proj.min(), values.min()), max(benign_proj.max(), values.max()), 40)
            benign_metric = sample_1d(benign_proj, args.metric_sample_size, rng)
            values_metric = sample_1d(values, args.metric_sample_size, rng)
            js = js_divergence_1d(benign_metric, values_metric, bins)
            sigma = median_heuristic_bandwidth(benign_metric, values_metric)
            mmd = mmd_rbf_unbiased(benign_metric, values_metric, sigma=sigma)
            wass = float(wasserstein_distance(benign_metric, values_metric))
            log_lik = benign_kde.logpdf(values)
            ll_mean = float(np.mean(log_lik))
            ll_p5 = float(np.percentile(log_lik, 5))
            _corr_diff_mat, corr_diff_mean = correlation_difference(x_benign_mut, x_adv_mut)

            metric_lines.append(
                f"{label}: JS={js:.3f}, MMD={mmd:.3f}, W1={wass:.3f}, dCorr={corr_diff_mean:.3f}"
            )
            summary_rows.append(
                {
                    "category": category,
                    "method": label,
                    "js_divergence": js,
                    "mmd_rbf": mmd,
                    "wasserstein_pca": wass,
                    "corr_absdiff_mean": corr_diff_mean,
                    "loglik_mean": ll_mean,
                    "loglik_p5": ll_p5,
                    "n_samples": int(values.shape[0]),
                    "vae_key": args.vae_key if label == "VAE" else "",
                }
            )

        for trough in troughs:
            ax.axvline(trough, color="#555555", linestyle="--", linewidth=1.4, alpha=0.85, zorder=2)

        ax.set_title(f"{category}: Benign vs Adversarial PCA Density", fontsize=16, fontweight="bold", pad=12)
        ax.set_xlabel("PCA Projection (mutable subspace)", fontsize=13)
        ax.set_ylabel("Density", fontsize=13)
        ax.grid(False)
        ax.tick_params(axis="both", labelsize=11)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(True)
        ax.spines["bottom"].set_visible(True)
        ax.set_ylim(bottom=rug_levels[-1] - 0.01 * (y_max - y_min))
        ax.legend(loc="upper left", frameon=False)
        ax.text(
            0.98,
            0.98,
            "\n".join(metric_lines),
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9.5,
            bbox={
                "boxstyle": "round,pad=0.5",
                "facecolor": "white",
                "alpha": 0.85,
                "edgecolor": "lightgray",
            },
        )
        fig.tight_layout()

        safe_name = category.lower().replace("&", "and").replace(" ", "_")
        out_path = args.out_dir / f"{safe_name}_benign_overlay_kde.png"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"[OK] {category} -> {out_path}")

        pgd_diff, pgd_corr_mean = correlation_difference(x_benign_mut, mutable_methods["PGD"])
        vae_diff, vae_corr_mean = correlation_difference(x_benign_mut, mutable_methods["VAE"])
        fig_hm, axes_hm = plt.subplots(
            1,
            2,
            figsize=(12.5, 5.2),
            sharex=True,
            sharey=True,
            constrained_layout=True,
        )
        vmax = float(np.nanmax([pgd_diff, vae_diff]))
        for ax_hm, mat, title in zip(
            axes_hm,
            [pgd_diff, vae_diff],
            [f"PGD vs Benign (mean |delta r|={pgd_corr_mean:.3f})",
             f"VAE vs Benign (mean |delta r|={vae_corr_mean:.3f})"],
        ):
            sns.heatmap(
                mat,
                ax=ax_hm,
                cmap="mako",
                vmin=0.0,
                vmax=vmax if vmax > 0 else 1e-6,
                square=True,
                cbar=False,
                xticklabels=False,
                yticklabels=False,
            )
            ax_hm.set_title(title, fontsize=13, fontweight="bold", pad=10)
            ax_hm.set_xlabel("Mutable features", fontsize=11)
        axes_hm[0].set_ylabel("Mutable features", fontsize=11)
        cbar = fig_hm.colorbar(axes_hm[1].collections[0], ax=axes_hm, shrink=0.85, pad=0.02)
        cbar.set_label("|delta correlation|", fontsize=11)
        fig_hm.suptitle(f"{category}: Correlation Difference Heatmaps", fontsize=15, fontweight="bold", y=0.98)
        heatmap_path = args.out_dir / f"{safe_name}_correlation_difference_heatmap.png"
        fig_hm.savefig(heatmap_path, dpi=300, bbox_inches="tight")
        plt.close(fig_hm)
        print(f"[OK] {category} -> {heatmap_path}")

    summary_df = pd.DataFrame(summary_rows)
    summary_path = args.out_dir / "benign_overlay_metrics_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved summary metrics: {summary_path}")


if __name__ == "__main__":
    main()
