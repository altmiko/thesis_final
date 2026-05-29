from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.spatial.distance import jensenshannon
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


SEED = 42
DEFAULT_CATEGORIES = ["DoS", "Mirai", "BruteForce"]


def load_bundle(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def select_subspace(bundle: dict[str, np.ndarray], mode: str) -> np.ndarray:
    n_features = int(np.asarray(bundle["X_original"]).shape[1])
    if mode == "full":
        return np.arange(n_features)
    mask_type = np.asarray(bundle.get("mask_type", []), dtype=str)
    if mask_type.size != n_features:
        return np.arange(n_features)
    mutable = np.flatnonzero(mask_type != "Frozen")
    return mutable if mutable.size else np.arange(n_features)


def js_divergence_1d(a: np.ndarray, b: np.ndarray, bins: np.ndarray) -> float:
    hist_a, _ = np.histogram(a, bins=bins, density=True)
    hist_b, _ = np.histogram(b, bins=bins, density=True)
    hist_a = hist_a.astype(np.float64) + 1e-12
    hist_b = hist_b.astype(np.float64) + 1e-12
    hist_a /= hist_a.sum()
    hist_b /= hist_b.sum()
    return float(jensenshannon(hist_a, hist_b, base=2.0) ** 2)


def draw_kde(
    ax: plt.Axes,
    values: np.ndarray,
    *,
    color: str,
    label: str,
    linestyle: str = "-",
    alpha: float = 0.35,
    hatch: str | None = None,
) -> None:
    sns.kdeplot(
        x=values,
        ax=ax,
        color=color,
        fill=False,
        linewidth=2.5,
        linestyle=linestyle,
        label=label,
        warn_singular=False,
        clip_on=False,
    )
    line = ax.lines[-1]
    x_data = line.get_xdata()
    y_data = line.get_ydata()
    ax.fill_between(
        x_data,
        y_data,
        0,
        facecolor=color,
        edgecolor=color,
        alpha=alpha,
        hatch=hatch,
        linewidth=0.0,
        zorder=1,
    )
    line.set_zorder(3)


def style_axis(ax: plt.Axes, title: str, show_ylabel: bool) -> None:
    ax.set_title(title, fontsize=20, fontweight="bold", pad=14)
    ax.set_xlabel("PCA Projection (mutable subspace)", fontsize=16, labelpad=10)
    ax.set_ylabel("Density" if show_ylabel else "", fontsize=16, labelpad=10)
    ax.grid(False)
    ax.tick_params(axis="both", labelsize=14, pad=6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)
    ax.margins(x=0.04)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Wide PCA-KDE distributional fidelity plot for CICIoT2023."
    )
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=Path("results/attacks/thesis_bundle.npz"),
        help="Path to thesis bundle NPZ.",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=DEFAULT_CATEGORIES,
        help="Attack categories to plot, one subplot per category.",
    )
    parser.add_argument(
        "--vae-key",
        default="X_adv_latent_cw",
        choices=["X_adv_latent_pgd", "X_adv_latent_cw"],
        help="Bundle key for the VAE-generated adversarial distribution.",
    )
    parser.add_argument(
        "--subspace",
        default="mutable",
        choices=["mutable", "full"],
        help="Feature subspace used for PCA projection.",
    )
    parser.add_argument(
        "--no-standardize",
        action="store_true",
        help="Disable per-category standardization before PCA.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("thesis_figures/distributional_fidelity.png"),
        help="Output PNG path.",
    )
    args = parser.parse_args()

    sns.set_theme(style="white")
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "axes.labelsize": 16,
            "axes.titlesize": 20,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 14,
            "savefig.dpi": 300,
            "figure.dpi": 120,
        }
    )

    bundle = load_bundle(args.artifacts)
    y_true = np.asarray(bundle["y_true"]).astype(str)
    x_original = np.asarray(bundle["X_original"], dtype=np.float32)
    x_pgd = np.asarray(bundle["X_adv_input_pgd"], dtype=np.float32)
    x_cw = np.asarray(bundle["X_adv_input_cw"], dtype=np.float32)
    x_vae = np.asarray(bundle[args.vae_key], dtype=np.float32)
    feature_idx = select_subspace(bundle, args.subspace)
    axis_label = (
        "PCA Projection (standardized mutable subspace)"
        if args.subspace == "mutable"
        else "PCA Projection (standardized full feature space)"
    )
    if args.no_standardize:
        axis_label = axis_label.replace("standardized ", "")

    colors = {
        "Original": "#7a7a7a",
        "PGD": "#4c9f50",
        "CW": "#d88432",
        "VAE (Ours)": "#2f6db5",
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, len(args.categories), figsize=(20, 7), sharey=True)
    if len(args.categories) == 1:
        axes = [axes]

    for idx, category in enumerate(args.categories):
        ax = axes[idx]
        mask = y_true == category
        if not np.any(mask):
            raise ValueError(f"Category '{category}' not found in bundle labels.")

        x_orig_cat = x_original[mask]
        x_pgd_cat = x_pgd[mask]
        x_cw_cat = x_cw[mask]
        x_vae_cat = x_vae[mask]

        x_orig_sub = x_orig_cat[:, feature_idx]
        x_pgd_sub = x_pgd_cat[:, feature_idx]
        x_cw_sub = x_cw_cat[:, feature_idx]
        x_vae_sub = x_vae_cat[:, feature_idx]
        if args.no_standardize:
            x_orig_plot = x_orig_sub
            x_pgd_plot = x_pgd_sub
            x_cw_plot = x_cw_sub
            x_vae_plot = x_vae_sub
        else:
            scaler = StandardScaler().fit(x_orig_sub)
            x_orig_plot = scaler.transform(x_orig_sub)
            x_pgd_plot = scaler.transform(x_pgd_sub)
            x_cw_plot = scaler.transform(x_cw_sub)
            x_vae_plot = scaler.transform(x_vae_sub)

        pca = PCA(n_components=1, random_state=SEED)
        pca.fit(x_orig_plot)

        proj_orig = pca.transform(x_orig_plot).ravel()
        proj_pgd = pca.transform(x_pgd_plot).ravel()
        proj_cw = pca.transform(x_cw_plot).ravel()
        proj_vae = pca.transform(x_vae_plot).ravel()

        combined = np.concatenate([proj_orig, proj_pgd, proj_cw, proj_vae])
        bins = np.histogram_bin_edges(combined, bins="fd")
        if len(bins) < 10:
            bins = np.linspace(combined.min(), combined.max(), 40)

        js_pgd = js_divergence_1d(proj_orig, proj_pgd, bins)
        js_cw = js_divergence_1d(proj_orig, proj_cw, bins)
        js_vae = js_divergence_1d(proj_orig, proj_vae, bins)

        print(
            f"{category}: JS(Original,PGD)={js_pgd:.4f} | "
            f"JS(Original,CW)={js_cw:.4f} | "
            f"JS(Original,VAE)={js_vae:.4f}"
        )

        draw_kde(
            ax,
            proj_orig,
            color=colors["Original"],
            label="Original",
            linestyle="--",
            alpha=0.30,
            hatch="///",
        )
        draw_kde(
            ax,
            proj_pgd,
            color=colors["PGD"],
            label="PGD",
            alpha=0.35,
        )
        draw_kde(
            ax,
            proj_cw,
            color=colors["CW"],
            label="CW",
            alpha=0.35,
        )
        draw_kde(
            ax,
            proj_vae,
            color=colors["VAE (Ours)"],
            label="VAE (Ours)",
            alpha=0.35,
        )

        style_axis(ax, category, show_ylabel=(idx == 0))
        ax.set_xlabel(axis_label, fontsize=16, labelpad=10)
        ax.text(
            0.98,
            0.97,
            f"Original-PGD JS: {js_pgd:.3f}\nOriginal-CW JS: {js_cw:.3f}\nOriginal-VAE JS: {js_vae:.3f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=12,
            bbox={
                "boxstyle": "round,pad=0.5",
                "facecolor": "white",
                "alpha": 0.8,
                "edgecolor": "lightgray",
            },
        )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.03),
        ncol=4,
        frameon=False,
        fontsize=14,
    )
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.17, top=0.78, wspace=0.40, hspace=0.40)
    fig.savefig(args.out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
