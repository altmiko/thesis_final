from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.spatial.distance import jensenshannon
from sklearn.decomposition import PCA


SEED = 42
DEFAULT_CATEGORIES = ["DoS", "Mirai", "BruteForce"]
INPUT_ATTACK_MAP = {
    "pgd": "X_adv_input_pgd",
    "cw": "X_adv_input_cw",
}
VAE_ATTACK_MAP = {
    "pgd": "X_adv_latent_pgd",
    "cw": "X_adv_latent_cw",
}


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


def style_axis(ax: plt.Axes, title: str) -> None:
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("PCA Projection", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)
    ax.tick_params(axis="both", labelsize=9)


def draw_kde(
    ax: plt.Axes,
    values: np.ndarray,
    *,
    color: str,
    label: str,
    linestyle: str = "-",
    alpha: float = 0.4,
    hatch: str | None = None,
) -> None:
    sns.kdeplot(
        x=values,
        ax=ax,
        color=color,
        fill=False,
        linewidth=1.8,
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
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="PCA-KDE distributional fidelity plot for CICIoT2023 attacks.")
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
        "--input-attack",
        choices=sorted(INPUT_ATTACK_MAP),
        default="cw",
        help="Unconstrained adversarial source.",
    )
    parser.add_argument(
        "--vae-attack",
        choices=sorted(VAE_ATTACK_MAP),
        default="cw",
        help="VAE-generated adversarial source.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("thesis_figures/distributional_fidelity.png"),
        help="Output PNG path.",
    )
    args = parser.parse_args()

    sns.set_style("white")
    plt.rcParams.update(
        {
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "savefig.dpi": 300,
            "figure.dpi": 120,
        }
    )

    bundle = load_bundle(args.artifacts)
    y_true = np.asarray(bundle["y_true"]).astype(str)
    x_original = np.asarray(bundle["X_original"], dtype=np.float32)
    x_input = np.asarray(bundle[INPUT_ATTACK_MAP[args.input_attack]], dtype=np.float32)
    x_vae = np.asarray(bundle[VAE_ATTACK_MAP[args.vae_attack]], dtype=np.float32)

    args.out.parent.mkdir(parents=True, exist_ok=True)

    n_cols = len(args.categories)
    fig_width = max(7.0, 2.3 * n_cols)
    fig, axes = plt.subplots(1, n_cols, figsize=(fig_width, 2.8), squeeze=False)

    for idx, category in enumerate(args.categories):
        ax = axes[0, idx]
        mask = y_true == category
        if not np.any(mask):
            raise ValueError(f"Category '{category}' not found in bundle labels.")

        x_orig_cat = x_original[mask]
        x_input_cat = x_input[mask]
        x_vae_cat = x_vae[mask]

        pca = PCA(n_components=1, random_state=SEED)
        pca.fit(x_orig_cat)

        proj_orig = pca.transform(x_orig_cat).ravel()
        proj_input = pca.transform(x_input_cat).ravel()
        proj_vae = pca.transform(x_vae_cat).ravel()

        combined = np.concatenate([proj_orig, proj_input, proj_vae])
        bins = np.histogram_bin_edges(combined, bins="fd")
        if len(bins) < 10:
            bins = np.linspace(combined.min(), combined.max(), 32)

        js_input = js_divergence_1d(proj_orig, proj_input, bins)
        js_vae = js_divergence_1d(proj_orig, proj_vae, bins)

        print(
            f"{category}: JS(Original,{args.input_attack.upper()})={js_input:.4f} | "
            f"JS(Original,VAE-{args.vae_attack.upper()})={js_vae:.4f}"
        )

        draw_kde(
            ax,
            proj_orig,
            color="#6e6e6e",
            label="Original",
            linestyle="--",
            alpha=0.30,
            hatch="///",
        )
        draw_kde(
            ax,
            proj_input,
            color="#5b8f29",
            label=args.input_attack.upper(),
            linestyle="-",
            alpha=0.40,
        )
        draw_kde(
            ax,
            proj_vae,
            color="#2b6cb0",
            label="VAE (Ours)",
            linestyle="-",
            alpha=0.40,
        )

        style_axis(ax, category)
        ax.legend(loc="upper left", frameon=False)
        ax.text(
            0.98,
            0.98,
            f"JS vs {args.input_attack.upper()}: {js_input:.3f}\nJS vs VAE: {js_vae:.3f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8.5,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#bbbbbb", "alpha": 0.9},
        )

    fig.tight_layout()
    fig.savefig(args.out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
