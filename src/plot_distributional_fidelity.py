"""PCA-projected KDE density plot comparing original vs PGD/CW vs VAE adversarial traffic.

NetDiffuser-style distributional-fidelity figure for CICIoT2023.

Per attack category:
    1. Restrict to the 22 fully mutable features (PERTURBATION_MASK == 1.0) so the
       projection lives in the subspace where perturbations actually happen.
    2. Clip every column to [1st, 99th] percentile of the *original* samples to stop
       latent-attack outliers from compressing PCA.
    3. Fit PCA(1) on clipped originals; project all four populations.
    4. Overlay KDEs and annotate JS divergence vs the original.

Reads results/attacks/thesis_bundle.npz, writes the PNG given by --out.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.spatial.distance import jensenshannon
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "preprocessing"))
from feature_groups import PERTURBATION_MASK  # noqa: E402

SEED = 42
DEFAULT_CATEGORIES = ["DoS", "Mirai", "BruteForce"]

CURVES = [
    # (label,        bundle key,            color,    linestyle, alpha, hatch)
    ("Original",    "X_original",          "#6e6e6e", "--",      0.30, "///"),
    ("PGD",         "X_adv_input_pgd",     "#d62728", "-",       0.40, None),
    ("CW",          "X_adv_input_cw",      "#2ca02c", "-",       0.40, None),
    ("VAE (Ours)",  "X_adv_latent_pgd",    "#1f77b4", "-",       0.50, None),
]


def js_divergence_1d(a: np.ndarray, b: np.ndarray, bins: np.ndarray) -> float:
    ha, _ = np.histogram(a, bins=bins, density=True)
    hb, _ = np.histogram(b, bins=bins, density=True)
    ha = ha.astype(np.float64) + 1e-12
    hb = hb.astype(np.float64) + 1e-12
    ha /= ha.sum()
    hb /= hb.sum()
    return float(jensenshannon(ha, hb, base=2.0) ** 2)


def draw_kde(ax, values, *, color, label, linestyle, alpha, hatch):
    sns.kdeplot(x=values, ax=ax, color=color, fill=False, linewidth=1.6,
                linestyle=linestyle, label=label, warn_singular=False, clip_on=True)
    line = ax.lines[-1]
    ax.fill_between(line.get_xdata(), line.get_ydata(), 0,
                    facecolor=color, edgecolor=color, alpha=alpha,
                    hatch=hatch, linewidth=0.0)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--artifacts", type=Path, default=ROOT / "results/attacks/thesis_bundle.npz")
    p.add_argument("--categories", nargs="+", default=DEFAULT_CATEGORIES)
    p.add_argument("--vae-source", choices=["pgd", "cw"], default="pgd",
                   help="Which latent attack to label as 'VAE (Ours)'.")
    p.add_argument("--clip-pct", type=float, default=1.0,
                   help="Per-feature percentile clip range (clip-pct, 100-clip-pct).")
    p.add_argument("--out", type=Path, default=ROOT / "results/figures/distributional_fidelity.png")
    args = p.parse_args()

    curves = list(CURVES)
    curves[-1] = (curves[-1][0], f"X_adv_latent_{args.vae_source}",
                  curves[-1][2], curves[-1][3], curves[-1][4], curves[-1][5])

    mask = np.asarray(PERTURBATION_MASK) == 1.0
    mutable_idx = np.where(mask)[0]
    print(f"Restricting PCA to {mutable_idx.size} mutable features (PERTURBATION_MASK==1.0)")

    bundle = np.load(args.artifacts, allow_pickle=True)
    y_true = np.asarray(bundle["y_true"]).astype(str)
    arrays = {key: np.asarray(bundle[key], dtype=np.float32)[:, mutable_idx]
              for _, key, *_ in curves}
    X_orig = arrays["X_original"]

    # --- L2 sanity print (in mutable subspace) ----------------------------------
    print("\nL2 sanity (mutable subspace, mean per-sample distance vs Original):")
    for label, key, *_ in curves[1:]:
        d = np.linalg.norm(arrays[key] - X_orig, axis=1)
        print(f"  {label:<11s} mean={d.mean():.4f}  median={np.median(d):.4f}  max={d.max():.4f}")

    sns.set_style("white")
    plt.rcParams.update({"axes.labelsize": 11, "axes.titlesize": 11,
                         "legend.fontsize": 9, "xtick.labelsize": 9,
                         "ytick.labelsize": 9, "savefig.dpi": 300})

    n = len(args.categories)
    fig, axes = plt.subplots(1, n, figsize=(max(7.0, 2.5 * n), 3.0), squeeze=False)
    js_log: dict[str, dict[str, float]] = {}

    for col, category in enumerate(args.categories):
        ax = axes[0, col]
        sel = y_true == category
        if not sel.any():
            raise ValueError(f"Category {category!r} not in bundle.")

        Xo = X_orig[sel]
        lo = np.percentile(Xo, args.clip_pct, axis=0)
        hi = np.percentile(Xo, 100.0 - args.clip_pct, axis=0)
        eps = 1e-6
        hi = np.where(hi - lo < eps, lo + eps, hi)

        clipped = {key: np.clip(arrays[key][sel], lo, hi) for _, key, *_ in curves}
        pca = PCA(n_components=1, random_state=SEED).fit(clipped["X_original"])
        projections = {key: pca.transform(clipped[key]).ravel() for key in clipped}

        combined = np.concatenate(list(projections.values()))
        bins = np.histogram_bin_edges(combined, bins="fd")
        if len(bins) < 12:
            bins = np.linspace(combined.min(), combined.max(), 32)

        js_log[category] = {}
        ref = projections["X_original"]
        for label, key, color, ls, alpha, hatch in curves:
            draw_kde(ax, projections[key], color=color, label=label,
                     linestyle=ls, alpha=alpha, hatch=hatch)
            if key != "X_original":
                js_log[category][label] = js_divergence_1d(ref, projections[key], bins)

        ax.set_title(category)
        ax.set_xlabel("PCA Projection (mutable subspace)")
        ax.set_ylabel("Density" if col == 0 else "")
        ax.grid(False)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        if ax.get_legend() is not None:
            ax.get_legend().remove()

        text = "\n".join(f"JS({lbl})={v:.3f}" for lbl, v in js_log[category].items())
        ax.text(1.02, 0.98, text, transform=ax.transAxes, ha="left", va="top",
                fontsize=8.5,
                bbox={"boxstyle": "round,pad=0.3", "facecolor": "white",
                      "edgecolor": "#bbbbbb", "alpha": 0.9})

    handles = [plt.Line2D([0], [0], color=c[2], linestyle=c[3], linewidth=2.0, label=c[0])
               for c in curves]
    fig.legend(handles=handles, loc="upper center", ncol=len(curves),
               bbox_to_anchor=(0.5, 1.02), frameon=False)

    fig.tight_layout(rect=(0, 0, 0.92, 0.94))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("\nJS divergence (base 2) vs Original:")
    for cat, row in js_log.items():
        kv = "  ".join(f"{k}={v:.4f}" for k, v in row.items())
        print(f"  {cat:<12s} {kv}")
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
