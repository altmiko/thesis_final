"""Thesis visualization & statistical validation suite.

Generates the full Section A-F figure set for the VAE-based on-manifold
adversarial attack thesis. Reads pipeline artifacts produced upstream by
training/attack scripts and writes all PDFs + a single statistical
summary CSV to ./thesis_figures/.

Usage:
    python src/thesis_visualizations.py \
        --artifacts results/attacks/thesis_bundle.npz \
        --out thesis_figures

The artifacts file is expected to be a single .npz (or a directory of
.npy/.npz files -- both are supported) containing the keys documented
in `REQUIRED_KEYS` below. Missing optional keys cause the corresponding
figure(s) to be skipped with a warning rather than a crash.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Global config
# ---------------------------------------------------------------------------
SEED = 42
np.random.seed(SEED)

sns.set_style("whitegrid")
plt.rcParams.update({
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.dpi": 300,
    "figure.dpi": 120,
})

PALETTE = {
    "original":   "#1f77b4",
    "latent_pgd": "#d62728",
    "latent_cw":  "#2ca02c",
    "input_pgd":  "#ff7f0e",
    "input_cw":   "#9467bd",
    "evaded":     "#d62728",
    "not_evaded": "#999999",
    "Full":       "#2ca02c",
    "Partial":    "#ffbf00",
    "Frozen":     "#d62728",
}

SCATTER_COLORS = {
    "original": "#000000",
    "latent_pgd": "#e7a3a3",
    "latent_cw": "#a9d8b1",
    "input_pgd": "#f4c59a",
    "input_cw": "#c9b8e8",
}
SCATTER_ALPHA_ORIGINAL = 0.95
SCATTER_ALPHA_ADV = 0.80

CATEGORY_PALETTE = "tab10"

ATTACK_TYPES = ["latent_pgd", "latent_cw", "input_pgd", "input_cw"]
ATTACK_LABELS = {
    "latent_pgd": "Latent-PGD",
    "latent_cw":  "Latent-CW",
    "input_pgd":  "Input-PGD",
    "input_cw":   "Input-CW",
}

REQUIRED_KEYS = [
    "X_original", "X_adv_latent_pgd", "X_adv_latent_cw",
    "X_adv_input_pgd", "X_adv_input_cw",
    "z_original", "z_perturbed_pgd", "z_perturbed_cw",
    "y_true", "y_pred_original",
    "y_pred_adv_latent_pgd", "y_pred_adv_latent_cw",
    "y_pred_adv_input_pgd",  "y_pred_adv_input_cw",
    "evasion_mask_latent_pgd", "evasion_mask_latent_cw",
    "evasion_mask_input_pgd",  "evasion_mask_input_cw",
    "protocol_valid_latent_pgd", "protocol_valid_latent_cw",
    "protocol_valid_input_pgd",  "protocol_valid_input_cw",
    "feature_names", "mask_type", "X_reconstructed",
]

OPTIONAL_KEYS = ["multimetric_table", "category_asr_table"]


def load_artifacts(path: Path) -> Dict[str, np.ndarray]:
    data: Dict[str, np.ndarray] = {}
    if path.is_file():
        with np.load(path, allow_pickle=True) as npz:
            for k in npz.files:
                data[k] = npz[k]
    elif path.is_dir():
        for f in path.iterdir():
            if f.suffix == ".npy":
                data[f.stem] = np.load(f, allow_pickle=True)
            elif f.suffix == ".npz":
                with np.load(f, allow_pickle=True) as npz:
                    for k in npz.files:
                        data[k] = npz[k]
            elif f.suffix == ".csv" and f.stem in OPTIONAL_KEYS:
                data[f.stem] = pd.read_csv(f)
    else:
        raise FileNotFoundError(f"Artifacts path not found: {path}")

    missing = [k for k in REQUIRED_KEYS if k not in data]
    if missing:
        print(f"[WARN] Missing required keys (some figures will be skipped): {missing}")
    return data


def save_fig(fig, out_dir: Path, name: str, checklist: dict) -> None:
    out = out_dir / name
    try:
        fig.tight_layout()
        fig.savefig(out, format="pdf", bbox_inches="tight")
        checklist[name] = "OK"
        print(f"  [OK]   {name}")
    except Exception as e:
        checklist[name] = f"FAIL: {e}"
        print(f"  [FAIL] {name}: {e}")
    finally:
        plt.close(fig)


def _try(section: str, fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except KeyError as e:
        print(f"[SKIP {section}] missing input: {e}")
    except Exception as e:
        print(f"[ERR  {section}] {type(e).__name__}: {e}")


def _tsne(X: np.ndarray, perplexity: int = 30, n_iter: int = 1000) -> np.ndarray:
    n = X.shape[0]
    perp = min(perplexity, max(5, (n - 1) // 3))
    kwargs = {
        "n_components": 2,
        "perplexity": perp,
        "random_state": SEED,
        "init": "pca",
        "learning_rate": "auto",
    }
    try:
        return TSNE(max_iter=n_iter, **kwargs).fit_transform(X)
    except TypeError:
        return TSNE(n_iter=n_iter, **kwargs).fit_transform(X)


def _umap(X: np.ndarray) -> Optional[np.ndarray]:
    try:
        import umap
    except ImportError:
        print("  [SKIP UMAP] umap-learn not installed")
        return None
    reducer = umap.UMAP(
        n_neighbors=15, min_dist=0.1, metric="euclidean", random_state=SEED,
    )
    return reducer.fit_transform(X)


def _labels(y: np.ndarray) -> np.ndarray:
    return y.astype(str)


def _table_to_df(table) -> pd.DataFrame:
    if isinstance(table, pd.DataFrame):
        return table.copy()
    if isinstance(table, np.ndarray) and getattr(table.dtype, "names", None):
        return pd.DataFrame.from_records(table)
    if isinstance(table, np.recarray):
        return pd.DataFrame.from_records(table)
    if hasattr(table, "item"):
        return pd.DataFrame(table.item())
    return pd.DataFrame(table)


# ---------------------------------------------------------------------------
# SECTION A
# ---------------------------------------------------------------------------
def section_a(data, out, checklist):
    print("\n[Section A] Dimensionality reduction")
    z = data["z_original"]
    y = _labels(data["y_true"])
    categories = sorted(np.unique(y).tolist())
    cat_to_color = dict(zip(categories, sns.color_palette(CATEGORY_PALETTE, len(categories))))

    def a1():
        emb = _tsne(z)
        fig, ax = plt.subplots(figsize=(7, 6))
        for c in categories:
            m = y == c
            ax.scatter(emb[m, 0], emb[m, 1], s=8, alpha=0.6, color=cat_to_color[c], label=c)
        ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
        ax.set_title("VAE Latent Space by Attack Category")
        ax.legend(markerscale=2)
        save_fig(fig, out, "tsne_latent_by_category.pdf", checklist)
    _try("A1", a1)

    def a2():
        zp = data["z_perturbed_pgd"]; zc = data["z_perturbed_cw"]
        combo = np.vstack([z, zp, zc])
        emb = _tsne(combo); n = z.shape[0]
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(emb[:n, 0], emb[:n, 1], s=8, alpha=SCATTER_ALPHA_ORIGINAL,
                   c=SCATTER_COLORS["original"], marker="o", label="Original")
        ax.scatter(emb[n:2*n, 0], emb[n:2*n, 1], s=8, alpha=SCATTER_ALPHA_ADV,
                   c=SCATTER_COLORS["latent_pgd"], marker="^", label="Latent-PGD")
        ax.scatter(emb[2*n:, 0], emb[2*n:, 1], s=8, alpha=SCATTER_ALPHA_ADV,
                   c=SCATTER_COLORS["latent_cw"], marker="D", label="Latent-CW")
        ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
        ax.set_title("Latent Space: Original vs Perturbed")
        ax.legend(markerscale=2)
        save_fig(fig, out, "tsne_latent_original_vs_perturbed.pdf", checklist)
    _try("A2", a2)

    def a3():
        Xo = data["X_original"]; Xlat = data["X_adv_latent_cw"]; Xin = data["X_adv_input_cw"]
        combo = np.vstack([Xo, Xlat, Xin])
        emb = _tsne(combo); n = Xo.shape[0]
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(emb[:n, 0], emb[:n, 1], s=8, alpha=SCATTER_ALPHA_ORIGINAL,
                   c=SCATTER_COLORS["original"], label="Original")
        ax.scatter(emb[n:2*n, 0], emb[n:2*n, 1], s=8, alpha=SCATTER_ALPHA_ADV,
                   c=SCATTER_COLORS["latent_cw"], label="Latent-Adv (CW)")
        ax.scatter(emb[2*n:, 0], emb[2*n:, 1], s=8, alpha=SCATTER_ALPHA_ADV,
                   c=SCATTER_COLORS["input_cw"], label="Input-Adv (CW)")
        ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
        ax.set_title("Input Space: On-Manifold vs Off-Manifold Attacks")
        ax.legend(markerscale=2)
        save_fig(fig, out, "tsne_input_space_comparison.pdf", checklist)
    _try("A3", a3)

    def a4():
        emb = _umap(z)
        if emb is not None:
            fig, ax = plt.subplots(figsize=(7, 6))
            for c in categories:
                m = y == c
                ax.scatter(emb[m, 0], emb[m, 1], s=8, alpha=0.6, color=cat_to_color[c], label=c)
            ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
            ax.set_title("VAE Latent Space (UMAP) by Attack Category")
            ax.legend(markerscale=2)
            save_fig(fig, out, "umap_latent_by_category.pdf", checklist)

        zp = data["z_perturbed_pgd"]; zc = data["z_perturbed_cw"]
        emb = _umap(np.vstack([z, zp, zc]))
        if emb is not None:
            n = z.shape[0]
            fig, ax = plt.subplots(figsize=(7, 6))
            ax.scatter(emb[:n, 0], emb[:n, 1], s=8, alpha=SCATTER_ALPHA_ORIGINAL,
                       c=SCATTER_COLORS["original"], marker="o", label="Original")
            ax.scatter(emb[n:2*n, 0], emb[n:2*n, 1], s=8, alpha=SCATTER_ALPHA_ADV,
                       c=SCATTER_COLORS["latent_pgd"], marker="^", label="Latent-PGD")
            ax.scatter(emb[2*n:, 0], emb[2*n:, 1], s=8, alpha=SCATTER_ALPHA_ADV,
                       c=SCATTER_COLORS["latent_cw"], marker="D", label="Latent-CW")
            ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
            ax.set_title("Latent Space (UMAP): Original vs Perturbed")
            ax.legend(markerscale=2)
            save_fig(fig, out, "umap_latent_original_vs_perturbed.pdf", checklist)

        Xo = data["X_original"]; Xlat = data["X_adv_latent_cw"]; Xin = data["X_adv_input_cw"]
        emb = _umap(np.vstack([Xo, Xlat, Xin]))
        if emb is not None:
            n = Xo.shape[0]
            fig, ax = plt.subplots(figsize=(7, 6))
            ax.scatter(emb[:n, 0], emb[:n, 1], s=8, alpha=SCATTER_ALPHA_ORIGINAL,
                       c=SCATTER_COLORS["original"], label="Original")
            ax.scatter(emb[n:2*n, 0], emb[n:2*n, 1], s=8, alpha=SCATTER_ALPHA_ADV,
                       c=SCATTER_COLORS["latent_cw"], label="Latent-Adv (CW)")
            ax.scatter(emb[2*n:, 0], emb[2*n:, 1], s=8, alpha=SCATTER_ALPHA_ADV,
                       c=SCATTER_COLORS["input_cw"], label="Input-Adv (CW)")
            ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
            ax.set_title("Input Space (UMAP): On-Manifold vs Off-Manifold Attacks")
            ax.legend(markerscale=2)
            save_fig(fig, out, "umap_input_space_comparison.pdf", checklist)
    _try("A4", a4)

    pca = PCA(n_components=2, random_state=SEED).fit(z)
    z_pca = pca.transform(z); ev = pca.explained_variance_ratio_

    def a5():
        fig, ax = plt.subplots(figsize=(7, 6))
        for c in categories:
            m = y == c
            ax.scatter(z_pca[m, 0], z_pca[m, 1], s=8, alpha=0.6, color=cat_to_color[c], label=c)
        ax.set_xlabel(f"PC1 ({ev[0]*100:.1f}%)")
        ax.set_ylabel(f"PC2 ({ev[1]*100:.1f}%)")
        ax.set_title("PCA of VAE Latent Space by Attack Category")
        ax.legend(markerscale=2)
        save_fig(fig, out, "pca_latent_by_category.pdf", checklist)
    _try("A5", a5)

    def a6():
        ev_mask = np.asarray(data["evasion_mask_latent_cw"]).astype(bool)
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(z_pca[~ev_mask, 0], z_pca[~ev_mask, 1], s=8, alpha=0.4, c=PALETTE["not_evaded"], label="Not evaded")
        ax.scatter(z_pca[ev_mask, 0], z_pca[ev_mask, 1], s=8, alpha=0.5, c=PALETTE["evaded"], label="Evaded")
        for c in categories:
            m = y == c
            if m.sum() == 0:
                continue
            cx, cy = z_pca[m, 0].mean(), z_pca[m, 1].mean()
            ax.text(cx, cy, c, fontsize=9, weight="bold", ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="black", alpha=0.7))
        ax.set_xlabel(f"PC1 ({ev[0]*100:.1f}%)")
        ax.set_ylabel(f"PC2 ({ev[1]*100:.1f}%)")
        ax.set_title("Latent PCA by Evasion Success (Latent-CW)")
        ax.legend(markerscale=2)
        save_fig(fig, out, "pca_latent_by_evasion.pdf", checklist)
    _try("A6", a6)

    def a7():
        emb = _umap(z)
        if emb is None:
            return
        ev_mask = np.asarray(data["evasion_mask_latent_cw"]).astype(bool)
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(emb[~ev_mask, 0], emb[~ev_mask, 1], s=8, alpha=0.4, c=PALETTE["not_evaded"], label="Not evaded")
        ax.scatter(emb[ev_mask, 0], emb[ev_mask, 1], s=8, alpha=0.5, c=PALETTE["evaded"], label="Evaded")
        for c in categories:
            m = y == c
            if m.sum() == 0:
                continue
            cx, cy = emb[m, 0].mean(), emb[m, 1].mean()
            ax.text(cx, cy, c, fontsize=9, weight="bold", ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="black", alpha=0.7))
        ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
        ax.set_title("Latent UMAP by Evasion Success (Latent-CW)")
        ax.legend(markerscale=2)
        save_fig(fig, out, "umap_latent_by_evasion.pdf", checklist)
    _try("A7", a7)


# ---------------------------------------------------------------------------
# SECTION B
# ---------------------------------------------------------------------------
def section_b(data, out, checklist, stats_rows):
    print("\n[Section B] Distributional fidelity")
    Xo = data["X_original"]; Xlat = data["X_adv_latent_cw"]; Xin = data["X_adv_input_cw"]
    feat_names = [str(f) for f in np.asarray(data["feature_names"]).tolist()]
    mask_type = [str(m) for m in np.asarray(data["mask_type"]).tolist()]

    delta_latent = np.abs(Xlat - Xo).mean(axis=0)
    delta_input = np.abs(Xin - Xo).mean(axis=0)
    composite = np.maximum(delta_latent, delta_input)
    top10 = np.argsort(-composite)[:10]

    def b1():
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        for ax, idx in zip(axes.ravel(), top10):
            try:
                sns.kdeplot(Xo[:, idx], ax=ax, color=PALETTE["original"], label="Original", warn_singular=False)
                sns.kdeplot(Xlat[:, idx], ax=ax, color="orange", label="Latent-CW", warn_singular=False)
                sns.kdeplot(Xin[:, idx], ax=ax, color="red", label="Input-CW", warn_singular=False)
            except Exception:
                pass
            ax.set_title(feat_names[idx], fontsize=9)
            ax.set_xlabel(""); ax.set_ylabel("")
        axes[0, 0].legend(fontsize=8)
        fig.suptitle("KDE: Top-10 Most-Perturbed Features", fontsize=12)
        save_fig(fig, out, "kde_top10_features.pdf", checklist)
    _try("B1", b1)

    def b2():
        c_o = np.corrcoef(Xo, rowvar=False)
        c_l = np.corrcoef(Xlat, rowvar=False)
        c_i = np.corrcoef(Xin, rowvar=False)
        tri = np.tril(np.ones_like(c_o, dtype=bool))
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        for ax, M, ttl in zip(axes,
                              [np.where(tri, c_o, np.nan),
                               np.where(tri, c_l, np.nan),
                               np.where(tri, c_i, np.nan)],
                              ["Original", "Latent-CW Adv", "Input-CW Adv"]):
            im = ax.imshow(M, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
            ax.set_title(ttl); ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=axes, shrink=0.7, label="Pearson r")
        save_fig(fig, out, "correlation_heatmap_comparison.pdf", checklist)
    _try("B2", b2)

    def b3():
        c_o = np.corrcoef(Xo, rowvar=False)
        c_l = np.corrcoef(Xlat, rowvar=False)
        c_i = np.corrcoef(Xin, rowvar=False)
        d_l = np.abs(c_o - c_l); d_i = np.abs(c_o - c_i)
        vmax = max(np.nanmax(d_l), np.nanmax(d_i))
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for ax, M, ttl in zip(axes, [d_l, d_i],
                              ["|corr(Orig) - corr(Latent-Adv)|",
                               "|corr(Orig) - corr(Input-Adv)|"]):
            im = ax.imshow(M, cmap="viridis", vmin=0, vmax=vmax, aspect="auto")
            ax.set_title(ttl); ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=axes, shrink=0.7, label="|delta correlation|")
        save_fig(fig, out, "correlation_difference.pdf", checklist)
    _try("B3", b3)

    def b4():
        rows = []
        for i, fn in enumerate(feat_names):
            ks_l = stats.ks_2samp(Xo[:, i], Xlat[:, i])
            ks_i = stats.ks_2samp(Xo[:, i], Xin[:, i])
            try:
                w_l = stats.wasserstein_distance(Xo[:, i], Xlat[:, i])
                w_i = stats.wasserstein_distance(Xo[:, i], Xin[:, i])
            except Exception:
                w_l = w_i = np.nan
            rows.append({
                "feature_name": fn,
                "mask_type": mask_type[i] if i < len(mask_type) else "?",
                "ks_stat_latent": ks_l.statistic,
                "p_value_latent": ks_l.pvalue,
                "ks_stat_input":  ks_i.statistic,
                "p_value_input":  ks_i.pvalue,
                "wasserstein_latent": w_l,
                "wasserstein_input":  w_i,
                "significant_latent_005": bool(ks_l.pvalue < 0.05),
                "significant_input_005":  bool(ks_i.pvalue < 0.05),
            })
        df = pd.DataFrame(rows)
        df.to_csv(out / "ks_wasserstein_results.csv", index=False)
        checklist["ks_wasserstein_results.csv"] = "OK"
        for r in rows:
            stats_rows.append({"test": "KS_latent", "feature": r["feature_name"],
                               "statistic": r["ks_stat_latent"], "p_value": r["p_value_latent"]})
            stats_rows.append({"test": "KS_input", "feature": r["feature_name"],
                               "statistic": r["ks_stat_input"], "p_value": r["p_value_input"]})
            stats_rows.append({"test": "Wasserstein_latent", "feature": r["feature_name"],
                               "statistic": r["wasserstein_latent"], "p_value": np.nan})
            stats_rows.append({"test": "Wasserstein_input", "feature": r["feature_name"],
                               "statistic": r["wasserstein_input"], "p_value": np.nan})

        x = np.arange(len(feat_names)); width = 0.4
        fig, ax = plt.subplots(figsize=(14, 5))
        colors = [PALETTE.get(mt, "gray") for mt in df["mask_type"]]
        ax.bar(x - width/2, df["ks_stat_latent"], width, color=colors, edgecolor="black", lw=0.4, label="Latent-CW")
        ax.bar(x + width/2, df["ks_stat_input"],  width, color=colors, edgecolor="black", lw=0.4, hatch="//", label="Input-CW")
        ax.set_xticks(x); ax.set_xticklabels(feat_names, rotation=90, fontsize=7)
        ax.set_ylabel("KS statistic")
        ax.set_title("Per-Feature KS Statistic (color = perturbation mask type)")
        handles = [Line2D([0], [0], color=PALETTE["Full"], lw=8, label="Full"),
                   Line2D([0], [0], color=PALETTE["Partial"], lw=8, label="Partial"),
                   Line2D([0], [0], color=PALETTE["Frozen"], lw=8, label="Frozen"),
                   Line2D([0], [0], color="black", lw=0, marker="s", label="Latent-CW (solid)"),
                   Line2D([0], [0], color="black", lw=0, marker="s",
                          markerfacecolor="white", label="Input-CW (hatched)")]
        ax.legend(handles=handles, fontsize=8)
        save_fig(fig, out, "ks_statistic_by_feature.pdf", checklist)
    _try("B4", b4)


# ---------------------------------------------------------------------------
# SECTION C
# ---------------------------------------------------------------------------
def section_c(data, out, checklist):
    print("\n[Section C] Perturbation analysis")
    Xo = data["X_original"]
    feat_names = [str(f) for f in np.asarray(data["feature_names"]).tolist()]
    mask_type = [str(m) for m in np.asarray(data["mask_type"]).tolist()]
    y = _labels(data["y_true"])
    adv = {at: data[f"X_adv_{at}"] for at in ATTACK_TYPES}

    def c1():
        mat = np.column_stack([np.abs(adv[at] - Xo).mean(axis=0) for at in ATTACK_TYPES])
        order = sorted(range(len(feat_names)),
                       key=lambda i: {"Full": 0, "Partial": 1, "Frozen": 2}.get(mask_type[i], 3))
        mat = mat[order]
        ordered_names = [feat_names[i] for i in order]
        ordered_masks = [mask_type[i] for i in order]
        fig, ax = plt.subplots(figsize=(7, 12))
        sns.heatmap(mat, annot=True, fmt=".2f",
                    xticklabels=[ATTACK_LABELS[a] for a in ATTACK_TYPES],
                    yticklabels=ordered_names, cmap="rocket_r", ax=ax,
                    cbar_kws={"label": "mean |delta|"}, annot_kws={"size": 7})
        for i, mt in enumerate(ordered_masks):
            ax.add_patch(plt.Rectangle((-0.5, i), 0.4, 1, color=PALETTE.get(mt, "gray"),
                                        clip_on=False, lw=0))
        ax.set_title("Mean Absolute Perturbation per Feature x Attack")
        save_fig(fig, out, "perturbation_heatmap.pdf", checklist)
    _try("C1", c1)

    def c2():
        fig, ax = plt.subplots(figsize=(7, 5))
        for at in ATTACK_TYPES:
            d = np.linalg.norm(adv[at] - Xo, axis=1)
            ds = np.sort(d); cdf = np.arange(1, len(ds) + 1) / len(ds)
            line, = ax.plot(ds, cdf, label=ATTACK_LABELS[at], color=PALETTE[at])
            ax.axvline(d.mean(), color=line.get_color(), ls="--", alpha=0.6)
        ax.set_xlabel("Per-sample L2 perturbation norm")
        ax.set_ylabel("Cumulative fraction of samples")
        ax.set_title("Empirical CDF of L2 Distortion")
        ax.legend()
        save_fig(fig, out, "l2_distortion_cdf.pdf", checklist)
    _try("C2", c2)

    def c3():
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        for ax, at in zip(axes.ravel(), ATTACK_TYPES):
            d = np.linalg.norm(adv[at] - Xo, axis=1)
            df = pd.DataFrame({"L2": d, "Category": y})
            sns.boxplot(data=df, x="Category", y="L2", ax=ax,
                        color=PALETTE[at], showfliers=False)
            ax.set_title(ATTACK_LABELS[at])
            ax.tick_params(axis="x", rotation=30)
        fig.suptitle("Per-Sample L2 Distortion by Attack Category", fontsize=13)
        save_fig(fig, out, "l2_by_category_boxplot.pdf", checklist)
    _try("C3", c3)


# ---------------------------------------------------------------------------
# SECTION D
# ---------------------------------------------------------------------------
def section_d(data, out, checklist):
    print("\n[Section D] VAE reconstruction quality")
    Xo = data["X_original"]; Xr = data["X_reconstructed"]
    feat_names = [str(f) for f in np.asarray(data["feature_names"]).tolist()]
    mask_type = [str(m) for m in np.asarray(data["mask_type"]).tolist()]
    Xlat = data.get("X_adv_latent_cw", Xo)
    top6 = np.argsort(-np.abs(Xlat - Xo).mean(axis=0))[:6]

    def d1():
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        for ax, idx in zip(axes.ravel(), top6):
            o = Xo[:, idx]; r = Xr[:, idx]
            ax.scatter(o, r, s=4, alpha=0.3, color=PALETTE["original"])
            lo, hi = min(o.min(), r.min()), max(o.max(), r.max())
            ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, label="y=x")
            try:
                slope, intercept, r_val, *_ = stats.linregress(o, r)
                xs = np.linspace(lo, hi, 50)
                ax.plot(xs, slope * xs + intercept, color="red", lw=1, label=f"R^2={r_val**2:.2f}")
            except Exception:
                pass
            ax.set_title(feat_names[idx], fontsize=10)
            ax.set_xlabel("Original"); ax.set_ylabel("Reconstructed")
            ax.legend(fontsize=8)
        fig.suptitle("VAE Reconstruction Scatter: Top-6 Perturbed Features", fontsize=12)
        save_fig(fig, out, "vae_reconstruction_scatter.pdf", checklist)
    _try("D1", d1)

    def d2():
        mse = ((Xo - Xr) ** 2).mean(axis=1)
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.hist(mse, bins=60, color=PALETTE["original"], alpha=0.7, edgecolor="black", lw=0.4)
        ax.axvline(mse.mean(), color="red", ls="--", label=f"mean={mse.mean():.4f}")
        ax.axvline(np.median(mse), color="black", ls=":", label=f"median={np.median(mse):.4f}")
        ax.set_xlabel("Per-sample MSE"); ax.set_ylabel("Count")
        ax.set_title("VAE Reconstruction Error Distribution")
        ax.legend()
        save_fig(fig, out, "vae_reconstruction_error_hist.pdf", checklist)
    _try("D2", d2)

    def d3():
        mae = np.abs(Xo - Xr).mean(axis=0)
        colors = [PALETTE.get(mt, "gray") for mt in mask_type]
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.bar(np.arange(len(feat_names)), mae, color=colors, edgecolor="black", lw=0.4)
        ax.set_xticks(np.arange(len(feat_names)))
        ax.set_xticklabels(feat_names, rotation=90, fontsize=7)
        ax.set_ylabel("Mean absolute reconstruction error")
        ax.set_title("Per-Feature VAE Reconstruction MAE (color = mask type)")
        handles = [Line2D([0], [0], color=PALETTE["Full"], lw=8, label="Full"),
                   Line2D([0], [0], color=PALETTE["Partial"], lw=8, label="Partial"),
                   Line2D([0], [0], color=PALETTE["Frozen"], lw=8, label="Frozen")]
        ax.legend(handles=handles)
        save_fig(fig, out, "vae_reconstruction_mae_by_feature.pdf", checklist)
    _try("D3", d3)


# ---------------------------------------------------------------------------
# SECTION E
# ---------------------------------------------------------------------------
def _build_multimetric_fallback(data) -> pd.DataFrame:
    Xo = data["X_original"]; rows = []
    for at in ATTACK_TYPES:
        Xa = data[f"X_adv_{at}"]
        ev = np.asarray(data[f"evasion_mask_{at}"]).astype(bool)
        pv = np.asarray(data[f"protocol_valid_{at}"]).astype(bool)
        l2 = np.linalg.norm(Xa - Xo, axis=1)
        rows.append({
            "model": "aggregate", "attack_type": at,
            "ASR": ev.mean(), "ASR_Valid": (ev & pv).mean(),
            "Protocol_Valid": pv.mean(), "Mask_Valid": np.nan,
            "IDSR": (~ev).mean(), "L2_mean": l2.mean(),
        })
    return pd.DataFrame(rows)


def section_e(data, out, checklist):
    print("\n[Section E] Multi-metric summary")
    if "multimetric_table" in data:
        mm = data["multimetric_table"]
        try:
            mm = _table_to_df(mm)
        except Exception:
            mm = _build_multimetric_fallback(data)
    else:
        mm = _build_multimetric_fallback(data)

    def e1():
        axes_labels = ["ASR", "ASR_Valid", "Protocol_Valid", "Mask_Valid", "IDSR", "1/L2_norm"]
        models = list(mm["model"].unique())
        n_models = len(models)
        ncols = min(n_models, 3); nrows = int(np.ceil(n_models / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows),
                                 subplot_kw=dict(polar=True), squeeze=False)
        l2_max = mm["L2_mean"].replace(0, np.nan).max()
        angles = np.linspace(0, 2 * np.pi, len(axes_labels), endpoint=False).tolist()
        angles += angles[:1]
        for ax, mdl in zip(axes.ravel(), models):
            sub = mm[mm["model"] == mdl]
            for _, r in sub.iterrows():
                inv_l2 = 0.0
                if pd.notna(r["L2_mean"]) and r["L2_mean"] > 0 and pd.notna(l2_max):
                    inv_l2 = 1.0 - min(r["L2_mean"] / l2_max, 1.0)
                vals = [r.get("ASR", np.nan), r.get("ASR_Valid", np.nan),
                        r.get("Protocol_Valid", np.nan), r.get("Mask_Valid", np.nan),
                        r.get("IDSR", np.nan), inv_l2]
                vals = [0.0 if pd.isna(v) else float(v) for v in vals]
                vals += vals[:1]
                at = r["attack_type"]
                ax.plot(angles, vals, color=PALETTE.get(at, "black"),
                        label=ATTACK_LABELS.get(at, at), lw=1.5)
                ax.fill(angles, vals, color=PALETTE.get(at, "black"), alpha=0.1)
            ax.set_xticks(angles[:-1]); ax.set_xticklabels(axes_labels, fontsize=8)
            ax.set_ylim(0, 1); ax.set_title(mdl, y=1.08)
            ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=7)
        for ax in axes.ravel()[n_models:]:
            ax.axis("off")
        save_fig(fig, out, "radar_multimodel.pdf", checklist)
    _try("E1", e1)

    def e2():
        if "category_asr_table" in data:
            ct = data["category_asr_table"]
            ct = _table_to_df(ct)
        else:
            y = _labels(data["y_true"])
            rows = []
            for at in ATTACK_TYPES:
                ev = np.asarray(data[f"evasion_mask_{at}"]).astype(bool)
                pv = np.asarray(data[f"protocol_valid_{at}"]).astype(bool)
                for c in np.unique(y):
                    m = y == c
                    if m.sum() == 0:
                        continue
                    rows.append({"model": "aggregate", "attack_type": at,
                                 "category": c, "ASR_Valid": (ev[m] & pv[m]).mean() * 100})
            ct = pd.DataFrame(rows)
        ct = ct.copy()
        ct["row"] = ct["model"].astype(str) + " | " + ct["attack_type"].astype(str)
        pivot = ct.pivot_table(index="row", columns="category", values="ASR_Valid", aggfunc="mean")
        fig, ax = plt.subplots(figsize=(max(8, 0.7 * pivot.shape[1] + 4),
                                        0.4 * pivot.shape[0] + 3))
        sns.heatmap(pivot, annot=True, fmt=".1f", cmap="RdBu_r", center=0,
                    cbar_kws={"label": "ASR_Valid (%)"}, ax=ax)
        ax.set_title("Category-Level ASR_Valid Heatmap")
        ax.set_xlabel("Category"); ax.set_ylabel("model | attack")
        save_fig(fig, out, "category_asr_heatmap.pdf", checklist)
    _try("E2", e2)


# ---------------------------------------------------------------------------
# SECTION F
# ---------------------------------------------------------------------------
def section_f(data, out, checklist, stats_rows):
    print("\n[Section F] Latent geometry")
    z = data["z_original"]; y = _labels(data["y_true"])

    def f1():
        pv = np.asarray(data["protocol_valid_latent_cw"]).astype(bool)
        z_pert = data["z_perturbed_cw"]
        rng = np.random.default_rng(SEED)
        categories = np.unique(y)
        rows = []
        for _ in range(5):
            cat = rng.choice(categories)
            idxs = np.flatnonzero(y == cat)
            if len(idxs) < 2:
                continue
            a, b = rng.choice(idxs, size=2, replace=False)
            za, zb = z[a], z[b]
            alphas = np.linspace(0, 1, 10)
            valids = []
            for alpha in alphas:
                z_interp = (1 - alpha) * za + alpha * zb
                d = np.linalg.norm(z_pert - z_interp, axis=1)
                nn = int(np.argmin(d))
                valids.append(bool(pv[nn]))
            rows.append((f"{cat}: {a}<->{b}", valids))
        if not rows:
            print("  [SKIP F1] not enough samples")
            return
        labels = [r[0] for r in rows]
        mat = np.array([r[1] for r in rows], dtype=int)
        fig, ax = plt.subplots(figsize=(8, 1 + 0.5 * len(rows)))
        sns.heatmap(mat, ax=ax, cmap=["#d62728", "#2ca02c"], cbar=False,
                    linewidths=0.5, linecolor="white", yticklabels=labels,
                    xticklabels=[f"{a:.1f}" for a in np.linspace(0, 1, 10)],
                    annot=mat, fmt="d", annot_kws={"size": 8, "color": "white"})
        ax.set_xlabel("Interpolation alpha (a -> b)")
        ax.set_title("Latent Interpolation: Protocol Validity per Step\n(green = valid, red = invalid)")
        save_fig(fig, out, "latent_interpolation_validity.pdf", checklist)
    _try("F1", f1)

    def f2():
        rng = np.random.default_rng(SEED)
        ev = np.asarray(data["evasion_mask_latent_cw"]).astype(bool)
        pv = np.asarray(data["protocol_valid_latent_cw"]).astype(bool)
        valid_attack = ev & pv
        cats, spreads, asrs = [], [], []
        for c in np.unique(y):
            m = np.flatnonzero(y == c)
            if len(m) < 2:
                continue
            sample = m if len(m) <= 500 else rng.choice(m, size=500, replace=False)
            zs = z[sample]
            d = np.linalg.norm(zs[:, None, :] - zs[None, :, :], axis=-1)
            iu = np.triu_indices_from(d, k=1)
            spreads.append(float(d[iu].mean()))
            asrs.append(float(valid_attack[m].mean() * 100))
            cats.append(str(c))
        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(cats, spreads, color=sns.color_palette(CATEGORY_PALETTE, len(cats)),
                       edgecolor="black", lw=0.5)
        ax.set_ylabel("Mean intra-cluster latent L2 distance")
        ax.set_xlabel("Attack category")
        ax.set_title("Latent Cluster Spread per Category (annotation = ASR_Valid %)")
        ax.tick_params(axis="x", rotation=30)
        for b, asr in zip(bars, asrs):
            ax.text(b.get_x() + b.get_width()/2, b.get_height(),
                    f"{asr:.1f}%", ha="center", va="bottom", fontsize=8)
        save_fig(fig, out, "latent_cluster_spread_vs_asr.pdf", checklist)
        if len(spreads) >= 3:
            rho, p = stats.spearmanr(spreads, asrs)
            print(f"  Spearman(cluster_spread, ASR_Valid) = rho={rho:.3f}, p={p:.3g}")
            stats_rows.append({"test": "Spearman_clusterSpread_vs_ASRValid",
                               "feature": "n/a", "statistic": rho, "p_value": p})
    _try("F2", f2)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--artifacts", type=Path, required=True,
                   help="Path to .npz bundle or directory of .npy/.npz files")
    p.add_argument("--out", type=Path, default=Path("thesis_figures"))
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    data = load_artifacts(args.artifacts)

    checklist: Dict[str, str] = {}
    stats_rows: list = []

    section_a(data, args.out, checklist)
    section_b(data, args.out, checklist, stats_rows)
    section_c(data, args.out, checklist)
    section_d(data, args.out, checklist)
    section_e(data, args.out, checklist)
    section_f(data, args.out, checklist, stats_rows)

    if stats_rows:
        pd.DataFrame(stats_rows).to_csv(args.out / "statistical_summary.csv", index=False)
        checklist["statistical_summary.csv"] = "OK"

    print("\n" + "=" * 60)
    print("FIGURE GENERATION CHECKLIST")
    print("=" * 60)
    expected = [
        "tsne_latent_by_category.pdf", "tsne_latent_original_vs_perturbed.pdf",
        "tsne_input_space_comparison.pdf",
        "umap_latent_by_category.pdf", "umap_latent_original_vs_perturbed.pdf",
        "umap_input_space_comparison.pdf",
        "pca_latent_by_category.pdf", "pca_latent_by_evasion.pdf",
        "umap_latent_by_evasion.pdf",
        "kde_top10_features.pdf",
        "correlation_heatmap_comparison.pdf", "correlation_difference.pdf",
        "ks_wasserstein_results.csv", "ks_statistic_by_feature.pdf",
        "perturbation_heatmap.pdf",
        "l2_distortion_cdf.pdf", "l2_by_category_boxplot.pdf",
        "vae_reconstruction_scatter.pdf",
        "vae_reconstruction_error_hist.pdf",
        "vae_reconstruction_mae_by_feature.pdf",
        "radar_multimodel.pdf", "category_asr_heatmap.pdf",
        "latent_interpolation_validity.pdf",
        "latent_cluster_spread_vs_asr.pdf",
        "statistical_summary.csv",
    ]
    for name in expected:
        status = checklist.get(name, "MISSING")
        mark = "[OK]  " if status == "OK" else "[--]  "
        suffix = "" if status == "OK" else f" ({status})"
        print(f"  {mark}{name}{suffix}")
    print("=" * 60)


if __name__ == "__main__":
    main()
