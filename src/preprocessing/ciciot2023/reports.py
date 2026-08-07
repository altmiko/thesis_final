"""Reports / evidence for the downsampling pipeline — one entry point per spec step.

Consolidates what used to be four scripts into `--mode` subcommands:

    python -m src.preprocessing.ciciot2023.reports diagnostics    # §6.2 multi-modality + §6.3 crosstab
    python -m src.preprocessing.ciciot2023.reports verify         # §6.5 leakage guards
    python -m src.preprocessing.ciciot2023.reports evidence       # §7.4 fidelity / support figures
    python -m src.preprocessing.ciciot2023.reports sensitivity    # §7.5 temporal-leakage bound [--device --epochs]

Pure logic + IO helpers live in `pipeline.py` (imported as `pl`); split primitives
in `splitter.py`; the sampler in `sampler.py`.
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.cluster import MiniBatchKMeans  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.preprocessing import RobustScaler  # noqa: E402

from config import paths  # noqa: E402
# pipeline imports pandas/pyarrow; it MUST load before torch (Windows DLL order).
from src.preprocessing.ciciot2023 import pipeline as pl  # noqa: E402
from src.preprocessing.ciciot2023.splitter import assert_forward_chaining, natkey  # noqa: E402

import torch  # noqa: E402  (after pl/pyarrow)
from src.classifiers.models import get_model  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("reports").info

# Sub-attack counts per majority category (spec §6.4) for the diagnostic crosstab.
MAJORITY_CATEGORIES: dict[str, int] = {
    "DDoS": 12, "DoS": 4, "Mirai": 3, "Recon": 5, "Spoofing": 2, "Benign": 1,
}
K_GRID = (10, 15, 20)
DIAG_SAMPLE = 50_000
SCALER_FIT_SAMPLE = 2_000_000
MODELS = ("mlp", "cnn", "lstm", "dualpath")  # gradient-attackable classifiers (§7.5)
TRAIN_CAP_PER_CLASS = 20_000
TEST_EVAL_CAP = 300_000
BATCH = 4096


# ── §6.2/§6.3 diagnostics ─────────────────────────────────────────────────


def _fit_diag_scaler(x_cont_all: np.ndarray, train_mask: np.ndarray, seed: int) -> RobustScaler:
    """RobustScaler on a large train-only sample of continuous features. Per-column
    (median/IQR) so it matches the pipeline's full-feature scaler on these cols."""
    train_idx = np.flatnonzero(train_mask)
    rng = np.random.default_rng(seed)
    if train_idx.size > SCALER_FIT_SAMPLE:
        train_idx = rng.choice(train_idx, SCALER_FIT_SAMPLE, replace=False)
    sc = RobustScaler()
    sc.fit(x_cont_all[train_idx])
    return sc


def _cluster_size_stats(sizes: np.ndarray) -> dict:
    sizes = np.sort(sizes)[::-1]
    total = int(sizes.sum())
    frac = sizes / total
    # Gini of cluster-size distribution: 0 = perfectly uniform, →1 = one giant mode.
    n = len(sizes)
    idx = np.arange(1, n + 1)
    gini = float((2 * np.sum(idx * np.sort(sizes)) / (n * sizes.sum())) - (n + 1) / n)
    return {
        "n_clusters": n,
        "sizes_desc": sizes.astype(int).tolist(),
        "max_over_min": float(sizes.max() / max(1, sizes.min())),
        "cv": float(sizes.std() / sizes.mean()),
        "gini": gini,
        "top_cluster_frac": float(frac[0]),
    }


def run_diagnostics() -> dict:
    paths.ensure_dirs()
    log("Loading metadata + computing split …")
    meta = pl.load_metadata()
    split, plans, _ = pl.compute_split(meta)
    train_mask = split == 0
    category = meta["category"].astype(str).to_numpy()
    label = meta["Label"].astype(str).to_numpy()

    log("Loading continuous features (%d cols) …", len(pl.CONTINUOUS_FEATURES))
    x_cont = pl.load_features(columns=pl.CONTINUOUS_FEATURES)

    log("Fitting diagnostic RobustScaler on train sample …")
    scaler = _fit_diag_scaler(x_cont, train_mask, paths.SEED)

    report: dict = {"k_grid": list(K_GRID), "diag_sample": DIAG_SAMPLE, "categories": {}}

    for cat, n_sub in MAJORITY_CATEGORIES.items():
        cat_train_idx = np.flatnonzero(train_mask & (category == cat))
        n_cat = cat_train_idx.size
        rng = np.random.default_rng(paths.SEED)
        sample_idx = (
            rng.choice(cat_train_idx, DIAG_SAMPLE, replace=False)
            if n_cat > DIAG_SAMPLE else cat_train_idx
        )
        Xs = scaler.transform(x_cont[sample_idx])
        subtypes = label[sample_idx]

        cat_entry: dict = {
            "n_train_rows": int(n_cat),
            "n_subattacks": n_sub,
            "diag_n": int(sample_idx.size),
            "per_k": {},
        }
        # cluster-size distribution figure across the k grid; choose the k whose
        # clusters best recover the known sub-attack labels (spec §6.3 purity).
        fig, axes = plt.subplots(1, len(K_GRID), figsize=(4 * len(K_GRID), 3.2), squeeze=False)
        best_k = best_stats = best_labels = best_ct = None
        best_purity = -1.0
        for j, k in enumerate(K_GRID):
            km = MiniBatchKMeans(n_clusters=min(k, sample_idx.size), random_state=paths.SEED,
                                 batch_size=4096, n_init="auto")
            labels_k = km.fit_predict(Xs)
            _, counts = np.unique(labels_k, return_counts=True)
            stats = _cluster_size_stats(counts)
            ct_k = pd.crosstab(labels_k, subtypes, normalize="index")
            purity_k = float(ct_k.max(axis=1).mean()) if ct_k.size else 0.0
            stats["cluster_purity_vs_subtype"] = purity_k
            cat_entry["per_k"][str(k)] = stats
            ax = axes[0][j]
            ax.bar(range(len(counts)), np.sort(counts)[::-1], color="#3b6ea5")
            ax.set_title(f"{cat}  k={k}\nGini={stats['gini']:.2f} purity={purity_k:.2f}")
            ax.set_xlabel("cluster (size-sorted)")
            ax.set_ylabel("rows")
            # Prefer higher purity; tie-break toward fewer clusters (simpler).
            if purity_k > best_purity + 1e-9:
                best_purity, best_k, best_stats, best_labels, best_ct = (
                    purity_k, k, stats, labels_k, ct_k
                )
        fig.tight_layout()
        fig.savefig(paths.DIAGNOSTICS_DIR / f"clustersize_{cat}.png", dpi=110)
        plt.close(fig)

        # Verdict (spec §6.2): skewed sizes → multimodal → cluster; else unimodal.
        gini = best_stats["gini"]
        verdict = "multimodal" if gini >= 0.30 else "unimodal"
        cat_entry["verdict"] = verdict
        cat_entry["recommended_k"] = int(best_k)
        cat_entry["recommended_sampler"] = (
            "cluster_proportional_floor" if verdict == "multimodal" else "random_floor"
        )

        # §6.3 crosstab at the chosen k: does clustering recover sub-attacks?
        ct = best_ct
        ct.to_csv(paths.DIAGNOSTICS_DIR / f"crosstab_{cat}_k{best_k}.csv")
        purity = best_purity
        cat_entry["cluster_purity_vs_subtype"] = purity

        fig, ax = plt.subplots(figsize=(max(4, 0.5 * ct.shape[1] + 2), 0.35 * ct.shape[0] + 2))
        im = ax.imshow(ct.values, aspect="auto", cmap="viridis", vmin=0, vmax=1)
        ax.set_xticks(range(ct.shape[1]))
        ax.set_xticklabels(ct.columns, rotation=90, fontsize=6)
        ax.set_yticks(range(ct.shape[0]))
        ax.set_yticklabels(ct.index, fontsize=7)
        ax.set_title(f"{cat}: cluster × subtype (k={best_k}), purity={purity:.2f}")
        fig.colorbar(im, ax=ax, fraction=0.03)
        fig.tight_layout()
        fig.savefig(paths.DIAGNOSTICS_DIR / f"crosstab_{cat}.png", dpi=110)
        plt.close(fig)

        report["categories"][cat] = cat_entry
        log(
            "  %-9s n_train=%9d  k*=%2d  Gini=%.2f  purity=%.2f  -> %s",
            cat, n_cat, best_k, gini, purity, verdict,
        )

    out = paths.DIAGNOSTICS_DIR / "multimodality.json"
    out.write_text(json.dumps(report, indent=2))
    log("Wrote %s", out)
    return report


# ── §6.5 leakage guards ───────────────────────────────────────────────────
def run_verify() -> None:
    meta = pl.load_metadata()
    split, plans, runs = pl.compute_split(meta, ds_val := 0.10, ds_test := 0.20)
    labels = meta["Label"].astype(str).to_numpy()
    checks: list[str] = []

    # 1. shard disjointness across splits (forward_chain classes)
    for plan in plans.values():
        assert_forward_chaining(plan)
        if plan.protocol == "forward_chain":
            sets = [set(plan.train_shards), set(plan.val_shards), set(plan.test_shards)]
            for a in range(3):
                for b in range(a + 1, 3):
                    assert not (sets[a] & sets[b]), f"{plan.label}: shard in two splits"
    checks.append("no shard appears in two splits (forward_chain)")

    # 2. every class present in all 3 splits
    for lbl in plans:
        present = set(np.unique(split[labels == lbl]).tolist())
        assert present == {0, 1, 2}, f"{lbl}: splits present = {present}"
    checks.append("every class present in train/val/test")

    # 3. forward-chaining on row indices within each class (no test row precedes train)
    for lbl in plans:
        rows = np.flatnonzero(labels == lbl)
        s = split[rows]
        tr, va, te = rows[s == 0], rows[s == 1], rows[s == 2]
        # global row order == time order (parquet is time-ordered); the max train
        # row index must precede the min test row index within the class.
        if plans[lbl].protocol in ("block", "two_shard_hybrid"):
            assert tr.max() < va.min() <= te.min() or tr.max() < te.min(), (
                f"{lbl}: block ordering violated"
            )
    checks.append("no test row precedes a train row within a class")

    # 4. val/test untouched by the sampler — counts equal the plan totals
    man = json.loads(paths.RUN_MANIFEST.read_text())
    plan_val = sum(p.n_val for p in plans.values())
    plan_test = sum(p.n_test for p in plans.values())
    assert man["split_row_counts"]["val"] == plan_val, (man["split_row_counts"]["val"], plan_val)
    assert man["split_row_counts"]["test"] == plan_test, (man["split_row_counts"]["test"], plan_test)
    checks.append(f"val/test untouched by sampler (val={plan_val:,} test={plan_test:,})")

    # 5. train post-sampling total matches manifest and array on disk
    y_train = np.load(paths.processed("y_train.npy"), mmap_mode="r")
    assert y_train.shape[0] == man["train_after_sampling"], (
        y_train.shape[0], man["train_after_sampling"]
    )
    checks.append(f"train sampled total consistent ({man['train_after_sampling']:,})")

    print("LEAKAGE GUARDS PASSED (spec §6.5):")
    for c in checks:
        print("  [OK]", c)
    print(f"\nsplit protocol counts: {man['split_protocol_counts']}")
    print(f"train {man['train_before_sampling']:,} -> {man['train_after_sampling']:,}  "
          f"val {plan_val:,}  test {plan_test:,}")


# ── §7.4 evidence figures ─────────────────────────────────────────────────
def per_split_class_counts() -> None:
    man = json.loads(paths.RUN_MANIFEST.read_text())
    rows = []
    for lbl, p in sorted(man["per_class_split"].items()):
        rows.append({
            "label": lbl, "protocol": p["protocol"], "n_shards": p["n_shards"],
            "train_natural": p["n_train"], "val": p["n_val"], "test": p["n_test"],
        })
    df = pd.DataFrame(rows)
    # attach post-sampling train counts by category (train arrays are sampled)
    df.to_csv(paths.DIAGNOSTICS_DIR / "per_split_class_counts.csv", index=False)

    fig, ax = plt.subplots(figsize=(11, 0.32 * len(df) + 1.5))
    ax.axis("off")
    tbl = ax.table(cellText=df.values, colLabels=df.columns, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(6.5); tbl.scale(1, 1.15)
    ax.set_title("Per-class split (natural counts) — spec §7.4", fontsize=10)
    fig.tight_layout()
    fig.savefig(paths.DIAGNOSTICS_DIR / "per_split_class_counts.png", dpi=130)
    plt.close(fig)
    log("wrote per_split_class_counts.{csv,png}  (%d classes)", len(df))


def fidelity_figures() -> None:
    """Faithfully reproduce the pipeline's cleaned+scaled space and its persisted
    kept indices, then show full-class vs kept overlap per majority category."""
    log("loading metadata + split + full features …")
    man = json.loads(paths.RUN_MANIFEST.read_text())
    upper = np.array([man["clip_upper_train"][f] for f in pl.FEATURE_NAMES], dtype=np.float32)

    meta = pl.load_metadata()
    split, _, _ = pl.compute_split(meta, pl.VAL_FRAC, pl.TEST_FRAC)
    category = meta["category"].astype(str).to_numpy()
    train_mask = split == 0

    X = pl.load_features()                      # full 39 features
    pl.clip_round(X, upper)                      # identical cleaning to the pipeline
    scaler = pickle.load(open(paths.processed("scaler.pkl"), "rb"))
    center = scaler.center_[pl.CONTINUOUS_IDX]
    scale = scaler.scale_[pl.CONTINUOUS_IDX]
    feats = pl.CONTINUOUS_FEATURES

    kept_global = np.load(paths.processed("train_kept_indices.npy"))
    kept_set = np.zeros(X.shape[0], dtype=bool)
    kept_set[kept_global] = True

    fidelity_summary: dict = {}
    for cat in pl.CAP_PER_CATEGORY:              # 6 majority (capped) categories
        cat_rows = np.flatnonzero(train_mask & (category == cat))
        Xc = (X[cat_rows][:, pl.CONTINUOUS_IDX] - center) / scale
        kept_local = np.flatnonzero(kept_set[cat_rows])   # kept rows within this class
        rng = np.random.default_rng(paths.SEED)
        n_show = min(20000, cat_rows.size)
        full_s = rng.choice(Xc.shape[0], n_show, replace=False)
        kept_s = kept_local if kept_local.size <= n_show else rng.choice(kept_local, n_show, replace=False)

        # Cluster with the SAME clean+scaled input the pipeline used (seed 42) so
        # the clusters here are the ones the sampler allocated over.
        km = MiniBatchKMeans(n_clusters=pl.K_PER_CATEGORY[cat], random_state=paths.SEED,
                             batch_size=8192, n_init="auto")
        clabels = km.fit_predict(Xc)
        kept_mask = np.zeros(Xc.shape[0], dtype=bool); kept_mask[kept_local] = True
        iqr_full = np.subtract(*np.percentile(Xc, [75, 25], axis=0))

        # WITHIN-CLUSTER fidelity — the honest claim (spec §7.4 / §5.2): the pick
        # inside a cluster is uniform random, so per-cluster pre vs post medians
        # match. Normalise each shift by *that cluster's own* IQR (not the global
        # IQR, which RobustScaler floors to 1 for near-constant features and would
        # inflate the ratio for a feature whose scaled magnitude is large).
        # (The WHOLE-class marginal is *intentionally* reshaped: the floor
        # up-weights rare modes — that is mode preservation, not a fidelity loss.)
        per_cluster_err = []
        for c in np.unique(clabels):
            m = clabels == c
            post_m = m & kept_mask
            if post_m.sum() < 5:
                continue
            iqr_c = np.subtract(*np.percentile(Xc[m], [75, 25], axis=0))
            iqr_c = np.where(iqr_c > 1e-9, iqr_c, np.nan)  # constant-in-cluster -> skip
            dmed_c = np.abs(np.median(Xc[m], axis=0) - np.median(Xc[post_m], axis=0)) / iqr_c
            per_cluster_err.append(dmed_c)
        stacked = np.nan_to_num(np.array(per_cluster_err), nan=0.0)  # constant-in-cluster -> 0 error
        wc_err = stacked.max(axis=0) if stacked.size else np.zeros(len(feats))
        fidelity_summary[cat] = {
            "within_cluster_max_norm_median_shift": float(wc_err.max()),
            "within_cluster_mean_norm_median_shift": float(wc_err.mean()),
            "kept": int(kept_local.size),
        }

        # ── Fig A: manifold-support overlap (PCA-2D) + within-cluster fidelity ─
        pca = PCA(n_components=2, random_state=paths.SEED).fit(Xc[full_s])
        Pf, Pk = pca.transform(Xc[full_s]), pca.transform(Xc[kept_s])
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
        ax = axes[0]
        ax.scatter(Pf[:, 0], Pf[:, 1], s=4, alpha=0.25, label=f"full train (n={cat_rows.size:,})", color="#888")
        ax.scatter(Pk[:, 0], Pk[:, 1], s=4, alpha=0.35, label=f"kept (n={kept_local.size:,})", color="#c0392b")
        ax.set_title(f"{cat}: manifold-support overlap (full vs kept)")
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.legend(markerscale=3, fontsize=8)

        ax = axes[1]
        yb = np.arange(len(feats))
        ax.barh(yb, wc_err, color="#2e7d32")
        ax.set_yticks(yb); ax.set_yticklabels(feats, fontsize=6)
        ax.set_xlabel("max within-cluster |Δ median| / IQR  (≈0 ⇒ shape preserved)")
        ax.set_title(f"{cat}: within-cluster fidelity\nmax={wc_err.max():.3f}  mean={wc_err.mean():.3f}")
        fig.tight_layout()
        fig.savefig(paths.DIAGNOSTICS_DIR / f"prepost_overlap_{cat}.png", dpi=120)
        plt.close(fig)

        # ── Fig B: within-cluster pre/post histograms (largest 3 clusters) ────
        fidx = int(np.argmax(iqr_full))          # highest-IQR feature = most to see
        sizes = np.bincount(clabels)
        top = np.argsort(sizes)[::-1][:3]
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), squeeze=False)
        for j, c in enumerate(top):
            m = clabels == c
            ax = axes[0][j]
            pre, post = Xc[m, fidx], Xc[m & kept_mask, fidx]
            lo, hi = np.percentile(pre, [1, 99])
            bins = np.linspace(lo, hi, 40) if hi > lo else 40
            ax.hist(pre, bins=bins, density=True, alpha=0.5, label=f"pre (n={m.sum():,})", color="#888")
            ax.hist(post, bins=bins, density=True, alpha=0.5, label=f"post (n={post.size:,})", color="#c0392b")
            ax.set_title(f"{cat} cl{c}: {feats[fidx]}", fontsize=9)
            ax.legend(fontsize=7)
        fig.suptitle(f"{cat}: within-cluster pre vs post ({feats[fidx]}) — shape preserved", fontsize=10)
        fig.tight_layout()
        fig.savefig(paths.DIAGNOSTICS_DIR / f"prepost_within_cluster_{cat}.png", dpi=120)
        plt.close(fig)
        log("  %-9s kept=%d  within-cluster fidelity max=%.3f mean=%.3f (IQR-normalised)",
            cat, kept_local.size, float(wc_err.max()), float(wc_err.mean()))

    (paths.DIAGNOSTICS_DIR / "fidelity_summary.json").write_text(json.dumps(fidelity_summary, indent=2))


def run_evidence() -> None:
    paths.ensure_dirs()
    per_split_class_counts()
    fidelity_figures()
    log("Evidence figures written to %s", paths.DIAGNOSTICS_DIR)


# ── §7.5 sensitivity study ────────────────────────────────────────────────


def _shuffled_split(labels: np.ndarray, rows: np.ndarray, seed: int) -> np.ndarray:
    """Row-level shuffled split over ``rows`` at the same 70/10/20 per-class
    proportions the temporal split uses — the leakage-blind counterfactual."""
    split = np.full(labels.shape[0], -1, dtype=np.int8)
    rng = np.random.default_rng(seed)
    for lbl in np.unique(labels[rows]):
        idx = rows[labels[rows] == lbl]
        idx = rng.permutation(idx)
        n = idx.size
        n_test = max(1, round(n * pl.TEST_FRAC))
        n_val = max(1, round(n * pl.VAL_FRAC))
        split[idx[: n - n_val - n_test]] = 0
        split[idx[n - n_val - n_test : n - n_test]] = 1
        split[idx[n - n_test :]] = 2
    return split


def _cap_train(split: np.ndarray, y: np.ndarray, seed: int) -> np.ndarray:
    """Random per-class train cap (identical procedure for both variants — this
    study isolates the *split*, not the sampler)."""
    rng = np.random.default_rng(seed)
    keep = []
    tr = np.flatnonzero((split == 0) & (y >= 0))  # y<0 = non-fc class -> excluded
    for c in np.unique(y[tr]):
        idx = tr[y[tr] == c]
        keep.append(idx if idx.size <= TRAIN_CAP_PER_CLASS
                    else rng.choice(idx, TRAIN_CAP_PER_CLASS, replace=False))
    out = np.concatenate(keep)
    out.sort()
    return out


def _train_eval(model_type, Xtr, ytr, Xte, yte, n_classes, device, epochs, seed) -> float:
    torch.manual_seed(seed)
    model = get_model(model_type, num_features=Xtr.shape[1], num_classes=n_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = torch.nn.CrossEntropyLoss()
    Xtr_t = torch.from_numpy(Xtr)
    ytr_t = torch.from_numpy(ytr.astype(np.int64))
    n = Xtr.shape[0]
    model.train()
    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, BATCH):
            b = perm[i : i + BATCH]
            xb = Xtr_t[b].to(device)
            yb = ytr_t[b].to(device)
            opt.zero_grad()
            out = model(xb)
            loss = lossf(out, yb)
            loss.backward()
            opt.step()
    # eval
    model.eval()
    correct = 0
    with torch.no_grad():
        for i in range(0, Xte.shape[0], BATCH):
            xb = torch.from_numpy(Xte[i : i + BATCH]).to(device)
            pred = model(xb).argmax(1).cpu().numpy()
            correct += int((pred == yte[i : i + BATCH]).sum())
    return correct / Xte.shape[0]


def run_sensitivity(device: str = "cuda", epochs: int = 4) -> dict:
    device = device if (device == "cuda" and torch.cuda.is_available()) else "cpu"
    log("device=%s epochs=%d", device, epochs)
    t0 = time.time()

    meta = pl.load_metadata()
    split_fc, plans, _ = pl.compute_split(meta, pl.VAL_FRAC, pl.TEST_FRAC)
    labels = meta["Label"].astype(str).to_numpy()

    fc_classes = sorted(l for l, p in plans.items() if p.protocol == "forward_chain")
    log("%d forward-chain (shard-splittable) classes", len(fc_classes))
    rows = np.flatnonzero(np.isin(labels, fc_classes))

    # contiguous 0..K-1 label ids over the 17 classes
    lut = {l: i for i, l in enumerate(fc_classes)}
    y = np.array([lut.get(l, -1) for l in labels], dtype=np.int64)

    log("loading + cleaning features …")
    X = pl.load_features()
    # clean with train-only clip from the FORWARD-CHAIN train (consistent bound);
    # both variants use the same cleaned X (cleaning is not the variable here).
    q = np.percentile(X[split_fc == 0], pl.CLIP_PERCENTILE, axis=0).astype(np.float32)
    pl.clip_round(X, q)

    variants = {
        "forward_chain": split_fc,
        "shuffled": _shuffled_split(labels, rows, paths.SEED),
    }
    results: dict = {"n_classes": len(fc_classes), "classes": fc_classes, "per_model": {}}
    acc: dict = {}
    rng = np.random.default_rng(paths.SEED)
    for vname, split in variants.items():
        tr_idx = _cap_train(split, y, paths.SEED)
        te_idx = np.flatnonzero((split == 2) & (y >= 0))  # fc classes only
        if te_idx.size > TEST_EVAL_CAP:
            te_idx = np.sort(rng.choice(te_idx, TEST_EVAL_CAP, replace=False))
        sc = RobustScaler().fit(X[tr_idx])
        Xtr = sc.transform(X[tr_idx]).astype(np.float32)
        Xte = sc.transform(X[te_idx]).astype(np.float32)
        ytr, yte = y[tr_idx], y[te_idx]
        log("[%s] train=%d test=%d", vname, Xtr.shape[0], Xte.shape[0])
        acc[vname] = {}
        for m in MODELS:
            a = _train_eval(m, Xtr, ytr, Xte, yte, len(fc_classes), device, epochs, paths.SEED)
            acc[vname][m] = a
            log("    %-9s acc=%.4f", m, a)

    for m in MODELS:
        delta = acc["shuffled"][m] - acc["forward_chain"][m]
        results["per_model"][m] = {
            "acc_shuffled": acc["shuffled"][m],
            "acc_forward_chain": acc["forward_chain"][m],
            "leakage_inflation": delta,
        }
    deltas = [results["per_model"][m]["leakage_inflation"] for m in MODELS]
    results["mean_leakage_inflation"] = float(np.mean(deltas))
    results["epochs"] = epochs
    results["train_cap_per_class"] = TRAIN_CAP_PER_CLASS
    results["runtime_seconds"] = round(time.time() - t0, 1)

    out = paths.DIAGNOSTICS_DIR / "sensitivity_study.json"
    paths.ensure_dirs()
    out.write_text(json.dumps(results, indent=2))
    log("=== mean temporal-leakage inflation: %+.4f accuracy (shuffled - forward_chain) ===",
        results["mean_leakage_inflation"])
    log("wrote %s", out)
    return results

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=["diagnostics", "verify", "evidence", "sensitivity"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--epochs", type=int, default=4)
    args = ap.parse_args()
    if args.mode == "diagnostics":
        run_diagnostics()
    elif args.mode == "verify":
        run_verify()
    elif args.mode == "evidence":
        run_evidence()
    elif args.mode == "sensitivity":
        run_sensitivity(device=args.device, epochs=args.epochs)


if __name__ == "__main__":
    main()
