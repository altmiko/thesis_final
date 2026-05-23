"""
Plot perturbation statistics for 8-class restart attack analysis.

Expected inputs:
- results/attacks/attack_restart_summary_8class.csv
- results/attacks/feature_perturbation_8class.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results" / "attacks"
FIG_DIR = ROOT / "figures" / "appendix"

MODEL_ORDER = ["MLP", "LSTM", "CNN-LSTM"]
ATTACK_ORDER = [
    ("FGSM", "0.05"),
    ("FGSM", "0.10"),
    ("FGSM", "0.30"),
    ("PGD", "0.05"),
    ("PGD", "0.10"),
    ("PGD", "0.30"),
    ("CW", "0"),
]


def _sort_summary(df: pd.DataFrame) -> pd.DataFrame:
    order_map = {k: i for i, k in enumerate(ATTACK_ORDER)}
    df = df.copy()
    df["_attack_key"] = list(zip(df["attack"], df["eps"]))
    df["_attack_ord"] = df["_attack_key"].map(order_map)
    model_map = {k: i for i, k in enumerate(MODEL_ORDER)}
    df["_model_ord"] = df["model"].map(model_map).fillna(999)
    df.sort_values(["_attack_ord", "_model_ord"], inplace=True)
    return df


def _plot_norm_bars(summary_df: pd.DataFrame) -> None:
    labels = [f"{a} {e}" if a != "CW" else "CW" for a, e in ATTACK_ORDER]
    x = np.arange(len(labels), dtype=np.float64)
    width = 0.24

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    colors = {"MLP": "#1b9e77", "LSTM": "#d95f02", "CNN-LSTM": "#7570b3"}

    for i, model in enumerate(MODEL_ORDER):
        subset = summary_df[summary_df["model"] == model]
        means_l2: List[float] = []
        stds_l2: List[float] = []
        means_linf: List[float] = []
        stds_linf: List[float] = []

        for attack, eps in ATTACK_ORDER:
            row = subset[(subset["attack"] == attack) & (subset["eps"] == eps)]
            if row.empty:
                means_l2.append(np.nan)
                stds_l2.append(0.0)
                means_linf.append(np.nan)
                stds_linf.append(0.0)
            else:
                means_l2.append(float(row["mean_l2_raw"].iloc[0]))
                stds_l2.append(float(row["std_l2_raw"].iloc[0]))
                means_linf.append(float(row["mean_linf_raw"].iloc[0]))
                stds_linf.append(float(row["std_linf_raw"].iloc[0]))

        offset = (i - 1) * width
        axes[0].bar(x + offset, means_l2, width=width, yerr=stds_l2, capsize=3, label=model, color=colors.get(model))
        axes[1].bar(x + offset, means_linf, width=width, yerr=stds_linf, capsize=3, label=model, color=colors.get(model))

    axes[0].set_title("Mean ± Std L2 Perturbation (Raw Space)")
    axes[1].set_title("Mean ± Std Linf Perturbation (Raw Space)")
    axes[0].set_ylabel("L2")
    axes[1].set_ylabel("Linf")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=30, ha="right")
    axes[0].legend(loc="upper left", ncol=3)
    axes[0].grid(alpha=0.25, axis="y")
    axes[1].grid(alpha=0.25, axis="y")

    fig.tight_layout()
    out_path = FIG_DIR / "perturbation_norms_8class_models.pdf"
    fig.savefig(out_path)
    plt.close(fig)


def _choose_representative_eps(feature_df: pd.DataFrame, attack: str) -> str:
    sub = feature_df[feature_df["attack"] == attack]
    if sub.empty:
        return ""
    if attack in {"FGSM", "PGD"}:
        eps_numeric = pd.to_numeric(sub["eps"], errors="coerce")
        max_eps = eps_numeric.max()
        return f"{max_eps:.2f}"
    return "0"


def _plot_top10_features(feature_df: pd.DataFrame) -> None:
    for model in MODEL_ORDER:
        for attack in ["FGSM", "PGD", "CW"]:
            eps = _choose_representative_eps(feature_df[(feature_df["model"] == model)], attack)
            if eps == "":
                continue

            sub = feature_df[
                (feature_df["model"] == model)
                & (feature_df["attack"] == attack)
                & (feature_df["eps"] == eps)
            ].copy()
            if sub.empty:
                continue

            top = sub.sort_values("mean_abs_delta_raw", ascending=False).head(10)
            top = top.iloc[::-1]

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(top["feature_name"], top["mean_abs_delta_raw"], color="#2c7fb8")
            ax.set_xlabel("Mean |delta| (raw space)")
            ax.set_title(f"Top 10 Perturbed Features: {model} {attack} eps={eps} (8-class)")
            ax.grid(alpha=0.2, axis="x")

            for i, v in enumerate(top["mean_abs_delta_raw"].tolist()):
                ax.text(v, i, f" {v:.4f}", va="center", fontsize=8)

            fig.tight_layout()
            model_slug = model.lower().replace("-", "_")
            out_path = FIG_DIR / f"top10_features_{model_slug}_{attack.lower()}_8class.pdf"
            fig.savefig(out_path)
            plt.close(fig)


def _plot_heatmap(feature_df: pd.DataFrame) -> None:
    rows = []
    for model in MODEL_ORDER:
        for attack in ["FGSM", "PGD", "CW"]:
            eps = _choose_representative_eps(feature_df[(feature_df["model"] == model)], attack)
            if eps == "":
                continue
            label = f"{model}\n{attack} eps={eps}"
            sub = feature_df[
                (feature_df["model"] == model)
                & (feature_df["attack"] == attack)
                & (feature_df["eps"] == eps)
            ][["feature_name", "mean_abs_delta_raw"]]
            if sub.empty:
                continue
            sub = sub.set_index("feature_name").rename(columns={"mean_abs_delta_raw": label})
            rows.append(sub)

    if not rows:
        return

    matrix = pd.concat(rows, axis=1).fillna(0.0)
    matrix = matrix.loc[sorted(matrix.index)]

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(matrix.values, aspect="auto", cmap="YlOrRd")
    ax.set_yticks(np.arange(len(matrix.index)))
    ax.set_yticklabels(matrix.index)
    ax.set_xticks(np.arange(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=40, ha="right")
    ax.set_title("Feature Perturbation Heatmap (Mean |delta|, raw space, representative eps)")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Mean |delta|")

    fig.tight_layout()
    out_path = FIG_DIR / "feature_heatmap_8class_models.pdf"
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot restart attack perturbation statistics for 8-class.")
    parser.parse_args()

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    summary_path = RESULTS_DIR / "attack_restart_summary_8class.csv"
    feature_path = RESULTS_DIR / "feature_perturbation_8class.csv"

    if not summary_path.exists() or not feature_path.exists():
        raise FileNotFoundError(
            "Missing analysis CSV files. Run analyze_attack_restarts.py before plotting."
        )

    summary_df = pd.read_csv(summary_path, dtype={"eps": str})
    feature_df = pd.read_csv(feature_path, dtype={"eps": str})

    summary_df = _sort_summary(summary_df)

    _plot_norm_bars(summary_df)
    _plot_top10_features(feature_df)
    _plot_heatmap(feature_df)

    print(f"Saved figures to: {FIG_DIR}")


if __name__ == "__main__":
    main()
