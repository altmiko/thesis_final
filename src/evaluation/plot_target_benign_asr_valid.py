from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = REPO_ROOT / "results" / "he_idsr" / "he_idsr_by_classifier.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "he_idsr"

INCLUDED_CLASSIFIERS = ["MLP", "CNN", "LSTM", "CNN-LSTM"]
ATTACKS = [
    ("targeted-benign-latent-pgd", "Latent PGD", "#2E86AB"),
    ("targeted-benign-latent-cw", "Latent CW", "#6A4C93"),
    ("cinput-pgd-target-benign", "Constrained input PGD", "#F18F01"),
    ("cinput-cw-target-benign", "Constrained input CW", "#3A923A"),
]


def _load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"attack_method", "attack_goal", "classifier", "ASR_valid"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    selected_attacks = [attack for attack, _label, _color in ATTACKS]
    target = df.loc[
        (df["attack_goal"] == "target-benign")
        & df["attack_method"].isin(selected_attacks),
        ["attack_method", "classifier", "ASR_valid"],
    ].copy()
    target = target.loc[target["classifier"].isin(INCLUDED_CLASSIFIERS)].copy()

    expected = len(INCLUDED_CLASSIFIERS) * len(ATTACKS)
    if len(target) != expected:
        raise ValueError(
            f"Expected {expected} classifier/attack rows, found {len(target)}"
        )
    if target.duplicated(["attack_method", "classifier"]).any():
        raise ValueError("Duplicate classifier/attack rows found")
    averaged = (
        target.groupby("attack_method", sort=False, as_index=False)
        .agg(ASR_valid=("ASR_valid", "mean"), model_count=("classifier", "nunique"))
    )
    if not (averaged["model_count"] == len(INCLUDED_CLASSIFIERS)).all():
        raise ValueError("Each attack must contain all included classifiers")
    order = [attack for attack, _label, _color in ATTACKS]
    return averaged.set_index("attack_method").loc[order].reset_index()


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "legend.fontsize": 9.5,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _axis_limit(df: pd.DataFrame) -> float:
    maximum = float(df["ASR_valid"].max())
    return max(0.10, np.ceil((maximum + 0.035) / 0.05) * 0.05)


def plot_vertical(df: pd.DataFrame, output_dir: Path) -> None:
    _style()
    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    x = np.arange(len(ATTACKS))
    limit = _axis_limit(df)
    labels = [label for _attack, label, _color in ATTACKS]
    colors = [color for _attack, _label, color in ATTACKS]
    values = df["ASR_valid"].to_numpy(dtype=float)
    bars = ax.bar(
        x,
        values,
        width=0.62,
        color=colors,
        edgecolor="white",
        linewidth=0.8,
    )
    ax.bar_label(
        bars,
        labels=[f"{value * 100:.2f}%" for value in values],
        padding=4,
        fontsize=10,
    )

    ax.set_title(
        "Average Valid ASR for Target-to-Benign Attacks",
        pad=25,
        weight="bold",
    )
    ax.text(
        0.5,
        1.01,
        "Macro-average across MLP, CNN, LSTM, and CNN-LSTM; DualPath excluded",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=9.5,
        color="#4D4D4D",
    )
    ax.set_xlabel("Attack")
    ax.set_ylabel("Average ASR valid")
    ax.set_xticks(x, labels, rotation=12, ha="right")
    ax.set_ylim(0.0, limit)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()

    for suffix in ("png", "pdf"):
        fig.savefig(
            output_dir / f"target_benign_asr_valid_vertical.{suffix}",
            bbox_inches="tight",
        )
    plt.close(fig)


def plot_horizontal(df: pd.DataFrame, output_dir: Path) -> None:
    _style()
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    y = np.arange(len(ATTACKS))
    limit = _axis_limit(df)
    labels = [label for _attack, label, _color in ATTACKS]
    colors = [color for _attack, _label, color in ATTACKS]
    values = df["ASR_valid"].to_numpy(dtype=float)
    bars = ax.barh(
        y,
        values,
        height=0.62,
        color=colors,
        edgecolor="white",
        linewidth=0.8,
    )
    ax.bar_label(
        bars,
        labels=[f"{value * 100:.2f}%" for value in values],
        padding=5,
        fontsize=10,
    )

    ax.set_title(
        "Average Valid ASR for Target-to-Benign Attacks",
        pad=25,
        weight="bold",
    )
    ax.text(
        0.5,
        1.01,
        "Macro-average across MLP, CNN, LSTM, and CNN-LSTM; DualPath excluded",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=9.5,
        color="#4D4D4D",
    )
    ax.set_xlabel("Average ASR valid")
    ax.set_ylabel("Attack")
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0.0, limit)
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.grid(axis="x", color="#D9D9D9", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()

    for suffix in ("png", "pdf"):
        fig.savefig(
            output_dir / f"target_benign_asr_valid_horizontal.{suffix}",
            bbox_inches="tight",
        )
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot target-to-Benign ASR_valid for latent and constrained attacks."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = _load_data(args.input)
    data.to_csv(
        args.output_dir / "target_benign_asr_valid_average_excluding_dualpath.csv",
        index=False,
    )
    plot_vertical(data, args.output_dir)
    plot_horizontal(data, args.output_dir)

    print(
        args.output_dir
        / "target_benign_asr_valid_average_excluding_dualpath.csv"
    )
    print(args.output_dir / "target_benign_asr_valid_vertical.png")
    print(args.output_dir / "target_benign_asr_valid_vertical.pdf")
    print(args.output_dir / "target_benign_asr_valid_horizontal.png")
    print(args.output_dir / "target_benign_asr_valid_horizontal.pdf")


if __name__ == "__main__":
    main()
