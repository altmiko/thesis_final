#!/usr/bin/env python
"""
Generate thesis-ready EDA figures for the CICIoT2023 preprocessing pipeline.

The script visualizes three existing pipeline stages without modifying data:
raw/staged input, cleaned input, and final RobustScaler-transformed arrays.
Every completed figure is written as PDF and SVG and recorded in a caption
manifest for direct use when preparing LaTeX figures.
"""

from __future__ import annotations

import argparse
import json
import math
import textwrap
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter, MaxNLocator
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RANDOM_SEED = 42
NAVY = "#1f3a5f"
SAGE = "#8a9a5b"
LIGHT_BLUE = "#8fa8bd"
LIGHT_SAGE = "#bec7a1"
WARM_GRAY = "#8b8580"
MUTED_RED = "#9f5f5f"
BACKGROUND = "#f7f7f4"

CLASS_ORDER = [
    "Benign",
    "BruteForce",
    "DDoS",
    "DoS",
    "Mirai",
    "Recon",
    "Spoofing",
    "Web",
]

CLASS_COLORS = {
    "Benign": SAGE,
    "BruteForce": NAVY,
    "DDoS": "#9f5f5f",
    "DoS": "#b7835a",
    "Mirai": "#786b8f",
    "Recon": "#55758d",
    "Spoofing": "#658b83",
    "Web": "#9a805d",
}

FEATURE_NAMES = [
    "Header_Length",
    "Protocol Type",
    "Time_To_Live",
    "Rate",
    "fin_flag_number",
    "syn_flag_number",
    "rst_flag_number",
    "psh_flag_number",
    "ack_flag_number",
    "ece_flag_number",
    "cwr_flag_number",
    "ack_count",
    "syn_count",
    "fin_count",
    "rst_count",
    "HTTP",
    "HTTPS",
    "DNS",
    "Telnet",
    "SMTP",
    "SSH",
    "IRC",
    "TCP",
    "UDP",
    "DHCP",
    "ARP",
    "ICMP",
    "IGMP",
    "IPv",
    "LLC",
    "Tot sum",
    "Min",
    "Max",
    "AVG",
    "Std",
    "Tot size",
    "IAT",
    "Number",
    "Variance",
]

BINARY_FEATURES = [
    "HTTP",
    "HTTPS",
    "DNS",
    "Telnet",
    "SMTP",
    "SSH",
    "IRC",
    "TCP",
    "UDP",
    "DHCP",
    "ARP",
    "ICMP",
    "IGMP",
    "IPv",
    "LLC",
]

CONTINUOUS_FEATURES = [
    name
    for name in FEATURE_NAMES
    if name != "Protocol Type" and name not in BINARY_FEATURES
]

DEFAULT_HEAVY_FEATURES = ["Rate", "Tot sum", "IAT", "Number", "Variance"]

CATEGORY_MAP = {
    "DDOS-ICMP_FLOOD": "DDoS",
    "DDOS-UDP_FLOOD": "DDoS",
    "DDOS-TCP_FLOOD": "DDoS",
    "DDOS-PSHACK_FLOOD": "DDoS",
    "DDOS-SYN_FLOOD": "DDoS",
    "DDOS-RSTFINFLOOD": "DDoS",
    "DDOS-SYNONYMOUSIP_FLOOD": "DDoS",
    "DDOS-UDP_FRAGMENTATION": "DDoS",
    "DDOS-ACK_FRAGMENTATION": "DDoS",
    "DDOS-ICMP_FRAGMENTATION": "DDoS",
    "DDOS-HTTP_FLOOD": "DDoS",
    "DDOS-SLOWLORIS": "DDoS",
    "DOS-UDP_FLOOD": "DoS",
    "DOS-TCP_FLOOD": "DoS",
    "DOS-SYN_FLOOD": "DoS",
    "DOS-HTTP_FLOOD": "DoS",
    "MIRAI-GREETH_FLOOD": "Mirai",
    "MIRAI-UDPPLAIN": "Mirai",
    "MIRAI-GREIP_FLOOD": "Mirai",
    "BENIGN": "Benign",
    "MITM-ARPSPOOFING": "Spoofing",
    "DNS_SPOOFING": "Spoofing",
    "RECON-PINGSWEEP": "Recon",
    "RECON-OSSCAN": "Recon",
    "RECON-PORTSCAN": "Recon",
    "RECON-HOSTDISCOVERY": "Recon",
    "VULNERABILITYSCAN": "Recon",
    "BROWSERHIJACKING": "Web",
    "BACKDOOR_MALWARE": "Web",
    "XSS": "Web",
    "SQLINJECTION": "Web",
    "COMMANDINJECTION": "Web",
    "UPLOADING_ATTACK": "Web",
    "DICTIONARYBRUTEFORCE": "BruteForce",
}

THOUSANDS = FuncFormatter(lambda value, _: f"{value:,.0f}")
DECIMALS = FuncFormatter(lambda value, _: f"{value:,.3f}")


def format_axis_value(value: float) -> str:
    """Format axis values without collapsing sub-unit or extreme magnitudes."""
    if value == 0:
        return "0"
    magnitude = abs(value)
    if magnitude >= 10_000_000:
        return f"{value:.0e}".replace("e+", "e")
    if magnitude >= 1_000:
        return f"{value:,.0f}"
    if magnitude >= 1:
        return f"{value:g}"
    return f"{value:.2g}"


COMPACT_NUMBERS = FuncFormatter(lambda value, _: format_axis_value(value))


def compact_symlog_ticks(maximum: float) -> list[float]:
    """Return at most four in-range power-of-ten ticks plus zero."""
    if maximum < 1:
        return [0.0]
    maximum_exponent = max(0, int(math.floor(math.log10(maximum))))
    tick_count = min(4, maximum_exponent + 1)
    exponents = sorted(
        set(
            int(value)
            for value in np.linspace(
                0,
                maximum_exponent,
                num=tick_count,
            )
        )
    )
    return [0.0] + [10.0**exponent for exponent in exponents]


def configure_style() -> None:
    """Apply the shared muted thesis style to all figures."""
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "axes.facecolor": BACKGROUND,
            "figure.facecolor": "white",
            "axes.edgecolor": "#5f6468",
            "axes.grid": True,
            "grid.color": "#d8d8d3",
            "grid.linewidth": 0.6,
            "grid.alpha": 0.7,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


class FigureManifest:
    """Append stable figure-number and caption records to the output manifest."""

    def __init__(self, outdir: Path) -> None:
        self.path = Path(outdir) / "figure_manifest.txt"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            "CICIoT2023 EDA figure manifest\n"
            "================================\n",
            encoding="utf-8",
        )
        self.count = 0

    def append(
        self,
        figure_number: int,
        filename_stem: str,
        caption: str,
    ) -> None:
        line = (
            f"Figure {figure_number:02d} | "
            f"{filename_stem}.[pdf|svg] | {caption.strip()}\n"
        )
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line)
        self.count += 1


def save_fig(
    fig: plt.Figure,
    name: str,
    outdir: Path,
    manifest: FigureManifest,
    figure_number: int,
    caption: str,
) -> None:
    """Save one figure in both required vector formats and update the manifest."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(outdir / f"{name}.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(outdir / f"{name}.svg", bbox_inches="tight", dpi=300)
    plt.close(fig)
    manifest.append(figure_number, name, caption)
    print(f"[Figure {figure_number:02d}] wrote {name}.pdf and {name}.svg")


@dataclass
class ScaledData:
    """Memmapped scaled arrays and their aligned metadata."""

    directory: Path
    feature_names: list[str]
    class_names: list[str]
    x_train: np.ndarray
    y_train: np.ndarray | None
    x_val: np.ndarray | None
    y_val: np.ndarray | None
    x_test: np.ndarray | None
    y_test: np.ndarray | None


# ---------------------------------------------------------------------------
# Loading and numerical helpers
# ---------------------------------------------------------------------------


def warn(message: str) -> None:
    """Print and emit a visible warning without terminating the pipeline."""
    print(f"WARNING: {message}")
    warnings.warn(message, RuntimeWarning, stacklevel=2)


def load_frame(path: Path | None, stage_name: str) -> pd.DataFrame | None:
    """Load a parquet or CSV stage, returning None when the path is unavailable."""
    if path is None:
        warn(f"{stage_name} path was not supplied; skipping that stage.")
        return None
    path = Path(path)
    if not path.exists():
        warn(f"{stage_name} input does not exist: {path}; skipping that stage.")
        return None
    print(f"Loading {stage_name}: {path}")
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        frame = pd.read_parquet(path)
    elif suffix in {".csv", ".txt", ".csv.gz", ".gz"}:
        frame = pd.read_csv(path, low_memory=False)
    else:
        raise ValueError(
            f"Unsupported {stage_name} format '{path.suffix}'. Use parquet or CSV."
        )
    print(f"Loaded {stage_name}: {len(frame):,} rows x {len(frame.columns):,} columns")
    return frame


def _load_json_list(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [str(item) for item in payload]
    if isinstance(payload, dict):
        for key in ("feature_names", "features", "columns", "class_names"):
            value = payload.get(key)
            if isinstance(value, list):
                if value and isinstance(value[0], dict) and "feature" in value[0]:
                    return [str(item["feature"]) for item in value]
                return [str(item) for item in value]
    raise ValueError(f"JSON file does not contain a usable list: {path}")


def load_feature_names(
    explicit_path: Path | None,
    scaled_dir: Path,
    n_features: int,
) -> list[str]:
    """Load feature order from JSON, with a 39-feature schema fallback."""
    candidates: list[Path] = []
    if explicit_path is not None:
        explicit_path = Path(explicit_path)
        candidates.append(explicit_path)
        if not explicit_path.exists():
            warn(
                f"Feature-name JSON does not exist: {explicit_path}; "
                "trying scaled-directory defaults."
            )
    candidates.extend(
        [
            scaled_dir / "feature_names.json",
            scaled_dir / "features.json",
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            names = _load_json_list(candidate)
            if len(names) != n_features:
                raise ValueError(
                    f"{candidate} has {len(names)} feature names, "
                    f"but X_train has {n_features} columns."
                )
            return names
    if n_features == len(FEATURE_NAMES):
        warn(
            "No feature-name JSON was found; using the canonical Modified "
            "Schema A order embedded in this script."
        )
        return FEATURE_NAMES.copy()
    raise FileNotFoundError(
        "A feature-name JSON is required because the scaled array does not "
        f"have the canonical {len(FEATURE_NAMES)} columns."
    )


def _load_optional_npy(path: Path) -> np.ndarray | None:
    return np.load(path, mmap_mode="r") if path.exists() else None


def load_scaled_data(
    scaled_path: Path | None,
    feature_names_path: Path | None,
) -> ScaledData | None:
    """Load scaled train/validation/test arrays and optional 8-class labels."""
    if scaled_path is None:
        warn("Scaled path was not supplied; skipping transformed-stage figures.")
        return None
    scaled_path = Path(scaled_path)
    if not scaled_path.exists():
        warn(
            f"Scaled input does not exist: {scaled_path}; "
            "skipping transformed-stage figures."
        )
        return None

    if scaled_path.is_dir():
        directory = scaled_path
        train_path = directory / "X_train.npy"
    else:
        directory = scaled_path.parent
        train_path = scaled_path

    if not train_path.exists():
        warn(f"Scaled train array not found: {train_path}; skipping scaled stage.")
        return None

    x_train = np.load(train_path, mmap_mode="r")
    if x_train.ndim != 2:
        raise ValueError(f"Expected a 2D X_train array, got shape {x_train.shape}.")

    feature_names = load_feature_names(
        feature_names_path,
        directory,
        x_train.shape[1],
    )

    class_names_path = directory / "category_names.json"
    class_names = (
        _load_json_list(class_names_path)
        if class_names_path.exists()
        else CLASS_ORDER.copy()
    )

    data = ScaledData(
        directory=directory,
        feature_names=feature_names,
        class_names=class_names,
        x_train=x_train,
        y_train=_load_optional_npy(directory / "y_train_cat.npy"),
        x_val=_load_optional_npy(directory / "X_val.npy"),
        y_val=_load_optional_npy(directory / "y_val_cat.npy"),
        x_test=_load_optional_npy(directory / "X_test.npy"),
        y_test=_load_optional_npy(directory / "y_test_cat.npy"),
    )

    for split_name, x_values, y_values in (
        ("train", data.x_train, data.y_train),
        ("validation", data.x_val, data.y_val),
        ("test", data.x_test, data.y_test),
    ):
        if x_values is not None and x_values.shape[1] != len(feature_names):
            raise ValueError(
                f"Scaled {split_name} feature count {x_values.shape[1]} "
                f"does not match {len(feature_names)} names."
            )
        if x_values is not None and y_values is not None:
            if len(x_values) != len(y_values):
                raise ValueError(
                    f"Scaled {split_name} data/label lengths differ: "
                    f"{len(x_values)} vs {len(y_values)}."
                )

    print(
        f"Loaded scaled train array: {data.x_train.shape[0]:,} rows x "
        f"{data.x_train.shape[1]:,} features"
    )
    return data


def resolve_label_series(frame: pd.DataFrame, requested_column: str) -> pd.Series:
    """Resolve an 8-class label series, including mapping the raw Label column."""
    candidates = [requested_column, "label_8class", "category", "Category", "Label"]
    selected = next((name for name in candidates if name in frame.columns), None)
    if selected is None:
        raise KeyError(
            f"No label column found. Tried: {', '.join(dict.fromkeys(candidates))}"
        )

    labels = frame[selected]
    as_strings = labels.astype(str)
    mapped = as_strings.map(CATEGORY_MAP)
    known_raw = as_strings.isin(CATEGORY_MAP)
    if bool(known_raw.all()):
        return mapped.rename(requested_column)
    return as_strings.rename(requested_column)


def normalize_scaled_labels(
    labels: np.ndarray | None,
    class_names: Sequence[str],
) -> np.ndarray | None:
    """Convert encoded scaled labels to readable category names."""
    if labels is None:
        return None
    values = np.asarray(labels)
    if np.issubdtype(values.dtype, np.number):
        result = np.empty(len(values), dtype=object)
        for index, value in enumerate(values.astype(np.int64, copy=False)):
            result[index] = (
                class_names[value]
                if 0 <= value < len(class_names)
                else f"Class {value}"
            )
        return result.astype(str)
    return values.astype(str)


def stratified_indices(
    labels: Sequence[object],
    sample_size: int,
    seed: int = RANDOM_SEED,
) -> np.ndarray:
    """Return an exact-size deterministic sample with proportional class coverage."""
    labels_array = np.asarray(labels)
    n_rows = len(labels_array)
    if sample_size <= 0:
        raise ValueError("sample_size must be positive.")
    if sample_size >= n_rows:
        return np.arange(n_rows, dtype=np.int64)

    classes, inverse, counts = np.unique(
        labels_array,
        return_inverse=True,
        return_counts=True,
    )
    ideal = counts.astype(np.float64) * sample_size / n_rows
    allocation = np.floor(ideal).astype(np.int64)

    if sample_size >= len(classes):
        allocation[(counts > 0) & (allocation == 0)] = 1
    allocation = np.minimum(allocation, counts)

    while int(allocation.sum()) > sample_size:
        removable = np.where(allocation > 1)[0]
        if removable.size == 0:
            removable = np.where(allocation > 0)[0]
        excess = allocation[removable] - ideal[removable]
        allocation[removable[np.argmax(excess)]] -= 1

    remainder_order = np.argsort(-(ideal - np.floor(ideal)))
    while int(allocation.sum()) < sample_size:
        changed = False
        for class_index in remainder_order:
            if allocation[class_index] < counts[class_index]:
                allocation[class_index] += 1
                changed = True
                if int(allocation.sum()) == sample_size:
                    break
        if not changed:
            break

    rng = np.random.default_rng(seed)
    selected: list[np.ndarray] = []
    for class_index, take in enumerate(allocation):
        if take == 0:
            continue
        candidates = np.flatnonzero(inverse == class_index)
        selected.append(rng.choice(candidates, size=int(take), replace=False))
    result = np.concatenate(selected).astype(np.int64, copy=False)
    rng.shuffle(result)
    return result


def sample_frame(
    frame: pd.DataFrame,
    labels: Sequence[object],
    sample_size: int,
) -> pd.DataFrame:
    """Create a deterministic stratified plotting sample from a dataframe."""
    indices = stratified_indices(labels, min(sample_size, len(frame)))
    return frame.iloc[indices]


def available_features(
    requested: Iterable[str],
    columns: Iterable[str],
    context: str,
) -> list[str]:
    """Filter a feature list to present columns and warn about omissions."""
    column_set = set(columns)
    present = [feature for feature in requested if feature in column_set]
    missing = [feature for feature in requested if feature not in column_set]
    if missing:
        warn(f"{context} is missing features: {', '.join(missing)}")
    return present


def finite_values(values: Sequence[object]) -> np.ndarray:
    """Convert a one-dimensional sequence to finite float64 values."""
    array = np.asarray(values, dtype=np.float64)
    return array[np.isfinite(array)]


def choose_histogram_bins(
    values: np.ndarray,
    bins: int = 70,
) -> tuple[np.ndarray, str]:
    """Choose linear or logarithmic bins while retaining raw measurement units."""
    finite = finite_values(values)
    if finite.size == 0:
        return np.linspace(0.0, 1.0, bins + 1), "linear"

    positive = finite[finite > 0]
    if (
        finite.min(initial=0.0) >= 0.0
        and positive.size >= 2
        and positive.max() / max(positive.min(), np.finfo(float).tiny) >= 100.0
    ):
        lower = max(float(positive.min()), np.finfo(float).tiny)
        upper = float(positive.max())
        return np.geomspace(lower, upper, bins + 1), "log"

    lower = float(finite.min())
    upper = float(finite.max())
    if math.isclose(lower, upper):
        padding = max(abs(lower) * 0.05, 0.5)
        lower -= padding
        upper += padding
    return np.linspace(lower, upper, bins + 1), "linear"


def class_color(label: str) -> str:
    """Return a stable class color, including a neutral fallback."""
    return CLASS_COLORS.get(str(label), WARM_GRAY)


def ordered_classes(labels: Sequence[object]) -> list[str]:
    """Order known CICIoT categories consistently and append unknown labels."""
    present = [str(value) for value in pd.unique(np.asarray(labels))]
    known = [label for label in CLASS_ORDER if label in present]
    unknown = sorted(set(present) - set(known))
    return known + unknown


def scatter_by_class(
    ax: plt.Axes,
    coordinates: np.ndarray,
    labels: np.ndarray,
) -> None:
    """Draw a consistent class-colored vector scatter plot."""
    for label in ordered_classes(labels):
        mask = labels == label
        ax.scatter(
            coordinates[mask, 0],
            coordinates[mask, 1],
            s=8,
            alpha=0.42,
            color=class_color(label),
            edgecolors="none",
            label=label,
        )
    ax.legend(markerscale=2.0, ncol=2, loc="best")


def compute_psi_jsd(
    train_values: Sequence[float],
    test_values: Sequence[float],
    bins: int = 10,
    epsilon: float = 1e-8,
) -> tuple[float, float]:
    """Compute quantile-binned PSI and base-2 Jensen-Shannon divergence."""
    train = finite_values(train_values)
    test = finite_values(test_values)
    if train.size == 0 or test.size == 0:
        return float("nan"), float("nan")

    quantile_edges = np.unique(
        np.quantile(train, np.linspace(0.0, 1.0, bins + 1))
    ).astype(np.float64, copy=False)
    if quantile_edges.size == 1:
        constant = float(quantile_edges[0])
        edges = np.array(
            [
                -np.inf,
                np.nextafter(constant, -np.inf),
                np.nextafter(constant, np.inf),
                np.inf,
            ],
            dtype=np.float64,
        )
    elif quantile_edges.size < bins + 1:
        midpoints = quantile_edges[:-1] + np.diff(quantile_edges) / 2.0
        edges = np.concatenate(([-np.inf], midpoints, [np.inf]))
    else:
        edges = quantile_edges.copy()
        edges[0] = -np.inf
        edges[-1] = np.inf
    train_counts, _ = np.histogram(train, bins=edges)
    test_counts, _ = np.histogram(test, bins=edges)

    train_prob = train_counts.astype(np.float64) + epsilon
    test_prob = test_counts.astype(np.float64) + epsilon
    train_prob /= train_prob.sum()
    test_prob /= test_prob.sum()

    psi = np.sum((test_prob - train_prob) * np.log(test_prob / train_prob))
    midpoint = 0.5 * (train_prob + test_prob)
    jsd = 0.5 * np.sum(train_prob * np.log2(train_prob / midpoint))
    jsd += 0.5 * np.sum(test_prob * np.log2(test_prob / midpoint))
    return float(psi), float(jsd)


def run_figure(description: str, function: Callable[[], None]) -> None:
    """Run one figure independently so a local data issue does not abort the rest."""
    try:
        function()
    except Exception as exc:
        warn(f"Skipped {description}: {type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# Stage 1: raw/staged figures
# ---------------------------------------------------------------------------


def figure_01_class_distribution(
    labels: pd.Series,
    outdir: Path,
    manifest: FigureManifest,
    log_y: bool,
) -> None:
    """Support the dataset-composition section with full raw 8-class counts."""
    counts = labels.value_counts()
    order = [label for label in CLASS_ORDER if label in counts.index]
    order.extend(sorted(set(counts.index.astype(str)) - set(order)))
    counts = counts.reindex(order).fillna(0).astype(np.int64)
    percentages = counts / counts.sum() * 100.0

    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    x_positions = np.arange(len(counts))
    bars = ax.bar(
        x_positions,
        counts.values,
        color=[class_color(label) for label in counts.index],
        width=0.72,
    )
    ax.set_xticks(x_positions)
    ax.set_xticklabels(counts.index, rotation=25, ha="right")
    ax.set_ylabel("Samples")
    ax.set_title("Raw/Staged CICIoT2023 Distribution by 8-Class Category")
    ax.yaxis.set_major_formatter(THOUSANDS)
    if log_y:
        ax.set_yscale("log")
        ax.set_ylim(bottom=max(0.8, counts[counts > 0].min() * 0.6))

    linear_offset = max(float(counts.max()) * 0.018, 1.0)
    for bar, count, percentage in zip(bars, counts.values, percentages.values):
        y_value = count * 1.12 if log_y else count + linear_offset
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y_value,
            f"{count:,}\n({percentage:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    save_fig(
        fig,
        "fig01_raw_class_distribution",
        outdir,
        manifest,
        1,
        "Distribution of the staged CICIoT2023 sample across the eight "
        "network-traffic categories, with absolute counts and dataset shares.",
    )


def figure_02_missing_infinite_summary(
    frame: pd.DataFrame,
    feature_names: Sequence[str],
    outdir: Path,
    manifest: FigureManifest,
) -> None:
    """Support the cleaning rationale with full-frame NaN and infinity counts."""
    features = available_features(
        feature_names,
        frame.columns,
        "Raw missing/infinite summary",
    )
    if not features:
        raise ValueError("No schema features are present.")

    nan_counts = pd.Series(
        {feature: int(frame[feature].isna().sum()) for feature in features}
    )
    inf_counts = pd.Series(
        {
            feature: int(
                np.isinf(
                    pd.to_numeric(frame[feature], errors="coerce").to_numpy(
                        dtype=np.float64,
                        copy=False,
                    )
                ).sum()
            )
            for feature in features
        }
    )
    order = (nan_counts + inf_counts).sort_values(ascending=True).index

    fig_height = max(7.0, 0.27 * len(features) + 2.0)
    fig, ax = plt.subplots(figsize=(11.5, fig_height))
    positions = np.arange(len(order))
    ax.barh(
        positions - 0.18,
        nan_counts.loc[order],
        height=0.36,
        color=NAVY,
        label="NaN",
    )
    ax.barh(
        positions + 0.18,
        inf_counts.loc[order],
        height=0.36,
        color=SAGE,
        label="+/- infinity",
    )
    ax.set_yticks(positions)
    ax.set_yticklabels(order)
    ax.set_xlabel("Invalid values")
    ax.set_title("Raw/Staged Missing and Infinite Values by Feature")
    ax.xaxis.set_major_formatter(THOUSANDS)
    ax.legend(loc="lower right")

    save_fig(
        fig,
        "fig02_raw_missing_infinite_summary",
        outdir,
        manifest,
        2,
        "Per-feature NaN and positive/negative infinity counts before the "
        "cleaning stage, computed over the complete staged dataframe.",
    )


def figure_03_raw_heavy_tail_histograms(
    frame: pd.DataFrame,
    plot_frame: pd.DataFrame,
    heavy_features: Sequence[str],
    outdir: Path,
    manifest: FigureManifest,
) -> None:
    """Support the clipping/log-transform rationale with pre-clean distributions."""
    features = available_features(
        heavy_features,
        frame.columns,
        "Raw heavy-tail histograms",
    )
    if not features:
        raise ValueError("None of the requested heavy-tailed features are present.")

    n_cols = 2 if len(features) <= 4 else 3
    n_rows = math.ceil(len(features) / n_cols)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5.1 * n_cols, 3.8 * n_rows),
        squeeze=False,
    )

    for ax, feature in zip(axes.flat, features):
        full = finite_values(pd.to_numeric(frame[feature], errors="coerce"))
        sample = finite_values(pd.to_numeric(plot_frame[feature], errors="coerce"))
        if full.size == 0 or sample.size == 0:
            ax.text(0.5, 0.5, "No finite values", ha="center", va="center")
            ax.set_title(feature)
            continue

        p9999 = float(np.quantile(full, 0.9999))
        bins, scale = choose_histogram_bins(full)
        histogram_values = sample[sample > 0] if scale == "log" else sample
        ax.hist(histogram_values, bins=bins, color=NAVY, alpha=0.82)
        ax.axvline(
            p9999,
            color=SAGE,
            linewidth=2.0,
            linestyle="--",
            label=f"99.99th pct. = {p9999:,.3g}",
        )
        if scale == "log":
            ax.set_xscale("log")
            zero_count = int(np.count_nonzero(sample == 0))
            if zero_count:
                ax.text(
                    0.02,
                    0.96,
                    f"Zeros in plot sample: {zero_count:,}",
                    transform=ax.transAxes,
                    va="top",
                    fontsize=8,
                )
        ax.set_yscale("log")
        ax.set_title(feature)
        ax.set_xlabel("Raw value")
        ax.set_ylabel("Frequency (log scale)")
        ax.legend(fontsize=8)

    for ax in axes.flat[len(features) :]:
        ax.axis("off")
    fig.suptitle(
        "Raw Heavy-Tailed Feature Distributions Before Clipping",
        fontsize=13,
        y=1.01,
    )
    save_fig(
        fig,
        "fig03_raw_heavy_tail_histograms",
        outdir,
        manifest,
        3,
        "Pre-cleaning distributions of selected heavy-tailed features in raw "
        "units; dashed lines mark the full-frame 99.99th percentiles.",
    )


def figure_04_raw_continuous_boxplots(
    frame: pd.DataFrame,
    continuous_features: Sequence[str],
    outdir: Path,
    manifest: FigureManifest,
) -> None:
    """Support the outlier-treatment section with full-frame robust box summaries."""
    features = available_features(
        continuous_features,
        frame.columns,
        "Raw continuous-feature boxplots",
    )
    if not features:
        raise ValueError("No continuous schema features are present.")

    n_cols = 4
    n_rows = math.ceil(len(features) / n_cols)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.1 * n_cols, 2.4 * n_rows),
        squeeze=False,
    )

    for ax, feature in zip(axes.flat, features):
        values = finite_values(pd.to_numeric(frame[feature], errors="coerce"))
        if values.size == 0:
            ax.text(0.5, 0.5, "No finite values", ha="center", va="center")
            ax.set_title(feature)
            continue

        minimum, q1, median, q3, p999, p9999, maximum = np.quantile(
            values,
            [0.0, 0.25, 0.5, 0.75, 0.999, 0.9999, 1.0],
        )
        iqr = q3 - q1
        lower_whisker = max(minimum, q1 - 1.5 * iqr)
        upper_whisker = min(maximum, q3 + 1.5 * iqr)
        stats = [
            {
                "label": "",
                "med": median,
                "q1": q1,
                "q3": q3,
                "whislo": lower_whisker,
                "whishi": upper_whisker,
                "fliers": [],
            }
        ]
        ax.bxp(
            stats,
            vert=False,
            showfliers=False,
            patch_artist=True,
            widths=0.45,
            boxprops={"facecolor": LIGHT_BLUE, "edgecolor": NAVY},
            medianprops={"color": NAVY, "linewidth": 1.8},
            whiskerprops={"color": NAVY},
            capprops={"color": NAVY},
        )
        markers = [p999, p9999, maximum]
        ax.scatter(
            markers,
            [1.0, 1.0, 1.0],
            marker="|",
            s=[35, 55, 75],
            color=[SAGE, "#6f7f45", MUTED_RED],
            zorder=3,
        )
        ax.set_yticks([])
        ax.set_title(feature, fontsize=9)
        use_symlog = (
            minimum >= 0
            and maximum > 1_000
            and maximum > 100 * max(float(q3), 1.0)
        )
        if use_symlog:
            linthresh = max(float(q3), 1.0)
            ax.set_xscale("symlog", linthresh=linthresh)
            ax.set_xticks(compact_symlog_ticks(float(maximum)))
        else:
            ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.xaxis.set_major_formatter(COMPACT_NUMBERS)
        ax.text(
            0.02,
            0.08,
            f"max={maximum:,.3g}",
            transform=ax.transAxes,
            fontsize=7,
        )

    for ax in axes.flat[len(features) :]:
        ax.axis("off")
    fig.suptitle(
        "Pre-Clip Boxplot Grid for the 23 Continuous Features",
        fontsize=13,
        y=1.005,
    )
    save_fig(
        fig,
        "fig04_raw_continuous_boxplot_grid",
        outdir,
        manifest,
        4,
        "Full-frame robust box summaries for all 23 continuous features before "
        "clipping; markers identify the 99.9th percentile, 99.99th percentile, "
        "and observed maximum.",
    )


# ---------------------------------------------------------------------------
# Stage 2: cleaning figures
# ---------------------------------------------------------------------------


def figure_05_cleaning_before_after_histograms(
    raw_frame: pd.DataFrame,
    clean_frame: pd.DataFrame,
    raw_plot: pd.DataFrame,
    clean_plot: pd.DataFrame,
    heavy_features: Sequence[str],
    outdir: Path,
    manifest: FigureManifest,
) -> None:
    """Support the cleaning-effects section with raw-versus-clean overlays."""
    features = [
        feature
        for feature in heavy_features
        if feature in raw_frame.columns and feature in clean_frame.columns
    ]
    if not features:
        raise ValueError("No requested heavy features exist in both stages.")

    n_cols = 2 if len(features) <= 4 else 3
    n_rows = math.ceil(len(features) / n_cols)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5.1 * n_cols, 3.8 * n_rows),
        squeeze=False,
    )

    for ax, feature in zip(axes.flat, features):
        raw_full = finite_values(pd.to_numeric(raw_frame[feature], errors="coerce"))
        raw_values = finite_values(
            pd.to_numeric(raw_plot[feature], errors="coerce")
        )
        clean_values = finite_values(
            pd.to_numeric(clean_plot[feature], errors="coerce")
        )
        if raw_full.size == 0 or raw_values.size == 0 or clean_values.size == 0:
            ax.text(0.5, 0.5, "Insufficient finite values", ha="center", va="center")
            ax.set_title(feature)
            continue

        clip_point = float(np.quantile(raw_full, 0.9999))
        bins, scale = choose_histogram_bins(raw_full)
        if scale == "log":
            raw_values = raw_values[raw_values > 0]
            clean_values = clean_values[clean_values > 0]
        ax.hist(
            raw_values,
            bins=bins,
            density=True,
            histtype="step",
            linewidth=1.6,
            color=NAVY,
            label="Before cleaning",
        )
        ax.hist(
            clean_values,
            bins=bins,
            density=True,
            histtype="stepfilled",
            linewidth=1.2,
            color=SAGE,
            alpha=0.28,
            label="After cleaning",
        )
        ax.axvline(
            clip_point,
            color=MUTED_RED,
            linestyle="--",
            linewidth=1.5,
            label="Raw 99.99th pct.",
        )
        if scale == "log":
            ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(feature)
        ax.set_xlabel("Feature value")
        ax.set_ylabel("Density (log scale)")
        ax.legend(fontsize=7)

    for ax in axes.flat[len(features) :]:
        ax.axis("off")
    fig.suptitle(
        "Heavy-Tailed Features Before and After Cleaning",
        fontsize=13,
        y=1.01,
    )
    save_fig(
        fig,
        "fig05_cleaning_before_after_histograms",
        outdir,
        manifest,
        5,
        "Overlaid raw and cleaned distributions showing the combined effect of "
        "99.99th-percentile clipping, non-negativity enforcement, and coercion.",
    )


def figure_06_clean_spearman_heatmap(
    clean_frame: pd.DataFrame,
    continuous_features: Sequence[str],
    outdir: Path,
    manifest: FigureManifest,
) -> None:
    """Support perturbation-mask tiering with cleaned full-frame correlations."""
    features = available_features(
        continuous_features,
        clean_frame.columns,
        "Clean Spearman heatmap",
    )
    if len(features) < 2:
        raise ValueError("At least two continuous features are required.")

    print("Computing full-frame Spearman correlation matrix...")
    numeric = clean_frame[features].apply(pd.to_numeric, errors="coerce")
    correlation = numeric.corr(method="spearman")
    cmap = LinearSegmentedColormap.from_list(
        "navy_sage_diverging",
        [NAVY, "#f3f1ea", SAGE],
    )

    fig, ax = plt.subplots(figsize=(13.2, 11.3))
    image = ax.imshow(
        correlation.to_numpy(),
        cmap=cmap,
        vmin=-1.0,
        vmax=1.0,
        aspect="equal",
    )
    positions = np.arange(len(features))
    ax.set_xticks(positions)
    ax.set_yticks(positions)
    ax.set_xticklabels(features, rotation=90, fontsize=7)
    ax.set_yticklabels(features, fontsize=7)
    ax.grid(False)
    ax.set_title(
        f"Cleaned Continuous-Feature Spearman Correlation (n={len(clean_frame):,})"
    )
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Spearman rho")

    save_fig(
        fig,
        "fig06_clean_spearman_correlation",
        outdir,
        manifest,
        6,
        "Spearman correlation matrix for the 23 cleaned continuous features, "
        "used to motivate correlation-aware perturbation-mask tiering.",
    )


def _split_label_map(scaled: ScaledData | None) -> dict[str, np.ndarray]:
    if scaled is None:
        return {}
    mapping: dict[str, np.ndarray] = {}
    for name, labels in (
        ("Train", scaled.y_train),
        ("Validation", scaled.y_val),
        ("Test", scaled.y_test),
    ):
        normalized = normalize_scaled_labels(labels, scaled.class_names)
        if normalized is not None:
            mapping[name] = normalized
    return mapping


def figure_07_clean_class_count_table(
    clean_labels: pd.Series,
    split_labels: dict[str, np.ndarray],
    outdir: Path,
    manifest: FigureManifest,
) -> None:
    """Support sample-accounting with clean and available split class counts."""
    all_labels = set(clean_labels.astype(str))
    for values in split_labels.values():
        all_labels.update(values.astype(str))
    order = [label for label in CLASS_ORDER if label in all_labels]
    order.extend(sorted(all_labels - set(order)))

    table_data = pd.DataFrame(index=order)
    table_data["Clean"] = clean_labels.astype(str).value_counts().reindex(order).fillna(0)
    for split_name, labels in split_labels.items():
        counts = pd.Series(labels).value_counts().reindex(order).fillna(0)
        table_data[split_name] = counts
    table_data = table_data.astype(np.int64)
    table_data.loc["Total"] = table_data.sum(axis=0)

    display = table_data.apply(
        lambda column: column.map(lambda value: f"{value:,}")
    )
    fig_height = max(4.8, 0.48 * len(display) + 1.8)
    fig, ax = plt.subplots(figsize=(9.5, fig_height))
    ax.axis("off")
    table = ax.table(
        cellText=display.values,
        rowLabels=display.index,
        colLabels=display.columns,
        cellLoc="right",
        rowLoc="left",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.45)
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor("#d0d0ca")
        if row == 0:
            cell.set_facecolor(NAVY)
            cell.set_text_props(color="white", weight="bold")
        elif row == len(display):
            cell.set_facecolor(LIGHT_SAGE)
            cell.set_text_props(weight="bold")
        elif column == -1:
            cell.set_facecolor("#e9ece8")
    ax.set_title("Cleaned and Split Sample Counts by 8-Class Category", pad=18)

    save_fig(
        fig,
        "fig07_clean_class_count_table",
        outdir,
        manifest,
        7,
        "Per-category sample counts after cleaning, with train, validation, and "
        "test counts included when aligned scaled-label arrays are available.",
    )


# ---------------------------------------------------------------------------
# Stage 3: transformed figures
# ---------------------------------------------------------------------------


def scaled_continuous_features(feature_names: Sequence[str]) -> list[str]:
    """Resolve the 23 continuous schema features in scaled-array order."""
    return [
        name
        for name in feature_names
        if name != "Protocol Type" and name not in BINARY_FEATURES
    ]


def compute_feature_iqrs(
    values: np.ndarray,
    feature_names: Sequence[str],
    selected_features: Sequence[str],
) -> dict[str, float]:
    """Compute per-feature IQRs one column at a time to limit peak memory."""
    index = {name: position for position, name in enumerate(feature_names)}
    result: dict[str, float] = {}
    for feature in selected_features:
        column = finite_values(values[:, index[feature]])
        if column.size == 0:
            result[feature] = float("nan")
        else:
            q1, q3 = np.quantile(column, [0.25, 0.75])
            result[feature] = float(q3 - q1)
    return result


def figure_08_scaled_feature_distributions(
    scaled: ScaledData,
    representative_features: Sequence[str],
    plot_sample: int,
    near_zero_threshold: float,
    outdir: Path,
    manifest: FigureManifest,
) -> None:
    """Support transform validation and near-zero-IQR freeze governance."""
    continuous = scaled_continuous_features(scaled.feature_names)
    representatives = [
        feature for feature in representative_features if feature in continuous
    ]
    if not representatives:
        representatives = continuous[: min(5, len(continuous))]
    if not representatives:
        raise ValueError("No continuous features are available in scaled arrays.")

    iqrs = compute_feature_iqrs(
        scaled.x_train,
        scaled.feature_names,
        continuous,
    )
    near_zero = [
        feature
        for feature, iqr in iqrs.items()
        if np.isfinite(iqr) and iqr <= near_zero_threshold
    ]

    labels = normalize_scaled_labels(scaled.y_train, scaled.class_names)
    if labels is None:
        labels = np.repeat("All samples", len(scaled.x_train))
    sample_indices = stratified_indices(
        labels,
        min(plot_sample, len(scaled.x_train)),
    )
    sample = np.asarray(scaled.x_train[sample_indices], dtype=np.float64)
    feature_index = {
        name: position for position, name in enumerate(scaled.feature_names)
    }

    n_cols = 2 if len(representatives) <= 4 else 3
    n_rows = math.ceil(len(representatives) / n_cols)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5.1 * n_cols, 3.7 * n_rows),
        squeeze=False,
    )
    for ax, feature in zip(axes.flat, representatives):
        values = finite_values(sample[:, feature_index[feature]])
        ax.hist(values, bins=70, color=NAVY, alpha=0.82)
        ax.axvline(0.0, color=SAGE, linestyle="--", linewidth=1.8, label="0")
        median = float(np.median(values)) if values.size else float("nan")
        iqr = iqrs[feature]
        ax.set_title(
            f"{feature}\nmedian={median:,.3f}, IQR={iqr:,.3g}",
            color=MUTED_RED if feature in near_zero else "black",
        )
        ax.set_xlabel("Transformed value")
        ax.set_ylabel("Frequency")
        ax.yaxis.set_major_formatter(THOUSANDS)
        if feature in near_zero:
            ax.text(
                0.98,
                0.94,
                "NEAR-ZERO IQR: frozen",
                transform=ax.transAxes,
                ha="right",
                va="top",
                color=MUTED_RED,
                weight="bold",
                fontsize=8,
            )

    near_zero_text = (
        ", ".join(near_zero)
        if near_zero
        else "None among the 23 continuous features"
    )
    unused_axes = list(axes.flat[len(representatives) :])
    for ax in unused_axes:
        ax.axis("off")
    note = (
        f"Near-zero-IQR features\n(threshold {near_zero_threshold:g})\n\n"
        f"{textwrap.fill(near_zero_text, width=38)}"
    )
    if unused_axes:
        unused_axes[0].text(
            0.05,
            0.92,
            note,
            transform=unused_axes[0].transAxes,
            va="top",
            ha="left",
            fontsize=9,
            color=MUTED_RED if near_zero else WARM_GRAY,
            bbox={
                "boxstyle": "round,pad=0.6",
                "facecolor": "#f0f1ec",
                "edgecolor": "#c8cbc2",
            },
        )
    else:
        fig.text(
            0.5,
            0.005,
            textwrap.fill(note.replace("\n\n", ": "), width=110),
            ha="center",
            fontsize=8,
            color=MUTED_RED if near_zero else WARM_GRAY,
        )
    fig.suptitle(
        "Representative Continuous Features After the Final Transform",
        fontsize=13,
        y=1.02,
    )

    save_fig(
        fig,
        "fig08_scaled_continuous_distributions",
        outdir,
        manifest,
        8,
        "Representative continuous-feature distributions after configured log1p "
        "transforms and train-fitted RobustScaler scaling; near-zero-IQR features "
        "are explicitly flagged for freezing during adversarial search.",
    )


def figure_09_scaled_pca(
    scaled: ScaledData,
    plot_sample: int,
    outdir: Path,
    manifest: FigureManifest,
) -> None:
    """Support transformed-space structure analysis with a 2D PCA projection."""
    labels = normalize_scaled_labels(scaled.y_train, scaled.class_names)
    if labels is None:
        raise ValueError("y_train_cat.npy is required for the class-colored PCA.")
    indices = stratified_indices(labels, min(plot_sample, len(labels)))
    values = np.asarray(scaled.x_train[indices], dtype=np.float64)
    finite_rows = np.isfinite(values).all(axis=1)
    values = values[finite_rows]
    sampled_labels = labels[indices][finite_rows]
    if len(values) < 3:
        raise ValueError("Too few finite scaled samples for PCA.")

    pca = PCA(n_components=2, random_state=RANDOM_SEED)
    coordinates = pca.fit_transform(values)
    explained = pca.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(10.5, 8.0))
    scatter_by_class(ax, coordinates, sampled_labels)
    ax.set_xlabel(f"PC1 ({explained[0]:.2%} explained variance)")
    ax.set_ylabel(f"PC2 ({explained[1]:.2%} explained variance)")
    ax.set_title(
        "PCA of Scaled CICIoT2023 Features "
        f"(PC1 + PC2 = {explained.sum():.2%})"
    )

    save_fig(
        fig,
        "fig09_scaled_pca_by_class",
        outdir,
        manifest,
        9,
        "Two-dimensional PCA projection of a stratified scaled-train sample, "
        "colored by the eight traffic categories with PC1/PC2 variance reported.",
    )


def figure_10_scaled_embedding(
    scaled: ScaledData,
    embedding_sample: int,
    perplexity: float,
    outdir: Path,
    manifest: FigureManifest,
) -> None:
    """Support nonlinear cluster inspection using UMAP or a t-SNE fallback."""
    labels = normalize_scaled_labels(scaled.y_train, scaled.class_names)
    if labels is None:
        raise ValueError(
            "y_train_cat.npy is required for the class-colored nonlinear embedding."
        )
    indices = stratified_indices(labels, min(embedding_sample, len(labels)))
    values = np.asarray(scaled.x_train[indices], dtype=np.float64)
    finite_rows = np.isfinite(values).all(axis=1)
    values = values[finite_rows]
    sampled_labels = labels[indices][finite_rows]
    if len(values) < 5:
        raise ValueError("Too few finite scaled samples for nonlinear embedding.")

    pre_components = min(30, values.shape[1], len(values) - 1)
    reduced = PCA(
        n_components=pre_components,
        random_state=RANDOM_SEED,
    ).fit_transform(values)

    method_name = "UMAP"
    filename = "fig10_scaled_umap_by_class"
    try:
        import umap

        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=min(30, len(reduced) - 1),
            min_dist=0.1,
            metric="euclidean",
            random_state=RANDOM_SEED,
            n_jobs=1,
        )
        coordinates = reducer.fit_transform(reduced)
        subtitle = f"n={len(reduced):,}, n_neighbors={min(30, len(reduced) - 1)}"
    except Exception as exc:
        warn(
            f"UMAP could not be used ({type(exc).__name__}: {exc}); "
            "falling back to t-SNE."
        )
        method_name = "t-SNE"
        filename = "fig10_scaled_tsne_by_class"
        effective_perplexity = min(float(perplexity), (len(reduced) - 1) / 3.0)
        effective_perplexity = max(2.0, effective_perplexity)
        reducer = TSNE(
            n_components=2,
            perplexity=effective_perplexity,
            learning_rate="auto",
            init="pca",
            random_state=RANDOM_SEED,
        )
        coordinates = reducer.fit_transform(reduced)
        subtitle = (
            f"n={len(reduced):,}, perplexity={effective_perplexity:,.1f}, "
            f"KL={reducer.kl_divergence_:,.3f}"
        )

    fig, ax = plt.subplots(figsize=(10.5, 8.0))
    scatter_by_class(ax, coordinates, sampled_labels)
    ax.set_xlabel(f"{method_name} dimension 1")
    ax.set_ylabel(f"{method_name} dimension 2")
    ax.set_title(f"{method_name} of Scaled CICIoT2023 Features ({subtitle})")

    save_fig(
        fig,
        filename,
        outdir,
        manifest,
        10,
        f"{method_name} projection of a small stratified scaled-train sample, "
        "colored by the eight traffic categories to reveal nonlinear cluster "
        "structure.",
    )


def compute_shift_table(
    scaled: ScaledData,
    continuous_features: Sequence[str],
    bins: int,
) -> pd.DataFrame:
    """Compute full train-versus-test PSI and JSD for each continuous feature."""
    if scaled.x_test is None:
        raise FileNotFoundError("X_test.npy is required for PSI/JSD analysis.")
    feature_index = {
        name: position for position, name in enumerate(scaled.feature_names)
    }
    rows: list[dict[str, float | str]] = []
    for feature in continuous_features:
        index = feature_index[feature]
        psi, jsd = compute_psi_jsd(
            scaled.x_train[:, index],
            scaled.x_test[:, index],
            bins=bins,
        )
        rows.append({"feature": feature, "PSI": psi, "JSD": jsd})
    return pd.DataFrame(rows).sort_values("PSI", ascending=False).reset_index(drop=True)


def figure_11_train_test_shift(
    scaled: ScaledData,
    psi_bins: int,
    outdir: Path,
    manifest: FigureManifest,
) -> Path:
    """Support distribution-shift checks with per-feature PSI and JSD."""
    continuous = scaled_continuous_features(scaled.feature_names)
    if not continuous:
        raise ValueError("No continuous features are available for shift analysis.")
    print("Computing full train-versus-test PSI and JSD...")
    shift = compute_shift_table(scaled, continuous, bins=psi_bins)
    csv_path = Path(outdir) / "train_test_distribution_shift.csv"
    shift.to_csv(csv_path, index=False)

    fig, axes = plt.subplots(1, 2, figsize=(15.0, 9.2))
    for ax, metric, color in (
        (axes[0], "PSI", NAVY),
        (axes[1], "JSD", SAGE),
    ):
        ordered = shift.sort_values(metric, ascending=True)
        positions = np.arange(len(ordered))
        ax.barh(positions, ordered[metric], color=color, alpha=0.88)
        ax.set_yticks(positions)
        ax.set_yticklabels(ordered["feature"], fontsize=8)
        ax.set_xlabel(metric)
        ax.set_title(f"Train vs Test {metric}")
        ax.xaxis.set_major_formatter(DECIMALS)
        if metric == "PSI":
            ax.axvline(
                0.10,
                color="#b7835a",
                linestyle="--",
                linewidth=1.2,
                label="PSI = 0.10",
            )
            ax.axvline(
                0.25,
                color=MUTED_RED,
                linestyle=":",
                linewidth=1.4,
                label="PSI = 0.25",
            )
            ax.legend(loc="lower right")
    fig.suptitle(
        "Scaled Continuous-Feature Distribution Shift: Train vs Test",
        fontsize=13,
        y=1.01,
    )

    save_fig(
        fig,
        "fig11_scaled_train_test_distribution_shift",
        outdir,
        manifest,
        11,
        "Per-feature train-versus-test Population Stability Index and "
        "Jensen-Shannon divergence in scaled space; numeric values are exported "
        "to train_test_distribution_shift.csv.",
    )
    return csv_path


# ---------------------------------------------------------------------------
# Command-line entrypoint
# ---------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line paths and plotting controls."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate vector EDA figures before cleaning, after cleaning, and "
            "after the CICIoT2023 final feature transform."
        )
    )
    parser.add_argument(
        "--raw",
        type=Path,
        default=Path("data/processed/raw_loaded.parquet"),
        help="Raw/staged parquet or CSV path.",
    )
    parser.add_argument(
        "--clean",
        type=Path,
        default=Path("data/processed/cleaned_data.parquet"),
        help="Cleaned parquet or CSV path.",
    )
    parser.add_argument(
        "--scaled",
        type=Path,
        default=Path("data/processed"),
        help="Directory containing scaled arrays, or the X_train.npy path.",
    )
    parser.add_argument(
        "--feature-names",
        type=Path,
        default=None,
        help="Optional feature-name JSON; otherwise search the scaled directory.",
    )
    parser.add_argument(
        "--labels",
        default="label_8class",
        help="8-class label column for raw and cleaned dataframes.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("eda_figures"),
        help="Output directory for vector figures, manifest, and shift CSV.",
    )
    parser.add_argument(
        "--plot-sample",
        type=int,
        default=50_000,
        help="Maximum stratified sample used for general plotting.",
    )
    parser.add_argument(
        "--embedding-sample",
        type=int,
        default=5_000,
        help="Maximum stratified sample used for UMAP/t-SNE.",
    )
    parser.add_argument(
        "--perplexity",
        type=float,
        default=40.0,
        help="t-SNE perplexity when UMAP is unavailable.",
    )
    parser.add_argument(
        "--heavy-features",
        nargs="+",
        default=DEFAULT_HEAVY_FEATURES,
        help="Heavy-tailed feature names used in raw and cleaning comparisons.",
    )
    parser.add_argument(
        "--scaled-features",
        nargs="+",
        default=None,
        help="Representative continuous features for transformed histograms.",
    )
    parser.add_argument(
        "--class-log-y",
        action="store_true",
        help="Use a logarithmic y-axis for the raw class distribution.",
    )
    parser.add_argument(
        "--psi-bins",
        type=int,
        default=10,
        help="Number of train-quantile bins for PSI and JSD.",
    )
    parser.add_argument(
        "--near-zero-iqr",
        type=float,
        default=1e-6,
        help="IQR threshold used to flag frozen transformed features.",
    )
    args = parser.parse_args(argv)
    if args.plot_sample <= 0 or args.embedding_sample <= 0:
        parser.error("--plot-sample and --embedding-sample must be positive.")
    if args.perplexity <= 0:
        parser.error("--perplexity must be positive.")
    if args.psi_bins < 2:
        parser.error("--psi-bins must be at least 2.")
    if args.near_zero_iqr < 0:
        parser.error("--near-zero-iqr cannot be negative.")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    """Generate all figures whose stage inputs are available."""
    args = parse_args(argv)
    np.random.seed(RANDOM_SEED)
    configure_style()
    args.outdir.mkdir(parents=True, exist_ok=True)
    manifest = FigureManifest(args.outdir)

    raw_frame = load_frame(args.raw, "raw/staged")
    clean_frame = load_frame(args.clean, "cleaned")
    scaled = load_scaled_data(args.scaled, args.feature_names)

    raw_labels: pd.Series | None = None
    raw_plot: pd.DataFrame | None = None
    if raw_frame is not None:
        try:
            raw_labels = resolve_label_series(raw_frame, args.labels)
            raw_plot = sample_frame(raw_frame, raw_labels, args.plot_sample)
        except Exception as exc:
            warn(f"Raw labels could not be resolved: {type(exc).__name__}: {exc}")

        if raw_labels is not None and raw_plot is not None:
            run_figure(
                "Figure 1 class distribution",
                lambda: figure_01_class_distribution(
                    raw_labels,
                    args.outdir,
                    manifest,
                    args.class_log_y,
                ),
            )
            run_figure(
                "Figure 2 missing/infinite summary",
                lambda: figure_02_missing_infinite_summary(
                    raw_frame,
                    FEATURE_NAMES,
                    args.outdir,
                    manifest,
                ),
            )
            run_figure(
                "Figure 3 raw heavy-tail histograms",
                lambda: figure_03_raw_heavy_tail_histograms(
                    raw_frame,
                    raw_plot,
                    args.heavy_features,
                    args.outdir,
                    manifest,
                ),
            )
            run_figure(
                "Figure 4 raw boxplot grid",
                lambda: figure_04_raw_continuous_boxplots(
                    raw_frame,
                    CONTINUOUS_FEATURES,
                    args.outdir,
                    manifest,
                ),
            )
    else:
        warn("Stage 1 figures were skipped because raw/staged data is unavailable.")

    clean_labels: pd.Series | None = None
    clean_plot: pd.DataFrame | None = None
    if clean_frame is not None:
        try:
            clean_labels = resolve_label_series(clean_frame, args.labels)
            clean_plot = sample_frame(clean_frame, clean_labels, args.plot_sample)
        except Exception as exc:
            warn(f"Clean labels could not be resolved: {type(exc).__name__}: {exc}")

        if raw_frame is not None and raw_plot is not None and clean_plot is not None:
            run_figure(
                "Figure 5 before/after cleaning comparison",
                lambda: figure_05_cleaning_before_after_histograms(
                    raw_frame,
                    clean_frame,
                    raw_plot,
                    clean_plot,
                    args.heavy_features,
                    args.outdir,
                    manifest,
                ),
            )
        else:
            warn(
                "Figure 5 requires both raw and cleaned data and was skipped."
            )

        run_figure(
            "Figure 6 cleaned Spearman heatmap",
            lambda: figure_06_clean_spearman_heatmap(
                clean_frame,
                CONTINUOUS_FEATURES,
                args.outdir,
                manifest,
            ),
        )
        if clean_labels is not None:
            run_figure(
                "Figure 7 cleaned class count table",
                lambda: figure_07_clean_class_count_table(
                    clean_labels,
                    _split_label_map(scaled),
                    args.outdir,
                    manifest,
                ),
            )
    else:
        warn("Stage 2 figures were skipped because cleaned data is unavailable.")

    shift_csv = args.outdir / "train_test_distribution_shift.csv"
    if scaled is not None:
        representative_features = (
            args.scaled_features
            if args.scaled_features is not None
            else args.heavy_features
        )
        run_figure(
            "Figure 8 scaled distributions",
            lambda: figure_08_scaled_feature_distributions(
                scaled,
                representative_features,
                args.plot_sample,
                args.near_zero_iqr,
                args.outdir,
                manifest,
            ),
        )
        run_figure(
            "Figure 9 scaled PCA",
            lambda: figure_09_scaled_pca(
                scaled,
                args.plot_sample,
                args.outdir,
                manifest,
            ),
        )
        run_figure(
            "Figure 10 scaled UMAP/t-SNE",
            lambda: figure_10_scaled_embedding(
                scaled,
                args.embedding_sample,
                args.perplexity,
                args.outdir,
                manifest,
            ),
        )
        run_figure(
            "Figure 11 train/test distribution shift",
            lambda: figure_11_train_test_shift(
                scaled,
                args.psi_bins,
                args.outdir,
                manifest,
            ),
        )
    else:
        warn("Stage 3 and cross-stage figures were skipped; scaled data unavailable.")

    print("\nEDA figure generation summary")
    print("-----------------------------")
    print(f"Figures completed: {manifest.count}")
    print(f"Vector files written: {manifest.count * 2}")
    print(f"Output directory: {args.outdir.resolve()}")
    print(f"Manifest: {manifest.path.resolve()}")
    print(
        f"PSI/JSD CSV: {shift_csv.resolve()}"
        if shift_csv.exists()
        else "PSI/JSD CSV: not written (scaled test data unavailable or figure skipped)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
