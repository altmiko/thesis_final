from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import binomtest, chi2

try:
    from statsmodels.stats.contingency_tables import mcnemar as _statsmodels_mcnemar
except ModuleNotFoundError:
    _statsmodels_mcnemar = None


BOOTSTRAP_SEED = 42
N_BOOTSTRAP = 10_000
ALPHA = 0.05


@dataclass(frozen=True)
class BootstrapCI:
    asr_valid_mean: float
    asr_valid_ci_low: float
    asr_valid_ci_high: float

    def as_dict(self) -> dict[str, float]:
        return {
            "asr_valid_mean": self.asr_valid_mean,
            "asr_valid_ci_low": self.asr_valid_ci_low,
            "asr_valid_ci_high": self.asr_valid_ci_high,
        }


@dataclass(frozen=True)
class McNemarResult:
    statistic: float
    pvalue: float


def _run_mcnemar(table: list[list[int]], *, exact: bool) -> McNemarResult:
    if _statsmodels_mcnemar is not None:
        result = _statsmodels_mcnemar(table, exact=exact, correction=not exact)
        return McNemarResult(statistic=float(result.statistic), pvalue=float(result.pvalue))

    b = int(table[0][1])
    c = int(table[1][0])
    if exact:
        statistic = float(min(b, c))
        pvalue = float(binomtest(k=min(b, c), n=b + c, p=0.5, alternative="two-sided").pvalue)
        return McNemarResult(statistic=statistic, pvalue=pvalue)

    statistic = float(((abs(b - c) - 1.0) ** 2) / (b + c)) if (b + c) > 0 else 0.0
    pvalue = float(1.0 - chi2.cdf(statistic, df=1))
    return McNemarResult(statistic=statistic, pvalue=pvalue)


def bootstrap_asr_valid_ci(
    outcomes: np.ndarray,
    *,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, float]:
    outcomes = np.asarray(outcomes, dtype=np.bool_)
    if outcomes.ndim != 1:
        raise ValueError(f"Expected a 1D boolean array, got shape {outcomes.shape}")
    if outcomes.size == 0:
        raise ValueError("Cannot bootstrap an empty outcomes array")

    rng = np.random.default_rng(seed)
    sample_size = outcomes.size
    bootstrap_means = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        sample_idx = rng.integers(0, sample_size, size=sample_size)
        bootstrap_means[i] = outcomes[sample_idx].mean()

    stats = BootstrapCI(
        asr_valid_mean=float(outcomes.mean()),
        asr_valid_ci_low=float(np.percentile(bootstrap_means, 2.5)),
        asr_valid_ci_high=float(np.percentile(bootstrap_means, 97.5)),
    )
    return stats.as_dict()


def format_ci_percent(mean: float, low: float, high: float) -> str:
    return f"{mean * 100.0:.2f}% [{low * 100.0:.2f}%, {high * 100.0:.2f}%]"


def build_bootstrap_summary_df(
    per_sample_df: pd.DataFrame,
    *,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = BOOTSTRAP_SEED,
) -> pd.DataFrame:
    required_cols = {"model_name", "attack_type", "success", "protocol_valid", "mask_valid"}
    missing = required_cols - set(per_sample_df.columns)
    if missing:
        raise ValueError(f"Missing required columns for bootstrap summary: {sorted(missing)}")

    rows: list[dict[str, Any]] = []
    grouped = per_sample_df.groupby(["model_name", "attack_type"], sort=True)
    for (model_name, attack_type), group in grouped:
        success = group["success"].astype(bool).to_numpy()
        valid_success = (
            group["success"].astype(bool)
            & group["protocol_valid"].astype(bool)
            & group["mask_valid"].astype(bool)
        ).to_numpy()

        asr_stats = bootstrap_asr_valid_ci(success, n_bootstrap=n_bootstrap, seed=seed)
        valid_stats = bootstrap_asr_valid_ci(valid_success, n_bootstrap=n_bootstrap, seed=seed)
        rows.append(
            {
                "model_name": model_name,
                "attack_type": attack_type,
                "asr_mean": asr_stats["asr_valid_mean"],
                "asr_ci_low": asr_stats["asr_valid_ci_low"],
                "asr_ci_high": asr_stats["asr_valid_ci_high"],
                "asr_formatted": format_ci_percent(
                    asr_stats["asr_valid_mean"],
                    asr_stats["asr_valid_ci_low"],
                    asr_stats["asr_valid_ci_high"],
                ),
                "asr_valid_mean": valid_stats["asr_valid_mean"],
                "asr_valid_ci_low": valid_stats["asr_valid_ci_low"],
                "asr_valid_ci_high": valid_stats["asr_valid_ci_high"],
                "asr_valid_formatted": format_ci_percent(
                    valid_stats["asr_valid_mean"],
                    valid_stats["asr_valid_ci_low"],
                    valid_stats["asr_valid_ci_high"],
                ),
                "n": int(len(group)),
            }
        )
    return pd.DataFrame(rows)


def map_true_label_to_category(true_label: Any) -> str:
    label = str(true_label).strip()
    label_upper = label.upper()

    if "MIRAI" in label_upper:
        return "Mirai"
    if "DDOS" in label_upper:
        return "DDoS"
    if label_upper == "DOS" or "DOS" in label_upper:
        return "DoS"
    if "RECON" in label_upper or "SCAN" in label_upper:
        return "Recon"
    if "WEB" in label_upper or "HTTP" in label_upper or "SQL" in label_upper or "XSS" in label_upper:
        return "Web"
    return "Other"


def build_per_category_tables(per_sample_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    required_cols = {
        "sample_id",
        "true_label",
        "predicted_label",
        "protocol_valid",
        "mask_valid",
        "attack_type",
        "model_name",
    }
    missing = required_cols - set(per_sample_df.columns)
    if missing:
        raise ValueError(f"Missing required columns for per-category table: {sorted(missing)}")

    df = per_sample_df.copy()
    df["protocol_valid"] = df["protocol_valid"].astype(bool)
    df["mask_valid"] = df["mask_valid"].astype(bool)
    df["success"] = df["predicted_label"].astype(str) != df["true_label"].astype(str)
    df["category"] = df["true_label"].map(map_true_label_to_category)
    df["asr_valid"] = df["success"] & df["protocol_valid"] & df["mask_valid"]

    grouped = (
        df.groupby(["model_name", "attack_type", "category"], sort=True)
        .agg(
            n=("sample_id", "count"),
            asr_valid_rate=("asr_valid", "mean"),
        )
        .reset_index()
    )
    grouped["display"] = grouped.apply(
        lambda row: f"{row['asr_valid_rate'] * 100.0:.2f}%{'*' if int(row['n']) < 30 else ''}",
        axis=1,
    )
    grouped["count_display"] = grouped.apply(
        lambda row: f"{int(row['n'])}{'*' if int(row['n']) < 30 else ''}",
        axis=1,
    )

    pivot_values = grouped.pivot(
        index=["model_name", "attack_type"],
        columns="category",
        values="display",
    ).reset_index()
    pivot_counts = grouped.pivot(
        index=["model_name", "attack_type"],
        columns="category",
        values="count_display",
    ).reset_index()
    return grouped, pivot_values, pivot_counts


def build_mcnemar_summary_df(per_sample_df: pd.DataFrame) -> pd.DataFrame:
    required_cols = {"model_name", "sample_id", "attack_type", "success"}
    missing = required_cols - set(per_sample_df.columns)
    if missing:
        raise ValueError(f"Missing required columns for McNemar test: {sorted(missing)}")

    rows: list[dict[str, Any]] = []
    latent_df = per_sample_df[per_sample_df["attack_type"].isin(["latent-pgd", "latent-cw"])].copy()
    if latent_df.empty:
        raise ValueError("No latent-pgd / latent-cw rows found in per-sample data")

    for model_name, model_df in latent_df.groupby("model_name", sort=True):
        pgd = (
            model_df[model_df["attack_type"] == "latent-pgd"]
            .sort_values("sample_id")
            .set_index("sample_id")["success"]
            .astype(bool)
        )
        cw = (
            model_df[model_df["attack_type"] == "latent-cw"]
            .sort_values("sample_id")
            .set_index("sample_id")["success"]
            .astype(bool)
        )

        if len(pgd) != len(cw):
            raise ValueError(
                f"McNemar pairing requires equal N for model={model_name}, got {len(pgd)} vs {len(cw)}"
            )
        if not pgd.index.equals(cw.index):
            raise ValueError(f"McNemar pairing requires matching sample_id order for model={model_name}")

        pgd_outcomes = pgd.to_numpy()
        cw_outcomes = cw.to_numpy()
        b = int(np.sum(pgd_outcomes & ~cw_outcomes))
        c = int(np.sum(~pgd_outcomes & cw_outcomes))
        table = [
            [int(np.sum(pgd_outcomes & cw_outcomes)), b],
            [c, int(np.sum(~pgd_outcomes & ~cw_outcomes))],
        ]

        exact = (b + c) < 25
        result = _run_mcnemar(table, exact=exact)
        p_value = float(result.pvalue)
        statistic = float(result.statistic)
        significant = bool(p_value < ALPHA)
        interpretation = (
            f"significant difference detected (p={p_value:.6g})"
            if significant
            else f"latent-CW and latent-PGD are not significantly different (p={p_value:.6g})"
        )
        rows.append(
            {
                "model": model_name,
                "b": b,
                "c": c,
                "statistic": statistic,
                "p_value": p_value,
                "significant": significant,
                "interpretation": interpretation,
            }
        )

    return pd.DataFrame(rows)


def save_dataframe(df: pd.DataFrame, csv_path: Path) -> None:
    df.to_csv(csv_path, index=False)


def save_latex_table(df: pd.DataFrame, tex_path: Path, *, index: bool = False) -> None:
    tex_path.write_text(df.to_latex(index=index, escape=False), encoding="utf-8")


def save_markdown_table(df: pd.DataFrame, md_path: Path, *, index: bool = False, title: str | None = None) -> None:
    lines: list[str] = []
    if title:
        lines.append(f"# {title}")
        lines.append("")
    lines.append(df.to_markdown(index=index))
    lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")
