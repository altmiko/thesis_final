"""
Run post-attack domain-validity analysis in raw feature space.

Outputs:
- results/attacks/shock_table.csv
- results/attacks/violation_breakdown.csv
- figures/asr_raw_vs_valid.png
- tables/impossible_traffic_{model}_{attack}.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.preprocessing.feature_groups import FEATURE_NAMES
from src.evaluation.validity_analysis import compute_asr_valid
from src.evaluation.validity_analysis import generate_impossible_traffic_exhibit
from src.evaluation.validity_analysis import inverse_transform_results
from src.evaluation.validity_analysis import parse_attack_filename
from src.evaluation.validity_analysis import validate_adversarial_examples
from src.evaluation.validity_analysis import validate_feature_matrix


ROOT = Path(__file__).resolve().parents[2]
ATTACKS_DIR = ROOT / "results" / "attacks"
SCALER_PATH = ROOT / "data" / "processed" / "scaler.pkl"
TABLES_DIR = ROOT / "tables"
FIGURES_DIR = ROOT / "figures"


def _fmt_pct(x: float) -> str:
    return f"{x * 100.0:.1f}%"


def _ensure_paths(attacks_dir: Path, scaler_path: Path) -> None:
    needed = [attacks_dir, scaler_path]
    missing = [str(p) for p in needed if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required paths:\n" + "\n".join(missing))
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def _discover_attack_files(attacks_dir: Path) -> List[Path]:
    files = sorted([p for p in attacks_dir.glob("attack_*.npz") if p.is_file()])
    if not files:
        raise FileNotFoundError(f"No attack .npz files found in {attacks_dir}")
    return files


def _sanity_binary_pgd_030(attacks_dir: Path, scaler_path: Path) -> Dict[str, Any]:
    sanity_file = attacks_dir / "attack_binary_pgd_0.30.npz"
    if not sanity_file.exists():
        raise FileNotFoundError(f"Sanity file not found: {sanity_file}")

    res = inverse_transform_results(str(sanity_file), str(scaler_path))

    clean_val = validate_feature_matrix(res["X_clean_raw"], FEATURE_NAMES)
    adv_val = validate_adversarial_examples(res["X_adv_raw"], FEATURE_NAMES)

    clean_rate = clean_val["validity_rate"]
    adv_rate = adv_val["validity_rate"]

    print("=" * 96)
    print("SANITY CHECK (binary + PGD eps=0.30)")
    print("=" * 96)
    print(f"clean validity rate: {_fmt_pct(clean_rate)}")
    print(f"adversarial validity rate: {_fmt_pct(adv_rate)}")

    if clean_rate < 0.995:
        raise RuntimeError(
            "Clean inverse-transformed samples are not ~100% valid. "
            "Likely scaler or feature-order mismatch; stopping for debugging."
        )

    if adv_rate >= 0.999:
        raise RuntimeError(
            "Adversarial validity is ~100%; expected at least some invalid samples. "
            "Validator wiring may be incorrect."
        )

    print("sanity check passed")
    return {
        "clean_validity_rate": clean_rate,
        "adv_validity_rate": adv_rate,
    }


def _plot_asr_raw_vs_valid(shock_df: pd.DataFrame, out_path: Path) -> None:
    subset = shock_df[
        (shock_df["Attack"].str.upper() == "PGD")
        & (np.isclose(shock_df["Eps"].astype(float), 0.30))
        & (shock_df["Model"].isin(["binary", "8class", "34class"]))
    ].copy()

    if subset.empty:
        print("Skipping plot: no PGD eps=0.30 rows found in shock table.")
        return

    order = ["binary", "8class", "34class"]
    subset["Model"] = pd.Categorical(subset["Model"], categories=order, ordered=True)
    subset = subset.sort_values("Model")

    x = np.arange(len(subset))
    width = 0.34

    fig, ax = plt.subplots(figsize=(9, 5.5))
    raw_vals = subset["ASR_raw"].to_numpy(dtype=float) * 100.0
    valid_vals = subset["ASR_valid"].to_numpy(dtype=float) * 100.0

    ax.bar(x - width / 2, raw_vals, width, label="ASR_raw", color="#D1495B")
    ax.bar(x + width / 2, valid_vals, width, label="ASR_valid", color="#2E86AB")

    ax.set_title("PGD eps=0.30: Raw vs Domain-Valid Attack Success")
    ax.set_ylabel("Rate (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(subset["Model"].tolist())
    ax.set_ylim(0, max(raw_vals.max(), valid_vals.max()) * 1.20)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()

    for i, (r, v) in enumerate(zip(raw_vals, valid_vals)):
        ax.text(i - width / 2, r + 0.8, f"{r:.1f}%", ha="center", va="bottom", fontsize=9)
        ax.text(i + width / 2, v + 0.8, f"{v:.1f}%", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _print_table(title: str, df: pd.DataFrame) -> None:
    print("\n" + title)
    if df.empty:
        print("(empty)")
        return
    print(df.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run validity analysis over attack .npz files")
    parser.add_argument("--attacks-dir", type=str, default=str(ATTACKS_DIR))
    parser.add_argument("--scaler-path", type=str, default=str(SCALER_PATH))
    parser.add_argument("--sanity-only", action="store_true")
    args = parser.parse_args()

    attacks_dir = Path(args.attacks_dir)
    scaler_path = Path(args.scaler_path)

    _ensure_paths(attacks_dir=attacks_dir, scaler_path=scaler_path)
    attack_files = _discover_attack_files(attacks_dir)

    _sanity_binary_pgd_030(attacks_dir, scaler_path)
    if args.sanity_only:
        print("Sanity-only mode complete.")
        return

    shock_rows: List[Dict[str, Any]] = []
    rule_totals: Dict[str, int] = {}
    total_adv_samples = 0

    processed: Dict[Tuple[str, str, str], Dict[str, Any]] = {}

    print("\nProcessing attack files...")
    for npz_path in attack_files:
        meta = parse_attack_filename(str(npz_path))
        result = inverse_transform_results(str(npz_path), str(scaler_path))

        clean_validation = validate_feature_matrix(result["X_clean_raw"], FEATURE_NAMES)
        adv_validation = validate_adversarial_examples(result["X_adv_raw"], FEATURE_NAMES)
        asr_valid_metrics = compute_asr_valid(result, adv_validation["per_sample_valid"])

        asr_raw = float(asr_valid_metrics["attack_success_rate_raw"])
        asr_valid = float(asr_valid_metrics["attack_success_rate_valid"])
        validity_rate = float(adv_validation["validity_rate"])
        gap = asr_raw - asr_valid

        print(
            f"{meta['model']} + {meta['attack'].upper()} eps={meta['eps_label']}: "
            f"ASR_raw={_fmt_pct(asr_raw)}, "
            f"Validity={_fmt_pct(validity_rate)}, "
            f"ASR_valid={_fmt_pct(asr_valid)}"
        )

        shock_rows.append(
            {
                "Model": meta["model"],
                "Attack": meta["attack"].upper(),
                "Eps": float(meta["eps"]),
                "ASR_raw": asr_raw,
                "Validity Rate": validity_rate,
                "ASR_valid": asr_valid,
                "Gap (raw-valid)": gap,
                "Clean Validity Rate": float(clean_validation["validity_rate"]),
                "Validity Among Successful": float(
                    asr_valid_metrics["validity_rate_among_successful"]
                ),
                "Samples Originally Correct": int(
                    asr_valid_metrics["samples_originally_correct"]
                ),
                "Successful Raw": int(asr_valid_metrics["samples_successful_raw"]),
                "Successful Valid": int(asr_valid_metrics["samples_successful_valid"]),
            }
        )

        for rule, cnt in adv_validation["per_rule_violation_counts"].items():
            rule_totals[rule] = rule_totals.get(rule, 0) + int(cnt)

        total_adv_samples += int(result["X_adv_raw"].shape[0])

        key = (meta["model"], meta["attack"], f"{meta['eps']:.2f}")
        processed[key] = {
            "meta": meta,
            "result": result,
            "clean_validation": clean_validation,
            "adv_validation": adv_validation,
            "asr_valid_metrics": asr_valid_metrics,
        }

    shock_df = pd.DataFrame(shock_rows)
    shock_df = shock_df.sort_values(by=["Model", "Attack", "Eps"]).reset_index(drop=True)

    shock_out = attacks_dir / "shock_table.csv"
    shock_df.to_csv(shock_out, index=False)

    shock_print = shock_df.copy()
    for col in ["ASR_raw", "Validity Rate", "ASR_valid", "Gap (raw-valid)", "Clean Validity Rate", "Validity Among Successful"]:
        shock_print[col] = shock_print[col].map(_fmt_pct)
    shock_print["Eps"] = shock_print["Eps"].map(lambda x: f"{x:.2f}")
    _print_table("Shock Table", shock_print[["Model", "Attack", "Eps", "ASR_raw", "Validity Rate", "ASR_valid", "Gap (raw-valid)"]])

    exhibit_targets = [
        ("binary", "pgd", "0.30"),
        ("8class", "pgd", "0.30"),
        ("34class", "pgd", "0.30"),
    ]

    for model_name, attack_name, eps_label in exhibit_targets:
        key = (model_name, attack_name, eps_label)
        if key not in processed:
            print(f"Skipping exhibit for {model_name}/{attack_name}/{eps_label}: file not found")
            continue

        pack = processed[key]
        result = pack["result"]
        adv_validation = pack["adv_validation"]

        exhibit_df = generate_impossible_traffic_exhibit(
            X_clean_raw=result["X_clean_raw"],
            X_adv_raw=result["X_adv_raw"],
            feature_names=FEATURE_NAMES,
            violation_details=adv_validation["violation_details"],
            y_pred_clean=result["y_pred_clean"],
            y_pred_adv=result["y_pred_adv"],
            y_true=result["y_true"],
            n_examples=5,
        )

        exhibit_out = TABLES_DIR / f"impossible_traffic_{model_name}_{attack_name}.csv"
        exhibit_df.to_csv(exhibit_out, index=False)

        _print_table(
            f"Impossible Traffic Exhibit: {model_name} + {attack_name.upper()} eps={eps_label}",
            exhibit_df,
        )
        print(f"Saved exhibit: {exhibit_out}")

    violation_rows = []
    for rule, total_cnt in sorted(rule_totals.items(), key=lambda kv: kv[1], reverse=True):
        violation_rows.append(
            {
                "rule_name": rule,
                "total_violations": int(total_cnt),
                "total_adv_samples": int(total_adv_samples),
                "violation_rate": float(total_cnt / total_adv_samples) if total_adv_samples > 0 else 0.0,
            }
        )

    violation_df = pd.DataFrame(violation_rows)
    violation_out = attacks_dir / "violation_breakdown.csv"
    violation_df.to_csv(violation_out, index=False)

    violation_print = violation_df.copy()
    if not violation_print.empty:
        violation_print["violation_rate"] = violation_print["violation_rate"].map(_fmt_pct)
    _print_table("Per-rule Violation Breakdown", violation_print)

    plot_out = FIGURES_DIR / "asr_raw_vs_valid.png"
    _plot_asr_raw_vs_valid(shock_df, plot_out)

    print("\nOutputs saved:")
    print(f"- {shock_out}")
    print(f"- {violation_out}")
    print(f"- {plot_out}")


if __name__ == "__main__":
    main()
