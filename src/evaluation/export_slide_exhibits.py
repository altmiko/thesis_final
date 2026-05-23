"""
Create compact, slide-ready versions of impossible traffic exhibits.

Input files are expected from run_validity_analysis.py:
- tables/impossible_traffic_binary_pgd.csv
- tables/impossible_traffic_8class_pgd.csv
- tables/impossible_traffic_34class_pgd.csv

Outputs:
- tables/impossible_traffic_binary_pgd_slide.csv
- tables/impossible_traffic_8class_pgd_slide.csv
- tables/impossible_traffic_34class_pgd_slide.csv
- tables/impossible_traffic_slide_pack.csv
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLES_DIR = ROOT / "tables"

TARGETS = [
    ("binary", "pgd"),
    ("8class", "pgd"),
    ("34class", "pgd"),
]


def _pretty_feature_name(name: str) -> str:
    text = name.replace("_", " ")
    return text


def _to_float(value: object) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _compact_exhibit(df: pd.DataFrame, model: str, attack: str, top_k: int = 15) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=[
                "Case",
                "Example",
                "Clean pred",
                "Adv pred",
                "Feature",
                "Clean value",
                "Adversarial value",
                "Delta",
                "Why impossible",
            ]
        )

    work = df.copy()
    work["delta_abs"] = work["delta"].map(_to_float).abs()
    work["has_reason"] = work["violation_description"].fillna("") != ""

    # Keep the strongest rows for non-ML audiences: explicit rule violations first,
    # then larger absolute feature shifts.
    work = work.sort_values(
        by=["violates_rule", "has_reason", "delta_abs", "example_rank"],
        ascending=[False, False, False, True],
    )

    # Prefer rule-violating rows for the compact table.
    violating = work[work["violates_rule"] == True].copy()
    if len(violating) < top_k:
        fill = work[~work.index.isin(violating.index)]
        pick = pd.concat([violating, fill], axis=0).head(top_k)
    else:
        pick = violating.head(top_k)

    pick = pick.copy()
    pick["Feature"] = pick["feature"].map(_pretty_feature_name)
    pick["Clean value"] = pick["clean_value"].map(lambda x: f"{float(x):.4f}")
    pick["Adversarial value"] = pick["adv_value"].map(lambda x: f"{float(x):.4f}")
    pick["Delta"] = pick["delta"].map(lambda x: f"{float(x):.4f}")
    pick["Why impossible"] = pick["violation_description"].fillna("")

    pick["Case"] = f"{model} + {attack.upper()} eps=0.30"
    pick["Example"] = pick["example_rank"].map(lambda x: f"Ex{int(x)}")
    pick["Clean pred"] = pick["clean_prediction"].astype(int)
    pick["Adv pred"] = pick["adv_prediction"].astype(int)

    out = pick[
        [
            "Case",
            "Example",
            "Clean pred",
            "Adv pred",
            "Feature",
            "Clean value",
            "Adversarial value",
            "Delta",
            "Why impossible",
        ]
    ].reset_index(drop=True)

    return out


def main() -> None:
    outputs: List[pd.DataFrame] = []

    for model, attack in TARGETS:
        source = TABLES_DIR / f"impossible_traffic_{model}_{attack}.csv"
        if not source.exists():
            raise FileNotFoundError(
                f"Missing source exhibit: {source}. Run run_validity_analysis.py first."
            )

        src_df = pd.read_csv(source)
        slide_df = _compact_exhibit(src_df, model=model, attack=attack, top_k=15)

        out_path = TABLES_DIR / f"impossible_traffic_{model}_{attack}_slide.csv"
        slide_df.to_csv(out_path, index=False)
        outputs.append(slide_df)

        print(f"Saved: {out_path}")
        print(slide_df.to_string(index=False))
        print()

    pack_df = pd.concat(outputs, axis=0, ignore_index=True)
    pack_out = TABLES_DIR / "impossible_traffic_slide_pack.csv"
    pack_df.to_csv(pack_out, index=False)
    print(f"Saved: {pack_out}")


if __name__ == "__main__":
    main()
