from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = str(_REPO_ROOT / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from attack.statistical_analysis import (  # noqa: E402
    build_bootstrap_summary_df,
    build_mcnemar_summary_df,
    build_per_category_tables,
    save_dataframe,
    save_markdown_table,
    save_latex_table,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute bootstrap CIs, per-category ASR_Valid, and McNemar tests.")
    parser.add_argument("--per-sample-csv", required=True, help="Path to per-sample attack results CSV.")
    parser.add_argument("--output-dir", required=True, help="Directory to write analysis artifacts.")
    args = parser.parse_args()

    per_sample_csv = Path(args.per_sample_csv)
    if not per_sample_csv.exists():
        raise FileNotFoundError(
            f"Per-sample CSV not found at {per_sample_csv}. "
            "The current rerun artifacts only include aggregate tables; this analysis requires per-sample outputs."
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(per_sample_csv)

    bootstrap_df = build_bootstrap_summary_df(df)
    save_dataframe(bootstrap_df, output_dir / "bootstrap_summary.csv")
    save_latex_table(bootstrap_df, output_dir / "bootstrap_summary.tex")
    save_markdown_table(bootstrap_df, output_dir / "bootstrap_summary.md", title="Bootstrap Summary")

    grouped_df, pivot_df, counts_df = build_per_category_tables(df)
    save_dataframe(grouped_df, output_dir / "per_category_long.csv")
    save_dataframe(pivot_df, output_dir / "per_category_pivot.csv")
    save_dataframe(counts_df, output_dir / "per_category_counts.csv")
    save_latex_table(pivot_df, output_dir / "per_category_pivot.tex")
    save_markdown_table(pivot_df, output_dir / "per_category_pivot.md", title="Per-Category ASR Valid Pivot")
    save_latex_table(counts_df, output_dir / "per_category_counts.tex")
    save_markdown_table(counts_df, output_dir / "per_category_counts.md", title="Per-Category Sample Counts")

    mcnemar_df = build_mcnemar_summary_df(df)
    save_dataframe(mcnemar_df, output_dir / "mcnemar_latent_pgd_vs_cw.csv")
    save_latex_table(mcnemar_df, output_dir / "mcnemar_latent_pgd_vs_cw.tex")
    save_markdown_table(mcnemar_df, output_dir / "mcnemar_latent_pgd_vs_cw.md", title="McNemar Latent PGD vs Latent CW")

    print(f"Bootstrap summary saved to {output_dir / 'bootstrap_summary.csv'}")
    print(f"Per-category pivot saved to {output_dir / 'per_category_pivot.csv'}")
    print(f"Per-category LaTeX saved to {output_dir / 'per_category_pivot.tex'}")
    print(f"McNemar summary saved to {output_dir / 'mcnemar_latent_pgd_vs_cw.csv'}")
    for row in mcnemar_df.itertuples(index=False):
        print(f"{row.model}: {row.interpretation}")


if __name__ == "__main__":
    main()
