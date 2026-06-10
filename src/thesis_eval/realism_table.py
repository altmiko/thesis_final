"""Join persisted realism metrics into the thesis main-text table."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


CLASS_ORDER = [
    "DoS",
    "DDoS",
    "Mirai",
    "BruteForce",
    "Recon",
    "Web",
    "Spoofing",
]
METHOD_ORDER = ["PGD", "CW", "Latent-PGD", "Latent-CW"]
MAHALANOBIS_MAP = {
    "PGD": ("PGD", "shared"),
    "CW": ("CW", "shared"),
    "Latent-PGD": ("latentPGD_untgt", "mlp"),
    "Latent-CW": ("latentCW_untgt", "mlp"),
}
NON_REALISM = {
    "Mirai": (
        "dagger",
        "Collapsed-latent-dimension pathology; large distance is not "
        "healthy novelty.",
    ),
    "Web": (
        "double_dagger",
        "Out-of-distribution generation; large distance is not healthy "
        "novelty.",
    ),
}


@dataclass(frozen=True)
class SourcePaths:
    wasserstein_pc1: Path
    wasserstein_perfeature: Path
    validity: Path
    validity_bundles: Path
    mahalanobis: Path | None
    mahalanobis_expected: Path
    mahalanobis_diagnostics: Path


def _require_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required {label}: {path}")
    return path


def _require_dir(path: Path, label: str) -> Path:
    if not path.is_dir():
        raise FileNotFoundError(f"Missing required {label}: {path}")
    return path


def discover_sources(repo_root: Path) -> SourcePaths:
    root = Path(repo_root).resolve()
    worktree_results = (
        root
        / ".worktrees"
        / "thesis-eval-suite"
        / "results"
        / "thesis_eval"
    )
    expected_maha = (
        root / "results" / "thesis_eval" / "data" / "F6_l2_vs_maha.csv"
    )
    candidates = [
        expected_maha,
        worktree_results / "data" / "F6_l2_vs_maha.csv",
    ]
    mahalanobis = next((path for path in candidates if path.is_file()), None)
    bundle_candidates = [
        root / "results" / "thesis_eval" / "bundles",
        worktree_results / "bundles",
    ]
    validity_bundles = next(
        (path for path in bundle_candidates if path.is_dir()), None
    )
    if validity_bundles is None:
        raise FileNotFoundError(
            "Missing required class-resolved validity bundles; checked: "
            + ", ".join(str(path) for path in bundle_candidates)
        )
    diagnostics = (
        mahalanobis.parent / "F6_covariance_diagnostics.csv"
        if mahalanobis is not None
        else expected_maha.parent / "F6_covariance_diagnostics.csv"
    )
    return SourcePaths(
        wasserstein_pc1=_require_file(
            root / "wasserstein_pc1.csv", "Wasserstein-PC1 source"
        ),
        wasserstein_perfeature=_require_file(
            root / "wasserstein_perfeature.csv",
            "per-feature Wasserstein source",
        ),
        validity=_require_file(
            root
            / "results"
            / "attacks"
            / "thesis_bundle.multimetric.csv",
            "validity source",
        ),
        validity_bundles=_require_dir(
            validity_bundles, "class-resolved validity bundles"
        ),
        mahalanobis=mahalanobis,
        mahalanobis_expected=expected_maha,
        mahalanobis_diagnostics=_require_file(
            diagnostics, "Mahalanobis covariance diagnostics"
        ),
    )


def source_mapping_lines(sources: SourcePaths) -> list[str]:
    maha_path = (
        str(sources.mahalanobis)
        if sources.mahalanobis is not None
        else f"# TODO; expected {sources.mahalanobis_expected}"
    )
    return [
        (
            "W (PC1) <- "
            f"{sources.wasserstein_pc1} :: "
            "attack_class + method -> value, ci_low, ci_high"
        ),
        (
            "W (mean per-feature) <- "
            f"{sources.wasserstein_perfeature} :: "
            "mean(value) over 39 features by attack_class + method"
        ),
        (
            "Validity % <- "
            f"{sources.validity_bundles} :: mapped population/model/class "
            "bundle -> mean(protocol_valid) * 100; "
            f"{sources.validity} retained as aggregate audit source"
        ),
        (
            "Mahalanobis (mean) <- "
            f"{maha_path} :: tied covariance; mean(maha) by "
            "source_class + mapped population/model; estimator metadata <- "
            f"{sources.mahalanobis_diagnostics}"
        ),
    ]


def _load_pc1(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, keep_default_na=False)
    required = {"attack_class", "method", "value", "ci_low", "ci_high", "status"}
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"{path} missing columns: {sorted(missing)}")
    selected = frame[
        frame["attack_class"].isin(CLASS_ORDER)
        & frame["method"].isin(METHOD_ORDER)
    ].copy()
    if len(selected) != len(CLASS_ORDER) * len(METHOD_ORDER):
        raise RuntimeError(
            f"{path} must contain exactly 28 requested class-method rows"
        )
    if not selected["status"].eq("ok").all():
        bad = selected.loc[selected["status"] != "ok", ["attack_class", "method"]]
        raise RuntimeError(
            f"{path} contains unavailable requested rows: "
            f"{bad.to_dict(orient='records')}"
        )
    for column in ("value", "ci_low", "ci_high"):
        selected[column] = pd.to_numeric(selected[column], errors="raise")
    return selected


def _load_perfeature_means(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, keep_default_na=False)
    required = {
        "attack_class",
        "method",
        "feature_index",
        "value",
        "status",
    }
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"{path} missing columns: {sorted(missing)}")
    selected = frame[
        frame["attack_class"].isin(CLASS_ORDER)
        & frame["method"].isin(METHOD_ORDER)
    ].copy()
    if not selected["status"].eq("ok").all():
        raise RuntimeError(f"{path} has unavailable requested feature rows")
    counts = selected.groupby(["attack_class", "method"]).size()
    if (
        len(counts) != len(CLASS_ORDER) * len(METHOD_ORDER)
        or not counts.eq(39).all()
    ):
        raise RuntimeError(
            f"{path} must contain exactly 39 rows per class-method"
        )
    selected["value"] = pd.to_numeric(selected["value"], errors="raise")
    return (
        selected.groupby(["attack_class", "method"], as_index=False)["value"]
        .mean()
        .rename(columns={"value": "wasserstein_mean_per_feature"})
    )


def _load_validity(
    aggregate_path: Path,
    bundles_dir: Path,
) -> dict[tuple[str, str], tuple[float, int]]:
    frame = pd.read_csv(aggregate_path, keep_default_na=False)
    required = {"model", "attack_type", "Protocol_Valid"}
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(
            f"{aggregate_path} missing columns: {sorted(missing)}"
        )

    values: dict[tuple[str, str], tuple[float, int]] = {}
    for attack_class in CLASS_ORDER:
        for method in METHOD_ORDER:
            population, model = MAHALANOBIS_MAP[method]
            bundle_path = (
                bundles_dir
                / f"{population}__{model}__{attack_class}.npz"
            )
            if not bundle_path.is_file():
                raise FileNotFoundError(
                    "Missing class-resolved validity bundle: "
                    f"{bundle_path}"
                )
            with np.load(bundle_path, allow_pickle=False) as bundle:
                if "protocol_valid" not in bundle.files:
                    raise RuntimeError(
                        f"{bundle_path} has no protocol_valid vector"
                    )
                protocol_valid = np.asarray(
                    bundle["protocol_valid"], dtype=bool
                )
            if protocol_valid.ndim != 1 or protocol_valid.size == 0:
                raise RuntimeError(
                    f"{bundle_path} protocol_valid must be a non-empty vector"
                )
            values[(attack_class, method)] = (
                float(protocol_valid.mean() * 100.0),
                int(protocol_valid.size),
            )
    return values


def _load_mahalanobis_diagnostics(
    path: Path,
) -> dict[str, dict[str, Any]]:
    frame = pd.read_csv(path, keep_default_na=False)
    required = {
        "source_class",
        "covariance_component",
        "covariance_mode",
        "covariance_estimator",
        "empirical_cond_number",
        "regularized_cond_number",
        "shrinkage_alpha",
        "n_reference",
        "latent_dim",
        "score_definition",
    }
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"{path} missing columns: {sorted(missing)}")
    tied = frame[
        frame["source_class"].isin(CLASS_ORDER)
        & (frame["covariance_mode"] == "tied")
        & (frame["covariance_component"] == "tied")
    ].copy()
    if len(tied) != len(CLASS_ORDER):
        raise RuntimeError(
            f"{path} must contain one tied diagnostic row per attack class"
        )
    if tied["source_class"].duplicated().any():
        raise RuntimeError(f"{path} has duplicate tied class diagnostics")
    for column in (
        "empirical_cond_number",
        "regularized_cond_number",
        "shrinkage_alpha",
        "n_reference",
        "latent_dim",
    ):
        tied[column] = pd.to_numeric(tied[column], errors="raise")
    return tied.set_index("source_class").to_dict(orient="index")


def _load_mahalanobis(
    path: Path | None, expected_path: Path
) -> tuple[dict[tuple[str, str], tuple[float, int]], list[str]]:
    if path is None:
        todos = [
            (
                f"# TODO {attack_class}/{method}: Mahalanobis source absent; "
                f"expected {expected_path}"
            )
            for attack_class in CLASS_ORDER
            for method in METHOD_ORDER
        ]
        return {}, todos
    frame = pd.read_csv(path, keep_default_na=False)
    required = {"population", "model", "source_class", "maha"}
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"{path} missing columns: {sorted(missing)}")
    frame["maha"] = pd.to_numeric(frame["maha"], errors="raise")
    values: dict[tuple[str, str], tuple[float, int]] = {}
    todos: list[str] = []
    for attack_class in CLASS_ORDER:
        for method in METHOD_ORDER:
            population, model = MAHALANOBIS_MAP[method]
            selected = frame[
                (frame["source_class"] == attack_class)
                & (frame["population"] == population)
                & (frame["model"] == model)
            ]["maha"]
            if selected.empty:
                todos.append(
                    f"# TODO {attack_class}/{method}: no persisted "
                    f"Mahalanobis rows in {path}"
                )
                continue
            values[(attack_class, method)] = (
                float(selected.mean()),
                int(selected.size),
            )
    return values, todos


def build_realism_table(
    sources: SourcePaths,
) -> tuple[pd.DataFrame, list[str]]:
    pc1 = _load_pc1(sources.wasserstein_pc1)
    perfeature = _load_perfeature_means(sources.wasserstein_perfeature)
    validity = _load_validity(
        sources.validity,
        sources.validity_bundles,
    )
    mahalanobis, todos = _load_mahalanobis(
        sources.mahalanobis, sources.mahalanobis_expected
    )
    diagnostics = _load_mahalanobis_diagnostics(
        sources.mahalanobis_diagnostics
    )
    pc1_lookup = pc1.set_index(["attack_class", "method"])
    perfeature_lookup = perfeature.set_index(["attack_class", "method"])

    rows: list[dict[str, Any]] = []
    for attack_class in CLASS_ORDER:
        marker, reason = NON_REALISM.get(attack_class, ("", ""))
        diagnostic = diagnostics[attack_class]
        for method in METHOD_ORDER:
            pc1_row = pc1_lookup.loc[(attack_class, method)]
            maha_value = mahalanobis.get((attack_class, method))
            validity_value, validity_n = validity[(attack_class, method)]
            population, model = MAHALANOBIS_MAP[method]
            rows.append(
                {
                    "attack_class": attack_class,
                    "method": method,
                    "wasserstein_pc1": float(pc1_row["value"]),
                    "wasserstein_pc1_ci_low": float(pc1_row["ci_low"]),
                    "wasserstein_pc1_ci_high": float(pc1_row["ci_high"]),
                    "wasserstein_mean_per_feature": float(
                        perfeature_lookup.loc[
                            (attack_class, method),
                            "wasserstein_mean_per_feature",
                        ]
                    ),
                    "validity_percent": validity_value,
                    "validity_n": validity_n,
                    "validity_scope": "class_method_bundle",
                    "validity_model": model,
                    "validity_population": population,
                    "mahalanobis_mean": (
                        maha_value[0] if maha_value is not None else "# TODO"
                    ),
                    "mahalanobis_covariance_mode": diagnostic[
                        "covariance_mode"
                    ],
                    "mahalanobis_covariance_estimator": diagnostic[
                        "covariance_estimator"
                    ],
                    "mahalanobis_score_definition": diagnostic[
                        "score_definition"
                    ],
                    "mahalanobis_empirical_condition": diagnostic[
                        "empirical_cond_number"
                    ],
                    "mahalanobis_regularized_condition": diagnostic[
                        "regularized_cond_number"
                    ],
                    "mahalanobis_shrinkage_alpha": diagnostic[
                        "shrinkage_alpha"
                    ],
                    "mahalanobis_reference_n": int(
                        diagnostic["n_reference"]
                    ),
                    "mahalanobis_latent_dim": int(
                        diagnostic["latent_dim"]
                    ),
                    "mahalanobis_n": (
                        maha_value[1] if maha_value is not None else "# TODO"
                    ),
                    "non_realism_flag": bool(marker),
                    "non_realism_marker": marker,
                    "non_realism_reason": reason,
                    "wasserstein_pc1_source": str(sources.wasserstein_pc1),
                    "wasserstein_perfeature_source": str(
                        sources.wasserstein_perfeature
                    ),
                    "validity_source": str(sources.validity),
                    "validity_bundle_source": str(
                        sources.validity_bundles
                        / f"{population}__{model}__{attack_class}.npz"
                    ),
                    "mahalanobis_source": (
                        str(sources.mahalanobis)
                        if sources.mahalanobis is not None
                        else f"# TODO: expected {sources.mahalanobis_expected}"
                    ),
                    "mahalanobis_diagnostics_source": str(
                        sources.mahalanobis_diagnostics
                    ),
                }
            )
    return pd.DataFrame(rows), todos


def _display_class(attack_class: str, marker: str, *, latex: bool) -> str:
    if not latex:
        if marker == "dagger":
            return f"{attack_class}\u2020"
        if marker == "double_dagger":
            return f"{attack_class}\u2021"
        return attack_class
    if marker == "dagger":
        return (
            f"{attack_class}\\textsuperscript{{\\dagger}}"
            if latex
            else f"{attack_class}†"
        )
    if marker == "double_dagger":
        return (
            f"{attack_class}\\textsuperscript{{\\ddagger}}"
            if latex
            else f"{attack_class}‡"
        )
    return attack_class


def _format_float(value: Any, digits: int = 3) -> str:
    if isinstance(value, str):
        return value
    return f"{float(value):.{digits}f}"


def render_markdown(table: pd.DataFrame) -> str:
    lines = [
        "# Unified Realism Table",
        "",
        (
            "Class-grouped method sub-rows are used for readability. This table "
            "is the source of truth for chapter prose and on-figure annotations; "
            "downstream consumers should read `realism_table.csv` rather than "
            "recompute its metrics."
        ),
        "",
        (
            "| Attack class | Method | W (PC1), 95% CI | "
            "W (mean per-feature) | Validity % | Mahalanobis^2 (mean) |"
        ),
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for attack_class in CLASS_ORDER:
        subset = table[table["attack_class"] == attack_class]
        for position, row in enumerate(subset.itertuples(index=False)):
            class_cell = (
                _display_class(
                    row.attack_class, row.non_realism_marker, latex=False
                )
                if position == 0
                else ""
            )
            pc1 = (
                f"{row.wasserstein_pc1:.3f} "
                f"({row.wasserstein_pc1_ci_low:.3f}-"
                f"{row.wasserstein_pc1_ci_high:.3f})"
            )
            lines.append(
                f"| {class_cell} | {row.method} | {pc1} | "
                f"{row.wasserstein_mean_per_feature:.3f} | "
                f"{row.validity_percent:.2f} | "
                f"{_format_float(row.mahalanobis_mean)} |"
            )
    lines.extend(
        [
            "",
            "**Footnotes**",
            "",
            (
                "1. **W (PC1)** reports the persisted point estimate with its "
                "bootstrap 95% CI. **W (mean per-feature)** is the arithmetic "
                "mean of the 39 persisted feature distances; the complete "
                "vector remains in `wasserstein_perfeature.csv`."
            ),
            (
                "2. **Validity %** is class-resolved: each cell is the mean of "
                "the persisted `protocol_valid` vector in the exact bundle "
                "used for that method and class. Input attacks use the shared "
                "bundle and latent attacks use the MLP bundle. The older "
                "method-level multimetric CSV is retained only as an aggregate "
                "audit source."
            ),
            (
                "3. **Mahalanobis^2 (mean)** is the mean persisted squared "
                "Mahalanobis score, minimized over the clean-malicious and "
                "clean-Benign centers. A single tied covariance is fitted per "
                "source class and shared by all four methods. It now uses "
                "always-on Ledoit-Wolf shrinkage; empirical and regularized "
                "condition numbers are recorded in `realism_table.csv` and "
                "`F6_covariance_diagnostics.csv`. The `shared`/`mlp` labels "
                "describe bundle provenance, not different covariance models."
            ),
            (
                "4. † **Mirai non-realism case:** collapsed-latent-dimension "
                "pathology. ‡ **Web non-realism case:** out-of-distribution "
                "generation, including unusually large latent mean-per-feature "
                "Wasserstein distances. These rows must not be read as healthy "
                "novelty."
            ),
            (
                "5. A large latent Wasserstein distance is healthy novelty only "
                "when validity is high **and** squared Mahalanobis remains "
                "on-manifold. "
                "For Mirai/Web, large distance instead indicates collapse/OOD."
            ),
            (
                "6. All class-specific latent populations used for Wasserstein "
                "and Mahalanobis are n=100. Bootstrap CIs are therefore wide; "
                "cross-class differences within overlapping CIs should not be "
                "over-interpreted."
            ),
            (
                "7. MMD and JS are appendix-only: MMD uses n=100 and is noisy, "
                "so it is not suitable for ranking; JS is bin-sensitive. See "
                "`Wasserstein_MMD_results.md`."
            ),
            "",
            (
                "**Interpretation guard:** shrinkage substantially reduces the "
                "largest Latent-PGD scores, especially Recon, but does not "
                "remove the separation from Latent-CW. The evidence therefore "
                "supports an artifact-amplified, not artifact-only, "
                "Latent-PGD off-manifold effect."
            ),
            "",
        ]
    )
    for index, line in enumerate(lines):
        if line.startswith("4. "):
            lines[index] = (
                "4. \u2020 **Mirai non-realism case:** collapsed-latent-"
                "dimension pathology. \u2021 **Web non-realism case:** "
                "out-of-distribution generation, including unusually large "
                "latent mean-per-feature Wasserstein distances. These rows "
                "must not be read as healthy novelty."
            )
    return "\n".join(lines)


def _latex_metric(value: Any) -> str:
    if isinstance(value, str):
        return "\\# TODO"
    return f"{float(value):.3f}"


def render_latex(table: pd.DataFrame) -> str:
    lines = [
        "\\begin{table}[H]",
        "\\centering",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\renewcommand{\\arraystretch}{1.12}",
        (
            "\\begin{tabular}{|"
            ">{\\centering\\arraybackslash}p{1.35cm}|"
            ">{\\centering\\arraybackslash}p{1.35cm}|"
            ">{\\centering\\arraybackslash}p{2.25cm}|"
            ">{\\centering\\arraybackslash}p{1.75cm}|"
            ">{\\centering\\arraybackslash}p{1.25cm}|"
            ">{\\centering\\arraybackslash}p{1.75cm}|}"
        ),
        "\\hline",
        (
            "\\makecell{Attack\\\\class} & "
            "\\makecell{Method} & "
            "\\makecell{W (PC1)\\\\estimate (95\\% CI)} & "
            "\\makecell{W (mean\\\\per-feature)} & "
            "\\makecell{Validity\\\\(\\%)} & "
            "\\makecell{Mean squared\\\\Mahalanobis (LW)} \\\\"
        ),
        "\\hline",
    ]
    for attack_class in CLASS_ORDER:
        subset = table[table["attack_class"] == attack_class]
        for position, row in enumerate(subset.itertuples(index=False)):
            class_cell = (
                _display_class(
                    row.attack_class, row.non_realism_marker, latex=True
                )
                if position == 0
                else ""
            )
            pc1 = (
                f"{row.wasserstein_pc1:.3f} "
                f"({row.wasserstein_pc1_ci_low:.3f}--"
                f"{row.wasserstein_pc1_ci_high:.3f})"
            )
            lines.append(
                f"{class_cell} & {row.method} & {pc1} & "
                f"{row.wasserstein_mean_per_feature:.3f} & "
                f"{row.validity_percent:.2f} & "
                f"{_latex_metric(row.mahalanobis_mean)} \\\\"
            )
            lines.append("\\hline")
    lines.extend(
        [
            "\\end{tabular}",
            "\\vspace{2pt}",
            "\\begin{minipage}{0.99\\linewidth}",
            "\\scriptsize",
            (
                "\\textit{Notes.} W (PC1) gives the persisted estimate and "
                "bootstrap 95\\% CI; mean-per-feature W averages the 39 "
                "persisted distances. Validity is class-resolved from each "
                "exact shared input or MLP latent bundle's persisted "
                "\\texttt{protocol\\_valid} vector. Mean squared Mahalanobis "
                "uses one clean-reference tied covariance per source class, "
                "shared across all methods and regularized with Ledoit--Wolf "
                "shrinkage. Shared/MLP labels identify bundle provenance, not "
                "different covariance references. "
            ),
            "",
            (
                "\\textsuperscript{\\dagger}Mirai is a non-realism case caused "
                "by collapsed latent dimensions. "
                "\\textsuperscript{\\ddagger}Web is a non-realism case showing "
                "out-of-distribution generation. A large latent W is healthy "
                "novelty only when validity is high and Mahalanobis remains "
                "on-manifold; Mirai/Web instead indicate collapse/OOD. "
            ),
            "",
            (
                "All class-specific latent populations used for Wasserstein and "
                "Mahalanobis are $n=100$, so bootstrap CIs are wide and "
                "overlapping cross-class intervals should not be "
                "over-interpreted. MMD ($n=100$, noisy, not for ranking) and JS "
                "(bin-sensitive) are appendix-only; see "
                "\\texttt{Wasserstein\\_MMD\\_results.md}. Shrinkage reduces "
                "the largest Latent-PGD scores but does not remove their "
                "separation from Latent-CW."
            ),
            "\\end{minipage}",
            (
                "\\caption{Unified realism comparison of distributional "
                "distance, protocol validity, and stabilized latent squared "
                "Mahalanobis score.}"
            ),
            "\\label{tab:unified-realism}",
            "\\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(
    output_dir: Path, table: pd.DataFrame
) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "realism_table.csv"
    md_path = output_dir / "realism_table.md"
    tex_path = output_dir / "realism_table.tex"
    table.to_csv(csv_path, index=False)
    md_path.write_text(render_markdown(table), encoding="utf-8")
    tex_path.write_text(render_latex(table), encoding="utf-8")
    return md_path, tex_path, csv_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Join persisted realism metrics without recomputation."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    parser.add_argument("--output-dir", type=Path, default=Path.cwd())
    args = parser.parse_args()
    sources = discover_sources(args.repo_root)
    print("Cell-source mapping:")
    for line in source_mapping_lines(sources):
        print(f"  {line}")
    table, todos = build_realism_table(sources)
    print("TODO cells:")
    if todos:
        for todo in todos:
            print(f"  {todo}")
    else:
        print("  none")
    paths = write_outputs(args.output_dir, table)
    for path in paths:
        print(f"Wrote {path.resolve()}")


if __name__ == "__main__":
    main()
