from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.thesis_eval import realism_table


def _write_sources(tmp_path: Path) -> realism_table.SourcePaths:
    pc1_rows = []
    perfeature_rows = []
    maha_rows = []
    method_to_population_model = {
        "PGD": ("PGD", "shared"),
        "CW": ("CW", "shared"),
        "Latent-PGD": ("latentPGD_untgt", "mlp"),
        "Latent-CW": ("latentCW_untgt", "mlp"),
    }
    for class_index, attack_class in enumerate(realism_table.CLASS_ORDER):
        for method_index, method in enumerate(realism_table.METHOD_ORDER):
            value = float(class_index + method_index + 1)
            pc1_rows.append(
                {
                    "attack_class": attack_class,
                    "method": method,
                    "value": value,
                    "ci_low": value - 0.1,
                    "ci_high": value + 0.1,
                    "status": "ok",
                }
            )
            for feature_index in range(39):
                perfeature_rows.append(
                    {
                        "attack_class": attack_class,
                        "method": method,
                        "feature_index": feature_index,
                        "value": value + feature_index,
                        "status": "ok",
                    }
                )
            population, model = method_to_population_model[method]
            maha_rows.extend(
                {
                    "population": population,
                    "model": model,
                    "source_class": attack_class,
                    "maha": value + offset,
                }
                for offset in (0.0, 2.0)
            )

    pc1_path = tmp_path / "wasserstein_pc1.csv"
    perfeature_path = tmp_path / "wasserstein_perfeature.csv"
    validity_path = tmp_path / "thesis_bundle.multimetric.csv"
    maha_path = tmp_path / "F6_l2_vs_maha.csv"
    diagnostics_path = tmp_path / "F6_covariance_diagnostics.csv"
    bundles_dir = tmp_path / "bundles"
    bundles_dir.mkdir()
    pd.DataFrame(pc1_rows).to_csv(pc1_path, index=False)
    pd.DataFrame(perfeature_rows).to_csv(perfeature_path, index=False)
    pd.DataFrame(
        [
            {
                "model": "MLP",
                "attack_type": attack_type,
                "Protocol_Valid": validity,
            }
            for attack_type, validity in (
                ("input_pgd", 0.1),
                ("input_cw", 0.2),
                ("latent_pgd", 0.9),
                ("latent_cw", 1.0),
            )
        ]
    ).to_csv(validity_path, index=False)
    pd.DataFrame(maha_rows).to_csv(maha_path, index=False)
    pd.DataFrame(
        [
            {
                "source_class": attack_class,
                "covariance_component": "tied",
                "covariance_mode": "tied",
                "covariance_estimator": "ledoit_wolf",
                "empirical_cond_number": 1000.0 + class_index,
                "regularized_cond_number": 100.0 + class_index,
                "shrinkage_alpha": 0.01,
                "n_reference": 40000,
                "latent_dim": 16,
                "score_definition": (
                    "squared_mahalanobis_min_over_centers"
                ),
            }
            for class_index, attack_class in enumerate(
                realism_table.CLASS_ORDER
            )
        ]
    ).to_csv(diagnostics_path, index=False)
    for class_index, attack_class in enumerate(realism_table.CLASS_ORDER):
        for method_index, method in enumerate(realism_table.METHOD_ORDER):
            population, model = method_to_population_model[method]
            valid_count = (class_index + method_index) % 3
            protocol_valid = np.array(
                [True] * valid_count + [False] * (2 - valid_count),
                dtype=bool,
            )
            np.savez_compressed(
                bundles_dir / f"{population}__{model}__{attack_class}.npz",
                protocol_valid=protocol_valid,
            )
    return realism_table.SourcePaths(
        wasserstein_pc1=pc1_path,
        wasserstein_perfeature=perfeature_path,
        validity=validity_path,
        validity_bundles=bundles_dir,
        mahalanobis=maha_path,
        mahalanobis_expected=maha_path,
        mahalanobis_diagnostics=diagnostics_path,
    )


def test_build_table_joins_metrics_and_flags_non_realism(
    tmp_path: Path,
) -> None:
    sources = _write_sources(tmp_path)

    table, todos = realism_table.build_realism_table(sources)

    assert len(table) == 28
    assert todos == []
    dos_pgd = table[
        (table["attack_class"] == "DoS") & (table["method"] == "PGD")
    ].iloc[0]
    assert dos_pgd["wasserstein_pc1"] == 1.0
    assert dos_pgd["wasserstein_mean_per_feature"] == np.mean(
        np.arange(1.0, 40.0)
    )
    assert dos_pgd["validity_percent"] == 0.0
    assert dos_pgd["validity_scope"] == "class_method_bundle"
    ddos_pgd = table[
        (table["attack_class"] == "DDoS") & (table["method"] == "PGD")
    ].iloc[0]
    assert ddos_pgd["validity_percent"] == 50.0
    assert dos_pgd["mahalanobis_mean"] == 2.0
    assert dos_pgd["mahalanobis_covariance_mode"] == "tied"
    assert dos_pgd["mahalanobis_covariance_estimator"] == "ledoit_wolf"
    assert dos_pgd["mahalanobis_empirical_condition"] == 1000.0
    assert dos_pgd["mahalanobis_regularized_condition"] == 100.0
    assert (
        dos_pgd["mahalanobis_score_definition"]
        == "squared_mahalanobis_min_over_centers"
    )
    assert table[table["attack_class"] == "Mirai"][
        "non_realism_flag"
    ].all()
    assert table[table["attack_class"] == "Web"][
        "non_realism_flag"
    ].all()


def test_missing_mahalanobis_emits_todo_cells(tmp_path: Path) -> None:
    sources = _write_sources(tmp_path)
    sources = realism_table.SourcePaths(
        wasserstein_pc1=sources.wasserstein_pc1,
        wasserstein_perfeature=sources.wasserstein_perfeature,
        validity=sources.validity,
        validity_bundles=sources.validity_bundles,
        mahalanobis=None,
        mahalanobis_expected=tmp_path / "expected" / "F6_l2_vs_maha.csv",
        mahalanobis_diagnostics=sources.mahalanobis_diagnostics,
    )

    table, todos = realism_table.build_realism_table(sources)

    assert len(todos) == 28
    assert set(table["mahalanobis_mean"]) == {"# TODO"}
    assert all("expected" in todo for todo in todos)


def test_renderers_follow_markdown_and_latex_contract(
    tmp_path: Path,
) -> None:
    table, _ = realism_table.build_realism_table(_write_sources(tmp_path))

    markdown = realism_table.render_markdown(table)
    latex = realism_table.render_latex(table)

    assert "| Attack class | Method | W (PC1)" in markdown
    assert "Mirai\u2020" in markdown
    assert "Web\u2021" in markdown
    assert "n=100" in markdown
    assert "MMD" in markdown and "JS" in markdown
    assert "squared Mahalanobis" in markdown
    assert "Ledoit-Wolf" in markdown
    assert "\\begin{table}[H]" in latex
    assert "\\makecell{" in latex
    assert "p{" in latex
    assert "\\centering\\arraybackslash" in latex
    assert "\\caption{" in latex
    assert latex.index("\\caption{") > latex.index("\\end{tabular}")
    assert "\\hline" in latex
