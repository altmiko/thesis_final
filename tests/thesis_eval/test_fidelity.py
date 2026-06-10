from __future__ import annotations

from pathlib import Path

import numpy as np

from src.thesis_eval.metrics import fidelity
from src.thesis_eval import fidelity_tables


def test_wasserstein_per_feature_returns_expected_shift() -> None:
    real = np.array([[0.0, 1.0], [1.0, 3.0], [2.0, 5.0]])
    generated = real + np.array([2.0, -1.0])

    distances = fidelity.wasserstein_per_feature(real, generated)

    np.testing.assert_allclose(distances, np.array([2.0, 1.0]))


def test_pc1_is_fit_on_clean_population_only() -> None:
    clean = np.array(
        [
            [-2.0, 0.0],
            [-1.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
        ]
    )
    generated = clean + np.array([3.0, 100.0])

    projection = fidelity.fit_clean_pc1(clean)
    clean_pc1 = projection.transform(clean).ravel()
    generated_pc1 = projection.transform(generated).ravel()

    assert abs(projection.components_[0, 0]) == 1.0
    assert abs(projection.components_[0, 1]) == 0.0
    assert fidelity.wasserstein_1d(clean_pc1, generated_pc1) == 3.0


def test_unbiased_mmd2_detects_shift_with_positive_bandwidth() -> None:
    rng = np.random.default_rng(7)
    real = rng.normal(size=(80, 4))
    same_distribution = rng.normal(size=(80, 4))
    shifted = rng.normal(loc=2.0, size=(80, 4))

    bandwidth = fidelity.median_heuristic_bandwidth(real, shifted)
    same_mmd2 = fidelity.unbiased_mmd2(real, same_distribution, bandwidth)
    shifted_mmd2 = fidelity.unbiased_mmd2(real, shifted, bandwidth)

    assert bandwidth > 0.0
    assert shifted_mmd2 > same_mmd2


def test_bootstrap_intervals_are_seeded_and_ordered() -> None:
    rng = np.random.default_rng(11)
    real = rng.normal(size=60)
    generated = rng.normal(loc=0.5, size=60)

    first = fidelity.bootstrap_wasserstein_1d(
        real, generated, n_boot=100, seed=42
    )
    second = fidelity.bootstrap_wasserstein_1d(
        real, generated, n_boot=100, seed=42
    )

    assert first == second
    assert first.value >= 0.0
    assert first.ci_low <= first.ci_high


def test_mmd_bootstrap_is_seeded_and_uses_fixed_bandwidth() -> None:
    rng = np.random.default_rng(19)
    real = rng.normal(size=(50, 3))
    generated = rng.normal(loc=0.4, size=(50, 3))
    bandwidth = fidelity.median_heuristic_bandwidth(real, generated)

    first = fidelity.bootstrap_mmd2(
        real,
        generated,
        bandwidth=bandwidth,
        n_boot=80,
        seed=42,
    )
    second = fidelity.bootstrap_mmd2(
        real,
        generated,
        bandwidth=bandwidth,
        n_boot=80,
        seed=42,
    )

    assert first == second
    assert first.ci_low <= first.ci_high


def test_population_discovery_marks_cinput_without_vectors_as_unavailable(
    tmp_path: Path,
) -> None:
    run_dir = (
        tmp_path
        / "outputs"
        / "latent_attacks"
        / "constrained_input_baselines_20260602_032651_seed42"
    )
    run_dir.mkdir(parents=True)
    (run_dir / "per_sample_results.csv").write_text(
        "sample_id,attack_type\n"
        "1,cinput-pgd\n"
        "2,cinput-cw\n",
        encoding="utf-8",
    )

    status = fidelity_tables.inspect_cinput_vectors(tmp_path)

    assert status.available is False
    assert status.path == run_dir
    assert "no full 39-feature adversarial vectors" in status.reason


def test_population_discovery_fails_if_a_cinput_method_is_missing(
    tmp_path: Path,
) -> None:
    run_dir = (
        tmp_path
        / "outputs"
        / "latent_attacks"
        / "constrained_input_baselines_20260602_032651_seed42"
    )
    run_dir.mkdir(parents=True)
    (run_dir / "per_sample_results.csv").write_text(
        "sample_id,attack_type\n1,cinput-pgd\n", encoding="utf-8"
    )

    with np.testing.assert_raises_regex(
        RuntimeError, "missing required populations.*cinput-cw"
    ):
        fidelity_tables.inspect_cinput_vectors(tmp_path)


def test_unavailable_metric_row_uses_na_and_todo_marker() -> None:
    row = fidelity_tables.unavailable_metric_row(
        attack_class="DoS",
        method="cInput CPGD",
        metric="mmd2_fullspace",
        reason="full vectors missing",
    )

    assert row["value"] == "N/A"
    assert row["ci_low"] == "N/A"
    assert row["status"] == "# TODO: full vectors missing"


def test_compute_class_metrics_emits_all_methods_and_feature_rows() -> None:
    rng = np.random.default_rng(23)
    clean = rng.normal(size=(60, 3))
    pgd = rng.normal(loc=0.2, size=(45, 3))
    populations = {
        "PGD": pgd,
        "CW": None,
        "Latent-PGD": None,
        "Latent-CW": None,
        "cInput CPGD": None,
        "cInput CCW": None,
    }
    source_paths = {method: f"/tmp/{method}" for method in populations}

    frames = fidelity_tables.compute_class_metrics(
        attack_class="DoS",
        clean=clean,
        populations=populations,
        source_paths=source_paths,
        unavailable_reasons={
            method: "vectors missing"
            for method, values in populations.items()
            if values is None
        },
        feature_names=["a", "b", "c"],
        mmd_n=20,
        n_boot=20,
        bootstrap_n=30,
        js_bins=10,
        seed=42,
    )

    assert len(frames["wasserstein_pc1"]) == 6
    assert len(frames["wasserstein_perfeature"]) == 18
    assert len(frames["mmd_fullspace"]) == 6
    assert len(frames["js_pc1_appendix"]) == 6
    pgd_pc1 = frames["wasserstein_pc1"].query("method == 'PGD'").iloc[0]
    cinput_pc1 = frames["wasserstein_pc1"].query(
        "method == 'cInput CPGD'"
    ).iloc[0]
    assert pgd_pc1["status"] == "ok"
    assert cinput_pc1["value"] == "N/A"


def test_render_markdown_contains_required_tables_and_reading_note() -> None:
    rows = []
    for attack_class in fidelity_tables.CLASS_ORDER:
        for method in fidelity_tables.METHOD_ORDER:
            rows.append(
                {
                    "attack_class": attack_class,
                    "method": method,
                    "value": 0.1,
                    "ci_low": 0.05,
                    "ci_high": 0.15,
                    "status": "ok",
                    "bandwidth": 1.2,
                }
            )
    frames = {
        "wasserstein_pc1": fidelity_tables.dataframe(rows),
        "wasserstein_perfeature": fidelity_tables.dataframe(
            [
                {
                    "attack_class": row["attack_class"],
                    "method": row["method"],
                    "feature_index": 0,
                    "feature": "a",
                    "value": row["value"],
                    "status": row["status"],
                }
                for row in rows
            ]
        ),
        "mmd_fullspace": fidelity_tables.dataframe(rows),
        "js_pc1_appendix": fidelity_tables.dataframe(rows),
    }

    markdown = fidelity_tables.render_markdown(
        frames,
        manifest={
            "seed": 42,
            "n_boot": 1000,
            "mmd_subsample_n": 100,
            "js_bins": 40,
            "scaling": "RobustScaler-transformed model space",
            "artifact_hash": "abc123",
            "library_versions": {"numpy": "2.0.1"},
        },
    )

    assert "Table 1 — Wasserstein (PC1)" in markdown
    assert "Table 2 — Wasserstein (mean per-feature, 39 features)" in markdown
    assert "Table 3 — MMD² (full feature space)" in markdown
    assert "Appendix Table A1 — JS divergence (PC1)" in markdown
    assert "lower values mean closer to clean malicious traffic" in markdown
    assert "0% protocol-validity" in markdown
    assert "Observed sigma range: 1.200000 to 1.200000" in markdown
