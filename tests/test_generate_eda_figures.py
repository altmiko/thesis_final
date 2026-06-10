from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import generate_eda_figures as eda


def test_resolve_label_series_maps_ciciot_labels_to_eight_classes() -> None:
    frame = pd.DataFrame(
        {
            "Label": [
                "BENIGN",
                "DDOS-ICMP_FLOOD",
                "DICTIONARYBRUTEFORCE",
            ]
        }
    )

    labels = eda.resolve_label_series(frame, "label_8class")

    assert labels.tolist() == ["Benign", "DDoS", "BruteForce"]


def test_stratified_indices_are_deterministic_and_preserve_classes() -> None:
    labels = np.array(["A"] * 70 + ["B"] * 20 + ["C"] * 10)

    first = eda.stratified_indices(labels, sample_size=20, seed=42)
    second = eda.stratified_indices(labels, sample_size=20, seed=42)

    assert np.array_equal(first, second)
    assert len(first) == 20
    assert set(labels[first]) == {"A", "B", "C"}


def test_compute_psi_jsd_is_zero_for_identical_distributions() -> None:
    values = np.linspace(-2.0, 2.0, 1000)

    psi, jsd = eda.compute_psi_jsd(values, values.copy(), bins=10)

    assert psi == 0.0
    assert jsd == 0.0


def test_compute_psi_jsd_detects_distribution_shift() -> None:
    train = np.linspace(-1.0, 1.0, 1000)
    test = np.linspace(1.0, 3.0, 1000)

    psi, jsd = eda.compute_psi_jsd(train, test, bins=10)

    assert psi > 0.1
    assert jsd > 0.01


def test_compute_psi_jsd_detects_shift_from_constant_train_feature() -> None:
    train = np.zeros(1000)
    test = np.concatenate([np.zeros(500), np.ones(500)])

    psi, jsd = eda.compute_psi_jsd(train, test, bins=10)

    assert psi > 0.1
    assert jsd > 0.01


def test_format_axis_value_preserves_small_values_and_compacts_large_values() -> None:
    assert eda.format_axis_value(0.1) == "0.1"
    assert eda.format_axis_value(1_000) == "1,000"
    assert eda.format_axis_value(10_000_000_000) == "1e10"


def test_compact_symlog_ticks_stay_within_observed_maximum() -> None:
    ticks = eda.compact_symlog_ticks(11_500)

    assert ticks[0] == 0.0
    assert max(ticks) <= 11_500
    assert len(ticks) <= 5


def test_save_fig_writes_both_vector_formats_and_manifest(tmp_path: Path) -> None:
    manifest = eda.FigureManifest(tmp_path)
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])

    eda.save_fig(
        fig,
        "stage1_example",
        tmp_path,
        manifest,
        figure_number=1,
        caption="Example caption.",
    )

    assert (tmp_path / "stage1_example.pdf").is_file()
    assert (tmp_path / "stage1_example.svg").is_file()
    text = (tmp_path / "figure_manifest.txt").read_text(encoding="utf-8")
    assert "Figure 01" in text
    assert "stage1_example.[pdf|svg]" in text
    assert "Example caption." in text
