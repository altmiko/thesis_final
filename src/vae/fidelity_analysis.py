"""
Population-level fidelity analysis for adversarial CICIoT2023 samples.

This module is intentionally separate from ``PhysicsValidator`` because the
metrics below are distributional and cannot be evaluated per sample.
"""

from __future__ import annotations


def compute_feature_distribution_drift(*args, **kwargs):
    """TODO: compare clean vs adversarial per-feature distributions.

    Planned metrics:
    - Kolmogorov-Smirnov statistic per feature
    - Wasserstein distance per feature

    This should operate on full attack outputs, not individual rows.
    """
    raise NotImplementedError("Feature-distribution fidelity metrics are a follow-up task.")


def compute_correlation_preservation(*args, **kwargs):
    """TODO: compare clean vs adversarial feature correlation structure.

    Planned metric:
    - Frobenius norm of the difference between correlation matrices

    This should operate on full attack outputs, not individual rows.
    """
    raise NotImplementedError("Correlation-preservation metrics are a follow-up task.")
