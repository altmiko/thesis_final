"""Distributional fidelity metrics for scaled CICIoT2023 feature vectors."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.distance import jensenshannon, pdist
from scipy.stats import wasserstein_distance
from sklearn.decomposition import PCA


@dataclass(frozen=True)
class BootstrapResult:
    value: float
    ci_low: float
    ci_high: float


def _as_1d(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size < 2:
        raise ValueError(f"{name} must contain at least two observations")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def _as_2d(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] < 2:
        raise ValueError(f"{name} must have shape (N, D) with N >= 2")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def wasserstein_1d(real: np.ndarray, generated: np.ndarray) -> float:
    real_1d = _as_1d(real, "real")
    generated_1d = _as_1d(generated, "generated")
    return float(wasserstein_distance(real_1d, generated_1d))


def wasserstein_per_feature(
    real: np.ndarray, generated: np.ndarray
) -> np.ndarray:
    real_2d = _as_2d(real, "real")
    generated_2d = _as_2d(generated, "generated")
    if real_2d.shape[1] != generated_2d.shape[1]:
        raise ValueError("real and generated must have the same feature count")
    return np.asarray(
        [
            wasserstein_distance(real_2d[:, index], generated_2d[:, index])
            for index in range(real_2d.shape[1])
        ],
        dtype=np.float64,
    )


def fit_clean_pc1(clean: np.ndarray) -> PCA:
    clean_2d = _as_2d(clean, "clean")
    model = PCA(n_components=1, svd_solver="full")
    model.fit(clean_2d)
    return model


def bootstrap_wasserstein_1d(
    real: np.ndarray,
    generated: np.ndarray,
    *,
    n_boot: int,
    seed: int,
    max_samples: int = 2000,
) -> BootstrapResult:
    if n_boot < 1:
        raise ValueError("n_boot must be >= 1")
    if max_samples < 2:
        raise ValueError("max_samples must be >= 2")
    real_1d = _as_1d(real, "real")
    generated_1d = _as_1d(generated, "generated")
    point = wasserstein_1d(real_1d, generated_1d)
    real_n = min(real_1d.size, max_samples)
    generated_n = min(generated_1d.size, max_samples)
    rng = np.random.default_rng(seed)
    estimates = np.empty(n_boot, dtype=np.float64)
    for index in range(n_boot):
        real_sample = real_1d[
            rng.integers(0, real_1d.size, size=real_n)
        ]
        generated_sample = generated_1d[
            rng.integers(0, generated_1d.size, size=generated_n)
        ]
        estimates[index] = wasserstein_distance(
            real_sample, generated_sample
        )
    ci_low, ci_high = np.percentile(estimates, [2.5, 97.5])
    return BootstrapResult(point, float(ci_low), float(ci_high))


def median_heuristic_bandwidth(
    real: np.ndarray, generated: np.ndarray
) -> float:
    real_2d = _as_2d(real, "real")
    generated_2d = _as_2d(generated, "generated")
    if real_2d.shape[1] != generated_2d.shape[1]:
        raise ValueError("real and generated must have the same feature count")
    pooled = np.vstack([real_2d, generated_2d])
    distances = pdist(pooled, metric="euclidean")
    positive = distances[np.isfinite(distances) & (distances > 0.0)]
    if positive.size == 0:
        return 1.0
    return float(np.median(positive))


def _rbf_kernel(
    left: np.ndarray, right: np.ndarray, bandwidth: float
) -> np.ndarray:
    if not np.isfinite(bandwidth) or bandwidth <= 0.0:
        raise ValueError("bandwidth must be finite and > 0")
    left_sq = np.sum(left * left, axis=1)[:, None]
    right_sq = np.sum(right * right, axis=1)[None, :]
    squared_distance = np.maximum(
        left_sq + right_sq - 2.0 * left @ right.T, 0.0
    )
    return np.exp(-squared_distance / (2.0 * bandwidth * bandwidth))


def _mmd2_from_kernels(
    kernel_xx: np.ndarray,
    kernel_yy: np.ndarray,
    kernel_xy: np.ndarray,
) -> float:
    m = kernel_xx.shape[0]
    n = kernel_yy.shape[0]
    xx = (kernel_xx.sum() - np.trace(kernel_xx)) / (m * (m - 1))
    yy = (kernel_yy.sum() - np.trace(kernel_yy)) / (n * (n - 1))
    xy = kernel_xy.mean()
    return float(xx + yy - 2.0 * xy)


def unbiased_mmd2(
    real: np.ndarray, generated: np.ndarray, bandwidth: float
) -> float:
    real_2d = _as_2d(real, "real")
    generated_2d = _as_2d(generated, "generated")
    if real_2d.shape[1] != generated_2d.shape[1]:
        raise ValueError("real and generated must have the same feature count")
    return _mmd2_from_kernels(
        _rbf_kernel(real_2d, real_2d, bandwidth),
        _rbf_kernel(generated_2d, generated_2d, bandwidth),
        _rbf_kernel(real_2d, generated_2d, bandwidth),
    )


def bootstrap_mmd2(
    real: np.ndarray,
    generated: np.ndarray,
    *,
    bandwidth: float,
    n_boot: int,
    seed: int,
) -> BootstrapResult:
    if n_boot < 1:
        raise ValueError("n_boot must be >= 1")
    real_2d = _as_2d(real, "real")
    generated_2d = _as_2d(generated, "generated")
    if real_2d.shape[1] != generated_2d.shape[1]:
        raise ValueError("real and generated must have the same feature count")
    kernel_xx = _rbf_kernel(real_2d, real_2d, bandwidth)
    kernel_yy = _rbf_kernel(generated_2d, generated_2d, bandwidth)
    kernel_xy = _rbf_kernel(real_2d, generated_2d, bandwidth)
    point = _mmd2_from_kernels(kernel_xx, kernel_yy, kernel_xy)
    rng = np.random.default_rng(seed)
    estimates = np.empty(n_boot, dtype=np.float64)
    for index in range(n_boot):
        real_idx = rng.integers(0, real_2d.shape[0], size=real_2d.shape[0])
        generated_idx = rng.integers(
            0, generated_2d.shape[0], size=generated_2d.shape[0]
        )
        estimates[index] = _mmd2_from_kernels(
            kernel_xx[np.ix_(real_idx, real_idx)],
            kernel_yy[np.ix_(generated_idx, generated_idx)],
            kernel_xy[np.ix_(real_idx, generated_idx)],
        )
    ci_low, ci_high = np.percentile(estimates, [2.5, 97.5])
    return BootstrapResult(point, float(ci_low), float(ci_high))


def js_divergence_1d(
    real: np.ndarray, generated: np.ndarray, *, bins: int
) -> float:
    if bins < 2:
        raise ValueError("bins must be >= 2")
    real_1d = _as_1d(real, "real")
    generated_1d = _as_1d(generated, "generated")
    lower = min(float(real_1d.min()), float(generated_1d.min()))
    upper = max(float(real_1d.max()), float(generated_1d.max()))
    if lower == upper:
        return 0.0
    edges = np.linspace(lower, upper, bins + 1)
    real_hist, _ = np.histogram(real_1d, bins=edges)
    generated_hist, _ = np.histogram(generated_1d, bins=edges)
    epsilon = np.finfo(np.float64).eps
    real_prob = real_hist.astype(np.float64) + epsilon
    generated_prob = generated_hist.astype(np.float64) + epsilon
    real_prob /= real_prob.sum()
    generated_prob /= generated_prob.sum()
    return float(jensenshannon(real_prob, generated_prob, base=2.0) ** 2)
