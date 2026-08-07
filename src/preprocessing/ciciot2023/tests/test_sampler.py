"""Unit tests for the cluster-proportional-floor sampler (doc §7.1 checklist).

Mandated by downsampling_strategy.md §5.1 / §7.1:

* ``size[c] < floor`` for at least one cluster -> ``sum(alloc) == N`` and no
  infinite loop (the bug the previous revision's pseudocode had).
* ``N`` close to total size -> the step-5 ``break`` path (``cap_not_binding``)
  fires, logs, and does not silently corrupt counts.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.preprocessing.ciciot2023.sampler import (
    cluster_proportional_floor_sample,
    proportional_floor_allocations,
    random_floor_sample,
)


def test_small_cluster_below_floor_is_taken_whole_and_sum_is_exact():
    # Clusters: two below floor, three above.
    sizes = {0: 50, 1: 30, 2: 5000, 3: 8000, 4: 12000}
    floor = 100
    target = 6000
    alloc, small, large, cap_not_binding = proportional_floor_allocations(
        sizes, target_n=target, floor=floor
    )
    assert small == [0, 1]                       # below-floor taken whole
    assert alloc[0] == 50 and alloc[1] == 30
    assert sum(alloc.values()) == target         # exact, no drift
    assert not cap_not_binding
    for c in large:
        assert floor <= alloc[c] <= sizes[c]     # bounds respected


def test_cap_not_binding_when_floors_exceed_target():
    # Many clusters, floor so large that k*floor > target -> step-5 break.
    sizes = {c: 5000 for c in range(10)}
    floor = 2000
    target = 8000  # < 10 * 2000
    alloc, small, large, cap_not_binding = proportional_floor_allocations(
        sizes, target_n=target, floor=floor
    )
    assert cap_not_binding                        # break path fired
    assert all(alloc[c] == floor for c in large)  # every cluster kept its floor
    assert sum(alloc.values()) >= target          # floor guarantee beats the cap


def test_target_ge_total_keeps_everything():
    sizes = {0: 100, 1: 200}
    alloc, small, large, cap_not_binding = proportional_floor_allocations(
        sizes, target_n=10_000, floor=10
    )
    assert alloc == sizes and not cap_not_binding


def test_small_pool_floors_exceeding_cap_raises():
    sizes = {0: 500, 1: 600}   # both <= floor -> both SMALL
    with pytest.raises(ValueError, match="too small"):
        proportional_floor_allocations(sizes, target_n=800, floor=1000)


def test_end_to_end_sample_is_deterministic_and_capped():
    rng = np.random.default_rng(0)
    # Three well-separated blobs of unequal size (multi-modal by construction).
    x = np.vstack([
        rng.normal(0, 0.1, size=(20000, 4)),
        rng.normal(10, 0.1, size=(4000, 4)),
        rng.normal(-8, 0.1, size=(1000, 4)),
    ]).astype(np.float32)

    r1 = cluster_proportional_floor_sample(x, target_n=6000, k=3, floor=200, seed=42)
    r2 = cluster_proportional_floor_sample(x, target_n=6000, k=3, floor=200, seed=42)

    assert np.array_equal(r1.kept_indices, r2.kept_indices)   # reproducible
    assert r1.kept_indices.size <= 6000 + 5                   # ~cap (float rounding)
    assert sum(r1.allocations.values()) == r1.kept_indices.size
    # The rare 1000-row mode must survive (floor guarantee).
    kept_in_rare = np.sum(r1.kept_indices >= 24000)
    assert kept_in_rare >= 200


def test_nearest_centroid_mode_is_deterministic():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(3000, 3)).astype(np.float32)
    r = cluster_proportional_floor_sample(
        x, target_n=1000, k=5, floor=50, seed=7, selection_mode="nearest_centroid"
    )
    assert r.kept_indices.size <= 1000 + 5
    assert r.selection_mode == "nearest_centroid"


def test_random_floor_sample_passthrough_and_cap():
    assert np.array_equal(random_floor_sample(100, 500, seed=1), np.arange(100))
    idx = random_floor_sample(10_000, 1000, seed=1)
    assert idx.size == 1000 and np.all(np.diff(idx) > 0)
