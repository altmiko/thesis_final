"""Intra-class clustering-based undersampling — selection variant (doc §5).

*"intra-class clustering-based undersampling (selection variant) with
cluster-proportional-with-floor allocation."*

Per majority class we cluster its (train-only, scaled, continuous) rows with
MiniBatchKMeans and keep a budget ``N`` of **real rows**, allocated to clusters
proportional to their size but with a per-cluster ``floor`` so no dense mode is
statistically erased (the P2 panel's objection to random undersampling, doc §2).

The allocation implements the corrected SMALL/LARGE pool split of doc §5.1:
clusters no larger than the floor are taken **whole** (they cannot satisfy the
floor otherwise) and removed from the proportional pool; the previous revision's
``max(floor, raw)`` with a ``≤ size`` clamp was unsatisfiable when
``size[c] < floor`` and would spin or silently break the cap.

Rules that are non-negotiable (doc §5.3): keep real rows only (never synthetic
centroids — those break the domain validator and corrupt the VAE manifold),
cluster on continuous features only, cluster on the train split only.

The module is dataset-agnostic: the continuous-feature matrix, cap, k, and floor
are all passed in, so the two ablation datasets reuse it unchanged (doc §10).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from sklearn.cluster import MiniBatchKMeans

logger = logging.getLogger(__name__)

SelectionMode = Literal["random_within", "nearest_centroid"]


@dataclass
class ClusterSampleResult:
    """Outcome of one class's cluster-selection, recorded in the run manifest."""

    kept_indices: np.ndarray                 # indices into the input rows
    cluster_sizes: dict[int, int]            # cluster -> #rows in that cluster
    allocations: dict[int, int]              # cluster -> #rows kept
    selection_mode: SelectionMode
    small_pool: list[int] = field(default_factory=list)   # clusters taken whole
    large_pool: list[int] = field(default_factory=list)
    cap_not_binding: bool = False            # True if step-5 break fired (doc §5.1)
    target_n: int = 0


def proportional_floor_allocations(
    cluster_sizes: dict[int, int],
    target_n: int,
    floor: int,
) -> tuple[dict[int, int], list[int], list[int], bool]:
    """Allocate ``target_n`` across clusters, proportional-to-size with a floor.

    Returns ``(alloc, small_pool, large_pool, cap_not_binding)``.

    Implements doc §5.1 steps 2–5:

    * **SMALL** = clusters with ``size <= floor`` -> taken whole.
    * **LARGE** = clusters with ``size >  floor``  -> share the remaining budget
      proportional to size, each clamped to ``[floor, size]``.
    * ``cap_not_binding`` is ``True`` when the floors alone meet or exceed the
      budget so the cap does not actually reduce the class (step-5 ``break``);
      the floor guarantee wins over the cap and the caller is warned.
    """
    if target_n <= 0:
        raise ValueError("target_n must be positive")
    if floor < 0:
        raise ValueError("floor must be non-negative")

    total = sum(cluster_sizes.values())
    if target_n >= total:  # nothing to reduce; keep everything
        return dict(cluster_sizes), [], list(cluster_sizes), False

    small = {c: s for c, s in cluster_sizes.items() if s <= floor}
    large = {c: s for c, s in cluster_sizes.items() if s > floor}

    alloc: dict[int, int] = {c: s for c, s in small.items()}  # take SMALL whole
    remaining = target_n - sum(small.values())
    if remaining < 0:
        raise ValueError(
            f"target_n={target_n} too small: SMALL-pool floors "
            f"(sum={sum(small.values())}) already exceed the cap"
        )

    if not large:
        # Everything landed in the SMALL pool; cannot reduce further.
        return alloc, sorted(small), [], True

    lower = {c: floor for c in large}
    upper = dict(large)  # = size[c]
    cap_not_binding = False

    if sum(lower.values()) >= remaining:
        # Floors alone meet/exceed the budget -> give every LARGE cluster its
        # floor and accept sum > remaining (doc §5.1 step-5 break path).
        for c in large:
            alloc[c] = floor
        cap_not_binding = True
        return alloc, sorted(small), sorted(large), cap_not_binding

    # Feasible: distribute (remaining - sum floor) proportional to size, bounded
    # by each cluster's headroom (size - floor). Bounded largest-remainder.
    extra_budget = remaining - sum(lower.values())
    headroom = {c: upper[c] - floor for c in large}
    weights = {c: float(upper[c]) for c in large}

    extra = _bounded_proportional(extra_budget, weights, headroom)
    for c in large:
        alloc[c] = lower[c] + extra[c]

    assert sum(alloc.values()) == target_n, (
        f"allocation drift: {sum(alloc.values())} != {target_n}"
    )
    return alloc, sorted(small), sorted(large), cap_not_binding


def _bounded_proportional(
    budget: int,
    weights: dict[int, float],
    headroom: dict[int, int],
) -> dict[int, int]:
    """Distribute integer ``budget`` across keys proportional to ``weights``,
    never exceeding each key's ``headroom``. Water-filling + largest-remainder.

    Precondition: ``0 <= budget <= sum(headroom)`` (guaranteed by caller).
    """
    keys = list(weights)
    give = {c: 0 for c in keys}
    remaining = budget
    active = [c for c in keys if headroom[c] > 0]

    # Water-fill in float, saturating capped clusters, until stable.
    while remaining > 0 and active:
        wsum = sum(weights[c] for c in active)
        if wsum <= 0:
            break
        target = {c: give[c] + remaining * weights[c] / wsum for c in active}
        newly_saturated = []
        for c in active:
            cap = headroom[c]
            want = target[c]
            if want >= cap:
                if give[c] < cap:
                    give[c] = cap
                newly_saturated.append(c)
        if newly_saturated:
            active = [c for c in active if c not in newly_saturated]
            remaining = budget - sum(give.values())
            continue
        # No new saturation this pass -> settle fractionally, then break to the
        # integer largest-remainder step below.
        for c in active:
            give[c] = int(np.floor(target[c]))
        break

    # Integer largest-remainder to place the residual exactly, respecting caps.
    placed = sum(give.values())
    residual = budget - placed
    if residual > 0:
        # Rank unsaturated keys by fractional shortfall proxy (weight), give 1 each.
        candidates = sorted(
            (c for c in keys if give[c] < headroom[c]),
            key=lambda c: weights[c],
            reverse=True,
        )
        i = 0
        while residual > 0 and candidates:
            c = candidates[i % len(candidates)]
            if give[c] < headroom[c]:
                give[c] += 1
                residual -= 1
            else:
                candidates.remove(c)
                if not candidates:
                    break
                continue
            i += 1
    return give


def cluster_proportional_floor_sample(
    x_continuous: np.ndarray,
    target_n: int,
    k: int,
    floor: int,
    seed: int,
    selection_mode: SelectionMode = "random_within",
    batch_size: int = 8192,
    precomputed_labels: np.ndarray | None = None,
    precomputed_centers: np.ndarray | None = None,
) -> ClusterSampleResult:
    """Select ``≈target_n`` real rows from one class via cluster-proportional-floor.

    ``x_continuous`` must already be scaled and contain only continuous features
    (doc §5.3). Returned ``kept_indices`` refer to rows of ``x_continuous``.

    A row is kept **iff** (1) KMeans assigned it to cluster ``c`` and (2) its
    position in cluster ``c``'s seeded shuffle is ``< alloc[c]`` (doc §5.2). The
    within-cluster pick is a seeded uniform shuffle — safe because a cluster is
    by construction a region of near-identical rows.
    """
    x = np.asarray(x_continuous)
    if x.ndim != 2:
        raise ValueError("x_continuous must be 2D")
    if selection_mode not in ("random_within", "nearest_centroid"):
        raise ValueError(f"unknown selection_mode: {selection_mode}")

    n_rows = x.shape[0]
    if n_rows == 0:
        return ClusterSampleResult(
            np.array([], dtype=np.int64), {}, {}, selection_mode, target_n=target_n
        )
    if n_rows <= target_n:  # nothing to reduce
        return ClusterSampleResult(
            np.arange(n_rows, dtype=np.int64),
            {0: n_rows},
            {0: n_rows},
            selection_mode,
            target_n=target_n,
        )

    n_clusters = max(1, min(int(k), n_rows))
    if precomputed_labels is not None:
        labels = np.asarray(precomputed_labels)
        centers = precomputed_centers
    else:
        model = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=seed,
            batch_size=min(batch_size, n_rows),
            n_init="auto",
        )
        labels = model.fit_predict(x)
        centers = model.cluster_centers_

    unique, counts = np.unique(labels, return_counts=True)
    cluster_sizes = {int(c): int(n) for c, n in zip(unique, counts)}

    alloc, small_pool, large_pool, cap_not_binding = proportional_floor_allocations(
        cluster_sizes, target_n=target_n, floor=floor
    )
    if cap_not_binding:
        logger.warning(
            "cap not binding: floors meet/exceed target_n=%d (kept=%d of %d). "
            "Lower N or leave this class uncapped (doc §5.1 step-5 break).",
            target_n,
            sum(alloc.values()),
            n_rows,
        )

    rng = np.random.default_rng(seed)
    kept: list[np.ndarray] = []
    for c in sorted(alloc):
        take = alloc[c]
        if take <= 0:
            continue
        idx = np.flatnonzero(labels == c)
        if take >= idx.size:
            kept.append(idx)
        elif selection_mode == "random_within":
            kept.append(rng.permutation(idx)[:take])
        else:  # nearest_centroid (doc §5.2 optional deterministic variant)
            if centers is None:
                raise ValueError("nearest_centroid requires cluster centers")
            d = np.sum((x[idx] - centers[c]) ** 2, axis=1)
            kept.append(idx[np.argsort(d, kind="stable")[:take]])

    kept_indices = np.concatenate(kept).astype(np.int64)
    kept_indices.sort()
    return ClusterSampleResult(
        kept_indices=kept_indices,
        cluster_sizes=cluster_sizes,
        allocations={int(c): int(v) for c, v in alloc.items()},
        selection_mode=selection_mode,
        small_pool=small_pool,
        large_pool=large_pool,
        cap_not_binding=cap_not_binding,
        target_n=target_n,
    )


def random_floor_sample(
    n_rows: int, target_n: int, seed: int
) -> np.ndarray:
    """Plain seeded uniform undersample — honest fallback for a class the
    multi-modality diagnostic (doc §6.2) found to be unimodal, where
    proportional-with-floor ≈ random and clustering would be theatre."""
    if n_rows <= target_n:
        return np.arange(n_rows, dtype=np.int64)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n_rows)[:target_n].astype(np.int64)
    idx.sort()
    return idx
