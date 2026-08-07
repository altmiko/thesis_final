"""Leakage-critical split protocol for CICIoT2023 (downsampling_strategy.md §4).

The 309 CSV shards are **not** independent capture sessions. CIC's pipeline
(``mergecap`` → ``tcpdump -C <size>`` → ``DPKT``) merges each attack's PCAPs
into one continuous stream, splits that stream *by byte budget*, then extracts
flow features. Therefore:

* ``source_csv_filename`` indexes a **contiguous temporal segment** of one
  capture, not an independent run (doc §1).
* the numeric suffix (``...Flood2.pcap.csv``) is a ``tcpdump`` sequence number,
  so **natural-numeric shard order == wall-clock order**.

We split forward in time — the latest segments go to test — so no training flow
post-dates a test flow (no data snooping, cf. Arp et al. P3). This module
implements the two primitives and the per-class dispatch of doc §4.2/§4.3.

It knows nothing about features or sampling: given, for one class, the list of
its shards (and their row counts, in file order), it returns which rows land in
train / val / test. All logic is deterministic and order-preserving; validity
depends on the parquet build being a pure concatenator (doc §4.3).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np


# ── Natural-numeric ordering ─────────────────────────────────────────────────
def natkey(path: str) -> list:
    """Natural-numeric sort key: ``'Flood2'`` sorts before ``'Flood10'``.

    Splitting a string on digit runs and casting the digit runs to ``int``
    yields a mixed list that sorts numerically on the numeric fields and
    lexically on the rest — recovering ``tcpdump`` sequence (== wall-clock)
    order, which plain lexicographic order would interleave.

    The trailing alphabetic file extension (``.pcap.csv``) is stripped first so
    that the suffix-less **base** shard — ``tcpdump`` sequence 0, the earliest
    segment (e.g. ``DDoS-ICMP_Flood.pcap.csv``) — sorts *before* ``...Flood1``.
    Without stripping, the glued extension makes the base's first token longer
    than the numbered stem and it would wrongly sort after sequence 1. (This is
    a corrected, strictly-more-faithful form of the key sketched in doc §4.2.)
    """
    stem = re.sub(r"(\.[A-Za-z]+)+$", "", path)
    return [int(tok) if tok.isdigit() else tok for tok in re.split(r"(\d+)", stem)]


# ── Shard-level forward-chaining split (≥3 shards) ───────────────────────────
def forward_chain_shards(
    shards: list[str],
    val_frac: float = 0.10,
    test_frac: float = 0.20,
) -> tuple[list[str], list[str], list[str]]:
    """Assign whole shards of ONE class to (train, val, test) forward in time.

    ``shards`` is the list of ``source_csv_filename`` for the class. The latest
    ``test_frac`` of shards (natural-numeric order) become test, the next
    ``val_frac`` become val, the earliest remainder is train. Requires ≥3
    shards so all three splits are non-empty.
    """
    if not 0 < test_frac < 1 or not 0 < val_frac < 1 or val_frac + test_frac >= 1:
        raise ValueError("val_frac/test_frac must be in (0,1) and sum < 1")
    s = sorted(shards, key=natkey)
    n = len(s)
    n_test = max(1, round(n * test_frac))
    n_val = max(1, round(n * val_frac))
    if n - n_test - n_val < 1:
        raise ValueError(
            f"too few shards ({n}) for a 3-way forward-chaining split; "
            "use the block/hybrid fallback"
        )
    train = s[: n - n_val - n_test]
    val = s[n - n_val - n_test : n - n_test]
    test = s[n - n_test :]
    return train, val, test


# ── Contiguous-block split within a single shard (1 shard) ───────────────────
def block_split_single_shard(
    n_rows: int,
    val_frac: float = 0.10,
    test_frac: float = 0.20,
) -> tuple[slice, slice, slice]:
    """Contiguous, order-preserving temporal cut of one shard's rows.

    Returns (train, val, test) as index ``slice`` objects over ``[0, n_rows)``.
    This performs, at row granularity, the same operation ``tcpdump`` already
    performed at folder granularity (doc §4.3) — a proxy for temporal
    separation, weaker than a true shard boundary. Only valid because row
    order was preserved by the parquet build.
    """
    if n_rows < 3:
        raise ValueError(f"need ≥3 rows for a 3-way block split, got {n_rows}")
    n_test = int(n_rows * test_frac)
    n_val = int(n_rows * val_frac)
    n_test = max(1, n_test)
    n_val = max(1, n_val)
    n_train = n_rows - n_val - n_test
    if n_train < 1:
        raise ValueError(f"block split leaves no train rows for n_rows={n_rows}")
    return (
        slice(0, n_train),
        slice(n_train, n_train + n_val),
        slice(n_train + n_val, n_rows),
    )


def block_split_two_way(n_rows: int, val_frac_of_block: float) -> tuple[slice, slice]:
    """Contiguous 2-way cut of one shard into (train, val); used by the
    two-shard hybrid where the later shard is already the whole test split."""
    if n_rows < 2:
        raise ValueError(f"need ≥2 rows for a 2-way block split, got {n_rows}")
    n_val = max(1, int(n_rows * val_frac_of_block))
    n_train = n_rows - n_val
    if n_train < 1:
        raise ValueError(f"two-way block split leaves no train rows for n_rows={n_rows}")
    return slice(0, n_train), slice(n_train, n_rows)


# ── Per-class dispatch ───────────────────────────────────────────────────────
SplitName = str  # "train" | "val" | "test"
Protocol = str   # "forward_chain" | "two_shard_hybrid" | "block"


@dataclass
class ClassSplitPlan:
    """How one class's rows were partitioned, for the run manifest."""

    label: str
    protocol: Protocol
    n_shards: int
    train_shards: list[str] = field(default_factory=list)
    val_shards: list[str] = field(default_factory=list)
    test_shards: list[str] = field(default_factory=list)
    # For block/hybrid classes: (shard, split, start, stop) row ranges.
    block_ranges: list[tuple[str, SplitName, int, int]] = field(default_factory=list)
    n_train: int = 0
    n_val: int = 0
    n_test: int = 0


def plan_class_split(
    label: str,
    shard_order: list[str],
    shard_rows: dict[str, int],
    val_frac: float = 0.10,
    test_frac: float = 0.20,
) -> ClassSplitPlan:
    """Decide the split protocol for one class and return a concrete plan.

    Parameters
    ----------
    label:        34-class label of this class.
    shard_order:  the class's shards in natural-numeric (wall-clock) order.
    shard_rows:   rows per shard.

    Dispatch (doc §4.3 rule table)::

        ≥3 shards -> forward-chaining shard split
        2 shards  -> later shard = test; earlier shard block-split into train/val
        1 shard   -> contiguous block split into train/val/test
    """
    s = sorted(shard_order, key=natkey)
    n = len(s)
    plan = ClassSplitPlan(label=label, protocol="", n_shards=n)

    if n >= 3:
        train, val, test = forward_chain_shards(s, val_frac, test_frac)
        plan.protocol = "forward_chain"
        plan.train_shards, plan.val_shards, plan.test_shards = train, val, test
        plan.n_train = sum(shard_rows[x] for x in train)
        plan.n_val = sum(shard_rows[x] for x in val)
        plan.n_test = sum(shard_rows[x] for x in test)

    elif n == 2:
        earlier, later = s[0], s[1]
        plan.protocol = "two_shard_hybrid"
        plan.test_shards = [later]
        plan.n_test = shard_rows[later]
        # Split the earlier shard into train/val, keeping val_frac of the whole
        # class's non-test rows. The earlier shard is the entire train+val pool.
        n_earlier = shard_rows[earlier]
        # val_frac is defined over the whole class; renormalise onto the earlier
        # shard (which holds the train+val portion).
        non_test_frac = 1.0 - test_frac
        val_frac_of_block = min(0.9, val_frac / non_test_frac) if non_test_frac > 0 else 0.1
        tr_slice, va_slice = block_split_two_way(n_earlier, val_frac_of_block)
        plan.block_ranges = [
            (earlier, "train", tr_slice.start, tr_slice.stop),
            (earlier, "val", va_slice.start, va_slice.stop),
        ]
        plan.train_shards = [earlier]
        plan.val_shards = [earlier]
        plan.n_train = tr_slice.stop - tr_slice.start
        plan.n_val = va_slice.stop - va_slice.start

    elif n == 1:
        shard = s[0]
        plan.protocol = "block"
        tr, va, te = block_split_single_shard(shard_rows[shard], val_frac, test_frac)
        plan.block_ranges = [
            (shard, "train", tr.start, tr.stop),
            (shard, "val", va.start, va.stop),
            (shard, "test", te.start, te.stop),
        ]
        plan.train_shards = [shard]
        plan.val_shards = [shard]
        plan.test_shards = [shard]
        plan.n_train = tr.stop - tr.start
        plan.n_val = va.stop - va.start
        plan.n_test = te.stop - te.start
    else:
        raise ValueError(f"class {label} has no shards")

    return plan


def assert_forward_chaining(plan: ClassSplitPlan) -> None:
    """Leakage guard (doc §6.5): no test row precedes any train row within a class.

    For shard-split classes, the max natkey of any train shard must be < the
    min natkey of any test shard. For block/hybrid classes, contiguous ordering
    is guaranteed by construction (train range precedes val precedes test).
    """
    if plan.protocol == "forward_chain":
        train_keys = [natkey(x) for x in plan.train_shards]
        val_keys = [natkey(x) for x in plan.val_shards]
        test_keys = [natkey(x) for x in plan.test_shards]
        assert max(train_keys) < min(val_keys), (
            f"{plan.label}: train shard order not before val ({plan.train_shards} vs {plan.val_shards})"
        )
        assert max(val_keys) < min(test_keys), (
            f"{plan.label}: val shard order not before test ({plan.val_shards} vs {plan.test_shards})"
        )
        overlap = (set(plan.train_shards) | set(plan.val_shards)) & set(plan.test_shards)
        assert not overlap, f"{plan.label}: shard appears in test and train/val: {overlap}"
    elif plan.protocol in ("block", "two_shard_hybrid"):
        # Ranges are emitted train-before-val-before-test on a single shard, or
        # (hybrid) train/val on an earlier shard than the whole-shard test.
        by_shard: dict[str, list[tuple[str, int, int]]] = {}
        for shard, split, start, stop in plan.block_ranges:
            by_shard.setdefault(shard, []).append((split, start, stop))
        for shard, ranges in by_shard.items():
            order = {"train": 0, "val": 1, "test": 2}
            ranges_sorted = sorted(ranges, key=lambda r: order[r[0]])
            for a, b in zip(ranges_sorted, ranges_sorted[1:]):
                assert a[2] <= b[1], (
                    f"{plan.label}: block ranges overlap on {shard}: {a} vs {b}"
                )


# ── Parquet bridge: shard runs → per-row split assignment (pure numpy) ───────
SPLIT_CODE = {"train": 0, "val": 1, "test": 2}
SPLIT_NAME = {v: k for k, v in SPLIT_CODE.items()}


@dataclass
class ShardRun:
    """One contiguous run of rows for a single shard, in parquet order."""

    shard: str
    label: str
    start: int          # global start row index (inclusive)
    n_rows: int

    @property
    def stop(self) -> int:
        return self.start + self.n_rows


def build_shard_runs(shard_col: np.ndarray, label_col: np.ndarray) -> list[ShardRun]:
    """Detect contiguous shard runs from the row-ordered ``source_csv_filename``
    column. The parquet is a pure shard-by-shard concatenation, so each shard is
    exactly one contiguous run; we assert that (a shard appearing twice would
    mean the build shuffled and block splits would be invalid, doc §4.3)."""
    n = len(shard_col)
    if n == 0:
        return []
    # Boundaries where the shard changes.
    change = np.flatnonzero(shard_col[1:] != shard_col[:-1]) + 1
    starts = np.concatenate(([0], change))
    stops = np.concatenate((change, [n]))
    runs: list[ShardRun] = []
    seen: set[str] = set()
    for s, e in zip(starts, stops):
        shard = str(shard_col[s])
        if shard in seen:
            raise ValueError(
                f"shard {shard!r} appears in >1 contiguous run — parquet is not a "
                "pure concatenator; block/temporal splits would be invalid (doc §4.3)"
            )
        seen.add(shard)
        runs.append(ShardRun(shard=shard, label=str(label_col[s]), start=int(s), n_rows=int(e - s)))
    return runs


def plan_all_classes(
    runs: list[ShardRun],
    val_frac: float = 0.10,
    test_frac: float = 0.20,
) -> dict[str, ClassSplitPlan]:
    """Build a per-class :class:`ClassSplitPlan` from the shard runs."""
    by_label: dict[str, list[ShardRun]] = {}
    for r in runs:
        by_label.setdefault(r.label, []).append(r)
    plans: dict[str, ClassSplitPlan] = {}
    for label, class_runs in by_label.items():
        shard_rows = {r.shard: r.n_rows for r in class_runs}
        plan = plan_class_split(
            label, list(shard_rows), shard_rows, val_frac=val_frac, test_frac=test_frac
        )
        assert_forward_chaining(plan)
        plans[label] = plan
    return plans


def assign_row_splits(
    runs: list[ShardRun], plans: dict[str, ClassSplitPlan], n_rows: int
) -> np.ndarray:
    """Return an ``int8`` array (0=train,1=val,2=test, -1=unassigned) of length
    ``n_rows`` mapping every parquet row to its split, per the class plans."""
    split = np.full(n_rows, -1, dtype=np.int8)
    run_by_shard = {r.shard: r for r in runs}

    for plan in plans.values():
        if plan.protocol == "forward_chain":
            for shard_list, code in (
                (plan.train_shards, 0), (plan.val_shards, 1), (plan.test_shards, 2)
            ):
                for shard in shard_list:
                    r = run_by_shard[shard]
                    split[r.start : r.stop] = code
        else:  # block / two_shard_hybrid
            # Whole-shard test split of the hybrid.
            for shard in plan.test_shards:
                if plan.protocol == "two_shard_hybrid":
                    r = run_by_shard[shard]
                    split[r.start : r.stop] = 2
            for shard, split_name, lo, hi in plan.block_ranges:
                r = run_by_shard[shard]
                split[r.start + lo : r.start + hi] = SPLIT_CODE[split_name]

    return split