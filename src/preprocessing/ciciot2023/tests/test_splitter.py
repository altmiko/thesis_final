"""Unit tests for the leakage-critical split protocol (doc §4, §6.5)."""
from __future__ import annotations

import pytest

from src.preprocessing.ciciot2023.splitter import (
    assert_forward_chaining,
    block_split_single_shard,
    forward_chain_shards,
    natkey,
    plan_class_split,
)


def test_natkey_orders_numerically_not_lexically():
    shards = ["F10.csv", "F2.csv", "F1.csv", "F.csv"]
    ordered = sorted(shards, key=natkey)
    assert ordered == ["F.csv", "F1.csv", "F2.csv", "F10.csv"]


def test_forward_chain_latest_shards_to_test():
    shards = [f"DDoS-ICMP_Flood{i}.pcap.csv" for i in range(1, 28)]  # 27 shards
    shards.append("DDoS-ICMP_Flood.pcap.csv")  # the base (earliest) file
    train, val, test = forward_chain_shards(shards, val_frac=0.10, test_frac=0.20)
    assert len(train) + len(val) + len(test) == len(shards)
    # test is the newest slice; train the oldest.
    assert max(natkey(x) for x in train) < min(natkey(x) for x in val)
    assert max(natkey(x) for x in val) < min(natkey(x) for x in test)


def test_forward_chain_rejects_too_few_shards():
    with pytest.raises(ValueError, match="too few shards"):
        forward_chain_shards(["a1.csv", "a2.csv"])


def test_block_split_is_contiguous_and_ordered():
    tr, va, te = block_split_single_shard(1000, val_frac=0.10, test_frac=0.20)
    assert (tr.start, tr.stop) == (0, 700)
    assert (va.start, va.stop) == (700, 800)
    assert (te.start, te.stop) == (800, 1000)
    # no test row precedes any train row
    assert tr.stop <= va.start <= te.start


def test_plan_forward_chain_class_passes_forward_chaining_guard():
    label = "DDOS-ICMP_FLOOD"
    shards = [f"DDoS-ICMP_Flood{i}.pcap.csv" for i in range(1, 6)]
    rows = {s: 1000 for s in shards}
    plan = plan_class_split(label, shards, rows)
    assert plan.protocol == "forward_chain"
    assert plan.n_train + plan.n_val + plan.n_test == 5000
    assert_forward_chaining(plan)  # must not raise


def test_plan_single_shard_uses_block_and_ranges_are_disjoint():
    plan = plan_class_split("XSS", ["XSS.pcap.csv"], {"XSS.pcap.csv": 3846})
    assert plan.protocol == "block"
    assert plan.n_train + plan.n_val + plan.n_test == 3846
    assert_forward_chaining(plan)
    # ranges cover the shard exactly with no overlap
    ranges = sorted(plan.block_ranges, key=lambda r: r[2])
    assert ranges[0][2] == 0 and ranges[-1][3] == 3846
    for a, b in zip(ranges, ranges[1:]):
        assert a[3] == b[2]


def test_plan_two_shard_hybrid():
    plan = plan_class_split(
        "DOS-HTTP_FLOOD",
        ["DoS-HTTP_Flood.pcap.csv", "DoS-HTTP_Flood1.pcap.csv"],
        {"DoS-HTTP_Flood.pcap.csv": 40684, "DoS-HTTP_Flood1.pcap.csv": 31173},
    )
    assert plan.protocol == "two_shard_hybrid"
    # later shard is the whole test split
    assert plan.test_shards == ["DoS-HTTP_Flood1.pcap.csv"]
    assert plan.n_test == 31173
    # earlier shard block-split into train/val, disjoint and covering it
    assert plan.n_train + plan.n_val == 40684
    assert_forward_chaining(plan)
