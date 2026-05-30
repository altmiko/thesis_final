"""
Comprehensive domain validation pass across full CICIoT2023 processed splits.

This script validates inverse-transformed train/val/test feature matrices using
the domain validator, writes detailed reports, and runs synthetic corruption
tests to verify rule coverage.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from src.preprocessing.feature_groups import BINARY_FEATURES, CATEGORY_MAP, FEATURE_NAMES, INTEGER_FEATURES
from src.attack.validator import VALID_PROTOCOLS, validate_batch


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
OUT_DIR = ROOT / "results" / "validation"

OUT_TXT = OUT_DIR / "full_dataset_validation_report.txt"
OUT_PER_CLASS = OUT_DIR / "per_class_validity.csv"
OUT_PER_RULE = OUT_DIR / "per_rule_breakdown.csv"
OUT_SYNTH = OUT_DIR / "synthetic_corruption_tests.csv"


SPLITS = {
    "train": ("X_train.npy", "y_train.npy", "y_train_cat.npy"),
    "val": ("X_val.npy", "y_val.npy", "y_val_cat.npy"),
    "test": ("X_test.npy", "y_test.npy", "y_test_cat.npy"),
}


NONNEG_FEATURES = [
    "Header_Length",
    "Rate",
    "Time_To_Live",
    "Tot sum",
    "Min",
    "Max",
    "AVG",
    "Std",
    "Tot size",
    "IAT",
    "Number",
    "Variance",
    "fin_flag_number",
    "syn_flag_number",
    "rst_flag_number",
    "psh_flag_number",
    "ack_flag_number",
    "ece_flag_number",
    "cwr_flag_number",
    "ack_count",
    "syn_count",
    "fin_count",
    "rst_count",
]


@dataclass
class SplitSummary:
    name: str
    total: int
    valid: int

    @property
    def rate(self) -> float:
        return float(self.valid / self.total) if self.total > 0 else 0.0


class TeeWriter:
    def __init__(self, path: Path) -> None:
        self._fh = path.open("w", encoding="utf-8", newline="")

    def write(self, line: str = "") -> None:
        print(line)
        self._fh.write(line + "\n")

    def close(self) -> None:
        self._fh.close()


def fmt_pct(x: float) -> str:
    return f"{x * 100.0:.2f}%"


def fmt_num(x: float) -> str:
    s = f"{x:.6f}"
    s = s.rstrip("0").rstrip(".") if "." in s else s
    return s


def rule_firing_for_one_sample(sample_raw: np.ndarray) -> List[str]:
    vr = validate_batch(sample_raw.reshape(1, -1), FEATURE_NAMES)
    fired = [rule for rule, mask in vr.violations_per_rule.items() if bool(mask[0])]
    fired.sort()
    return fired


def run_synthetic_corruption_tests(base_sample: np.ndarray) -> List[Dict[str, object]]:
    idx = {f: i for i, f in enumerate(FEATURE_NAMES)}

    tests = [
        {
            "test_id": "T1",
            "description": "Protocol Type = 5.7 (non-integer)",
            "changes": [("Protocol Type", 5.7)],
            "expected": ["R_protocol_valid"],
        },
        {
            "test_id": "T2",
            "description": "Protocol Type = 4 (not in allowlist)",
            "changes": [("Protocol Type", 4.0)],
            "expected": ["R_protocol_valid"],
        },
        {
            "test_id": "T3",
            "description": "TCP = 0.5 (fractional binary)",
            "changes": [("TCP", 0.5)],
            "expected": ["R_binary_TCP"],
        },
        {
            "test_id": "T4",
            "description": "fin_flag_number = -1 (negative count)",
            "changes": [("fin_flag_number", -1.0)],
            "expected": ["R_nonneg_fin_flag_number"],
        },
        {
            "test_id": "T5",
            "description": "Min = 100, Max = 50 (Min > Max)",
            "changes": [("Min", 100.0), ("Max", 50.0)],
            "expected": ["R_min_leq_max"],
        },
        {
            "test_id": "T6",
            "description": "Variance=100, Std=5 (Var != Std^2)",
            "changes": [("Variance", 100.0), ("Std", 5.0)],
            "expected": ["R_var_eq_std_sq"],
        },
        {
            "test_id": "T7",
            "description": "Number = -3 (negative packet count)",
            "changes": [("Number", -3.0)],
            "expected": ["R_pkts_positive"],
        },
        {
            "test_id": "T8",
            "description": "AVG = 200 while Min=60, Max=100",
            "changes": [("Min", 60.0), ("Max", 100.0), ("AVG", 200.0)],
            "expected": ["R_avg_in_range"],
        },
        {
            "test_id": "T9",
            "description": "Multiple: Protocol=5.7, TCP=0.3, Min>Max",
            "changes": [("Protocol Type", 5.7), ("TCP", 0.3), ("Min", 100.0), ("Max", 50.0)],
            "expected": ["R_protocol_valid", "R_binary_TCP", "R_min_leq_max"],
        },
    ]

    out: List[Dict[str, object]] = []
    for t in tests:
        sample = np.array(base_sample, dtype=np.float32, copy=True)
        for feat, val in t["changes"]:
            sample[idx[feat]] = np.float32(val)

        actual = rule_firing_for_one_sample(sample)
        expected = sorted(t["expected"])

        missing = [r for r in expected if r not in actual]
        status = "PASS" if not missing else "FAIL"

        out.append(
            {
                "test_id": t["test_id"],
                "corruption_description": t["description"],
                "expected_rules": "; ".join(expected),
                "actual_rules": "; ".join(actual),
                "status": status,
                "missing_rules": "; ".join(missing),
            }
        )

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate full CICIoT2023 processed dataset via domain rules")
    parser.add_argument("--chunk-size", type=int, default=200000)
    parser.add_argument("--max-failure-examples", type=int, default=20)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(DATA_DIR / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(DATA_DIR / "label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)
    with open(DATA_DIR / "category_encoder.pkl", "rb") as f:
        category_encoder = pickle.load(f)

    with open(DATA_DIR / "class_to_category.json", "r", encoding="utf-8") as f:
        class_to_category = json.load(f)

    n_features = len(FEATURE_NAMES)
    n_classes_34 = len(label_encoder.classes_)
    n_classes_8 = len(category_encoder.classes_)

    split_summaries: Dict[str, SplitSummary] = {}
    split_rule_fails: Dict[str, Dict[str, int]] = {}

    global_rule_fails: Dict[str, int] = {}
    total_samples_all = 0

    class_total_by_split: Dict[str, np.ndarray] = {}
    class_pass_by_split: Dict[str, np.ndarray] = {}
    cat_total_by_split: Dict[str, np.ndarray] = {}
    cat_pass_by_split: Dict[str, np.ndarray] = {}

    protocol_dev_min = np.inf
    protocol_dev_max = -np.inf
    binary_dev_min = np.inf
    binary_dev_max = -np.inf

    idx = {f: i for i, f in enumerate(FEATURE_NAMES)}
    proto_idx = idx["Protocol Type"]
    binary_idx = np.array([idx[f] for f in BINARY_FEATURES], dtype=np.int64)
    int_idx = np.array([idx[f] for f in INTEGER_FEATURES], dtype=np.int64)
    nonneg_idx = np.array([idx[f] for f in NONNEG_FEATURES], dtype=np.int64)

    independent_totals = {
        "min_le_avg_le_max": 0,
        "var_eq_std_sq_5pct": 0,
        "binary_within_0p01": 0,
        "integer_within_0p01": 0,
        "protocol_allowlist_within_0p01": 0,
        "nonneg_ge_neg0p01": 0,
    }

    failing_examples: List[Dict[str, object]] = []
    first_raw_for_synth: np.ndarray | None = None

    tee = TeeWriter(OUT_TXT)

    tee.write("FULL DATASET VALIDATION RUN")
    tee.write("=" * 80)
    tee.write(f"Features expected: {n_features}")
    tee.write(f"Chunk size: {args.chunk_size}")
    tee.write(f"34-class labels: {n_classes_34}")
    tee.write(f"8-class labels: {n_classes_8}")
    tee.write()

    for split_name, (x_file, y34_file, y8_file) in SPLITS.items():
        x_path = DATA_DIR / x_file
        y34_path = DATA_DIR / y34_file
        y8_path = DATA_DIR / y8_file

        x_scaled = np.load(x_path, mmap_mode="r")
        y34 = np.load(y34_path)
        y8 = np.load(y8_path)

        if x_scaled.ndim != 2 or x_scaled.shape[1] != n_features:
            raise ValueError(f"{x_file} has shape {x_scaled.shape}, expected (*, {n_features})")
        if len(x_scaled) != len(y34) or len(x_scaled) != len(y8):
            raise ValueError(f"Length mismatch for split {split_name}")

        n = len(x_scaled)
        total_samples_all += n

        class_total = np.zeros(n_classes_34, dtype=np.int64)
        class_pass = np.zeros(n_classes_34, dtype=np.int64)
        cat_total = np.zeros(n_classes_8, dtype=np.int64)
        cat_pass = np.zeros(n_classes_8, dtype=np.int64)
        split_rule_fails[split_name] = {}

        split_valid = 0

        tee.write(f"Split: {split_name}")
        tee.write(f"  X shape: {x_scaled.shape}")
        tee.write(f"  y_34 shape: {y34.shape}")
        tee.write(f"  y_8 shape: {y8.shape}")

        for start in range(0, n, args.chunk_size):
            end = min(start + args.chunk_size, n)

            x_chunk_scaled = np.array(x_scaled[start:end], dtype=np.float32, copy=True)
            x_chunk_raw = np.asarray(scaler.inverse_transform(x_chunk_scaled), dtype=np.float32)
            y34_chunk = np.asarray(y34[start:end], dtype=np.int64)
            y8_chunk = np.asarray(y8[start:end], dtype=np.int64)

            if first_raw_for_synth is None and len(x_chunk_raw) > 0:
                first_raw_for_synth = np.array(x_chunk_raw[0], dtype=np.float32, copy=True)

            vr = validate_batch(x_chunk_raw, FEATURE_NAMES)
            valid = vr.overall_valid
            split_valid += int(valid.sum())

            for rule, fail_mask in vr.violations_per_rule.items():
                fail_count = int(np.asarray(fail_mask, dtype=np.int64).sum())
                split_rule_fails[split_name][rule] = split_rule_fails[split_name].get(rule, 0) + fail_count
                global_rule_fails[rule] = global_rule_fails.get(rule, 0) + fail_count

            class_total += np.bincount(y34_chunk, minlength=n_classes_34)
            class_pass += np.bincount(y34_chunk, weights=valid.astype(np.int64), minlength=n_classes_34).astype(np.int64)
            cat_total += np.bincount(y8_chunk, minlength=n_classes_8)
            cat_pass += np.bincount(y8_chunk, weights=valid.astype(np.int64), minlength=n_classes_8).astype(np.int64)

            protocol_vals = x_chunk_raw[:, proto_idx]
            proto_dev = np.abs(protocol_vals - np.round(protocol_vals))
            protocol_dev_min = min(protocol_dev_min, float(proto_dev.min()))
            protocol_dev_max = max(protocol_dev_max, float(proto_dev.max()))

            binary_vals = x_chunk_raw[:, binary_idx]
            bin_dev = np.minimum(np.abs(binary_vals), np.abs(binary_vals - 1.0))
            binary_dev_min = min(binary_dev_min, float(bin_dev.min()))
            binary_dev_max = max(binary_dev_max, float(bin_dev.max()))

            min_vals = x_chunk_raw[:, idx["Min"]]
            avg_vals = x_chunk_raw[:, idx["AVG"]]
            max_vals = x_chunk_raw[:, idx["Max"]]
            mask_min_avg_max = (min_vals <= (avg_vals + 0.01)) & (avg_vals <= (max_vals + 0.01))

            std_vals = x_chunk_raw[:, idx["Std"]]
            var_vals = x_chunk_raw[:, idx["Variance"]]
            exp_var = std_vals ** 2
            abs_err = np.abs(var_vals - exp_var)
            rel_err = abs_err / (np.abs(exp_var) + 1e-8)
            mask_var_std = (abs_err <= 0.01) | (rel_err <= 0.05)

            mask_binary = (bin_dev <= 0.01).all(axis=1)
            int_vals = x_chunk_raw[:, int_idx]
            int_dev = np.abs(int_vals - np.round(int_vals))
            mask_integer = (int_dev <= 0.01).all(axis=1)

            protocol_dist_allow = np.min(
                np.abs(protocol_vals[:, None] - np.array(sorted(VALID_PROTOCOLS), dtype=np.float32)[None, :]),
                axis=1,
            )
            mask_proto_allow = protocol_dist_allow <= 0.01

            nonneg_vals = x_chunk_raw[:, nonneg_idx]
            mask_nonneg = (nonneg_vals >= -0.01).all(axis=1)

            independent_totals["min_le_avg_le_max"] += int(mask_min_avg_max.sum())
            independent_totals["var_eq_std_sq_5pct"] += int(mask_var_std.sum())
            independent_totals["binary_within_0p01"] += int(mask_binary.sum())
            independent_totals["integer_within_0p01"] += int(mask_integer.sum())
            independent_totals["protocol_allowlist_within_0p01"] += int(mask_proto_allow.sum())
            independent_totals["nonneg_ge_neg0p01"] += int(mask_nonneg.sum())

            if len(failing_examples) < args.max_failure_examples:
                bad_local = np.where(~valid)[0]
                for loc in bad_local:
                    if len(failing_examples) >= args.max_failure_examples:
                        break
                    gidx = start + int(loc)
                    fired = [r for r, m in vr.violations_per_rule.items() if bool(np.asarray(m)[loc])]
                    cls_idx = int(y34_chunk[loc])
                    cls_name = str(label_encoder.inverse_transform(np.array([cls_idx], dtype=np.int64))[0])
                    failing_examples.append(
                        {
                            "split": split_name,
                            "global_index": gidx,
                            "class_index": cls_idx,
                            "class_name": cls_name,
                            "violated_rules": fired,
                            "Protocol Type": float(x_chunk_raw[loc, proto_idx]),
                            "TCP": float(x_chunk_raw[loc, idx["TCP"]]),
                            "UDP": float(x_chunk_raw[loc, idx["UDP"]]),
                            "Min": float(x_chunk_raw[loc, idx["Min"]]),
                            "AVG": float(x_chunk_raw[loc, idx["AVG"]]),
                            "Max": float(x_chunk_raw[loc, idx["Max"]]),
                            "Std": float(x_chunk_raw[loc, idx["Std"]]),
                            "Variance": float(x_chunk_raw[loc, idx["Variance"]]),
                            "Number": float(x_chunk_raw[loc, idx["Number"]]),
                        }
                    )

        split_summaries[split_name] = SplitSummary(split_name, total=n, valid=split_valid)
        class_total_by_split[split_name] = class_total
        class_pass_by_split[split_name] = class_pass
        cat_total_by_split[split_name] = cat_total
        cat_pass_by_split[split_name] = cat_pass

        tee.write(f"  Valid all-rules: {split_valid}/{n} ({fmt_pct(split_valid / n)})")
        failed_rules = [(r, c) for r, c in split_rule_fails[split_name].items() if c > 0]
        if failed_rules:
            tee.write("  Failing rules in this split:")
            for r, c in sorted(failed_rules, key=lambda kv: (-kv[1], kv[0])):
                tee.write(f"    {r}: {c}")
        else:
            tee.write("  Failing rules in this split: none")
        tee.write()

    if first_raw_for_synth is None:
        raise RuntimeError("Could not obtain base sample for synthetic tests")

    rule_names = sorted(global_rule_fails.keys())
    n_rules = len(rule_names)

    tee.write("PER-RULE ANALYSIS (ALL SPLITS COMBINED)")
    tee.write("-" * 80)
    tee.write("rule_name | total_checked | total_passing | total_failing | failure_rate")
    per_rule_rows = []
    for rule in sorted(rule_names, key=lambda r: (-global_rule_fails[r], r)):
        fail = int(global_rule_fails[rule])
        total_checked = total_samples_all
        total_passing = total_checked - fail
        rate = fail / total_checked if total_checked > 0 else 0.0
        tee.write(f"{rule} | {total_checked} | {total_passing} | {fail} | {fmt_pct(rate)}")
        per_rule_rows.append(
            {
                "rule_name": rule,
                "total_checked": total_checked,
                "total_passing": total_passing,
                "total_failing": fail,
                "failure_rate": rate,
            }
        )

    zero_fail_rules = sum(1 for r in rule_names if global_rule_fails[r] == 0)
    if zero_fail_rules == n_rules:
        tee.write(f"All {n_rules} rules pass on all {total_samples_all} samples across all splits.")
    tee.write()

    tee.write("PER-CLASS VALIDITY BREAKDOWN (34-CLASS)")
    tee.write("-" * 80)
    tee.write("split | class_index | class_name | category_name | total | passing | validity_rate")

    per_class_rows = []
    for split_name in SPLITS.keys():
        totals = class_total_by_split[split_name]
        passing = class_pass_by_split[split_name]
        for cls_idx in range(n_classes_34):
            total = int(totals[cls_idx])
            passed = int(passing[cls_idx])
            rate = (passed / total) if total > 0 else 0.0
            cls_name = str(label_encoder.inverse_transform(np.array([cls_idx], dtype=np.int64))[0])
            cat_name = str(class_to_category.get(cls_name, CATEGORY_MAP.get(cls_name, "Unknown")))
            tee.write(
                f"{split_name} | {cls_idx} | {cls_name} | {cat_name} | {total} | {passed} | {fmt_pct(rate)}"
            )
            per_class_rows.append(
                {
                    "class_index": cls_idx,
                    "class_name": cls_name,
                    "category_name": cat_name,
                    "split": split_name,
                    "total_samples": total,
                    "passing": passed,
                    "validity_rate": rate,
                }
            )
    tee.write()

    tee.write("PER-CATEGORY VALIDITY BREAKDOWN (8-CLASS)")
    tee.write("-" * 80)
    tee.write("split | category_index | category_name | total | passing | validity_rate")
    for split_name in SPLITS.keys():
        totals = cat_total_by_split[split_name]
        passing = cat_pass_by_split[split_name]
        for cat_idx in range(n_classes_8):
            total = int(totals[cat_idx])
            passed = int(passing[cat_idx])
            rate = (passed / total) if total > 0 else 0.0
            cat_name = str(category_encoder.inverse_transform(np.array([cat_idx], dtype=np.int64))[0])
            tee.write(f"{split_name} | {cat_idx} | {cat_name} | {total} | {passed} | {fmt_pct(rate)}")
    tee.write()

    tee.write("FLOAT PRECISION DIAGNOSTICS (AFTER INVERSE TRANSFORM)")
    tee.write("-" * 80)
    tee.write(f"Protocol Type deviation from nearest integer: min={fmt_num(protocol_dev_min)}, max={fmt_num(protocol_dev_max)}")
    tee.write(f"Binary feature deviation from nearest of {{0,1}}: min={fmt_num(binary_dev_min)}, max={fmt_num(binary_dev_max)}")
    tee.write("Validator behavior note: protocol/binary/integer checks round values before rule tests.")
    tee.write()

    tee.write("INDEPENDENT INTER-FEATURE CONSISTENCY CHECKS")
    tee.write("-" * 80)
    for k, passed in independent_totals.items():
        rate = (passed / total_samples_all) if total_samples_all > 0 else 0.0
        tee.write(f"{k}: {passed}/{total_samples_all} ({fmt_pct(rate)})")
    tee.write()

    tee.write("SYNTHETIC CORRUPTION TESTS")
    tee.write("-" * 80)
    synth_rows = run_synthetic_corruption_tests(first_raw_for_synth)
    synth_pass = 0
    for row in synth_rows:
        status = str(row["status"])
        if status == "PASS":
            synth_pass += 1
        tee.write(
            f"{row['test_id']} | {row['corruption_description']} | expected=[{row['expected_rules']}] | "
            f"actual=[{row['actual_rules']}] | {status}"
        )
        if row["missing_rules"]:
            tee.write(f"  Missing expected rules: {row['missing_rules']}")
    tee.write()

    if failing_examples:
        tee.write("CRITICAL: CLEAN DATA VALIDATION FAILURES DETECTED")
        tee.write("-" * 80)
        for ex in failing_examples:
            tee.write(
                f"split={ex['split']} idx={ex['global_index']} class={ex['class_index']} {ex['class_name']} "
                f"rules={ex['violated_rules']}"
            )
            tee.write(
                f"  Protocol={fmt_num(ex['Protocol Type'])}, TCP={fmt_num(ex['TCP'])}, UDP={fmt_num(ex['UDP'])}, "
                f"Min={fmt_num(ex['Min'])}, AVG={fmt_num(ex['AVG'])}, Max={fmt_num(ex['Max'])}, "
                f"Std={fmt_num(ex['Std'])}, Variance={fmt_num(ex['Variance'])}, Number={fmt_num(ex['Number'])}"
            )
        tee.write()

    train_s = split_summaries["train"]
    val_s = split_summaries["val"]
    test_s = split_summaries["test"]
    total_valid_all = train_s.valid + val_s.valid + test_s.valid
    total_rate = total_valid_all / total_samples_all if total_samples_all > 0 else 0.0

    rules_zero = sum(1 for r in rule_names if global_rule_fails[r] == 0)
    synth_total = len(synth_rows)

    verdict = "PASS"
    verdict_detail = "all clean data valid, validator catches all synthetic corruptions"
    if total_valid_all != total_samples_all or synth_pass != synth_total:
        verdict = "FAIL"
        if total_valid_all != total_samples_all and synth_pass != synth_total:
            verdict_detail = "clean-data failures present and synthetic corruption tests missed"
        elif total_valid_all != total_samples_all:
            verdict_detail = "clean-data failures present"
        else:
            verdict_detail = "synthetic corruption tests missed"

    tee.write("FULL DATASET VALIDATION SUMMARY")
    tee.write("=" * 80)
    tee.write(
        f"Train: {train_s.total:,} samples - {train_s.valid:,}/{train_s.total:,} valid ({fmt_pct(train_s.rate)})"
    )
    tee.write(f"Val:   {val_s.total:,} samples - {val_s.valid:,}/{val_s.total:,} valid ({fmt_pct(val_s.rate)})")
    tee.write(
        f"Test:  {test_s.total:,} samples - {test_s.valid:,}/{test_s.total:,} valid ({fmt_pct(test_s.rate)})"
    )
    tee.write(
        f"Total: {total_samples_all:,} samples - {total_valid_all:,}/{total_samples_all:,} valid ({fmt_pct(total_rate)})"
    )
    tee.write()
    tee.write(f"Validator rules checked: {n_rules}")
    tee.write(f"Rules with zero failures: {rules_zero}/{n_rules}")
    tee.write(f"Synthetic corruption tests passed: {synth_pass}/{synth_total}")
    tee.write()
    tee.write(f"VERDICT: [{verdict} - {verdict_detail}]")

    with OUT_PER_CLASS.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "class_index",
                "class_name",
                "category_name",
                "split",
                "total_samples",
                "passing",
                "validity_rate",
            ],
        )
        writer.writeheader()
        writer.writerows(per_class_rows)

    with OUT_PER_RULE.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "rule_name",
                "total_checked",
                "total_passing",
                "total_failing",
                "failure_rate",
            ],
        )
        writer.writeheader()
        writer.writerows(per_rule_rows)

    with OUT_SYNTH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["test_id", "corruption_description", "expected_rules", "actual_rules", "status"],
        )
        writer.writeheader()
        for row in synth_rows:
            writer.writerow(
                {
                    "test_id": row["test_id"],
                    "corruption_description": row["corruption_description"],
                    "expected_rules": row["expected_rules"],
                    "actual_rules": row["actual_rules"],
                    "status": row["status"],
                }
            )

    tee.close()

    print("=" * 80)
    print("Validation run complete")
    print(f"Report: {OUT_TXT}")
    print(f"Per-class CSV: {OUT_PER_CLASS}")
    print(f"Per-rule CSV: {OUT_PER_RULE}")
    print(f"Synthetic tests CSV: {OUT_SYNTH}")


if __name__ == "__main__":
    main()
