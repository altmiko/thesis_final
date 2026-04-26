"""
Generate supervisor-ready adversarial sample exhibits and attack summaries.

This script reads selected unconstrained attack outputs, inverse-transforms clean
and adversarial samples to raw feature space, validates samples, selects
high-value flipped-invalid examples, and writes a text report plus two CSV files.
"""

from __future__ import annotations

import argparse
import csv
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

from feature_groups import (
    BINARY_FEATURES,
    CATEGORY_MAP,
    FEATURE_NAMES,
    IMMUTABLE_FEATURES,
    INTEGER_FEATURES,
)
from validator import VALID_PROTOCOLS
from validator import validate_batch


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
ATTACKS_DIR = ROOT / "results" / "attacks"
OUT_TEXT = ATTACKS_DIR / "sample_exhibits.txt"
OUT_EXHIBITS_CSV = ATTACKS_DIR / "sample_exhibits_all.csv"
OUT_SUMMARY_CSV = ATTACKS_DIR / "attack_summaries.csv"

REQUIRED_ATTACK_FILES = [
    "attack_binary_fgsm_0.30.npz",
    "attack_binary_pgd_0.30.npz",
    "attack_binary_cw_0.npz",
    "attack_binary_pgd_0.05.npz",
    "attack_8class_pgd_0.30.npz",
    "attack_34class_pgd_0.30.npz",
    "attack_binary_fgsm_0.05.npz",
]

NONNEG_FEATURES = {
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
}

MANUAL_CHECK_TARGETS = {
    "Protocol Type",
    "Min",
    "AVG",
    "Max",
    "Std",
    "Variance",
    "Number",
} | set(BINARY_FEATURES) | set(INTEGER_FEATURES) | NONNEG_FEATURES


@dataclass
class AttackData:
    attack_config: str
    task_name: str
    attack_name: str
    eps_label: str
    x_clean_scaled: np.ndarray
    x_adv_scaled: np.ndarray
    x_clean_raw: np.ndarray
    x_adv_raw: np.ndarray
    y_true: np.ndarray
    y_pred_clean: np.ndarray
    y_pred_adv: np.ndarray


class TeePrinter:
    def __init__(self, text_path: Path) -> None:
        self._fh = text_path.open("w", encoding="utf-8", newline="")

    def println(self, line: str = "") -> None:
        print(line)
        self._fh.write(line + "\n")

    def close(self) -> None:
        self._fh.close()


def mask_label(mask_val: float) -> str:
    if np.isclose(mask_val, 0.0):
        return "FROZEN"
    if np.isclose(mask_val, 0.3):
        return "0.3"
    if np.isclose(mask_val, 1.0):
        return "1.0"
    return f"{mask_val:.3f}"


def parse_attack_file_name(filename: str) -> Tuple[str, str, str]:
    stem = Path(filename).stem
    parts = stem.split("_")
    if len(parts) != 4 or parts[0] != "attack":
        raise ValueError(f"Unexpected attack filename format: {filename}")
    return parts[1], parts[2], parts[3]


def decode_label(task_name: str, label: int, label_encoder, category_encoder) -> str:
    if task_name == "binary":
        return "Benign" if int(label) == 0 else "Attack"
    if task_name == "8class":
        return str(category_encoder.inverse_transform(np.array([int(label)], dtype=np.int64))[0])
    if task_name == "34class":
        return str(label_encoder.inverse_transform(np.array([int(label)], dtype=np.int64))[0])
    raise ValueError(f"Unknown task name: {task_name}")


def protocol_status(value: float, treat_adv: bool) -> str:
    nearest = int(np.round(value))
    if treat_adv and abs(value - nearest) > 0.01:
        return f"INVALID (non-integer: {value:.4f})"
    if nearest not in VALID_PROTOCOLS:
        return f"INVALID (not in allowlist: {nearest})"
    return str(nearest)


def build_manual_feature_violations(row_adv: np.ndarray) -> Tuple[Dict[str, List[str]], List[str]]:
    idx = {name: i for i, name in enumerate(FEATURE_NAMES)}
    out: Dict[str, List[str]] = {}
    global_msgs: List[str] = []

    def add(feat: str, msg: str) -> None:
        out.setdefault(feat, []).append(msg)

    proto = float(row_adv[idx["Protocol Type"]])
    proto_nearest = int(np.round(proto))
    if abs(proto - proto_nearest) > 0.01:
        add("Protocol Type", "Protocol invalid: non-integer")
    elif proto_nearest not in VALID_PROTOCOLS:
        add("Protocol Type", "Protocol invalid: not in allowlist")

    for feat in BINARY_FEATURES:
        val = float(row_adv[idx[feat]])
        if abs(val - round(val)) > 0.01 or int(round(val)) not in (0, 1):
            add(feat, "Binary constraint violated (must be 0/1)")

    for feat in INTEGER_FEATURES:
        val = float(row_adv[idx[feat]])
        if abs(val - round(val)) > 0.01:
            add(feat, "Integer constraint violated")

    for feat in NONNEG_FEATURES:
        val = float(row_adv[idx[feat]])
        if val < 0:
            add(feat, "Non-negative constraint violated")

    min_val = float(row_adv[idx["Min"]])
    avg_val = float(row_adv[idx["AVG"]])
    max_val = float(row_adv[idx["Max"]])
    if min_val > max_val:
        add("Min", "Min > Max")
        add("Max", "Min > Max")
    if avg_val < min_val or avg_val > max_val:
        add("AVG", "AVG outside [Min, Max]")

    std_val = float(row_adv[idx["Std"]])
    var_val = float(row_adv[idx["Variance"]])
    exp_var = std_val * std_val
    rel_err = abs(var_val - exp_var) / (exp_var + 1e-8)
    if rel_err > 0.05:
        add("Std", "Variance mismatch vs Std^2")
        add("Variance", "Variance mismatch vs Std^2")

    num_val = float(row_adv[idx["Number"]])
    if num_val < 1:
        add("Number", "Packet count must be >= 1")
    if abs(num_val - round(num_val)) > 0.5:
        add("Number", "Packet count must be integer-like")

    if min_val > max_val:
        global_msgs.append(f"Ordering invalid: Min={min_val:.4f} > Max={max_val:.4f}")
    if avg_val < min_val or avg_val > max_val:
        global_msgs.append(
            f"AVG out of range: Min={min_val:.4f}, AVG={avg_val:.4f}, Max={max_val:.4f}"
        )
    if rel_err > 0.05:
        global_msgs.append(
            f"Variance mismatch: Std={std_val:.4f}, Variance={var_val:.4f}, Std^2={exp_var:.4f}"
        )

    return out, global_msgs


def ensure_files_exist() -> List[Path]:
    files = [ATTACKS_DIR / name for name in REQUIRED_ATTACK_FILES]
    missing = [str(p) for p in files if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required attack files:\n" + "\n".join(missing))
    return files


def load_attack_data(npz_path: Path, scaler) -> AttackData:
    with np.load(npz_path, allow_pickle=True) as data:
        expected = ["X_clean", "X_adv", "y_true", "y_pred_clean", "y_pred_adv"]
        for key in expected:
            if key not in data.files:
                raise KeyError(f"{npz_path.name} missing key: {key}")

        x_clean_scaled = np.asarray(data["X_clean"], dtype=np.float32)
        x_adv_scaled = np.asarray(data["X_adv"], dtype=np.float32)
        y_true = np.asarray(data["y_true"], dtype=np.int64)
        y_pred_clean = np.asarray(data["y_pred_clean"], dtype=np.int64)
        y_pred_adv = np.asarray(data["y_pred_adv"], dtype=np.int64)

    if x_clean_scaled.shape != x_adv_scaled.shape:
        raise ValueError(f"Shape mismatch in {npz_path.name}: clean={x_clean_scaled.shape}, adv={x_adv_scaled.shape}")
    if x_clean_scaled.ndim != 2 or x_clean_scaled.shape[1] != len(FEATURE_NAMES):
        raise ValueError(
            f"Unexpected feature shape in {npz_path.name}: {x_clean_scaled.shape}, expected (*, {len(FEATURE_NAMES)})"
        )

    n = x_clean_scaled.shape[0]
    for arr_name, arr in [
        ("y_true", y_true),
        ("y_pred_clean", y_pred_clean),
        ("y_pred_adv", y_pred_adv),
    ]:
        if arr.shape[0] != n:
            raise ValueError(f"Length mismatch in {npz_path.name}: {arr_name} has {arr.shape[0]}, expected {n}")

    x_clean_raw = np.asarray(scaler.inverse_transform(x_clean_scaled), dtype=np.float32)
    x_adv_raw = np.asarray(scaler.inverse_transform(x_adv_scaled), dtype=np.float32)

    task_name, attack_name, eps_label = parse_attack_file_name(npz_path.name)
    return AttackData(
        attack_config=npz_path.stem,
        task_name=task_name,
        attack_name=attack_name,
        eps_label=eps_label,
        x_clean_scaled=x_clean_scaled,
        x_adv_scaled=x_adv_scaled,
        x_clean_raw=x_clean_raw,
        x_adv_raw=x_adv_raw,
        y_true=y_true,
        y_pred_clean=y_pred_clean,
        y_pred_adv=y_pred_adv,
    )


def compute_rule_counts_on_mask(
    violations_per_rule: Dict[str, np.ndarray],
    mask: np.ndarray,
) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for rule, rule_mask in violations_per_rule.items():
        counts[rule] = int(np.asarray(rule_mask, dtype=bool)[mask].sum())
    return counts


def select_samples(
    adv_data: AttackData,
    adv_valid: np.ndarray,
    n_violations: np.ndarray,
    delta_linf_raw: np.ndarray,
    delta_l2_raw: np.ndarray,
    changed_count: np.ndarray,
    max_samples: int,
) -> List[int]:
    clean_correct = adv_data.y_pred_clean == adv_data.y_true
    flipped = adv_data.y_pred_adv != adv_data.y_true
    invalid_adv = ~adv_valid

    candidate_idx = np.where(clean_correct & flipped & invalid_adv)[0]
    if len(candidate_idx) == 0:
        return []

    score_rows = [
        (
            int(i),
            int(n_violations[i]),
            float(delta_linf_raw[i]),
            float(delta_l2_raw[i]),
            int(changed_count[i]),
            int(adv_data.y_true[i]),
        )
        for i in candidate_idx
    ]

    score_rows.sort(key=lambda r: (-r[1], -r[2], -r[3], -r[4], r[0]))

    selected: List[int] = []
    covered_classes = set()

    for idx, _, _, _, _, true_cls in score_rows:
        if true_cls in covered_classes:
            continue
        selected.append(idx)
        covered_classes.add(true_cls)
        if len(selected) >= max_samples:
            return selected

    for idx, *_ in score_rows:
        if idx in selected:
            continue
        selected.append(idx)
        if len(selected) >= max_samples:
            break

    return selected


def print_feature_reference_table(pr: TeePrinter, mask: np.ndarray) -> None:
    pr.println("=" * 120)
    pr.println("FEATURE REFERENCE TABLE (39 features)")
    pr.println("=" * 120)
    pr.println("Idx | Feature           | Mask   | Type      | Constraint summary")
    pr.println("-" * 120)
    for i, feat in enumerate(FEATURE_NAMES):
        if feat in BINARY_FEATURES:
            ftype = "binary"
            constraint = "must be 0/1"
        elif feat in INTEGER_FEATURES:
            ftype = "integer"
            constraint = "integer and non-negative"
        else:
            ftype = "continuous"
            if feat in NONNEG_FEATURES:
                constraint = "non-negative"
            else:
                constraint = "domain-specific"

        if feat == "Protocol Type":
            constraint = "allowed protocol in {0,1,2,6,17,47}"
        if feat == "Time_To_Live":
            constraint = "range [0,255]"
        if feat == "Number":
            constraint = "packet count >=1 and integer-like"
        if feat == "AVG":
            constraint = "must lie within [Min, Max]"
        if feat == "Variance":
            constraint = "approximately Std^2 (5% tolerance)"

        pr.println(
            f"{i:>3d} | {feat:<17} | {mask_label(float(mask[i])):<6} | {ftype:<9} | {constraint}"
        )
    pr.println()


def _fmt_label(x: float) -> str:
    return f"{x:.4f}"


def _changed_feature_indices(clean_row: np.ndarray, adv_row: np.ndarray, threshold: float = 0.001) -> List[int]:
    delta = np.abs(adv_row - clean_row)
    idx = np.where(delta > threshold)[0]
    out = list(idx.tolist())
    out.sort(key=lambda j: -abs(float(adv_row[j] - clean_row[j])))
    return out


def print_baselines_for_classes(
    pr: TeePrinter,
    adv_data: AttackData,
    selected_indices: List[int],
    label_encoder,
    category_encoder,
) -> None:
    classes = sorted({int(adv_data.y_true[i]) for i in selected_indices})
    if not classes:
        return

    pr.println("Clean baseline sample per class present in selected exhibits")
    pr.println("-" * 120)
    for cls in classes:
        idx = int(np.where(adv_data.y_true == cls)[0][0])
        cls_name = decode_label(adv_data.task_name, cls, label_encoder, category_encoder)
        proto = float(adv_data.x_clean_raw[idx, FEATURE_NAMES.index("Protocol Type")])
        pr.println(
            f"class={cls_name:<20} sample={idx:>6d} Header_Length={adv_data.x_clean_raw[idx, FEATURE_NAMES.index('Header_Length')]:.4f} "
            f"Rate={adv_data.x_clean_raw[idx, FEATURE_NAMES.index('Rate')]:.4f} Protocol={protocol_status(proto, treat_adv=False)}"
        )
    pr.println()


def validate_encoder_coverage(
    pr: TeePrinter,
    data_by_file: Iterable[AttackData],
    label_encoder,
    category_encoder,
) -> None:
    pr.println("Label decode coverage check")
    pr.println("-" * 120)
    for attack_data in data_by_file:
        present = np.unique(attack_data.y_true)
        ok = True
        for y in present:
            try:
                _ = decode_label(attack_data.task_name, int(y), label_encoder, category_encoder)
            except Exception:
                ok = False
                break
        pr.println(
            f"{attack_data.attack_config:<30} labels={len(present):>3d} decode_coverage={'OK' if ok else 'FAIL'}"
        )
    pr.println()


def run(args: argparse.Namespace) -> None:
    ATTACKS_DIR.mkdir(parents=True, exist_ok=True)
    required_files = ensure_files_exist()

    with open(DATA_DIR / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(DATA_DIR / "label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)
    with open(DATA_DIR / "category_encoder.pkl", "rb") as f:
        category_encoder = pickle.load(f)

    perturbation_mask = np.load(DATA_DIR / "perturbation_mask.npy")
    if perturbation_mask.shape[0] != len(FEATURE_NAMES):
        raise ValueError(
            f"Mask length mismatch: got {perturbation_mask.shape[0]}, expected {len(FEATURE_NAMES)}"
        )

    loaded_data: List[AttackData] = [load_attack_data(path, scaler) for path in required_files]

    pr = TeePrinter(OUT_TEXT)
    all_exhibit_rows: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []

    pr.println("=" * 120)
    pr.println("ADVERSARIAL SAMPLE EXHIBITS")
    pr.println("=" * 120)
    pr.println("Preflight sanity checks")
    pr.println("-" * 120)
    pr.println(f"Feature count check: {len(FEATURE_NAMES)}")
    pr.println(f"Mask length check: {perturbation_mask.shape[0]}")
    pr.println(f"Scaler center_[:5]: {np.array2string(np.asarray(scaler.center_)[:5], precision=6)}")
    pr.println(f"Scaler scale_[:5]:  {np.array2string(np.asarray(scaler.scale_)[:5], precision=6)}")
    validate_encoder_coverage(pr, loaded_data, label_encoder, category_encoder)
    print_feature_reference_table(pr, perturbation_mask)

    for attack_data in loaded_data:
        clean_val = validate_batch(attack_data.x_clean_raw, FEATURE_NAMES)
        adv_val = validate_batch(attack_data.x_adv_raw, FEATURE_NAMES)

        clean_valid = clean_val.overall_valid
        adv_valid = adv_val.overall_valid
        clean_valid_rate = float(clean_valid.mean())

        clean_correct = attack_data.y_pred_clean == attack_data.y_true
        flipped = attack_data.y_pred_adv != attack_data.y_true
        candidate_mask = clean_correct & flipped

        delta_scaled = attack_data.x_adv_scaled - attack_data.x_clean_scaled
        delta_raw = attack_data.x_adv_raw - attack_data.x_clean_raw

        linf_scaled = np.max(np.abs(delta_scaled), axis=1)
        l2_scaled = np.linalg.norm(delta_scaled, axis=1)
        linf_raw = np.max(np.abs(delta_raw), axis=1)
        l2_raw = np.linalg.norm(delta_raw, axis=1)
        changed_count = (np.abs(delta_raw) > 0.001).sum(axis=1)

        n_violations = np.zeros(len(attack_data.y_true), dtype=np.int64)
        for mask in adv_val.violations_per_rule.values():
            n_violations += np.asarray(mask, dtype=np.int64)

        selected = select_samples(
            adv_data=attack_data,
            adv_valid=adv_valid,
            n_violations=n_violations,
            delta_linf_raw=linf_raw,
            delta_l2_raw=l2_raw,
            changed_count=changed_count,
            max_samples=args.max_samples_per_attack,
        )

        pr.println("=" * 120)
        pr.println(
            f"ATTACK: {attack_data.attack_config} (task={attack_data.task_name}, attack={attack_data.attack_name}, eps={attack_data.eps_label})"
        )
        pr.println("=" * 120)
        pr.println(f"Clean validity rate: {clean_valid_rate * 100.0:.2f}% ({int(clean_valid.sum())}/{len(clean_valid)})")

        n_clean_correct = int(clean_correct.sum())
        n_flipped = int(candidate_mask.sum())
        n_flipped_valid = int((candidate_mask & adv_valid).sum())
        clean_acc = float(clean_correct.mean())
        asr_raw = float(n_flipped / n_clean_correct) if n_clean_correct > 0 else 0.0
        validity_rate_flipped = float(n_flipped_valid / n_flipped) if n_flipped > 0 else 0.0
        asr_valid = float(n_flipped_valid / n_clean_correct) if n_clean_correct > 0 else 0.0

        per_rule_counts_flipped = compute_rule_counts_on_mask(
            adv_val.violations_per_rule,
            candidate_mask,
        )
        most_viol_rule = "NONE"
        most_viol_count = 0
        if per_rule_counts_flipped:
            most_viol_rule, most_viol_count = max(per_rule_counts_flipped.items(), key=lambda kv: kv[1])
            if most_viol_count == 0:
                most_viol_rule = "NONE"

        pct_viol = float(most_viol_count / n_flipped) if n_flipped > 0 else 0.0

        pr.println("Attack-level summary")
        pr.println("-" * 120)
        pr.println(f"Clean accuracy: {clean_acc * 100.0:.2f}%")
        pr.println(f"ASR_raw: {asr_raw * 100.0:.2f}% (flipped={n_flipped})")
        pr.println(f"Validity rate among flipped: {validity_rate_flipped * 100.0:.2f}%")
        pr.println(f"ASR_valid: {asr_valid * 100.0:.2f}%")

        if n_flipped > 0:
            linf_raw_flip = linf_raw[candidate_mask]
            l2_raw_flip = l2_raw[candidate_mask]
            changed_flip = changed_count[candidate_mask]
            pr.println(
                f"Raw perturbation L_inf (mean+-std): {linf_raw_flip.mean():.4f} +- {linf_raw_flip.std(ddof=0):.4f}"
            )
            pr.println(
                f"Raw perturbation L2   (mean+-std): {l2_raw_flip.mean():.4f} +- {l2_raw_flip.std(ddof=0):.4f}"
            )
            pr.println(
                f"Changed features among flipped (median/mean): {np.median(changed_flip):.1f}/{changed_flip.mean():.2f}"
            )
        else:
            pr.println("Raw perturbation stats unavailable: no flipped samples")

        nonzero_rule_lines = []
        for rule, count in sorted(per_rule_counts_flipped.items(), key=lambda kv: (-kv[1], kv[0])):
            if count <= 0:
                continue
            pct = 100.0 * count / max(n_flipped, 1)
            nonzero_rule_lines.append(f"{rule}: {count} ({pct:.2f}%)")

        if nonzero_rule_lines:
            pr.println("Most violated rules over flipped")
            for line in nonzero_rule_lines[:10]:
                pr.println(f"  {line}")
        else:
            pr.println("Most violated rules over flipped: none")

        if n_flipped > 0:
            mean_abs_delta = np.mean(np.abs(delta_raw[candidate_mask]), axis=0)
            nonzero_idx = np.where(mean_abs_delta > 0)[0]
            if len(nonzero_idx) > 0:
                order_desc = nonzero_idx[np.argsort(-mean_abs_delta[nonzero_idx])]
                order_asc = nonzero_idx[np.argsort(mean_abs_delta[nonzero_idx])]
                pr.println("Top 5 most perturbed features (mean |delta_raw| over flipped)")
                for j in order_desc[:5]:
                    pr.println(f"  {FEATURE_NAMES[int(j)]}: {mean_abs_delta[int(j)]:.6f}")
                pr.println("Top 5 least perturbed non-zero features (mean |delta_raw| over flipped)")
                for j in order_asc[:5]:
                    pr.println(f"  {FEATURE_NAMES[int(j)]}: {mean_abs_delta[int(j)]:.6f}")

        print_baselines_for_classes(pr, attack_data, selected, label_encoder, category_encoder)

        if len(selected) < args.max_samples_per_attack:
            pr.println(
                f"Selected exhibits: {len(selected)} (requested up to {args.max_samples_per_attack}; shortage due to strict criteria)"
            )
        else:
            pr.println(f"Selected exhibits: {len(selected)}")
        pr.println()

        frozen_idx = np.where(np.isclose(perturbation_mask, 0.0))[0]

        for rank, sample_idx in enumerate(selected, start=1):
            clean_row = attack_data.x_clean_raw[sample_idx]
            adv_row = attack_data.x_adv_raw[sample_idx]
            clean_row_scaled = attack_data.x_clean_scaled[sample_idx]
            adv_row_scaled = attack_data.x_adv_scaled[sample_idx]

            delta_row_raw = adv_row - clean_row
            delta_row_scaled = adv_row_scaled - clean_row_scaled

            violated_rules = [
                rule
                for rule, mask in adv_val.violations_per_rule.items()
                if bool(np.asarray(mask, dtype=bool)[sample_idx])
            ]

            manual_feat_msgs, manual_global = build_manual_feature_violations(adv_row)
            changed_idx = _changed_feature_indices(clean_row, adv_row, threshold=0.001)

            true_label = decode_label(
                attack_data.task_name,
                int(attack_data.y_true[sample_idx]),
                label_encoder,
                category_encoder,
            )
            clean_pred_label = decode_label(
                attack_data.task_name,
                int(attack_data.y_pred_clean[sample_idx]),
                label_encoder,
                category_encoder,
            )
            adv_pred_label = decode_label(
                attack_data.task_name,
                int(attack_data.y_pred_adv[sample_idx]),
                label_encoder,
                category_encoder,
            )

            pr.println("-" * 120)
            pr.println(
                f"EXHIBIT {rank}: sample_index={sample_idx} attack={attack_data.attack_name} model={attack_data.task_name}"
            )
            pr.println("-" * 120)
            pr.println(
                f"True={true_label} | CleanPred={clean_pred_label} ({'correct' if attack_data.y_pred_clean[sample_idx] == attack_data.y_true[sample_idx] else 'wrong'}) "
                f"| AdvPred={adv_pred_label} ({'flipped' if attack_data.y_pred_adv[sample_idx] != attack_data.y_true[sample_idx] else 'not-flipped'})"
            )
            pr.println(
                f"Perturbation scaled: L_inf={np.max(np.abs(delta_row_scaled)):.4f}, L2={np.linalg.norm(delta_row_scaled):.4f}"
            )
            pr.println(
                f"Perturbation raw:    L_inf={np.max(np.abs(delta_row_raw)):.4f}, L2={np.linalg.norm(delta_row_raw):.4f}"
            )
            pr.println("Changed features (|delta_raw| > 0.001, sorted desc by |delta_raw|)")
            pr.println("Feature            | CleanRaw    | AdvRaw      | DeltaRaw    | Mask   | Violation")
            pr.println("-" * 120)

            for feat_idx in changed_idx:
                feat = FEATURE_NAMES[feat_idx]
                clean_val = float(clean_row[feat_idx])
                adv_feat_val = float(adv_row[feat_idx])
                delta_val = adv_feat_val - clean_val
                mval = float(perturbation_mask[feat_idx])

                messages: List[str] = []
                if feat == "Protocol Type":
                    clean_proto = protocol_status(clean_val, treat_adv=False)
                    adv_proto = protocol_status(adv_feat_val, treat_adv=True)
                    messages.append(f"Protocol clean={clean_proto}, adv={adv_proto}")
                messages.extend(manual_feat_msgs.get(feat, []))
                if not messages and feat in MANUAL_CHECK_TARGETS:
                    messages.append("-")

                violation_text = " | ".join(messages) if messages else "-"

                pr.println(
                    f"{feat:<18} | {_fmt_label(clean_val):>10} | {_fmt_label(adv_feat_val):>10} | {_fmt_label(delta_val):>10} | {mask_label(mval):<6} | {violation_text}"
                )

                all_exhibit_rows.append(
                    {
                        "attack_config": attack_data.attack_config,
                        "sample_index": int(sample_idx),
                        "feature_name": feat,
                        "original_raw": float(clean_val),
                        "perturbed_raw": float(adv_feat_val),
                        "delta_raw": float(delta_val),
                        "mask_value": float(mval),
                        "violation": violation_text,
                        "true_class": true_label,
                        "clean_pred": clean_pred_label,
                        "adv_pred": adv_pred_label,
                    }
                )

            pr.println("Validation summary")
            pr.println(
                f"Adversarial sample valid: {'YES' if bool(adv_valid[sample_idx]) else 'NO'}; total violated rules: {len(violated_rules)}"
            )
            pr.println("Violated rules list")
            if violated_rules:
                for rule in violated_rules:
                    pr.println(f"  {rule}")
            else:
                pr.println("  None")
            if manual_global:
                pr.println("Manual global checks")
                for msg in manual_global:
                    pr.println(f"  {msg}")

            pr.println("Frozen-feature check (mask==0.0)")
            frozen_changed = 0
            for j in frozen_idx:
                diff = float(delta_row_raw[int(j)])
                status = "changed" if abs(diff) > 0.001 else "unchanged"
                if status == "changed":
                    frozen_changed += 1
                pr.println(f"  {FEATURE_NAMES[int(j)]}: {status} (delta={diff:.6f})")
            pr.println(f"Frozen features changed: {frozen_changed}/{len(frozen_idx)}")
            pr.println()

        summary_rows.append(
            {
                "attack_config": attack_data.attack_config,
                "clean_acc": clean_acc,
                "asr_raw": asr_raw,
                "validity_rate": validity_rate_flipped,
                "asr_valid": asr_valid,
                "total_flipped": n_flipped,
                "most_violated_rule": most_viol_rule,
                "pct_violated": pct_viol,
                "mean_linf_raw": float(linf_raw[candidate_mask].mean()) if n_flipped > 0 else 0.0,
                "mean_l2_raw": float(l2_raw[candidate_mask].mean()) if n_flipped > 0 else 0.0,
                "median_features_changed": float(np.median(changed_count[candidate_mask])) if n_flipped > 0 else 0.0,
            }
        )

    pr.close()

    with OUT_EXHIBITS_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "attack_config",
                "sample_index",
                "feature_name",
                "original_raw",
                "perturbed_raw",
                "delta_raw",
                "mask_value",
                "violation",
                "true_class",
                "clean_pred",
                "adv_pred",
            ],
        )
        writer.writeheader()
        writer.writerows(all_exhibit_rows)

    with OUT_SUMMARY_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "attack_config",
                "clean_acc",
                "asr_raw",
                "validity_rate",
                "asr_valid",
                "total_flipped",
                "most_violated_rule",
                "pct_violated",
                "mean_linf_raw",
                "mean_l2_raw",
                "median_features_changed",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print("=" * 120)
    print("Export complete")
    print(f"Text report: {OUT_TEXT}")
    print(f"Exhibit CSV: {OUT_EXHIBITS_CSV}")
    print(f"Summary CSV: {OUT_SUMMARY_CSV}")
    print(f"Rows in sample_exhibits_all.csv: {len(all_exhibit_rows)}")
    print(f"Rows in attack_summaries.csv: {len(summary_rows)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate adversarial sample exhibits and summaries.")
    parser.add_argument(
        "--max-samples-per-attack",
        type=int,
        default=5,
        help="Maximum number of selected exhibits per attack configuration.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Reserved for future stochastic tie-breaks; currently deterministic selection is used.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    _ = args.seed
    run(args)