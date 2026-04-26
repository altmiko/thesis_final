"""
Generate a compact adversarial exhibit report for binary framing.

Outputs 3 models x 3 attacks x 5 samples each (45 exhibits), reusing existing
attack artifacts where possible and generating missing artifacts when needed.
"""

from __future__ import annotations

import argparse
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.model_selection import train_test_split
import torch

from adversarial_attacks import load_model
from adversarial_attacks import run_attack
from feature_groups import BINARY_FEATURES, FEATURE_NAMES
from validator import VALID_PROTOCOLS
from validator import validate_batch


ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "data" / "processed"
ATTACKS_DIR = ROOT / "results" / "attacks"
OUT_PATH = ATTACKS_DIR / "compact_exhibits.txt"


ATTACKS: Sequence[Tuple[str, Optional[float]]] = (
    ("fgsm", 0.30),
    ("pgd", 0.30),
    ("cw", None),
)

EPS_LABEL = {
    ("fgsm", 0.30): "FGSM epsilon=0.30",
    ("pgd", 0.30): "PGD epsilon=0.30",
    ("cw", None): "C&W",
}

SUMMARY_ATTACK_LABEL = {
    ("fgsm", 0.30): "FGSM 0.30",
    ("pgd", 0.30): "PGD 0.30",
    ("cw", None): "C&W",
}

MODEL_LABEL = {
    "mlp_binary": "MLP",
    "serial_binary": "CNN-LSTM",
    "cnn_binary": "CNN",
    "lstm_binary": "LSTM",
    "dualpath_binary": "DUALPATH",
}


@dataclass
class ComboStats:
    model_tag: str
    attack_name: str
    eps: Optional[float]
    asr_raw: float
    validity: float
    asr_valid: float
    flipped_count: int
    valid_flipped_count: int
    avg_violations: float
    avg_frozen_changed: float
    frozen_total: int


def fmt_num(x: float) -> str:
    if np.isnan(x):
        return "nan"
    s = f"{x:.2f}"
    if s.endswith("00"):
        return f"{x:.1f}"
    if s.endswith("0"):
        return s[:-1]
    return s


def eps_file_label(eps: Optional[float]) -> str:
    if eps is None:
        return "0"
    return f"{eps:.2f}"


def binary_label(y: int) -> str:
    return "Benign" if int(y) == 0 else "Attack"


def select_three_models() -> List[str]:
    existing = {p.stem for p in MODELS_DIR.glob("*_binary.pt")}
    if not existing:
        raise FileNotFoundError("No binary checkpoints found in models directory")

    selected: List[str] = []

    def pick(preferred: Sequence[str]) -> Optional[str]:
        for tag in preferred:
            if tag in existing and tag not in selected:
                return tag
        return None

    first = pick(["mlp_binary", "serial_binary", "cnn_binary", "lstm_binary", "dualpath_binary"])
    if first is None:
        raise RuntimeError("Could not choose first model")
    selected.append(first)

    second = pick(["serial_binary", "cnn_binary", "dualpath_binary", "lstm_binary", "mlp_binary"])
    if second is None:
        raise RuntimeError("Could not choose second model")
    selected.append(second)

    third = pick(["lstm_binary", "dualpath_binary", "cnn_binary", "serial_binary", "mlp_binary"])
    if third is None:
        for cand in sorted(existing):
            if cand not in selected:
                third = cand
                break
    if third is None:
        raise RuntimeError("Could not choose third model")
    selected.append(third)

    return selected


def feature_mask_label(v: float) -> str:
    if np.isclose(v, 0.0):
        return "FROZEN"
    if np.isclose(v, 0.3):
        return "0.3"
    if np.isclose(v, 1.0):
        return "1.0"
    return fmt_num(v)


def build_feature_reference(mask: np.ndarray, max_line_len: int = 130) -> List[str]:
    tokens = []
    for feat, m in zip(FEATURE_NAMES, mask):
        token = f"{feat.replace(' ', '_')}[{feature_mask_label(float(m))}]"
        tokens.append(token)

    lines: List[str] = [f"FEATURES ({len(FEATURE_NAMES)}):"]
    current = ""
    for tok in tokens:
        candidate = tok if not current else (current + " " + tok)
        if len(candidate) > max_line_len:
            lines.append(current)
            current = tok
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def stratified_subsample_indices(y: np.ndarray, sample_size: int, seed: int) -> np.ndarray:
    if sample_size > len(y):
        raise ValueError(f"sample_size {sample_size} exceeds dataset size {len(y)}")
    idx_all = np.arange(len(y))
    idx_keep, _ = train_test_split(
        idx_all,
        train_size=sample_size,
        random_state=seed,
        stratify=y,
        shuffle=True,
    )
    return idx_keep


def resolve_attack_npz(model_tag: str, attack_name: str, eps: Optional[float]) -> Tuple[Path, List[Path]]:
    eps_lbl = eps_file_label(eps)
    preferred = ATTACKS_DIR / f"attack_{model_tag}_{attack_name}_{eps_lbl}.npz"

    candidates = [preferred]
    if model_tag == "mlp_binary":
        candidates.append(ATTACKS_DIR / f"attack_binary_{attack_name}_{eps_lbl}.npz")

    for c in candidates:
        if c.exists():
            return c, candidates
    return preferred, candidates


def ensure_attack_artifact(
    model_tag: str,
    attack_name: str,
    eps: Optional[float],
    x_sub: np.ndarray,
    y_sub: np.ndarray,
    num_features: int,
    device: str,
    batch_size: int,
) -> Path:
    existing, _ = resolve_attack_npz(model_tag, attack_name, eps)
    if existing.exists():
        return existing

    ckpt = MODELS_DIR / f"{model_tag}.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt}")

    model = load_model(
        model_path=str(ckpt),
        num_features=num_features,
        num_classes=2,
        device=device,
    )
    use_cudnn_workaround = ("lstm" in model_tag) or ("serial" in model_tag)
    if use_cudnn_workaround:
        prev_cudnn = torch.backends.cudnn.enabled
        torch.backends.cudnn.enabled = False
        try:
            result = run_attack(
                model=model,
                X=x_sub,
                y=y_sub,
                attack_name=attack_name,
                eps=0.0 if eps is None else float(eps),
                batch_size=batch_size,
                device=device,
            )
        finally:
            torch.backends.cudnn.enabled = prev_cudnn
    else:
        result = run_attack(
            model=model,
            X=x_sub,
            y=y_sub,
            attack_name=attack_name,
            eps=0.0 if eps is None else float(eps),
            batch_size=batch_size,
            device=device,
        )

    out_path, _ = resolve_attack_npz(model_tag, attack_name, eps)
    np.savez_compressed(
        out_path,
        X_adv=result["X_adv"],
        X_clean=result["X_clean"],
        y_true=result["y_true"],
        y_pred_clean=result["y_pred_clean"],
        y_pred_adv=result["y_pred_adv"],
        attack_name=result["attack_name"],
        eps=result["eps"],
    )
    return out_path


def load_attack_npz(npz_path: Path, scaler) -> Dict[str, np.ndarray]:
    with np.load(npz_path, allow_pickle=True) as data:
        for k in ["X_clean", "X_adv", "y_true", "y_pred_clean", "y_pred_adv"]:
            if k not in data.files:
                raise KeyError(f"Missing key {k} in {npz_path.name}")
        x_clean = np.asarray(data["X_clean"], dtype=np.float32)
        x_adv = np.asarray(data["X_adv"], dtype=np.float32)
        y_true = np.asarray(data["y_true"], dtype=np.int64)
        y_pred_clean = np.asarray(data["y_pred_clean"], dtype=np.int64)
        y_pred_adv = np.asarray(data["y_pred_adv"], dtype=np.int64)

    if x_clean.shape != x_adv.shape:
        raise ValueError(f"Shape mismatch in {npz_path.name}")
    if x_clean.ndim != 2 or x_clean.shape[1] != len(FEATURE_NAMES):
        raise ValueError(f"Unexpected feature matrix shape in {npz_path.name}: {x_clean.shape}")
    if len(y_true) != len(x_clean):
        raise ValueError(f"Length mismatch in {npz_path.name}")

    x_clean_raw = np.asarray(scaler.inverse_transform(x_clean), dtype=np.float32)
    x_adv_raw = np.asarray(scaler.inverse_transform(x_adv), dtype=np.float32)
    return {
        "x_clean": x_clean,
        "x_adv": x_adv,
        "x_clean_raw": x_clean_raw,
        "x_adv_raw": x_adv_raw,
        "y_true": y_true,
        "y_pred_clean": y_pred_clean,
        "y_pred_adv": y_pred_adv,
    }


def map_rule_to_features(rule: str) -> List[str]:
    if rule.startswith("R_nonneg_"):
        return [rule.replace("R_nonneg_", "", 1)]
    if rule.startswith("R_binary_"):
        return [rule.replace("R_binary_", "", 1)]

    mapping = {
        "R_protocol_valid": ["Protocol Type"],
        "R_proto_tcp": ["Protocol Type", "TCP"],
        "R_proto_udp": ["Protocol Type", "UDP"],
        "R_proto_icmp": ["Protocol Type", "ICMP"],
        "R_proto_igmp": ["Protocol Type", "IGMP"],
        "R_min_leq_max": ["Min", "Max"],
        "R_avg_in_range": ["AVG", "Min", "Max"],
        "R_var_eq_std_sq": ["Variance", "Std"],
        "R_ttl_range": ["Time_To_Live"],
        "R_pkts_positive": ["Number"],
        "R_pkts_integer": ["Number"],
    }
    return mapping.get(rule, [])


def add_tag(tags: Dict[str, set], feat: str, tag: str) -> None:
    tags.setdefault(feat, set()).add(tag)


def feature_violation_tags(
    idx_map: Dict[str, int],
    clean_row: np.ndarray,
    adv_row: np.ndarray,
    violated_rules: List[str],
) -> Dict[str, set]:
    tags: Dict[str, set] = {}

    for rule in violated_rules:
        feats = map_rule_to_features(rule)
        for feat in feats:
            if rule.startswith("R_nonneg_"):
                add_tag(tags, feat, "NEGATIVE")
            elif rule.startswith("R_binary_"):
                add_tag(tags, feat, "FRACTIONAL BINARY")
            elif rule == "R_protocol_valid":
                add_tag(tags, feat, "NOT VALID IANA PROTOCOL")
            elif rule in {"R_proto_tcp", "R_proto_udp", "R_proto_icmp", "R_proto_igmp"}:
                add_tag(tags, feat, "PROTO MISMATCH")
            elif rule == "R_min_leq_max":
                add_tag(tags, "Min", "MIN>MAX")
                add_tag(tags, "Max", "MIN>MAX")
            elif rule == "R_avg_in_range":
                add_tag(tags, "AVG", "AVG OUTSIDE RANGE")
            elif rule == "R_var_eq_std_sq":
                add_tag(tags, "Variance", "VAR!=STD^2")
                add_tag(tags, "Std", "VAR!=STD^2")
            elif rule == "R_ttl_range":
                add_tag(tags, feat, "TTL OUT OF RANGE")
            elif rule == "R_pkts_positive":
                add_tag(tags, "Number", "NEGATIVE")
            elif rule == "R_pkts_integer":
                add_tag(tags, "Number", "NON-INTEGER")

    for feat in BINARY_FEATURES:
        j = idx_map[feat]
        v = float(adv_row[j])
        if abs(v - round(v)) > 0.01:
            add_tag(tags, feat, "FRACTIONAL BINARY")
        elif int(round(v)) not in (0, 1):
            add_tag(tags, feat, "NOT IN {0,1}")

    proto = float(adv_row[idx_map["Protocol Type"]])
    p_round = int(round(proto))
    if abs(proto - p_round) > 0.01:
        add_tag(tags, "Protocol Type", "NOT VALID IANA PROTOCOL")
    elif p_round not in VALID_PROTOCOLS:
        add_tag(tags, "Protocol Type", "NOT VALID IANA PROTOCOL")

    for feat in [
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
        "Number",
    ]:
        j = idx_map[feat]
        v = float(adv_row[j])
        if abs(v - round(v)) > 0.01:
            add_tag(tags, feat, "NON-INTEGER")
        if v < 0:
            add_tag(tags, feat, "NEGATIVE")

    min_v = float(adv_row[idx_map["Min"]])
    max_v = float(adv_row[idx_map["Max"]])
    avg_v = float(adv_row[idx_map["AVG"]])
    if min_v > max_v:
        add_tag(tags, "Min", f"MIN>MAX (Max={fmt_num(max_v)})")
        add_tag(tags, "Max", f"MIN>MAX (Min={fmt_num(min_v)})")
    if avg_v < min_v or avg_v > max_v:
        add_tag(tags, "AVG", "AVG OUTSIDE RANGE")

    std_v = float(adv_row[idx_map["Std"]])
    var_v = float(adv_row[idx_map["Variance"]])
    rel_err = abs(var_v - std_v * std_v) / (std_v * std_v + 1e-8)
    if rel_err > 0.05:
        add_tag(tags, "Variance", "VAR!=STD^2")
        add_tag(tags, "Std", "VAR!=STD^2")

    return tags


def sample_indices_to_show(
    y_true: np.ndarray,
    y_pred_clean: np.ndarray,
    y_pred_adv: np.ndarray,
    adv_valid: np.ndarray,
    n_violations: np.ndarray,
    linf_raw: np.ndarray,
    target_n: int,
) -> List[int]:
    clean_correct = y_pred_clean == y_true
    flipped = y_pred_adv != y_true
    invalid_adv = ~adv_valid

    cand = np.where(clean_correct & flipped & invalid_adv)[0]
    if len(cand) == 0:
        cand = np.where(clean_correct & flipped)[0]
    rows = [(int(i), int(n_violations[i]), float(linf_raw[i]), int(y_true[i])) for i in cand]
    rows.sort(key=lambda x: (-x[1], -x[2], x[0]))

    selected: List[int] = []
    covered = set()
    for i, _, _, cls in rows:
        if cls in covered:
            continue
        selected.append(i)
        covered.add(cls)
        if len(selected) >= target_n:
            return selected

    for i, _, _, _ in rows:
        if i in selected:
            continue
        selected.append(i)
        if len(selected) >= target_n:
            break

    return selected


def active_transport_feature(idx_map: Dict[str, int], clean_row: np.ndarray) -> List[str]:
    tcp = float(clean_row[idx_map["TCP"]])
    udp = float(clean_row[idx_map["UDP"]])
    if round(tcp) == 1 and round(udp) != 1:
        return ["TCP"]
    if round(udp) == 1 and round(tcp) != 1:
        return ["UDP"]
    return ["TCP", "UDP"]


def build_compact_lines(
    model_tag: str,
    attack_name: str,
    eps: Optional[float],
    attack_data: Dict[str, np.ndarray],
    perturb_mask: np.ndarray,
    max_violation_features: int,
) -> Tuple[List[str], ComboStats]:
    idx_map = {f: i for i, f in enumerate(FEATURE_NAMES)}

    x_clean_raw = attack_data["x_clean_raw"]
    x_adv_raw = attack_data["x_adv_raw"]
    y_true = attack_data["y_true"]
    y_pred_clean = attack_data["y_pred_clean"]
    y_pred_adv = attack_data["y_pred_adv"]

    val_adv = validate_batch(x_adv_raw, FEATURE_NAMES)
    adv_valid = val_adv.overall_valid

    clean_correct = y_pred_clean == y_true
    flipped = y_pred_adv != y_true
    n_clean_correct = int(clean_correct.sum())
    n_flipped = int((clean_correct & flipped).sum())
    n_valid_flipped = int((clean_correct & flipped & adv_valid).sum())

    asr_raw = float(n_flipped / n_clean_correct) if n_clean_correct > 0 else 0.0
    validity = float(n_valid_flipped / n_flipped) if n_flipped > 0 else 0.0
    asr_valid = float(n_valid_flipped / n_clean_correct) if n_clean_correct > 0 else 0.0

    delta_raw = x_adv_raw - x_clean_raw
    linf_raw = np.max(np.abs(delta_raw), axis=1)

    n_viol = np.zeros(len(y_true), dtype=np.int64)
    for mask in val_adv.violations_per_rule.values():
        n_viol += np.asarray(mask, dtype=np.int64)

    chosen = sample_indices_to_show(
        y_true=y_true,
        y_pred_clean=y_pred_clean,
        y_pred_adv=y_pred_adv,
        adv_valid=adv_valid,
        n_violations=n_viol,
        linf_raw=linf_raw,
        target_n=5,
    )

    lines: List[str] = []
    model_label = MODEL_LABEL.get(model_tag, model_tag.upper())
    eps_title = EPS_LABEL[(attack_name, eps)]
    lines.append(
        f"--- {model_label} + {eps_title}: ASR_raw={asr_raw*100:.1f}%, Validity={validity*100:.1f}%, "
        f"ASR_valid={asr_valid*100:.1f}% ({n_flipped} flipped, {n_valid_flipped} valid) ---"
    )

    frozen_idx = np.where(np.isclose(perturb_mask, 0.0))[0]
    frozen_total = int(len(frozen_idx))

    sample_viols: List[int] = []
    sample_frozen_changed: List[int] = []

    for rank in range(5):
        if rank < len(chosen):
            i = chosen[rank]
        else:
            lines.append(f"[{model_label} | {eps_title} | Sample {rank+1}/5] unavailable")
            continue

        clean_row = x_clean_raw[i]
        adv_row = x_adv_raw[i]
        deltas = adv_row - clean_row

        violated_rules = [
            rule for rule, mask in val_adv.violations_per_rule.items() if bool(np.asarray(mask, dtype=bool)[i])
        ]
        tags = feature_violation_tags(idx_map, clean_row, adv_row, violated_rules)

        violating_feats = [f for f in FEATURE_NAMES if tags.get(f)]
        violating_feats.sort(key=lambda f: -abs(float(deltas[idx_map[f]])))
        picked_viol = violating_feats[:max_violation_features]
        hidden = max(0, len(violating_feats) - len(picked_viol))

        always = ["Protocol Type"] + active_transport_feature(idx_map, clean_row) + [
            "Min",
            "Max",
            "AVG",
            "Variance",
            "Std",
        ]

        show_feats: List[str] = []
        for f in always:
            if f not in show_feats:
                show_feats.append(f)
        for f in picked_viol:
            if f not in show_feats:
                show_feats.append(f)

        frozen_changed = int((np.abs(deltas[frozen_idx]) > 0.001).sum())

        sample_viols.append(len(violating_feats))
        sample_frozen_changed.append(frozen_changed)

        true_lbl = binary_label(int(y_true[i]))
        pred_lbl = binary_label(int(y_pred_adv[i]))
        flip_tag = "FLIPPED" if y_pred_adv[i] != y_true[i] else "NOT FLIPPED"

        shown_viol_count = sum(1 for f in show_feats if tags.get(f))
        lines.append(
            f"[{model_label} | {eps_title} | Sample {rank+1}/5] "
            f"True: {true_lbl} -> Pred: {pred_lbl} ({flip_tag}) | Linf={fmt_num(float(linf_raw[i]))} | "
            f"Violations: {len(violating_feats)} | Frozen: {frozen_changed}/{frozen_total} | "
            f"Shown: {shown_viol_count}/{len(violating_feats)}"
        )

        for f in show_feats:
            j = idx_map[f]
            c = float(clean_row[j])
            a = float(adv_row[j])

            if f == "Protocol Type":
                cr = int(round(c))
                if cr == 6:
                    c_str = f"{fmt_num(c)} (TCP)"
                elif cr == 17:
                    c_str = f"{fmt_num(c)} (UDP)"
                elif cr == 1:
                    c_str = f"{fmt_num(c)} (ICMP)"
                else:
                    c_str = fmt_num(c)
            else:
                c_str = fmt_num(c)

            viol_str = ", ".join(sorted(tags.get(f, []))) if tags.get(f) else "-"
            lines.append(f"{f} | {c_str} | {fmt_num(a)} | {viol_str}")

    avg_viol = float(np.mean(sample_viols)) if sample_viols else 0.0
    avg_frozen = float(np.mean(sample_frozen_changed)) if sample_frozen_changed else 0.0

    stats = ComboStats(
        model_tag=model_tag,
        attack_name=attack_name,
        eps=eps,
        asr_raw=asr_raw,
        validity=validity,
        asr_valid=asr_valid,
        flipped_count=n_flipped,
        valid_flipped_count=n_valid_flipped,
        avg_violations=avg_viol,
        avg_frozen_changed=avg_frozen,
        frozen_total=frozen_total,
    )
    return lines, stats


def assemble_report(
    models: Sequence[str],
    perturb_mask: np.ndarray,
    combo_payloads: Dict[Tuple[str, str, Optional[float]], Dict[str, np.ndarray]],
    max_violation_features: int,
) -> Tuple[List[str], List[ComboStats]]:
    lines: List[str] = []
    stats_all: List[ComboStats] = []

    lines.extend(build_feature_reference(perturb_mask))
    lines.append("")

    for model_tag in models:
        for attack_name, eps in ATTACKS:
            payload = combo_payloads[(model_tag, attack_name, eps)]
            combo_lines, combo_stats = build_compact_lines(
                model_tag=model_tag,
                attack_name=attack_name,
                eps=eps,
                attack_data=payload,
                perturb_mask=perturb_mask,
                max_violation_features=max_violation_features,
            )
            lines.extend(combo_lines)
            stats_all.append(combo_stats)

    lines.append("SUMMARY")
    lines.append("MODEL | ATTACK | ASR_raw | Validity | ASR_valid | Avg_Violations | Avg_Frozen_Changed")
    for s in stats_all:
        model_lbl = MODEL_LABEL.get(s.model_tag, s.model_tag.upper())
        attack_lbl = SUMMARY_ATTACK_LABEL[(s.attack_name, s.eps)]
        lines.append(
            f"{model_lbl} | {attack_lbl} | {s.asr_raw*100:.1f}% | {s.validity*100:.1f}% | {s.asr_valid*100:.1f}% | "
            f"{s.avg_violations:.1f} | {s.avg_frozen_changed:.1f}/{s.frozen_total}"
        )

    return lines, stats_all


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate compact adversarial exhibits")
    p.add_argument("--sample-size", type=int, default=50000)
    p.add_argument(
        "--cw-sample-size",
        type=int,
        default=5000,
        help="Optional reduced CW sample size using same stratified seed logic.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--line-budget", type=int, default=400)
    p.add_argument("--max-violation-features", type=int, default=15)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    ATTACKS_DIR.mkdir(parents=True, exist_ok=True)

    with open(DATA_DIR / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    perturb_mask = np.load(DATA_DIR / "perturbation_mask.npy")
    if len(perturb_mask) != len(FEATURE_NAMES):
        raise ValueError("perturbation mask length mismatch")

    models = select_three_models()
    print(f"Selected models: {models}")

    x_test = np.load(DATA_DIR / "X_test.npy", mmap_mode="r")
    y_test_bin = np.load(DATA_DIR / "y_test_bin.npy")
    idx_main = stratified_subsample_indices(y_test_bin, sample_size=args.sample_size, seed=args.seed)
    x_sub_main = np.array(x_test[idx_main], dtype=np.float32, copy=True)
    y_sub_main = np.array(y_test_bin[idx_main], dtype=np.int64, copy=True)

    if args.cw_sample_size is not None:
        idx_cw = stratified_subsample_indices(y_test_bin, sample_size=args.cw_sample_size, seed=args.seed)
        x_sub_cw = np.array(x_test[idx_cw], dtype=np.float32, copy=True)
        y_sub_cw = np.array(y_test_bin[idx_cw], dtype=np.int64, copy=True)
    else:
        x_sub_cw = x_sub_main
        y_sub_cw = y_sub_main

    combo_payloads: Dict[Tuple[str, str, Optional[float]], Dict[str, np.ndarray]] = {}

    for model_tag in models:
        for attack_name, eps in ATTACKS:
            out_npz = ensure_attack_artifact(
                model_tag=model_tag,
                attack_name=attack_name,
                eps=eps,
                x_sub=x_sub_cw if attack_name == "cw" else x_sub_main,
                y_sub=y_sub_cw if attack_name == "cw" else y_sub_main,
                num_features=x_sub_main.shape[1],
                device=args.device,
                batch_size=args.batch_size,
            )
            print(f"Using attack artifact: {out_npz.name}")
            combo_payloads[(model_tag, attack_name, eps)] = load_attack_npz(out_npz, scaler)

    caps = [args.max_violation_features, 12, 10, 8, 6, 5, 4, 3, 2, 1, 0]
    caps = [c for c in caps if c <= args.max_violation_features]
    if not caps:
        caps = [args.max_violation_features]

    chosen_lines: Optional[List[str]] = None
    chosen_cap = caps[0]
    chosen_stats: List[ComboStats] = []
    for cap in caps:
        lines, stats = assemble_report(
            models=models,
            perturb_mask=perturb_mask,
            combo_payloads=combo_payloads,
            max_violation_features=cap,
        )
        chosen_lines = lines
        chosen_cap = cap
        chosen_stats = stats
        if len(lines) <= args.line_budget:
            break

    assert chosen_lines is not None

    text = "\n".join(chosen_lines) + "\n"
    print(text, end="")
    OUT_PATH.write_text(text, encoding="utf-8")

    print(f"Saved: {OUT_PATH}")
    print(f"Total lines: {len(chosen_lines)}")
    print(f"Max violated features shown per sample: {chosen_cap}")
    print(f"Summary rows: {len(chosen_stats)}")


if __name__ == "__main__":
    main()
