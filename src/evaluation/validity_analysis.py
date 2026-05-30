"""
Utilities for inverse-transforming and validating adversarial attack outputs.

This module bridges scaled attack artifacts (.npz) and domain validation
in raw CICIoT2023 feature space.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.attack.validator import validate_batch


def _to_python_scalar(value: Any) -> Any:
    if isinstance(value, np.ndarray) and value.shape == ():
        return value.item()
    return value


def parse_attack_filename(npz_path: str) -> Dict[str, Any]:
    """Parse attack filename like attack_binary_pgd_0.30.npz into metadata."""
    stem = Path(npz_path).stem
    parts = stem.split("_")
    if len(parts) < 4 or parts[0] != "attack":
        raise ValueError(f"Unexpected attack file name format: {Path(npz_path).name}")

    # Support model tags that include underscores by taking the last two tokens
    # as attack and eps, with the rest forming the model name. Restart-aware
    # files add a trailing token such as r10.
    num_restarts: Optional[int] = None
    if parts[-1].lower().startswith("r") and parts[-1][1:].isdigit():
        num_restarts = int(parts[-1][1:])
        eps_str = parts[-2]
        attack_name = parts[-3].lower()
        model_name = "_".join(parts[1:-3])
    else:
        eps_str = parts[-1]
        attack_name = parts[-2].lower()
        model_name = "_".join(parts[1:-2])
    if not model_name:
        raise ValueError(f"Missing model name in attack file: {Path(npz_path).name}")
    eps = float(eps_str) if eps_str.upper() != "N/A" else 0.0

    return {
        "model": model_name,
        "attack": attack_name,
        "eps": eps,
        "eps_label": eps_str,
        "num_restarts": num_restarts,
        "file_stem": stem,
    }


def inverse_transform_results(npz_path: str, scaler_path: str) -> Dict[str, Any]:
    """
    Load attack result arrays and inverse-transform X_clean/X_adv to raw space.

    Returns a dictionary containing the original arrays and two additional keys:
    X_clean_raw and X_adv_raw.
    """
    with np.load(npz_path, allow_pickle=True) as data:
        result: Dict[str, Any] = {k: data[k] for k in data.files}

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    x_clean = np.asarray(result["X_clean"], dtype=np.float32)
    x_adv = np.asarray(result["X_adv"], dtype=np.float32)

    x_clean_raw = scaler.inverse_transform(x_clean)
    x_adv_raw = scaler.inverse_transform(x_adv)

    for key in ["attack_name", "eps"]:
        if key in result:
            result[key] = _to_python_scalar(result[key])

    result["X_clean_raw"] = np.asarray(x_clean_raw, dtype=np.float32)
    result["X_adv_raw"] = np.asarray(x_adv_raw, dtype=np.float32)
    return result


def validate_feature_matrix(X_raw: np.ndarray, feature_names: List[str]) -> Dict[str, Any]:
    """Run rule validation and package per-sample and per-rule outputs."""
    val_result = validate_batch(X_raw, feature_names)

    per_rule_violation_counts = {
        rule: int(mask.sum()) for rule, mask in val_result.violations_per_rule.items()
    }

    violation_details: List[Dict[str, Any]] = []
    all_rules = list(val_result.violations_per_rule.keys())
    for idx in range(val_result.n_samples):
        violated = [rule for rule in all_rules if bool(val_result.violations_per_rule[rule][idx])]
        violation_details.append(
            {
                "sample_index": int(idx),
                "violated_rules": violated,
                "n_violations": int(len(violated)),
            }
        )

    return {
        "validity_rate": float(val_result.validity_rate),
        "per_rule_violation_counts": per_rule_violation_counts,
        "per_sample_valid": np.asarray(val_result.overall_valid, dtype=bool),
        "violation_details": violation_details,
        "per_rule_violation_rate": val_result.per_rule_violation_rate(),
    }


def validate_adversarial_examples(X_adv_raw: np.ndarray, feature_names: List[str]) -> Dict[str, Any]:
    """
    Validate adversarial samples in raw feature space.

    Returns validity rate, per-rule violation counts, per-sample validity,
    and per-sample violated rule details.
    """
    return validate_feature_matrix(X_adv_raw, feature_names)


def compute_asr_valid(result: Dict[str, Any], per_sample_valid: np.ndarray) -> Dict[str, Any]:
    """
    Compute ASR_valid and related ratios.

    ASR_valid = (# clean-correct and adv-misclassified and adv-valid) / (# clean-correct)
    """
    y_true = np.asarray(result["y_true"], dtype=np.int64)
    y_pred_clean = np.asarray(result["y_pred_clean"], dtype=np.int64)
    y_pred_adv = np.asarray(result["y_pred_adv"], dtype=np.int64)
    valid = np.asarray(per_sample_valid, dtype=bool)

    if len(valid) != len(y_true):
        raise ValueError(
            f"Length mismatch: per_sample_valid={len(valid)}, y_true={len(y_true)}"
        )

    clean_correct = y_pred_clean == y_true
    successful_attack = clean_correct & (y_pred_adv != y_true)
    successful_and_valid = successful_attack & valid

    clean_correct_count = int(clean_correct.sum())
    success_count = int(successful_attack.sum())
    valid_success_count = int(successful_and_valid.sum())

    asr_raw = float(success_count / clean_correct_count) if clean_correct_count > 0 else 0.0
    asr_valid = float(valid_success_count / clean_correct_count) if clean_correct_count > 0 else 0.0
    validity_rate_among_successful = (
        float(valid_success_count / success_count) if success_count > 0 else 0.0
    )

    return {
        "samples_originally_correct": clean_correct_count,
        "samples_successful_raw": success_count,
        "samples_successful_valid": valid_success_count,
        "attack_success_rate_raw": asr_raw,
        "attack_success_rate_valid": asr_valid,
        "validity_rate_among_successful": validity_rate_among_successful,
        "clean_correct_mask": clean_correct,
        "successful_attack_mask": successful_attack,
        "successful_valid_mask": successful_and_valid,
    }


def _rules_to_features(rule_name: str) -> List[str]:
    if rule_name.startswith("R_nonneg_"):
        return [rule_name.replace("R_nonneg_", "", 1)]
    if rule_name.startswith("R_binary_"):
        return [rule_name.replace("R_binary_", "", 1)]

    mapping = {
        "R_protocol_valid": ["Protocol Type"],
        "R_proto_tcp": ["Protocol Type", "TCP"],
        "R_proto_udp": ["Protocol Type", "UDP"],
        "R_proto_icmp": ["Protocol Type", "ICMP"],
        "R_proto_igmp": ["Protocol Type", "IGMP"],
        "R_min_leq_max": ["Min", "Max"],
        "R_avg_in_range": ["Min", "AVG", "Max"],
        "R_var_eq_std_sq": ["Std", "Variance"],
        "R_ttl_range": ["Time_To_Live"],
        "R_pkts_positive": ["Number"],
        "R_pkts_integer": ["Number"],
    }
    return mapping.get(rule_name, [])


def _describe_violation(rule_name: str, adv_row: pd.Series) -> str:
    if rule_name == "R_protocol_valid":
        val = float(adv_row.get("Protocol Type", np.nan))
        return (
            "Protocol Type must map to allowed values {0,1,2,6,17,47}; "
            f"got {val:.4f}"
        )
    if rule_name.startswith("R_binary_"):
        feat = rule_name.replace("R_binary_", "", 1)
        val = float(adv_row.get(feat, np.nan))
        return f"{feat} must be binary (0/1); got {val:.4f}"
    if rule_name.startswith("R_nonneg_"):
        feat = rule_name.replace("R_nonneg_", "", 1)
        val = float(adv_row.get(feat, np.nan))
        return f"{feat} must be non-negative; got {val:.4f}"
    if rule_name == "R_min_leq_max":
        return (
            f"Min must be <= Max; got Min={float(adv_row.get('Min', np.nan)):.4f}, "
            f"Max={float(adv_row.get('Max', np.nan)):.4f}"
        )
    if rule_name == "R_avg_in_range":
        return (
            "AVG must lie in [Min, Max]; got "
            f"Min={float(adv_row.get('Min', np.nan)):.4f}, "
            f"AVG={float(adv_row.get('AVG', np.nan)):.4f}, "
            f"Max={float(adv_row.get('Max', np.nan)):.4f}"
        )
    if rule_name == "R_var_eq_std_sq":
        std_val = float(adv_row.get("Std", np.nan))
        var_val = float(adv_row.get("Variance", np.nan))
        return (
            f"Variance should match Std^2 (within tolerance); "
            f"Std={std_val:.4f}, Variance={var_val:.4f}"
        )
    if rule_name == "R_ttl_range":
        ttl = float(adv_row.get("Time_To_Live", np.nan))
        return f"Time_To_Live must be in [0,255]; got {ttl:.4f}"
    if rule_name == "R_pkts_positive":
        n = float(adv_row.get("Number", np.nan))
        return f"Number (packet count) must be >= 1; got {n:.4f}"
    if rule_name == "R_pkts_integer":
        n = float(adv_row.get("Number", np.nan))
        return f"Number (packet count) must be integer-like; got {n:.4f}"
    if rule_name == "R_proto_tcp":
        return "TCP indicator implies Protocol Type should be 6"
    if rule_name == "R_proto_udp":
        return "UDP indicator implies Protocol Type should be 17"
    if rule_name == "R_proto_icmp":
        return "ICMP indicator implies Protocol Type should be 1"
    if rule_name == "R_proto_igmp":
        return "IGMP indicator implies Protocol Type should be 2"
    return rule_name


def generate_impossible_traffic_exhibit(
    X_clean_raw: np.ndarray,
    X_adv_raw: np.ndarray,
    feature_names: List[str],
    violation_details: List[Dict[str, Any]],
    y_pred_clean: np.ndarray,
    y_pred_adv: np.ndarray,
    n_examples: int = 10,
    y_true: Optional[np.ndarray] = None,
    significant_delta: float = 0.01,
) -> pd.DataFrame:
    """
    Build a feature-level table for invalid successful adversarial examples.

    Includes changed features (|delta| > significant_delta) and features tied
    to violated rules.
    """
    df_clean = pd.DataFrame(X_clean_raw, columns=feature_names)
    df_adv = pd.DataFrame(X_adv_raw, columns=feature_names)

    y_pred_clean = np.asarray(y_pred_clean, dtype=np.int64)
    y_pred_adv = np.asarray(y_pred_adv, dtype=np.int64)

    if y_true is not None:
        y_true_arr = np.asarray(y_true, dtype=np.int64)
        clean_correct = y_pred_clean == y_true_arr
        successful_flip = clean_correct & (y_pred_adv != y_true_arr)
    else:
        successful_flip = y_pred_adv != y_pred_clean

    invalid_mask = np.array(
        [len(item.get("violated_rules", [])) > 0 for item in violation_details], dtype=bool
    )

    selected_indices = np.where(successful_flip & invalid_mask)[0][:n_examples]
    rows: List[Dict[str, Any]] = []

    for rank, sample_idx in enumerate(selected_indices, start=1):
        violated_rules = violation_details[sample_idx].get("violated_rules", [])
        violated_features = set()
        for rule in violated_rules:
            violated_features.update(_rules_to_features(rule))

        changed_features = []
        for feat in feature_names:
            delta = float(df_adv.at[sample_idx, feat] - df_clean.at[sample_idx, feat])
            if abs(delta) > significant_delta or feat in violated_features:
                changed_features.append(feat)

        for feat in changed_features:
            clean_val = float(df_clean.at[sample_idx, feat])
            adv_val = float(df_adv.at[sample_idx, feat])
            delta = adv_val - clean_val

            feat_rules = [r for r in violated_rules if feat in _rules_to_features(r)]
            descriptions = [_describe_violation(r, df_adv.loc[sample_idx]) for r in feat_rules]

            rows.append(
                {
                    "example_rank": int(rank),
                    "sample_index": int(sample_idx),
                    "clean_prediction": int(y_pred_clean[sample_idx]),
                    "adv_prediction": int(y_pred_adv[sample_idx]),
                    "feature": feat,
                    "clean_value": round(clean_val, 4),
                    "adv_value": round(adv_val, 4),
                    "delta": round(delta, 4),
                    "violates_rule": bool(len(feat_rules) > 0),
                    "violated_rules": "; ".join(feat_rules),
                    "violation_description": " | ".join(descriptions),
                }
            )

    exhibit_df = pd.DataFrame(rows)
    if exhibit_df.empty:
        return exhibit_df

    exhibit_df = exhibit_df.sort_values(
        by=["example_rank", "violates_rule", "feature"],
        ascending=[True, False, True],
    ).reset_index(drop=True)
    return exhibit_df
