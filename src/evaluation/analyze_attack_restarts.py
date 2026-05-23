"""
Analyze restart-based adversarial attack outputs and compute perturbation statistics.

Outputs:
- results/attacks/attack_restart_summary_8class.csv
- results/attacks/feature_perturbation_8class.csv
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from src.preprocessing.feature_groups import FEATURE_NAMES


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results" / "attacks"
DATA_DIR = ROOT / "data" / "processed"

FILE_PATTERN = re.compile(
    r"^attack_(?P<model_tag>mlp|lstm|serial)_8class_(?P<attack>fgsm|pgd|cw)_(?P<eps>[0-9]+(?:\.[0-9]+)?)_r(?P<restarts>[0-9]+)\.npz$"
)


def _safe_mean_std(values: np.ndarray) -> tuple[float, float]:
    if values.size == 0:
        return float("nan"), float("nan")
    return float(values.mean()), float(values.std())


def _load_attack_files() -> List[Path]:
    files: List[Path] = []
    for p in RESULTS_DIR.glob("attack_*_8class_*_r*.npz"):
        if FILE_PATTERN.match(p.name):
            files.append(p)
    files.sort()
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute perturbation stats for 8-class restart attacks.")
    parser.add_argument("--delta-threshold", type=float, default=1e-12)
    parser.add_argument(
        "--num-restarts",
        type=int,
        default=10,
        help="Only include NPZ files with this restart count (default: 10).",
    )
    args = parser.parse_args()

    attack_files = _load_attack_files()
    if not attack_files:
        raise FileNotFoundError("No restart attack NPZ files found under results/attacks.")

    scaler = pd.read_pickle(DATA_DIR / "scaler.pkl")
    if len(FEATURE_NAMES) != 39:
        raise ValueError(f"Expected 39 features, got {len(FEATURE_NAMES)}")

    summary_rows: List[Dict[str, object]] = []
    feature_rows: List[Dict[str, object]] = []

    for path in attack_files:
        m = FILE_PATTERN.match(path.name)
        if m is None:
            continue

        model_tag = str(m.group("model_tag"))
        attack = str(m.group("attack")).upper()
        eps_str = str(m.group("eps"))
        restarts = int(m.group("restarts"))
        if attack in {"PGD", "CW"} and args.num_restarts is not None and restarts != args.num_restarts:
            continue
        if attack == "FGSM" and restarts != 1:
            continue

        with np.load(path, allow_pickle=True) as data:
            x_clean = np.asarray(data["X_clean"], dtype=np.float64)
            x_adv = np.asarray(data["X_adv"], dtype=np.float64)
            y_true = np.asarray(data["y_true"], dtype=np.int64)
            y_pred_clean = np.asarray(data["y_pred_clean"], dtype=np.int64)
            y_pred_adv = np.asarray(data["y_pred_adv"], dtype=np.int64)
            model_name = str(data["model_name"]) if "model_name" in data.files else model_tag.upper()
            sample_size = int(data["sample_size"]) if "sample_size" in data.files else int(len(y_true))

        if x_clean.shape[1] != len(FEATURE_NAMES):
            raise ValueError(f"Feature mismatch in {path.name}: {x_clean.shape[1]} vs {len(FEATURE_NAMES)}")

        clean_correct = y_pred_clean == y_true
        flipped = clean_correct & (y_pred_adv != y_true)

        clean_correct_count = int(clean_correct.sum())
        flipped_count = int(flipped.sum())

        clean_acc = float(clean_correct.mean())
        asr_raw = float(flipped_count / clean_correct_count) if clean_correct_count > 0 else 0.0

        delta_scaled = x_adv - x_clean
        linf_scaled = np.max(np.abs(delta_scaled), axis=1)
        l2_scaled = np.linalg.norm(delta_scaled, ord=2, axis=1)

        x_clean_raw = scaler.inverse_transform(x_clean)
        x_adv_raw = scaler.inverse_transform(x_adv)
        delta_raw = x_adv_raw - x_clean_raw

        linf_raw = np.max(np.abs(delta_raw), axis=1)
        l2_raw = np.linalg.norm(delta_raw, ord=2, axis=1)

        mean_linf_scaled, std_linf_scaled = _safe_mean_std(linf_scaled)
        mean_l2_scaled, std_l2_scaled = _safe_mean_std(l2_scaled)
        mean_linf_raw, std_linf_raw = _safe_mean_std(linf_raw)
        mean_l2_raw, std_l2_raw = _safe_mean_std(l2_raw)

        summary_rows.append(
            {
                "model": model_name,
                "model_tag": model_tag,
                "attack": attack,
                "eps": eps_str,
                "sample_size": sample_size,
                "num_restarts": restarts,
                "clean_acc": clean_acc,
                "asr_raw": asr_raw,
                "total_flipped": flipped_count,
                "mean_linf_scaled": mean_linf_scaled,
                "std_linf_scaled": std_linf_scaled,
                "mean_l2_scaled": mean_l2_scaled,
                "std_l2_scaled": std_l2_scaled,
                "mean_linf_raw": mean_linf_raw,
                "std_linf_raw": std_linf_raw,
                "mean_l2_raw": mean_l2_raw,
                "std_l2_raw": std_l2_raw,
                "attack_file": str(path.relative_to(ROOT)),
            }
        )

        abs_delta_raw = np.abs(delta_raw)
        mean_abs = abs_delta_raw.mean(axis=0)
        std_abs = abs_delta_raw.std(axis=0)
        min_abs = abs_delta_raw.min(axis=0)
        max_abs = abs_delta_raw.max(axis=0)

        if flipped_count > 0:
            abs_delta_flipped = abs_delta_raw[flipped]
            mean_abs_flipped = abs_delta_flipped.mean(axis=0)
            std_abs_flipped = abs_delta_flipped.std(axis=0)
        else:
            mean_abs_flipped = np.full(abs_delta_raw.shape[1], np.nan, dtype=np.float64)
            std_abs_flipped = np.full(abs_delta_raw.shape[1], np.nan, dtype=np.float64)

        rank_order = np.argsort(-mean_abs)
        ranks = np.empty_like(rank_order)
        ranks[rank_order] = np.arange(1, len(rank_order) + 1)

        changed_mask = mean_abs > args.delta_threshold
        for idx, fname in enumerate(FEATURE_NAMES):
            if not changed_mask[idx]:
                continue
            feature_rows.append(
                {
                    "model": model_name,
                    "model_tag": model_tag,
                    "attack": attack,
                    "eps": eps_str,
                    "feature_name": fname,
                    "mean_abs_delta_raw": float(mean_abs[idx]),
                    "std_abs_delta_raw": float(std_abs[idx]),
                    "min_abs_delta_raw": float(min_abs[idx]),
                    "max_abs_delta_raw": float(max_abs[idx]),
                    "mean_abs_delta_raw_flipped": float(mean_abs_flipped[idx]),
                    "std_abs_delta_raw_flipped": float(std_abs_flipped[idx]),
                    "feature_rank": int(ranks[idx]),
                }
            )

    summary_df = pd.DataFrame(summary_rows)
    feature_df = pd.DataFrame(feature_rows)

    summary_out = RESULTS_DIR / "attack_restart_summary_8class.csv"
    feature_out = RESULTS_DIR / "feature_perturbation_8class.csv"

    summary_df.sort_values(["model_tag", "attack", "eps"], inplace=True)
    feature_df.sort_values(["model_tag", "attack", "eps", "feature_rank"], inplace=True)

    summary_df.to_csv(summary_out, index=False)
    feature_df.to_csv(feature_out, index=False)

    print(f"Saved: {summary_out}")
    print(f"Saved: {feature_out}")
    print(f"Rows: summary={len(summary_df)}, feature={len(feature_df)}")


if __name__ == "__main__":
    main()
