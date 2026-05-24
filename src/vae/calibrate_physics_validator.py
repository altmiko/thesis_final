"""
Calibrate post-hoc physics plausibility checks against clean training data.

For each 8-class category, this script samples up to 5,000 training rows,
inverse-transforms them to raw space, runs ``PhysicsValidator``, and writes
per-rule pass rates to ``physics_calibration.json``.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np

from vae.physics_validator import PhysicsValidator


SEED = 42
MAX_SAMPLES_PER_CLASS = 5000
FEATURE_NAMES = [
    "Header_Length",
    "Protocol Type",
    "Time_To_Live",
    "Rate",
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
    "HTTP",
    "HTTPS",
    "DNS",
    "Telnet",
    "SMTP",
    "SSH",
    "IRC",
    "TCP",
    "UDP",
    "DHCP",
    "ARP",
    "ICMP",
    "IGMP",
    "IPv",
    "LLC",
    "Tot sum",
    "Min",
    "Max",
    "AVG",
    "Std",
    "Tot size",
    "IAT",
    "Number",
    "Variance",
]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_pickle(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def _sample_indices(mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    indices = np.flatnonzero(mask)
    if indices.size <= MAX_SAMPLES_PER_CLASS:
        return indices
    return np.sort(rng.choice(indices, size=MAX_SAMPLES_PER_CLASS, replace=False))


def main() -> None:
    root = _repo_root()
    data_dir = root / "data" / "processed"
    out_path = root / "physics_calibration.json"

    x_train = np.load(data_dir / "X_train.npy", mmap_mode="r")
    y_train = np.load(data_dir / "y_train_cat.npy")
    scaler = _load_pickle(data_dir / "scaler.pkl")
    category_encoder = _load_pickle(data_dir / "category_encoder.pkl")

    validator = PhysicsValidator(feature_names=FEATURE_NAMES)
    rng = np.random.default_rng(SEED)

    calibration: dict[str, dict[str, float]] = {}
    for class_idx, class_name in enumerate(category_encoder.classes_):
        sampled_idx = _sample_indices(y_train == class_idx, rng)
        x_scaled = np.asarray(x_train[sampled_idx], dtype=np.float64)
        x_raw = scaler.inverse_transform(x_scaled)
        result = validator.validate_batch(x_raw)
        calibration[str(class_name)] = {
            rule: float(result["per_rule_pass_rate"][rule])
            for rule in result["rules_implemented"]
        }
        print(
            f"{class_name}: n={len(sampled_idx)} "
            + ", ".join(
                f"{rule}={calibration[str(class_name)][rule]:.4f}"
                for rule in result["rules_implemented"]
            )
        )

    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(calibration, handle, indent=2, sort_keys=True)

    print(f"\nSaved calibration to {out_path}")


if __name__ == "__main__":
    main()
