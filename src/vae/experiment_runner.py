"""
Run a small grid of targeted VAE experiments for a single class and save results.

Example:
    python src/vae/experiment_runner.py --device cpu --class-name Web
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PYTHON = Path("C:/Users/T2530985/.conda/envs/thesis/python.exe")


def _deep_update(base: dict, override: dict) -> dict:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_update(result[key], value)
        else:
            result[key] = value
    return result


def _load_default_config() -> dict:
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from vae.config import DEFAULT_CONFIG

    return deepcopy(DEFAULT_CONFIG)


def _experiment_grid(class_name: str) -> list[tuple[str, dict]]:
    if class_name == "Mirai":
        return [
            (
                "structured_beta100_constraint000",
                {
                    "use_structured_continuous_decoder": True,
                    "constraint_loss_weight": 0.0,
                },
            ),
            (
                "structured_beta100_constraint005",
                {
                    "use_structured_continuous_decoder": True,
                    "constraint_loss_weight": 0.05,
                },
            ),
            (
                "structured_beta100_constraint010",
                {
                    "use_structured_continuous_decoder": True,
                    "constraint_loss_weight": 0.10,
                },
            ),
            (
                "structured_beta075_constraint005",
                {
                    "use_structured_continuous_decoder": True,
                    "constraint_loss_weight": 0.05,
                    "beta_target": {class_name: 0.75},
                },
            ),
            (
                "structured_beta050_constraint005",
                {
                    "use_structured_continuous_decoder": True,
                    "constraint_loss_weight": 0.05,
                    "beta_target": {class_name: 0.50},
                },
            ),
            (
                "structured_beta025_constraint005",
                {
                    "use_structured_continuous_decoder": True,
                    "constraint_loss_weight": 0.05,
                    "beta_target": {class_name: 0.25},
                },
            ),
        ]
    if class_name == "Web":
        return [
            (
                "wide24_beta050_protocol3_baseline",
                {
                    "latent_dim": {class_name: 24},
                    "beta_target": {class_name: 0.50},
                    "encoder_hidden": [256, 128],
                    "decoder_hidden": [128, 256],
                    "protocol_loss_weight": 3.0,
                    "use_structured_continuous_decoder": False,
                    "constraint_loss_weight": 0.0,
                },
            ),
            (
                "wide24_beta050_protocol3_weighted_normalized_light",
                {
                    "latent_dim": {class_name: 24},
                    "beta_target": {class_name: 0.50},
                    "encoder_hidden": [256, 128],
                    "decoder_hidden": [128, 256],
                    "protocol_loss_weight": 3.0,
                    "use_structured_continuous_decoder": False,
                    "constraint_loss_weight": 0.0,
                    "normalize_feature_loss_weights": True,
                    "continuous_feature_loss_weights": {
                        "Time_To_Live": 1.5,
                        "IAT": 1.75,
                    },
                    "binary_feature_loss_weights": {
                        "HTTP": 1.5,
                        "HTTPS": 2.0,
                        "DNS": 1.25,
                    },
                },
            ),
            (
                "wide24_beta050_protocol3_binary_weighted_normalized_light",
                {
                    "latent_dim": {class_name: 24},
                    "beta_target": {class_name: 0.50},
                    "encoder_hidden": [256, 128],
                    "decoder_hidden": [128, 256],
                    "protocol_loss_weight": 3.0,
                    "use_structured_continuous_decoder": False,
                    "constraint_loss_weight": 0.0,
                    "normalize_feature_loss_weights": True,
                    "binary_feature_loss_weights": {
                        "HTTP": 1.5,
                        "HTTPS": 2.5,
                        "DNS": 1.25,
                    },
                },
            ),
            (
                "wide24_beta050_protocol3_weighted_normalized_mild",
                {
                    "latent_dim": {class_name: 24},
                    "beta_target": {class_name: 0.50},
                    "encoder_hidden": [256, 128],
                    "decoder_hidden": [128, 256],
                    "protocol_loss_weight": 3.0,
                    "use_structured_continuous_decoder": False,
                    "constraint_loss_weight": 0.0,
                    "normalize_feature_loss_weights": True,
                    "continuous_feature_loss_weights": {
                        "Time_To_Live": 2.0,
                        "IAT": 2.5,
                    },
                    "binary_feature_loss_weights": {
                        "HTTP": 2.0,
                        "HTTPS": 3.0,
                        "DNS": 2.0,
                    },
                },
            ),
            (
                "wide24_beta050_protocol3_structured",
                {
                    "latent_dim": {class_name: 24},
                    "beta_target": {class_name: 0.50},
                    "encoder_hidden": [256, 128],
                    "decoder_hidden": [128, 256],
                    "protocol_loss_weight": 3.0,
                    "use_structured_continuous_decoder": True,
                    "structured_continuous_mode": "full",
                    "constraint_loss_weight": 0.0,
                },
            ),
            (
                "wide24_beta050_protocol3_structured_constraint005",
                {
                    "latent_dim": {class_name: 24},
                    "beta_target": {class_name: 0.50},
                    "encoder_hidden": [256, 128],
                    "decoder_hidden": [128, 256],
                    "protocol_loss_weight": 3.0,
                    "use_structured_continuous_decoder": True,
                    "structured_continuous_mode": "full",
                    "constraint_loss_weight": 0.05,
                },
            ),
            (
                "wide24_beta050_protocol3_selective_constraint005",
                {
                    "latent_dim": {class_name: 24},
                    "beta_target": {class_name: 0.50},
                    "encoder_hidden": [256, 128],
                    "decoder_hidden": [128, 256],
                    "protocol_loss_weight": 3.0,
                    "use_structured_continuous_decoder": True,
                    "structured_continuous_mode": "selective",
                    "constraint_loss_weight": 0.05,
                },
            ),
            (
                "wide24_beta025_protocol3_structured_constraint005",
                {
                    "latent_dim": {class_name: 24},
                    "beta_target": {class_name: 0.25},
                    "encoder_hidden": [256, 128],
                    "decoder_hidden": [128, 256],
                    "protocol_loss_weight": 3.0,
                    "use_structured_continuous_decoder": True,
                    "structured_continuous_mode": "full",
                    "constraint_loss_weight": 0.05,
                },
            ),
            (
                "wide24_beta025_protocol3_selective_constraint005",
                {
                    "latent_dim": {class_name: 24},
                    "beta_target": {class_name: 0.25},
                    "encoder_hidden": [256, 128],
                    "decoder_hidden": [128, 256],
                    "protocol_loss_weight": 3.0,
                    "use_structured_continuous_decoder": True,
                    "structured_continuous_mode": "selective",
                    "constraint_loss_weight": 0.05,
                },
            ),
            (
                "wide24_beta025_protocol4_structured_constraint005",
                {
                    "latent_dim": {class_name: 24},
                    "beta_target": {class_name: 0.25},
                    "encoder_hidden": [256, 128],
                    "decoder_hidden": [128, 256],
                    "protocol_loss_weight": 4.0,
                    "use_structured_continuous_decoder": True,
                    "structured_continuous_mode": "full",
                    "constraint_loss_weight": 0.05,
                },
            ),
        ]

    base_grid: list[tuple[str, dict]] = [
        (
            "baseline_refresh",
            {
                "constraint_loss_weight": 0.0,
            },
        ),
        (
            "constraint005",
            {
                "constraint_loss_weight": 0.05,
            },
        ),
        (
            "constraint010",
            {
                "constraint_loss_weight": 0.10,
            },
        ),
        (
            "constraint010_beta075_latent24",
            {
                "latent_dim": {class_name: 24},
                "beta_target": {class_name: 0.75},
                "constraint_loss_weight": 0.10,
            },
        ),
    ]
    return base_grid


def _write_override(path: Path, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _run_train_all(class_name: str, override_path: Path | None, device: str) -> None:
    cmd = [
        str(DEFAULT_PYTHON),
        "src/vae/train_all.py",
        "--device",
        device,
        "--classes",
        class_name,
    ]
    if override_path is not None:
        cmd.extend(["--config", str(override_path)])

    subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        check=True,
    )


def _load_result_row(class_name: str, experiment_name: str) -> dict:
    summary_path = REPO_ROOT / "results" / "vae" / "summary.csv"
    diag_path = REPO_ROOT / "results" / "vae" / f"diagnostics_{class_name}.json"

    with open(summary_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    row = next(r for r in rows if r["Class"] == class_name)

    with open(diag_path, encoding="utf-8") as f:
        diag = json.load(f)

    worst = diag["per_feature_recon"]["worst_5_features"]
    return {
        "experiment": experiment_name,
        "class_name": class_name,
        "best_val_loss": float(row["best_val_loss"]),
        "final_kl": float(row["final_kl"]),
        "collapsed_dims": int(row["collapsed_dims"]),
        "unconditional_validity_pre_pct": float(row["unconditional_validity_pre_pct"]),
        "unconditional_validity_pct": float(row["unconditional_validity_pct"]),
        "unconditional_repair_pct": float(row["unconditional_repair_pct"]),
        "conditional_validity_pre_pct": float(row["conditional_validity_pre_pct"]),
        "conditional_validity_pct": float(row["conditional_validity_pct"]),
        "conditional_repair_pct": float(row["conditional_repair_pct"]),
        "protocol_accuracy_pct": float(row["protocol_accuracy_pct"]),
        "worst_feature": worst[0]["feature"] if worst else "",
        "worst_feature_error": float(worst[0]["error"]) if worst else float("nan"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run targeted VAE experiment sweeps.")
    parser.add_argument("--class-name", required=True, help="Single class name, e.g. Web")
    parser.add_argument("--device", default="cpu", help="PyTorch device string")
    args = parser.parse_args()

    class_name = args.class_name
    base_config = _load_default_config()

    out_dir = REPO_ROOT / "results" / "vae" / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / f"{class_name.lower()}_experiments.csv"

    rows: list[dict] = []
    for experiment_name, override in _experiment_grid(class_name):
        override_payload = _deep_update(base_config, override)
        override_path = out_dir / f"{class_name.lower()}_{experiment_name}.json"
        _write_override(override_path, override_payload)
        _run_train_all(class_name, override_path, args.device)
        rows.append(_load_result_row(class_name, experiment_name))

    with open(results_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
