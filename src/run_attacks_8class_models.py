"""
Run 8-class adversarial attacks across MLP, LSTM, and CNN-LSTM (serial) models.

This script supports multi-restart evaluation for PGD and CW. FGSM is executed
as a single-run baseline.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from adversarial_attacks import compute_attack_metrics
from adversarial_attacks import load_model
from adversarial_attacks import run_attack_with_restarts


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results" / "attacks"


MODEL_CONFIGS = [
    {
        "model_tag": "mlp",
        "label": "MLP",
        "checkpoint": "mlp_8class.pt",
        "num_classes": 8,
    },
    {
        "model_tag": "lstm",
        "label": "LSTM",
        "checkpoint": "lstm_8class.pt",
        "num_classes": 8,
    },
    {
        "model_tag": "serial",
        "label": "CNN-LSTM",
        "checkpoint": "serial_8class.pt",
        "num_classes": 8,
    },
]


ATTACK_GRID: List[Tuple[str, Optional[float]]] = [
    ("fgsm", 0.05),
    ("fgsm", 0.10),
    ("fgsm", 0.30),
    ("pgd", 0.05),
    ("pgd", 0.10),
    ("pgd", 0.30),
    ("cw", None),
]


def _format_eps_label(eps: Optional[float]) -> str:
    if eps is None:
        return "N/A"
    return f"{eps:.2f}"


def _format_eps_filename(eps: Optional[float]) -> str:
    if eps is None:
        return "0"
    return f"{eps:.2f}"


def _stratified_indices(y: np.ndarray, sample_size: int, random_state: int) -> np.ndarray:
    if sample_size > len(y):
        raise ValueError(f"sample_size {sample_size} cannot exceed dataset size {len(y)}")

    idx_all = np.arange(len(y))
    idx_keep, _ = train_test_split(
        idx_all,
        train_size=sample_size,
        random_state=random_state,
        stratify=y,
        shuffle=True,
    )
    return idx_keep


def _ensure_requirements() -> None:
    needed = [
        DATA_DIR / "X_test.npy",
        DATA_DIR / "y_test_cat.npy",
        MODELS_DIR / "mlp_8class.pt",
        MODELS_DIR / "lstm_8class.pt",
        MODELS_DIR / "serial_8class.pt",
    ]
    missing = [p for p in needed if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(str(p) for p in missing))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 8-class attacks for MLP/LSTM/CNN-LSTM with restart support.")
    parser.add_argument("--device", default="cuda", help="Device: cuda or cpu")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--sample-size", type=int, default=25000)
    parser.add_argument("--num-restarts", type=int, default=10, help="Restarts for PGD/CW")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out-prefix",
        default="attack",
        help="Output filename prefix in results/attacks (default: attack)",
    )
    args = parser.parse_args()

    if args.num_restarts < 1:
        raise ValueError("--num-restarts must be >= 1")

    np.random.seed(args.seed)

    _ensure_requirements()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    x_test = np.load(DATA_DIR / "X_test.npy", mmap_mode="r")
    y_full = np.load(DATA_DIR / "y_test_cat.npy")

    idx = _stratified_indices(y_full, sample_size=args.sample_size, random_state=args.seed)
    x_main = np.array(x_test[idx], dtype=np.float32, copy=True)
    y_main = np.array(y_full[idx], dtype=np.int64, copy=True)

    vals, cnts = np.unique(y_main, return_counts=True)
    print(f"Subsample distribution (8-class, N={len(y_main)}):")
    for cls, cnt in zip(vals, cnts):
        print(f"  class={int(cls):2d} count={int(cnt):7d}")

    summary_rows: List[Dict[str, object]] = []
    total_runs = len(MODEL_CONFIGS) * len(ATTACK_GRID)
    run_counter = 0

    for config in MODEL_CONFIGS:
        model = load_model(
            str(MODELS_DIR / config["checkpoint"]),
            num_features=x_main.shape[1],
            num_classes=int(config["num_classes"]),
            device=args.device,
        )

        for attack_name, eps in ATTACK_GRID:
            run_counter += 1
            eps_value = 0.0 if eps is None else float(eps)
            num_restarts = args.num_restarts if attack_name in {"pgd", "cw"} else 1

            start = time.perf_counter()
            result = run_attack_with_restarts(
                model=model,
                X=x_main,
                y=y_main,
                attack_name=attack_name,
                eps=eps_value,
                num_restarts=num_restarts,
                batch_size=args.batch_size,
                device=args.device,
                base_seed=args.seed,
            )
            elapsed = time.perf_counter() - start

            metrics = compute_attack_metrics(result)

            eps_for_file = _format_eps_filename(eps)
            out_file = (
                RESULTS_DIR
                / f"{args.out_prefix}_{config['model_tag']}_8class_{attack_name}_{eps_for_file}_r{num_restarts}.npz"
            )
            np.savez_compressed(
                out_file,
                X_adv=result["X_adv"],
                X_clean=result["X_clean"],
                y_true=result["y_true"],
                y_pred_clean=result["y_pred_clean"],
                y_pred_adv=result["y_pred_adv"],
                selected_restart=result.get("selected_restart", np.zeros(len(y_main), dtype=np.int32)),
                restart_samples_flipped=result.get("restart_samples_flipped", np.array([], dtype=np.int64)),
                attack_name=result["attack_name"],
                eps=result["eps"],
                num_restarts=result.get("num_restarts", num_restarts),
                model_name=config["label"],
                model_tag=config["model_tag"],
                task_name="8class",
                sample_size=len(y_main),
                seed=args.seed,
            )

            restart_counts = np.asarray(result.get("restart_samples_flipped", np.array([], dtype=np.int64)))
            restart_mean = float(restart_counts.mean()) if restart_counts.size else np.nan
            restart_std = float(restart_counts.std()) if restart_counts.size else np.nan

            print(
                f"[{run_counter}/{total_runs}] {config['label']} + {attack_name.upper()} eps={_format_eps_label(eps)} "
                f"r={num_restarts}: ASR_raw={metrics['attack_success_rate_raw'] * 100.0:.2f}% "
                f"clean_acc={metrics['clean_accuracy'] * 100.0:.2f}% time={elapsed:.2f}s"
            )

            summary_rows.append(
                {
                    "model": config["label"],
                    "model_tag": config["model_tag"],
                    "task": "8class",
                    "attack": attack_name.upper(),
                    "eps": _format_eps_label(eps),
                    "num_restarts": num_restarts,
                    "sample_size": len(y_main),
                    "clean_acc": metrics["clean_accuracy"],
                    "asr_raw": metrics["attack_success_rate_raw"],
                    "samples_flipped": metrics["samples_flipped"],
                    "mean_linf_scaled": metrics["mean_l_inf"],
                    "mean_l2_scaled": metrics["mean_l2"],
                    "restart_flipped_mean": restart_mean,
                    "restart_flipped_std": restart_std,
                    "elapsed_sec": elapsed,
                    "output_file": str(out_file.relative_to(ROOT)),
                }
            )

    summary_df = pd.DataFrame(summary_rows)
    summary_path = RESULTS_DIR / "attack_summary_8class_models_restarts.csv"
    summary_df.to_csv(summary_path, index=False)

    pretty_df = summary_df.copy()
    pretty_df["clean_acc"] = (pretty_df["clean_acc"] * 100.0).map(lambda x: f"{x:.1f}%")
    pretty_df["asr_raw"] = (pretty_df["asr_raw"] * 100.0).map(lambda x: f"{x:.1f}%")
    pretty_df["mean_linf_scaled"] = pretty_df["mean_linf_scaled"].map(lambda x: f"{x:.4f}")
    pretty_df["mean_l2_scaled"] = pretty_df["mean_l2_scaled"].map(lambda x: f"{x:.4f}")

    print("\nSummary table")
    print(
        pretty_df[
            [
                "model",
                "attack",
                "eps",
                "num_restarts",
                "clean_acc",
                "asr_raw",
                "samples_flipped",
                "mean_linf_scaled",
                "mean_l2_scaled",
            ]
        ].to_string(index=False)
    )
    print(f"\nSaved summary: {summary_path}")


if __name__ == "__main__":
    main()
