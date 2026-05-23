"""
Run unconstrained FGSM, PGD, and CW attacks on trained baselines.

This script executes attack experiments for binary, 8-class, and 34-class
framing checkpoints and stores detailed outputs in results/attacks.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.attack.adversarial_attacks import compute_attack_metrics
from src.attack.adversarial_attacks import load_model
from src.attack.adversarial_attacks import run_attack


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results" / "attacks"


MODEL_CONFIGS = [
    {
        "name": "binary",
        "checkpoint": "mlp_binary.pt",
        "num_classes": 2,
        "y_file": "y_test_bin.npy",
        "class_names_file": None,
    },
    {
        "name": "8class",
        "checkpoint": "mlp_8class.pt",
        "num_classes": 8,
        "y_file": "y_test_cat.npy",
        "class_names_file": "category_names.json",
    },
    {
        "name": "34class",
        "checkpoint": "mlp_34class.pt",
        "num_classes": 34,
        "y_file": "y_test.npy",
        "class_names_file": "class_names.json",
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


def _load_json_names(path: Optional[Path]) -> Optional[List[str]]:
    if path is None or not path.exists():
        return None

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return [str(x) for x in data]

    if isinstance(data, dict):
        try:
            keys_sorted = sorted(data.keys(), key=lambda k: int(k))
        except ValueError:
            keys_sorted = sorted(data.keys())
        return [str(data[k]) for k in keys_sorted]

    return None


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


def _print_distribution(name: str, y: np.ndarray) -> None:
    vals, cnts = np.unique(y, return_counts=True)
    total = len(y)
    print(f"Subsample distribution for {name} (N={total}):")
    for cls, cnt in zip(vals, cnts):
        pct = 100.0 * float(cnt) / float(total)
        print(f"  class={int(cls):2d} count={int(cnt):7d} ({pct:6.3f}%)")


def _ensure_requirements() -> None:
    needed = [
        DATA_DIR / "X_test.npy",
        DATA_DIR / "y_test_bin.npy",
        DATA_DIR / "y_test_cat.npy",
        DATA_DIR / "y_test.npy",
        MODELS_DIR / "mlp_binary.pt",
        MODELS_DIR / "mlp_8class.pt",
        MODELS_DIR / "mlp_34class.pt",
    ]
    missing = [p for p in needed if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(str(p) for p in missing))


def _run_sanity_check(device: str, batch_size: int) -> None:
    print("=" * 90)
    print("SANITY CHECK: FGSM eps=0.3 on binary model (N=1000)")
    print("=" * 90)

    x_test = np.load(DATA_DIR / "X_test.npy", mmap_mode="r")
    y_bin = np.load(DATA_DIR / "y_test_bin.npy")

    idx = _stratified_indices(y_bin, sample_size=1000, random_state=42)
    x_sample = np.array(x_test[idx], dtype=np.float32, copy=True)
    y_sample = np.array(y_bin[idx], dtype=np.int64, copy=True)

    model = load_model(
        str(MODELS_DIR / "mlp_binary.pt"),
        num_features=x_sample.shape[1],
        num_classes=2,
        device=device,
    )

    start = time.perf_counter()
    result = run_attack(
        model=model,
        X=x_sample,
        y=y_sample,
        attack_name="fgsm",
        eps=0.3,
        batch_size=batch_size,
        device=device,
    )
    elapsed = time.perf_counter() - start

    metrics = compute_attack_metrics(result)

    checks: List[Tuple[bool, str]] = []
    checks.append((metrics["attack_success_rate_raw"] > 0.0, "ASR_raw > 0"))
    checks.append((metrics["clean_accuracy"] >= 0.90, "clean accuracy >= 0.90"))
    checks.append(
        (
            result["X_adv"].shape == result["X_clean"].shape == (1000, 39),
            "X_adv and X_clean shapes equal (1000, 39)",
        )
    )
    checks.append((not np.allclose(result["X_adv"], result["X_clean"]), "X_adv differs from X_clean"))
    linf = np.max(np.abs(result["X_adv"] - result["X_clean"]), axis=1)
    checks.append((bool(np.all(linf <= 0.3001)), "max per-sample L_inf <= eps"))

    print(f"Sanity elapsed: {elapsed:.2f}s")
    print(f"Clean accuracy: {metrics['clean_accuracy'] * 100.0:.2f}%")
    print(f"ASR_raw: {metrics['attack_success_rate_raw'] * 100.0:.2f}%")
    print("First sample, features 0-4")
    print("  clean:", np.array2string(result["X_clean"][0, :5], precision=6))
    print("  adv:  ", np.array2string(result["X_adv"][0, :5], precision=6))

    failed = [desc for ok, desc in checks if not ok]
    if failed:
        print("Sanity checks FAILED:")
        for desc in failed:
            print(f"  - {desc}")
        print("Diagnostics:")
        print(f"  X_adv dtype={result['X_adv'].dtype}, X_clean dtype={result['X_clean'].dtype}")
        print(f"  linf max={float(linf.max()):.6f}, linf mean={float(linf.mean()):.6f}")
        raise RuntimeError("Stopping because sanity checks failed.")

    print("Sanity checks PASSED")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run unconstrained adversarial attacks on trained baselines.")
    parser.add_argument("--device", default="cuda", help="Device: cuda or cpu")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--sample-size", type=int, default=50000)
    parser.add_argument(
        "--cw-sample-size",
        type=int,
        default=None,
        help="Optional reduced sample size for CW only (e.g., 10000 if CW is too slow).",
    )
    parser.add_argument("--skip-sanity", action="store_true", help="Skip pre-suite sanity checks")
    parser.add_argument("--sanity-only", action="store_true", help="Run sanity checks and exit")
    args = parser.parse_args()

    _ensure_requirements()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if not args.skip_sanity:
        _run_sanity_check(device=args.device, batch_size=args.batch_size)
    if args.sanity_only:
        print("Sanity-only mode complete. Exiting before full attack suite.")
        return

    x_test = np.load(DATA_DIR / "X_test.npy", mmap_mode="r")

    summary_rows: List[Dict[str, object]] = []
    total_runs = len(MODEL_CONFIGS) * len(ATTACK_GRID)
    run_counter = 0

    for config in MODEL_CONFIGS:
        y_full = np.load(DATA_DIR / config["y_file"])
        idx_main = _stratified_indices(y_full, sample_size=args.sample_size, random_state=42)
        x_main = np.array(x_test[idx_main], dtype=np.float32, copy=True)
        y_main = np.array(y_full[idx_main], dtype=np.int64, copy=True)

        _print_distribution(config["name"], y_main)

        cw_sample_size = args.cw_sample_size
        if cw_sample_size is None:
            x_cw = x_main
            y_cw = y_main
        else:
            idx_cw = _stratified_indices(y_full, sample_size=cw_sample_size, random_state=42)
            x_cw = np.array(x_test[idx_cw], dtype=np.float32, copy=True)
            y_cw = np.array(y_full[idx_cw], dtype=np.int64, copy=True)
            print(
                f"CW sample reduced for {config['name']}: "
                f"{len(y_main)} -> {len(y_cw)} samples"
            )

        class_names_path = DATA_DIR / config["class_names_file"] if config["class_names_file"] else None
        class_names = _load_json_names(class_names_path)
        if config["name"] == "binary" and class_names is None:
            class_names = ["benign", "attack"]

        model = load_model(
            str(MODELS_DIR / config["checkpoint"]),
            num_features=x_main.shape[1],
            num_classes=int(config["num_classes"]),
            device=args.device,
        )

        for attack_name, eps in ATTACK_GRID:
            run_counter += 1

            if attack_name == "cw":
                x_run = x_cw
                y_run = y_cw
            else:
                x_run = x_main
                y_run = y_main

            start = time.perf_counter()
            result = run_attack(
                model=model,
                X=x_run,
                y=y_run,
                attack_name=attack_name,
                eps=0.0 if eps is None else float(eps),
                batch_size=args.batch_size,
                device=args.device,
            )
            elapsed = time.perf_counter() - start

            metrics = compute_attack_metrics(result, class_names=class_names)

            eps_for_file = _format_eps_filename(eps)
            out_file = RESULTS_DIR / f"attack_{config['name']}_{attack_name}_{eps_for_file}.npz"
            np.savez_compressed(
                out_file,
                X_adv=result["X_adv"],
                X_clean=result["X_clean"],
                y_true=result["y_true"],
                y_pred_clean=result["y_pred_clean"],
                y_pred_adv=result["y_pred_adv"],
                attack_name=result["attack_name"],
                eps=result["eps"],
            )

            progress_line = (
                f"[{run_counter}/{total_runs}] {config['checkpoint'].replace('.pt', '')} + "
                f"{attack_name.upper()} eps={_format_eps_label(eps)}: "
                f"ASR_raw={metrics['attack_success_rate_raw'] * 100.0:.1f}%, "
                f"clean_acc={metrics['clean_accuracy'] * 100.0:.1f}%, "
                f"time={elapsed:.2f}s"
            )
            print(progress_line)

            summary_rows.append(
                {
                    "Model": config["name"],
                    "Attack": attack_name.upper(),
                    "Eps": _format_eps_label(eps),
                    "Clean Acc": metrics["clean_accuracy"],
                    "ASR_raw": metrics["attack_success_rate_raw"],
                    "Samples Flipped": metrics["samples_flipped"],
                    "Mean L_inf": metrics["mean_l_inf"],
                    "Mean L2": metrics["mean_l2"],
                    "Elapsed Sec": elapsed,
                }
            )

    summary_df = pd.DataFrame(summary_rows)
    summary_path = RESULTS_DIR / "attack_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    pretty_df = summary_df.copy()
    pretty_df["Clean Acc"] = (pretty_df["Clean Acc"] * 100.0).map(lambda x: f"{x:.1f}%")
    pretty_df["ASR_raw"] = (pretty_df["ASR_raw"] * 100.0).map(lambda x: f"{x:.1f}%")
    pretty_df["Samples Flipped"] = pretty_df["Samples Flipped"].map(lambda x: f"{int(x):,}")
    pretty_df["Mean L_inf"] = pretty_df["Mean L_inf"].map(lambda x: f"{x:.3f}")
    pretty_df["Mean L2"] = pretty_df["Mean L2"].map(lambda x: f"{x:.3f}")
    print("\nSummary table")
    print(pretty_df[["Model", "Attack", "Eps", "Clean Acc", "ASR_raw", "Samples Flipped", "Mean L_inf", "Mean L2"]].to_string(index=False))
    print(f"\nSaved summary: {summary_path}")


if __name__ == "__main__":
    main()
