from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import matplotlib.pyplot as plt
import torch


_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = str(_REPO_ROOT / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from attack.run_targeted_benign_new_vae_methods import (  # noqa: E402
    VAE_METHOD_RUN_TAGS,
    _parse_methods,
)


_TARGETED_SCRIPT = _REPO_ROOT / "src" / "attack" / "run_targeted_benign_latent_pgd.py"
SWEEP_COLUMNS = [
    "sweep_id",
    "run_id",
    "run_dir",
    "vae_method",
    "vae_run_tag",
    "model",
    "model_tag",
    "source_class",
    "attack_config",
    "target_loss",
    "kappa",
    "lambda_latent_l2",
    "n",
    "raw_attack_to_benign_rate",
    "post_repair_benign_rate",
    "protocol_validity_rate",
    "mask_compliance_rate",
    "raw_g1g8_validity_rate",
    "joint_validity_rate",
    "final_asr_valid",
    "joint_target_success_rate",
    "target_success_given_joint_valid",
    "idsr",
    "mean_l2_latent",
    "mean_l2_input",
    "attempts_per_sample_mean",
    "seed",
]


def _parse_kappas(value: str) -> list[float]:
    kappas: list[float] = []
    for raw in str(value).replace(";", ",").split(","):
        text = raw.strip()
        if not text:
            continue
        kappas.append(float(text))
    if not kappas:
        raise ValueError("At least one kappa value is required")
    return kappas


def _kappa_tag(kappa: float) -> str:
    text = f"{float(kappa):g}".replace("-", "neg").replace(".", "p")
    return text


def _read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _safe_float(value: Any, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    return float(value)


def _build_command(
    *,
    args: argparse.Namespace,
    method: str,
    kappa: float,
    run_output_root: Path,
) -> list[str]:
    run_tag = VAE_METHOD_RUN_TAGS[method]
    cmd = [
        sys.executable,
        str(_TARGETED_SCRIPT),
        "--vae-run-tag",
        run_tag,
        "--seed",
        str(args.seed),
        "--device",
        str(args.device),
        "--models",
        str(args.models),
        "--samples-per-class",
        str(args.samples_per_class),
        "--epsilon",
        str(args.epsilon),
        "--alpha",
        str(args.alpha),
        "--num-steps",
        str(args.num_steps),
        "--num-restarts",
        str(args.num_restarts),
        "--restart-strategy",
        str(args.restart_strategy),
        "--target-loss",
        str(args.target_loss),
        "--target-class",
        str(args.target_class),
        "--kappa",
        str(kappa),
        "--lambda-latent-l2",
        str(args.lambda_latent_l2),
        "--gmm-split",
        str(args.gmm_split),
        "--gmm-components",
        str(args.gmm_components),
        "--gmm-fit-max-samples",
        str(args.gmm_fit_max_samples),
        "--selection-batch-size",
        str(args.selection_batch_size),
        "--output-root",
        str(run_output_root),
    ]
    if args.detector_max_samples is not None:
        cmd.extend(["--detector-max-samples", str(args.detector_max_samples)])
    if args.force_refit_gmm:
        cmd.append("--force-refit-gmm")
    return cmd


def _latest_completed_run(root: Path) -> Path:
    candidates = [
        path
        for path in root.glob("*")
        if path.is_dir()
        and (path / "summary.csv").exists()
        and (path / "per_class.csv").exists()
        and (path / "per_sample_results.csv").exists()
    ]
    if not candidates:
        raise FileNotFoundError(f"No completed targeted run found under {root}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _aggregate_group(
    *,
    rows: list[dict[str, str]],
    metric_row: dict[str, str] | None,
    sweep_id: str,
    run_id: str,
    run_dir: Path,
    method: str,
    vae_run_tag: str,
    source_class: str,
    attack_config: str,
    args: argparse.Namespace,
    kappa: float,
) -> dict[str, Any]:
    n = len(rows)
    if n == 0:
        raise ValueError("Cannot aggregate an empty group")

    target_success = [_truthy(row.get("target_success")) for row in rows]
    protocol_valid = [_truthy(row.get("protocol_valid")) for row in rows]
    mask_valid = [_truthy(row.get("mask_valid")) for row in rows]
    raw_valid = [_truthy(row.get("raw_g1g8_valid")) for row in rows]
    joint_valid = [_truthy(row.get("joint_valid")) for row in rows]

    def rate(values: list[bool]) -> float:
        return float(sum(1 for value in values if value) / n)

    target_and_protocol = [
        bool(target_success[idx] and protocol_valid[idx])
        for idx in range(n)
    ]
    target_and_joint = [
        bool(target_success[idx] and joint_valid[idx])
        for idx in range(n)
    ]
    joint_count = sum(1 for value in joint_valid if value)
    target_joint_count = sum(1 for value in target_and_joint if value)

    metric_row = metric_row or {}
    return {
        "sweep_id": sweep_id,
        "run_id": run_id,
        "run_dir": str(run_dir),
        "vae_method": method,
        "vae_run_tag": vae_run_tag,
        "model": rows[0].get("model", ""),
        "model_tag": rows[0].get("model_tag", ""),
        "source_class": source_class,
        "attack_config": attack_config,
        "target_loss": args.target_loss,
        "kappa": float(kappa),
        "lambda_latent_l2": float(args.lambda_latent_l2),
        "n": int(n),
        "raw_attack_to_benign_rate": rate(target_success),
        "post_repair_benign_rate": rate(target_success),
        "protocol_validity_rate": rate(protocol_valid),
        "mask_compliance_rate": rate(mask_valid),
        "raw_g1g8_validity_rate": rate(raw_valid),
        "joint_validity_rate": rate(joint_valid),
        "final_asr_valid": rate(target_and_protocol),
        "joint_target_success_rate": rate(target_and_joint),
        "target_success_given_joint_valid": (
            float(target_joint_count / joint_count) if joint_count > 0 else 0.0
        ),
        "idsr": _safe_float(metric_row.get("idsr"), 0.0),
        "mean_l2_latent": _safe_float(metric_row.get("mean_l2_latent"), 0.0),
        "mean_l2_input": _safe_float(metric_row.get("mean_l2_input"), 0.0),
        "attempts_per_sample_mean": float(args.num_restarts),
        "seed": int(args.seed),
    }


def _collect_run_rows(
    *,
    sweep_id: str,
    run_dir: Path,
    method: str,
    kappa: float,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    summary_rows = _read_csv(run_dir / "summary.csv")
    per_class_rows = _read_csv(run_dir / "per_class.csv")
    sample_rows = _read_csv(run_dir / "per_sample_results.csv")

    if not sample_rows:
        raise ValueError(f"No per-sample rows found in {run_dir}")

    config = json.dumps(
        {
            "attack": "targeted_benign_latent_pgd",
            "target_loss": args.target_loss,
            "kappa": float(kappa),
            "lambda_latent_l2": float(args.lambda_latent_l2),
            "epsilon": float(args.epsilon),
            "alpha": float(args.alpha),
            "num_steps": int(args.num_steps),
            "num_restarts": int(args.num_restarts),
            "restart_strategy": str(args.restart_strategy),
        },
        sort_keys=True,
    )

    per_class_metrics = {
        (row.get("model_tag", ""), row.get("class_name", "")): row
        for row in per_class_rows
    }
    summary_metrics = {
        row.get("model_tag", ""): row
        for row in summary_rows
    }

    by_model_source: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    by_model: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in sample_rows:
        model_tag = row.get("model_tag", "")
        source_class = row.get("source_class", "")
        by_model_source[(model_tag, source_class)].append(row)
        by_model[model_tag].append(row)

    vae_run_tag = sample_rows[0].get("vae_run_tag", "")
    rows_out: list[dict[str, Any]] = []
    for (model_tag, source_class), rows in sorted(by_model_source.items()):
        rows_out.append(
            _aggregate_group(
                rows=rows,
                metric_row=per_class_metrics.get((model_tag, source_class)),
                sweep_id=sweep_id,
                run_id=run_dir.name,
                run_dir=run_dir,
                method=method,
                vae_run_tag=vae_run_tag,
                source_class=source_class,
                attack_config=config,
                args=args,
                kappa=kappa,
            )
        )

    for model_tag, rows in sorted(by_model.items()):
        rows_out.append(
            _aggregate_group(
                rows=rows,
                metric_row=summary_metrics.get(model_tag),
                sweep_id=sweep_id,
                run_id=run_dir.name,
                run_dir=run_dir,
                method=method,
                vae_run_tag=vae_run_tag,
                source_class="ALL",
                attack_config=config,
                args=args,
                kappa=kappa,
            )
        )
    return rows_out


def _plot_by_model(rows: list[dict[str, Any]], output_path: Path) -> None:
    model_rows = [row for row in rows if row["source_class"] == "ALL"]
    if not model_rows:
        return

    methods = sorted({str(row["vae_method"]) for row in model_rows})
    fig, axes = plt.subplots(
        1,
        len(methods),
        figsize=(6.0 * len(methods), 4.2),
        squeeze=False,
        sharey=True,
    )
    for axis, method in zip(axes[0], methods):
        subset = [row for row in model_rows if row["vae_method"] == method]
        model_tags = sorted({str(row["model"]) for row in subset})
        for model in model_tags:
            series = sorted(
                [row for row in subset if row["model"] == model],
                key=lambda row: float(row["kappa"]),
            )
            axis.plot(
                [float(row["kappa"]) for row in series],
                [float(row["final_asr_valid"]) * 100.0 for row in series],
                marker="o",
                label=model,
            )
        axis.set_title(f"{method}: aggregate")
        axis.set_xlabel("kappa")
        axis.grid(alpha=0.25)
    axes[0][0].set_ylabel("final_asr_valid (%)")
    axes[0][-1].legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_by_source(rows: list[dict[str, Any]], output_path: Path) -> None:
    source_rows = [row for row in rows if row["source_class"] != "ALL"]
    if not source_rows:
        return

    grouped: dict[tuple[str, str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in source_rows:
        grouped[(str(row["vae_method"]), str(row["source_class"]), float(row["kappa"]))].append(row)

    averaged: list[dict[str, Any]] = []
    for (method, source_class, kappa), group_rows in grouped.items():
        n_total = sum(int(row["n"]) for row in group_rows)
        if n_total == 0:
            continue
        value = sum(float(row["final_asr_valid"]) * int(row["n"]) for row in group_rows) / n_total
        averaged.append(
            {
                "vae_method": method,
                "source_class": source_class,
                "kappa": kappa,
                "final_asr_valid": value,
            }
        )

    methods = sorted({str(row["vae_method"]) for row in averaged})
    fig, axes = plt.subplots(
        1,
        len(methods),
        figsize=(6.0 * len(methods), 4.2),
        squeeze=False,
        sharey=True,
    )
    for axis, method in zip(axes[0], methods):
        subset = [row for row in averaged if row["vae_method"] == method]
        source_classes = sorted({str(row["source_class"]) for row in subset})
        for source_class in source_classes:
            series = sorted(
                [row for row in subset if row["source_class"] == source_class],
                key=lambda row: float(row["kappa"]),
            )
            axis.plot(
                [float(row["kappa"]) for row in series],
                [float(row["final_asr_valid"]) * 100.0 for row in series],
                marker="o",
                label=source_class,
            )
        axis.set_title(f"{method}: source mean")
        axis.set_xlabel("kappa")
        axis.grid(alpha=0.25)
    axes[0][0].set_ylabel("final_asr_valid (%)")
    axes[0][-1].legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _write_config_snapshot(
    *,
    path: Path,
    args: argparse.Namespace,
    kappas: list[float],
    sweep_dir: Path,
    commands: list[dict[str, Any]],
) -> None:
    payload = {
        "phase": "targeted_benign_kappa_sweep",
        "created_at": datetime.now().isoformat(),
        "sweep_dir": str(sweep_dir),
        "seed": int(args.seed),
        "methods": _parse_methods(args.methods),
        "kappas": kappas,
        "models": str(args.models),
        "samples_per_class": int(args.samples_per_class),
        "attack": {
            "target_loss": str(args.target_loss),
            "target_class": str(args.target_class),
            "epsilon": float(args.epsilon),
            "alpha": float(args.alpha),
            "num_steps": int(args.num_steps),
            "num_restarts": int(args.num_restarts),
            "restart_strategy": str(args.restart_strategy),
            "lambda_latent_l2": float(args.lambda_latent_l2),
        },
        "gmm": {
            "split": str(args.gmm_split),
            "components": int(args.gmm_components),
            "fit_max_samples": args.gmm_fit_max_samples,
            "force_refit": bool(args.force_refit_gmm),
        },
        "repair_in_loop": False,
        "repair_note": "No separate repair-in-loop path exists in the current targeted runner; post_repair_benign_rate equals raw_attack_to_benign_rate.",
        "commands": commands,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a targeted-Benign CW-margin kappa sweep and collect valid-ASR outputs."
    )
    parser.add_argument("--methods", default="gaussian,laplace", help="Comma-separated methods, or all.")
    parser.add_argument("--kappa-grid", default="0,5,10,20,40")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--models", default="all", help="Comma-separated model tags or all.")
    parser.add_argument("--samples-per-class", type=int, default=100)
    parser.add_argument("--epsilon", type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--num-steps", type=int, default=40)
    parser.add_argument("--num-restarts", type=int, default=5)
    parser.add_argument("--restart-strategy", default="encoded+jitter+gmm")
    parser.add_argument("--target-loss", default="cw-margin", choices=["ce", "cw-margin"])
    parser.add_argument("--target-class", default="Benign", choices=["Benign"])
    parser.add_argument("--lambda-latent-l2", type=float, default=0.0)
    parser.add_argument("--gmm-split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--gmm-components", type=int, default=5)
    parser.add_argument("--gmm-fit-max-samples", type=int, default=50000)
    parser.add_argument("--force-refit-gmm", action="store_true")
    parser.add_argument("--detector-max-samples", type=int, default=None)
    parser.add_argument("--selection-batch-size", type=int, default=8192)
    parser.add_argument(
        "--output-root",
        default=None,
        help="Sweep output root; defaults to outputs/latent_attacks/kappa_sweeps.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands and write config without running attacks.")
    args = parser.parse_args()

    kappas = _parse_kappas(args.kappa_grid)
    methods = _parse_methods(args.methods)
    output_root = (
        Path(args.output_root).resolve()
        if args.output_root
        else _REPO_ROOT / "outputs" / "latent_attacks" / "kappa_sweeps"
    )
    output_root.mkdir(parents=True, exist_ok=True)
    sweep_id = f"kappa_sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}_s{args.seed}"
    sweep_dir = output_root / sweep_id
    sweep_dir.mkdir(parents=True, exist_ok=False)
    # Keep nested runner paths short enough for Windows path limits; the
    # targeted runner appends a long VAE-tagged directory name and NPZ names.
    runs_root = sweep_dir / "r"
    logs_root = sweep_dir / "l"
    runs_root.mkdir(parents=True, exist_ok=False)
    logs_root.mkdir(parents=True, exist_ok=False)

    all_rows: list[dict[str, Any]] = []
    commands: list[dict[str, Any]] = []

    for kappa in kappas:
        for method in methods:
            run_output_root = runs_root / f"k{_kappa_tag(kappa)}_{method[0]}"
            run_output_root.mkdir(parents=True, exist_ok=True)
            cmd = _build_command(
                args=args,
                method=method,
                kappa=float(kappa),
                run_output_root=run_output_root,
            )
            command_record: dict[str, Any] = {
                "method": method,
                "kappa": float(kappa),
                "command": cmd,
                "run_output_root": str(run_output_root),
                "status": "dry_run" if args.dry_run else "pending",
            }
            commands.append(command_record)

            print(f"\n=== Kappa sweep: method={method} kappa={kappa:g} ===", flush=True)
            print(" ".join(cmd), flush=True)
            if args.dry_run:
                continue

            log_path = logs_root / f"{method}_kappa_{_kappa_tag(kappa)}.log"
            with open(log_path, "w", encoding="utf-8") as log_file:
                completed = subprocess.run(
                    cmd,
                    cwd=str(_REPO_ROOT),
                    text=True,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            command_record["returncode"] = int(completed.returncode)
            command_record["log_path"] = str(log_path)
            if completed.returncode != 0:
                command_record["status"] = "failed"
                _write_config_snapshot(
                    path=sweep_dir / "config_snapshot.json",
                    args=args,
                    kappas=kappas,
                    sweep_dir=sweep_dir,
                    commands=commands,
                )
                raise RuntimeError(f"Run failed for method={method} kappa={kappa:g}; see {log_path}")

            run_dir = _latest_completed_run(run_output_root)
            command_record["status"] = "completed"
            command_record["run_dir"] = str(run_dir)
            print(f"Completed: {run_dir}", flush=True)
            all_rows.extend(
                _collect_run_rows(
                    sweep_id=sweep_id,
                    run_dir=run_dir,
                    method=method,
                    kappa=float(kappa),
                    args=args,
                )
            )

    _write_config_snapshot(
        path=sweep_dir / "config_snapshot.json",
        args=args,
        kappas=kappas,
        sweep_dir=sweep_dir,
        commands=commands,
    )

    if args.dry_run:
        print(f"\nDry run complete. Config written to {sweep_dir / 'config_snapshot.json'}")
        return

    csv_path = sweep_dir / "kappa_sweep_valid_asr.csv"
    _write_csv(csv_path, all_rows, SWEEP_COLUMNS)
    _plot_by_model(all_rows, sweep_dir / "kappa_sweep_valid_asr_by_model.png")
    _plot_by_source(all_rows, sweep_dir / "kappa_sweep_valid_asr_by_source_class.png")

    print("\n=== Kappa Sweep Complete ===")
    print(f"Output directory: {sweep_dir}")
    print(f"CSV: {csv_path}")
    print(f"Plot by model: {sweep_dir / 'kappa_sweep_valid_asr_by_model.png'}")
    print(f"Plot by source class: {sweep_dir / 'kappa_sweep_valid_asr_by_source_class.png'}")


if __name__ == "__main__":
    main()
