from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch


_REPO_ROOT = Path(__file__).resolve().parents[2]
_TARGETED_SCRIPT = _REPO_ROOT / "src" / "attack" / "run_targeted_benign_latent_pgd.py"

VAE_METHOD_RUN_TAGS = {
    "gaussian": "gaussian_anticollapse_beta05_freebits01_20260529_173512",
    "laplace": "laplace_rerun_20260529",
}


def _parse_methods(value: str) -> list[str]:
    methods = [item.strip().lower() for item in value.split(",") if item.strip()]
    if not methods or methods == ["all"]:
        return list(VAE_METHOD_RUN_TAGS)

    unknown = [method for method in methods if method not in VAE_METHOD_RUN_TAGS]
    if unknown:
        raise ValueError(f"Unknown methods {unknown}; valid methods: {sorted(VAE_METHOD_RUN_TAGS)} or all")
    return methods


def _append_optional(cmd: list[str], flag: str, value: object | None) -> None:
    if value is not None:
        cmd.extend([flag, str(value)])


def _build_command(args: argparse.Namespace, method: str) -> list[str]:
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
        str(args.kappa),
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
    ]
    _append_optional(cmd, "--detector-max-samples", args.detector_max_samples)
    _append_optional(cmd, "--output-root", args.output_root)
    if args.force_refit_gmm:
        cmd.append("--force-refit-gmm")
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run targeted-to-Benign latent PGD for the tagged Gaussian and Laplace VAE reruns."
    )
    parser.add_argument("--methods", default="gaussian,laplace", help="Comma-separated methods, or all.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--models", default="all", help="Comma-separated model tags or all.")
    parser.add_argument("--samples-per-class", type=int, default=100)
    parser.add_argument("--epsilon", type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--num-steps", type=int, default=40)
    parser.add_argument("--num-restarts", type=int, default=5)
    parser.add_argument("--restart-strategy", default="encoded+jitter+gmm")
    parser.add_argument("--target-loss", default="ce", choices=["ce", "cw-margin"])
    parser.add_argument("--target-class", default="Benign", choices=["Benign"])
    parser.add_argument("--kappa", type=float, default=0.0)
    parser.add_argument("--lambda-latent-l2", type=float, default=0.0)
    parser.add_argument("--gmm-split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--gmm-components", type=int, default=5)
    parser.add_argument("--gmm-fit-max-samples", type=int, default=50000)
    parser.add_argument("--force-refit-gmm", action="store_true")
    parser.add_argument("--detector-max-samples", type=int, default=None)
    parser.add_argument("--selection-batch-size", type=int, default=8192)
    parser.add_argument("--output-root", default=None)
    args = parser.parse_args()

    for method in _parse_methods(args.methods):
        cmd = _build_command(args, method)
        print(f"\n=== Targeted Benign PGD: {method} ({VAE_METHOD_RUN_TAGS[method]}) ===", flush=True)
        print(" ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=str(_REPO_ROOT), check=True)


if __name__ == "__main__":
    main()
