from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = str(_REPO_ROOT / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from attack.adversarial_attacks import load_model  # noqa: E402
from attack.latent_cw import latent_cw_attack  # noqa: E402
from attack.latent_infra import (  # noqa: E402
    AttackRouter,
    PerturbationMask,
    ProtocolValidator,
    load_split,
    set_global_seed,
)
from attack.latent_pgd import classifier_logits, latent_pgd_attack  # noqa: E402
from attack.run_all_models_attack_rerun import MODEL_SPECS  # noqa: E402
from preprocessing.feature_groups import FEATURE_NAMES  # noqa: E402
from vae.config import CLASS_TO_ID, CLASSES  # noqa: E402


RUN_PATTERNS = {
    "gaussian": "new_vae_attacks_gaussian_anticollapse_beta05_freebits01_20260529_173512_*_seed42",
    "laplace": "new_vae_attacks_laplace_rerun_20260529_*_seed42",
}
ATTACKS = ["latent-pgd", "latent-cw"]
SUMMARY_COLUMNS = [
    "sample_key",
    "run_label",
    "vae_run_tag",
    "model",
    "model_tag",
    "attack",
    "rank_in_combo",
    "sample_id",
    "source_class",
    "true_label",
    "full_run_predicted_label",
    "regenerated_pred_before",
    "regenerated_pred_after",
    "full_run_success",
    "regenerated_success",
    "protocol_valid",
    "mask_valid",
    "joint_valid",
    "input_l2_scaled",
    "latent_l2",
]
FEATURE_COLUMNS = SUMMARY_COLUMNS + [
    "feature_index",
    "feature",
    "original_raw",
    "adversarial_raw",
    "delta_raw",
    "original_scaled",
    "adversarial_scaled",
    "delta_scaled",
]


def _load_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _latest_complete_run(pattern: str) -> Path:
    root = _REPO_ROOT / "outputs" / "latent_attacks"
    candidates = sorted(root.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    for candidate in candidates:
        if (candidate / "summary.csv").exists() and (candidate / "per_sample_results.csv").exists():
            return candidate
    raise FileNotFoundError(f"No completed attack run found for pattern: {pattern}")


def _resolve_attack_run_dir(label: str, override: str | None) -> Path:
    if override:
        path = Path(override).resolve()
        if not path.exists():
            raise FileNotFoundError(f"{label} attack run directory not found: {path}")
        return path
    return _latest_complete_run(RUN_PATTERNS[label])


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


def _fmt_value(value: Any) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    if value is None:
        return ""
    try:
        x = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(x):
        return str(x)
    if abs(x) < 1e-12:
        return "0"
    return f"{x:.6g}"


def _load_run_context(run_dir: Path) -> tuple[str, dict[str, Any]]:
    config = _load_json(run_dir / "config_snapshot.json")
    vae_run = config["vae_run"]
    manifest = _load_json(Path(vae_run["manifest_path"]))
    return str(vae_run["run_tag"]), manifest


def _candidate_rows(
    *,
    run_dir: Path,
    model_tag: str,
    attack: str,
    pool_size: int,
) -> list[dict[str, str]]:
    rows = _read_csv(run_dir / "per_sample_results.csv")
    matching = [
        row
        for row in rows
        if row.get("model_tag") == model_tag and row.get("attack_type") == attack
    ]
    successes = [row for row in matching if _truthy(row.get("success"))]
    fallback = [row for row in matching if not _truthy(row.get("success"))]
    return (successes + fallback)[:pool_size]


def _run_attack(
    *,
    attack: str,
    vae: torch.nn.Module,
    classifier: torch.nn.Module,
    mask: PerturbationMask,
    scaler: Any,
    x_batch: torch.Tensor,
    y_batch: torch.Tensor,
    device: str,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    if attack == "latent-pgd":
        return latent_pgd_attack(
            vae=vae,
            classifier=classifier,
            mask=mask,
            x_original=x_batch,
            y_true=y_batch,
            scaler=scaler,
            epsilon=args.epsilon,
            alpha=args.alpha,
            num_steps=args.num_steps,
            random_start=args.random_start,
            device=device,
        )
    if attack == "latent-cw":
        return latent_cw_attack(
            vae=vae,
            classifier=classifier,
            mask=mask,
            x_original=x_batch,
            y_true=y_batch,
            scaler=scaler,
            lambda_conf=args.lambda_conf,
            kappa=args.kappa,
            num_iterations=args.num_iterations,
            learning_rate=args.learning_rate,
            convergence_threshold=args.convergence_threshold,
            device=device,
        )
    raise KeyError(f"Unsupported attack: {attack}")


def _generate_candidates(
    *,
    run_label: str,
    vae_run_tag: str,
    router: AttackRouter,
    classifier: torch.nn.Module,
    model_label: str,
    model_tag: str,
    attack: str,
    candidates: list[dict[str, str]],
    split_test: dict[str, Any],
    mask: PerturbationMask,
    protocol_validator: ProtocolValidator,
    device: str,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    by_class: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in candidates:
        by_class[row["source_class"]].append(row)

    generated: list[dict[str, Any]] = []
    for source_class, class_rows in by_class.items():
        class_id = CLASS_TO_ID[source_class]
        sample_ids = np.asarray([int(row["sample_id"]) for row in class_rows], dtype=np.int64)
        x_batch = torch.from_numpy(split_test["X"][sample_ids].astype(np.float32))
        y_batch = torch.from_numpy(split_test["y_8"][sample_ids].astype(np.int64))

        vae = router.get_vae(class_id)
        x_adv, z_adv, metadata = _run_attack(
            attack=attack,
            vae=vae,
            classifier=classifier,
            mask=mask,
            scaler=router.scaler,
            x_batch=x_batch,
            y_batch=y_batch,
            device=device,
            args=args,
        )

        with torch.no_grad():
            logits_before = classifier_logits(classifier, x_batch.to(device), device=device).cpu()
            logits_after = classifier_logits(classifier, x_adv.to(device), device=device).cpu()
            pred_before = torch.argmax(logits_before, dim=1)
            pred_after = torch.argmax(logits_after, dim=1)
            success = pred_after != y_batch.cpu()
            protocol_valid = protocol_validator.validate(x_adv, already_scaled=True).cpu()
            mask_valid = mask.verify(x_adv.cpu(), x_batch.cpu())["all_compliant"].cpu()
            joint_valid = protocol_valid & mask_valid
            input_l2 = torch.linalg.norm(x_adv.cpu() - x_batch.cpu(), dim=1)
            latent_l2 = torch.linalg.norm(z_adv.cpu() - metadata["z_orig"].cpu(), dim=1)

        original_raw = router.scaler.inverse_transform(x_batch.numpy().astype(np.float64))
        adversarial_raw = router.scaler.inverse_transform(x_adv.cpu().numpy().astype(np.float64))

        for pos, row in enumerate(class_rows):
            generated.append(
                {
                    "run_label": run_label,
                    "vae_run_tag": vae_run_tag,
                    "model": model_label,
                    "model_tag": model_tag,
                    "attack": attack,
                    "sample_id": int(row["sample_id"]),
                    "source_class": source_class,
                    "true_label": row["true_label"],
                    "full_run_predicted_label": row["predicted_label"],
                    "regenerated_pred_before": CLASSES[int(pred_before[pos].item())],
                    "regenerated_pred_after": CLASSES[int(pred_after[pos].item())],
                    "full_run_success": _truthy(row["success"]),
                    "regenerated_success": bool(success[pos].item()),
                    "protocol_valid": bool(protocol_valid[pos].item()),
                    "mask_valid": bool(mask_valid[pos].item()),
                    "joint_valid": bool(joint_valid[pos].item()),
                    "input_l2_scaled": float(input_l2[pos].item()),
                    "latent_l2": float(latent_l2[pos].item()),
                    "original_scaled_vector": x_batch[pos].numpy().astype(np.float64),
                    "adversarial_scaled_vector": x_adv.cpu().numpy()[pos].astype(np.float64),
                    "original_raw_vector": original_raw[pos],
                    "adversarial_raw_vector": adversarial_raw[pos],
                }
            )

    return generated


def _select_generated(generated: list[dict[str, Any]], samples_per_combo: int) -> list[dict[str, Any]]:
    successes = [row for row in generated if row["regenerated_success"]]
    fallback = [row for row in generated if not row["regenerated_success"]]
    return (successes + fallback)[:samples_per_combo]


def _rows_for_sample(
    sample: dict[str, Any],
    *,
    rank_in_combo: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    sample_key = (
        f"{sample['run_label']}|{sample['model_tag']}|{sample['attack']}|"
        f"{rank_in_combo}|{sample['sample_id']}"
    )
    summary = {
        key: sample[key]
        for key in SUMMARY_COLUMNS
        if key in sample and key not in {"sample_key", "rank_in_combo"}
    }
    summary["sample_key"] = sample_key
    summary["rank_in_combo"] = rank_in_combo

    feature_rows: list[dict[str, Any]] = []
    orig_raw = sample["original_raw_vector"]
    adv_raw = sample["adversarial_raw_vector"]
    orig_scaled = sample["original_scaled_vector"]
    adv_scaled = sample["adversarial_scaled_vector"]
    for idx, feature in enumerate(FEATURE_NAMES):
        base = dict(summary)
        base.update(
            {
                "feature_index": idx,
                "feature": feature,
                "original_raw": float(orig_raw[idx]),
                "adversarial_raw": float(adv_raw[idx]),
                "delta_raw": float(adv_raw[idx] - orig_raw[idx]),
                "original_scaled": float(orig_scaled[idx]),
                "adversarial_scaled": float(adv_scaled[idx]),
                "delta_scaled": float(adv_scaled[idx] - orig_scaled[idx]),
            }
        )
        feature_rows.append(base)
    return summary, feature_rows


def _write_markdown(path: Path, sample_rows: list[dict[str, Any]], feature_rows: list[dict[str, Any]]) -> None:
    by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in feature_rows:
        by_sample[str(row["sample_key"])].append(row)

    lines = [
        "# Inverse-Transformed Attack Samples",
        "",
        "Two regenerated samples are shown for each VAE run, classifier, and latent attack method.",
        "",
        "## Sample Summary",
        "",
        "| Run | Model | Attack | Rank | Sample | Source | Before | After | Success | Joint Valid | L2 scaled |",
        "| --- | --- | --- | ---: | ---: | --- | --- | --- | --- | --- | ---: |",
    ]
    for row in sample_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["run_label"]),
                    str(row["model"]),
                    str(row["attack"]),
                    str(row["rank_in_combo"]),
                    str(row["sample_id"]),
                    str(row["source_class"]),
                    str(row["regenerated_pred_before"]),
                    str(row["regenerated_pred_after"]),
                    str(row["regenerated_success"]),
                    str(row["joint_valid"]),
                    _fmt_value(row["input_l2_scaled"]),
                ]
            )
            + " |"
        )

    for row in sample_rows:
        key = str(row["sample_key"])
        lines.extend(
            [
                "",
                f"## {row['run_label']} / {row['model']} / {row['attack']} / sample {row['sample_id']}",
                "",
                (
                    f"Source `{row['source_class']}`; regenerated prediction "
                    f"`{row['regenerated_pred_before']}` -> `{row['regenerated_pred_after']}`; "
                    f"success={row['regenerated_success']}; joint_valid={row['joint_valid']}."
                ),
                "",
                "| # | Feature | Original raw | Adversarial raw | Delta raw |",
                "| ---: | --- | ---: | ---: | ---: |",
            ]
        )
        for feature_row in by_sample[key]:
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(feature_row["feature_index"]),
                        str(feature_row["feature"]),
                        _fmt_value(feature_row["original_raw"]),
                        _fmt_value(feature_row["adversarial_raw"]),
                        _fmt_value(feature_row["delta_raw"]),
                    ]
                )
                + " |"
            )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export inverse-transformed feature-wise samples for new VAE attack reruns."
    )
    parser.add_argument("--gaussian-attack-run-dir", default=None)
    parser.add_argument("--laplace-attack-run-dir", default=None)
    parser.add_argument("--samples-per-combo", type=int, default=2)
    parser.add_argument("--candidate-pool-size", type=int, default=10)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epsilon", type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--num-steps", type=int, default=40)
    parser.add_argument("--random-start", action="store_true", default=True)
    parser.add_argument("--no-random-start", dest="random_start", action="store_false")
    parser.add_argument("--lambda-conf", type=float, default=1.0)
    parser.add_argument("--kappa", type=float, default=0.0)
    parser.add_argument("--num-iterations", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--convergence-threshold", type=float, default=1e-5)
    parser.add_argument(
        "--output-root",
        default=str(_REPO_ROOT / "outputs" / "latent_attacks" / "sample_exports"),
    )
    args = parser.parse_args()

    set_global_seed(args.seed)
    attack_run_dirs = {
        "gaussian": _resolve_attack_run_dir("gaussian", args.gaussian_attack_run_dir),
        "laplace": _resolve_attack_run_dir("laplace", args.laplace_attack_run_dir),
    }
    output_dir = Path(args.output_root) / datetime.now().strftime("new_vae_attack_samples_%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=False)

    split_test = load_split("test")
    all_sample_rows: list[dict[str, Any]] = []
    all_feature_rows: list[dict[str, Any]] = []

    for run_label, run_dir in attack_run_dirs.items():
        vae_run_tag, manifest = _load_run_context(run_dir)
        router = AttackRouter(device=args.device)
        router.manifest = manifest
        mask = PerturbationMask.from_preprocessing_artifacts()
        protocol_validator = ProtocolValidator(router.scaler)

        for spec in MODEL_SPECS:
            model_tag = str(spec["tag"])
            model_label = str(spec["label"])
            classifier = load_model(
                model_path=str(_REPO_ROOT / "models" / str(spec["checkpoint"])),
                num_features=len(FEATURE_NAMES),
                num_classes=len(CLASSES),
                device=args.device,
            )

            for attack in ATTACKS:
                candidates = _candidate_rows(
                    run_dir=run_dir,
                    model_tag=model_tag,
                    attack=attack,
                    pool_size=max(args.candidate_pool_size, args.samples_per_combo),
                )
                generated = _generate_candidates(
                    run_label=run_label,
                    vae_run_tag=vae_run_tag,
                    router=router,
                    classifier=classifier,
                    model_label=model_label,
                    model_tag=model_tag,
                    attack=attack,
                    candidates=candidates,
                    split_test=split_test,
                    mask=mask,
                    protocol_validator=protocol_validator,
                    device=args.device,
                    args=args,
                )
                selected = _select_generated(generated, args.samples_per_combo)
                if len(selected) < args.samples_per_combo:
                    raise RuntimeError(
                        f"Only {len(selected)} samples available for "
                        f"{run_label}/{model_tag}/{attack}"
                    )

                for rank, sample in enumerate(selected, start=1):
                    sample_row, feature_rows = _rows_for_sample(sample, rank_in_combo=rank)
                    all_sample_rows.append(sample_row)
                    all_feature_rows.extend(feature_rows)

            del classifier
            if torch.cuda.is_available() and str(args.device).startswith("cuda"):
                torch.cuda.empty_cache()

    _write_csv(output_dir / "sample_summary.csv", all_sample_rows, SUMMARY_COLUMNS)
    _write_csv(output_dir / "featurewise_samples.csv", all_feature_rows, FEATURE_COLUMNS)
    _write_markdown(output_dir / "featurewise_samples.md", all_sample_rows, all_feature_rows)

    run_context = {
        "attack_run_dirs": {key: str(path) for key, path in attack_run_dirs.items()},
        "samples_per_combo": int(args.samples_per_combo),
        "candidate_pool_size": int(args.candidate_pool_size),
        "attacks": ATTACKS,
        "models": MODEL_SPECS,
    }
    with open(output_dir / "export_context.json", "w", encoding="utf-8") as f:
        json.dump(run_context, f, indent=2)

    print(f"Output directory: {output_dir}")
    print(f"Sample rows: {len(all_sample_rows)}")
    print(f"Feature rows: {len(all_feature_rows)}")
    print()
    for row in all_sample_rows[:12]:
        print(
            f"{row['run_label']:8s} {row['model']:9s} {row['attack']:10s} "
            f"sample={row['sample_id']} {row['regenerated_pred_before']}->{row['regenerated_pred_after']} "
            f"success={row['regenerated_success']} joint={row['joint_valid']}"
        )


if __name__ == "__main__":
    main()
