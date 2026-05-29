from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = str(_REPO_ROOT / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from preprocessing.feature_groups import FEATURE_NAMES  # noqa: E402
from vae.config import CLASSES  # noqa: E402


RUN_PATTERNS = {
    "gaussian": "targeted_benign_pgd_gaussian_anticollapse_beta05_freebits01_20260529_173512_*_seed42",
    "laplace": "targeted_benign_pgd_laplace_rerun_20260529_*_seed42",
}
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
    "pred_before",
    "pred_after",
    "target_class",
    "target_success",
    "protocol_valid",
    "mask_valid",
    "raw_g1g8_valid",
    "joint_valid",
    "selected_restart",
    "input_l2_scaled",
    "latent_l2",
    "top_feature_changes",
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


def _load_pickle(path: Path) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _latest_complete_run(pattern: str) -> Path:
    root = _REPO_ROOT / "outputs" / "latent_attacks"
    candidates = sorted(root.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    for candidate in candidates:
        required = [
            candidate / "summary.csv",
            candidate / "per_sample_results.csv",
            candidate / "config_snapshot.json",
        ]
        if all(path.exists() for path in required):
            return candidate
    raise FileNotFoundError(f"No completed targeted benign run found for pattern: {pattern}")


def _resolve_attack_run_dir(label: str, override: str | None) -> Path:
    if override:
        path = Path(override).resolve()
        if not path.exists():
            raise FileNotFoundError(f"{label} targeted benign run directory not found: {path}")
        return path
    return _latest_complete_run(RUN_PATTERNS[label])


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


def _load_run_context(run_dir: Path) -> tuple[str, str]:
    config = _load_json(run_dir / "config_snapshot.json")
    vae_run = config.get("vae_run", {})
    run_tag = str(vae_run.get("run_tag", run_dir.name))
    attack_name = str(config.get("attack", {}).get("name", "targeted-benign-latent-pgd"))
    return run_tag, attack_name


def _npz_identity(npz_path: Path) -> tuple[str, str]:
    marker = "_targeted_benign_pgd_"
    stem = npz_path.stem
    if marker not in stem:
        raise ValueError(f"Unexpected targeted benign npz name: {npz_path.name}")
    model_tag, source_class = stem.split(marker, maxsplit=1)
    return model_tag, source_class


def _top_feature_changes(sample: dict[str, Any], *, top_n: int) -> str:
    orig_raw = sample["original_raw_vector"]
    adv_raw = sample["adversarial_raw_vector"]
    orig_scaled = sample["original_scaled_vector"]
    adv_scaled = sample["adversarial_scaled_vector"]
    delta_scaled = adv_scaled - orig_scaled
    delta_raw = adv_raw - orig_raw
    order = np.argsort(-np.abs(delta_scaled))

    parts: list[str] = []
    for idx in order:
        if abs(float(delta_scaled[idx])) <= 1e-12:
            continue
        feature = FEATURE_NAMES[int(idx)]
        parts.append(f"`{feature}` {_fmt_value(delta_raw[idx])}")
        if len(parts) >= int(top_n):
            break
    return "; ".join(parts)


def _load_samples_from_run(
    *,
    run_label: str,
    run_dir: Path,
    scaler: Any,
) -> tuple[str, list[dict[str, Any]]]:
    vae_run_tag, attack_name = _load_run_context(run_dir)
    per_sample_rows = _read_csv(run_dir / "per_sample_results.csv")
    rows_by_key: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in per_sample_rows:
        rows_by_key[(str(row["model_tag"]), str(row["source_class"]))].append(row)

    samples: list[dict[str, Any]] = []
    for npz_path in sorted(run_dir.glob("*_targeted_benign_pgd_*.npz")):
        model_tag, source_class = _npz_identity(npz_path)
        metadata_rows = rows_by_key[(model_tag, source_class)]
        data = np.load(npz_path)
        x_orig = np.asarray(data["x_orig"], dtype=np.float64)
        x_adv = np.asarray(data["x_adv"], dtype=np.float64)
        z_orig = np.asarray(data["z_orig"], dtype=np.float64)
        z_adv = np.asarray(data["z_adv"], dtype=np.float64)
        n_samples = int(x_orig.shape[0])

        if len(metadata_rows) != n_samples:
            raise ValueError(
                f"Metadata/sample count mismatch for {npz_path.name}: "
                f"{len(metadata_rows)} rows vs {n_samples} vectors"
            )

        orig_raw = scaler.inverse_transform(x_orig)
        adv_raw = scaler.inverse_transform(x_adv)
        input_l2 = np.linalg.norm(x_adv - x_orig, axis=1)
        latent_l2 = np.linalg.norm(z_adv - z_orig, axis=1)

        y_true = np.asarray(data["y_true"], dtype=np.int64)
        y_pred_before = np.asarray(data["y_pred_before"], dtype=np.int64)
        y_pred_after = np.asarray(data["y_pred_after"], dtype=np.int64)
        target_success = np.asarray(data["target_success"], dtype=np.bool_)
        protocol_valid = np.asarray(data["protocol_valid"], dtype=np.bool_)
        mask_valid = np.asarray(data["mask_valid"], dtype=np.bool_)
        raw_valid = np.asarray(data["raw_g1g8_valid"], dtype=np.bool_)
        joint_valid = np.asarray(data["joint_valid"], dtype=np.bool_)
        selected_restart = np.asarray(data["selected_restart"], dtype=np.int64)

        for pos, row in enumerate(metadata_rows):
            sample = {
                "run_label": run_label,
                "vae_run_tag": vae_run_tag,
                "model": row.get("model", model_tag),
                "model_tag": model_tag,
                "attack": attack_name,
                "sample_id": int(row["sample_id"]),
                "source_class": source_class,
                "true_label": CLASSES[int(y_true[pos])],
                "pred_before": CLASSES[int(y_pred_before[pos])],
                "pred_after": CLASSES[int(y_pred_after[pos])],
                "target_class": row.get("target_class", "Benign"),
                "target_success": bool(target_success[pos]),
                "protocol_valid": bool(protocol_valid[pos]),
                "mask_valid": bool(mask_valid[pos]),
                "raw_g1g8_valid": bool(raw_valid[pos]),
                "joint_valid": bool(joint_valid[pos]),
                "selected_restart": int(selected_restart[pos]),
                "input_l2_scaled": float(input_l2[pos]),
                "latent_l2": float(latent_l2[pos]),
                "original_scaled_vector": x_orig[pos],
                "adversarial_scaled_vector": x_adv[pos],
                "original_raw_vector": orig_raw[pos],
                "adversarial_raw_vector": adv_raw[pos],
            }
            samples.append(sample)

    return vae_run_tag, samples


def _select_samples(
    samples: list[dict[str, Any]],
    *,
    samples_per_combo: int,
) -> list[dict[str, Any]]:
    by_combo: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        by_combo[(str(sample["run_label"]), str(sample["model_tag"]))].append(sample)

    selected: list[dict[str, Any]] = []
    for combo in sorted(by_combo):
        combo_samples = by_combo[combo]
        ranked = sorted(
            combo_samples,
            key=lambda row: (
                not (row["target_success"] and row["joint_valid"]),
                not row["target_success"],
                not row["joint_valid"],
                float(row["input_l2_scaled"]),
                float(row["latent_l2"]),
                int(row["sample_id"]),
            ),
        )
        selected.extend(ranked[: int(samples_per_combo)])
    return selected


def _rows_for_sample(
    sample: dict[str, Any],
    *,
    rank_in_combo: int,
    top_n_features: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    sample_key = (
        f"{sample['run_label']}|{sample['model_tag']}|targeted_benign|"
        f"{rank_in_combo}|{sample['sample_id']}"
    )
    top_changes = _top_feature_changes(sample, top_n=top_n_features)
    summary = {
        key: sample[key]
        for key in SUMMARY_COLUMNS
        if key in sample and key not in {"sample_key", "rank_in_combo", "top_feature_changes"}
    }
    summary["sample_key"] = sample_key
    summary["rank_in_combo"] = rank_in_combo
    summary["top_feature_changes"] = top_changes

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
        "# Targeted Benign Inverse-Transformed Attack Samples",
        "",
        "Selected samples prioritize joint-valid attacks whose adversarial prediction is Benign, then smaller scaled input-space L2.",
        "",
        "## Sample Summary",
        "",
        "| Run | Model | Rank | Sample | Source -> After | Target Success | Joint Valid | L2 scaled | L2 latent | Top raw feature changes |",
        "| --- | --- | ---: | ---: | --- | --- | --- | ---: | ---: | --- |",
    ]
    for row in sample_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["run_label"]),
                    str(row["model"]),
                    str(row["rank_in_combo"]),
                    str(row["sample_id"]),
                    f"{row['source_class']} -> {row['pred_after']}",
                    str(row["target_success"]),
                    str(row["joint_valid"]),
                    _fmt_value(row["input_l2_scaled"]),
                    _fmt_value(row["latent_l2"]),
                    str(row["top_feature_changes"]),
                ]
            )
            + " |"
        )

    for row in sample_rows:
        key = str(row["sample_key"])
        lines.extend(
            [
                "",
                f"## {row['run_label']} / {row['model']} / sample {row['sample_id']}",
                "",
                (
                    f"Source `{row['source_class']}`; prediction "
                    f"`{row['pred_before']}` -> `{row['pred_after']}`; "
                    f"target_success={row['target_success']}; joint_valid={row['joint_valid']}."
                ),
                "",
                "| # | Feature | Original raw | Adversarial raw | Delta raw | Original scaled | Adversarial scaled | Delta scaled |",
                "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
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
                        _fmt_value(feature_row["original_scaled"]),
                        _fmt_value(feature_row["adversarial_scaled"]),
                        _fmt_value(feature_row["delta_scaled"]),
                    ]
                )
                + " |"
            )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export inverse-transformed feature-wise samples for targeted benign VAE attack runs."
    )
    parser.add_argument("--gaussian-attack-run-dir", default=None)
    parser.add_argument("--laplace-attack-run-dir", default=None)
    parser.add_argument("--samples-per-combo", type=int, default=2)
    parser.add_argument("--top-n-features", type=int, default=5)
    parser.add_argument(
        "--output-root",
        default=str(_REPO_ROOT / "outputs" / "latent_attacks" / "sample_exports"),
    )
    args = parser.parse_args()

    attack_run_dirs = {
        "gaussian": _resolve_attack_run_dir("gaussian", args.gaussian_attack_run_dir),
        "laplace": _resolve_attack_run_dir("laplace", args.laplace_attack_run_dir),
    }
    output_dir = Path(args.output_root) / datetime.now().strftime(
        "targeted_benign_attack_samples_%Y%m%d_%H%M%S"
    )
    output_dir.mkdir(parents=True, exist_ok=False)

    scaler = _load_pickle(_REPO_ROOT / "data" / "processed" / "scaler.pkl")
    all_samples: list[dict[str, Any]] = []
    vae_run_tags: dict[str, str] = {}
    for run_label, run_dir in attack_run_dirs.items():
        vae_run_tag, samples = _load_samples_from_run(
            run_label=run_label,
            run_dir=run_dir,
            scaler=scaler,
        )
        vae_run_tags[run_label] = vae_run_tag
        all_samples.extend(samples)

    selected = _select_samples(all_samples, samples_per_combo=args.samples_per_combo)
    sample_rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    combo_rank: dict[tuple[str, str], int] = defaultdict(int)
    for sample in selected:
        combo = (str(sample["run_label"]), str(sample["model_tag"]))
        combo_rank[combo] += 1
        sample_row, rows = _rows_for_sample(
            sample,
            rank_in_combo=combo_rank[combo],
            top_n_features=args.top_n_features,
        )
        sample_rows.append(sample_row)
        feature_rows.extend(rows)

    _write_csv(output_dir / "sample_summary.csv", sample_rows, SUMMARY_COLUMNS)
    _write_csv(output_dir / "featurewise_samples.csv", feature_rows, FEATURE_COLUMNS)
    _write_markdown(output_dir / "featurewise_samples.md", sample_rows, feature_rows)

    run_context = {
        "attack_run_dirs": {key: str(path) for key, path in attack_run_dirs.items()},
        "vae_run_tags": vae_run_tags,
        "samples_per_combo": int(args.samples_per_combo),
        "top_n_features": int(args.top_n_features),
        "selection_rule": "target_success and joint_valid first, then target_success, joint_valid, lower input_l2_scaled, lower latent_l2",
    }
    with open(output_dir / "export_context.json", "w", encoding="utf-8") as f:
        json.dump(run_context, f, indent=2)

    print(f"Output directory: {output_dir}")
    print(f"Sample rows: {len(sample_rows)}")
    print(f"Feature rows: {len(feature_rows)}")
    print()
    for row in sample_rows:
        print(
            f"{row['run_label']:8s} {row['model']:9s} "
            f"sample={row['sample_id']} {row['source_class']}->{row['pred_after']} "
            f"target={row['target_success']} joint={row['joint_valid']} "
            f"L2={_fmt_value(row['input_l2_scaled'])}"
        )


if __name__ == "__main__":
    main()
