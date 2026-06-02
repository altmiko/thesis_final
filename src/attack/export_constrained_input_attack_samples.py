from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
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
from attack.constrained_input_baselines import (  # noqa: E402
    VAEConstraintProjection,
    constrained_input_cw_attack,
    constrained_input_pgd_attack,
)
from attack.latent_infra import (  # noqa: E402
    PerturbationMask,
    ProtocolValidator,
    load_split,
    set_global_seed,
)
from attack.latent_pgd import classifier_logits  # noqa: E402
from attack.validator import validate_batch  # noqa: E402
from preprocessing.feature_groups import FEATURE_NAMES  # noqa: E402
from vae.config import CLASS_TO_ID, CLASSES  # noqa: E402


ATTACKS = (
    "cinput-pgd",
    "cinput-cw",
    "cinput-pgd-target-benign",
    "cinput-cw-target-benign",
)
SUMMARY_COLUMNS = [
    "sample_key",
    "model",
    "model_tag",
    "attack",
    "attack_goal",
    "target_class",
    "rank_in_attack",
    "sample_id",
    "source_class",
    "true_label",
    "full_run_predicted_label",
    "regenerated_pred_before",
    "regenerated_pred_after",
    "full_run_success",
    "regenerated_success",
    "full_run_target_success",
    "regenerated_target_success",
    "protocol_valid",
    "mask_valid",
    "raw_g1g8_valid",
    "joint_valid",
    "input_l2_scaled",
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


def _read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _latest_complete_run() -> Path:
    root = _REPO_ROOT / "outputs" / "latent_attacks"
    candidates = sorted(
        root.glob("constrained_input_baselines_*_seed*"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    required_names = ("config_snapshot.json", "summary.csv", "per_sample_results.csv")
    for candidate in candidates:
        if all((candidate / name).exists() for name in required_names):
            return candidate
    raise FileNotFoundError("No completed constrained-input baseline run found.")


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _stable_seed_offset(text: str) -> int:
    return sum((idx + 1) * ord(ch) for idx, ch in enumerate(text)) % 100_000


def _attack_goal(attack: str) -> str:
    return "target-benign" if str(attack).endswith("-target-benign") else "untargeted"


def _is_targeted_benign_attack(attack: str, params: dict[str, Any] | None = None) -> bool:
    if params is not None and "targeted" in params:
        return bool(params["targeted"])
    return _attack_goal(attack) == "target-benign"


def _target_class_id(value: Any = "Benign") -> int:
    if value is None:
        value = "Benign"
    if isinstance(value, str):
        return int(CLASS_TO_ID[value])
    return int(value)


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
    if abs(x) >= 1e5 or abs(x) < 1e-4:
        return f"{x:.4e}"
    return f"{x:.6g}"


def _describe_attack_params(attack: str, params: dict[str, Any]) -> str:
    targeted = _is_targeted_benign_attack(attack, params)
    target_part = ""
    if targeted:
        target_part = f", targeted=True, target_class={params.get('target_class', 'Benign')}"

    if attack.startswith("cinput-pgd"):
        return (
            f"epsilon={_fmt_value(params['epsilon'])}, alpha={_fmt_value(params['alpha'])}, "
            f"steps={params['num_steps']}, random_start={params['random_start']}{target_part}"
        )
    if attack.startswith("cinput-cw"):
        return (
            f"lambda_conf={_fmt_value(params['lambda_conf'])}, kappa={_fmt_value(params['kappa'])}, "
            f"iterations={params['num_iterations']}, lr={_fmt_value(params['learning_rate'])}, "
            f"convergence_threshold={_fmt_value(params['convergence_threshold'])}{target_part}"
        )
    return json.dumps(params, sort_keys=True)


def _diverse_take(rows: list[dict[str, str]], limit: int) -> list[dict[str, str]]:
    by_source: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in sorted(rows, key=lambda item: int(item["sample_id"])):
        by_source[str(row["source_class"])].append(row)

    selected: list[dict[str, str]] = []
    while len(selected) < limit and any(by_source.values()):
        for source_class in sorted(by_source):
            if by_source[source_class]:
                selected.append(by_source[source_class].pop(0))
                if len(selected) >= limit:
                    break
    return selected


def _candidate_rows(
    *,
    per_sample_rows: list[dict[str, str]],
    model_order: list[str],
    attack: str,
    candidate_pool_per_model: int,
) -> list[dict[str, str]]:
    by_model: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in per_sample_rows:
        if row.get("attack_type") != attack:
            continue
        success_value = row.get("target_success") if _attack_goal(attack) == "target-benign" else row.get("success")
        if _truthy(success_value) and _truthy(row.get("joint_valid")):
            by_model[str(row["model_tag"])].append(row)

    selected: list[dict[str, str]] = []
    for model_tag in model_order:
        selected.extend(_diverse_take(by_model.get(model_tag, []), candidate_pool_per_model))

    if selected:
        return selected

    fallback = [row for row in per_sample_rows if row.get("attack_type") == attack]
    return _diverse_take(fallback, candidate_pool_per_model * max(len(model_order), 1))


def _run_attack(
    *,
    attack: str,
    classifier: torch.nn.Module,
    projection: VAEConstraintProjection,
    x_batch: torch.Tensor,
    y_batch: torch.Tensor,
    config: dict[str, Any],
    device: str,
) -> torch.Tensor:
    params = config["attacks"][attack]
    targeted = _is_targeted_benign_attack(attack, params)
    target_class = _target_class_id(params.get("target_class", "Benign")) if targeted else CLASS_TO_ID["Benign"]

    if attack in {"cinput-pgd", "cinput-pgd-target-benign"}:
        x_adv, _metadata = constrained_input_pgd_attack(
            classifier=classifier,
            projection=projection,
            x_original=x_batch,
            y_true=y_batch,
            epsilon=float(params["epsilon"]),
            alpha=float(params["alpha"]),
            num_steps=int(params["num_steps"]),
            random_start=bool(params["random_start"]),
            device=device,
            targeted=targeted,
            target_class=target_class,
        )
        return x_adv
    if attack in {"cinput-cw", "cinput-cw-target-benign"}:
        x_adv, _metadata = constrained_input_cw_attack(
            classifier=classifier,
            projection=projection,
            x_original=x_batch,
            y_true=y_batch,
            lambda_conf=float(params["lambda_conf"]),
            kappa=float(params["kappa"]),
            num_iterations=int(params["num_iterations"]),
            learning_rate=float(params["learning_rate"]),
            convergence_threshold=float(params["convergence_threshold"]),
            device=device,
            targeted=targeted,
            target_class=target_class,
        )
        return x_adv
    raise KeyError(f"Unsupported attack: {attack}")


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
        parts.append(f"`{FEATURE_NAMES[int(idx)]}` {_fmt_value(delta_raw[idx])}")
        if len(parts) >= int(top_n):
            break
    return "; ".join(parts)


def _regenerate_candidates(
    *,
    run_dir: Path,
    config: dict[str, Any],
    candidate_rows: list[dict[str, str]],
    attack: str,
    device: str,
    seed: int,
) -> list[dict[str, Any]]:
    split_test = load_split("test")
    scaler = split_test["scaler"]
    mask = PerturbationMask.from_preprocessing_artifacts()
    protocol_validator = ProtocolValidator(scaler)
    projection = VAEConstraintProjection(scaler, mask, enable_physics=True, device=device)
    params = config["attacks"][attack]
    targeted = _is_targeted_benign_attack(attack, params)
    target_class = _target_class_id(params.get("target_class", "Benign")) if targeted else None
    target_class_name = CLASSES[int(target_class)] if target_class is not None else None

    model_specs = {str(spec["tag"]): spec for spec in config["models"]}
    by_model: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in candidate_rows:
        by_model[str(row["model_tag"])].append(row)

    generated: list[dict[str, Any]] = []
    for model_tag, rows in by_model.items():
        spec = model_specs[model_tag]
        set_global_seed(int(seed) + _stable_seed_offset(f"{attack}:{model_tag}"))
        classifier = load_model(
            model_path=str(_REPO_ROOT / "models" / str(spec["checkpoint"])),
            num_features=len(FEATURE_NAMES),
            num_classes=len(CLASSES),
            device=device,
        )

        sample_ids = np.asarray([int(row["sample_id"]) for row in rows], dtype=np.int64)
        x_batch = torch.from_numpy(split_test["X"][sample_ids].astype(np.float32))
        y_batch = torch.from_numpy(split_test["y_8"][sample_ids].astype(np.int64))
        x_adv = _run_attack(
            attack=attack,
            classifier=classifier,
            projection=projection,
            x_batch=x_batch,
            y_batch=y_batch,
            config=config,
            device=device,
        )

        with torch.no_grad():
            logits_before = classifier_logits(classifier, x_batch.to(device), device=device).cpu()
            logits_after = classifier_logits(classifier, x_adv.to(device), device=device).cpu()
            pred_before = torch.argmax(logits_before, dim=1)
            pred_after = torch.argmax(logits_after, dim=1)
            if targeted:
                success = pred_after == int(target_class)
            else:
                success = pred_after != y_batch.cpu()
            protocol_valid = protocol_validator.validate(x_adv, already_scaled=True).cpu()
            mask_valid = mask.verify(x_adv.cpu(), x_batch.cpu())["all_compliant"].cpu()
            input_l2 = torch.linalg.norm(x_adv.cpu() - x_batch.cpu(), dim=1)

        original_scaled = x_batch.numpy().astype(np.float64)
        adversarial_scaled = x_adv.cpu().numpy().astype(np.float64)
        original_raw = scaler.inverse_transform(original_scaled)
        adversarial_raw = scaler.inverse_transform(adversarial_scaled)
        raw_valid = validate_batch(adversarial_raw, FEATURE_NAMES).overall_valid.astype(np.bool_)

        for pos, row in enumerate(rows):
            generated.append(
                {
                    "source_run_dir": str(run_dir),
                    "model": row.get("model_name", str(spec["label"])),
                    "model_tag": model_tag,
                    "attack": attack,
                    "attack_goal": _attack_goal(attack),
                    "target_class": target_class_name,
                    "sample_id": int(row["sample_id"]),
                    "source_class": row["source_class"],
                    "true_label": CLASSES[int(y_batch[pos].item())],
                    "full_run_predicted_label": row.get("predicted_label", ""),
                    "regenerated_pred_before": CLASSES[int(pred_before[pos].item())],
                    "regenerated_pred_after": CLASSES[int(pred_after[pos].item())],
                    "full_run_success": _truthy(row.get("success")),
                    "regenerated_success": bool(success[pos].item()),
                    "full_run_target_success": (
                        _truthy(row.get("target_success")) if targeted else None
                    ),
                    "regenerated_target_success": bool(success[pos].item()) if targeted else None,
                    "protocol_valid": bool(protocol_valid[pos].item()),
                    "mask_valid": bool(mask_valid[pos].item()),
                    "raw_g1g8_valid": bool(raw_valid[pos]),
                    "joint_valid": bool(
                        protocol_valid[pos].item() and mask_valid[pos].item() and raw_valid[pos]
                    ),
                    "input_l2_scaled": float(input_l2[pos].item()),
                    "original_scaled_vector": original_scaled[pos],
                    "adversarial_scaled_vector": adversarial_scaled[pos],
                    "original_raw_vector": original_raw[pos],
                    "adversarial_raw_vector": adversarial_raw[pos],
                }
            )

    return generated


def _select_samples(
    samples: list[dict[str, Any]],
    *,
    samples_per_attack: int,
    attacks: tuple[str, ...] = ATTACKS,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for attack in attacks:
        attack_samples = [sample for sample in samples if sample["attack"] == attack]
        ranked = sorted(
            attack_samples,
            key=lambda row: (
                not (row["regenerated_success"] and row["joint_valid"]),
                not row["regenerated_success"],
                not row["joint_valid"],
                float(row["input_l2_scaled"]),
                str(row["model_tag"]),
                int(row["sample_id"]),
            ),
        )
        selected.extend(ranked[: int(samples_per_attack)])
    return selected


def _rows_for_sample(
    sample: dict[str, Any],
    *,
    rank_in_attack: int,
    top_n_features: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    sample_key = f"{sample['attack']}|{rank_in_attack}|{sample['model_tag']}|{sample['sample_id']}"
    top_changes = _top_feature_changes(sample, top_n=top_n_features)
    summary = {
        key: sample[key]
        for key in SUMMARY_COLUMNS
        if key in sample and key not in {"sample_key", "rank_in_attack", "top_feature_changes"}
    }
    summary["sample_key"] = sample_key
    summary["rank_in_attack"] = int(rank_in_attack)
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
                "feature_index": int(idx),
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


def _write_markdown(
    path: Path,
    *,
    run_dir: Path,
    config: dict[str, Any],
    sample_rows: list[dict[str, Any]],
    feature_rows: list[dict[str, Any]],
    candidate_pool_per_model: int,
    attacks: tuple[str, ...] = ATTACKS,
) -> None:
    by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in feature_rows:
        by_sample[str(row["sample_key"])].append(row)

    title = "Constrained-Input Attack Inverse-Transformed Samples"
    if attacks and all(_attack_goal(attack) == "target-benign" for attack in attacks):
        title = "Constrained-Input Target-Benign Inverse-Transformed Samples"

    lines = [
        f"# {title}",
        "",
        f"Source run: `{run_dir.relative_to(_REPO_ROOT)}`.",
        "",
        (
            "The feature tables below are regenerated from selected `per_sample_results.csv` "
            "sample IDs because the constrained-input runner saves metadata, not the full "
            "adversarial vectors. Deltas are adversarial minus clean after "
            "`scaler.inverse_transform`."
        ),
        "",
        "Selection rule: candidate rows are successful and joint-valid in the source run; "
        f"up to {candidate_pool_per_model} diverse rows per model are regenerated for each attack; "
        "the displayed rows are the lowest scaled-L2 regenerated successful joint-valid examples "
        "per attack, with fallback to the best regenerated rows if needed.",
        "",
        "## Attack Parameters",
        "",
        "| Attack | Parameters |",
        "| --- | --- |",
        *[
            f"| {attack} | {_describe_attack_params(attack, config['attacks'][attack])} |"
            for attack in attacks
            if attack in config.get("attacks", {})
        ],
        "",
        "## Sample Summary",
        "",
        "| Attack | Goal | Target | Model | Rank | Sample | Source -> After | Goal Success | Joint Valid | Raw G1-G8 | L2 scaled | Top raw feature changes |",
        "| --- | --- | --- | --- | ---: | ---: | --- | --- | --- | --- | ---: | --- |",
    ]
    for row in sample_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["attack"]),
                    str(row.get("attack_goal") or ""),
                    str(row.get("target_class") or ""),
                    str(row["model"]),
                    str(row["rank_in_attack"]),
                    str(row["sample_id"]),
                    f"{row['source_class']} -> {row['regenerated_pred_after']}",
                    str(row["regenerated_success"]),
                    str(row["joint_valid"]),
                    str(row["raw_g1g8_valid"]),
                    _fmt_value(row["input_l2_scaled"]),
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
                f"## {row['attack']} / {row['model']} / sample {row['sample_id']}",
                "",
                (
                    f"Goal `{row.get('attack_goal') or 'untargeted'}`"
                    f"{' target `' + str(row['target_class']) + '`' if row.get('target_class') else ''}; "
                    f"source `{row['source_class']}`; regenerated prediction "
                    f"`{row['regenerated_pred_before']}` -> `{row['regenerated_pred_after']}`; "
                    f"goal_success={row['regenerated_success']}; protocol_valid={row['protocol_valid']}; "
                    f"mask_valid={row['mask_valid']}; raw_g1g8_valid={row['raw_g1g8_valid']}; "
                    f"joint_valid={row['joint_valid']}; scaled L2={_fmt_value(row['input_l2_scaled'])}."
                ),
                "",
                "| # | Feature | Clean raw | C-input raw | Delta raw | Clean scaled | C-input scaled | Delta scaled |",
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
        description="Export inverse-transformed feature-wise samples for constrained-input PGD/CW and target-Benign variants."
    )
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--samples-per-attack", type=int, default=2)
    parser.add_argument("--candidate-pool-per-model", type=int, default=4)
    parser.add_argument("--top-n-features", type=int, default=5)
    parser.add_argument(
        "--attacks",
        default=None,
        help="Optional comma-separated attack filter, e.g. cinput-pgd-target-benign,cinput-cw-target-benign.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-md", default=str(_REPO_ROOT / "c-attack_inverse_result.md"))
    parser.add_argument(
        "--output-summary-csv",
        default=str(_REPO_ROOT / "outputs" / "latent_attacks" / "sample_exports" / "cinput_sample_summary.csv"),
    )
    parser.add_argument(
        "--output-feature-csv",
        default=str(_REPO_ROOT / "outputs" / "latent_attacks" / "sample_exports" / "cinput_featurewise_samples.csv"),
    )
    args = parser.parse_args()

    if int(args.samples_per_attack) < 1:
        raise ValueError("--samples-per-attack must be >= 1")
    if int(args.candidate_pool_per_model) < 1:
        raise ValueError("--candidate-pool-per-model must be >= 1")

    run_dir = Path(args.run_dir).resolve() if args.run_dir else _latest_complete_run()
    config = _load_json(run_dir / "config_snapshot.json")
    per_sample_rows = _read_csv(run_dir / "per_sample_results.csv")
    model_order = [str(spec["tag"]) for spec in config["models"]]
    if args.attacks:
        attacks_to_export = tuple(item.strip() for item in str(args.attacks).split(",") if item.strip())
        unknown = [attack for attack in attacks_to_export if attack not in ATTACKS]
        if unknown:
            raise ValueError(f"Unknown attacks {unknown}; valid choices: {list(ATTACKS)}")
    else:
        attacks_to_export = ATTACKS

    all_generated: list[dict[str, Any]] = []
    for attack in attacks_to_export:
        if attack not in config.get("attacks", {}):
            continue
        candidates = _candidate_rows(
            per_sample_rows=per_sample_rows,
            model_order=model_order,
            attack=attack,
            candidate_pool_per_model=int(args.candidate_pool_per_model),
        )
        all_generated.extend(
            _regenerate_candidates(
                run_dir=run_dir,
                config=config,
                candidate_rows=candidates,
                attack=attack,
                device=str(args.device),
                seed=int(args.seed),
            )
        )

    selected = _select_samples(
        all_generated,
        samples_per_attack=int(args.samples_per_attack),
        attacks=attacks_to_export,
    )
    sample_rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    attack_rank: dict[str, int] = defaultdict(int)
    for sample in selected:
        attack_rank[str(sample["attack"])] += 1
        sample_row, rows = _rows_for_sample(
            sample,
            rank_in_attack=attack_rank[str(sample["attack"])],
            top_n_features=int(args.top_n_features),
        )
        sample_rows.append(sample_row)
        feature_rows.extend(rows)

    output_md = Path(args.output_md)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    _write_markdown(
        output_md,
        run_dir=run_dir,
        config=config,
        sample_rows=sample_rows,
        feature_rows=feature_rows,
        candidate_pool_per_model=int(args.candidate_pool_per_model),
        attacks=attacks_to_export,
    )

    output_summary_csv = Path(args.output_summary_csv)
    output_feature_csv = Path(args.output_feature_csv)
    output_summary_csv.parent.mkdir(parents=True, exist_ok=True)
    output_feature_csv.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(output_summary_csv, sample_rows, SUMMARY_COLUMNS)
    _write_csv(output_feature_csv, feature_rows, FEATURE_COLUMNS)

    print(f"Markdown: {output_md}")
    print(f"Summary CSV: {output_summary_csv}")
    print(f"Feature CSV: {output_feature_csv}")
    print(f"Selected samples: {len(sample_rows)}")
    for row in sample_rows:
        print(
            f"{row['attack']:10s} {row['model']:9s} sample={row['sample_id']} "
            f"goal={row.get('attack_goal') or 'untargeted'} "
            f"target={row.get('target_class') or ''} "
            f"{row['source_class']}->{row['regenerated_pred_after']} "
            f"goal_success={row['regenerated_success']} joint={row['joint_valid']} "
            f"L2={_fmt_value(row['input_l2_scaled'])}"
        )


if __name__ == "__main__":
    main()
