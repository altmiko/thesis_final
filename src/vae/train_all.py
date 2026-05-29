"""
Orchestrator: train all 8 per-class β-VAEs and run diagnostics.

Usage:
    python src/vae/train_all.py --device cpu
    python src/vae/train_all.py --device cuda --classes Benign,DDoS
    python src/vae/train_all.py --config path/to/override.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Path bootstrap — ensure src/ is on sys.path before local imports
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = str(_REPO_ROOT / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from vae.config import CLASS_TO_ID, CLASSES, DEFAULT_CONFIG  # noqa: E402
from copy import deepcopy  # noqa: E402
from vae.dataset import PerClassDataset  # noqa: E402
from vae.diagnostics import run_diagnostics  # noqa: E402
from vae.model import MixedInputBetaVAE  # noqa: E402
from vae.schema import get_partition  # noqa: E402
from vae.train import _load_8class_labels, train_one_vae  # noqa: E402

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SHA-256 helper
# ---------------------------------------------------------------------------

def compute_sha256(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _resolve_model_hparams(
    config_like: dict,
    class_name: str,
) -> tuple[int, tuple[int, ...], tuple[int, ...], int, bool, bool, str, float, tuple[float, float]]:
    latent_cfg = config_like["latent_dim"]
    latent_dim = latent_cfg[class_name] if isinstance(latent_cfg, dict) else int(latent_cfg)
    encoder_hidden = tuple(config_like.get("encoder_hidden", [128, 64]))
    decoder_hidden = tuple(config_like.get("decoder_hidden", [64, 128]))
    protocol_embed_dim = int(config_like.get("protocol_embed_dim", 4))
    use_structured_continuous_decoder = bool(
        config_like.get("use_structured_continuous_decoder", False)
    )
    use_structured_physics_decoder = bool(
        config_like.get("use_structured_physics_decoder", False)
    )
    structured_continuous_mode = str(config_like.get("structured_continuous_mode", "full"))
    structured_std_floor = float(
        config_like.get(
            "structured_std_floor",
            0.01 if use_structured_continuous_decoder else 0.0,
        )
    )
    latent_logvar_bounds = (
        float(config_like.get("latent_logvar_floor", -6.0)),
        float(config_like.get("latent_logvar_ceiling", 6.0)),
    )
    return (
        latent_dim,
        encoder_hidden,
        decoder_hidden,
        protocol_embed_dim,
        use_structured_continuous_decoder,
        use_structured_physics_decoder,
        structured_continuous_mode,
        structured_std_floor,
        latent_logvar_bounds,
    )


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def _setup_logging(log_path: Path) -> None:
    """Configure root logger to emit to stdout and a log file (append)."""
    log_path.parent.mkdir(parents=True, exist_ok=True)

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root = logging.getLogger()
    root.setLevel(logging.INFO)

    # stdout handler
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    root.addHandler(sh)

    # file handler (append)
    fh = logging.FileHandler(str(log_path), mode="a", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    root.addHandler(fh)


# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------

def _load_manifest(manifest_path: Path) -> dict:
    if manifest_path.exists():
        with open(manifest_path, encoding="utf-8") as f:
            return json.load(f)
    # Minimal skeleton if missing
    return {"schema_version": "1.0", "checkpoints": {}, "diagnostics": {}}


def _deep_update(base: dict, override: dict) -> dict:
    """Recursively merge override into base, updating nested dicts rather than replacing them."""
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_update(result[key], value)
        else:
            result[key] = value
    return result


def _save_manifest(manifest: dict, manifest_path: Path) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def _resolve_repo_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _format_run_template(
    template: str,
    *,
    class_id: int,
    class_name: str,
    run_tag: str | None,
) -> str:
    return template.format(
        class_id=class_id,
        class_name=class_name,
        run_tag=run_tag or "",
    )


def _config_for_class(
    base_config: dict,
    *,
    class_id: int,
    class_name: str,
    run_tag: str | None,
    results_dir: Path,
    default_results_dir: Path,
) -> dict:
    class_config = deepcopy(base_config)

    checkpoint_template = class_config.get("checkpoint_name_template")
    if checkpoint_template:
        class_config["checkpoint_name_override"] = _format_run_template(
            str(checkpoint_template),
            class_id=class_id,
            class_name=class_name,
            run_tag=run_tag,
        )
    elif run_tag and not class_config.get("checkpoint_name_override"):
        class_config["checkpoint_name_override"] = (
            f"vae_class_{class_id}_{class_name}_{run_tag}.pt"
        )

    curves_template = class_config.get("curves_name_template")
    if curves_template:
        class_config["curves_name_override"] = _format_run_template(
            str(curves_template),
            class_id=class_id,
            class_name=class_name,
            run_tag=run_tag,
        )
    elif run_tag and not class_config.get("curves_name_override"):
        class_config["curves_name_override"] = f"curves_{class_name}_{run_tag}.png"

    if results_dir != default_results_dir and not class_config.get("curves_dir_override"):
        class_config["curves_dir_override"] = str(results_dir)

    return class_config


# ---------------------------------------------------------------------------
# Phase 10 gate checks
# ---------------------------------------------------------------------------

def _check_gates(
    summary_rows: list[dict],
    config: dict,
    manifest: dict,
    summary_md_path: Path,
) -> dict[str, bool]:
    """Evaluate all Phase 10 gates and return a name→bool dict."""
    gates: dict[str, bool] = {}

    # Gate 1: All 8 checkpoints present
    ckpt_dir = _REPO_ROOT / "models" / "vae"
    all_present = True
    for cls in CLASSES:
        cid = CLASS_TO_ID[cls]
        ckpt_info = manifest.get("checkpoints", {}).get(cls, {})
        p = Path(ckpt_info.get("path", ckpt_dir / f"vae_class_{cid}_{cls}.pt"))
        if not p.exists():
            all_present = False
            break
    gates["Gate1_all_checkpoints_exist"] = all_present

    # Gate 2: No VAE has >50% collapsed dims
    gate2 = True
    for row in summary_rows:
        latent_dim = int(row.get("latent_dim", 16))
        collapsed = row.get("collapsed_dims", 0)
        if collapsed > 0.5 * latent_dim:
            gate2 = False
            break
    gates["Gate2_no_excess_collapse"] = gate2

    # Gate 3: Raw conditional validity >= 60% for every class
    gate3 = all(
        row.get("conditional_validity_pre_pct", 0.0) >= 60.0
        for row in summary_rows
    )
    gates["Gate3_raw_conditional_validity_60pct"] = gate3

    # Gate 4: Raw unconditional validity >= 30% for every class
    gate4 = all(
        row.get("unconditional_validity_pre_pct", 0.0) >= 30.0
        for row in summary_rows
    )
    gates["Gate4_raw_unconditional_validity_30pct"] = gate4

    # Gate 4b: Postprocessed unconditional validity >= 30% for every class
    gate4b = all(
        row.get("unconditional_validity_pct", 0.0) >= 30.0
        for row in summary_rows
    )
    gates["Gate4b_postprocessed_unconditional_validity_30pct"] = gate4b

    # Gate 5: Protocol accuracy >= 95% for every class
    gate5 = all(
        row.get("protocol_accuracy_pct", 0.0) >= 95.0
        for row in summary_rows
    )
    gates["Gate5_protocol_accuracy_95pct"] = gate5

    # Gate 6: No NaN/Inf in per-feature recon errors
    gate6 = True
    for cls_name, diag_info in manifest.get("diagnostics", {}).items():
        diag_path_str = diag_info.get("path")
        if diag_path_str and Path(diag_path_str).exists():
            with open(diag_path_str, encoding="utf-8") as f:
                diag = json.load(f)
            pfr = diag.get("per_feature_recon", {})
            for sub_dict_key in ("recon_nll_continuous", "recon_bce_independent_binary"):
                for feat, val in pfr.get(sub_dict_key, {}).items():
                    if val is None or (isinstance(val, float) and (math.isnan(val) or math.isinf(val))):
                        gate6 = False
                        break
    gates["Gate6_no_nan_inf_recon"] = gate6

    # Gate 7: protocol_binary_consistency == 1.0 for every class
    gate7 = True
    for cls_name, diag_info in manifest.get("diagnostics", {}).items():
        diag_path_str = diag_info.get("path")
        if diag_path_str and Path(diag_path_str).exists():
            with open(diag_path_str, encoding="utf-8") as f:
                diag = json.load(f)
            pbc = diag.get("unconditional_validity", {}).get("protocol_binary_consistency")
            if pbc is None or abs(float(pbc) - 1.0) > 1e-9:
                gate7 = False
                break
    gates["Gate7_protocol_binary_consistency"] = gate7

    # Gate 8: summary.md exists
    gates["Gate8_summary_md_exists"] = summary_md_path.exists()

    return gates


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------

_SUMMARY_COLS = [
    "Class",
    "latent_dim",
    "n_train",
    "n_val",
    "best_val_loss",
    "final_kl",
    "collapsed_dims",
    "unconditional_validity_pre_pct",
    "unconditional_validity_pct",
    "unconditional_repair_pct",
    "conditional_validity_pre_pct",
    "conditional_validity_pct",
    "conditional_repair_pct",
    "protocol_accuracy_pct",
]


def _validation_metrics(validation_block: dict) -> tuple[float, float, float]:
    pre_rate = validation_block.get("pre_postprocess_validity_rate")
    post_rate = validation_block.get("overall_validity_rate")
    repair_rate = validation_block.get("postprocess_repair_rate")
    return (
        round(float(pre_rate) * 100.0, 2) if pre_rate is not None else float("nan"),
        round(float(post_rate) * 100.0, 2) if post_rate is not None else float("nan"),
        round(float(repair_rate) * 100.0, 2) if repair_rate is not None else float("nan"),
    )


def _build_summary_row(metrics: dict, diag_result: dict) -> dict:
    collapse = diag_result.get("posterior_collapse", {})
    uncond = diag_result.get("unconditional_validity", {})
    cond = diag_result.get("conditional_validity", {})
    pfr = diag_result.get("per_feature_recon", {})

    uncond_pre_pct, uncond_post_pct, uncond_repair_pct = _validation_metrics(uncond)
    cond_pre_pct, cond_post_pct, cond_repair_pct = _validation_metrics(cond)
    proto_acc = pfr.get("protocol_top1_accuracy")

    return {
        "Class": metrics["class_name"],
        "n_train": metrics["n_train"],
        "n_val": metrics["n_val"],
        "best_val_loss": round(float(metrics["best_val_loss"]), 6),
        "final_kl": round(float(metrics["final_kl"]), 6),
        "latent_dim": int(metrics.get("latent_dim", 16)),
        "collapsed_dims": int(collapse.get("collapsed_dim_count", 0)),
        "unconditional_validity_pre_pct": uncond_pre_pct,
        "unconditional_validity_pct": uncond_post_pct,
        "unconditional_repair_pct": uncond_repair_pct,
        "conditional_validity_pre_pct": cond_pre_pct,
        "conditional_validity_pct": cond_post_pct,
        "conditional_repair_pct": cond_repair_pct,
        "protocol_accuracy_pct": round(float(proto_acc) * 100.0, 2) if proto_acc is not None else float("nan"),
    }


def _load_artifact_summary_row(
    class_name: str,
    device: str,
    manifest: dict,
) -> dict | None:
    ckpt_info = manifest.get("checkpoints", {}).get(class_name)
    diag_info = manifest.get("diagnostics", {}).get(class_name)
    if not ckpt_info or not diag_info:
        return None

    ckpt_path = Path(ckpt_info["path"])
    diag_path = Path(diag_info["path"])
    if not ckpt_path.exists() or not diag_path.exists():
        return None

    with open(diag_path, encoding="utf-8") as f:
        diag_result = json.load(f)

    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    val_history = ckpt.get("val_history", {})
    final_kl = float("nan")
    if isinstance(val_history, dict) and val_history.get("kl"):
        final_kl = float(val_history["kl"][-1])

    metrics = {
        "class_name": class_name,
        "n_train": int(ckpt_info.get("n_train", 0)),
        "n_val": int(ckpt_info.get("n_val", 0)),
        "best_val_loss": float(ckpt_info.get("best_val_loss", float("nan"))),
        "final_kl": final_kl,
        "latent_dim": int(ckpt_info.get("latent_dim", 16)),
    }
    return _build_summary_row(metrics, diag_result)


def _rows_to_markdown(rows: list[dict], cols: list[str]) -> str:
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    lines = [header, sep]
    for row in rows:
        line = "| " + " | ".join(str(row.get(c, "")) for c in cols) + " |"
        lines.append(line)
    return "\n".join(lines)


def _rows_to_csv(rows: list[dict], cols: list[str], out_path: Path) -> None:
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    root = _REPO_ROOT

    # ------------------------------------------------------------------
    # 1. Build config
    # ------------------------------------------------------------------
    config = deepcopy(DEFAULT_CONFIG)
    applied_config_path: Path | None = None

    if args.config:
        override_path = Path(args.config)
        if not override_path.exists():
            print(f"Config override file not found: {override_path}", file=sys.stderr)
            sys.exit(1)
        with open(override_path, encoding="utf-8") as f:
            overrides = json.load(f)
        config = _deep_update(config, overrides)
        applied_config_path = override_path

    run_tag = (args.run_tag or str(config.get("run_tag", "") or "")).strip() or None
    default_results_vae = root / "results" / "vae"
    results_dir_arg = args.results_dir or config.get("results_dir")
    if results_dir_arg:
        results_vae = _resolve_repo_path(root, results_dir_arg)
    elif run_tag:
        results_vae = default_results_vae / run_tag
    else:
        results_vae = default_results_vae
    results_vae.mkdir(parents=True, exist_ok=True)

    log_name = str(config.get("training_log_name", "training_log.txt"))
    log_path = results_vae / log_name
    _setup_logging(log_path)

    logger.info("=== VAE Orchestrator starting ===")
    logger.info("Device: %s", args.device)
    if applied_config_path is not None:
        logger.info("Applied config overrides from %s", applied_config_path)
    if run_tag:
        logger.info("Run tag: %s", run_tag)
    logger.info("Results directory: %s", results_vae)

    # ------------------------------------------------------------------
    # 2. Determine which classes to train
    # ------------------------------------------------------------------
    if args.classes:
        requested = [c.strip() for c in args.classes.split(",")]
        unknown = [c for c in requested if c not in CLASS_TO_ID]
        if unknown:
            logger.error("Unknown class names: %s. Valid: %s", unknown, CLASSES)
            sys.exit(1)
        # Sort by class ID to maintain deterministic ordering
        classes_to_train = sorted(requested, key=lambda c: CLASS_TO_ID[c])
    else:
        classes_to_train = list(CLASSES)  # already in ID order

    logger.info("Classes to train (in order): %s", classes_to_train)

    # ------------------------------------------------------------------
    # 3. Load shared data once (for building val_ds per class)
    # ------------------------------------------------------------------
    logger.info("Loading shared val arrays ...")
    X_train = np.load(str(root / "data" / "processed" / "X_train.npy"))
    y_train_34 = np.load(str(root / "data" / "processed" / "y_train.npy"))
    X_val = np.load(str(root / "data" / "processed" / "X_val.npy"))
    y_val_34 = np.load(str(root / "data" / "processed" / "y_val.npy"))

    with open(str(root / "data" / "processed" / "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)

    y_val_8 = _load_8class_labels(root, y_val_34, "val")
    y_train_8 = _load_8class_labels(root, y_train_34, "train")
    partition = get_partition()
    shared_arrays = {
        "X_train": X_train,
        "y_train_34": y_train_34,
        "X_val": X_val,
        "y_val_34": y_val_34,
        "y_train_8": y_train_8,
        "y_val_8": y_val_8,
        "scaler": scaler,
    }

    # ------------------------------------------------------------------
    # 4. Load manifest
    # ------------------------------------------------------------------
    manifest_path_arg = args.manifest_path or config.get("manifest_path")
    if manifest_path_arg:
        manifest_path = _resolve_repo_path(root, manifest_path_arg)
    elif results_vae != default_results_vae:
        manifest_path = results_vae / "vae_run_manifest.json"
    else:
        manifest_path = root / "vae_run_manifest.json"
    logger.info("Manifest path: %s", manifest_path)
    manifest = _load_manifest(manifest_path)

    # ------------------------------------------------------------------
    # 5. Train loop
    # ------------------------------------------------------------------
    summary_rows: list[dict] = []

    for class_name in classes_to_train:
        class_id = CLASS_TO_ID[class_name]
        logger.info(
            "========== Training class %d/%s ==========",
            class_id, class_name,
        )

        if args.diagnostics_only:
            metrics_row = _load_artifact_summary_row(class_name, args.device, manifest)
            if metrics_row is None:
                logger.error(
                    "Diagnostics-only mode requires existing checkpoint and manifest entry for %s",
                    class_name,
                )
                sys.exit(1)
            metrics = {
                "class_name": class_name,
                "n_train": metrics_row["n_train"],
                "n_val": metrics_row["n_val"],
                "best_val_loss": metrics_row["best_val_loss"],
                "final_kl": metrics_row["final_kl"],
                "latent_dim": metrics_row["latent_dim"],
                "epochs_trained": int(manifest["checkpoints"][class_name].get("epochs_trained", 0)),
            }
        else:
            class_config = _config_for_class(
                config,
                class_id=class_id,
                class_name=class_name,
                run_tag=run_tag,
                results_dir=results_vae,
                default_results_dir=default_results_vae,
            )
            metrics = train_one_vae(
                class_id,
                class_name,
                class_config,
                args.device,
                shared_arrays=shared_arrays,
            )

        # (a/b) Reload best checkpoint
        if args.diagnostics_only:
            ckpt_path = Path(manifest["checkpoints"][class_name]["path"])
        else:
            ckpt_path = Path(
                metrics.get(
                    "checkpoint_path",
                    root / "models" / "vae" / f"vae_class_{class_id}_{class_name}.pt",
                )
            )
        logger.info("Reloading checkpoint from %s", ckpt_path)
        ckpt = torch.load(str(ckpt_path), map_location=args.device, weights_only=False)

        ckpt_config = ckpt.get("config", config)
        (
            latent_dim,
            encoder_hidden,
            decoder_hidden,
            protocol_embed_dim,
            use_structured_continuous_decoder,
            use_structured_physics_decoder,
            structured_continuous_mode,
            structured_std_floor,
            latent_logvar_bounds,
        ) = _resolve_model_hparams(
            ckpt_config,
            class_name,
        )

        model = MixedInputBetaVAE(
            partition=partition,
            latent_dim=latent_dim,
            protocol_embed_dim=protocol_embed_dim,
            encoder_hidden=encoder_hidden,
            decoder_hidden=decoder_hidden,
            n_pseudo_binary=0,
            use_structured_continuous_decoder=use_structured_continuous_decoder,
            use_structured_physics_decoder=use_structured_physics_decoder,
            structured_continuous_mode=structured_continuous_mode,
            structured_std_floor=structured_std_floor,
            latent_logvar_bounds=latent_logvar_bounds,
        )
        load_result = model.load_state_dict(ckpt["state_dict"], strict=False)
        if load_result.missing_keys or load_result.unexpected_keys:
            logger.info(
                "Checkpoint compatibility note for %s: missing_keys=%s unexpected_keys=%s",
                class_name,
                load_result.missing_keys,
                load_result.unexpected_keys,
            )
        model.register_protocol_references(scaler)
        model = model.to(args.device)
        model.eval()

        # (c) Build val_ds and run diagnostics
        val_ds = PerClassDataset(X_val, y_val_8, class_id, scaler, partition)
        diag_result = run_diagnostics(
            class_id, class_name, model, val_ds, scaler, partition, args.device,
            continuous_likelihood=str(ckpt_config.get("continuous_likelihood", "gaussian")),
            output_dir=results_vae,
        )

        # Diagnostics output path
        diag_path = results_vae / f"diagnostics_{class_name}.json"

        # (d) Update manifest
        manifest["checkpoints"][class_name] = {
            "path": str(ckpt_path),
            "sha256": compute_sha256(ckpt_path),
            "best_val_loss": metrics["best_val_loss"],
            "epochs_trained": metrics["epochs_trained"],
            "n_train": metrics["n_train"],
            "n_val": metrics["n_val"],
            "latent_dim": latent_dim,
        }
        manifest["diagnostics"][class_name] = {
            "path": str(diag_path),
            "collapsed_dim_count": diag_result["posterior_collapse"]["collapsed_dim_count"],
            "unconditional_validity_pre_postprocess": diag_result["unconditional_validity"].get("pre_postprocess_validity_rate"),
            "unconditional_validity_raw": diag_result["unconditional_validity"].get("overall_validity_rate_raw"),
            "unconditional_validity": diag_result["unconditional_validity"]["overall_validity_rate"],
            "unconditional_validity_postprocess": diag_result["unconditional_validity"].get("overall_validity_rate_postprocess"),
            "unconditional_postprocess_repair_rate": diag_result["unconditional_validity"].get("postprocess_repair_rate"),
            "conditional_validity_pre_postprocess": diag_result["conditional_validity"].get("pre_postprocess_validity_rate"),
            "conditional_validity_raw": diag_result["conditional_validity"].get("overall_validity_rate_raw"),
            "conditional_validity": diag_result["conditional_validity"]["overall_validity_rate"],
            "conditional_validity_postprocess": diag_result["conditional_validity"].get("overall_validity_rate_postprocess"),
            "conditional_postprocess_repair_rate": diag_result["conditional_validity"].get("postprocess_repair_rate"),
            "protocol_accuracy": diag_result["per_feature_recon"]["protocol_top1_accuracy"],
        }
        _save_manifest(manifest, manifest_path)
        logger.info("Manifest updated for class %s", class_name)

        # (e) Accumulate summary row
        row = _build_summary_row(metrics, diag_result)
        summary_rows.append(row)

        logger.info(
            "Class %s done: best_val_loss=%.4f epochs=%d "
            "collapsed=%d uncond_pre=%.2f%% uncond_post=%.2f%% cond_pre=%.2f%% cond_post=%.2f%% proto_acc=%.2f%%",
            class_name,
            metrics["best_val_loss"],
            metrics["epochs_trained"],
            row["collapsed_dims"],
            row["unconditional_validity_pre_pct"] if not math.isnan(row["unconditional_validity_pre_pct"]) else float("nan"),
            row["unconditional_validity_pct"] if not math.isnan(row["unconditional_validity_pct"]) else float("nan"),
            row["conditional_validity_pre_pct"] if not math.isnan(row["conditional_validity_pre_pct"]) else float("nan"),
            row["conditional_validity_pct"] if not math.isnan(row["conditional_validity_pct"]) else float("nan"),
            row["protocol_accuracy_pct"] if not math.isnan(row["protocol_accuracy_pct"]) else float("nan"),
        )

    # ------------------------------------------------------------------
    # 6. Summary table
    # ------------------------------------------------------------------
    if len(summary_rows) < len(CLASSES):
        seen_classes = {row["Class"] for row in summary_rows}
        for class_name in CLASSES:
            if class_name in seen_classes:
                continue
            row = _load_artifact_summary_row(class_name, args.device, manifest)
            if row is not None:
                summary_rows.append(row)

    summary_rows.sort(key=lambda row: CLASS_TO_ID[row["Class"]])

    logger.info("===== Training Summary =====")
    header_line = " | ".join(f"{c:>30}" if c == "Class" else f"{c}" for c in _SUMMARY_COLS)
    logger.info(header_line)
    for row in summary_rows:
        vals = " | ".join(str(row.get(c, "")) for c in _SUMMARY_COLS)
        logger.info(vals)

    # Save Markdown
    summary_md_path = results_vae / "summary.md"
    md_content = (
        "# VAE Training Summary\n\n"
        "Important: treat `*_validity_pre_pct` as the honest decoder metric. "
        "`*_validity_pre_pct` is validity in raw space immediately after "
        "`scaler.inverse_transform`, before any repair. `*_validity_pct` is validity after "
        "`raw_postprocess()`, and `*_repair_pct` is the fraction of samples rescued by that "
        "repair step.\n\n"
        + _rows_to_markdown(summary_rows, _SUMMARY_COLS)
        + "\n"
    )
    with open(summary_md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    logger.info("Summary Markdown saved to %s", summary_md_path)

    # Save CSV
    summary_csv_path = results_vae / "summary.csv"
    _rows_to_csv(summary_rows, _SUMMARY_COLS, summary_csv_path)
    logger.info("Summary CSV saved to %s", summary_csv_path)

    # ------------------------------------------------------------------
    # 7. Phase 10 gate checks
    # ------------------------------------------------------------------
    # Reload manifest to pick up any updates from current run
    manifest = _load_manifest(manifest_path)
    gates = _check_gates(summary_rows, config, manifest, summary_md_path)

    logger.info("===== Phase 10 Gate Checks =====")
    all_pass = True
    for gate_name, passed in gates.items():
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        logger.info("  %-45s %s", gate_name, status)

    if all_pass:
        logger.info("All Phase 10 gates PASSED.")
    else:
        logger.warning("One or more Phase 10 gates FAILED — review logs above.")

    logger.info("=== VAE Orchestrator complete ===")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train all 8 per-class β-VAEs and run diagnostics."
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="PyTorch device string, e.g. 'cpu' or 'cuda' (default: cpu)",
    )
    parser.add_argument(
        "--classes",
        default=None,
        help=(
            "Comma-separated class names to train, e.g. 'Benign,DDoS'. "
            "Default: all 8 classes."
        ),
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to a JSON file with config overrides (merged on top of DEFAULT_CONFIG).",
    )
    parser.add_argument(
        "--run-tag",
        default=None,
        help=(
            "Optional tag for non-destructive reruns. When set, checkpoints get "
            "the tag in their filename and outputs go under results/vae/<tag>."
        ),
    )
    parser.add_argument(
        "--results-dir",
        default=None,
        help=(
            "Optional output directory for logs, diagnostics, summary files, and "
            "curves. Relative paths are resolved from the repository root."
        ),
    )
    parser.add_argument(
        "--manifest-path",
        default=None,
        help=(
            "Optional manifest path. Relative paths are resolved from the "
            "repository root. Defaults to vae_run_manifest.json unless a tagged "
            "or custom results directory is used."
        ),
    )
    parser.add_argument(
        "--diagnostics-only",
        action="store_true",
        help="Skip training, reload existing checkpoints, and refresh diagnostics/summary artifacts only.",
    )
    args = parser.parse_args()
    main(args)
