"""
Reconstruction accuracy for the 8 per-class MixedInputBetaVAE models.

Unlike diagnostics.py (which reports per-feature NLL/BCE — the *training*
likelihood, not an interpretable accuracy), this script measures how faithfully
each VAE reconstructs a held-out validation sample under the deterministic
posterior-mean reconstruction (z = mu, no sampling):

    x_scaled --encode--> mu --decode_to_39(hard)--> x_recon

Metrics are decomposed by feature type, mirroring the model's four decoder heads:

  Continuous (23 feats, raw space):
      R^2, RMSE, MAE, NRMSE (RMSE / std of the true feature).
      Headline = mean R^2 over non-degenerate continuous features.

  Independent binary (11 feats) + derived binary (4 feats):
      exact bit accuracy = fraction of decoded bits equal to the true bit.

  Protocol Type (categorical, 6-way allowlist):
      top-1 accuracy = fraction with argmax(protocol_logits) == true index.

  Aggregate (per class):
      binary_mean_accuracy (15 binary cols), protocol_accuracy,
      categorical_exact_match (all 11 independent bits AND protocol correct).

The canonical checkpoint set is whatever ``vae_run_manifest.json`` points at —
i.e. the exact models the latent attacks load via AttackRouter.get_vae.

Outputs:
  results/vae/reconstruction_accuracy.json          (full per-feature detail)
  results/vae/reconstruction_accuracy_summary.csv   (per-class headline table)
  results/vae/reconstruction_accuracy_summary.md    (same, Markdown)

Usage:
  python src/vae/reconstruction_accuracy.py --device cuda
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

# --- Path bootstrap: ensure src/ is importable (mirrors train_all.py) ---
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = str(_REPO_ROOT / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from preprocessing.feature_groups import FEATURE_NAMES  # noqa: E402
from vae.config import CLASS_TO_ID, CLASSES  # noqa: E402
from vae.dataset import PerClassDataset  # noqa: E402
from vae.model import PROTOCOL_REFERENCE_BUFFERS, MixedInputBetaVAE  # noqa: E402
from vae.schema import derive_binaries_from_protocol_index, get_partition  # noqa: E402
from vae.train import _load_8class_labels  # noqa: E402
from vae.train_all import _resolve_model_hparams  # noqa: E402

logger = logging.getLogger(__name__)

_EPS = 1e-12


# ---------------------------------------------------------------------------
# Model loading (mirrors AttackRouter.get_vae — strict load against manifest)
# ---------------------------------------------------------------------------

def _load_vae(
    class_name: str,
    manifest: dict,
    partition: dict,
    scaler,
    device: str,
) -> MixedInputBetaVAE:
    ckpt_info = manifest["checkpoints"][class_name]
    ckpt_path = Path(ckpt_info["path"])
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    ckpt_config = ckpt.get("config", {})

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
    ) = _resolve_model_hparams(ckpt_config, class_name)

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
    missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
    missing_real = [k for k in missing if k not in PROTOCOL_REFERENCE_BUFFERS]
    if missing_real or unexpected:
        raise RuntimeError(
            f"State dict mismatch loading VAE for class {class_name}: "
            f"missing={missing_real}, unexpected={list(unexpected)}"
        )
    model.register_protocol_references(scaler)
    model = model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Per-class reconstruction accuracy
# ---------------------------------------------------------------------------

def _reconstruction_accuracy_for_class(
    class_id: int,
    class_name: str,
    model: MixedInputBetaVAE,
    val_ds: PerClassDataset,
    scaler,
    partition: dict,
    device: str,
    batch_size: int = 4096,
) -> dict:
    continuous_idx = partition["continuous_idx"]          # 23
    independent_binary_idx = partition["independent_binary_idx"]  # 11
    derived_binary_idx = partition["derived_binary_idx"]  # 4
    n_cont = len(continuous_idx)
    n_ind_bin = len(independent_binary_idx)
    n_der_bin = len(derived_binary_idx)

    # Continuous streaming accumulators (raw space, float64).
    sum_abs_err = np.zeros(n_cont, dtype=np.float64)
    sum_sq_err = np.zeros(n_cont, dtype=np.float64)
    sum_true = np.zeros(n_cont, dtype=np.float64)
    sum_true_sq = np.zeros(n_cont, dtype=np.float64)

    # Categorical accumulators.
    ind_bin_correct = np.zeros(n_ind_bin, dtype=np.int64)
    der_bin_correct = np.zeros(n_der_bin, dtype=np.int64)
    proto_correct = 0
    exact_match = 0  # all 11 independent bits AND protocol correct
    n_total = 0

    loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    with torch.no_grad():
        for batch in loader:
            x = batch["x_scaled"].to(device)
            target_ind_bin = batch["target_ind_binary"].to(device)   # (B, 11) {0,1}
            target_proto_idx = batch["target_proto_idx"].to(device)  # (B,) long
            B = x.shape[0]

            # Deterministic reconstruction: z = posterior mean.
            mu, _ = model.encode(x)
            x_recon_scaled, meta = model.decode_to_39(mu, mode="hard")

            # --- Continuous (raw space) ---
            true_raw = scaler.inverse_transform(x.detach().cpu().numpy().astype(np.float64))
            recon_raw = scaler.inverse_transform(
                x_recon_scaled.detach().cpu().numpy().astype(np.float64)
            )
            true_c = true_raw[:, continuous_idx]
            recon_c = recon_raw[:, continuous_idx]
            err = recon_c - true_c
            sum_abs_err += np.abs(err).sum(axis=0)
            sum_sq_err += (err ** 2).sum(axis=0)
            sum_true += true_c.sum(axis=0)
            sum_true_sq += (true_c ** 2).sum(axis=0)

            # --- Independent binary (logits > 0 => bit 1) ---
            pred_bits = (meta["binary_logits"] > 0.0).to(target_ind_bin.dtype)  # (B, 11)
            ind_correct_mat = pred_bits == target_ind_bin                       # (B, 11)
            ind_bin_correct += ind_correct_mat.sum(dim=0).cpu().numpy().astype(np.int64)

            # --- Protocol top-1 ---
            pred_proto_idx = meta["protocol_idx_batch"]  # (B,) long
            proto_correct_vec = pred_proto_idx == target_proto_idx
            proto_correct += int(proto_correct_vec.sum().item())

            # --- Derived binary (deterministic fn of protocol index) ---
            pred_der = derive_binaries_from_protocol_index(pred_proto_idx)    # (B, 4)
            true_der = derive_binaries_from_protocol_index(target_proto_idx)  # (B, 4)
            der_bin_correct += (pred_der == true_der).sum(dim=0).cpu().numpy().astype(np.int64)

            # --- Exact categorical row match (independent bits + protocol) ---
            all_ind_ok = ind_correct_mat.all(dim=1)  # (B,)
            exact_match += int((all_ind_ok & proto_correct_vec).sum().item())

            n_total += B

    n = float(n_total)

    # --- Continuous per-feature metrics ---
    mae = sum_abs_err / n
    mse = sum_sq_err / n
    rmse = np.sqrt(mse)
    true_mean = sum_true / n
    ss_res = sum_sq_err
    ss_tot = sum_true_sq - n * (true_mean ** 2)  # = sum((true - mean)^2)
    var_true = ss_tot / n
    std_true = np.sqrt(np.maximum(var_true, 0.0))

    cont_features: dict[str, dict] = {}
    r2_valid: list[float] = []
    nrmse_valid: list[float] = []
    for i, col in enumerate(continuous_idx):
        fname = FEATURE_NAMES[col]
        degenerate = ss_tot[i] <= _EPS  # feature ~constant across val split
        r2 = None if degenerate else float(1.0 - ss_res[i] / ss_tot[i])
        nrmse = None if std_true[i] <= _EPS else float(rmse[i] / std_true[i])
        cont_features[fname] = {
            "r2": r2,
            "rmse": float(rmse[i]),
            "mae": float(mae[i]),
            "nrmse": nrmse,
            "true_std": float(std_true[i]),
            "degenerate": bool(degenerate),
        }
        if r2 is not None:
            r2_valid.append(r2)
        if nrmse is not None:
            nrmse_valid.append(nrmse)

    cont_mean_r2 = float(np.mean(r2_valid)) if r2_valid else None
    cont_median_r2 = float(np.median(r2_valid)) if r2_valid else None
    cont_mean_nrmse = float(np.mean(nrmse_valid)) if nrmse_valid else None
    cont_median_nrmse = float(np.median(nrmse_valid)) if nrmse_valid else None
    # Mean R^2 is dominated by a handful of near-constant features (tiny SS_tot
    # => huge negative R^2), so report robust aggregates as the headline:
    # the median and the fraction of features reconstructed with R^2 >= 0.5.
    cont_frac_r2_ge_0p5 = (
        float(np.mean([r >= 0.5 for r in r2_valid])) if r2_valid else None
    )

    # --- Binary per-feature accuracy ---
    ind_bin_acc = ind_bin_correct / n
    der_bin_acc = der_bin_correct / n
    ind_bin_features = {
        FEATURE_NAMES[col]: float(ind_bin_acc[i])
        for i, col in enumerate(independent_binary_idx)
    }
    der_bin_features = {
        FEATURE_NAMES[col]: float(der_bin_acc[i])
        for i, col in enumerate(derived_binary_idx)
    }

    all_bin_acc = np.concatenate([ind_bin_acc, der_bin_acc])
    binary_mean_accuracy = float(all_bin_acc.mean())
    protocol_accuracy = float(proto_correct / n)
    categorical_exact_match = float(exact_match / n)

    return {
        "class_id": int(class_id),
        "class_name": class_name,
        "n_val": int(n_total),
        "headline": {
            "continuous_median_r2": cont_median_r2,
            "continuous_frac_r2_ge_0p5": cont_frac_r2_ge_0p5,
            "continuous_median_nrmse": cont_median_nrmse,
            "continuous_mean_r2": cont_mean_r2,
            "continuous_mean_nrmse": cont_mean_nrmse,
            "continuous_n_features_scored": int(len(r2_valid)),
            "binary_mean_accuracy": binary_mean_accuracy,
            "independent_binary_mean_accuracy": float(ind_bin_acc.mean()),
            "derived_binary_mean_accuracy": float(der_bin_acc.mean()),
            "protocol_accuracy": protocol_accuracy,
            "categorical_exact_match": categorical_exact_match,
        },
        "continuous_features": cont_features,
        "independent_binary_features": ind_bin_features,
        "derived_binary_features": der_bin_features,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

_SUMMARY_COLS = [
    "class_name",
    "n_val",
    "continuous_median_r2",
    "continuous_frac_r2_ge_0p5",
    "continuous_median_nrmse",
    "binary_mean_accuracy",
    "protocol_accuracy",
    "categorical_exact_match",
]


def _fmt(v) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def main() -> None:
    parser = argparse.ArgumentParser(description="VAE reconstruction accuracy.")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="PyTorch device string (default: cuda if available, else cpu).",
    )
    parser.add_argument(
        "--classes",
        default=None,
        help="Comma-separated class names (default: all 8).",
    )
    parser.add_argument(
        "--manifest",
        default=str(_REPO_ROOT / "vae_run_manifest.json"),
        help="Path to the VAE run manifest (default: repo-root manifest).",
    )
    parser.add_argument("--batch-size", type=int, default=4096)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    device = args.device
    logger.info("Device: %s", device)

    root = _REPO_ROOT
    with open(args.manifest, encoding="utf-8") as f:
        manifest = json.load(f)

    X_val = np.load(str(root / "data" / "processed" / "X_val.npy"))
    y_val_34 = np.load(str(root / "data" / "processed" / "y_val.npy"))
    with open(str(root / "data" / "processed" / "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    y_val_8 = _load_8class_labels(root, y_val_34, "val")
    partition = get_partition()

    if args.classes:
        requested = [c.strip() for c in args.classes.split(",")]
        classes = sorted(requested, key=lambda c: CLASS_TO_ID[c])
    else:
        classes = list(CLASSES)

    per_class: list[dict] = []
    for class_name in classes:
        class_id = CLASS_TO_ID[class_name]
        logger.info("=== Class %d (%s) ===", class_id, class_name)
        model = _load_vae(class_name, manifest, partition, scaler, device)
        val_ds = PerClassDataset(X_val, y_val_8, class_id, scaler, partition)
        res = _reconstruction_accuracy_for_class(
            class_id, class_name, model, val_ds, scaler, partition, device,
            batch_size=args.batch_size,
        )
        h = res["headline"]
        logger.info(
            "%s: n=%d  cont_R2(median)=%s  frac_R2>=0.5=%s  bin_acc=%.4f  proto_acc=%.4f  cat_exact=%.4f",
            class_name, res["n_val"],
            _fmt(h["continuous_median_r2"]), _fmt(h["continuous_frac_r2_ge_0p5"]),
            h["binary_mean_accuracy"], h["protocol_accuracy"], h["categorical_exact_match"],
        )
        per_class.append(res)

    # --- Sample-weighted overall aggregate ---
    total_n = sum(r["n_val"] for r in per_class)

    def _wmean(key: str) -> float | None:
        vals = [(r["n_val"], r["headline"][key]) for r in per_class if r["headline"][key] is not None]
        if not vals:
            return None
        w = sum(n for n, _ in vals)
        return float(sum(n * v for n, v in vals) / w) if w > 0 else None

    overall = {
        "total_n_val": int(total_n),
        "continuous_median_r2": _wmean("continuous_median_r2"),
        "continuous_frac_r2_ge_0p5": _wmean("continuous_frac_r2_ge_0p5"),
        "continuous_median_nrmse": _wmean("continuous_median_nrmse"),
        "continuous_mean_r2": _wmean("continuous_mean_r2"),
        "continuous_mean_nrmse": _wmean("continuous_mean_nrmse"),
        "binary_mean_accuracy": _wmean("binary_mean_accuracy"),
        "protocol_accuracy": _wmean("protocol_accuracy"),
        "categorical_exact_match": _wmean("categorical_exact_match"),
    }

    out_dir = root / "results" / "vae"
    out_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "description": "Per-class VAE reconstruction accuracy (deterministic posterior-mean reconstruction on the validation split).",
        "reconstruction": "z = mu (no sampling); decode_to_39(mode='hard'); continuous compared in raw space.",
        "device": device,
        "manifest": str(args.manifest),
        "overall_sample_weighted": overall,
        "per_class": per_class,
    }
    json_path = out_dir / "reconstruction_accuracy.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    logger.info("Wrote %s", json_path)

    # --- Summary CSV + MD ---
    summary_rows = []
    for r in per_class:
        h = r["headline"]
        summary_rows.append({
            "class_name": r["class_name"],
            "n_val": r["n_val"],
            "continuous_median_r2": h["continuous_median_r2"],
            "continuous_frac_r2_ge_0p5": h["continuous_frac_r2_ge_0p5"],
            "continuous_median_nrmse": h["continuous_median_nrmse"],
            "binary_mean_accuracy": h["binary_mean_accuracy"],
            "protocol_accuracy": h["protocol_accuracy"],
            "categorical_exact_match": h["categorical_exact_match"],
        })
    summary_rows.append({
        "class_name": "OVERALL (weighted)",
        "n_val": overall["total_n_val"],
        "continuous_median_r2": overall["continuous_median_r2"],
        "continuous_frac_r2_ge_0p5": overall["continuous_frac_r2_ge_0p5"],
        "continuous_median_nrmse": overall["continuous_median_nrmse"],
        "binary_mean_accuracy": overall["binary_mean_accuracy"],
        "protocol_accuracy": overall["protocol_accuracy"],
        "categorical_exact_match": overall["categorical_exact_match"],
    })

    csv_path = out_dir / "reconstruction_accuracy_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_SUMMARY_COLS)
        writer.writeheader()
        writer.writerows(summary_rows)
    logger.info("Wrote %s", csv_path)

    md_lines = [
        "# VAE Reconstruction Accuracy",
        "",
        "Deterministic posterior-mean reconstruction (`z = mu`, no sampling) on the "
        "validation split. Continuous features compared in raw space; binary/protocol "
        "compared as exact categorical matches.",
        "",
        "| " + " | ".join(_SUMMARY_COLS) + " |",
        "| " + " | ".join(["---"] * len(_SUMMARY_COLS)) + " |",
    ]
    for row in summary_rows:
        md_lines.append("| " + " | ".join(_fmt(row[c]) for c in _SUMMARY_COLS) + " |")
    md_lines.append("")
    md_lines.append(
        "- **continuous_median_r2**: median coefficient of determination across "
        "non-degenerate continuous features (1.0 = perfect). Median, not mean, "
        "because a few near-constant features (tiny variance) give hugely negative "
        "R^2 that dominates the mean — see `continuous_mean_r2` in the JSON for the raw mean.\n"
        "- **continuous_frac_r2_ge_0p5**: fraction of continuous features reconstructed with R^2 >= 0.5.\n"
        "- **continuous_median_nrmse**: median RMSE normalised by each feature's std (lower is better).\n"
        "- **binary_mean_accuracy**: mean exact-bit accuracy over 11 independent + 4 derived binaries.\n"
        "- **protocol_accuracy**: protocol-type top-1 reconstruction accuracy.\n"
        "- **categorical_exact_match**: fraction of samples with all 11 independent bits AND protocol correct."
    )
    md_path = out_dir / "reconstruction_accuracy_summary.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    logger.info("Wrote %s", md_path)

    logger.info("=== DONE ===")
    print("\n=== VAE Reconstruction Accuracy (validation split) ===")
    print("| " + " | ".join(_SUMMARY_COLS) + " |")
    for row in summary_rows:
        print("| " + " | ".join(_fmt(row[c]) for c in _SUMMARY_COLS) + " |")


if __name__ == "__main__":
    main()
