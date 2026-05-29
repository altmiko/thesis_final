"""
Post-training diagnostics for MixedInputBetaVAE.
All outputs saved to results/vae/diagnostics_{class_name}.json.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch.utils.data import DataLoader

if TYPE_CHECKING:
    from sklearn.preprocessing import RobustScaler

    from vae.dataset import PerClassDataset
    from vae.model import MixedInputBetaVAE

logger = logging.getLogger(__name__)

RULE_GROUP_PREFIXES: dict[str, tuple[str, ...]] = {
    "G1_nonnegativity": ("R_nonneg_",),
    "G2_protocol_allowlist": ("R_protocol_valid",),
    "G3_binary_constraints": ("R_binary_",),
    "G4_protocol_indicator_consistency": ("R_proto_",),
    "G5_statistical_ordering": ("R_min_leq_max", "R_avg_in_range"),
    "G6_variance_std_consistency": ("R_var_eq_std_sq",),
    "G7_ttl_range": ("R_ttl_range",),
    "G8_packet_count": ("R_pkts_positive", "R_pkts_integer"),
}

# Canonical 39-dim CICIoT2023 feature order. Imported from the single source of
# truth (preprocessing.feature_groups) so this module, the validator, and
# preprocessing can never drift out of sync.
from preprocessing.feature_groups import FEATURE_NAMES  # noqa: E402

assert len(FEATURE_NAMES) == 39, f"Expected 39 feature names, got {len(FEATURE_NAMES)}"


# ---------------------------------------------------------------------------
# JSON serialization helpers
# ---------------------------------------------------------------------------

class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that converts numpy scalars and arrays to Python native types."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def _to_python(obj: Any) -> Any:
    """Recursively convert numpy types to Python native types."""
    if isinstance(obj, dict):
        return {k: _to_python(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_python(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, torch.Tensor):
        return obj.cpu().numpy().tolist()
    return obj


# ---------------------------------------------------------------------------
# Validator import
# ---------------------------------------------------------------------------

def _load_validator():
    """Import the validate_batch function from the attack validator.

    Returns (validate_batch_fn, feature_names) or (None, None) if not importable.
    The validator is at src/attack/validator.py. It takes (X: np.ndarray, feature_names: List[str])
    and returns a ValidationResult with per_rule_violation_rate() and validity_rate.
    """
    try:
        from attack.validator import validate_batch
        logger.info("Loaded validator via attack.validator.validate_batch")
        return validate_batch
    except ImportError:
        pass

    try:
        import importlib.util
        repo_root = Path(__file__).resolve().parents[2]
        spec = importlib.util.spec_from_file_location(
            "validator",
            repo_root / "src" / "attack" / "validator.py",
        )
        mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        logger.info("Loaded validator via direct file import")
        return mod.validate_batch
    except Exception as exc:
        logger.warning("Could not import validator: %s", exc)
        return None


def _group_rule_pass_rates(per_rule_pass_rate: dict[str, float]) -> dict[str, float]:
    grouped: dict[str, float] = {}
    for group_name, prefixes in RULE_GROUP_PREFIXES.items():
        matched = [
            pass_rate
            for rule_name, pass_rate in per_rule_pass_rate.items()
            if any(rule_name.startswith(prefix) for prefix in prefixes)
        ]
        if matched:
            grouped[group_name] = float(sum(matched) / len(matched))
    return grouped


def _summarize_validation(validate_batch_fn, raw_samples: np.ndarray) -> dict[str, Any]:
    if validate_batch_fn is None:
        return {
            "per_rule_pass_rate": {},
            "group_pass_rate": {},
            "overall_validity_rate": None,
            "n_valid": None,
            "n_invalid": None,
        }

    result = validate_batch_fn(raw_samples, FEATURE_NAMES)
    viol_rates = result.per_rule_violation_rate()
    per_rule_pass_rate = {k: float(1.0 - v) for k, v in viol_rates.items()}
    overall_validity_rate = float(result.validity_rate)
    return {
        "per_rule_pass_rate": per_rule_pass_rate,
        "group_pass_rate": _group_rule_pass_rates(per_rule_pass_rate),
        "overall_validity_rate": overall_validity_rate,
        "n_valid": int(result.overall_valid.sum()),
        "n_invalid": int((~result.overall_valid).sum()),
    }


# ---------------------------------------------------------------------------
# Diagnostic 1 — Posterior collapse check
# ---------------------------------------------------------------------------

def _diag_posterior_collapse(
    model: "MixedInputBetaVAE",
    val_ds: "PerClassDataset",
    device: str,
    collapse_threshold: float = 0.01,
) -> dict:
    """Compute per-latent-dim KL and flag collapsed dimensions."""
    loader = DataLoader(val_ds, batch_size=2048, shuffle=False, num_workers=0)

    kl_accum = torch.zeros(model.latent_dim, device=device)
    n_total = 0

    with torch.no_grad():
        for batch in loader:
            x = batch["x_scaled"].to(device)
            mu, logvar = model.encode(x)
            # KL per sample per dim: -0.5 * (1 + logvar - mu^2 - exp(logvar))
            kl = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())  # (B, latent_dim)
            kl_accum += kl.sum(dim=0)
            n_total += x.shape[0]

    per_dim_kl = (kl_accum / n_total).cpu().numpy()  # (latent_dim,)

    collapsed_mask = per_dim_kl < collapse_threshold
    collapsed_indices = [int(i) for i in np.where(collapsed_mask)[0]]

    logger.info(
        "Posterior collapse: %d/%d dims below threshold %.3f",
        len(collapsed_indices),
        model.latent_dim,
        collapse_threshold,
    )

    return {
        "per_dim_kl": per_dim_kl.tolist(),
        "collapsed_dim_count": len(collapsed_indices),
        "collapsed_dim_indices": collapsed_indices,
        "collapse_threshold": collapse_threshold,
    }


# ---------------------------------------------------------------------------
# Diagnostic 2 — Per-feature reconstruction error on val
# ---------------------------------------------------------------------------

def _diag_per_feature_recon(
    model: "MixedInputBetaVAE",
    val_ds: "PerClassDataset",
    partition: dict,
    device: str,
    continuous_likelihood: str = "gaussian",
) -> dict:
    """Compute per-feature reconstruction error on validation set."""
    continuous_idx = partition["continuous_idx"]
    independent_binary_idx = partition["independent_binary_idx"]

    n_continuous = len(continuous_idx)
    n_ind_bin = len(independent_binary_idx)

    # Accumulate per-column errors
    nll_accum = np.zeros(n_continuous, dtype=np.float64)
    bce_accum = np.zeros(n_ind_bin, dtype=np.float64)
    n_proto_correct = 0
    n_total = 0

    # Per-protocol accuracy tracking
    proto_correct_by_raw: dict[int, int] = {}
    proto_total_by_raw: dict[int, int] = {}

    loader = DataLoader(val_ds, batch_size=2048, shuffle=False, num_workers=0)

    from vae.schema import PROTOCOL_ALLOWLIST

    with torch.no_grad():
        for batch in loader:
            x = batch["x_scaled"].to(device)
            target_ind_bin = batch["target_ind_binary"].to(device)
            target_proto_idx = batch["target_proto_idx"].to(device)
            B = x.shape[0]

            out = model.forward(x)

            # --- Continuous: Gaussian NLL per column ---
            cont_mu = out["continuous_mu"]           # (B, 23)
            cont_logvar = out["continuous_logvar"]   # (B, 23)
            cont_target = x[:, continuous_idx]       # (B, 23) scaled values as targets

            # Reconstruction NLL per column, matching the trained likelihood.
            # Additive constants (log(2pi)/2, log 2) are dropped for comparability.
            if continuous_likelihood == "laplace":
                scale = torch.exp(cont_logvar)
                nll_per_sample = cont_logvar + (cont_target - cont_mu).abs() / (scale + 1e-8)
            else:
                var = torch.exp(cont_logvar)
                nll_per_sample = 0.5 * (cont_logvar + (cont_target - cont_mu).pow(2) / (var + 1e-8))
            # Mean over batch, per column
            nll_accum += nll_per_sample.sum(dim=0).cpu().numpy()

            # --- Independent binary: BCE per column ---
            binary_logits = out["binary_logits"]   # (B, 11)
            bce_per_sample = torch.nn.functional.binary_cross_entropy_with_logits(
                binary_logits, target_ind_bin, reduction="none"
            )  # (B, 11)
            bce_accum += bce_per_sample.sum(dim=0).cpu().numpy()

            # --- Protocol: top-1 accuracy ---
            proto_pred = out["protocol_logits"].argmax(dim=1)  # (B,)
            n_proto_correct += (proto_pred == target_proto_idx).sum().item()

            # Per-protocol-value accuracy
            target_np = target_proto_idx.cpu().numpy()
            pred_np = proto_pred.cpu().numpy()
            for b_idx in range(B):
                raw_val = PROTOCOL_ALLOWLIST[int(target_np[b_idx])]
                proto_total_by_raw[raw_val] = proto_total_by_raw.get(raw_val, 0) + 1
                if int(pred_np[b_idx]) == int(target_np[b_idx]):
                    proto_correct_by_raw[raw_val] = proto_correct_by_raw.get(raw_val, 0) + 1

            n_total += B

    # Average over samples
    nll_per_col = nll_accum / n_total
    bce_per_col = bce_accum / n_total
    proto_top1_acc = float(n_proto_correct / n_total) if n_total > 0 else 0.0

    # Per-protocol-value accuracy (only values that appear in val)
    proto_per_val_acc: dict[str, float] = {}
    for raw_val, total in sorted(proto_total_by_raw.items()):
        correct = proto_correct_by_raw.get(raw_val, 0)
        proto_per_val_acc[str(raw_val)] = float(correct / total) if total > 0 else 0.0

    # Map indices to feature names
    recon_nll_continuous = {
        FEATURE_NAMES[col_idx]: float(nll_per_col[i])
        for i, col_idx in enumerate(continuous_idx)
    }
    recon_bce_ind_bin = {
        FEATURE_NAMES[col_idx]: float(bce_per_col[i])
        for i, col_idx in enumerate(independent_binary_idx)
    }

    # Build unified ranking list for worst-5
    ranking: list[dict] = []
    for feat_name, val in recon_nll_continuous.items():
        ranking.append({"feature": feat_name, "type": "nll_continuous", "error": float(val)})
    for feat_name, val in recon_bce_ind_bin.items():
        ranking.append({"feature": feat_name, "type": "bce_independent_binary", "error": float(val)})

    ranking.sort(key=lambda d: d["error"], reverse=True)
    worst_5 = ranking[:5]

    logger.info(
        "Per-feature recon: protocol top-1=%.4f, worst feature=%s (%.4f)",
        proto_top1_acc,
        worst_5[0]["feature"] if worst_5 else "N/A",
        worst_5[0]["error"] if worst_5 else 0.0,
    )

    return {
        "recon_nll_continuous": recon_nll_continuous,
        "recon_bce_independent_binary": recon_bce_ind_bin,
        "protocol_top1_accuracy": proto_top1_acc,
        "protocol_per_value_accuracy": proto_per_val_acc,
        "worst_5_features": worst_5,
    }


# ---------------------------------------------------------------------------
# Diagnostic 3 — Unconditional sample validity
# ---------------------------------------------------------------------------

def _diag_unconditional_validity(
    model: "MixedInputBetaVAE",
    val_ds: "PerClassDataset",
    scaler: "RobustScaler",
    partition: dict,
    device: str,
    n_samples: int = 1000,
    validate_batch_fn=None,
    seed: int = 42,
) -> dict:
    """Sample z ~ N(0, I) and check domain validity of decoded samples."""
    from vae.schema import PROTOCOL_ALLOWLIST, protocol_index_to_raw

    # --- 1. Sample from prior ---
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    z = torch.randn(n_samples, model.latent_dim, device=device, generator=generator)

    with torch.no_grad():
        x_recon_scaled, decode_meta = model.decode_to_39(z, scaler, mode="hard")

    x_scaled_np = x_recon_scaled.cpu().numpy()  # (1000, 39)

    # --- 2. Inverse-transform to raw space and apply domain post-processing ---
    from vae.schema import raw_postprocess
    raw_samples_pre = scaler.inverse_transform(x_scaled_np.astype(np.float64))  # (1000, 39)
    raw_samples_post = raw_postprocess(raw_samples_pre)

    # --- 3. Validate in raw space, before and after postprocess ---
    try:
        pre_summary = _summarize_validation(validate_batch_fn, raw_samples_pre)
        post_summary = _summarize_validation(validate_batch_fn, raw_samples_post)
        if post_summary["overall_validity_rate"] is not None:
            logger.info(
                "Unconditional validity after raw_postprocess: %.4f overall (%d/%d valid)",
                post_summary["overall_validity_rate"],
                post_summary["n_valid"],
                n_samples,
            )
        if pre_summary["overall_validity_rate"] is not None:
            logger.info(
                "Unconditional validity before raw_postprocess: %.4f overall (%d/%d valid)",
                pre_summary["overall_validity_rate"],
                pre_summary["n_valid"],
                n_samples,
            )
    except Exception as exc:
        logger.warning("Validator call failed: %s", exc)
        pre_summary = {
            "per_rule_pass_rate": {},
            "group_pass_rate": {},
            "overall_validity_rate": None,
            "n_valid": None,
            "n_invalid": None,
        }
        post_summary = dict(pre_summary)

    # --- 4. Protocol-to-derived-binary consistency ---
    # decode_to_39 with mode='hard' always sets derived binaries deterministically
    # from the protocol argmax, so this should be 1.0 by construction.
    proto_idx_batch = decode_meta["protocol_idx_batch"].cpu().numpy()  # (1000,)

    # Column positions in the 39-dim output
    protocol_col = partition["protocol_idx"][0]   # 1
    derived_idx = partition["derived_binary_idx"]  # [22, 23, 26, 27] = TCP, UDP, ICMP, IGMP

    # In raw space: verify each sample's derived binary columns are consistent with its protocol
    # derived_binary_idx order: TCP=22, UDP=23, ICMP=26, IGMP=27
    # PROTOCOL_TO_BINARY: 6→TCP, 17→UDP, 1→ICMP, 2→IGMP
    # derived_binary_order in decode_to_39: [TCP, UDP, ICMP, IGMP] at positions derived_idx
    protocol_raw_col = np.round(raw_samples_post[:, protocol_col]).astype(int)  # (1000,)
    derived_cols_raw = np.round(raw_samples_post[:, derived_idx]).astype(int)   # (1000, 4)

    # Build expected derived binaries from protocol raw value
    _proto_to_derived_col = {6: 0, 17: 1, 1: 2, 2: 3}  # raw→col in derived_binary_order
    n_consistent = 0
    for i in range(n_samples):
        proto_val = int(protocol_raw_col[i])
        derived_row = derived_cols_raw[i]  # [TCP, UDP, ICMP, IGMP]
        expected_idx = _proto_to_derived_col.get(proto_val, None)
        consistent = True
        if expected_idx is not None:
            # If protocol has a derived binary, that column must be 1 and others 0
            for j in range(4):
                expected_val = 1 if j == expected_idx else 0
                if derived_row[j] != expected_val:
                    consistent = False
                    break
        else:
            # Protocol has no derived binary (HOPOPT=0, GRE=47): all derived must be 0
            if derived_row.any():
                consistent = False
        if consistent:
            n_consistent += 1

    proto_binary_consistency = float(n_consistent / n_samples)
    if proto_binary_consistency < 1.0:
        logger.error(
            "BUG: protocol_binary_consistency=%.6f (expected 1.0). "
            "%d/%d samples inconsistent.",
            proto_binary_consistency,
            n_samples - n_consistent,
            n_samples,
        )
    else:
        logger.info("Protocol-binary consistency: 1.0 (as expected)")

    # --- 5. Protocol distribution in generated samples ---
    proto_dist_gen: dict[str, float] = {}
    for proto_allowlist_idx in range(len(PROTOCOL_ALLOWLIST)):
        raw_val = PROTOCOL_ALLOWLIST[proto_allowlist_idx]
        count = int((proto_idx_batch == proto_allowlist_idx).sum())
        proto_dist_gen[str(raw_val)] = float(count / n_samples)

    # --- 6. Protocol distribution in the validation split (reference distribution) ---
    val_proto_indices = val_ds.target_protocol_index.numpy()
    proto_dist_val: dict[str, float] = {}
    n_val_proto = len(val_proto_indices)
    for proto_allowlist_idx in range(len(PROTOCOL_ALLOWLIST)):
        raw_val = PROTOCOL_ALLOWLIST[proto_allowlist_idx]
        count = int((val_proto_indices == proto_allowlist_idx).sum())
        if count > 0:
            proto_dist_val[str(raw_val)] = float(count / n_val_proto)

    return {
        "n_samples": n_samples,
        "validation_space": "raw_after_inverse_transform",
        "uses_raw_postprocess": True,
        "pre_postprocess": pre_summary,
        "post_postprocess": post_summary,
        "per_rule_pass_rate_raw": pre_summary["per_rule_pass_rate"],
        "group_pass_rate_raw": pre_summary["group_pass_rate"],
        "overall_validity_rate_raw": pre_summary["overall_validity_rate"],
        "per_rule_pass_rate_postprocess": post_summary["per_rule_pass_rate"],
        "group_pass_rate_postprocess": post_summary["group_pass_rate"],
        "overall_validity_rate_postprocess": post_summary["overall_validity_rate"],
        "headline_validity_stage": "post_postprocess",
        "per_rule_pass_rate": post_summary["per_rule_pass_rate"],
        "group_pass_rate": post_summary["group_pass_rate"],
        "overall_validity_rate": post_summary["overall_validity_rate"],
        "pre_postprocess_validity_rate": pre_summary["overall_validity_rate"],
        "postprocess_repair_rate": (
            float(post_summary["overall_validity_rate"] - pre_summary["overall_validity_rate"])
            if pre_summary["overall_validity_rate"] is not None
            and post_summary["overall_validity_rate"] is not None
            else None
        ),
        "protocol_binary_consistency": proto_binary_consistency,
        "protocol_distribution_generated": proto_dist_gen,
        "protocol_distribution_val": proto_dist_val,
        "raw_sample_head_before_postprocess": raw_samples_pre[:3].tolist(),
        "raw_sample_head_after_postprocess": raw_samples_post[:3].tolist(),
    }


# ---------------------------------------------------------------------------
# Diagnostic 4 — Conditional reconstruction validity
# ---------------------------------------------------------------------------

def _diag_conditional_validity(
    model: "MixedInputBetaVAE",
    val_ds: "PerClassDataset",
    scaler: "RobustScaler",
    partition: dict,
    device: str,
    max_samples: int = 1000,
    validate_batch_fn=None,
    seed: int = 42,
) -> dict:
    """Encode val samples, reparameterize, decode, and check domain validity."""

    n_use = min(len(val_ds), max_samples)

    # Collect samples
    x_list: list[torch.Tensor] = []
    n_collected = 0
    loader = DataLoader(val_ds, batch_size=512, shuffle=False, num_workers=0)
    for batch in loader:
        x_list.append(batch["x_scaled"])
        n_collected += batch["x_scaled"].shape[0]
        if n_collected >= n_use:
            break

    x_batch = torch.cat(x_list, dim=0)[:n_use].to(device)
    actual_n = x_batch.shape[0]

    with torch.no_grad():
        # Force eval mode sampling (reparameterize returns mu in eval mode)
        mu, logvar = model.encode(x_batch)
        # In eval mode reparameterize returns mu; to get stochastic samples we
        # add noise manually so the conditional validity test is meaningful
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        eps = torch.randn(mu.shape, device=device, dtype=mu.dtype, generator=generator)
        z = mu + eps * torch.exp(0.5 * logvar)
        x_recon_scaled, _ = model.decode_to_39(z, scaler, mode="hard")

    x_scaled_np = x_recon_scaled.cpu().numpy()
    from vae.schema import raw_postprocess
    raw_samples_pre = scaler.inverse_transform(x_scaled_np.astype(np.float64))
    raw_samples_post = raw_postprocess(raw_samples_pre)

    try:
        pre_summary = _summarize_validation(validate_batch_fn, raw_samples_pre)
        post_summary = _summarize_validation(validate_batch_fn, raw_samples_post)
        if post_summary["overall_validity_rate"] is not None:
            logger.info(
                "Conditional validity after raw_postprocess: %.4f overall (%d/%d valid)",
                post_summary["overall_validity_rate"],
                post_summary["n_valid"],
                actual_n,
            )
        if pre_summary["overall_validity_rate"] is not None:
            logger.info(
                "Conditional validity before raw_postprocess: %.4f overall (%d/%d valid)",
                pre_summary["overall_validity_rate"],
                pre_summary["n_valid"],
                actual_n,
            )
    except Exception as exc:
        logger.warning("Validator call failed in conditional validity: %s", exc)
        pre_summary = {
            "per_rule_pass_rate": {},
            "group_pass_rate": {},
            "overall_validity_rate": None,
            "n_valid": None,
            "n_invalid": None,
        }
        post_summary = dict(pre_summary)

    return {
        "n_samples": actual_n,
        "validation_space": "raw_after_inverse_transform",
        "uses_raw_postprocess": True,
        "pre_postprocess": pre_summary,
        "post_postprocess": post_summary,
        "per_rule_pass_rate_raw": pre_summary["per_rule_pass_rate"],
        "group_pass_rate_raw": pre_summary["group_pass_rate"],
        "overall_validity_rate_raw": pre_summary["overall_validity_rate"],
        "per_rule_pass_rate_postprocess": post_summary["per_rule_pass_rate"],
        "group_pass_rate_postprocess": post_summary["group_pass_rate"],
        "overall_validity_rate_postprocess": post_summary["overall_validity_rate"],
        "headline_validity_stage": "post_postprocess",
        "per_rule_pass_rate": post_summary["per_rule_pass_rate"],
        "group_pass_rate": post_summary["group_pass_rate"],
        "overall_validity_rate": post_summary["overall_validity_rate"],
        "pre_postprocess_validity_rate": pre_summary["overall_validity_rate"],
        "postprocess_repair_rate": (
            float(post_summary["overall_validity_rate"] - pre_summary["overall_validity_rate"])
            if pre_summary["overall_validity_rate"] is not None
            and post_summary["overall_validity_rate"] is not None
            else None
        ),
        "raw_sample_head_before_postprocess": raw_samples_pre[:3].tolist(),
        "raw_sample_head_after_postprocess": raw_samples_post[:3].tolist(),
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_diagnostics(
    class_id: int,
    class_name: str,
    model: "MixedInputBetaVAE",
    val_ds: "PerClassDataset",
    scaler: "RobustScaler",
    partition: dict,
    device: str = "cpu",
    continuous_likelihood: str = "gaussian",
    output_dir: Path | str | None = None,
    output_name: str | None = None,
) -> dict:
    """Run all post-training diagnostics for a single per-class β-VAE.

    Parameters
    ----------
    class_id:
        8-class integer label (0..7).
    class_name:
        Human-readable class name (e.g. 'Benign').
    model:
        Trained MixedInputBetaVAE, already placed on ``device``.
    val_ds:
        PerClassDataset for the validation split of this class.
    scaler:
        Fitted RobustScaler.
    partition:
        Feature partition dict from ``schema.get_partition()``.
    device:
        PyTorch device string.

    Returns
    -------
    dict with keys: 'class_id', 'class_name', 'posterior_collapse',
    'per_feature_recon', 'unconditional_validity', 'conditional_validity'.
    Also writes results/vae/diagnostics_{class_name}.json.
    """
    logger.info("=== Running diagnostics for class %d (%s) ===", class_id, class_name)

    model.eval()
    was_training = model.training  # always False after .eval() but kept for clarity
    _ = was_training

    # Load validator once
    validate_batch_fn = _load_validator()

    # --- Diagnostic 1 ---
    logger.info("[1/4] Posterior collapse check ...")
    d1 = _diag_posterior_collapse(model, val_ds, device)

    # --- Diagnostic 2 ---
    logger.info("[2/4] Per-feature reconstruction error ...")
    d2 = _diag_per_feature_recon(
        model, val_ds, partition, device, continuous_likelihood=continuous_likelihood
    )

    # --- Diagnostic 3 ---
    logger.info("[3/4] Unconditional sample validity (n=1000) ...")
    d3 = _diag_unconditional_validity(
        model, val_ds, scaler, partition, device,
        n_samples=1000,
        validate_batch_fn=validate_batch_fn,
        seed=1000 + int(class_id),
    )

    # --- Diagnostic 4 ---
    logger.info("[4/4] Conditional reconstruction validity ...")
    d4 = _diag_conditional_validity(
        model, val_ds, scaler, partition, device,
        max_samples=1000,
        validate_batch_fn=validate_batch_fn,
        seed=2000 + int(class_id),
    )

    result: dict = {
        "class_id": int(class_id),
        "class_name": str(class_name),
        "posterior_collapse": d1,
        "per_feature_recon": d2,
        "unconditional_validity": d3,
        "conditional_validity": d4,
    }

    # --- Serialize to JSON ---
    result_native = _to_python(result)

    repo_root = Path(__file__).resolve().parents[2]
    if output_dir is None:
        out_dir = repo_root / "results" / "vae"
    else:
        out_dir = Path(output_dir)
        if not out_dir.is_absolute():
            out_dir = repo_root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (output_name or f"diagnostics_{class_name}.json")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result_native, f, indent=2, cls=_NumpyEncoder)

    logger.info("Diagnostics written to %s", out_path)

    return result_native
