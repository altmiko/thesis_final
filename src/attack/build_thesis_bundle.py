"""Assemble the .npz bundle consumed by ``src/thesis_visualizations.py``.

Runs latent-PGD and latent-CW for ALL 5 classifiers (MLP, CNN, LSTM, Serial,
DualPath) on a stratified sample of the 8-class test split, harvests the
corresponding input-space PGD/CW arrays from existing
``results/attacks/attack_<model>_8class_{pgd,cw}_*.npz`` files, validates
every adversarial variant with the protocol validator, and writes a single
``.npz`` to ``results/attacks/thesis_bundle.npz``.

Per-sample arrays (X_*, z_*, evasion_mask_*, protocol_valid_*) belong to a
designated PRIMARY model (default: ``mlp``). The 5-model multi-metric and
per-category ASR tables are stored as picklable arrays under the keys
``multimetric_table`` and ``category_asr_table`` (and also dumped to sibling
CSVs for human inspection).

Run:
    python src/attack/build_thesis_bundle.py --device cuda \
        --samples-per-class 200 --primary mlp

This script is heavy. Default --samples-per-class=200 keeps t-SNE / pairwise
latent distance computations in the visualization step to minutes, not hours.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
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
    predict_labels,
    set_global_seed,
)
from attack.latent_pgd import latent_pgd_attack  # noqa: E402
from preprocessing.feature_groups import FEATURE_NAMES  # noqa: E402
from vae.config import CLASSES, ID_TO_CLASS  # noqa: E402

MODEL_KEYS = ["mlp", "cnn", "lstm", "serial", "dualpath"]
MODEL_LABELS = {
    "mlp": "MLP", "cnn": "CNN", "lstm": "LSTM",
    "serial": "CNN-LSTM", "dualpath": "DualPath",
}


# ---------------------------------------------------------------------------
# Sample selection
# ---------------------------------------------------------------------------
def stratified_indices(y: np.ndarray, per_class: int, rng: np.random.Generator) -> np.ndarray:
    picks: list[np.ndarray] = []
    for cid in np.unique(y):
        idx = np.flatnonzero(y == cid)
        if len(idx) == 0:
            continue
        take = min(per_class, len(idx))
        picks.append(rng.choice(idx, size=take, replace=False))
    out = np.concatenate(picks)
    rng.shuffle(out)
    return out


def load_classifier(name: str, device: str) -> torch.nn.Module:
    path = _REPO_ROOT / "models" / f"{name}_8class.pt"
    if not path.exists():
        raise FileNotFoundError(f"Missing classifier checkpoint: {path}")
    model = load_model(
        model_path=str(path),
        num_features=len(FEATURE_NAMES),
        num_classes=len(CLASSES),
        device=device,
    )
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Per-class latent attack: each sample is encoded/decoded by its own class VAE
# ---------------------------------------------------------------------------
def run_latent_attack_for_model(
    *,
    classifier: torch.nn.Module,
    router: AttackRouter,
    mask: PerturbationMask,
    X_scaled: np.ndarray,
    y_8: np.ndarray,
    attack: str,
    args: argparse.Namespace,
    device: str,
) -> dict[str, np.ndarray]:
    n = X_scaled.shape[0]
    X_adv = np.zeros_like(X_scaled, dtype=np.float32)
    z_orig_arr: np.ndarray | None = None
    z_adv_arr: np.ndarray | None = None
    x_recon_arr = np.zeros_like(X_scaled, dtype=np.float32)

    for cid in np.unique(y_8):
        idxs = np.flatnonzero(y_8 == cid)
        if len(idxs) == 0:
            continue
        vae = router.get_vae(int(cid))
        x_batch = torch.from_numpy(X_scaled[idxs].astype(np.float32)).to(device)
        y_batch = torch.from_numpy(y_8[idxs].astype(np.int64)).to(device)

        if attack == "pgd":
            x_adv_t, _z_dummy, meta = latent_pgd_attack(
                vae=vae, classifier=classifier, mask=mask,
                x_original=x_batch, y_true=y_batch, scaler=router.scaler,
                epsilon=args.pgd_eps, alpha=args.pgd_alpha,
                num_steps=args.pgd_steps, random_start=True, device=device,
            )
        elif attack == "cw":
            x_adv_t, _z_dummy, meta = latent_cw_attack(
                vae=vae, classifier=classifier, mask=mask,
                x_original=x_batch, y_true=y_batch, scaler=router.scaler,
                lambda_conf=args.cw_lambda, kappa=0.0,
                num_iterations=args.cw_iters, learning_rate=args.cw_lr,
                convergence_threshold=1e-5, device=device,
            )
        else:
            raise ValueError(attack)

        z_orig_t = meta["z_orig"]
        z_adv_t = meta["z_adv"]
        with torch.no_grad():
            x_recon_t, _ = vae.decode_to_39(z_orig_t, router.scaler, mode="hard")

        if z_orig_arr is None:
            latent_dim = int(z_orig_t.shape[1])
            z_orig_arr = np.zeros((n, latent_dim), dtype=np.float32)
            z_adv_arr = np.zeros((n, latent_dim), dtype=np.float32)

        X_adv[idxs] = x_adv_t.detach().cpu().numpy().astype(np.float32)
        z_orig_arr[idxs] = z_orig_t.detach().cpu().numpy().astype(np.float32)
        z_adv_arr[idxs] = z_adv_t.detach().cpu().numpy().astype(np.float32)
        x_recon_arr[idxs] = x_recon_t.detach().cpu().numpy().astype(np.float32)

    return {
        "X_adv": X_adv,
        "z_orig": z_orig_arr,
        "z_adv": z_adv_arr,
        "X_reconstructed": x_recon_arr,
    }


# ---------------------------------------------------------------------------
# Input attack harvest
# ---------------------------------------------------------------------------
def harvest_input_attacks(
    *, model_key: str, sel_idx: np.ndarray, X_test: np.ndarray,
) -> dict[str, np.ndarray] | None:
    attacks_dir = _REPO_ROOT / "results" / "attacks"
    candidates = {
        "pgd": [
            attacks_dir / f"attack_{model_key}_8class_pgd_0.30_r10.npz",
            attacks_dir / "attack_8class_pgd_0.30.npz" if model_key == "mlp" else None,
        ],
        "cw": [
            attacks_dir / f"attack_{model_key}_8class_cw_0_r10.npz",
            attacks_dir / "attack_8class_cw_0.npz" if model_key == "mlp" else None,
        ],
    }
    out: dict[str, np.ndarray] = {}
    for attack, paths in candidates.items():
        found = next((p for p in paths if p is not None and p.exists()), None)
        if found is None:
            print(f"  [WARN] no input-{attack} NPZ for {model_key}")
            return None
        d = np.load(found, allow_pickle=True)
        X_clean = d["X_clean"]; X_adv_full = d["X_adv"]
        needles = X_test[sel_idx]
        idx_in_full = _align_rows(X_clean, needles)
        if idx_in_full is None:
            print(f"  [WARN] exact alignment failed for input-{attack}/{model_key}; "
                  "using 1-NN fallback")
            idx_in_full = _nn_align(X_clean, needles)
        out[f"X_adv_input_{attack}"] = X_adv_full[idx_in_full].astype(np.float32)
        out[f"y_pred_adv_input_{attack}"] = d["y_pred_adv"][idx_in_full].astype(np.int64)
    return out


def _align_rows(haystack: np.ndarray, needles: np.ndarray) -> np.ndarray | None:
    if haystack.shape[1] != needles.shape[1]:
        return None
    h_view = haystack.view([("", haystack.dtype)] * haystack.shape[1]).ravel()
    n_view = needles.view([("", needles.dtype)] * needles.shape[1]).ravel()
    idx_map = {row.tobytes(): i for i, row in enumerate(h_view)}
    out = np.empty(len(n_view), dtype=np.int64)
    for j, row in enumerate(n_view):
        i = idx_map.get(row.tobytes())
        if i is None:
            return None
        out[j] = i
    return out


def _nn_align(haystack: np.ndarray, needles: np.ndarray) -> np.ndarray:
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=1).fit(haystack)
    _, idx = nn.kneighbors(needles)
    return idx.ravel()


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def metrics_for(
    *, X_orig: np.ndarray, X_adv: np.ndarray, y_true: np.ndarray,
    y_pred_adv: np.ndarray, protocol_valid: np.ndarray,
    mask_valid: np.ndarray | None = None,
) -> dict[str, float]:
    evasion = (y_pred_adv != y_true).astype(bool)
    pv = protocol_valid.astype(bool)
    mv = np.ones_like(pv, dtype=bool) if mask_valid is None else mask_valid.astype(bool)
    valid = pv & mv
    l2 = np.linalg.norm(X_adv - X_orig, axis=1)
    return {
        "ASR":            float(evasion.mean()),
        "ASR_Valid":      float((evasion & valid).mean()),
        "Protocol_Valid": float(pv.mean()),
        "Mask_Valid":     float(mv.mean()),
        "IDSR":           float((~evasion).mean()),
        "L2_mean":        float(l2.mean()),
    }


def mask_validity(mask: PerturbationMask, X_adv: np.ndarray, X_orig: np.ndarray) -> np.ndarray:
    verify = mask.verify(
        torch.from_numpy(X_adv.astype(np.float32)),
        torch.from_numpy(X_orig.astype(np.float32)),
    )
    return verify["all_compliant"].cpu().numpy().astype(bool)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--samples-per-class", type=int, default=200)
    p.add_argument("--primary", default="mlp", choices=MODEL_KEYS)
    p.add_argument("--pgd-eps",   type=float, default=0.5)
    p.add_argument("--pgd-alpha", type=float, default=0.05)
    p.add_argument("--pgd-steps", type=int,   default=40)
    p.add_argument("--cw-lambda", type=float, default=1.0)
    p.add_argument("--cw-iters",  type=int,   default=200)
    p.add_argument("--cw-lr",     type=float, default=0.01)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path,
                   default=_REPO_ROOT / "results" / "attacks" / "thesis_bundle.npz")
    args = p.parse_args()

    set_global_seed(args.seed)
    device = args.device
    rng = np.random.default_rng(args.seed)

    print(f"[1/6] Loading test split (device={device})")
    split = load_split("test")
    X_test = split["X"].astype(np.float32)
    y_test_8 = split["y_8"].astype(np.int64)
    scaler = split["scaler"]

    print(f"[2/6] Stratified sample: {args.samples_per_class} per class")
    sel = stratified_indices(y_test_8, args.samples_per_class, rng)
    X_orig = X_test[sel]
    y_true = y_test_8[sel]
    y_true_names = np.array([ID_TO_CLASS[int(c)] for c in y_true])
    print(f"  n_samples = {len(sel)}")

    print("[3/6] Initialising AttackRouter, ProtocolValidator, PerturbationMask")
    classifier_paths = {k: _REPO_ROOT / "models" / f"{k}_8class.pt" for k in MODEL_KEYS}
    router = AttackRouter(device=device, classifier_paths=classifier_paths)
    validator = ProtocolValidator(scaler)
    mask = PerturbationMask.from_preprocessing_artifacts()
    mask_type = ["Frozen"] * len(FEATURE_NAMES)
    for i in mask.full_indices:    mask_type[i] = "Full"
    for i in mask.partial_indices: mask_type[i] = "Partial"
    classifiers = {k: load_classifier(k, device) for k in MODEL_KEYS}

    per_model_per_sample: dict[str, dict[str, np.ndarray]] = {}
    multimetric_rows: list[dict[str, Any]] = []
    category_rows: list[dict[str, Any]] = []

    print(f"[4/6] Running latent + harvesting input attacks for {len(MODEL_KEYS)} models")
    for mk in MODEL_KEYS:
        print(f"  --- model: {mk} ---")
        clf = classifiers[mk]

        with torch.no_grad():
            y_pred_clean = predict_labels(
                clf, torch.from_numpy(X_orig), device=device
            ).numpy().astype(np.int64)

        print("    latent-PGD ...")
        lp = run_latent_attack_for_model(
            classifier=clf, router=router, mask=mask,
            X_scaled=X_orig, y_8=y_true, attack="pgd",
            args=args, device=device,
        )
        print("    latent-CW  ...")
        lc = run_latent_attack_for_model(
            classifier=clf, router=router, mask=mask,
            X_scaled=X_orig, y_8=y_true, attack="cw",
            args=args, device=device,
        )

        with torch.no_grad():
            y_lp = predict_labels(clf, torch.from_numpy(lp["X_adv"]), device=device).numpy().astype(np.int64)
            y_lc = predict_labels(clf, torch.from_numpy(lc["X_adv"]), device=device).numpy().astype(np.int64)

        inp = harvest_input_attacks(model_key=mk, sel_idx=sel, X_test=X_test)
        if inp is None:
            print(f"    [WARN] using clean placeholder for input-{mk}")
            inp = {
                "X_adv_input_pgd": X_orig.copy(),
                "X_adv_input_cw":  X_orig.copy(),
                "y_pred_adv_input_pgd": y_pred_clean.copy(),
                "y_pred_adv_input_cw":  y_pred_clean.copy(),
            }

        pv_lp = validator.validate(torch.from_numpy(lp["X_adv"])).cpu().numpy().astype(bool)
        pv_lc = validator.validate(torch.from_numpy(lc["X_adv"])).cpu().numpy().astype(bool)
        pv_ip = validator.validate(torch.from_numpy(inp["X_adv_input_pgd"])).cpu().numpy().astype(bool)
        pv_ic = validator.validate(torch.from_numpy(inp["X_adv_input_cw"])).cpu().numpy().astype(bool)
        mv_lp = mask_validity(mask, lp["X_adv"], X_orig)
        mv_lc = mask_validity(mask, lc["X_adv"], X_orig)
        mv_ip = mask_validity(mask, inp["X_adv_input_pgd"], X_orig)
        mv_ic = mask_validity(mask, inp["X_adv_input_cw"], X_orig)

        per_model_per_sample[mk] = {
            "X_adv_latent_pgd": lp["X_adv"], "z_perturbed_pgd": lp["z_adv"],
            "X_adv_latent_cw":  lc["X_adv"], "z_perturbed_cw":  lc["z_adv"],
            "z_original": lp["z_orig"], "X_reconstructed": lp["X_reconstructed"],
            "X_adv_input_pgd": inp["X_adv_input_pgd"],
            "X_adv_input_cw":  inp["X_adv_input_cw"],
            "y_pred_original":         y_pred_clean,
            "y_pred_adv_latent_pgd":   y_lp,
            "y_pred_adv_latent_cw":    y_lc,
            "y_pred_adv_input_pgd":    inp["y_pred_adv_input_pgd"],
            "y_pred_adv_input_cw":     inp["y_pred_adv_input_cw"],
            "evasion_mask_latent_pgd": (y_lp != y_true),
            "evasion_mask_latent_cw":  (y_lc != y_true),
            "evasion_mask_input_pgd":  (inp["y_pred_adv_input_pgd"] != y_true),
            "evasion_mask_input_cw":   (inp["y_pred_adv_input_cw"]  != y_true),
            "protocol_valid_latent_pgd": pv_lp,
            "protocol_valid_latent_cw":  pv_lc,
            "protocol_valid_input_pgd":  pv_ip,
            "protocol_valid_input_cw":   pv_ic,
            "mask_valid_latent_pgd": mv_lp,
            "mask_valid_latent_cw":  mv_lc,
            "mask_valid_input_pgd":  mv_ip,
            "mask_valid_input_cw":   mv_ic,
        }

        runs = {
            "latent_pgd": (lp["X_adv"],            y_lp,                        pv_lp, mv_lp),
            "latent_cw":  (lc["X_adv"],            y_lc,                        pv_lc, mv_lc),
            "input_pgd":  (inp["X_adv_input_pgd"], inp["y_pred_adv_input_pgd"], pv_ip, mv_ip),
            "input_cw":   (inp["X_adv_input_cw"],  inp["y_pred_adv_input_cw"],  pv_ic, mv_ic),
        }
        for atk, (Xa, ya, pv, mv) in runs.items():
            m = metrics_for(X_orig=X_orig, X_adv=Xa, y_true=y_true,
                            y_pred_adv=ya, protocol_valid=pv, mask_valid=mv)
            multimetric_rows.append({"model": MODEL_LABELS[mk], "attack_type": atk, **m})
            ev = (ya != y_true).astype(bool)
            valid = pv.astype(bool) & mv.astype(bool)
            for cname in np.unique(y_true_names):
                m_ = (y_true_names == cname)
                if m_.sum() == 0:
                    continue
                asr_valid = float((ev[m_] & valid[m_]).mean()) * 100.0
                category_rows.append({
                    "model": MODEL_LABELS[mk], "attack_type": atk,
                    "category": cname, "ASR_Valid": asr_valid,
                })

    print(f"[5/6] Assembling bundle (primary = {args.primary})")
    primary = per_model_per_sample[args.primary]
    mm_df = pd.DataFrame(multimetric_rows)
    ct_df = pd.DataFrame(category_rows)

    bundle: dict[str, Any] = {
        "X_original":     X_orig.astype(np.float32),
        "y_true":         y_true_names,
        "feature_names":  np.array(FEATURE_NAMES),
        "mask_type":      np.array(mask_type),
        "latent_attack_update": np.array("anchored_decoder_residual"),
        "latent_space_scope": np.array("per_class_vae"),
        **primary,
        # Store tables as structured arrays for round-trip-safe npz storage.
        # The viz script's loader detects these names and converts via pandas.
        "multimetric_table":  mm_df.to_records(index=False),
        "category_asr_table": ct_df.to_records(index=False),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **bundle)
    mm_df.to_csv(args.out.with_suffix(".multimetric.csv"), index=False)
    ct_df.to_csv(args.out.with_suffix(".category_asr.csv"), index=False)
    print(f"[6/6] Wrote {args.out}")
    print(f"      + {args.out.with_suffix('.multimetric.csv').name}")
    print(f"      + {args.out.with_suffix('.category_asr.csv').name}")
    print("Done.")


if __name__ == "__main__":
    main()
