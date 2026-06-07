"""
Inspect a per-class VAE's reconstruction on random TEST-split samples.

Takes a few random samples of the chosen class from the held-out test split,
runs them through that class's MixedInputBetaVAE (deterministic posterior-mean
reconstruction: z = mu, decode_to_39 hard), and prints a per-feature
input-vs-reconstruction comparison in raw space. Also reports aggregate
reconstruction accuracy over the full test set of that class using the same
methodology as reconstruction_accuracy.py.

Usage:
  python src/vae/inspect_benign_reconstruction.py --device cuda --n-display 5
  python src/vae/inspect_benign_reconstruction.py --class-name DDoS --n-display 5
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = str(_REPO_ROOT / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from preprocessing.feature_groups import FEATURE_NAMES  # noqa: E402
from vae.config import CLASS_TO_ID  # noqa: E402
from vae.dataset import PerClassDataset  # noqa: E402
from vae.schema import get_partition  # noqa: E402
from vae.train import _load_8class_labels  # noqa: E402
from vae.reconstruction_accuracy import (  # noqa: E402
    _load_vae,
    _reconstruction_accuracy_for_class,
)

logger = logging.getLogger(__name__)


def _feature_type(col: int, partition: dict) -> str:
    if col in partition["protocol_idx"]:
        return "protocol"
    if col in partition["continuous_idx"]:
        return "cont"
    if col in partition["independent_binary_idx"]:
        return "bin"
    if col in partition["derived_binary_idx"]:
        return "derived"
    return "?"


def _fmt_num(v: float) -> str:
    if abs(v) >= 1e4 or (v != 0 and abs(v) < 1e-3):
        return f"{v:.4g}"
    return f"{v:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--class-name", default="Benign", help="Class to inspect (e.g. Benign, DDoS).")
    parser.add_argument("--n-display", type=int, default=5, help="Samples shown in full per-feature detail.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--manifest", default=str(_REPO_ROOT / "vae_run_manifest.json"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    device = args.device
    root = _REPO_ROOT

    class_name = args.class_name
    if class_name not in CLASS_TO_ID:
        raise SystemExit(f"Unknown class {class_name!r}; valid: {list(CLASS_TO_ID)}")
    class_id = CLASS_TO_ID[class_name]

    with open(args.manifest, encoding="utf-8") as f:
        manifest = json.load(f)

    X_test = np.load(str(root / "data" / "processed" / "X_test.npy"))
    y_test_34 = np.load(str(root / "data" / "processed" / "y_test.npy"))
    with open(str(root / "data" / "processed" / "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    y_test_8 = _load_8class_labels(root, y_test_34, "test")
    partition = get_partition()

    class_test_idx = np.where(y_test_8 == class_id)[0]
    logger.info("%s test samples available: %d", class_name, len(class_test_idx))

    rng = np.random.default_rng(args.seed)
    disp_pos = rng.choice(len(class_test_idx), size=min(args.n_display, len(class_test_idx)), replace=False)
    disp_global_idx = np.sort(class_test_idx[disp_pos])
    X_disp = X_test[disp_global_idx]  # (n_display, 39) scaled

    model = _load_vae(class_name, manifest, partition, scaler, device)

    # --- Deterministic reconstruction of display samples ---
    x = torch.from_numpy(X_disp.astype(np.float32)).to(device)
    with torch.no_grad():
        mu, _ = model.encode(x)
        x_recon_scaled, meta = model.decode_to_39(mu, mode="hard")

    true_raw = scaler.inverse_transform(X_disp.astype(np.float64))                       # (n,39)
    recon_raw = scaler.inverse_transform(x_recon_scaled.cpu().numpy().astype(np.float64))  # (n,39)

    cont_idx = partition["continuous_idx"]
    bin_cols = partition["independent_binary_idx"] + partition["derived_binary_idx"]
    proto_col = partition["protocol_idx"][0]

    # ---------------- Per-sample detailed report ----------------
    print("\n" + "=" * 100)
    print(f"{class_name.upper()} VAE — deterministic reconstruction of {len(disp_global_idx)} random TEST samples (seed={args.seed})")
    print("z = posterior mean (mu), decode_to_39(mode='hard'); values shown in RAW space")
    print("=" * 100)

    samples_payload = []
    for i, gidx in enumerate(disp_global_idx):
        print(f"\n--- Sample #{i+1}  (test row {gidx}) ---")
        print(f"{'feat':>16s} {'type':>8s} {'input':>14s} {'recon':>14s} {'abs_err':>12s}  {'rel_err':>9s}")
        cont_abs_errs = []
        for col in range(39):
            ftype = _feature_type(col, partition)
            tv = float(true_raw[i, col])
            rv = float(recon_raw[i, col])
            if ftype in ("bin", "derived", "protocol"):
                tb = round(tv)
                rb = round(rv)
                mark = "ok" if tb == rb else "MISS"
                print(f"{FEATURE_NAMES[col]:>16s} {ftype:>8s} {tb:>14d} {rb:>14d} {'':>12s}  {mark:>9s}")
            else:
                ae = abs(rv - tv)
                rel = ae / (abs(tv) + 1e-9)
                cont_abs_errs.append(ae)
                print(f"{FEATURE_NAMES[col]:>16s} {ftype:>8s} {_fmt_num(tv):>14s} {_fmt_num(rv):>14s} {_fmt_num(ae):>12s}  {rel:>8.1%}")

        # Per-sample categorical accuracy
        true_bits = np.round(true_raw[i, bin_cols]).astype(int)
        recon_bits = np.round(recon_raw[i, bin_cols]).astype(int)
        bin_correct = int((true_bits == recon_bits).sum())
        proto_ok = round(true_raw[i, proto_col]) == round(recon_raw[i, proto_col])
        print(
            f"  -> continuous MAE={np.mean(cont_abs_errs):.4f} | "
            f"binary {bin_correct}/{len(bin_cols)} bits correct | "
            f"protocol {'match' if proto_ok else 'MISS'}"
        )
        samples_payload.append({
            "test_row": int(gidx),
            "continuous_mae": float(np.mean(cont_abs_errs)),
            "binary_bits_correct": bin_correct,
            "binary_bits_total": len(bin_cols),
            "protocol_match": bool(proto_ok),
            "input_raw": true_raw[i].tolist(),
            "recon_raw": recon_raw[i].tolist(),
        })

    # ---------------- Aggregate accuracy over full Benign test set ----------------
    print("\n" + "=" * 100)
    print(f"AGGREGATE reconstruction accuracy over ALL {len(class_test_idx)} {class_name} TEST samples")
    print("=" * 100)
    val_ds = PerClassDataset(X_test, y_test_8, class_id, scaler, partition)
    agg = _reconstruction_accuracy_for_class(
        class_id, class_name, model, val_ds, scaler, partition, device,
    )
    h = agg["headline"]
    print(json.dumps(h, indent=2))

    # Worst & best reconstructed continuous features (by R^2)
    cont = agg["continuous_features"]
    scored = [(k, v["r2"], v["nrmse"], v["rmse"], v["true_std"]) for k, v in cont.items() if v["r2"] is not None]
    scored.sort(key=lambda r: r[1])
    print("\nWorst-5 continuous features (lowest R^2):")
    for k, r2, nrmse, rmse, std in scored[:5]:
        print(f"  {k:>16s}  R2={r2:>12.4f}  NRMSE={nrmse:>8.3f}  RMSE={rmse:>12.4g}  true_std={std:>12.4g}")
    print("Best-5 continuous features (highest R^2):")
    for k, r2, nrmse, rmse, std in scored[-5:][::-1]:
        print(f"  {k:>16s}  R2={r2:>12.4f}  NRMSE={nrmse:>8.3f}  RMSE={rmse:>12.4g}  true_std={std:>12.4g}")

    # --- Save ---
    out = {
        "class": class_name,
        "split": "test",
        "seed": args.seed,
        "n_class_test": int(len(class_test_idx)),
        "reconstruction": "z = mu (deterministic); decode_to_39(mode='hard'); raw-space comparison",
        "display_samples": samples_payload,
        "aggregate": agg,
    }
    out_path = root / "results" / "vae" / f"{class_name.lower()}_test_reconstruction.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
