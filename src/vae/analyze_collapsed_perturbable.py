"""
Cross-reference VAE reconstruction quality against the latent-attack
perturbation mask.

For the Benign class (the one whose VAE collapses the volumetric features),
this prints, per continuous feature:
  - reconstruction R^2 (from results/vae/benign_test_reconstruction.json)
  - the latent-attack perturbation class: FULL / PARTIAL / FROZEN
    (from attack.latent_infra.PerturbationMask.from_preprocessing_artifacts)

The point: a feature that the VAE decoder collapses (R^2 <= 0) AND that the
attack is allowed to perturb is a feature the latent attack cannot actually
*steer* through z (the decoder output is ~constant there, so the decoder
residual x_decoded - x_anchor_decoded ~ 0); a collapsed feature that is FROZEN
is doubly irrelevant.

Usage:
  python src/vae/analyze_collapsed_perturbable.py --class-name Benign
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = str(_REPO_ROOT / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from preprocessing.feature_groups import FEATURE_NAMES  # noqa: E402
from attack.latent_infra import PerturbationMask  # noqa: E402
from vae.schema import get_partition  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--class-name", default="Benign")
    parser.add_argument(
        "--recon-json",
        default=None,
        help="Path to <class>_test_reconstruction.json (default: derived from class name).",
    )
    args = parser.parse_args()

    recon_path = (
        Path(args.recon_json)
        if args.recon_json
        else _REPO_ROOT / "results" / "vae" / f"{args.class_name.lower()}_test_reconstruction.json"
    )
    recon = json.load(open(recon_path, encoding="utf-8"))
    cont_feats = recon["aggregate"]["continuous_features"]  # name -> {r2, ...}

    partition = get_partition()
    mask = PerturbationMask.from_preprocessing_artifacts()
    full = set(mask.full_feature_names)
    partial = set(mask.partial_feature_names)
    frozen = set(mask.frozen_feature_names)

    def mask_class(feat: str) -> str:
        if feat in full:
            return "FULL"
        if feat in partial:
            return "PARTIAL"
        if feat in frozen:
            return "FROZEN"
        return "?"

    # --- Per continuous feature: R^2 vs perturbation class ---
    rows = []
    for col in partition["continuous_idx"]:
        fname = FEATURE_NAMES[col]
        r2 = cont_feats.get(fname, {}).get("r2")
        rows.append((fname, r2, mask_class(fname)))
    # Sort by R^2 ascending (worst reconstructed first); None (degenerate) last.
    rows.sort(key=lambda r: (r[1] is None, r[1] if r[1] is not None else 1e9))

    print("=" * 74)
    print(f"{args.class_name}: continuous-feature reconstruction R^2  vs  latent-attack mask")
    print("=" * 74)
    print(f"{'feature':>16s} {'R^2':>14s} {'recon':>10s}   {'attack_mask':>11s}")
    for fname, r2, mc in rows:
        r2s = "degenerate" if r2 is None else f"{r2:14.4f}"
        quality = "GOOD" if (r2 is not None and r2 >= 0.5) else ("ok" if (r2 is not None and r2 >= 0) else "COLLAPSED")
        if r2 is None:
            quality = "const"
        print(f"{fname:>16s} {r2s:>14s} {quality:>10s}   {mc:>11s}")

    # --- Focus: collapsed (R^2 <= 0) features and their perturbability ---
    collapsed = [(f, r2, mc) for (f, r2, mc) in rows if (r2 is not None and r2 <= 0.0)]
    print("\n" + "-" * 74)
    print(f"COLLAPSED continuous features (R^2 <= 0): {len(collapsed)}")
    print("-" * 74)
    n_full = sum(1 for _, _, mc in collapsed if mc == "FULL")
    n_partial = sum(1 for _, _, mc in collapsed if mc == "PARTIAL")
    n_frozen = sum(1 for _, _, mc in collapsed if mc == "FROZEN")
    for f, r2, mc in collapsed:
        print(f"  {f:>16s}  R^2={r2:9.3f}  -> {mc}")
    print(f"\n  collapsed & FULL (attack-perturbable, unbounded): {n_full}")
    print(f"  collapsed & PARTIAL (attack-perturbable, ±0.3 scaled): {n_partial}")
    print(f"  collapsed & FROZEN (attack cannot touch):              {n_frozen}")

    # --- Overall mask composition ---
    print("\n" + "-" * 74)
    print("Latent-attack perturbation mask composition (all 39 features)")
    print("-" * 74)
    print(f"  FULL    ({len(mask.full_indices):2d}): {sorted(full)}")
    print(f"  PARTIAL ({len(mask.partial_indices):2d}): {sorted(partial)}")
    print(f"  FROZEN  ({len(mask.frozen_indices):2d}): {sorted(frozen)}")


if __name__ == "__main__":
    main()
