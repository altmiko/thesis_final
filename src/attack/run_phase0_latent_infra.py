from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = str(_REPO_ROOT / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from attack.latent_infra import (  # noqa: E402
    AttackRouter,
    AttackRunLogger,
    MahalanobisOutlierDetector,
    PerturbationMask,
    build_per_class_dataset,
    encode_dataset_mu,
    load_collapsed_dims,
    load_validation_split,
    phase0_config_snapshot,
    set_global_seed,
)
from vae.config import CLASSES  # noqa: E402


def _format_group(indices: list[int]) -> str:
    parts = [f"{idx}:{name}" for idx, name in zip(indices, [CLASSES[0]] * 0)]
    _ = parts
    return ", ".join(f"{idx}:{name}" for idx, name in zip(indices, []))


def _feature_listing(indices: list[int]) -> list[str]:
    from preprocessing.feature_groups import FEATURE_NAMES

    return [f"{idx}:{FEATURE_NAMES[idx]}" for idx in indices]


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 0 checkpoint for latent-attack infrastructure.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    set_global_seed(args.seed)

    run_logger = AttackRunLogger.create(
        phase_name="phase0",
        seed=args.seed,
        config_snapshot=phase0_config_snapshot(args.seed, args.device),
    )

    mask = PerturbationMask.from_preprocessing_artifacts()
    router = AttackRouter(device=args.device)
    split = load_validation_split()
    collapsed_by_class = load_collapsed_dims()
    detector = MahalanobisOutlierDetector(latent_dim=16)

    print("=== Phase 0 Checkpoint ===")
    print(f"Output directory: {run_logger.run_dir}")
    print(f"Seed: {args.seed}")
    print()

    print("Perturbation mask groups")
    print(f"  Full ({len(mask.full_indices)}): {', '.join(_feature_listing(mask.full_indices))}")
    print(f"  Partial ({len(mask.partial_indices)}): {', '.join(_feature_listing(mask.partial_indices))}")
    print(f"  Frozen ({len(mask.frozen_indices)}): {', '.join(_feature_listing(mask.frozen_indices))}")
    print("  Partial delta bounds (scaled-space deltas):")
    for idx in mask.partial_indices:
        lower, upper = mask.partial_bounds_by_index()[idx]
        print(f"    {idx}:{_feature_listing([idx])[0].split(':', 1)[1]} -> [{lower:.3f}, {upper:.3f}]")
    print()

    run_logger.log("Perturbation mask")
    run_logger.log(f"Full: {mask.full_indices}")
    run_logger.log(f"Partial: {mask.partial_indices}")
    run_logger.log(f"Frozen: {mask.frozen_indices}")

    print("Active latent dimensions per class")
    for class_id, class_name in enumerate(CLASSES):
        collapsed = collapsed_by_class[class_id]
        active = [dim for dim in range(16) if dim not in collapsed]
        print(
            f"  {class_name:10s} active={active} "
            f"(k={len(active)}), collapsed={collapsed}"
        )
        run_logger.log(
            f"{class_name}: active={active} collapsed={collapsed}"
        )
    print()

    print("Mahalanobis clean self-outlier rates")
    for class_id, class_name in enumerate(CLASSES):
        dataset = build_per_class_dataset(
            class_id,
            X_val=split["X_val"],
            y_val_8=split["y_val_8"],
            scaler=split["scaler"],
            partition=split["partition"],
        )
        vae = router.get_vae(class_id)
        z_mu = encode_dataset_mu(vae, dataset, device=args.device)
        stats = detector.fit(class_id, z_mu, collapsed_dims=collapsed_by_class[class_id])
        outlier_rate = detector.outlier_rate(class_id, z_mu)
        print(
            f"  {class_name:10s} outlier_rate={outlier_rate * 100.0:6.2f}% "
            f"threshold95={stats['threshold_95']:.3f} "
            f"active_k={stats['effective_dimensionality']:2d} "
            f"pinv={'yes' if stats['used_pinv'] else 'no'}"
        )
        run_logger.log(
            f"{class_name}: outlier_rate={outlier_rate:.6f} "
            f"threshold95={stats['threshold_95']:.6f} "
            f"active_k={stats['effective_dimensionality']} "
            f"pinv={stats['used_pinv']}"
        )


if __name__ == "__main__":
    main()
