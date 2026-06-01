from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = str(_REPO_ROOT / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from attack.constrained_input_baselines import (  # noqa: E402
    VAEConstraintProjection,
    constrained_input_cw_attack,
    constrained_input_pgd_attack,
)
from attack.input_baselines import input_pgd_attack  # noqa: E402
from attack.latent_infra import PerturbationMask  # noqa: E402
from attack.validator import validate_batch  # noqa: E402
from preprocessing.feature_groups import FEATURE_NAMES  # noqa: E402


def _load_scaler():
    with open(_REPO_ROOT / "data" / "processed" / "scaler.pkl", "rb") as f:
        return pickle.load(f)


def _real_scaled_samples(n: int = 8) -> np.ndarray:
    x_test = np.load(_REPO_ROOT / "data" / "processed" / "X_test.npy")
    return x_test[:n].astype(np.float32)


def _tiny_classifier() -> torch.nn.Module:
    torch.manual_seed(0)
    model = torch.nn.Linear(39, 8)
    model.eval()
    return model


def test_hard_projection_satisfies_structured_rules():
    scaler = _load_scaler()
    mask = PerturbationMask.from_preprocessing_artifacts(_REPO_ROOT)
    projection = VAEConstraintProjection(scaler, mask, enable_physics=True, device="cpu")

    x0 = torch.from_numpy(_real_scaled_samples(16))
    x_pert = x0 + torch.randn_like(x0) * 2.0
    x_proj = projection.project(x_pert, x0, mode="hard")

    x_raw = scaler.inverse_transform(x_proj.detach().numpy().astype(np.float64))
    rates = validate_batch(x_raw, FEATURE_NAMES).per_rule_violation_rate()
    for rule in [
        "R_min_leq_max",
        "R_avg_in_range",
        "R_var_eq_std_sq",
        "R_pkts_positive",
        "R_pkts_integer",
    ]:
        assert rates.get(rule, 0.0) == 0.0, (rule, rates.get(rule))


def test_parity_with_vae_structured_decoder():
    from vae.model import MixedInputBetaVAE
    from vae.schema import get_partition

    partition = get_partition()
    scaler = _load_scaler()
    vae = MixedInputBetaVAE(
        partition=partition,
        latent_dim=16,
        use_structured_continuous_decoder=True,
        use_structured_physics_decoder=True,
        structured_continuous_mode="full",
        structured_std_floor=0.0,
    )
    vae.register_protocol_references(scaler)
    vae.eval()

    mask = PerturbationMask.from_preprocessing_artifacts(_REPO_ROOT)
    projection = VAEConstraintProjection(
        scaler,
        mask,
        enable_physics=True,
        structured_std_floor=0.0,
        device="cpu",
    )

    torch.manual_seed(0)
    cont_idx = partition["continuous_idx"]
    continuous_raw = torch.randn(32, len(cont_idx)) * 5.0

    vae_structured = vae._structure_continuous_raw(continuous_raw)
    full_raw = torch.zeros(32, len(FEATURE_NAMES))
    full_raw[:, cont_idx] = continuous_raw
    projected_structured = projection._structure_raw(full_raw, mode="soft")[:, cont_idx]

    assert torch.allclose(vae_structured, projected_structured, atol=1e-4), (
        vae_structured - projected_structured
    ).abs().max().item()


def test_soft_projection_is_differentiable():
    scaler = _load_scaler()
    mask = PerturbationMask.from_preprocessing_artifacts(_REPO_ROOT)
    projection = VAEConstraintProjection(scaler, mask, enable_physics=True, device="cpu")

    x0 = torch.from_numpy(_real_scaled_samples(4))
    x = (x0 + 0.1).clone().requires_grad_(True)
    out = projection.project(x, x0, mode="soft")
    out.sum().backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_constrained_pgd_smoke_and_validity_gain():
    scaler = _load_scaler()
    mask = PerturbationMask.from_preprocessing_artifacts(_REPO_ROOT)
    projection = VAEConstraintProjection(scaler, mask, enable_physics=True, device="cpu")
    classifier = _tiny_classifier()

    x0 = torch.from_numpy(_real_scaled_samples(16))
    y = torch.zeros(16, dtype=torch.long)

    x_constrained, meta = constrained_input_pgd_attack(
        classifier=classifier,
        projection=projection,
        x_original=x0,
        y_true=y,
        epsilon=0.5,
        alpha=0.05,
        num_steps=5,
        random_start=True,
        device="cpu",
    )
    x_plain, _ = input_pgd_attack(
        classifier=classifier,
        x_original=x0,
        y_true=y,
        epsilon=0.5,
        alpha=0.05,
        num_steps=5,
        random_start=True,
        device="cpu",
    )

    assert x_constrained.shape == x0.shape
    assert meta["constraint_projection"] is True

    raw_constrained = scaler.inverse_transform(x_constrained.numpy().astype(np.float64))
    raw_plain = scaler.inverse_transform(x_plain.numpy().astype(np.float64))
    assert validate_batch(raw_constrained, FEATURE_NAMES).validity_rate >= validate_batch(
        raw_plain,
        FEATURE_NAMES,
    ).validity_rate


def test_constrained_cw_smoke():
    scaler = _load_scaler()
    mask = PerturbationMask.from_preprocessing_artifacts(_REPO_ROOT)
    projection = VAEConstraintProjection(scaler, mask, enable_physics=True, device="cpu")
    classifier = _tiny_classifier()

    x0 = torch.from_numpy(_real_scaled_samples(16))
    y = torch.zeros(16, dtype=torch.long)

    x_adv, meta = constrained_input_cw_attack(
        classifier=classifier,
        projection=projection,
        x_original=x0,
        y_true=y,
        lambda_conf=1.0,
        kappa=0.0,
        num_iterations=5,
        learning_rate=0.05,
        convergence_threshold=1e-5,
        device="cpu",
    )

    assert x_adv.shape == x0.shape
    assert meta["constraint_projection"] is True
    assert "best_success_mask" in meta


if __name__ == "__main__":
    test_hard_projection_satisfies_structured_rules()
    test_parity_with_vae_structured_decoder()
    test_soft_projection_is_differentiable()
    test_constrained_pgd_smoke_and_validity_gain()
    test_constrained_cw_smoke()
    print("Constrained input baseline tests passed.")
