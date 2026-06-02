from __future__ import annotations

import inspect
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = str(_REPO_ROOT / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from attack.latent_cw import latent_cw_attack  # noqa: E402
from attack.latent_infra import PerturbationMask  # noqa: E402
from attack.latent_pgd import _attack_success_mask, _normalise_z_initializers, classifier_logits  # noqa: E402
from vae.model import MixedInputBetaVAE  # noqa: E402
from vae.schema import get_partition  # noqa: E402


def _load_scaler():
    with open(_REPO_ROOT / "data" / "processed" / "scaler.pkl", "rb") as f:
        return pickle.load(f)


def _real_scaled_samples(n: int = 8) -> np.ndarray:
    x_test = np.load(_REPO_ROOT / "data" / "processed" / "X_test.npy", mmap_mode="r")
    return np.array(x_test[:n], dtype=np.float32)


def _build_vae(scaler) -> MixedInputBetaVAE:
    torch.manual_seed(0)
    vae = MixedInputBetaVAE(partition=get_partition(), latent_dim=16)
    vae.register_protocol_references(scaler)
    vae.eval()
    return vae


def _tiny_classifier() -> torch.nn.Module:
    torch.manual_seed(0)
    model = torch.nn.Linear(39, 8)
    model.eval()
    return model


# --- #14: internal multi-restart must actually diversify ---------------------

def test_latent_cw_default_restart_jitter_is_nonzero():
    # With epsilon=0 + z_initializers=None every restart started at exactly
    # z_orig and Adam is deterministic, so num_restarts>1 was a no-op (fix.md
    # #14). A non-zero default jitter makes restarts actually differ.
    default = inspect.signature(latent_cw_attack).parameters["restart_jitter"].default
    assert float(default) > 0.0


def test_normalise_initializers_diversifies_restarts_with_jitter():
    z_orig = torch.zeros(4, 16)
    starts = _normalise_z_initializers(
        z_orig=z_orig,
        epsilon=0.05,
        random_start=False,
        num_restarts=3,
        z_initializers=None,
        project=False,
    )
    assert len(starts) == 3
    # Restart 0 is unjittered; later restarts must differ from it.
    assert torch.allclose(starts[0], z_orig)
    assert not torch.allclose(starts[1], starts[0])
    assert not torch.allclose(starts[2], starts[0])


# --- #13 / #16: success accounting is argmax-based and self-consistent --------

def _run_cw(scaler, *, targeted: bool, target_class: int = 3):
    vae = _build_vae(scaler)
    classifier = _tiny_classifier()
    mask = PerturbationMask.from_preprocessing_artifacts(_REPO_ROOT)
    x0 = torch.from_numpy(_real_scaled_samples(8))
    y = torch.zeros(8, dtype=torch.long)
    x_adv, _z, meta = latent_cw_attack(
        vae=vae,
        classifier=classifier,
        mask=mask,
        x_original=x0,
        y_true=y,
        scaler=scaler,
        lambda_conf=1.0,
        kappa=0.0,
        num_iterations=5,
        learning_rate=0.05,
        convergence_threshold=1e-5,
        device="cpu",
        targeted=targeted,
        target_class=target_class,
    )
    return classifier, x_adv, y, meta


def test_untargeted_latent_cw_success_mask_is_argmax_consistent():
    scaler = _load_scaler()
    classifier, x_adv, y, meta = _run_cw(scaler, targeted=False)
    logits = classifier_logits(classifier, x_adv, device="cpu")
    recomputed = _attack_success_mask(logits, y, targeted=False, target_class=0)
    assert torch.equal(meta["best_success_mask"].cpu(), recomputed.cpu())


def test_targeted_latent_cw_success_mask_is_argmax_consistent():
    scaler = _load_scaler()
    classifier, x_adv, y, meta = _run_cw(scaler, targeted=True, target_class=3)
    logits = classifier_logits(classifier, x_adv, device="cpu")
    recomputed = _attack_success_mask(logits, y, targeted=True, target_class=3)
    assert torch.equal(meta["best_success_mask"].cpu(), recomputed.cpu())


if __name__ == "__main__":
    test_latent_cw_default_restart_jitter_is_nonzero()
    test_normalise_initializers_diversifies_restarts_with_jitter()
    test_untargeted_latent_cw_success_mask_is_argmax_consistent()
    test_targeted_latent_cw_success_mask_is_argmax_consistent()
    print("latent_cw fix tests passed.")
