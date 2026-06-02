from __future__ import annotations

import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = str(_REPO_ROOT / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from vae.model import MixedInputBetaVAE  # noqa: E402
from vae.schema import get_partition  # noqa: E402


def _assert_raises(exc_type, fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except exc_type:
        return
    except Exception as exc:  # noqa: BLE001
        raise AssertionError(
            f"expected {exc_type.__name__}, got {type(exc).__name__}: {exc}"
        )
    raise AssertionError(f"expected {exc_type.__name__}, but no exception was raised")


def _build_vae() -> MixedInputBetaVAE:
    partition = get_partition()
    vae = MixedInputBetaVAE(partition=partition, latent_dim=16)
    vae.eval()
    return vae


def test_decode_to_39_rejects_invalid_mode():
    # A typo in ``mode`` silently flipped binary outputs to a hard threshold,
    # changing attack semantics inside every PGD/CW inner step (fix.md #7).
    vae = _build_vae()
    z = torch.zeros(2, vae.latent_dim)
    _assert_raises(ValueError, vae.decode_to_39, z, None, "harD")


def test_decode_to_39_accepts_valid_modes():
    vae = _build_vae()
    z = torch.zeros(2, vae.latent_dim)
    for mode in ("soft", "hard"):
        x_out, _ = vae.decode_to_39(z, None, mode)
        assert x_out.shape == (2, 39)


def test_encode_requires_registered_references():
    # Without register_protocol_references the placeholder ref buffer is all
    # zeros, so every sample would silently embed as protocol index 0 (fix.md
    # #15). encode() must refuse rather than produce wrong embeddings.
    vae = _build_vae()
    x = torch.zeros(2, 39)
    _assert_raises(RuntimeError, vae.encode, x)


if __name__ == "__main__":
    test_decode_to_39_rejects_invalid_mode()
    test_decode_to_39_accepts_valid_modes()
    test_encode_requires_registered_references()
    print("VAE guard tests passed.")
