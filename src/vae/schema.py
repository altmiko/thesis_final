"""
Path C feature-partitioning schema for the per-class β-VAE.

Centralizes all scaler-aware raw↔scaled conversions and protocol-to-binary mappings
so that model code stays clean and all scaler operations remain exact.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from sklearn.preprocessing import RobustScaler

# Canonical allowlist for Protocol Type column (raw integer values)
PROTOCOL_ALLOWLIST: list[int] = [0, 1, 2, 6, 17, 47]

# Maps raw protocol integer → name of the derived binary column
PROTOCOL_TO_BINARY: dict[int, str] = {6: "TCP", 17: "UDP", 1: "ICMP", 2: "IGMP"}

# Order of derived binary columns in the 39-dim output (TCP, UDP, ICMP, IGMP)
DERIVED_BINARY_ORDER: list[str] = ["TCP", "UDP", "ICMP", "IGMP"]

# Pre-built lookup: raw protocol value → index in DERIVED_BINARY_ORDER (-1 = no column)
_PROTOCOL_TO_DERIVED_IDX: dict[int, int] = {
    6: 0,   # TCP
    17: 1,  # UDP
    1: 2,   # ICMP
    2: 3,   # IGMP
}

# Pre-built lookup: raw protocol value → index in PROTOCOL_ALLOWLIST
_PROTOCOL_TO_ALLOWLIST_IDX: dict[int, int] = {
    v: i for i, v in enumerate(PROTOCOL_ALLOWLIST)
}


def get_partition(scaler: "RobustScaler | None" = None) -> dict[str, list[int]]:
    """Return the hard-coded feature partition for the 39-dim CICIoT2023 schema.

    The scaler parameter is accepted for API symmetry but is not used — the
    partition is determined by domain knowledge, not data statistics.
    """
    return {
        "protocol_idx": [1],
        "continuous_idx": [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 30, 31, 32, 33, 34, 35, 36, 37, 38],
        "independent_binary_idx": [15, 16, 17, 18, 19, 20, 21, 24, 25, 28, 29],
        "derived_binary_idx": [22, 23, 26, 27],
    }


def raw_to_protocol_index(raw_value: int) -> int:
    """Map a raw protocol integer to its 0-based index in PROTOCOL_ALLOWLIST.

    Raises ValueError for values outside the allowlist — callers should validate
    before calling if the source is untrusted.
    """
    try:
        return _PROTOCOL_TO_ALLOWLIST_IDX[raw_value]
    except KeyError:
        raise ValueError(
            f"Protocol value {raw_value!r} is not in PROTOCOL_ALLOWLIST {PROTOCOL_ALLOWLIST}"
        )


def protocol_index_to_raw(idx: int) -> int:
    """Map a 0-based PROTOCOL_ALLOWLIST index back to the raw integer protocol value."""
    try:
        return PROTOCOL_ALLOWLIST[idx]
    except IndexError:
        raise ValueError(
            f"Protocol index {idx} is out of range for PROTOCOL_ALLOWLIST (len={len(PROTOCOL_ALLOWLIST)})"
        )


def scaled_to_raw_protocol(
    x_scaled: np.ndarray,
    scaler: "RobustScaler",
    protocol_idx: int,
) -> np.ndarray:
    """Inverse-transform the protocol column of a scaled feature matrix to raw integers.

    x_scaled: (N, 39) scaled feature matrix.
    Returns: (N,) array of raw integer protocol values.

    The buffer trick isolates the single column so that all other columns are
    zeroed — inverse_transform operates on the full 39-dim space but only the
    protocol column carries meaningful values.
    """
    n = x_scaled.shape[0]
    buf = np.zeros((n, x_scaled.shape[1]), dtype=np.float64)
    buf[:, protocol_idx] = x_scaled[:, protocol_idx]
    raw = scaler.inverse_transform(buf)[:, protocol_idx]
    return np.round(raw).astype(np.int64)


def raw_protocol_to_scaled(
    raw_protocol_values: np.ndarray,
    scaler: "RobustScaler",
    protocol_idx: int,
    n_features: int = 39,
) -> np.ndarray:
    """Transform raw integer protocol values to the scaler's scaled space.

    raw_protocol_values: (N,) array of raw integers.
    Returns: (N,) array of scaled float values for the protocol column.

    Same buffer trick as scaled_to_raw_protocol — zeros in all other columns
    prevent cross-feature contamination during transform.
    """
    n = raw_protocol_values.shape[0]
    buf = np.zeros((n, n_features), dtype=np.float64)
    buf[:, protocol_idx] = raw_protocol_values.astype(np.float64)
    scaled = scaler.transform(buf)[:, protocol_idx]
    return scaled


def derive_binaries_from_protocol_index(
    protocol_idx_batch: torch.Tensor,
) -> torch.Tensor:
    """Derive the four protocol binary columns (TCP, UDP, ICMP, IGMP) from protocol indices.

    Input: (N,) long tensor of indices into PROTOCOL_ALLOWLIST (0..5).
    Output: (N, 4) float32 tensor with exactly one 1.0 per row for TCP/UDP/ICMP/IGMP
            protocols, and all zeros for protocols that have no derived binary column
            (HOPOPT=0, GRE=47).

    Implemented with broadcasting rather than loops — each of the four columns
    is independently derived by comparing the raw protocol value (recovered via
    PROTOCOL_ALLOWLIST indexing) to the target constant.
    """
    # Recover raw values via a vectorised lookup tensor.
    # allowlist_tensor[i] == PROTOCOL_ALLOWLIST[i] for all i.
    allowlist_tensor = torch.tensor(PROTOCOL_ALLOWLIST, dtype=torch.long, device=protocol_idx_batch.device)
    raw_values = allowlist_tensor[protocol_idx_batch]  # (N,)

    # Each column is a boolean comparison broadcast over the batch.
    tcp  = (raw_values == 6).float()   # index 0
    udp  = (raw_values == 17).float()  # index 1
    icmp = (raw_values == 1).float()   # index 2
    igmp = (raw_values == 2).float()   # index 3

    return torch.stack([tcp, udp, icmp, igmp], dim=1)  # (N, 4)


def apply_scaler_to_columns(
    values: np.ndarray,
    scaler: "RobustScaler",
    col_indices: list[int],
    n_features: int = 39,
) -> np.ndarray:
    """Scale raw values for a subset of columns using the fitted scaler.

    values: (N, len(col_indices)) raw values for the specified columns.
    Returns: (N, len(col_indices)) scaled values.

    The buffer trick ensures that the scaler's per-column statistics are applied
    correctly — each column uses its own center_ and scale_ from the original fit.
    """
    n = values.shape[0]
    buf = np.zeros((n, n_features), dtype=np.float64)
    buf[:, col_indices] = values.astype(np.float64)
    scaled = scaler.transform(buf)
    return scaled[:, col_indices]


def inverse_transform_columns(
    x_scaled: np.ndarray,
    scaler: "RobustScaler",
    col_indices: list[int],
    n_features: int = 39,
) -> np.ndarray:
    """Inverse-transform scaled values back to raw space for a subset of columns.

    x_scaled: either (N, len(col_indices)) — subset only — or (N, n_features) — full
              matrix. When the full matrix is passed, the selected columns are
              extracted before placing them into the buffer; when a subset is passed,
              they are placed directly.
    Returns: (N, len(col_indices)) raw values.

    Mirror of apply_scaler_to_columns — same buffer trick in the inverse direction.
    """
    n = x_scaled.shape[0]
    buf = np.zeros((n, n_features), dtype=np.float64)
    if x_scaled.shape[1] == n_features:
        # Full matrix supplied — extract the relevant columns into the buffer.
        buf[:, col_indices] = x_scaled[:, col_indices].astype(np.float64)
    else:
        # Subset supplied — must match len(col_indices).
        buf[:, col_indices] = x_scaled.astype(np.float64)
    raw = scaler.inverse_transform(buf)
    return raw[:, col_indices]


def raw_postprocess(x_raw: np.ndarray) -> np.ndarray:
    """Enforce domain constraints on inverse-transformed raw-space output.

    Called after scaler.inverse_transform, not during training. All rules mirror
    the validator's G1-G8 checks so that model output satisfies them by construction.

    Enforces (in order):
      G1  - All continuous features clipped to ≥ 0 (packet statistics are non-negative)
      G7  - TTL clamped to [0, 255]
      G5  - Min ≤ Max (swap if inverted)
      G5  - AVG clamped to [Min, Max]
      G6  - Variance = Std²
      G8  - Number rounded to nearest positive integer (≥ 1)
      G3  - All binary columns rounded and clipped to {0, 1}
    """
    from preprocessing.feature_groups import FEATURE_NAMES

    x = x_raw.copy()

    # Index lookups
    cont_idx = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                30, 31, 32, 33, 34, 35, 36, 37, 38]
    ttl_idx    = FEATURE_NAMES.index("Time_To_Live")
    min_idx    = FEATURE_NAMES.index("Min")
    max_idx    = FEATURE_NAMES.index("Max")
    avg_idx    = FEATURE_NAMES.index("AVG")
    std_idx    = FEATURE_NAMES.index("Std")
    var_idx    = FEATURE_NAMES.index("Variance")
    number_idx = FEATURE_NAMES.index("Number")
    all_binary_idx = [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]

    # G1: all continuous features ≥ 0
    x[:, cont_idx] = np.maximum(x[:, cont_idx], 0.0)
    # G7: TTL ≤ 255
    x[:, ttl_idx] = np.minimum(x[:, ttl_idx], 255.0)

    # G5: Min ≤ Max (swap if inverted)
    swap_mask = x[:, min_idx] > x[:, max_idx]
    x[swap_mask, min_idx], x[swap_mask, max_idx] = (
        x[swap_mask, max_idx].copy(), x[swap_mask, min_idx].copy()
    )
    # G5: Min ≤ AVG ≤ Max
    x[:, avg_idx] = np.clip(x[:, avg_idx], x[:, min_idx], x[:, max_idx])

    # G6: Variance = Std²
    x[:, var_idx] = x[:, std_idx] ** 2

    # G8: Number is a positive integer packet count (≥ 1)
    x[:, number_idx] = np.maximum(np.round(x[:, number_idx]), 1.0)

    # G3: binary columns ∈ {0, 1}
    x[:, all_binary_idx] = np.clip(np.round(x[:, all_binary_idx]), 0.0, 1.0)

    return x
