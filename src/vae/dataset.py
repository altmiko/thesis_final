"""
Per-class subset dataset for the Path C β-VAE.
Precomputes all targets once to avoid inverse-transforming in the training loop.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch.utils.data import Dataset

if TYPE_CHECKING:
    from sklearn.preprocessing import RobustScaler

from vae import schema

logger = logging.getLogger(__name__)


class PerClassDataset(Dataset):
    """Dataset that holds scaled features for a single 8-class label.

    All inverse-transform operations happen once at construction time so that
    the training loop receives pre-computed targets without any scaler calls
    per batch.

    Parameters
    ----------
    X_split:
        (N_total, 39) array of scaled features for a train/val split.
    y_split:
        (N_total,) integer array of 8-class labels (0..7).
    class_id:
        The 8-class integer label to subset on.
    scaler:
        Fitted RobustScaler used to inverse-transform columns into raw space.
    partition:
        Feature partition dict from ``schema.get_partition()``.
    n_pseudo_binary:
        Number of pseudo-binary columns (currently 0 for this project).
    """

    def __init__(
        self,
        X_split: np.ndarray,
        y_split: np.ndarray,
        class_id: int,
        scaler: "RobustScaler",
        partition: dict,
        n_pseudo_binary: int = 0,
    ) -> None:
        super().__init__()

        # --- 1. Subset to this class ---
        mask = y_split == class_id
        if not mask.any():
            raise ValueError(
                f"No samples found for class_id={class_id} in the provided split."
            )

        X_class = X_split[mask]  # (N, 39) scaled
        N = X_class.shape[0]

        # --- 2. Scaled input tensor (model input, kept as-is) ---
        self.x_scaled = torch.from_numpy(X_class.astype(np.float32))

        # --- 3. Independent binary targets (precomputed in raw space) ---
        # inverse_transform_columns returns (N, 11) raw values
        raw_ind_bin = schema.inverse_transform_columns(
            X_class,
            scaler,
            partition["independent_binary_idx"],
        )
        # Round and clip to valid {0, 1} range
        raw_ind_bin = np.clip(np.round(raw_ind_bin), 0.0, 1.0).astype(np.float32)
        self.target_independent_binary = torch.from_numpy(raw_ind_bin)

        # --- 4. Protocol class index targets (precomputed in raw space) ---
        raw_protocol_vals = schema.scaled_to_raw_protocol(
            X_class,
            scaler,
            partition["protocol_idx"][0],
        )  # (N,) int64

        protocol_indices = np.zeros(N, dtype=np.int64)
        for i, v in enumerate(raw_protocol_vals):
            try:
                protocol_indices[i] = schema.raw_to_protocol_index(int(v))
            except ValueError:
                logger.warning(
                    "Class %d sample %d: raw protocol %r not in allowlist; defaulting to index 0.",
                    class_id,
                    i,
                    int(v),
                )
                protocol_indices[i] = 0

        self.target_protocol_index = torch.from_numpy(protocol_indices)

        # --- 5. Metadata ---
        self.class_id = class_id
        self.n_samples = N

        logger.info("Class %d: %d samples loaded", class_id, N)

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> dict:
        return {
            "x_scaled": self.x_scaled[idx],
            "target_ind_binary": self.target_independent_binary[idx],
            "target_proto_idx": self.target_protocol_index[idx],
        }
