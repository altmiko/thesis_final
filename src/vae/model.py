"""
Path C: MixedInputBetaVAE
- Encoder input: 39-dim scaled vector with Protocol Type replaced by embedding
- Decoder: four separate heads (continuous, independent binary, protocol, pseudo-binary)
- Output adapter: decode_to_39 reconstructs full 39-dim scaled vector,
  with derived binaries (TCP/UDP/ICMP/IGMP) computed from protocol argmax
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn

if TYPE_CHECKING:
    from sklearn.preprocessing import RobustScaler

from vae.schema import (
    PROTOCOL_ALLOWLIST,
    apply_scaler_to_columns,
    derive_binaries_from_protocol_index,
    get_partition,
    protocol_index_to_raw,
    raw_protocol_to_scaled,
    raw_to_protocol_index,
    scaled_to_raw_protocol,
)

logger = logging.getLogger(__name__)

_N_PROTOCOL_CLASSES: int = len(PROTOCOL_ALLOWLIST)  # 6


def _continuous_pos(partition: dict, full_feature_idx: int) -> int:
    return partition["continuous_idx"].index(full_feature_idx)


def _build_encoder_body(input_dim: int, hidden: tuple) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev = input_dim
    for h in hidden:
        layers.append(nn.Linear(prev, h))
        layers.append(nn.ReLU())
        prev = h
    return nn.Sequential(*layers)


def _build_decoder_body(latent_dim: int, hidden: tuple) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev = latent_dim
    for h in hidden:
        layers.append(nn.Linear(prev, h))
        layers.append(nn.ReLU())
        prev = h
    return nn.Sequential(*layers)


class MixedInputBetaVAE(nn.Module):
    def __init__(
        self,
        partition: dict,
        latent_dim: int = 16,
        protocol_embed_dim: int = 4,
        encoder_hidden: tuple = (128, 64),
        decoder_hidden: tuple = (64, 128),
        n_pseudo_binary: int = 0,
        use_structured_continuous_decoder: bool = False,
        structured_continuous_mode: str = "full",
        structured_std_floor: float = 0.0,
        latent_logvar_bounds: tuple[float, float] = (-6.0, 6.0),
    ) -> None:
        super().__init__()

        self.partition = partition
        self.latent_dim = latent_dim
        self.protocol_embed_dim = protocol_embed_dim
        self.encoder_hidden = encoder_hidden
        self.decoder_hidden = decoder_hidden
        self.n_pseudo_binary = n_pseudo_binary
        self.use_structured_continuous_decoder = use_structured_continuous_decoder
        self.structured_continuous_mode = structured_continuous_mode
        self.structured_std_floor = structured_std_floor
        self.latent_logvar_bounds = latent_logvar_bounds

        self.n_continuous = len(partition["continuous_idx"])
        self.n_independent_binary = len(partition["independent_binary_idx"])

        self.protocol_embed = nn.Embedding(_N_PROTOCOL_CLASSES, protocol_embed_dim)

        encoder_input_dim = (
            self.n_continuous
            + self.n_independent_binary
            + n_pseudo_binary
            + protocol_embed_dim
        )

        self.encoder_body = _build_encoder_body(encoder_input_dim, encoder_hidden)
        self.encoder_out = nn.Linear(encoder_hidden[-1], 2 * latent_dim)

        self.decoder_body = _build_decoder_body(latent_dim, decoder_hidden)

        self.head_continuous_mu = nn.Linear(decoder_hidden[-1], self.n_continuous)
        self.head_continuous_logvar = nn.Linear(decoder_hidden[-1], self.n_continuous)
        self.head_binary = nn.Linear(decoder_hidden[-1], self.n_independent_binary)
        self.head_protocol = nn.Linear(decoder_hidden[-1], _N_PROTOCOL_CLASSES)

        if n_pseudo_binary > 0:
            self.head_pseudo: nn.Module | None = nn.Sequential(
                nn.Linear(decoder_hidden[-1], n_pseudo_binary),
                nn.Sigmoid(),
            )
        else:
            self.head_pseudo = None

        # Placeholder buffer — must call register_protocol_references(scaler) before
        # encode() produces meaningful protocol embeddings. The placeholder ensures
        # the model can be constructed and shapes verified without a scaler.
        self.register_buffer(
            "ref_proto_scaled",
            torch.zeros(_N_PROTOCOL_CLASSES, dtype=torch.float32),
        )
        self.register_buffer(
            "continuous_center",
            torch.zeros(self.n_continuous, dtype=torch.float32),
        )
        self.register_buffer(
            "continuous_scale",
            torch.ones(self.n_continuous, dtype=torch.float32),
        )

    def register_protocol_references(self, scaler: "RobustScaler") -> None:
        """Compute and register scaled reference values for each allowlist protocol.

        Must be called once after construction (before any training or inference)
        whenever the scaler is available. Stored as a non-parameter buffer so it
        survives state_dict save/load.
        """
        raw_vals = np.array(PROTOCOL_ALLOWLIST, dtype=np.float64)
        cont_idx = self.partition["continuous_idx"]
        scaled = raw_protocol_to_scaled(
            raw_vals,
            scaler,
            protocol_idx=self.partition["protocol_idx"][0],
            n_features=39,
        ).astype(np.float32)
        self.ref_proto_scaled = torch.tensor(scaled, dtype=torch.float32)
        self.continuous_center = torch.tensor(
            scaler.center_[cont_idx].astype(np.float32),
            dtype=torch.float32,
        )
        self.continuous_scale = torch.tensor(
            scaler.scale_[cont_idx].astype(np.float32),
            dtype=torch.float32,
        )

    def continuous_scaled_to_raw(self, continuous_scaled: torch.Tensor) -> torch.Tensor:
        """Map decoder continuous outputs from scaled space back to raw space."""
        center = self.continuous_center.to(continuous_scaled.device)
        scale = self.continuous_scale.to(continuous_scaled.device)
        return continuous_scaled * scale.unsqueeze(0) + center.unsqueeze(0)

    def continuous_raw_to_scaled(self, continuous_raw: torch.Tensor) -> torch.Tensor:
        """Map raw-space continuous values back to the scaler's feature space."""
        center = self.continuous_center.to(continuous_raw.device)
        scale = self.continuous_scale.to(continuous_raw.device)
        return (continuous_raw - center.unsqueeze(0)) / scale.unsqueeze(0)

    def _structure_continuous_raw(self, continuous_raw: torch.Tensor) -> torch.Tensor:
        """Apply by-construction constraints for the main raw-space consistency rules."""
        ttl_idx = _continuous_pos(self.partition, 2)
        min_idx = _continuous_pos(self.partition, 31)
        max_idx = _continuous_pos(self.partition, 32)
        avg_idx = _continuous_pos(self.partition, 33)
        std_idx = _continuous_pos(self.partition, 34)
        number_idx = _continuous_pos(self.partition, 37)
        variance_idx = _continuous_pos(self.partition, 38)

        if self.structured_continuous_mode == "full":
            # Full mode is the strongest repair path: all continuous features are
            # projected nonnegative before feature-specific constraints are applied.
            structured = continuous_raw.clamp_min(0.0)
        elif self.structured_continuous_mode == "selective":
            # Selective mode only enforces the cross-feature/domain rules that are
            # most semantically important, leaving the rest of the continuous head
            # untouched so the class can preserve a better likelihood fit.
            structured = continuous_raw.clone()
        else:
            raise ValueError(
                f"Unknown structured_continuous_mode={self.structured_continuous_mode!r}"
            )

        min_base = torch.nn.functional.softplus(continuous_raw[:, min_idx])
        avg_gap = torch.nn.functional.softplus(continuous_raw[:, avg_idx] - continuous_raw[:, min_idx])
        max_gap = torch.nn.functional.softplus(continuous_raw[:, max_idx] - continuous_raw[:, avg_idx])
        avg_val = min_base + avg_gap
        max_val = avg_val + max_gap

        std_val = self.structured_std_floor + torch.nn.functional.softplus(continuous_raw[:, std_idx])
        variance_val = std_val.square()

        number_pos = 1.0 + torch.nn.functional.softplus(continuous_raw[:, number_idx])
        number_val = number_pos + (torch.round(number_pos) - number_pos).detach()
        ttl_val = torch.sigmoid(continuous_raw[:, ttl_idx] / 32.0) * 255.0

        structured[:, ttl_idx] = ttl_val
        structured[:, min_idx] = min_base
        structured[:, avg_idx] = avg_val
        structured[:, max_idx] = max_val
        structured[:, std_idx] = std_val
        structured[:, variance_idx] = variance_val
        structured[:, number_idx] = number_val

        return structured

    def encode(self, x_39: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        device = x_39.device

        x_continuous = x_39[:, self.partition["continuous_idx"]]
        x_ind_bin = x_39[:, self.partition["independent_binary_idx"]]

        # CPU roundtrip: recover protocol index via nearest-reference lookup.
        # ref_proto_scaled holds the scaled value for each of the 6 allowlist
        # entries; argmin of absolute distance gives the protocol index.
        # This is exact when the input came from the same scaler that built the
        # references, and degrades gracefully (picks nearest) otherwise.
        proto_scaled_col = x_39[:, self.partition["protocol_idx"][0]]  # (N,) on device
        ref = self.ref_proto_scaled.to(device)  # (6,) on same device
        diffs = torch.abs(proto_scaled_col.unsqueeze(1) - ref.unsqueeze(0))  # (N, 6)
        proto_indices = diffs.argmin(dim=1)  # (N,) long, stays on device

        proto_emb = self.protocol_embed(proto_indices)  # (N, protocol_embed_dim)

        parts: list[torch.Tensor] = [x_continuous, x_ind_bin]
        if self.n_pseudo_binary > 0:
            pseudo_idx = self.partition.get("pseudo_binary_idx", [])
            x_pseudo = x_39[:, pseudo_idx]
            parts.append(x_pseudo)
        parts.append(proto_emb)

        h_in = torch.cat(parts, dim=1)
        h = self.encoder_body(h_in)
        out = self.encoder_out(h)

        mu = out[:, : self.latent_dim]
        logvar = out[:, self.latent_dim :]
        logvar = logvar.clamp(*self.latent_logvar_bounds)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return mu
        eps = torch.randn_like(mu)
        return mu + eps * torch.exp(0.5 * logvar)

    def decode_internal(self, z: torch.Tensor) -> dict:
        h = self.decoder_body(z)

        continuous_mu_unstructured = self.head_continuous_mu(h)
        continuous_mu_raw_unstructured = self.continuous_scaled_to_raw(continuous_mu_unstructured)
        if self.use_structured_continuous_decoder:
            continuous_mu_raw = self._structure_continuous_raw(continuous_mu_raw_unstructured)
            continuous_mu = self.continuous_raw_to_scaled(continuous_mu_raw)
        else:
            continuous_mu_raw = continuous_mu_raw_unstructured
            continuous_mu = continuous_mu_unstructured
        continuous_logvar = self.head_continuous_logvar(h).clamp(-7.0, 2.0)
        binary_logits = self.head_binary(h)
        protocol_logits = self.head_protocol(h)

        pseudo_binary_sigmoid: torch.Tensor | None = None
        if self.head_pseudo is not None:
            pseudo_binary_sigmoid = self.head_pseudo(h)

        return {
            "continuous_mu": continuous_mu,
            "continuous_mu_raw": continuous_mu_raw,
            "continuous_mu_raw_unstructured": continuous_mu_raw_unstructured,
            "continuous_logvar": continuous_logvar,
            "binary_logits": binary_logits,
            "protocol_logits": protocol_logits,
            "pseudo_binary_sigmoid": pseudo_binary_sigmoid,
        }

    def decode_to_39(
        self,
        z: torch.Tensor,
        scaler: "RobustScaler",
        mode: str = "soft",
    ) -> tuple[torch.Tensor, dict]:
        device = z.device
        N = z.shape[0]

        dec = self.decode_internal(z)

        continuous_mu = dec["continuous_mu"]

        if mode == "soft":
            binary_output = torch.sigmoid(dec["binary_logits"])
        else:
            binary_output = (dec["binary_logits"] > 0.0).float()

        protocol_idx_batch = dec["protocol_logits"].argmax(dim=1)  # (N,) long

        derived_raw = derive_binaries_from_protocol_index(protocol_idx_batch)  # (N, 4)
        derived_np = derived_raw.detach().cpu().numpy().astype(np.float64)
        derived_scaled_np = apply_scaler_to_columns(
            derived_np,
            scaler,
            col_indices=self.partition["derived_binary_idx"],
            n_features=39,
        ).astype(np.float32)
        derived_scaled = torch.from_numpy(derived_scaled_np).to(device)

        proto_idx_np = protocol_idx_batch.detach().cpu().numpy()
        raw_proto_np = np.array(
            [protocol_index_to_raw(int(i)) for i in proto_idx_np], dtype=np.float64
        )
        proto_scaled_np = raw_protocol_to_scaled(
            raw_proto_np,
            scaler,
            protocol_idx=self.partition["protocol_idx"][0],
            n_features=39,
        ).astype(np.float32)
        protocol_scaled = torch.from_numpy(proto_scaled_np).to(device)  # (N,)

        # binary_output is in raw {0,1} space (the head was trained with BCE against
        # raw targets). Convert to scaled space so inverse_transform yields valid {0,1}.
        # For most binary cols center_=0 so this is identity, but IPv/LLC have center_=1
        # which would otherwise produce raw=2.0 after inverse_transform.
        binary_np = binary_output.detach().cpu().numpy().astype(np.float64)
        binary_scaled_np = apply_scaler_to_columns(
            binary_np,
            scaler,
            col_indices=self.partition["independent_binary_idx"],
            n_features=39,
        ).astype(np.float32)
        binary_scaled = torch.from_numpy(binary_scaled_np).to(device)

        x_out = torch.zeros(N, 39, device=device, dtype=torch.float32)
        x_out[:, self.partition["continuous_idx"]] = continuous_mu
        x_out[:, self.partition["independent_binary_idx"]] = binary_scaled
        x_out[:, self.partition["derived_binary_idx"]] = derived_scaled
        x_out[:, self.partition["protocol_idx"]] = protocol_scaled.unsqueeze(1)

        decode_meta = {
            "protocol_logits": dec["protocol_logits"],
            "protocol_idx_batch": protocol_idx_batch,
            "binary_logits": dec["binary_logits"],
            "continuous_mu": dec["continuous_mu"],
            "continuous_mu_raw": dec["continuous_mu_raw"],
            "continuous_logvar": dec["continuous_logvar"],
        }

        return x_out, decode_meta

    def forward(self, x_39: torch.Tensor) -> dict:
        mu, logvar = self.encode(x_39)
        z = self.reparameterize(mu, logvar)
        dec = self.decode_internal(z)

        return {
            "mu": mu,
            "logvar": logvar,
            "z": z,
            "continuous_mu": dec["continuous_mu"],
            "continuous_mu_raw": dec["continuous_mu_raw"],
            "continuous_logvar": dec["continuous_logvar"],
            "binary_logits": dec["binary_logits"],
            "protocol_logits": dec["protocol_logits"],
            "pseudo_binary_sigmoid": dec["pseudo_binary_sigmoid"],
        }
