from __future__ import annotations

import json
import logging
import pickle
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.stats import chi2
from torch.utils.data import DataLoader

from attack.adversarial_attacks import load_model
from preprocessing.feature_groups import (
    FEATURE_NAMES,
    FULL_PERTURBABLE_OVERRIDE_FEATURES,
    MANUAL_CONCENTRATED_DECISIONS,
    MUTABLE_FEATURES,
)
from vae.config import CLASS_TO_ID, CLASSES, ID_TO_CLASS
from vae.dataset import PerClassDataset
from vae.model import MixedInputBetaVAE
from vae.schema import PROTOCOL_ALLOWLIST, get_partition, scaled_to_raw_protocol
from vae.train import _load_8class_labels
from vae.train_all import _resolve_model_hparams

LOGGER = logging.getLogger(__name__)

DEFAULT_PARTIAL_DELTA = 0.3
PROTOCOL_FEATURES = ["Protocol Type", "TCP", "UDP", "ICMP", "IGMP"]
PROTOCOL_FEATURE_INDICES = [FEATURE_NAMES.index(name) for name in PROTOCOL_FEATURES]


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _load_pickle(path: Path) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def _to_serializable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: _to_serializable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_serializable(v) for v in value]
    if isinstance(value, tuple):
        return [_to_serializable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


@dataclass
class AttackRunLogger:
    run_dir: Path
    seed: int
    summary_log_path: Path
    config_path: Path

    @classmethod
    def create(
        cls,
        *,
        phase_name: str,
        seed: int,
        config_snapshot: dict[str, Any],
        output_root: Path | None = None,
    ) -> "AttackRunLogger":
        root = output_root or (_repo_root() / "outputs" / "latent_attacks")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = root / f"{phase_name}_{timestamp}_seed{seed}"
        run_dir.mkdir(parents=True, exist_ok=False)

        config_path = run_dir / "config_snapshot.json"
        summary_log_path = run_dir / "summary.log"

        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(_to_serializable(config_snapshot), f, indent=2)

        with open(summary_log_path, "w", encoding="utf-8") as f:
            f.write(f"phase={phase_name}\n")
            f.write(f"seed={seed}\n")
            f.write(f"created_at={datetime.now().isoformat()}\n")

        return cls(
            run_dir=run_dir,
            seed=seed,
            summary_log_path=summary_log_path,
            config_path=config_path,
        )

    def log(self, message: str) -> None:
        with open(self.summary_log_path, "a", encoding="utf-8") as f:
            f.write(message.rstrip() + "\n")


@dataclass
class PerturbationMask:
    full_indices: list[int]
    partial_indices: list[int]
    frozen_indices: list[int]
    partial_lower_bounds: torch.Tensor
    partial_upper_bounds: torch.Tensor
    verify_tolerance: float = 1e-5

    @classmethod
    def from_preprocessing_artifacts(cls, repo_root: Path | None = None) -> "PerturbationMask":
        root = repo_root or _repo_root()
        near_zero = _load_json(root / "data" / "processed" / "near_zero_iqr_features.json")
        netdiffuser = _load_json(root / "data" / "processed" / "netdiffuser_categorization.json")

        mutable_set = set(MUTABLE_FEATURES)
        full_override = set(FULL_PERTURBABLE_OVERRIDE_FEATURES)
        discrete_set = set(netdiffuser["discrete"])
        relative_set = set(netdiffuser["relative"])

        near_zero_by_feature = {item["feature"]: item for item in near_zero.get("features", [])}
        auto_freeze = {
            item["feature"]
            for item in near_zero.get("features", [])
            if item.get("policy_action") == "auto_freeze"
        }

        for feat, decision in MANUAL_CONCENTRATED_DECISIONS.items():
            if feat not in mutable_set:
                continue
            if decision == "force_freeze":
                auto_freeze.add(feat)
            elif decision != "allow_mutable":
                raise ValueError(f"Unknown manual decision for {feat}: {decision}")

        capped_partial = {
            feat
            for feat, meta in near_zero_by_feature.items()
            if feat in mutable_set
            and meta.get("kind") in {"rare_signal", "concentrated"}
            and feat not in auto_freeze
        }

        full_indices: list[int] = []
        partial_indices: list[int] = []
        frozen_indices: list[int] = []

        for idx, feat in enumerate(FEATURE_NAMES):
            if feat in auto_freeze:
                frozen_indices.append(idx)
            elif feat in mutable_set and feat in full_override:
                full_indices.append(idx)
            elif feat in capped_partial:
                partial_indices.append(idx)
            elif feat in mutable_set and feat in discrete_set:
                full_indices.append(idx)
            elif feat in mutable_set and feat in relative_set:
                partial_indices.append(idx)
            else:
                frozen_indices.append(idx)

        partial_lower = torch.full((len(partial_indices),), -DEFAULT_PARTIAL_DELTA, dtype=torch.float32)
        partial_upper = torch.full((len(partial_indices),), DEFAULT_PARTIAL_DELTA, dtype=torch.float32)

        return cls(
            full_indices=full_indices,
            partial_indices=partial_indices,
            frozen_indices=frozen_indices,
            partial_lower_bounds=partial_lower,
            partial_upper_bounds=partial_upper,
        )

    @property
    def full_feature_names(self) -> list[str]:
        return [FEATURE_NAMES[idx] for idx in self.full_indices]

    @property
    def partial_feature_names(self) -> list[str]:
        return [FEATURE_NAMES[idx] for idx in self.partial_indices]

    @property
    def frozen_feature_names(self) -> list[str]:
        return [FEATURE_NAMES[idx] for idx in self.frozen_indices]

    def partial_bounds_by_index(self) -> dict[int, tuple[float, float]]:
        return {
            idx: (
                float(self.partial_lower_bounds[pos].item()),
                float(self.partial_upper_bounds[pos].item()),
            )
            for pos, idx in enumerate(self.partial_indices)
        }

    def apply(self, x_adv: torch.Tensor, x_original: torch.Tensor) -> torch.Tensor:
        if x_adv.shape != x_original.shape:
            raise ValueError(f"Shape mismatch: {x_adv.shape} vs {x_original.shape}")

        x_masked = x_adv.clone()
        device = x_adv.device
        dtype = x_adv.dtype

        if self.partial_indices:
            partial_idx = torch.tensor(self.partial_indices, device=device, dtype=torch.long)
            lower = self.partial_lower_bounds.to(device=device, dtype=dtype).unsqueeze(0)
            upper = self.partial_upper_bounds.to(device=device, dtype=dtype).unsqueeze(0)
            delta = x_adv[:, partial_idx] - x_original[:, partial_idx]
            delta = torch.clamp(delta, min=lower, max=upper)
            x_masked[:, partial_idx] = x_original[:, partial_idx] + delta

        if self.frozen_indices:
            frozen_idx = torch.tensor(self.frozen_indices, device=device, dtype=torch.long)
            x_masked[:, frozen_idx] = x_original[:, frozen_idx]

        return x_masked

    def verify(self, x_adv: torch.Tensor, x_original: torch.Tensor) -> dict[str, torch.Tensor]:
        if x_adv.shape != x_original.shape:
            raise ValueError(f"Shape mismatch: {x_adv.shape} vs {x_original.shape}")

        device = x_adv.device
        batch = x_adv.shape[0]
        true_mask = torch.ones(batch, dtype=torch.bool, device=device)

        if self.frozen_indices:
            frozen_idx = torch.tensor(self.frozen_indices, device=device, dtype=torch.long)
            frozen_ok = torch.all(x_adv[:, frozen_idx] == x_original[:, frozen_idx], dim=1)
        else:
            frozen_ok = true_mask

        if self.partial_indices:
            partial_idx = torch.tensor(self.partial_indices, device=device, dtype=torch.long)
            lower = self.partial_lower_bounds.to(device=device, dtype=x_adv.dtype).unsqueeze(0)
            upper = self.partial_upper_bounds.to(device=device, dtype=x_adv.dtype).unsqueeze(0)
            delta = x_adv[:, partial_idx] - x_original[:, partial_idx]
            tol = torch.tensor(self.verify_tolerance, device=device, dtype=x_adv.dtype)
            partial_ok = torch.all((delta >= (lower - tol)) & (delta <= (upper + tol)), dim=1)
        else:
            partial_ok = true_mask

        return {
            "frozen_exact": frozen_ok,
            "partial_within_bounds": partial_ok,
            "full_unconstrained": true_mask,
            "all_compliant": frozen_ok & partial_ok,
        }


class ProtocolValidator:
    def __init__(self, scaler: Any, protocol_feature_index: int | None = None) -> None:
        self.scaler = scaler
        self.protocol_feature_index = (
            FEATURE_NAMES.index("Protocol Type") if protocol_feature_index is None else protocol_feature_index
        )

    def validate(self, x_batch: torch.Tensor | np.ndarray, *, already_scaled: bool = True) -> torch.Tensor:
        if isinstance(x_batch, torch.Tensor):
            device = x_batch.device
            x_np = x_batch.detach().cpu().numpy()
        else:
            device = None
            x_np = np.asarray(x_batch)

        if already_scaled:
            proto_raw = scaled_to_raw_protocol(x_np, self.scaler, self.protocol_feature_index)
        else:
            proto_raw = np.round(x_np[:, self.protocol_feature_index]).astype(np.int64)

        valid = np.isin(proto_raw, np.array(PROTOCOL_ALLOWLIST, dtype=np.int64))
        valid_t = torch.from_numpy(valid.astype(np.bool_))
        return valid_t.to(device) if device is not None else valid_t


class AttackRouter:
    def __init__(
        self,
        *,
        repo_root: Path | None = None,
        device: str = "cpu",
        classifier_paths: dict[str, Path] | None = None,
    ) -> None:
        self.repo_root = repo_root or _repo_root()
        self.device = device
        self.partition = get_partition()
        self.scaler = _load_pickle(self.repo_root / "data" / "processed" / "scaler.pkl")
        self.manifest = _load_json(self.repo_root / "vae_run_manifest.json")
        self._vae_cache: dict[int, MixedInputBetaVAE] = {}
        self._classifier_cache: dict[str, Any] = {}
        self.classifier_paths = classifier_paths or {
            "mlp": self.repo_root / "models" / "mlp_8class.pt",
            "cnn": self.repo_root / "models" / "cnn_8class.pt",
            "lightgbm": self.repo_root / "models" / "lightgbm_8class.pkl",
        }

    def _normalise_classifier_name(self, classifier_name: str) -> str:
        key = classifier_name.strip().lower()
        alias_map = {
            "mlp-3l": "mlp",
            "mlp": "mlp",
            "cnn-1d": "cnn",
            "cnn": "cnn",
            "lightgbm": "lightgbm",
            "lgbm": "lightgbm",
        }
        if key not in alias_map:
            raise KeyError(f"Unknown classifier alias: {classifier_name}")
        return alias_map[key]

    def get_vae(self, class_id: int) -> MixedInputBetaVAE:
        if class_id in self._vae_cache:
            return self._vae_cache[class_id]

        class_name = ID_TO_CLASS[class_id]
        checkpoint_info = self.manifest["checkpoints"][class_name]
        ckpt_path = Path(checkpoint_info["path"])
        ckpt = torch.load(str(ckpt_path), map_location=self.device, weights_only=False)
        ckpt_config = ckpt.get("config", {})

        (
            latent_dim,
            encoder_hidden,
            decoder_hidden,
            protocol_embed_dim,
            use_structured_continuous_decoder,
            structured_continuous_mode,
            structured_std_floor,
            latent_logvar_bounds,
        ) = _resolve_model_hparams(ckpt_config, class_name)

        vae = MixedInputBetaVAE(
            partition=self.partition,
            latent_dim=latent_dim,
            protocol_embed_dim=protocol_embed_dim,
            encoder_hidden=encoder_hidden,
            decoder_hidden=decoder_hidden,
            n_pseudo_binary=0,
            use_structured_continuous_decoder=use_structured_continuous_decoder,
            structured_continuous_mode=structured_continuous_mode,
            structured_std_floor=structured_std_floor,
            latent_logvar_bounds=latent_logvar_bounds,
        )
        vae.load_state_dict(ckpt["state_dict"], strict=False)
        vae.register_protocol_references(self.scaler)
        vae = vae.to(self.device)
        vae.eval()
        self._vae_cache[class_id] = vae
        return vae

    def get_classifier(self, classifier_name: str) -> Any:
        key = self._normalise_classifier_name(classifier_name)
        if key in self._classifier_cache:
            return self._classifier_cache[key]

        path = self.classifier_paths[key]
        if not path.exists():
            raise FileNotFoundError(
                f"Classifier checkpoint for '{classifier_name}' not found at {path}"
            )

        if key in {"mlp", "cnn"}:
            model = load_model(
                model_path=str(path),
                num_features=len(FEATURE_NAMES),
                num_classes=len(CLASSES),
                device=self.device,
            )
        elif key == "lightgbm":
            model = _load_pickle(path)
        else:
            raise KeyError(f"Unsupported classifier key: {key}")

        self._classifier_cache[key] = model
        return model

    def route(self, class_id: int, classifier_name: str) -> dict[str, Any]:
        return {
            "class_id": class_id,
            "class_name": ID_TO_CLASS[class_id],
            "vae": self.get_vae(class_id),
            "classifier_name": self._normalise_classifier_name(classifier_name),
            "classifier": self.get_classifier(classifier_name),
            "scaler": self.scaler,
            "partition": self.partition,
        }


class MahalanobisOutlierDetector:
    def __init__(
        self,
        latent_dim: int = 16,
        *,
        target_clean_outlier_rate: float = 0.05,
        ridge_grid: tuple[float, ...] = (0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0),
    ) -> None:
        self.latent_dim = latent_dim
        self.target_clean_outlier_rate = target_clean_outlier_rate
        self.ridge_grid = ridge_grid
        self.stats_by_class: dict[int, dict[str, Any]] = {}

    def fit(
        self,
        class_id: int,
        z_benign_held_out: torch.Tensor,
        *,
        collapsed_dims: list[int] | None = None,
    ) -> dict[str, Any]:
        if z_benign_held_out.ndim != 2:
            raise ValueError(f"Expected 2D latent tensor, got {z_benign_held_out.shape}")

        collapsed = sorted(set(collapsed_dims or []))
        active_dims = [dim for dim in range(z_benign_held_out.shape[1]) if dim not in collapsed]
        if not active_dims:
            raise ValueError(f"Class {class_id} has no active latent dimensions")

        z_active = z_benign_held_out[:, active_dims]
        mu = z_active.mean(dim=0)
        centered = z_active - mu.unsqueeze(0)
        cov = torch.cov(centered.T)
        if cov.ndim == 0:
            cov = cov.reshape(1, 1)

        k_active = len(active_dims)
        threshold = float(chi2.ppf(0.95, df=k_active))

        best_precision: torch.Tensor | None = None
        best_rate: float | None = None
        best_ridge = 0.0
        best_score: float | None = None
        used_pinv = False
        identity = torch.eye(cov.shape[0], dtype=cov.dtype, device=cov.device)

        # Calibrate a light ridge on the empirical covariance so the nominal 95%
        # chi-square boundary has near-95% in-distribution coverage on clean
        # latent codes. This makes the detector materially more stable for the
        # high-collapse classes without changing the underlying MD definition.
        for ridge in self.ridge_grid:
            cov_adjusted = cov + float(ridge) * identity
            try:
                precision_candidate = torch.linalg.inv(cov_adjusted)
                used_pinv_candidate = False
            except RuntimeError:
                precision_candidate = torch.linalg.pinv(cov_adjusted)
                used_pinv_candidate = True

            md_sq = torch.einsum("bi,ij,bj->b", centered, precision_candidate, centered)
            clean_rate = float((md_sq > threshold).float().mean().item())
            score = abs(clean_rate - self.target_clean_outlier_rate)

            if best_score is None or score < best_score:
                best_score = score
                best_precision = precision_candidate
                best_rate = clean_rate
                best_ridge = float(ridge)
                used_pinv = used_pinv_candidate

        assert best_precision is not None
        assert best_rate is not None
        stats = {
            "mu": mu,
            "cov": cov,
            "precision": best_precision,
            "collapsed_dims": collapsed,
            "active_dims": active_dims,
            "effective_dimensionality": k_active,
            "threshold_95": threshold,
            "used_pinv": used_pinv,
            "ridge_lambda": best_ridge,
            "fit_clean_outlier_rate": best_rate,
        }
        self.stats_by_class[class_id] = stats
        return stats

    def mahalanobis_sq(self, class_id: int, z_batch: torch.Tensor) -> torch.Tensor:
        if class_id not in self.stats_by_class:
            raise KeyError(f"Class {class_id} has not been fit yet")

        stats = self.stats_by_class[class_id]
        active_dims = stats["active_dims"]
        z_active = z_batch[:, active_dims]
        diff = z_active - stats["mu"].to(z_batch.device).unsqueeze(0)
        precision = stats["precision"].to(z_batch.device)
        return torch.einsum("bi,ij,bj->b", diff, precision, diff)

    def outlier_mask(self, class_id: int, z_batch: torch.Tensor) -> torch.Tensor:
        stats = self.stats_by_class[class_id]
        md_sq = self.mahalanobis_sq(class_id, z_batch)
        return md_sq > stats["threshold_95"]

    def outlier_rate(self, class_id: int, z_batch: torch.Tensor) -> float:
        return float(self.outlier_mask(class_id, z_batch).float().mean().item())


def load_validation_split(repo_root: Path | None = None) -> dict[str, Any]:
    return load_split("val", repo_root=repo_root)


def load_split(split_name: str, repo_root: Path | None = None) -> dict[str, Any]:
    root = repo_root or _repo_root()
    X_split = np.load(root / "data" / "processed" / f"X_{split_name}.npy")
    y_split_34 = np.load(root / "data" / "processed" / f"y_{split_name}.npy")
    scaler = _load_pickle(root / "data" / "processed" / "scaler.pkl")
    y_split_8 = _load_8class_labels(root, y_split_34, split_name)
    partition = get_partition()
    return {
        "X": X_split,
        "y_8": y_split_8,
        "scaler": scaler,
        "partition": partition,
        "split_name": split_name,
    }


def build_per_class_dataset(
    class_id: int,
    *,
    X_val: np.ndarray,
    y_val_8: np.ndarray,
    scaler: Any,
    partition: dict[str, list[int]],
) -> PerClassDataset:
    return PerClassDataset(X_val, y_val_8, class_id, scaler, partition)


def encode_dataset_mu(
    model: MixedInputBetaVAE,
    dataset: PerClassDataset,
    *,
    device: str,
    batch_size: int = 2048,
) -> torch.Tensor:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    mu_batches: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            x = batch["x_scaled"].to(device)
            mu, _ = model.encode(x)
            mu_batches.append(mu.cpu())
    return torch.cat(mu_batches, dim=0)


def load_collapsed_dims(repo_root: Path | None = None) -> dict[int, list[int]]:
    root = repo_root or _repo_root()
    collapsed: dict[int, list[int]] = {}
    for class_name in CLASSES:
        diag_path = root / "results" / "vae" / f"diagnostics_{class_name}.json"
        diag = _load_json(diag_path)
        collapsed[CLASS_TO_ID[class_name]] = diag["posterior_collapse"]["collapsed_dim_indices"]
    return collapsed


def inverse_transform_scaled(x_scaled: torch.Tensor | np.ndarray, scaler: Any) -> np.ndarray:
    if isinstance(x_scaled, torch.Tensor):
        x_np = x_scaled.detach().cpu().numpy()
    else:
        x_np = np.asarray(x_scaled)
    return scaler.inverse_transform(x_np.astype(np.float64))


def reimpose_protocol_features(x_adv: torch.Tensor, x_original: torch.Tensor) -> torch.Tensor:
    if x_adv.shape != x_original.shape:
        raise ValueError(f"Shape mismatch: {x_adv.shape} vs {x_original.shape}")

    x_fixed = x_adv.clone()
    # Protocol Type and its four derived binary indicators are frozen by design,
    # so every decode step must overwrite them from the original sample while
    # keeping those values detached from the attack graph.
    x_fixed[:, PROTOCOL_FEATURE_INDICES] = x_original[:, PROTOCOL_FEATURE_INDICES].detach()
    return x_fixed


def predict_labels(classifier: Any, x_batch: torch.Tensor, *, device: str) -> torch.Tensor:
    if hasattr(classifier, "predict") and not isinstance(classifier, torch.nn.Module):
        preds = classifier.predict(x_batch.detach().cpu().numpy())
        return torch.as_tensor(preds, dtype=torch.long)

    if not isinstance(classifier, torch.nn.Module):
        raise TypeError(f"Unsupported classifier type: {type(classifier)!r}")

    classifier = classifier.to(device)
    classifier.eval()
    with torch.no_grad():
        logits = classifier(x_batch.to(device))
        if isinstance(logits, tuple):
            logits = logits[0]
        preds = torch.argmax(logits, dim=1)
    return preds.detach().cpu()


def phase0_config_snapshot(seed: int, device: str) -> dict[str, Any]:
    mask = PerturbationMask.from_preprocessing_artifacts()
    return {
        "seed": seed,
        "device": device,
        "latent_dim": 16,
        "protocol_allowlist": PROTOCOL_ALLOWLIST,
        "protocol_feature_indices": PROTOCOL_FEATURE_INDICES,
        "mask": {
            "full_indices": mask.full_indices,
            "partial_indices": mask.partial_indices,
            "frozen_indices": mask.frozen_indices,
            "partial_delta_bounds": mask.partial_bounds_by_index(),
        },
        "classes": CLASSES,
    }
