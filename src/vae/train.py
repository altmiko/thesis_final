"""
Single-class VAE training loop.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from preprocessing.feature_groups import FEATURE_NAMES
from vae.dataset import PerClassDataset
from vae.losses import BetaScheduler, compute_elbo
from vae.model import MixedInputBetaVAE
from vae.schema import PROTOCOL_ALLOWLIST, get_partition

logger = logging.getLogger(__name__)


def _compute_reconstruction_metric(
    model: MixedInputBetaVAE,
    data_loader: DataLoader,
    scaler,
    device: str,
    quantile: float | None = None,
) -> float:
    """Deterministic reconstruction error used for early stopping."""
    errs: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for batch in data_loader:
            x = batch["x_scaled"].to(device)
            mu, _ = model.encode(x)
            x_recon, _ = model.decode_to_39(mu, scaler, mode="hard")
            num = torch.linalg.norm(x - x_recon, dim=1)
            den = torch.linalg.norm(x, dim=1).clamp_min(1e-12)
            errs.append((num / den).cpu())
    if not errs:
        return float("inf")
    err_all = torch.cat(errs, dim=0)
    if quantile is None:
        return float(err_all.mean().item())
    return float(torch.quantile(err_all, float(quantile)).item())


def _build_feature_weight_tensor(
    partition_feature_indices: list[int],
    feature_weight_map: dict[str, float] | None,
    device: str,
    normalize: bool = True,
) -> torch.Tensor | None:
    if not feature_weight_map:
        return None

    weights = torch.ones(len(partition_feature_indices), dtype=torch.float32, device=device)
    for pos, full_idx in enumerate(partition_feature_indices):
        feature_name = FEATURE_NAMES[full_idx]
        if feature_name in feature_weight_map:
            weights[pos] = float(feature_weight_map[feature_name])

    if normalize:
        weights = weights / weights.mean().clamp_min(1e-8)

    return weights


def _build_sample_weight_array(
    x_scaled: torch.Tensor,
    sample_weight_rules: list[dict[str, Any]] | None,
) -> np.ndarray | None:
    if not sample_weight_rules:
        return None

    x_np = x_scaled.detach().cpu().numpy()
    weights = np.ones(x_np.shape[0], dtype=np.float64)

    for rule in sample_weight_rules:
        feature = str(rule["feature"])
        idx = FEATURE_NAMES.index(feature)
        mode = str(rule.get("mode", "high_quantile"))
        bonus = float(rule.get("bonus", 0.0))

        if mode == "high_quantile":
            threshold = np.quantile(x_np[:, idx], float(rule["quantile"]))
            mask = x_np[:, idx] >= threshold
        elif mode == "low_quantile":
            threshold = np.quantile(x_np[:, idx], float(rule["quantile"]))
            mask = x_np[:, idx] <= threshold
        elif mode == "binary_on":
            mask = x_np[:, idx] > 0.5
        else:
            raise ValueError(f"Unknown sample weight rule mode: {mode}")

        weights[mask] += bonus

    return weights


# ---------------------------------------------------------------------------
# 8-class label helpers
# ---------------------------------------------------------------------------

def _load_8class_labels(
    root: Path,
    y_34: np.ndarray,
    split_name: str,
) -> np.ndarray:
    """Load or derive 8-class (category) labels for a given split.

    Tries ``y_{split}_cat.npy`` first (produced by preprocessing), then
    falls back to mapping via CATEGORY_MAP.

    Parameters
    ----------
    root:
        Repository root path.
    y_34:
        (N,) array of 34-class encoded integer labels.
    split_name:
        One of ``'train'``, ``'val'``, ``'test'``.

    Returns
    -------
    (N,) array of 8-class integer labels (0..7).
    """
    cat_path = root / "data" / "processed" / f"y_{split_name}_cat.npy"
    if cat_path.exists():
        logger.info("Loading 8-class labels from %s", cat_path)
        return np.load(str(cat_path))

    # Fallback: derive via CATEGORY_MAP + label_encoder
    logger.warning(
        "y_%s_cat.npy not found — deriving 8-class labels from CATEGORY_MAP", split_name
    )
    import sys
    src_path = str(root / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

    from preprocessing.feature_groups import CATEGORY_MAP  # type: ignore

    # Load 34-class encoder to get string label names
    encoder_path = root / "data" / "processed" / "label_encoder.pkl"
    with open(str(encoder_path), "rb") as f:
        label_enc_34 = pickle.load(f)

    # Load 8-class encoder to get category integer indices
    cat_encoder_path = root / "data" / "processed" / "category_encoder.pkl"
    with open(str(cat_encoder_path), "rb") as f:
        cat_enc = pickle.load(f)

    # Map: 34-class int → label string → category string → 8-class int
    label_strings = label_enc_34.inverse_transform(y_34)  # (N,)
    category_strings = np.array(
        [CATEGORY_MAP.get(lbl, "Benign") for lbl in label_strings], dtype=object
    )
    y_8 = cat_enc.transform(category_strings).astype(np.int64)
    return y_8


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train_one_vae(
    class_id: int,
    class_name: str,
    config: dict,
    device: str = "cpu",
    shared_arrays: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Train a single per-class β-VAE.

    Parameters
    ----------
    class_id:
        8-class integer label (0..7).
    class_name:
        Human-readable class name, e.g. ``'DDoS'``.
    config:
        Hyperparameter dict (see ``vae.config`` for structure).
    device:
        PyTorch device string, e.g. ``'cpu'`` or ``'cuda'``.

    Returns
    -------
    Metrics dict with keys: class_id, class_name, n_train, n_val,
    best_val_loss, epochs_trained, final_kl, train_history, val_history,
    checkpoint_path.
    """
    # ------------------------------------------------------------------
    # 1. Seeds and determinism
    # ------------------------------------------------------------------
    torch.manual_seed(42)
    np.random.seed(42)
    torch.backends.cudnn.deterministic = True

    # ------------------------------------------------------------------
    # 2. Load data
    # ------------------------------------------------------------------
    root = Path(__file__).resolve().parents[2]
    logger.info("[Class %d/%s] Loading split arrays from %s", class_id, class_name, root)

    if shared_arrays is None:
        X_train = np.load(str(root / "data" / "processed" / "X_train.npy"))
        y_train_34 = np.load(str(root / "data" / "processed" / "y_train.npy"))
        X_val = np.load(str(root / "data" / "processed" / "X_val.npy"))
        y_val_34 = np.load(str(root / "data" / "processed" / "y_val.npy"))

        with open(str(root / "data" / "processed" / "scaler.pkl"), "rb") as f:
            scaler = pickle.load(f)
    else:
        X_train = shared_arrays["X_train"]
        y_train_34 = shared_arrays["y_train_34"]
        X_val = shared_arrays["X_val"]
        y_val_34 = shared_arrays["y_val_34"]
        scaler = shared_arrays["scaler"]

    # ------------------------------------------------------------------
    # 3. 8-class label mapping
    # ------------------------------------------------------------------
    if shared_arrays is None or "y_train_8" not in shared_arrays:
        y_train_8 = _load_8class_labels(root, y_train_34, "train")
        y_val_8 = _load_8class_labels(root, y_val_34, "val")
    else:
        y_train_8 = shared_arrays["y_train_8"]
        y_val_8 = shared_arrays["y_val_8"]

    n_train_class = int((y_train_8 == class_id).sum())
    n_val_class = int((y_val_8 == class_id).sum())
    logger.info(
        "[Class %d/%s] 8-class split: train=%d, val=%d",
        class_id, class_name, n_train_class, n_val_class,
    )

    # ------------------------------------------------------------------
    # 4. Datasets and loaders
    # ------------------------------------------------------------------
    partition = get_partition()
    train_ds = PerClassDataset(X_train, y_train_8, class_id, scaler, partition)
    val_ds = PerClassDataset(X_val, y_val_8, class_id, scaler, partition)

    sample_weight_array = _build_sample_weight_array(
        train_ds.x_scaled,
        config.get("train_sample_weight_rules"),
    )
    if sample_weight_array is not None:
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sample_weight_array, dtype=torch.double),
            num_samples=len(train_ds),
            replacement=True,
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=config["batch_size"],
            shuffle=False,
            sampler=sampler,
            num_workers=config.get("num_workers", 0),
            pin_memory=(device == "cuda"),
        )
        logger.info(
            "[Class %d/%s] Weighted sampling enabled: min=%.3f mean=%.3f max=%.3f",
            class_id,
            class_name,
            float(sample_weight_array.min()),
            float(sample_weight_array.mean()),
            float(sample_weight_array.max()),
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=config.get("num_workers", 0),
            pin_memory=(device == "cuda"),
        )
    val_loader = DataLoader(
        val_ds,
        batch_size=config["batch_size"] * 2,
        shuffle=False,
        num_workers=config.get("num_workers", 0),
        pin_memory=(device == "cuda"),
    )

    protocol_class_weights_t: torch.Tensor | None = None
    if config.get("use_protocol_class_weights", True):
        proto_counts = torch.bincount(
            train_ds.target_protocol_index,
            minlength=len(PROTOCOL_ALLOWLIST),
        ).float()
        power = float(config.get("protocol_class_weight_power", 0.5))
        proto_counts = proto_counts.clamp_min(1.0)
        protocol_class_weights_t = proto_counts.pow(-power)
        protocol_class_weights_t = protocol_class_weights_t / protocol_class_weights_t.mean()
        protocol_class_weights_t = protocol_class_weights_t.to(device)
        logger.info(
            "[Class %d/%s] Protocol class weights: %s",
            class_id,
            class_name,
            [round(float(x), 4) for x in protocol_class_weights_t.detach().cpu()],
        )

    continuous_feature_weights_t = _build_feature_weight_tensor(
        partition["continuous_idx"],
        config.get("continuous_feature_loss_weights"),
        device,
        normalize=bool(config.get("normalize_feature_loss_weights", True)),
    )
    binary_feature_weights_t = _build_feature_weight_tensor(
        partition["independent_binary_idx"],
        config.get("binary_feature_loss_weights"),
        device,
        normalize=bool(config.get("normalize_feature_loss_weights", True)),
    )
    raw_relative_feature_weights_t = _build_feature_weight_tensor(
        partition["continuous_idx"],
        config.get("raw_relative_feature_loss_weights"),
        device,
        normalize=bool(config.get("normalize_feature_loss_weights", True)),
    )

    if continuous_feature_weights_t is not None:
        logger.info(
            "[Class %d/%s] Continuous loss weights (normalized=%s): %s",
            class_id,
            class_name,
            bool(config.get("normalize_feature_loss_weights", True)),
            [
                (
                    FEATURE_NAMES[full_idx],
                    round(float(continuous_feature_weights_t[pos].detach().cpu()), 4),
                )
                for pos, full_idx in enumerate(partition["continuous_idx"])
                if FEATURE_NAMES[full_idx] in config.get("continuous_feature_loss_weights", {})
            ],
        )
    if binary_feature_weights_t is not None:
        logger.info(
            "[Class %d/%s] Binary loss weights (normalized=%s): %s",
            class_id,
            class_name,
            bool(config.get("normalize_feature_loss_weights", True)),
            [
                (
                    FEATURE_NAMES[full_idx],
                    round(float(binary_feature_weights_t[pos].detach().cpu()), 4),
                )
                for pos, full_idx in enumerate(partition["independent_binary_idx"])
                if FEATURE_NAMES[full_idx] in config.get("binary_feature_loss_weights", {})
            ],
        )
    if raw_relative_feature_weights_t is not None:
        logger.info(
            "[Class %d/%s] Raw-relative continuous weights (normalized=%s): %s",
            class_id,
            class_name,
            bool(config.get("normalize_feature_loss_weights", True)),
            [
                (
                    FEATURE_NAMES[full_idx],
                    round(float(raw_relative_feature_weights_t[pos].detach().cpu()), 4),
                )
                for pos, full_idx in enumerate(partition["continuous_idx"])
                if FEATURE_NAMES[full_idx] in config.get("raw_relative_feature_loss_weights", {})
            ],
        )

    # ------------------------------------------------------------------
    # 5. Model
    # ------------------------------------------------------------------
    # Resolve per-class latent dim: config['latent_dim'] may be a dict or scalar
    latent_cfg = config["latent_dim"]
    if isinstance(latent_cfg, dict):
        latent_dim = latent_cfg[class_name]
    else:
        latent_dim = int(latent_cfg)

    # Per-class beta target
    beta_cfg = config["beta_target"]
    if isinstance(beta_cfg, dict):
        beta_target = float(beta_cfg[class_name])
    else:
        beta_target = float(beta_cfg)

    use_structured_continuous_decoder = bool(
        config.get("use_structured_continuous_decoder", False)
    )
    structured_continuous_mode = str(config.get("structured_continuous_mode", "full"))
    structured_std_floor = float(
        config.get(
            "structured_std_floor",
            0.01 if use_structured_continuous_decoder else 0.0,
        )
    )

    model = MixedInputBetaVAE(
        partition=partition,
        latent_dim=latent_dim,
        protocol_embed_dim=config["protocol_embed_dim"],
        encoder_hidden=tuple(config.get("encoder_hidden", [128, 64])),
        decoder_hidden=tuple(config.get("decoder_hidden", [64, 128])),
        n_pseudo_binary=0,
        use_structured_continuous_decoder=use_structured_continuous_decoder,
        structured_continuous_mode=structured_continuous_mode,
        structured_std_floor=structured_std_floor,
        latent_logvar_bounds=(
            float(config.get("latent_logvar_floor", -6.0)),
            float(config.get("latent_logvar_ceiling", 6.0)),
        ),
    )
    model.register_protocol_references(scaler)
    model = model.to(device)

    logger.info(
        "[Class %d/%s] Model: latent_dim=%d, beta_target=%.4f, params=%d",
        class_id, class_name, latent_dim, beta_target,
        sum(p.numel() for p in model.parameters()),
    )

    # ------------------------------------------------------------------
    # 6. Optimizer and schedulers
    # ------------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["max_epochs"]
    )

    total_steps = config["max_epochs"] * len(train_loader)
    beta_scheduler = BetaScheduler(
        beta_target=beta_target,
        total_steps=total_steps,
        warmup_frac=config["warmup_frac"],
    )

    # ------------------------------------------------------------------
    # 7. Training loop
    # ------------------------------------------------------------------
    max_epochs: int = config["max_epochs"]
    patience: int = config["early_stop_patience"]
    best_val_loss: float = float("inf")
    patience_counter: int = 0
    epoch_count: int = 0
    final_kl: float = 0.0

    # Per-metric history lists for plotting
    _metric_keys = [
        "loss", "recon_continuous", "recon_independent_binary",
        "recon_protocol", "recon_continuous_raw_relative",
        "kl", "constraint_loss", "beta", "recon_metric",
    ]
    train_history: dict[str, list[float]] = {k: [] for k in _metric_keys}
    val_history: dict[str, list[float]] = {
        k: [] for k in _metric_keys if k != "beta"
    }

    # Directory for checkpoints
    checkpoint_dir = root / "models" / "vae"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_name = str(
        config.get("checkpoint_name_override", f"vae_class_{class_id}_{class_name}.pt")
    )
    checkpoint_path = checkpoint_dir / checkpoint_name
    early_stop_metric = str(config.get("early_stop_metric", "val_loss"))
    recon_metric_every = int(config.get("recon_metric_every_n_epochs", 1))
    recon_metric_quantile = config.get("recon_metric_quantile")
    best_monitor_value: float = float("inf")
    best_epoch: int = 0

    for epoch in range(1, max_epochs + 1):
        epoch_count = epoch

        # ---- Train ----
        model.train()
        train_accum: dict[str, float] = {k: 0.0 for k in _metric_keys}
        n_train_batches = 0

        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)

            out = model(batch["x_scaled"])
            x_cont_target_raw = model.continuous_scaled_to_raw(
                batch["x_scaled"][:, partition["continuous_idx"]]
            ).detach()

            beta = beta_scheduler.step()
            elbo = compute_elbo(
                batch["x_scaled"],
                out,
                partition,
                beta,
                target_ind_binary=batch["target_ind_binary"],
                target_protocol_idx=batch["target_proto_idx"],
                protocol_class_weights=protocol_class_weights_t,
                protocol_loss_weight=float(config.get("protocol_loss_weight", 1.0)),
                constraint_loss_weight=float(config.get("constraint_loss_weight", 0.0)),
                continuous_feature_weights=continuous_feature_weights_t,
                binary_feature_weights=binary_feature_weights_t,
                continuous_logvar_floor=float(config.get("continuous_logvar_floor", -4.0)),
                continuous_logvar_ceiling=float(config.get("continuous_logvar_ceiling", 2.0)),
                continuous_nll_per_sample_cap=config.get("continuous_nll_per_sample_cap"),
                free_bits_lambda=float(config.get("free_bits_lambda", 0.0)),
                continuous_target_raw=x_cont_target_raw,
                raw_relative_continuous_loss_weight=float(
                    config.get("raw_relative_continuous_loss_weight", 0.0)
                ),
                raw_relative_feature_weights=raw_relative_feature_weights_t,
                raw_relative_epsilon=float(config.get("raw_relative_epsilon", 1.0)),
                raw_relative_tail_focus_quantile=config.get("raw_relative_tail_focus_quantile"),
                raw_relative_tail_focus_weight=float(
                    config.get("raw_relative_tail_focus_weight", 0.0)
                ),
            )

            loss = elbo["loss"]
            if not torch.isfinite(loss):
                logger.warning(
                    "[Class %d/%s] Skipping non-finite train loss at epoch %d batch %d",
                    class_id,
                    class_name,
                    epoch,
                    n_train_batches + 1,
                )
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=float(config.get("grad_clip", 5.0)),
            )
            optimizer.step()

            for k in _metric_keys:
                if k == "beta":
                    train_accum[k] += beta
                elif k == "recon_metric":
                    continue
                else:
                    train_accum[k] += elbo[k].item()
            n_train_batches += 1

        for k in _metric_keys:
            if k == "recon_metric":
                train_history[k].append(float("nan"))
            else:
                train_history[k].append(train_accum[k] / max(n_train_batches, 1))

        # ---- Validation ----
        model.eval()
        val_accum: dict[str, float] = {
            k: 0.0 for k in _metric_keys if k not in {"beta", "recon_metric"}
        }
        n_val_batches = 0
        current_beta = beta_scheduler.current_beta

        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                out = model(batch["x_scaled"])
                x_cont_target_raw = model.continuous_scaled_to_raw(
                    batch["x_scaled"][:, partition["continuous_idx"]]
                ).detach()
                elbo = compute_elbo(
                    batch["x_scaled"],
                    out,
                    partition,
                    current_beta,
                    target_ind_binary=batch["target_ind_binary"],
                    target_protocol_idx=batch["target_proto_idx"],
                    protocol_class_weights=protocol_class_weights_t,
                    protocol_loss_weight=float(config.get("protocol_loss_weight", 1.0)),
                    constraint_loss_weight=float(config.get("constraint_loss_weight", 0.0)),
                    continuous_feature_weights=continuous_feature_weights_t,
                    binary_feature_weights=binary_feature_weights_t,
                    continuous_logvar_floor=float(config.get("continuous_logvar_floor", -4.0)),
                    continuous_logvar_ceiling=float(config.get("continuous_logvar_ceiling", 2.0)),
                    continuous_nll_per_sample_cap=config.get("continuous_nll_per_sample_cap"),
                    free_bits_lambda=float(config.get("free_bits_lambda", 0.0)),
                    continuous_target_raw=x_cont_target_raw,
                    raw_relative_continuous_loss_weight=float(
                        config.get("raw_relative_continuous_loss_weight", 0.0)
                    ),
                    raw_relative_feature_weights=raw_relative_feature_weights_t,
                    raw_relative_epsilon=float(config.get("raw_relative_epsilon", 1.0)),
                    raw_relative_tail_focus_quantile=config.get("raw_relative_tail_focus_quantile"),
                    raw_relative_tail_focus_weight=float(
                        config.get("raw_relative_tail_focus_weight", 0.0)
                    ),
                )
                if not torch.isfinite(elbo["loss"]):
                    logger.warning(
                        "[Class %d/%s] Skipping non-finite val loss at epoch %d batch %d",
                        class_id,
                        class_name,
                        epoch,
                        n_val_batches + 1,
                    )
                    continue
                for k in val_accum:
                    val_accum[k] += elbo[k].item()
                n_val_batches += 1

        for k in val_accum:
            val_history[k].append(val_accum[k] / max(n_val_batches, 1))

        if epoch % recon_metric_every == 0:
            recon_metric = _compute_reconstruction_metric(
                model=model,
                data_loader=val_loader,
                scaler=scaler,
                device=device,
                quantile=float(recon_metric_quantile) if recon_metric_quantile is not None else None,
            )
        else:
            recon_metric = val_history["recon_metric"][-1] if val_history["recon_metric"] else float("inf")
        val_history["recon_metric"].append(recon_metric)

        train_loss = train_history["loss"][-1]
        val_loss = val_history["loss"][-1]
        current_kl = val_history["kl"][-1]
        current_recon_metric = val_history["recon_metric"][-1]
        current_lr = optimizer.param_groups[0]["lr"]

        logger.info(
            "[Class %d/%s] Epoch %d/%d | train_loss=%.4f val_loss=%.4f"
            " recon_metric=%.4f constraint=%.4f beta=%.4f lr=%.2e",
            class_id, class_name, epoch, max_epochs,
            train_loss, val_loss, current_recon_metric,
            val_history["constraint_loss"][-1], current_beta, current_lr,
        )

        # ---- LR scheduler step ----
        lr_scheduler.step()

        # ---- Early stopping and checkpoint ----
        monitor_value = current_recon_metric if early_stop_metric == "recon_metric" else val_loss
        if monitor_value < best_monitor_value:
            best_monitor_value = monitor_value
            best_epoch = epoch
            best_val_loss = val_loss
            final_kl = current_kl
            patience_counter = 0
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "config": config,
                    "partition": partition,
                    "pseudo_binary_columns": [],
                    "protocol_allowlist": PROTOCOL_ALLOWLIST,
                    "val_history": val_history,
                    "best_val_loss": best_val_loss,
                    "best_monitor_value": best_monitor_value,
                    "best_epoch": best_epoch,
                    "early_stop_metric": early_stop_metric,
                    "epoch": epoch,
                    "class_id": class_id,
                    "class_name": class_name,
                },
                str(checkpoint_path),
            )
            logger.info(
                "[Class %d/%s] New best %s=%.4f — checkpoint saved.",
                class_id, class_name, early_stop_metric, best_monitor_value,
            )
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(
                    "[Class %d/%s] Early stop after %d epochs (patience=%d).",
                    class_id, class_name, epoch, patience,
                )
                break

    # ------------------------------------------------------------------
    # 8. Training curves
    # ------------------------------------------------------------------
    curves_dir = root / "results" / "vae"
    curves_dir.mkdir(parents=True, exist_ok=True)
    curves_name = str(config.get("curves_name_override", f"curves_{class_name}.png"))
    curves_path = curves_dir / curves_name

    fig, axes = plt.subplots(1, 6, figsize=(24, 4))
    fig.suptitle(f"VAE Training Curves — Class {class_id} ({class_name})", fontsize=12)

    # Subplot 0: total loss
    ax = axes[0]
    ax.plot(train_history["loss"], label="train")
    ax.plot(val_history["loss"], label="val")
    ax.set_title("Total Loss")
    ax.set_xlabel("Epoch")
    ax.legend()

    # Subplot 1: continuous recon
    ax = axes[1]
    ax.plot(train_history["recon_continuous"], label="train")
    ax.plot(val_history["recon_continuous"], label="val")
    ax.set_title("Continuous Recon")
    ax.set_xlabel("Epoch")
    ax.legend()

    # Subplot 2: binary + protocol recon
    ax = axes[2]
    train_binprot = [
        b + p
        for b, p in zip(
            train_history["recon_independent_binary"],
            train_history["recon_protocol"],
        )
    ]
    val_binprot = [
        b + p
        for b, p in zip(
            val_history["recon_independent_binary"],
            val_history["recon_protocol"],
        )
    ]
    ax.plot(train_binprot, label="train")
    ax.plot(val_binprot, label="val")
    ax.set_title("Binary + Protocol Recon")
    ax.set_xlabel("Epoch")
    ax.legend()

    # Subplot 3: KL
    ax = axes[3]
    ax.plot(train_history["kl"], label="train")
    ax.plot(val_history["kl"], label="val")
    ax.set_title("KL Divergence")
    ax.set_xlabel("Epoch")
    ax.legend()

    # Subplot 4: beta (train only — same schedule for both)
    ax = axes[4]
    ax.plot(train_history["beta"], label="beta", color="orange")
    ax.set_title("Beta Schedule")
    ax.set_xlabel("Epoch")
    ax.legend()

    ax = axes[5]
    ax.plot(val_history["recon_metric"], label="val")
    ax.set_title("Recon Metric")
    ax.set_xlabel("Epoch")
    ax.legend()

    fig.tight_layout()
    fig.savefig(str(curves_path), dpi=120)
    plt.close(fig)
    logger.info("[Class %d/%s] Curves saved to %s", class_id, class_name, curves_path)

    # ------------------------------------------------------------------
    # 9. Return metrics
    # ------------------------------------------------------------------
    return {
        "class_id": class_id,
        "class_name": class_name,
        "latent_dim": latent_dim,
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "best_val_loss": best_val_loss,
        "epochs_trained": epoch_count,
        "final_kl": final_kl,
        "best_monitor_value": best_monitor_value,
        "best_epoch": best_epoch,
        "early_stop_metric": early_stop_metric,
        "train_history": train_history,
        "val_history": val_history,
        "checkpoint_path": str(checkpoint_path),
    }
