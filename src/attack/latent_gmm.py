from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

os.environ.setdefault("OMP_NUM_THREADS", "1")

from sklearn.mixture import BayesianGaussianMixture

from attack.latent_infra import AttackRouter, build_per_class_dataset, load_split
from vae.config import CLASSES


@dataclass
class LatentGMMPrior:
    class_id: int
    class_name: str
    split_name: str
    n_components: int
    latent_dim: int
    n_fit_samples: int
    checkpoint_sha: str
    gmm: BayesianGaussianMixture

    def sample(
        self,
        n_samples: int,
        *,
        seed: int,
        device: str,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        samples = sample_bayesian_gmm(self.gmm, n_samples=n_samples, seed=seed)
        return torch.as_tensor(samples, dtype=dtype, device=device)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_cache_dir() -> Path:
    return _repo_root() / "outputs" / "latent_gmm_priors"


def _checkpoint_sha(router: AttackRouter, class_name: str) -> str:
    try:
        return str(router.manifest["checkpoints"][class_name]["sha256"])
    except KeyError:
        return "unknown"


def _cache_path(
    *,
    cache_dir: Path,
    class_name: str,
    split_name: str,
    n_components: int,
    max_fit_samples: int | None,
    checkpoint_sha: str,
) -> Path:
    sample_tag = "all" if max_fit_samples is None else str(int(max_fit_samples))
    sha_tag = checkpoint_sha[:12] if checkpoint_sha != "unknown" else "unknown"
    name = (
        f"latent_gmm_{class_name}_{split_name}_k{int(n_components)}_"
        f"n{sample_tag}_{sha_tag}.pkl"
    )
    return cache_dir / name


def _encode_sampled_mu(
    *,
    model: torch.nn.Module,
    x_scaled: torch.Tensor,
    device: str,
    max_fit_samples: int | None,
    seed: int,
    batch_size: int,
) -> np.ndarray:
    n_total = int(x_scaled.shape[0])
    n_use = n_total if max_fit_samples is None else min(n_total, int(max_fit_samples))

    if n_use < n_total:
        rng = np.random.default_rng(seed)
        indices = np.sort(rng.choice(n_total, size=n_use, replace=False))
    else:
        indices = np.arange(n_total)

    model = model.to(device)
    model.eval()
    mu_batches: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, n_use, batch_size):
            batch_idx = indices[start : start + batch_size]
            x_batch = x_scaled[batch_idx].to(device=device, dtype=torch.float32)
            mu, _ = model.encode(x_batch)
            mu_batches.append(mu.detach().cpu().numpy().astype(np.float32))

    if not mu_batches:
        raise RuntimeError("No latent samples were encoded for GMM fitting")
    return np.concatenate(mu_batches, axis=0)


def fit_or_load_latent_gmm(
    *,
    router: AttackRouter,
    class_id: int,
    split_name: str = "val",
    n_components: int = 5,
    max_fit_samples: int | None = 50000,
    cache_dir: Path | None = None,
    seed: int = 42,
    device: str = "cpu",
    batch_size: int = 2048,
    force_refit: bool = False,
    max_iter: int = 200,
) -> LatentGMMPrior:
    class_name = CLASSES[int(class_id)]
    checkpoint_sha = _checkpoint_sha(router, class_name)
    cache_root = cache_dir or _default_cache_dir()
    cache_root.mkdir(parents=True, exist_ok=True)
    path = _cache_path(
        cache_dir=cache_root,
        class_name=class_name,
        split_name=split_name,
        n_components=n_components,
        max_fit_samples=max_fit_samples,
        checkpoint_sha=checkpoint_sha,
    )

    if path.exists() and not force_refit:
        with open(path, "rb") as f:
            prior = pickle.load(f)
        if isinstance(prior, LatentGMMPrior):
            return prior

    split = load_split(split_name, repo_root=router.repo_root)
    dataset = build_per_class_dataset(
        int(class_id),
        X_val=split["X"],
        y_val_8=split["y_8"],
        scaler=split["scaler"],
        partition=split["partition"],
    )
    vae = router.get_vae(int(class_id))
    z_mu = _encode_sampled_mu(
        model=vae,
        x_scaled=dataset.x_scaled,
        device=device,
        max_fit_samples=max_fit_samples,
        seed=int(seed) + int(class_id),
        batch_size=int(batch_size),
    )

    if z_mu.ndim != 2:
        raise ValueError(f"Expected encoded latents to be 2D, got shape {z_mu.shape}")
    if z_mu.shape[0] < 2:
        raise ValueError(f"Need at least 2 latent samples for class {class_name}")

    effective_components = min(max(1, int(n_components)), int(z_mu.shape[0]))
    gmm = BayesianGaussianMixture(
        n_components=effective_components,
        covariance_type="full",
        random_state=int(seed) + int(class_id),
        max_iter=int(max_iter),
        weight_concentration_prior_type="dirichlet_process",
        init_params="kmeans",
    )
    gmm.fit(z_mu)

    prior = LatentGMMPrior(
        class_id=int(class_id),
        class_name=class_name,
        split_name=str(split_name),
        n_components=effective_components,
        latent_dim=int(z_mu.shape[1]),
        n_fit_samples=int(z_mu.shape[0]),
        checkpoint_sha=checkpoint_sha,
        gmm=gmm,
    )
    with open(path, "wb") as f:
        pickle.dump(prior, f)
    return prior


def _component_covariance(gmm: BayesianGaussianMixture, component_id: int) -> np.ndarray:
    covariance_type = str(gmm.covariance_type)
    covariances: Any = gmm.covariances_
    latent_dim = int(gmm.means_.shape[1])

    if covariance_type == "full":
        return np.asarray(covariances[component_id], dtype=np.float64)
    if covariance_type == "tied":
        return np.asarray(covariances, dtype=np.float64)
    if covariance_type == "diag":
        return np.diag(np.asarray(covariances[component_id], dtype=np.float64))
    if covariance_type == "spherical":
        return np.eye(latent_dim, dtype=np.float64) * float(covariances[component_id])
    raise ValueError(f"Unsupported covariance_type={covariance_type!r}")


def sample_bayesian_gmm(
    gmm: BayesianGaussianMixture,
    *,
    n_samples: int,
    seed: int,
) -> np.ndarray:
    weights = np.asarray(gmm.weights_, dtype=np.float64)
    weights = np.maximum(weights, 0.0)
    weights = weights / weights.sum()

    means = np.asarray(gmm.means_, dtype=np.float64)
    latent_dim = int(means.shape[1])
    rng = np.random.default_rng(int(seed))
    component_ids = rng.choice(len(weights), size=int(n_samples), replace=True, p=weights)
    samples = np.zeros((int(n_samples), latent_dim), dtype=np.float32)

    eye = np.eye(latent_dim, dtype=np.float64)
    for component_id in np.unique(component_ids):
        mask = component_ids == component_id
        n_component = int(mask.sum())
        cov = _component_covariance(gmm, int(component_id))
        cov = cov + 1e-6 * eye
        samples[mask] = rng.multivariate_normal(
            mean=means[int(component_id)],
            cov=cov,
            size=n_component,
            check_valid="ignore",
        ).astype(np.float32)

    return samples
