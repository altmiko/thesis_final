from __future__ import annotations

import json
from typing import Mapping

import torch

from attack.latent_gmm import LatentGMMPrior
from vae.config import CLASS_TO_ID, CLASSES, ID_TO_CLASS

DEFAULT_NUM_RESTARTS = 5
DEFAULT_RESTART_STRATEGY = "encoded+jitter+gmm"
DEFAULT_GMM_SPLIT = "val"
DEFAULT_GMM_COMPONENTS = 5
DEFAULT_GMM_FIT_MAX_SAMPLES = 50000

DEFAULT_EPSILON_BY_CLASS_NAME: dict[str, float] = {
    "Benign": 0.3,
    "BruteForce": 0.3,
    "DDoS": 0.8,
    "DoS": 0.8,
    "Mirai": 0.8,
    "Recon": 0.5,
    "Spoofing": 0.5,
    "Web": 1.0,
}
DEFAULT_EPSILON_BY_CLASS_SPEC = ",".join(
    f"{name}={value:g}" for name, value in DEFAULT_EPSILON_BY_CLASS_NAME.items()
)


def strategy_parts(restart_strategy: str) -> list[str]:
    return [
        part.strip().lower()
        for part in str(restart_strategy).replace(",", "+").split("+")
        if part.strip()
    ]


def strategy_uses_gmm(restart_strategy: str) -> bool:
    return "gmm" in set(strategy_parts(restart_strategy))


def parse_class_float_map(
    value: str | None,
    *,
    default_by_class_name: Mapping[str, float],
) -> dict[int, float]:
    """Parse class keyed float overrides.

    Accepted forms:
      - JSON object: {"Web": 1.0, "3": 0.8}
      - comma list: Web=1.0,DoS=0.8,3=0.8
      - single float: applied to all classes
    """
    if value is None or not str(value).strip():
        raw_items: Mapping[str, float] = default_by_class_name
    else:
        text = str(value).strip()
        try:
            scalar = float(text)
        except ValueError:
            scalar = None
        if scalar is not None:
            raw_items = {name: scalar for name in CLASSES}
        elif text.startswith("{"):
            parsed = json.loads(text)
            if not isinstance(parsed, dict):
                raise ValueError("Class float map JSON must be an object")
            raw_items = parsed
        else:
            parsed_items: dict[str, float] = {}
            for item in text.split(","):
                if not item.strip():
                    continue
                if "=" in item:
                    key, raw_val = item.split("=", 1)
                elif ":" in item:
                    key, raw_val = item.split(":", 1)
                else:
                    raise ValueError(
                        f"Expected class=value entry in {item!r}"
                    )
                parsed_items[key.strip()] = float(raw_val.strip())
            raw_items = parsed_items

    out: dict[int, float] = {}
    for raw_key, raw_value in raw_items.items():
        key = str(raw_key).strip()
        if key.isdigit():
            class_id = int(key)
            if class_id not in ID_TO_CLASS:
                raise ValueError(f"Unknown class id in float map: {class_id}")
        else:
            if key not in CLASS_TO_ID:
                raise ValueError(
                    f"Unknown class name in float map: {key!r}; "
                    f"expected one of {CLASSES}"
                )
            class_id = CLASS_TO_ID[key]
        out[class_id] = float(raw_value)
    return out


def class_float_value(
    class_id: int,
    values_by_class_id: Mapping[int, float],
    fallback: float,
) -> float:
    return float(values_by_class_id.get(int(class_id), fallback))


def build_latent_restart_initializers(
    *,
    vae: torch.nn.Module,
    x_batch: torch.Tensor,
    gmm_prior: LatentGMMPrior | None,
    epsilon: float,
    num_restarts: int,
    restart_strategy: str,
    seed: int,
    class_id: int,
    device: str,
    project_to_epsilon: bool = True,
) -> tuple[torch.Tensor, list[str]]:
    """Build a restart tensor shaped (R, batch, latent_dim)."""
    with torch.no_grad():
        z_orig, _ = vae.encode(x_batch.to(device=device, dtype=torch.float32))

    parts = strategy_parts(restart_strategy)
    part_set = set(parts)
    if "gmm" in part_set and gmm_prior is None:
        raise ValueError("restart_strategy includes gmm, but no gmm_prior was provided")

    z_starts = [z_orig.detach().clone()]
    labels = ["encoded"]

    restart_kinds: list[str] = []
    if "jitter" in part_set:
        restart_kinds.append("jitter")
    if "gmm" in part_set:
        restart_kinds.append("gmm")
    if not restart_kinds:
        restart_kinds.append("encoded")

    for restart_idx in range(1, max(1, int(num_restarts))):
        kind = restart_kinds[(restart_idx - 1) % len(restart_kinds)]
        if kind == "jitter":
            generator = torch.Generator(device=z_orig.device)
            generator.manual_seed(int(seed) + int(class_id) * 1009 + int(restart_idx))
            if float(epsilon) > 0.0:
                noise = torch.empty_like(z_orig).uniform_(
                    -float(epsilon),
                    float(epsilon),
                    generator=generator,
                )
                z_start = z_orig + noise
            else:
                z_start = z_orig.clone()
        elif kind == "gmm":
            assert gmm_prior is not None
            z_start = gmm_prior.sample(
                int(z_orig.shape[0]),
                seed=int(seed) + int(class_id) * 2003 + int(restart_idx),
                device=device,
                dtype=z_orig.dtype,
            )
        else:
            z_start = z_orig.clone()

        if project_to_epsilon and float(epsilon) > 0.0:
            z_start = torch.max(
                torch.min(z_start, z_orig + float(epsilon)),
                z_orig - float(epsilon),
            )
        z_starts.append(z_start.detach().clone())
        labels.append(kind)

    return torch.stack(z_starts, dim=0), labels
