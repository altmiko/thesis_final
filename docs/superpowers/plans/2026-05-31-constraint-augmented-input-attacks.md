# Constraint-Augmented Input-Space PGD/CW — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add input-space PGD/CW attacks that apply the VAE's exact constraint mechanics (structured-decoder G1–G8 + physics P2/P4/P5) directly to `x` with no latent encode/decode, plus a standalone runner across all 8-class neural models, so the manifold-vs-constraints ablation can be tabulated against the existing latent and plain-input attacks.

**Architecture:** A differentiable `VAEConstraintProjection` lifts `MixedInputBetaVAE._structure_continuous_raw` out of latent space and applies it (unscale → structure → rescale → mask → freeze-protocol) to a full 39-dim scaled vector each optimization step. Two attack functions mirror `input_baselines.py` but route every iterate through that projection. A standalone runner reuses the public `latent_infra` infrastructure to evaluate the full metric suite (ASR, ASR_valid, protocol/mask/raw-G1G8 validity, IDSR, L2) and emit the same CSV schema as the master harness.

**Tech Stack:** Python 3.10 (conda env `thesis`), PyTorch, NumPy, scikit-learn (`RobustScaler`), existing project modules under `src/`.

**Spec:** `docs/superpowers/specs/2026-05-31-constraint-augmented-input-attacks-design.md`

**Conventions:**
- Run everything with the project interpreter: `C:\Users\T2530985\.conda\envs\thesis\python.exe`.
- This repo has **no pytest dependency**. Tests are plain-`assert` functions with an `if __name__ == "__main__"` runner, executed with `python <testfile>`.
- All scripts assume CWD = repo root `D:/thesis_final` and insert `src/` on `sys.path` (mirroring existing runners).

---

## File Structure

| File | Responsibility | Action |
|---|---|---|
| `src/attack/constrained_input_baselines.py` | `VAEConstraintProjection` + `constrained_input_pgd_attack` + `constrained_input_cw_attack` | Create |
| `src/attack/run_constrained_input_baselines.py` | Standalone runner across 8-class models; full metric suite; CSV/MD/JSON output | Create |
| `tests/attack/test_constrained_input_baselines.py` | Unit + smoke tests for projection and attacks | Create |

No existing files are modified.

---

## Task 1: `VAEConstraintProjection`

**Files:**
- Create: `src/attack/constrained_input_baselines.py`
- Test: `tests/attack/test_constrained_input_baselines.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/attack/test_constrained_input_baselines.py`:

```python
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = str(_REPO_ROOT / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import pickle

from attack.constrained_input_baselines import VAEConstraintProjection
from attack.latent_infra import PerturbationMask
from attack.validator import validate_batch
from preprocessing.feature_groups import FEATURE_NAMES


def _load_scaler():
    with open(_REPO_ROOT / "data" / "processed" / "scaler.pkl", "rb") as f:
        return pickle.load(f)


def _one_real_scaled_sample(n: int = 8) -> np.ndarray:
    X = np.load(_REPO_ROOT / "data" / "processed" / "X_test.npy")
    return X[:n].astype(np.float32)


def test_hard_projection_satisfies_continuous_rules():
    scaler = _load_scaler()
    mask = PerturbationMask.from_preprocessing_artifacts()
    proj = VAEConstraintProjection(scaler, mask, enable_physics=True, device="cpu")

    x0 = torch.from_numpy(_one_real_scaled_sample(16))
    # Perturb the mutable aggregate features hard, then project.
    x_pert = x0 + torch.randn_like(x0) * 2.0
    x_proj = proj.project(x_pert, x0, mode="hard")

    x_raw = scaler.inverse_transform(x_proj.detach().numpy().astype(np.float64))
    vr = validate_batch(x_raw, FEATURE_NAMES)
    rates = vr.per_rule_violation_rate()
    # Structurally enforced rules must hold for all rows.
    for rule in ["R_min_leq_max", "R_avg_in_range", "R_var_eq_std_sq",
                 "R_pkts_positive", "R_pkts_integer"]:
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

    mask = PerturbationMask.from_preprocessing_artifacts()
    proj = VAEConstraintProjection(
        scaler, mask, enable_physics=True, structured_std_floor=0.0, device="cpu"
    )

    torch.manual_seed(0)
    cont_idx = partition["continuous_idx"]
    cont_raw = torch.randn(32, len(cont_idx)) * 5.0

    vae_struct = vae._structure_continuous_raw(cont_raw)  # (32, n_continuous)

    # Embed the same continuous raw values into a full-39 raw vector and run the
    # standalone structuring directly (bypassing scaler round-trip for parity).
    full_raw = torch.zeros(32, 39)
    full_raw[:, cont_idx] = cont_raw
    proj_struct_full = proj._structure_raw(full_raw, mode="soft")
    proj_struct = proj_struct_full[:, cont_idx]

    assert torch.allclose(vae_struct, proj_struct, atol=1e-4), (
        (vae_struct - proj_struct).abs().max().item()
    )


def test_soft_projection_is_differentiable():
    scaler = _load_scaler()
    mask = PerturbationMask.from_preprocessing_artifacts()
    proj = VAEConstraintProjection(scaler, mask, enable_physics=True, device="cpu")

    x0 = torch.from_numpy(_one_real_scaled_sample(4))
    x = (x0 + 0.1).clone().requires_grad_(True)
    out = proj.project(x, x0, mode="soft")
    out.sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


if __name__ == "__main__":
    test_hard_projection_satisfies_continuous_rules()
    test_parity_with_vae_structured_decoder()
    test_soft_projection_is_differentiable()
    print("Task 1 tests passed.")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `C:\Users\T2530985\.conda\envs\thesis\python.exe tests/attack/test_constrained_input_baselines.py`
Expected: `ModuleNotFoundError` / `ImportError: cannot import name 'VAEConstraintProjection'`.

- [ ] **Step 3: Implement the projection**

Create `src/attack/constrained_input_baselines.py` with this content:

```python
from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from attack.latent_infra import PerturbationMask, reimpose_protocol_features
from attack.latent_pgd import classifier_logits
from preprocessing.feature_groups import FEATURE_NAMES

# Raw-space indices for the 39-feature Modified Schema A. Mirrors
# vae.schema.get_partition()["continuous_idx"] and the binary columns.
_CONTINUOUS_IDX = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                   30, 31, 32, 33, 34, 35, 36, 37, 38]
_BINARY_IDX = [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]


class VAEConstraintProjection:
    """The VAE decoder's constraint mechanic, lifted out of latent space.

    Applies MixedInputBetaVAE._structure_continuous_raw's by-construction G1-G8
    (+ physics P2/P4/P5) enforcement to a full 39-dim *scaled* feature vector,
    so input-space PGD/CW can use the same mechanics the latent attack relies on.
    """

    def __init__(
        self,
        scaler: Any,
        mask: PerturbationMask,
        *,
        enable_physics: bool = True,
        structured_std_floor: float = 0.0,
        device: str = "cpu",
    ) -> None:
        center = np.asarray(scaler.center_, dtype=np.float32)
        scale = np.asarray(scaler.scale_, dtype=np.float32)
        if center.shape[0] != 39 or scale.shape[0] != 39:
            raise ValueError(
                f"Expected a 39-feature RobustScaler, got center={center.shape}, "
                f"scale={scale.shape}"
            )
        self.mask = mask
        self.enable_physics = bool(enable_physics)
        self.structured_std_floor = float(structured_std_floor)
        self.device = device

        self.center = torch.tensor(center, device=device)
        self.scale = torch.tensor(scale, device=device)
        self.continuous_idx = torch.tensor(_CONTINUOUS_IDX, dtype=torch.long, device=device)
        self.binary_idx = torch.tensor(_BINARY_IDX, dtype=torch.long, device=device)

        self.ttl_idx = FEATURE_NAMES.index("Time_To_Live")
        self.tot_sum_idx = FEATURE_NAMES.index("Tot sum")
        self.min_idx = FEATURE_NAMES.index("Min")
        self.max_idx = FEATURE_NAMES.index("Max")
        self.avg_idx = FEATURE_NAMES.index("AVG")
        self.std_idx = FEATURE_NAMES.index("Std")
        self.tot_size_idx = FEATURE_NAMES.index("Tot size")
        self.number_idx = FEATURE_NAMES.index("Number")
        self.variance_idx = FEATURE_NAMES.index("Variance")

    def to(self, device: str) -> "VAEConstraintProjection":
        self.device = device
        self.center = self.center.to(device)
        self.scale = self.scale.to(device)
        self.continuous_idx = self.continuous_idx.to(device)
        self.binary_idx = self.binary_idx.to(device)
        return self

    def _unscale(self, x_scaled: torch.Tensor) -> torch.Tensor:
        return x_scaled * self.scale.unsqueeze(0) + self.center.unsqueeze(0)

    def _rescale(self, x_raw: torch.Tensor) -> torch.Tensor:
        return (x_raw - self.center.unsqueeze(0)) / self.scale.unsqueeze(0)

    def _structure_raw(self, x_raw: torch.Tensor, mode: str) -> torch.Tensor:
        structured = x_raw.clone()
        # G1: non-negativity for every continuous feature ("full" mode).
        structured[:, self.continuous_idx] = x_raw[:, self.continuous_idx].clamp_min(0.0)

        # G5: Min <= AVG <= Max via stacked softplus gaps.
        min_base = F.softplus(x_raw[:, self.min_idx])
        avg_gap = F.softplus(x_raw[:, self.avg_idx] - x_raw[:, self.min_idx])
        max_gap = F.softplus(x_raw[:, self.max_idx] - x_raw[:, self.avg_idx])
        avg_val = min_base + avg_gap
        max_val = avg_val + max_gap

        # G8: Number positive integer via straight-through round.
        number_pos = 1.0 + F.softplus(x_raw[:, self.number_idx])
        number_val = number_pos + (torch.round(number_pos) - number_pos).detach()

        # G7: TTL in [0, 255].
        ttl_val = torch.sigmoid(x_raw[:, self.ttl_idx] / 32.0) * 255.0

        std_candidate = self.structured_std_floor + F.softplus(x_raw[:, self.std_idx])
        if self.enable_physics:
            std_cap = 0.5 * (max_val - min_base)  # P4
            std_val = torch.minimum(std_candidate, std_cap)
        else:
            std_val = std_candidate

        if self.enable_physics:
            # P5: singleton flows have no within-flow size variance.
            singleton = number_val <= 1.0
            singleton_size = F.softplus(x_raw[:, self.avg_idx])
            min_base = torch.where(singleton, singleton_size, min_base)
            avg_val = torch.where(singleton, singleton_size, avg_val)
            max_val = torch.where(singleton, singleton_size, max_val)
            std_val = torch.where(singleton, torch.zeros_like(std_val), std_val)

        variance_val = std_val.square()  # G6

        if self.enable_physics:
            tot_size_val = avg_val            # P3-duplicate semantics: Tot size == AVG
            tot_sum_val = number_val * avg_val  # P2
        else:
            tot_size_val = structured[:, self.tot_size_idx]
            tot_sum_val = structured[:, self.tot_sum_idx]

        structured[:, self.ttl_idx] = ttl_val
        structured[:, self.tot_sum_idx] = tot_sum_val
        structured[:, self.min_idx] = min_base
        structured[:, self.avg_idx] = avg_val
        structured[:, self.max_idx] = max_val
        structured[:, self.std_idx] = std_val
        structured[:, self.tot_size_idx] = tot_size_val
        structured[:, self.variance_idx] = variance_val
        structured[:, self.number_idx] = number_val

        # G3: binary columns. Soft keeps gradients; hard rounds to {0,1}.
        if mode == "hard":
            structured[:, self.binary_idx] = torch.clamp(
                torch.round(x_raw[:, self.binary_idx]), 0.0, 1.0
            )
        else:
            structured[:, self.binary_idx] = torch.clamp(
                x_raw[:, self.binary_idx], 0.0, 1.0
            )
        return structured

    def project(
        self,
        x_scaled: torch.Tensor,
        x_original_scaled: torch.Tensor,
        mode: str = "soft",
    ) -> torch.Tensor:
        if mode not in {"soft", "hard"}:
            raise ValueError(f"mode must be 'soft' or 'hard', got {mode!r}")
        x_raw = self._unscale(x_scaled)
        structured_raw = self._structure_raw(x_raw, mode)
        x_struct_scaled = self._rescale(structured_raw)
        x_masked = self.mask.apply(x_struct_scaled, x_original_scaled)
        return reimpose_protocol_features(x_masked, x_original_scaled)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `C:\Users\T2530985\.conda\envs\thesis\python.exe tests/attack/test_constrained_input_baselines.py`
Expected: `Task 1 tests passed.`

- [ ] **Step 5: Commit**

```bash
git add src/attack/constrained_input_baselines.py tests/attack/test_constrained_input_baselines.py
git commit -m "feat(attack): add VAEConstraintProjection (lifted decoder mechanic)"
```

---

## Task 2: The two constrained attacks

**Files:**
- Modify: `src/attack/constrained_input_baselines.py` (append two functions)
- Test: `tests/attack/test_constrained_input_baselines.py` (append smoke tests)

- [ ] **Step 1: Write the failing smoke tests**

Append to `tests/attack/test_constrained_input_baselines.py` (before the `if __name__` block):

```python
def _tiny_classifier() -> torch.nn.Module:
    torch.manual_seed(0)
    model = torch.nn.Linear(39, 8)
    model.eval()
    return model


def test_constrained_pgd_smoke_and_validity_gain():
    from attack.constrained_input_baselines import constrained_input_pgd_attack
    from attack.input_baselines import input_pgd_attack

    scaler = _load_scaler()
    mask = PerturbationMask.from_preprocessing_artifacts()
    proj = VAEConstraintProjection(scaler, mask, enable_physics=True, device="cpu")
    clf = _tiny_classifier()

    x0 = torch.from_numpy(_one_real_scaled_sample(16))
    y = torch.zeros(16, dtype=torch.long)

    x_c, meta_c = constrained_input_pgd_attack(
        classifier=clf, projection=proj, x_original=x0, y_true=y,
        epsilon=0.5, alpha=0.05, num_steps=5, random_start=True, device="cpu",
    )
    x_p, _ = input_pgd_attack(
        classifier=clf, x_original=x0, y_true=y,
        epsilon=0.5, alpha=0.05, num_steps=5, random_start=True, device="cpu",
    )
    assert x_c.shape == x0.shape
    assert meta_c["constraint_projection"] is True

    raw_c = scaler.inverse_transform(x_c.numpy().astype(np.float64))
    raw_p = scaler.inverse_transform(x_p.numpy().astype(np.float64))
    vr_c = validate_batch(raw_c, FEATURE_NAMES).validity_rate
    vr_p = validate_batch(raw_p, FEATURE_NAMES).validity_rate
    assert vr_c >= vr_p


def test_constrained_cw_smoke():
    from attack.constrained_input_baselines import constrained_input_cw_attack

    scaler = _load_scaler()
    mask = PerturbationMask.from_preprocessing_artifacts()
    proj = VAEConstraintProjection(scaler, mask, enable_physics=True, device="cpu")
    clf = _tiny_classifier()

    x0 = torch.from_numpy(_one_real_scaled_sample(16))
    y = torch.zeros(16, dtype=torch.long)

    x_adv, meta = constrained_input_cw_attack(
        classifier=clf, projection=proj, x_original=x0, y_true=y,
        lambda_conf=1.0, kappa=0.0, num_iterations=5, learning_rate=0.05,
        convergence_threshold=1e-5, device="cpu",
    )
    assert x_adv.shape == x0.shape
    assert meta["constraint_projection"] is True
    assert "best_success_mask" in meta
```

Add these two calls inside the `if __name__ == "__main__"` block (before the print):

```python
    test_constrained_pgd_smoke_and_validity_gain()
    test_constrained_cw_smoke()
```

- [ ] **Step 2: Run to verify failure**

Run: `C:\Users\T2530985\.conda\envs\thesis\python.exe tests/attack/test_constrained_input_baselines.py`
Expected: `ImportError: cannot import name 'constrained_input_pgd_attack'`.

- [ ] **Step 3: Implement the two attacks**

Append to `src/attack/constrained_input_baselines.py`:

```python
def constrained_input_pgd_attack(
    *,
    classifier: torch.nn.Module,
    projection: VAEConstraintProjection,
    x_original: torch.Tensor,
    y_true: torch.Tensor,
    epsilon: float,
    alpha: float,
    num_steps: int,
    random_start: bool,
    device: str,
) -> tuple[torch.Tensor, dict[str, Any]]:
    x_original = x_original.to(device=device, dtype=torch.float32)
    y_true = y_true.to(device=device, dtype=torch.long)
    classifier = classifier.to(device)
    classifier.eval()
    projection = projection.to(device)

    if float(epsilon) <= 0.0:
        x_pass = projection.project(x_original, x_original, mode="hard")
        return x_pass.detach(), {
            "loss_final": 0.0,
            "num_steps": int(num_steps),
            "random_start": bool(random_start),
            "epsilon": float(epsilon),
            "alpha": float(alpha),
            "zero_budget_passthrough": True,
            "constraint_projection": True,
            "x_orig": x_original.detach(),
            "x_adv": x_pass.detach(),
        }

    x_adv = x_original.clone()
    if random_start:
        x_adv = x_adv + torch.empty_like(x_adv).uniform_(-epsilon, epsilon)
        x_adv = torch.max(torch.min(x_adv, x_original + epsilon), x_original - epsilon)

    loss_value = 0.0
    for _ in range(int(num_steps)):
        x_adv = x_adv.detach().requires_grad_(True)
        x_proj = projection.project(x_adv, x_original, mode="soft")
        logits = classifier_logits(classifier, x_proj, device=device)
        loss = F.cross_entropy(logits, y_true)
        grad = torch.autograd.grad(loss, x_adv, retain_graph=False, create_graph=False)[0]
        with torch.no_grad():
            x_adv = x_adv + alpha * grad.sign()
            x_adv = torch.max(torch.min(x_adv, x_original + epsilon), x_original - epsilon)
            loss_value = float(loss.item())

    x_adv_final = projection.project(x_adv.detach(), x_original, mode="hard")
    return x_adv_final.detach(), {
        "loss_final": loss_value,
        "num_steps": int(num_steps),
        "random_start": bool(random_start),
        "epsilon": float(epsilon),
        "alpha": float(alpha),
        "zero_budget_passthrough": False,
        "constraint_projection": True,
        "x_orig": x_original.detach(),
        "x_adv": x_adv_final.detach(),
    }


def constrained_input_cw_attack(
    *,
    classifier: torch.nn.Module,
    projection: VAEConstraintProjection,
    x_original: torch.Tensor,
    y_true: torch.Tensor,
    lambda_conf: float,
    kappa: float,
    num_iterations: int,
    learning_rate: float,
    convergence_threshold: float,
    device: str,
) -> tuple[torch.Tensor, dict[str, Any]]:
    x_original = x_original.to(device=device, dtype=torch.float32)
    y_true = y_true.to(device=device, dtype=torch.long)
    classifier = classifier.to(device)
    classifier.eval()
    projection = projection.to(device)

    if float(lambda_conf) <= 0.0:
        x_pass = projection.project(x_original, x_original, mode="hard")
        return x_pass.detach(), {
            "loss_final": 0.0,
            "num_iterations": int(num_iterations),
            "learning_rate": float(learning_rate),
            "lambda_conf": float(lambda_conf),
            "kappa": float(kappa),
            "convergence_threshold": float(convergence_threshold),
            "zero_budget_passthrough": True,
            "constraint_projection": True,
            "iterations_run": 0,
            "converged_early": False,
            "x_orig": x_original.detach(),
            "x_adv": x_pass.detach(),
            "best_success_mask": torch.zeros(x_original.shape[0], dtype=torch.bool, device=device),
        }

    delta = torch.zeros_like(x_original, requires_grad=True)
    optimizer = torch.optim.Adam([delta], lr=learning_rate)

    best_delta = torch.zeros_like(x_original)
    best_delta_l2 = torch.full((x_original.shape[0],), float("inf"), device=device)
    best_success_mask = torch.zeros(x_original.shape[0], dtype=torch.bool, device=device)

    prev_delta = delta.detach().clone()
    converged_early = False
    loss_value = 0.0
    iterations_run = 0

    for iteration in range(int(num_iterations)):
        optimizer.zero_grad(set_to_none=True)
        x_proj = projection.project(x_original + delta, x_original, mode="soft")
        logits = classifier_logits(classifier, x_proj, device=device)
        true_logits = logits.gather(1, y_true.unsqueeze(1)).squeeze(1)
        masked_logits = logits.clone()
        masked_logits.scatter_(1, y_true.unsqueeze(1), float("-inf"))
        other_logits = masked_logits.max(dim=1).values
        margin = true_logits - other_logits

        conf_term = torch.clamp(margin + float(kappa), min=0.0)
        delta_l2_sq = delta.pow(2).sum(dim=1)
        loss = (float(lambda_conf) * conf_term + delta_l2_sq).mean()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            x_post = projection.project(x_original + delta.detach(), x_original, mode="hard")
            logits_post = classifier_logits(classifier, x_post, device=device)
            true_post = logits_post.gather(1, y_true.unsqueeze(1)).squeeze(1)
            masked_post = logits_post.clone()
            masked_post.scatter_(1, y_true.unsqueeze(1), float("-inf"))
            other_post = masked_post.max(dim=1).values
            success_mask = (true_post - other_post) <= 0.0

            delta_l2 = torch.linalg.norm(delta.detach(), dim=1)
            improved = success_mask & (delta_l2 < best_delta_l2)
            if improved.any():
                best_delta[improved] = delta.detach()[improved]
                best_delta_l2[improved] = delta_l2[improved]
                best_success_mask[improved] = True

            delta_shift = torch.linalg.norm(delta.detach() - prev_delta, dim=1)
            prev_delta = delta.detach().clone()
            loss_value = float(loss.item())
            iterations_run = iteration + 1
            if torch.max(delta_shift).item() < float(convergence_threshold):
                converged_early = True
                break

    with torch.no_grad():
        chosen_delta = torch.where(best_success_mask.unsqueeze(1), best_delta, delta.detach())
        x_adv_final = projection.project(x_original + chosen_delta, x_original, mode="hard")

    return x_adv_final.detach(), {
        "loss_final": loss_value,
        "num_iterations": int(num_iterations),
        "learning_rate": float(learning_rate),
        "lambda_conf": float(lambda_conf),
        "kappa": float(kappa),
        "convergence_threshold": float(convergence_threshold),
        "zero_budget_passthrough": False,
        "constraint_projection": True,
        "iterations_run": int(iterations_run),
        "converged_early": bool(converged_early),
        "x_orig": x_original.detach(),
        "x_adv": x_adv_final.detach(),
        "best_success_mask": best_success_mask.detach(),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `C:\Users\T2530985\.conda\envs\thesis\python.exe tests/attack/test_constrained_input_baselines.py`
Expected: `Task 1 tests passed.` (the print line still runs after all five tests).

- [ ] **Step 5: Commit**

```bash
git add src/attack/constrained_input_baselines.py tests/attack/test_constrained_input_baselines.py
git commit -m "feat(attack): add constrained input PGD/CW with in-loop projection"
```

---

## Task 3: Standalone runner

**Files:**
- Create: `src/attack/run_constrained_input_baselines.py`

- [ ] **Step 1: Implement the runner**

Create `src/attack/run_constrained_input_baselines.py`:

```python
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

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
from attack.adversarial_attacks import load_model  # noqa: E402
from attack.latent_infra import (  # noqa: E402
    AttackRouter,
    AttackRunLogger,
    MahalanobisOutlierDetector,
    PerturbationMask,
    ProtocolValidator,
    build_per_class_dataset,
    encode_dataset_mu,
    load_split,
    phase0_config_snapshot,
    predict_labels,
    set_global_seed,
)
from attack.latent_pgd import classifier_logits  # noqa: E402
from attack.validator import validate_batch  # noqa: E402
from preprocessing.feature_groups import FEATURE_NAMES  # noqa: E402
from vae.config import CLASS_TO_ID, CLASSES  # noqa: E402

DEFAULT_VAE_RUN_TAG = "gaussian_anticollapse_beta05_freebits01_20260529_173512"
SAMPLES_PER_SOURCE_CLASS = 100
SELECTION_BATCH_SIZE = 8192
SOURCE_CLASSES = [name for name in CLASSES if name != "Benign"]
MODEL_SPECS = [
    {"tag": "mlp", "label": "MLP", "checkpoint": "mlp_8class.pt"},
    {"tag": "cnn", "label": "CNN", "checkpoint": "cnn_8class.pt"},
    {"tag": "lstm", "label": "LSTM", "checkpoint": "lstm_8class.pt"},
    {"tag": "serial", "label": "CNN-LSTM", "checkpoint": "serial_8class.pt"},
    {"tag": "dualpath", "label": "DualPath", "checkpoint": "dualpath_8class.pt"},
]
ATTACK_ORDER = ["cinput-pgd", "cinput-cw"]
SUMMARY_COLUMNS = [
    "model", "attack", "n", "asr_overall", "asr_valid_only", "asr_invalid_only",
    "protocol_validity_rate", "mask_compliance_rate", "joint_validity_rate",
    "idsr", "mean_l2_input", "mean_l2_latent", "raw_g1g8_validity_rate",
    "successful_joint_valid_count",
]


def _load_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _resolve_vae_paths(args: argparse.Namespace) -> tuple[str, Path, Path]:
    run_tag = str(args.vae_run_tag).strip()
    manifest_path = (
        Path(args.vae_manifest)
        if args.vae_manifest
        else _REPO_ROOT / "results" / "vae" / run_tag / "vae_run_manifest.json"
    ).resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(f"VAE manifest not found: {manifest_path}")
    diagnostics_dir = (
        Path(args.vae_diagnostics_dir).resolve()
        if args.vae_diagnostics_dir
        else manifest_path.parent
    )
    if not run_tag:
        run_tag = manifest_path.parent.name
    return run_tag, manifest_path, diagnostics_dir


def _load_collapsed_dims_for_run(diagnostics_dir: Path, manifest: dict[str, Any]) -> dict[int, list[int]]:
    collapsed: dict[int, list[int]] = {}
    manifest_diag = manifest.get("diagnostics", {})
    for class_name in CLASSES:
        class_id = CLASS_TO_ID[class_name]
        diag_path = diagnostics_dir / f"diagnostics_{class_name}.json"
        md = manifest_diag.get(class_name, {})
        if not diag_path.exists() and md.get("path"):
            diag_path = Path(str(md["path"]))
        if not diag_path.exists():
            raise FileNotFoundError(f"Diagnostics not found for {class_name}: {diag_path}")
        diag = _load_json(diag_path)
        collapsed[class_id] = list(diag["posterior_collapse"]["collapsed_dim_indices"])
    return collapsed


def _fit_detector(router: AttackRouter, device: str, collapsed_by_class: dict[int, list[int]]) -> MahalanobisOutlierDetector:
    split_val = load_split("val")
    detector = MahalanobisOutlierDetector(latent_dim=16)
    for class_id, _name in enumerate(CLASSES):
        ds = build_per_class_dataset(
            class_id,
            X_val=split_val["X"], y_val_8=split_val["y_8"],
            scaler=split_val["scaler"], partition=split_val["partition"],
        )
        vae = router.get_vae(class_id)
        z_mu = encode_dataset_mu(vae, ds, device=device)
        detector.fit(class_id, z_mu.cpu(), collapsed_dims=collapsed_by_class[class_id])
    return detector


def _select_correct_source_samples(*, x_test, y_test, classifier, device, class_id, max_samples, selection_batch_size=SELECTION_BATCH_SIZE):
    class_indices = np.where(y_test == class_id)[0]
    if len(class_indices) == 0:
        raise RuntimeError(f"No test samples found for class_id={class_id}")
    pred_batches = []
    for start in range(0, len(class_indices), selection_batch_size):
        batch_indices = class_indices[start:start + selection_batch_size]
        preds = predict_labels(classifier, torch.from_numpy(x_test[batch_indices].astype(np.float32)), device=device).numpy()
        pred_batches.append(preds)
    preds = np.concatenate(pred_batches, axis=0)
    keep = class_indices[preds == class_id]
    return keep[:max_samples]


def _conditional_rate(numerator_mask: torch.Tensor, denom_mask: torch.Tensor) -> float:
    denom = int(denom_mask.sum().item())
    if denom == 0:
        return 0.0
    return float((numerator_mask & denom_mask).float().sum().item() / denom)


def _masked_rate(mask: torch.Tensor) -> float:
    if mask.numel() == 0:
        return 0.0
    return float(mask.float().mean().item())


def _raw_g1g8_validity(x_adv: torch.Tensor, scaler: Any) -> torch.Tensor:
    x_raw = scaler.inverse_transform(x_adv.detach().cpu().numpy().astype(np.float64))
    return torch.from_numpy(validate_batch(x_raw, FEATURE_NAMES).overall_valid.astype(np.bool_))


def _evaluate(*, attack_name, class_id, x_batch, y_batch, vae, classifier, projection,
              mask, protocol_validator, detector, scaler, device, args) -> dict[str, Any]:
    if attack_name == "cinput-pgd":
        x_adv, _meta = constrained_input_pgd_attack(
            classifier=classifier, projection=projection, x_original=x_batch, y_true=y_batch,
            epsilon=args.input_epsilon, alpha=args.input_alpha, num_steps=args.num_steps,
            random_start=args.random_start, device=device,
        )
    elif attack_name == "cinput-cw":
        x_adv, _meta = constrained_input_cw_attack(
            classifier=classifier, projection=projection, x_original=x_batch, y_true=y_batch,
            lambda_conf=args.lambda_conf, kappa=args.kappa, num_iterations=args.num_iterations,
            learning_rate=args.learning_rate, convergence_threshold=args.convergence_threshold,
            device=device,
        )
    else:
        raise KeyError(f"Unsupported attack: {attack_name}")

    with torch.no_grad():
        logits_after = classifier_logits(classifier, x_adv.to(device), device=device).cpu()
        pred_after = torch.argmax(logits_after, dim=1)
        success_mask = pred_after != y_batch.cpu()
        protocol_valid = protocol_validator.validate(x_adv, already_scaled=True).cpu()
        mask_compliance = mask.verify(x_adv.cpu(), x_batch.cpu())["all_compliant"].cpu()
        raw_valid = _raw_g1g8_validity(x_adv.cpu(), scaler)
        joint_valid = protocol_valid & mask_compliance & raw_valid
        invalid_mask = ~joint_valid
        z_reencoded, _ = vae.encode(x_adv.to(device))
        outlier_mask = detector.outlier_mask(class_id, z_reencoded.cpu()).cpu()
        input_l2 = torch.linalg.norm(x_adv.cpu() - x_batch.cpu(), dim=1)

    successful_joint = int((success_mask & joint_valid).sum().item())
    return {
        "class_name": CLASSES[class_id],
        "pred_after": pred_after.numpy().tolist(),
        "success_mask": success_mask.numpy().tolist(),
        "protocol_valid_mask": protocol_valid.numpy().tolist(),
        "mask_compliance_mask": mask_compliance.numpy().tolist(),
        "raw_g1g8_valid_mask": raw_valid.numpy().tolist(),
        "joint_valid_mask": joint_valid.numpy().tolist(),
        "n": int(x_batch.shape[0]),
        "asr_overall": _masked_rate(success_mask),
        "asr_valid_only": _conditional_rate(success_mask, joint_valid),
        "asr_invalid_only": _conditional_rate(success_mask, invalid_mask),
        "protocol_validity_rate": _masked_rate(protocol_valid),
        "mask_compliance_rate": _masked_rate(mask_compliance),
        "raw_g1g8_validity_rate": _masked_rate(raw_valid),
        "joint_validity_rate": _masked_rate(joint_valid),
        "idsr": _masked_rate(~outlier_mask),
        "mean_l2_input": float(input_l2.mean().item()),
        "mean_l2_latent": None,
        "valid_count": int(joint_valid.sum().item()),
        "invalid_count": int(invalid_mask.sum().item()),
        "successful_total_count": int(success_mask.sum().item()),
        "successful_valid_count": successful_joint,
        "successful_invalid_count": int((success_mask & invalid_mask).sum().item()),
        "successful_joint_valid_count": successful_joint,
    }


def _aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total_n = sum(r["n"] for r in rows)
    total_valid = sum(r["valid_count"] for r in rows)
    total_invalid = sum(r["invalid_count"] for r in rows)
    return {
        "n": int(total_n),
        "asr_overall": float(sum(r["successful_total_count"] for r in rows) / total_n),
        "asr_valid_only": float(sum(r["successful_valid_count"] for r in rows) / total_valid) if total_valid > 0 else 0.0,
        "asr_invalid_only": float(sum(r["successful_invalid_count"] for r in rows) / total_invalid) if total_invalid > 0 else 0.0,
        "protocol_validity_rate": float(sum(r["protocol_validity_rate"] * r["n"] for r in rows) / total_n),
        "mask_compliance_rate": float(sum(r["mask_compliance_rate"] * r["n"] for r in rows) / total_n),
        "joint_validity_rate": float(sum(r["joint_validity_rate"] * r["n"] for r in rows) / total_n),
        "idsr": float(sum(r["idsr"] * r["n"] for r in rows) / total_n),
        "mean_l2_input": float(sum(r["mean_l2_input"] * r["n"] for r in rows) / total_n),
        "mean_l2_latent": None,
        "raw_g1g8_validity_rate": float(sum(r["raw_g1g8_validity_rate"] * r["n"] for r in rows) / total_n),
        "successful_joint_valid_count": sum(int(r["successful_joint_valid_count"]) for r in rows),
    }


def _fmt_pct(v: Any) -> str:
    return "NA" if v is None else f"{float(v) * 100.0:.2f}%"


def _fmt_float(v: Any) -> str:
    return "NA" if v is None else f"{float(v):.4f}"


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    def cell(value: Any) -> Any:
        return json.dumps(value) if isinstance(value, (list, dict)) else value
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: cell(row.get(k, "")) for k in columns})


def _write_summary_md(path: Path, rows: list[dict[str, Any]]) -> None:
    headers = ["Model", "Attack", "N", "ASR", "ASR Valid", "Protocol", "Mask",
               "Raw G1-G8", "Joint Valid", "IDSR", "L2 Input"]
    lines = [
        "# Constraint-Augmented Input Attack Summary",
        "",
        f"Sample setting: {SAMPLES_PER_SOURCE_CLASS} correctly classified test samples per non-benign source class.",
        "Constraint set: full + physics (G1-G8 + P2/P4/P5). Final sample validity is measured, not forced.",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join([
            str(row["model"]), str(row["attack"]), str(row["n"]),
            _fmt_pct(row["asr_overall"]), _fmt_pct(row["asr_valid_only"]),
            _fmt_pct(row["protocol_validity_rate"]), _fmt_pct(row["mask_compliance_rate"]),
            _fmt_pct(row["raw_g1g8_validity_rate"]), _fmt_pct(row["joint_validity_rate"]),
            _fmt_pct(row["idsr"]), _fmt_float(row["mean_l2_input"]),
        ]) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Constraint-augmented input PGD/CW across 8-class models.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--vae-run-tag", default=DEFAULT_VAE_RUN_TAG)
    parser.add_argument("--vae-manifest", default=None)
    parser.add_argument("--vae-diagnostics-dir", default=None)
    parser.add_argument("--models", default="all")
    parser.add_argument("--samples-per-class", type=int, default=SAMPLES_PER_SOURCE_CLASS)
    parser.add_argument("--selection-batch-size", type=int, default=SELECTION_BATCH_SIZE)
    parser.add_argument("--input-epsilon", type=float, default=0.5)
    parser.add_argument("--input-alpha", type=float, default=0.05)
    parser.add_argument("--num-steps", type=int, default=40)
    parser.add_argument("--random-start", action="store_true", default=True)
    parser.add_argument("--no-random-start", dest="random_start", action="store_false")
    parser.add_argument("--lambda-conf", type=float, default=1.0)
    parser.add_argument("--kappa", type=float, default=0.0)
    parser.add_argument("--num-iterations", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--convergence-threshold", type=float, default=1e-5)
    args = parser.parse_args()

    requested = [t.strip().lower() for t in args.models.split(",") if t.strip()]
    by_tag = {s["tag"]: s for s in MODEL_SPECS}
    model_specs = list(MODEL_SPECS) if not requested or requested == ["all"] else [by_tag[t] for t in requested]

    run_tag, manifest_path, diagnostics_dir = _resolve_vae_paths(args)
    manifest = _load_json(manifest_path)
    collapsed_by_class = _load_collapsed_dims_for_run(diagnostics_dir, manifest)

    set_global_seed(args.seed)
    snapshot = phase0_config_snapshot(args.seed, args.device)
    snapshot.update({
        "phase": "constrained_input_baselines",
        "split": "test",
        "vae_run_tag": run_tag,
        "models": model_specs,
        "attacks": ATTACK_ORDER,
        "constraint_set": "full+physics",
        "hyperparameters": {
            "input_epsilon": args.input_epsilon, "input_alpha": args.input_alpha,
            "num_steps": args.num_steps, "random_start": args.random_start,
            "lambda_conf": args.lambda_conf, "kappa": args.kappa,
            "num_iterations": args.num_iterations, "learning_rate": args.learning_rate,
            "convergence_threshold": args.convergence_threshold,
        },
        "evaluation_note": "Final sample is NOT raw_postprocess-forced; validity is measured.",
    })
    run_logger = AttackRunLogger.create(
        phase_name="constrained_input_baselines", seed=args.seed, config_snapshot=snapshot,
    )

    router = AttackRouter(device=args.device)
    router.manifest = manifest
    mask = PerturbationMask.from_preprocessing_artifacts()
    protocol_validator = ProtocolValidator(router.scaler)
    projection = VAEConstraintProjection(router.scaler, mask, enable_physics=True, device=args.device)
    split_test = load_split("test")
    detector = _fit_detector(router, args.device, collapsed_by_class)

    classifiers = {
        spec["tag"]: load_model(
            model_path=str(_REPO_ROOT / "models" / spec["checkpoint"]),
            num_features=len(FEATURE_NAMES), num_classes=len(CLASSES), device=args.device,
        )
        for spec in model_specs
    }

    per_class_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for spec in model_specs:
        classifier = classifiers[spec["tag"]]
        selected: dict[int, np.ndarray] = {}
        for class_name in SOURCE_CLASSES:
            class_id = CLASS_TO_ID[class_name]
            selected[class_id] = _select_correct_source_samples(
                x_test=split_test["X"], y_test=split_test["y_8"], classifier=classifier,
                device=args.device, class_id=class_id, max_samples=args.samples_per_class,
                selection_batch_size=args.selection_batch_size,
            )
        for attack_name in ATTACK_ORDER:
            attack_rows: list[dict[str, Any]] = []
            for class_name in SOURCE_CLASSES:
                class_id = CLASS_TO_ID[class_name]
                sample_idx = selected[class_id]
                if sample_idx.size == 0:
                    continue
                x_batch = torch.from_numpy(split_test["X"][sample_idx].astype(np.float32))
                y_batch = torch.from_numpy(split_test["y_8"][sample_idx].astype(np.int64))
                vae = router.get_vae(class_id)
                row = _evaluate(
                    attack_name=attack_name, class_id=class_id, x_batch=x_batch, y_batch=y_batch,
                    vae=vae, classifier=classifier, projection=projection, mask=mask,
                    protocol_validator=protocol_validator, detector=detector,
                    scaler=router.scaler, device=args.device, args=args,
                )
                row.update({"vae_run_tag": run_tag, "model": spec["label"], "model_tag": spec["tag"], "attack": attack_name})
                attack_rows.append(row)
                per_class_rows.append(row)
                run_logger.log(json.dumps({k: v for k, v in row.items() if not k.endswith("_mask")}))
            if not attack_rows:
                continue
            summary = _aggregate_rows(attack_rows)
            summary.update({"vae_run_tag": run_tag, "model": spec["label"], "model_tag": spec["tag"], "attack": attack_name})
            summary_rows.append(summary)

    payload = {"vae_run_tag": run_tag, "summary": summary_rows, "per_class": per_class_rows}
    with open(run_logger.run_dir / "all_results.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    _write_csv(run_logger.run_dir / "summary.csv", summary_rows, ["vae_run_tag", "model_tag"] + SUMMARY_COLUMNS)
    _write_csv(
        run_logger.run_dir / "per_class.csv", per_class_rows,
        ["vae_run_tag", "model", "model_tag", "attack", "class_name", "n", "asr_overall",
         "asr_valid_only", "asr_invalid_only", "protocol_validity_rate", "mask_compliance_rate",
         "joint_validity_rate", "idsr", "mean_l2_input", "raw_g1g8_validity_rate",
         "successful_joint_valid_count"],
    )
    _write_summary_md(run_logger.run_dir / "summary.md", summary_rows)

    print("=== Constraint-Augmented Input Attack Summary ===")
    print(f"VAE run tag: {run_tag}")
    print(f"Output directory: {run_logger.run_dir}")
    for row in summary_rows:
        print(
            f"{row['model']:9s} {row['attack']:11s} "
            f"ASR={_fmt_pct(row['asr_overall']):>8s} ASR_valid={_fmt_pct(row['asr_valid_only']):>8s} "
            f"Proto={_fmt_pct(row['protocol_validity_rate']):>8s} Mask={_fmt_pct(row['mask_compliance_rate']):>8s} "
            f"Raw={_fmt_pct(row['raw_g1g8_validity_rate']):>8s} Joint={_fmt_pct(row['joint_validity_rate']):>8s} "
            f"IDSR={_fmt_pct(row['idsr']):>8s} L2_in={_fmt_float(row['mean_l2_input']):>8s}"
        )


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Runner smoke test (small)**

Run:
```bash
C:\Users\T2530985\.conda\envs\thesis\python.exe src/attack/run_constrained_input_baselines.py --device cpu --models mlp --samples-per-class 5 --num-steps 3 --num-iterations 5
```
Expected: prints a summary directory under `outputs/latent_attacks/constrained_input_baselines_*`, with two summary lines (cinput-pgd, cinput-cw) and non-zero `n`. Verify `summary.csv`, `summary.md`, `per_class.csv`, `all_results.json`, `config_snapshot.json` exist in that directory.

- [ ] **Step 3: Verify CSV columns line up with the master harness**

Run:
```bash
C:\Users\T2530985\.conda\envs\thesis\python.exe -c "import csv,glob,os; d=sorted(glob.glob('outputs/latent_attacks/constrained_input_baselines_*'))[-1]; print(d); r=next(csv.DictReader(open(os.path.join(d,'summary.csv')))); print(sorted(r.keys()))"
```
Expected: the printed keys include `asr_overall, asr_valid_only, idsr, joint_validity_rate, mask_compliance_rate, protocol_validity_rate, raw_g1g8_validity_rate, mean_l2_input` — the subset shared with `run_all_models_attack_rerun.py`'s `summary.csv`, so the rows can be concatenated.

- [ ] **Step 4: Commit**

```bash
git add src/attack/run_constrained_input_baselines.py
git commit -m "feat(attack): add standalone runner for constraint-augmented input attacks"
```

---

## Task 4: Full run (optional, GPU)

- [ ] **Step 1: Run across all 8-class models**

Run:
```bash
C:\Users\T2530985\.conda\envs\thesis\python.exe src/attack/run_constrained_input_baselines.py --device cuda
```
Expected: a `summary.md`/`summary.csv` with 10 rows (5 models × {cinput-pgd, cinput-cw}). Compare `asr_valid_only`, `raw_g1g8_validity_rate`, and `idsr` against the latent and plain-input rows in the most recent `run_all_models_attack_rerun.py` output to populate the manifold-vs-constraints table.

> Note: this task produces results only; no code changes. Do not commit output artifacts unless the project convention is to track them.

---

## Self-Review

**Spec coverage:**
- Unit 1 `VAEConstraintProjection` → Task 1. Covered.
- Unit 2 attacks → Task 2. Covered.
- Unit 3 runner (all 5 models, full metric suite incl. raw-G1G8 + IDSR, same CSV schema, standalone) → Task 3. Covered.
- Honesty constraint (no `raw_postprocess` on final sample) → enforced: `_evaluate` measures `raw_g1g8` on `x_adv` directly; attacks return the structured-but-not-postprocessed hard projection. Covered.
- Identical hyperparameters to plain-input baseline → runner defaults match `input_baselines` / phase-4. Covered.
- Testing strategy items 1–5 → Tasks 1–3 cover projection correctness, decoder parity, differentiability, attack smoke + validity gain, runner smoke. Covered.
- Out-of-scope (no edits to master harness / VAE / existing attacks; canonical full+physics; no plots) → respected.

**Placeholder scan:** No TBD/TODO; every code step is complete.

**Type consistency:** `VAEConstraintProjection.project(x_scaled, x_original_scaled, mode)` and `.to(device)` used consistently across Tasks 1–3; attack signatures (`projection=`, keyword-only) match between definition (Task 2) and runner calls (Task 3); metadata key `constraint_projection` asserted in tests and set in both attacks.
