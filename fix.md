# Code Audit & Proposed Fixes

**Scope:** `src/` (75 Python modules) + `tests/`, audited 2026‑06‑02 on branch `version2`.
**Method:** Deep read of the foundation/core layer and all recently‑modified files; whole‑tree
byte‑compile (clean); cross‑cutting pattern sweeps (bare excepts, mutable defaults, `== None`,
import roots, `sys.path` hacks). Findings marked **[verified]** were reproduced; others are
**[by inspection]**.

Severity legend: 🔴 high (run‑blocking / correctness) · 🟠 medium · 🟡 low / hygiene.

---

## 🔴 1. `sys.path.insert` points at the wrong directory in EDA + preprocessing entry points  **[verified]**

**Files:** `src/evaluation/eda_tables.py:8`, `src/evaluation/eda_figures_part1.py:12`,
`src/evaluation/eda_figures_part2.py:11`, `src/preprocessing/pipeline.py:16`

Each does:
```python
sys.path.insert(0, 'D:/thesis_final/src')
from src.preprocessing.feature_groups import ...   # needs repo ROOT on path, not src/
```
The inserted path is `…/src`, but `from src.preprocessing…` resolves only when the **repo root**
(`D:/thesis_final`, the parent of `src/`) is on `sys.path`. Verified:

```
# only …/src on path:
>>> import src.preprocessing.feature_groups
ModuleNotFoundError: No module named 'src.preprocessing'
# repo root on path:
>>> import src.preprocessing.feature_groups   # OK
```

So the insert is dead/misleading code. These scripts run **only** via
`python -m src.evaluation.eda_tables` from the repo root (where `-m` puts cwd on the path); the
insert contributes nothing and the documented direct invocation fails.

**Fix:** replace the line with a root insert and drop the hardcoded drive path, e.g.
```python
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root
```
(or change the imports to the `from preprocessing.feature_groups import …` style and insert `…/src`).

---

## 🔴 2. Documented run commands reference moved/renamed files  **[verified]**

`CLAUDE.md` lists e.g. `python src/eda_tables.py`, `python src/preprocessing.py`,
`python src/baseline_experiments.py`, `python src/run_attacks.py`. The files now live in
sub‑packages: `src/evaluation/eda_tables.py`, `src/preprocessing/pipeline.py`,
`src/classifiers/baseline_experiments.py`, `src/attack/run_attacks.py`. The documented commands
fail with *file not found*, and `baseline_experiments.py` additionally uses `from src.classifiers…`
so it needs `-m` from root.

**Fix:** update the Commands section of `CLAUDE.md` to the working invocation, e.g.
`python -m src.evaluation.eda_tables`, `python -m src.preprocessing.pipeline`,
`python -m src.classifiers.baseline_experiments`, etc. Pair with fix #3.

---

## 🔴 3. Two incompatible package‑import conventions across the tree  **[verified]**

The codebase mixes two roots that require **different** `sys.path` setups:

| Convention | Requires on path | Used by (examples) |
|---|---|---|
| `from src.X import …` | repo **root** | `preprocessing/pipeline.py`, `evaluation/*` (eda, delta_report, validity_analysis, analyze_attack_restarts), `classifiers/*` |
| `from attack.X` / `from vae.X` / `from preprocessing.X` | **`src/`** | all of `vae/*`, most of `attack/*`, `plot_benign_overlay_fidelity.py` |

Each module then patches `sys.path` ad hoc (16 different `sys.path.insert` sites, some to repo root,
some to `src/`, some to a hardcoded absolute path). `adversarial_attacks.py:18-20` even wraps the
import in `try/except ModuleNotFoundError` to tolerate both. This is fragile: whether any given
script imports successfully depends on the entry point *and* its private path hack.

**Fix (pick one and apply tree‑wide):**
- Standardize on absolute `src.` imports, delete every per‑file `sys.path.insert`, and add a single
  repo‑root `conftest.py` (for pytest) plus a documented `python -m src.…` convention; **or**
- Add a minimal `pyproject.toml`/`setup.cfg` and `pip install -e .`, so `import src.*` (or a top
  package name) resolves everywhere without path hacking.

---

## 🟠 4. `run_attack(perturbation_mask=…)` silently ignores the mask  **[by inspection]**

**File:** `src/attack/adversarial_attacks.py:296-297` (TODO), param at `:246`

The unconstrained‑attack design is intentional (the ASR_raw vs ASR_valid gap is the thesis
contribution — see `CLAUDE.md`). The problem is that `run_attack`/`run_attack_with_restarts` still
*accept* `perturbation_mask` and validate its shape (`:259-260`) but never apply it, so a caller who
passes a mask gets an unconstrained attack with **no signal** that the mask was dropped.

**Fix:** make the no‑op explicit — either implement the masked update
(`x_adv = x + (x_adv - x) * mask`) behind a flag, or `raise NotImplementedError`/`warnings.warn`
when a non‑`None` mask is supplied, so it cannot be silently ignored.

---

## 🟠 5. `AttackRouter` has a stale `lightgbm` path and no tree‑model aliases  **[by inspection]**

**File:** `src/attack/latent_infra.py:312-330, 410-413`

Defaults point at `models/lightgbm_8class.pkl` (doesn't exist — the project shipped **XGBoost /
RandomForest**, see memory + `models/{xgb,rf}_*.pkl`), and `_normalise_classifier_name` only knows
`mlp/cnn/lightgbm`. Routing a latent attack at a tree classifier raises `KeyError`/`FileNotFoundError`.

**Fix:** drop the `lightgbm` entry or repoint it to the real tree artifacts, add `xgb`/`rf` aliases,
and load tree models via `joblib.load` (the `predict_labels` helper at `:642` already handles
non‑`nn.Module` estimators).

---

## 🟠 6. Aggregate multiclass ROC‑AUC can raise on incomplete label coverage  **[by inspection]**

**File:** `src/classifiers/review_baselines.py:236-241`

```python
roc_auc_score(y_true, y_score, labels=labels, average="macro", multi_class="ovr")
```
The per‑class loop above (`:218-230`) defensively sets `NaN` for classes with <2 distinct labels,
but this aggregate call errors if any class in `labels` is absent from `y_true`. Full runs are safe
(stratified split guarantees all classes), but `--limit-samples` smoke tests can trigger it.

**Fix:** wrap in `try/except ValueError` (fall back to averaging the finite per‑class AUCs), or
compute the macro/weighted AUC directly from `auc_by_class`, skipping `NaN`s.

---

## 🟠 7. `decode_to_39` accepts any `mode` and silently treats non‑`"soft"` as hard  **[by inspection]**

**File:** `src/vae/model.py:391-394`

```python
if mode == "soft":
    binary_output = torch.sigmoid(dec["binary_logits"])
else:                     # "hardd", "Soft", anything → hard threshold
    binary_output = (dec["binary_logits"] > 0.0).float()
```
A typo silently changes attack semantics (this runs in every PGD/CW inner step). Note
`constrained_input_baselines.VAEConstraintProjection._structure_raw` *does* validate `mode`
(`:77-78`); `decode_to_39` should match.

**Fix:** `if mode not in {"soft","hard"}: raise ValueError(mode)` at the top.

---

## 🟡 8. Hardcoded absolute Windows paths block portability/CI  **[by inspection]**

**Files:** `src/preprocessing/pipeline.py` (`PROC_DIR`, `logs/`, `D:/thesis_final/config`,
lines 16/26‑28/33/517), all `src/evaluation/eda_*` (`D:/thesis_final/{figures,tables,data}`),
`src/vae/_rediag.py:3`.

`CLAUDE.md` acknowledges EDA/preprocessing use hardcoded `D:/thesis_final/`, but it prevents running
on any other machine/checkout. **Fix:** derive from `Path(__file__).resolve().parents[N]` (the
attack/model scripts already do this) and accept a `--data-dir`/`--out-dir` override.

---

## 🟡 9. Silent `except Exception: pass` hides real failures  **[by inspection]**

- `src/classifiers/tree_baselines.py:419-424` — swallows failure of
  `set_params(device="cpu")` / `set_param({"device":"cpu"})`. If both fail, the XGBoost model is
  persisted still pointing at CUDA, reproducing exactly the device‑mismatch bug that motivated this
  block (memory obs #96). At minimum `warnings.warn` on failure.
- `src/thesis_visualizations.py:424-425, 600-601` — plotting fallbacks; acceptable but should log.

---

## 🟡 10. Integration tests masquerade as unit tests; load full arrays for a few rows  **[by inspection]**

**File:** `tests/attack/test_constrained_input_baselines.py`

- `_real_scaled_samples` (`:31`) does `np.load(".../X_test.npy")[:n]` — loads the **entire** test
  matrix into RAM to slice 4–16 rows. Use `np.load(..., mmap_mode="r")[:n].copy()`.
- Every test depends on real artifacts (`scaler.pkl`, `X_test.npy`, the preprocessing JSONs) and on
  constructing a `MixedInputBetaVAE`; they cannot run on a fresh checkout/CI without the full data
  pipeline. Add synthetic fixtures for the pure‑math paths (`_structure_raw`, soft differentiability)
  and mark the artifact‑dependent ones with `@pytest.mark.integration`.
- This is also the project's *only* test file — 74 of 75 modules are untested. The constraint
  projection, validator rules (`validator.py`), and `compute_asr_valid`/`compute_attack_metrics` are
  high‑value, pure‑function targets worth covering.

---

## 🟡 11. Minor correctness/reporting nuances  **[by inspection]**

- `adversarial_attacks.compute_attack_metrics:444-446` — `mean_l2`/`mean_l_inf` average over **all**
  rows, including originally‑misclassified samples and CW rows that returned unchanged (Δ=0). This
  understates perturbation size. Consider reporting the mean over the `flipped` mask (or report both).
- `models.py:850` (`__main__` smoke test) uses `num_features=45`; the real schema is 39
  (`feature_groups.FEATURE_NAMES`). Harmless but misleading — align to `len(FEATURE_NAMES)`.
- `constrained_input_baselines.VAEConstraintProjection.to():62-68` moves only 4 of the tensors
  created in `__init__`; it's currently sufficient (the rest are Python ints / handled via
  `.to(x.device)` at use sites), but it reads as an incomplete `.to()`. Either move everything or
  drop the method and rely on the per‑call `.to(device)` calls already present.

---

## VAE / CPGD / latent‑attack deep dive

Focused second pass over the VAE core (`model`, `schema`, `losses`, `dataset`, `config`,
`diagnostics`, `physics_validator`), the constrained‑input ("CPGD") attack
(`constrained_input_baselines` + runner), and the latent attacks (`latent_pgd`, `latent_cw`,
`latent_gmm`, `latent_restarts`, `latent_infra`).

**Bottom line:** the *actively‑exercised* paths are correct — untargeted latent PGD/CW, the
**targeted latent PGD** path used by `run_targeted_benign_*`, and the constrained‑input PGD/CW all
trace cleanly and are backed by `tests/attack/test_constrained_input_baselines.py`
(`test_parity_with_vae_structured_decoder`). The structured‑decoder math matches between
`model._structure_continuous_raw` and `constrained_input_baselines._structure_raw`, the feature‑index
maps agree, the anchored decoder‑residual → mask → protocol‑reimpose composition is sound, and the
IDSR collapse/Mahalanobis active‑dim computation is correct. The findings below are **dormant or
low‑severity**; none corrupts current results.

### 🟠 12. `get_vae` loads checkpoints with `strict=False`, masking architecture mismatch  **[by inspection]**

**File:** `src/attack/latent_infra.py:385`

`vae.load_state_dict(ckpt["state_dict"], strict=False)`. If `_resolve_model_hparams` ever resolves
the wrong `decoder_hidden`/`latent_dim`/etc. for a checkpoint, missing keys are silently skipped and
the attack runs against a partly‑random decoder with no error. The protocol‑reference buffers it's
presumably tolerating are overwritten by `register_protocol_references` immediately after, so they
don't need `strict=False`.

**Fix:** load with `strict=True`; if buffer presence/absence is the reason for the relaxation,
capture `missing, unexpected = load_state_dict(..., strict=False)` and assert both are empty (modulo
the known protocol‑reference buffer names).

### 🟡 13. Targeted `latent_cw` uses the wrong CW objective and an inconsistent success test (dormant)  **[by inspection]**

**File:** `src/attack/latent_cw.py:125-133` (objective), `:154-162` (inner success), `:195` (final selector)

In the `targeted` branch the margin is `true_logits − target_logits` and inner success is
`true − target ≤ 0`. For multiclass targeted CW this is wrong twice: (1) the objective should drive
`target` above **max‑of‑all‑other‑classes‑excluding‑target**, not merely above the true class;
(2) the inner success mask disagrees with the final selector `_attack_success_mask` (`argmax ==
target`), so an inner "success" can be a real non‑success.

**Currently dormant** — all four call sites
(`build_thesis_bundle.py:127`, `export_new_vae_attack_samples.py:185`,
`run_all_models_attack_rerun.py:551`, `run_phase3_latent_cw.py:136/178`) use the default
`targeted=False`. The targeted **PGD** path does it correctly via `target_logit_margin`
(`latent_pgd.py:38-52`), which is why targeted‑Benign results are unaffected.

**Fix:** when porting CW to targeted, reuse the PGD `cw-margin` formulation
(`max_other_excluding_target − target_logit`) and define inner success as `argmax == target`.

### 🟡 14. `latent_cw` internal multi‑restart is a no‑op without GMM initializers (dormant)  **[verified by inspection]**

**File:** `src/attack/latent_cw.py:78-85`, with `_normalise_z_initializers` in `latent_pgd.py:194-203`

`latent_cw` calls `_normalise_z_initializers(epsilon=0.0, random_start=False)`. With
`z_initializers=None`, the per‑restart jitter is `uniform(−0.0, 0.0)` → every restart starts at
exactly `z_orig`, and CW's Adam trajectory is deterministic, so all `num_restarts` runs are
**identical**. `num_restarts>1` then burns compute and yields a misleading `restart_success_counts`
(all equal). It only diversifies when explicit GMM `z_initializers` are supplied.

Not wrong output in practice: `run_phase3`/`export_new_vae` use the default `num_restarts=1`, and
`run_all_models_attack_rerun.py:551` orchestrates restarts externally
(`num_restarts=1 if init is not None else num_restarts`), so the only way to hit it is the
`init is None` fallback with `num_restarts>1`.

**Fix:** drop the internal restart loop for CW, or inject a small fixed latent σ jitter for
`restart_idx>0` even when `epsilon==0`, so restarts actually differ.

### 🟡 15. `MixedInputBetaVAE.encode` silently returns wrong embeddings if references not registered  **[by inspection]**

**File:** `src/vae/model.py:317-322` (use), `:155` (registration)

The placeholder `ref_proto_scaled` buffer is all zeros, so `argmin` is always 0 → every sample embeds
as protocol index 0 if `register_protocol_references` was never called. Documented in the docstring,
guaranteed by the attack code today, but unguarded.

**Fix:** add a one‑line guard, e.g. `if torch.count_nonzero(self.ref_proto_scaled) == 0: raise
RuntimeError("call register_protocol_references(scaler) before encode()")`.

### 🟡 16. Inner soft‑vs‑hard success inconsistency in `latent_cw`  **[by inspection]**

**File:** `src/attack/latent_cw.py:144` (inner, `mode="soft"`) vs `:187` (final, `mode="hard"`)

Per‑restart `best_delta` is chosen using **soft**‑projection success, while the final cross‑restart
pick and the reported success use **hard**. The constrained‑input CW uses hard for both
(`constrained_input_baselines.py:291`). Final reported numbers are hard‑consistent, so this only
affects which low‑L2 delta wins *inside* a restart — minor. Align to hard for consistency.

### 🟡 17. Dead `len(resolved_hparams)==8` branch in `get_vae`  **[verified]**

**File:** `src/attack/latent_infra.py:355-366`

`_resolve_model_hparams` (`vae/train_all.py:57-93`) always returns a 9‑tuple now, so the 8‑tuple
branch is unreachable. Harmless drift — delete it (and the `len(...)` dispatch) to avoid implying two
live checkpoint formats.

> **Note:** the `decode_to_39` `mode`‑validation issue raised in this pass is already filed as **#7**
> above (it applies to both `decode_to_39` and is the counterpart of the guard that
> `constrained_input_baselines._structure_raw` already has).

### Verified correct in this pass (no action)

- Untargeted latent PGD/CW objectives and success masks; targeted latent **PGD** `cw-margin`.
- Anchored decoder residual (`apply_decoder_residual`) + `PerturbationMask.apply` +
  `reimpose_protocol_features` composition; epsilon‑ball projection.
- `schema` partition (39 = 23 continuous + 11 indep‑binary + 4 derived + 1 protocol),
  protocol↔binary derivation order, and the buffer‑trick scaler conversions.
- `losses.compute_elbo` (Gaussian/Laplace NLL, free‑bits KL, constraint/physics penalties),
  `BetaScheduler`.
- `diagnostics._diag_posterior_collapse` per‑dim KL and collapsed‑dim indices → IDSR active dims.
- `physics_validator` P2/P4/P5 (P1/P3/P6/P7/P8 intentionally omitted).
- `latent_gmm` sampling / caching and `latent_restarts` initializer construction.
- `constrained_input_pgd_attack` / `constrained_input_cw_attack` (untargeted) objectives and the
  hard‑projection success accounting in `run_constrained_input_baselines._evaluate`.

---

## Suggested order of work

1. **#1–#3 (imports/paths/docs)** first — they gate the ability to *run* the pipeline reproducibly,
   and #1/#2 are reproducible breakages today.
2. **#4–#7** — small, localized correctness/robustness fixes.
3. **#8–#11** — hygiene; bundle with the import refactor (#3) since they touch the same files.

## Not bugs (verified during audit — no action needed)

- `validator.py` G6 `R_var_eq_std_sq` uses `&` (abs **and** rel tolerance) — intentional relative
  tolerance away from zero.
- `run_attack` RNN handling (`adversarial_attacks.py:267-272`): putting single‑layer LSTMs in train
  mode with dropout/BN kept in eval is correct (enables cuDNN backward; predictions stay
  deterministic). The newer latent path uses `cudnn.flags(enabled=False)` instead — also fine.
- VAE↔input‑space structured‑decoder parity is covered by
  `test_parity_with_vae_structured_decoder` and the feature‑index maps match between
  `model._structure_continuous_raw` and `constrained_input_baselines._structure_raw`.
- No mutable default arguments and no `== None` comparisons anywhere in `src/`.
