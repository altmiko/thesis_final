# Constraint-Augmented Input-Space PGD/CW — Design Spec

**Date:** 2026-05-31
**Branch:** version2
**Status:** Approved design, pending implementation plan

## Problem

The thesis's central result is the gap between raw attack success (ASR_raw) and
domain-validity-constrained success (ASR_valid). The proposed contribution is a
VAE-based **on-manifold latent attack** that perturbs the 16-dim latent code `z`
and decodes through a per-class β-VAE whose structured decoder enforces the
G1–G8 domain rules (plus physics rules P2/P4/P5) by construction.

A supervisor is expected to ask:

> "What if you augment ordinary PGD/CW with the **same constraint mechanics** the
> VAE attack uses — does plain PGD then match the latent attack's validity and
> success, or does the VAE manifold contribute something beyond constraint
> enforcement?"

This spec defines an **ablation** that answers that question: input-space PGD/CW
that apply the VAE's exact constraint mechanics directly to the feature vector
`x`, **without** any latent encode/decode. Comparing this against (a) plain
unconstrained input PGD/CW and (b) the full latent VAE attack isolates the
manifold's contribution from the constraints' contribution.

## Background: how the existing pieces work

- **VAE structured decoder** — `MixedInputBetaVAE._structure_continuous_raw`
  (`src/vae/model.py:229`). In raw space it reparameterizes continuous decoder
  outputs so domain rules hold by construction, fully differentiably:
  - G1 non-negativity (`clamp_min`/softplus)
  - G5 ordering `Min ≤ AVG ≤ Max` via stacked softplus gaps
  - G6 `Variance = Std²`
  - G7 `TTL = sigmoid(raw/32)·255 ∈ [0,255]`
  - G8 `Number = 1 + softplus(·)` then straight-through round (positive integer)
  - Physics (when `use_structured_physics_decoder=True`): P2 `Tot sum = Number·AVG`
    and `Tot size = AVG`; P4 `Std ≤ 0.5·(Max−Min)`; P5 singleton flows get
    zero variance.
  - Protocol is an argmax over a 6-value allowlist; TCP/UDP/ICMP/IGMP are
    *derived* from it, so G2/G4 hold by construction.
- **Hard projection** — `raw_postprocess` (`src/vae/schema.py:198`): a
  non-differentiable clip/round/reorder mirroring G1–G8, applied post-hoc in
  diagnostics. **Deliberately not used by this ablation's final sample** (see
  "Honesty constraints").
- **Latent attack** — `latent_pgd_attack` / `latent_cw_attack`
  (`src/attack/latent_pgd.py`, `latent_cw.py`). Each step: `decode_to_39(z)` →
  `apply_decoder_residual` (anchored to cancel VAE reconstruction bias) →
  `PerturbationMask` (full/partial/frozen) → `reimpose_protocol_features`
  (protocol frozen) → classifier.
- **Plain input baseline** — `input_pgd_attack` / `input_cw_attack`
  (`src/attack/input_baselines.py`): PGD/CW on `x` with **zero** constraint
  enforcement.
- **Master harness** — `run_all_models_attack_rerun.py` already runs
  `latent-pgd`, `latent-cw`, `input-pgd`, `input-cw` across the 5 8-class neural
  models (MLP, CNN, LSTM, CNN-LSTM `serial`, DualPath) on 100 correctly-classified
  test samples per non-benign source class, computing ASR, ASR_valid, protocol /
  mask / raw G1-G8 validity, IDSR (in-distribution success via Mahalanobis on the
  re-encoded latent), and L2. It emits `summary.csv`, `summary.md`,
  `per_class.csv`, `per_sample_results.csv`, `all_results.json`. **This file is
  currently modified in the working tree and is left untouched by this work.**

## Decisions (from brainstorming)

1. **Enforcement = differentiable in-loop.** The projection uses the structured
   decoder's softplus/sigmoid/straight-through math (not the hard
   `raw_postprocess` clamp), applied every PGD/CW step so gradients flow through
   the constraint surface — the truest analog of "the same mechanics as the VAE
   attack."
2. **Comparison surface = all 8-class neural models.** MLP, CNN, LSTM, CNN-LSTM
   (`serial`), DualPath; same 100-per-source-class test sampling as the master
   harness; full metric suite including raw G1-G8 validity and IDSR.
3. **Integration = standalone.** A new attack file plus a separate runner.
   `run_all_models_attack_rerun.py` is **not** modified. The runner emits the
   same CSV schema so its rows can be concatenated next to the existing
   latent/plain-input rows.
4. **Constraint set = single canonical "full + physics".** Uniform across all
   classes and models (rather than per-class VAE config), because the question is
   "PGD + the constraint mechanics," not "PGD + each class's trained VAE config."

## Architecture

Three units, each independently understandable and testable.

### Unit 1 — `VAEConstraintProjection` (new file `src/attack/constrained_input_baselines.py`)

A differentiable module that lifts the VAE decoder's constraint mechanic out of
latent space and applies it to a full 39-dim **scaled** feature vector.

- **Construction:** built from the fitted `RobustScaler` (`center_`, `scale_`
  per feature → tensors) and a `PerturbationMask`. Holds raw feature indices from
  `FEATURE_NAMES`.
- **`project(x_scaled, x_original_scaled, mode)` → x_scaled:**
  1. Differentiable affine unscale `x_raw = x_scaled · scale_ + center_`.
  2. Apply structured reparameterization at raw indices (G1/G5/G6/G7/G8 + P2/P4/P5),
     replicating `_structure_continuous_raw` exactly for the canonical
     `full` + `physics` configuration.
  3. Binaries: `sigmoid` in `mode="soft"`; straight-through round to {0,1} in
     `mode="hard"` (G3).
  4. Re-scale to scaled space.
  5. Apply `PerturbationMask.apply(x_candidate, x_original)` (full/partial/frozen),
     then `reimpose_protocol_features(·, x_original)` (Protocol Type + derived
     TCP/UDP/ICMP/IGMP frozen → G2/G4).
- **What it deliberately omits:** the anchored decoder residual
  (`apply_decoder_residual`). That term cancels VAE reconstruction bias; with
  input-space attacks `x` is the real sample, not a decode, so there is no bias to
  cancel.
- **Interface contract:** pure function of its inputs; no global state; same
  device/dtype as `x_scaled`; output shape == input shape.

### Unit 2 — the attacks (`src/attack/constrained_input_baselines.py`)

Two functions mirroring `input_baselines.py` signatures, with an added `projection`
(or `mask` + `scaler`, from which the projection is built):

- **`constrained_input_pgd_attack(...)`** — L∞ PGD identical to
  `input_pgd_attack` (random start, `α·grad.sign()`, ε-ball projection in scaled
  space) except each iterate is passed through `projection.project(x, x0, "soft")`
  before `classifier_logits`. The returned `x_adv` is produced with
  `mode="hard"`. Same zero-budget passthrough behavior when `ε ≤ 0`.
- **`constrained_input_cw_attack(...)`** — CW identical to `input_cw_attack`
  (Adam on δ, `λ·clamp(margin+κ,0) + ‖δ‖²`, early-stop on δ-shift) except the
  decoded point for logits/success is `projection.project(x0+δ, x0, …)` (soft for
  loss, hard for success/selection). Best-δ selection uses the hard sample.
- Both return `(x_adv, metadata)` with the same metadata keys as the plain-input
  variants, plus `constraint_projection=True`.

### Unit 3 — runner (`src/attack/run_constrained_input_baselines.py`)

Standalone, self-contained. Reuses by import the safe, pure infrastructure from
`latent_infra` (`AttackRouter`, `PerturbationMask`, `ProtocolValidator`,
`MahalanobisOutlierDetector`, `load_split`, `predict_labels`, `set_global_seed`,
`build_per_class_dataset`, `encode_dataset_mu`, `load_collapsed_dims`,
`AttackRunLogger`, `phase0_config_snapshot`) and `validate_batch`.

- **Sampling:** 100 correctly-classified test samples per non-benign source class
  (re-uses the master harness's selection rule), per model.
- **Attacks:** `cinput-pgd`, `cinput-cw`. Hyperparameters identical to the plain
  input baseline (`ε=0.5, α=0.05, steps=40`; CW `λ=1.0, κ=0, iters=200, lr=0.01`).
- **Metrics (full suite, per sample → per class → per model):** ASR_overall;
  ASR_valid / ASR_invalid (conditioned on joint validity); protocol validity;
  mask compliance; **raw G1-G8 validity** (`validate_batch` on inverse-transformed
  `x_adv`); joint validity = protocol ∧ mask ∧ raw-G1G8; **IDSR** (re-encode
  `x_adv` through the per-class VAE, Mahalanobis outlier check via the
  detector fit on validation latents); mean L2 in input space. `mean_l2_latent`
  is `None` (no latent perturbation).
- **VAE usage:** only for IDSR measurement (re-encode + detector) — never in the
  attack optimization. The detector is fit exactly as in the master harness
  (`_fit_detector` logic, val split, collapsed dims from the chosen VAE run).
- **Outputs:** `summary.csv`, `summary.md`, `per_class.csv`,
  `per_sample_results.csv`, `all_results.json`, `config_snapshot.json` — same
  column schema as the master harness (`SUMMARY_COLUMNS`) so rows concatenate
  cleanly. Written under the standard
  `outputs/latent_attacks/<phase>_<ts>_seed<seed>/` run dir via `AttackRunLogger`
  (timestamp format `%Y%m%d_%H%M%S`).
- **CLI:** `--seed`, `--device`, `--models` (default all), `--vae-run-tag`
  (default `DEFAULT_VAE_RUN_TAG`), plus the attack hyperparameters, mirroring the
  master harness flags that are relevant.

## Data flow

```
test split (scaled X) ──select 100 correct/class/model──> x_batch, y_batch
   │
   ├─ cinput-pgd / cinput-cw  (optimize x in scaled space)
   │        each step: x ─unscale→ structure(G1-G8,P2/P4/P5) ─rescale→ mask ─> reimpose protocol ─> classifier
   │        final:  x_adv (hard projection)
   │
   └─ metrics:
        success      = argmax(classifier(x_adv)) != y
        protocol     = ProtocolValidator(x_adv)
        mask         = PerturbationMask.verify(x_adv, x_batch)
        raw_g1g8     = validate_batch(inverse_transform(x_adv))
        joint_valid  = protocol & mask & raw_g1g8
        idsr         = ~detector.outlier(class, vae.encode(x_adv))
        l2_input     = ‖x_adv − x_batch‖₂
```

## Honesty constraints (important for defensibility)

- The final `x_adv` is **not** run through `raw_postprocess`. Raw G1-G8 validity is
  *measured*, not trivially forced to 100%. The structured projection makes most
  of G1-G8 hold by construction (the same way the VAE does), but mask re-clamping
  in scaled space, the scaled↔raw round-trip, and binary handling can still leave
  residual violations — exactly as in the latent attack.
- Optimizer settings are identical to the plain input baseline; the **only**
  difference is the projection. This keeps "plain-input vs constrained-input" a
  clean one-variable ablation.
- The VAE attack additionally has the learned per-class manifold and the outlier
  detector; the augmented attack gets only the constraint mechanics. The
  remaining gap (in ASR_valid, raw G1-G8 validity, and especially IDSR) is the
  evidence about what the manifold contributes.

## Testing strategy

1. **Projection correctness (unit):** for a batch of random scaled vectors, the
   **hard**-projected output, inverse-transformed, passes `validate_batch` for
   G1, G3, G5, G6, G7, G8 at ~100% on the structurally-enforced features
   (protocol/derived frozen from a valid original). Confirms the lifted mechanic
   reproduces the decoder's guarantees.
2. **Projection ≈ decoder parity (unit):** feeding the VAE decoder's own
   continuous raw outputs through the standalone projection yields the same
   structured continuous values the in-model `_structure_continuous_raw` produces
   (within float tolerance) for the `full`+`physics` config. Guards against drift
   from the source mechanic.
3. **Differentiability (unit):** `project(x, x0, "soft")` returns a tensor with a
   grad path back to `x` (gradient is finite, non-NaN) for the dims that are
   not frozen.
4. **Attack smoke (integration):** on a tiny batch (e.g., 16 samples, 5 steps),
   both attacks run end-to-end on CPU, return correct shapes, and
   `constrained-input` raw G1-G8 validity is **strictly higher** than
   plain-`input` on the same batch (sanity that the projection does something).
5. **Runner smoke (integration):** `--models mlp --samples-per-class 5
   --num-steps 3 --num-iterations 5` produces all output files with the expected
   columns and non-empty rows.

## Risks / open points

- **softplus reparameterization distorts magnitudes.** Applying `softplus` to
  already-large packet statistics every step shifts values near zero. This is the
  VAE's actual mechanic, so it is the faithful comparison; documented, not "fixed."
- **Scaler affine assumption.** Differentiable unscale assumes `RobustScaler`'s
  per-feature affine transform (`center_`, `scale_`). True for this project's
  scaler; asserted at construction.
- **CW with in-loop projection** may converge differently than plain CW; that is
  expected and is part of the result, not a bug.

## Out of scope

- Modifying `run_all_models_attack_rerun.py`, the VAE model, or any existing
  attack.
- Per-class VAE-config-matched projection (canonical full+physics only).
- New plots / thesis figures (this spec delivers the attack + runner + CSVs only;
  visualization can be a follow-up).
- Re-running or altering the latent / plain-input baselines.
