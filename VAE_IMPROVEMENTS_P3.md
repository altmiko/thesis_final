# VAE Improvements P3: Targeted-Benign Latent Attacks

Workspace: `D:\thesis_final`

Date: 2026-06-01

## Goal

Improve targeted-to-Benign latent attacks while preserving the thesis constraint story:

- Attack samples should be classified as `Benign`.
- Samples should remain protocol-valid, mask-compliant, raw G1-G8 valid, and jointly valid.
- Improvements should be measured primarily by joint target success, not raw target success alone.

## Hard Constraints

These constraints are non-negotiable for every P3 implementation and rerun.

- Do not relax, disable, or modify the 49-rule validator.
- Do not relax, disable, or modify the protocol allowlist.
- Do not relax, disable, or modify the perturbation mask: 11 Full / 9 Partial / 19 Frozen.
- Do not relax, disable, or modify the repair step to inflate ASR.
- Do not count attack-to-attack misclassification as success. Success means the predicted class is exactly `Benign`.
- Keep preprocessing, splits, scaler, and feature schema identical to the current pipeline.
- Keep the `RobustScaler` and current train/val/test split artifacts unchanged.
- Preserve apples-to-apples comparisons across models, baselines, and reruns.
- Every new behavior must be toggleable through config or CLI flags so old-vs-new ablations can be run.
- Preserve and version all checkpoints.
- Never overwrite existing run artifacts.
- Write every new run to a new output directory.
- Update `run_manifest.json` or the run-local manifest/config snapshot with the exact settings, seed, checkpoint paths, and code/config options used.

## Orient Before Code

Before implementing P3 code, first read and summarize the current wiring. Do this before editing files.

Files/modules to inspect:

- Existing targeted attack runner: `src\attack\run_targeted_benign_latent_pgd.py`
- Wrapper for Gaussian/Laplace targeted runs: `src\attack\run_targeted_benign_new_vae_methods.py`
- Shared latent PGD primitive: `src\attack\latent_pgd.py`
- Latent restart/GMM logic: `src\attack\latent_restarts.py`, `src\attack\latent_gmm.py`
- Attack infrastructure, decode path, repair/mask/protocol validation helpers: `src\attack\latent_infra.py`
- Validator and raw G1-G8 checks: `src\attack\validator.py`
- Physics validator if used by repair/validity flow: `src\vae\physics_validator.py`
- Evaluation and metrics exporters: `src\evaluation\validity_analysis.py`, `src\evaluation\run_validity_analysis.py`, `src\attack\run_attack_statistics.py`
- Existing result exporters for targeted benign samples: `src\attack\export_targeted_benign_attack_samples.py`
- Relevant config/run manifests: `config\run_manifest.json`, `vae_run_manifest.json`, `results\vae\*\vae_run_manifest.json`

Orientation summary must answer:

- How samples are selected for attack.
- How latent `z` is initialized and perturbed.
- How `z` is decoded to scaled feature space.
- Where repair is applied.
- Whether repair is inside or outside the optimization loop.
- Where protocol, mask, raw G1-G8, joint validity, and IDSR are computed.
- Where logits and predictions are computed pre-repair and post-repair.
- Where success is currently checked.
- Whether current targeted success already means `pred_after == Benign`.
- Where per-class/per-sample/summary metrics are written.
- Exactly where each P3 upgrade should hook in.

Ask before any structural refactor. Local, toggleable additions are allowed after the orientation pass, but moving ownership boundaries or rewriting shared attack APIs should be checkpointed first.

Current targeted-benign rerun command:

```powershell
python src\attack\run_targeted_benign_new_vae_methods.py `
  --methods gaussian,laplace `
  --models all `
  --samples-per-class 100 `
  --num-restarts 5 `
  --restart-strategy encoded+jitter+gmm `
  --force-refit-gmm
```

Recent outputs:

- Gaussian: `outputs\latent_attacks\targeted_benign_pgd_gaussian_anticollapse_beta05_freebits01_20260529_173512_20260601_055256_seed42\summary.csv`
- Laplace: `outputs\latent_attacks\targeted_benign_pgd_laplace_rerun_20260529_20260601_055517_seed42\summary.csv`

## Current Baseline

Gaussian joint target success:

| Model | Target success | Joint target success | Joint validity | IDSR | Mean L2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| CNN | 5.00% | 5.00% | 98.40% | 91.60% | 4.1489 |
| CNN-LSTM | 3.83% | 3.83% | 96.83% | 93.83% | 4.8747 |
| DualPath | 4.43% | 4.14% | 85.14% | 91.43% | 8.4570 |
| LSTM | 5.29% | 4.86% | 89.29% | 93.43% | 10.0490 |
| MLP | 4.14% | 3.43% | 86.29% | 92.43% | 9.5753 |

Laplace joint target success:

| Model | Target success | Joint target success | Joint validity | IDSR | Mean L2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| CNN | 6.20% | 5.80% | 97.80% | 94.80% | 3.2264 |
| CNN-LSTM | 3.50% | 3.17% | 96.00% | 93.50% | 3.4220 |
| DualPath | 4.00% | 3.57% | 91.86% | 94.14% | 3.5341 |
| LSTM | 4.86% | 4.29% | 90.86% | 93.57% | 4.3241 |
| MLP | 4.14% | 3.43% | 91.86% | 93.14% | 3.5867 |

## Diagnosis

The current targeted-benign attack is stricter than the non-targeted latent attacks. It tries to keep each sample on the source attack-class VAE manifold while forcing the classifier to output `Benign`. That means the attack is searching for classifier failures that are still valid and source-coherent, not merely nearby points in input space.

This matches the on-manifold adversarial-example literature: on-manifold examples can exist, but they are often harder to find and can behave more like hard test examples than like ordinary norm-bounded perturbations.

The low 3-6% joint target success may therefore reflect one or more of:

- The attack objective is too weak for targeted optimization.
- The PGD step schedule is not adaptive enough.
- Source-only GMM restarts do not seed the search near Benign decision regions.
- The global latent budget is too small for some source classes.
- Some source-class manifolds have little valid overlap with the Benign decision region.
- Class-specific VAE latent spaces make "move toward Benign" poorly defined.

## Required P3 Upgrade Sequence

Implement and evaluate upgrades in this order. Upgrades 1 and 2 are the first implementation checkpoint. Do not implement upgrades 3-6 blind; pause after 1-2 and review the valid_ASR-vs-kappa results before proceeding.

### Upgrade 1: Benign-Targeted CW-Style Latent Loss

Reformulate the targeted attack objective toward `Benign`. Keep the existing untargeted/legacy objective path available behind a flag for ablation.

Required loss:

```text
loss = max(max_{i != benign} Z_i - Z_benign, -kappa) + lambda * ||delta_z||_2
```

Where:

- `Z` are target-model logits on the decoded sample.
- If `repair_in_loop=true`, `Z` should be logits on the repaired sample or on the straight-through repaired sample.
- `delta_z` is the latent perturbation from the original encoded latent.
- Optimization should drive the Benign logit above every non-Benign logit.

Required flags:

```text
--target-loss ce|cw-margin
--target-class Benign
--kappa 0.0
--lambda-latent-l2 0.0
```

Success definition:

```text
predicted_class_after_attack == Benign
```

Do not count any attack-to-attack transition as success.

Expected hook points:

- Add the loss option in `src\attack\latent_pgd.py` if the shared PGD primitive owns the classifier objective.
- Add targeted-benign-specific selection/logging in `src\attack\run_targeted_benign_latent_pgd.py`.
- Keep old behavior selectable through `--target-loss ce` or an equivalent legacy flag.

### Upgrade 2: Confidence Margin Kappa Sweep

Make `kappa` configurable and add a sweep utility.

Initial grid:

```text
0, 5, 10, 20, 40
```

Required outputs:

- CSV of valid_ASR versus `kappa`.
- Plot of valid_ASR versus `kappa`.
- Breakdown per model.
- Breakdown per attack category/source class.
- Post-repair target success where repair is enabled.

Recommended output columns:

```text
run_id
vae_run_tag
model
model_tag
source_class
attack_config
kappa
n
raw_attack_to_benign_rate
post_repair_benign_rate
protocol_validity_rate
mask_compliance_rate
raw_g1g8_validity_rate
joint_validity_rate
final_asr_valid
mean_l2_latent
mean_l2_input
attempts_per_sample_mean
seed
```

Required checkpoint:

- Run upgrades 1 and 2.
- Show valid_ASR versus `kappa`.
- Confirm baselines are unchanged.
- Confirm no validator/mask/protocol/repair constraints were relaxed.
- Wait before proceeding to upgrades 3-6.

### Upgrade 3: Repair-In-The-Loop

Current repair is likely post-hoc:

```text
attack -> decode -> repair -> validate/check success
```

P3 should add a toggle to optimize for the post-repair outcome:

```text
attack step -> decode -> repair or repair relaxation -> logits/loss -> update z
```

Required flag:

```text
--repair-in-loop true|false
```

If repair is differentiable:

- Use the repaired sample directly in the loss.

If repair is non-differentiable:

- Implement a straight-through estimator where the forward pass uses repaired values and the backward pass routes gradients through the unrepaired decoded tensor.
- If STE is unstable or invalid for some repair operations, add a score-based/search outer loop that evaluates repaired candidates and selects the best post-repair Benign-valid candidate.

Guardrail:

- Repair logic must not be weakened.
- Evaluation must clearly distinguish pre-repair raw success from post-repair still-Benign valid success.

### Upgrade 4: Restrict Attack to Active Latent Dimensions

Per class, identify collapsed/inactive latent dimensions using posterior variance and/or KL contribution from the VAE diagnostics.

Required behavior:

- Rank latent dimensions by KL contribution or equivalent activity signal.
- Expose a config for how many top active dimensions to attack.
- Freeze inactive dimensions during latent perturbation.
- Report active dimensions per class in run outputs.

Suggested flags:

```text
--active-dim-mode all|noncollapsed|topk
--active-dim-topk 16
--active-dim-source diagnostics
```

Expected hook points:

- Load collapsed dims from diagnostics in `run_targeted_benign_latent_pgd.py`.
- Apply an active-dim mask inside the latent optimizer/restart initializer.
- Log active dims in config snapshot and per-class output.

Important note:

- Mirai may remain near 0% even after active-dim restriction. That is a valid finding, not something to force.

### Upgrade 5: Multi-Start Best-of-N Latent Initialization

For each sample, run `N` latent initializations and keep the best result.

Required behavior:

- `N` must be configurable.
- Selection should prefer valid Benign evasion first.
- If no valid Benign evasion exists, keep the candidate with best post-repair target margin/loss.
- Record attempts per sample.

Suggested flags:

```text
--n-restarts 5
--restart-strategy encoded+jitter+gmm
```

Required reporting:

```text
attempts_per_sample
selected_restart
selected_restart_label
restart_target_success_counts
restart_valid_benign_counts
```

### Upgrade 6: Per-Class Attack Tuning

Allow these values to be set per attack category/source class:

- `kappa`
- step size / alpha
- `n_restarts`
- active-dim count

Suggested flags:

```text
--kappa-by-class "DDoS=10,DoS=10,Mirai=0,Recon=5,Spoofing=5,Web=10"
--alpha-by-class "DDoS=0.08,DoS=0.08,Mirai=0.05,Recon=0.05,Spoofing=0.05,Web=0.08"
--n-restarts-by-class "DDoS=10,DoS=10,Mirai=10,Recon=5,Spoofing=5,Web=10"
--active-dim-topk-by-class "Mirai=7,DoS=10,DDoS=12,Recon=12,Spoofing=12,Web=16"
```

Guardrail:

- Per-class tuning must be reported transparently and cannot be hidden inside aggregate metrics.

## Research Takeaways

### On-Manifold Latent Attacks

David Stutz's CVPR 2019 work shows that on-manifold adversarial examples can be generated by optimizing in latent space through a decoder, but also notes that these examples are harder to find when models generalize well.

Source: https://davidstutz.de/on-manifold-adversarial-examples/

Relevance here:

- Your targeted-benign attack is effectively an on-manifold latent attack.
- It should be expected to have lower success than unconstrained PGD/CW.
- Stronger optimization and better latent priors are needed before concluding that the boundary is unreachable.

### Targeted Margin Losses

Carlini and Wagner attacks are a standard reference for stronger targeted attacks. Their key practical lesson is that targeted attacks benefit from direct logit-margin objectives and confidence margins, not just cross-entropy.

Source: https://arxiv.org/abs/1608.04644

Relevance here:

- Add a targeted CW-style loss in latent space.
- Optimize `benign_logit - max_other_logit` directly.
- Add a confidence parameter `kappa` to push examples deeper into the Benign region.

### APGD and DLR-Style Losses

AutoAttack introduced APGD variants to avoid weak robustness evaluations caused by poor step sizes and objective-function failures. The useful idea for this repo is not necessarily the full AutoAttack suite, but the adaptive PGD mechanics and robust margin-style losses.

Source: https://arxiv.org/abs/2003.01690

Relevance here:

- The non-targeted runner already has adaptive PGD machinery.
- The targeted-benign runner should get the same treatment.
- Track best target margin over time, halve step size on stagnation, and preserve the best candidate per sample.

### Generative Target Conditioning

Song et al. construct unrestricted adversarial examples by searching a class-conditional generator's latent space conditioned on a desired class.

Source: https://arxiv.org/abs/1805.07894

Relevance here:

- Current GMM restarts are source-class priors only.
- Target-aware seeds may matter more for targeted-Benign success.
- A shared class-conditional VAE would make target-conditioned latent movement cleaner than separate per-class VAEs.

### Constrained Tabular Attacks

Recent tabular attack work argues that image-style PGD is insufficient for tabular data because tabular examples have categorical, immutability, and feature-relationship constraints. CAPGD/CAA combine adaptive gradient attacks with constraint handling and search.

Sources:

- https://arxiv.org/abs/2406.00775
- https://arxiv.org/abs/2311.04503

Relevance here:

- A pure latent attack may fail to exploit valid mutable feature directions after decoding.
- A small constrained feature-space polish step after latent PGD could improve target success while preserving validity.

### IoT/NIDS Realism

IoT intrusion-detection robustness work emphasizes that valid network-flow examples must satisfy both domain validity and class coherence. A sample that fools a model but violates traffic semantics is not a useful cyber-attack example.

Source: https://link.springer.com/article/10.1007/s12243-023-00953-y

Relevance here:

- Keep joint validity as a hard reporting metric.
- Do not improve target success by weakening protocol, mask, or raw G1-G8 checks.
- Per-class feasibility analysis is important: some source classes may be hard to turn into Benign while preserving class coherence.

## Ranked Improvement Plan

### 1. Add Targeted Margin PGD

Replace or supplement targeted cross-entropy with a direct target-margin objective.

For a target class `t = Benign`:

```text
target_margin = logit_t - max(logit_k for k != t)
loss_to_minimize = -target_margin
```

Or CW-style:

```text
loss = max(max_other_logit - benign_logit + kappa, 0)
```

Expected effect:

- Better optimization signal when cross-entropy saturates.
- More useful failure diagnostics because best margin can be tracked even when target success is false.

Implementation target:

- `src\attack\run_targeted_benign_latent_pgd.py`
- `src\attack\latent_pgd.py` if the shared PGD primitive needs a new loss option.

Suggested flags:

```text
--target-loss ce|margin|cw-margin
--kappa 0.0
```

### 2. Add Adaptive Targeted PGD

Port the adaptive PGD logic from the non-targeted rerun path into targeted-benign PGD.

Mechanics:

- Track target margin or CW loss every `checkpoint_interval`.
- If improvement stalls, reduce alpha by `rho`.
- Stop alpha decay at `min_alpha`.
- Preserve the best candidate per sample by target success, joint validity, target margin, then L2.

Suggested flags:

```text
--adaptive-pgd
--checkpoint-interval 10
--rho 0.75
--min-alpha 1e-4
```

Expected effect:

- Reduces wasted runs from a fixed step size.
- Gives a thesis-friendly comparison to APGD-style literature.

### 3. Add Target-Aware Restarts

The current restart pool uses:

```text
encoded, jitter, gmm, jitter, gmm
```

The GMM prior is source-class latent space. Add target-aware seeds:

```text
encoded, jitter, source_gmm, benign_anchor, boundary_anchor
```

Possible target-aware seed types:

- `benign_anchor`: nearest Benign latent neighbor decoded/re-encoded into the source VAE, then clipped to the source epsilon ball.
- `benign_gmm`: sample from a Benign prior, map through a cross-class bridge if available.
- `boundary_anchor`: latent points from previous failed runs with high Benign margin but not yet target-successful.
- `successful_archive`: replay latent starts from earlier successful joint-valid targeted-benign examples for the same source class/model.

Expected effect:

- Helps the search start near regions that already have Benign evidence.
- Especially useful for `Recon`, `Spoofing`, and `Web`, where current successes mostly come from these classes.

Implementation target:

- `src\attack\run_targeted_benign_latent_pgd.py`
- `src\attack\latent_restarts.py` if generalized.

### 4. Use Class-Specific Targeted Budgets

The targeted-benign runner currently uses global:

```text
--epsilon 0.5
--alpha 0.05
```

But the non-targeted runner already uses class-specific latent radii. Add the same idea here.

Initial sweep:

| Class | Epsilon candidates |
| --- | --- |
| BruteForce | 0.3, 0.5, 0.8 |
| DDoS | 0.5, 0.8, 1.0, 1.2 |
| DoS | 0.5, 0.8, 1.0, 1.2 |
| Mirai | 0.5, 0.8, 1.0, 1.2 |
| Recon | 0.5, 0.8 |
| Spoofing | 0.5, 0.8 |
| Web | 0.5, 0.8, 1.0 |

Suggested flags:

```text
--epsilon-by-class "BruteForce=0.5,DDoS=0.8,DoS=0.8,Mirai=0.8,Recon=0.5,Spoofing=0.5,Web=1.0"
--alpha-ratio 0.1
```

Expected effect:

- More fair attack budget across classes.
- Useful to determine whether zero-success source classes are budget-limited or structurally hard.

### 5. Improve Candidate Selection

Current selection policy:

1. Target success.
2. Joint validity.
3. Lower input L2.

Add target margin before L2:

1. Target success.
2. Joint validity.
3. Higher Benign target margin.
4. Lower input L2.
5. Lower latent L2.

Also save these fields in per-sample outputs:

```text
benign_logit
max_other_logit
benign_margin
target_confidence
best_restart_label
```

Expected effect:

- Better selection among failed candidates.
- Better warm starts for future runs.
- Clearer per-class failure analysis.

### 6. Add Targeted Latent CW

Implement a targeted latent CW attack parallel to the current targeted PGD.

Objective:

```text
minimize latent_l2 + lambda_conf * max(max_other_logit - benign_logit + kappa, 0)
```

Selection:

- Target success first.
- Joint validity second.
- Higher target margin third.
- Lower input L2 fourth.

Suggested flags:

```text
--attacks targeted-pgd,targeted-cw
--lambda-conf 1.0
--kappa 0.0
--num-iterations 200
--learning-rate 0.01
```

Expected effect:

- CW often handles targeted attacks better than PGD.
- Provides a clean thesis comparison against the current targeted PGD baseline.

### 7. Add a Constrained Feature-Space Polish Step

After latent PGD/CW produces `x_adv`, run a small local search over mutable features only.

Constraints:

- Preserve perturbation mask.
- Preserve protocol and protocol indicators.
- Re-run raw G1-G8 validity after every candidate.
- Accept only candidates that improve Benign margin or target success while staying jointly valid.

Candidate methods:

- Coordinate search over top-k mutable gradient features.
- CAPGD-inspired adaptive constrained PGD.
- Small random search around decoded values with projection/repair.

Expected effect:

- Can recover valid Benign target success from near-boundary latent outputs.
- Especially helpful when decoder outputs are valid but not classifier-optimal.

### 8. Train a Shared CVAE or Cross-Class Latent Bridge

This is the larger architectural upgrade.

Problem with current per-class VAEs:

- Each class has its own latent space.
- "Move toward Benign" is not geometrically meaningful inside a source-class VAE.
- Benign priors cannot be directly sampled in a source VAE latent coordinate system.

Potential solutions:

- Train one shared CVAE conditioned on class label.
- Train a cross-class latent translator from source latent codes to Benign-like latent codes.
- Train a joint VAE with class embeddings and class-conditional decoder heads.

Expected effect:

- Makes target-conditioned search natural.
- Allows counterfactual attacks such as: preserve source identity constraints while optimizing under a Benign condition.

Risk:

- Larger implementation and retraining cost.
- Requires careful validation that the shared model does not weaken class-specific reconstruction quality.

## Experiment Ladder

Run these in order so each improvement is attributable.

### E0: Baseline Reproduction

Current rerun:

```powershell
python src\attack\run_targeted_benign_new_vae_methods.py `
  --methods gaussian,laplace `
  --models all `
  --samples-per-class 100 `
  --num-restarts 5 `
  --restart-strategy encoded+jitter+gmm `
  --force-refit-gmm
```

### E1: Margin Loss

```powershell
python src\attack\run_targeted_benign_new_vae_methods.py `
  --methods gaussian,laplace `
  --models all `
  --samples-per-class 100 `
  --num-restarts 5 `
  --restart-strategy encoded+jitter+gmm `
  --target-loss cw-margin `
  --kappa 0.0
```

### E2: Adaptive Targeted PGD

```powershell
python src\attack\run_targeted_benign_new_vae_methods.py `
  --methods gaussian,laplace `
  --models all `
  --samples-per-class 100 `
  --num-restarts 5 `
  --restart-strategy encoded+jitter+gmm `
  --target-loss cw-margin `
  --adaptive-pgd `
  --checkpoint-interval 10 `
  --rho 0.75 `
  --min-alpha 1e-4
```

### E3: More Steps and Restarts

```powershell
python src\attack\run_targeted_benign_new_vae_methods.py `
  --methods gaussian,laplace `
  --models all `
  --samples-per-class 100 `
  --num-steps 100 `
  --num-restarts 10 `
  --restart-strategy encoded+jitter+gmm `
  --target-loss cw-margin `
  --adaptive-pgd
```

### E4: Class-Specific Budget Sweep

Start with:

```powershell
python src\attack\run_targeted_benign_new_vae_methods.py `
  --methods gaussian,laplace `
  --models all `
  --samples-per-class 100 `
  --num-restarts 10 `
  --restart-strategy encoded+jitter+gmm `
  --target-loss cw-margin `
  --adaptive-pgd `
  --epsilon-by-class "BruteForce=0.5,DDoS=0.8,DoS=0.8,Mirai=0.8,Recon=0.5,Spoofing=0.5,Web=1.0"
```

Then widen only classes whose best target margins improve but do not cross zero.

### E5: Target-Aware Restarts

```powershell
python src\attack\run_targeted_benign_new_vae_methods.py `
  --methods gaussian,laplace `
  --models all `
  --samples-per-class 100 `
  --num-restarts 10 `
  --restart-strategy encoded+jitter+gmm+benign_anchor+boundary_anchor `
  --target-loss cw-margin `
  --adaptive-pgd
```

### E6: Targeted Latent CW

```powershell
python src\attack\run_targeted_benign_new_vae_methods.py `
  --methods gaussian,laplace `
  --models all `
  --samples-per-class 100 `
  --attacks targeted-pgd,targeted-cw `
  --num-restarts 10 `
  --restart-strategy encoded+jitter+gmm+benign_anchor `
  --lambda-conf 1.0 `
  --kappa 0.0
```

### E7: Latent Plus Constrained Polish

```powershell
python src\attack\run_targeted_benign_new_vae_methods.py `
  --methods gaussian,laplace `
  --models all `
  --samples-per-class 100 `
  --num-restarts 10 `
  --restart-strategy encoded+jitter+gmm+benign_anchor `
  --target-loss cw-margin `
  --adaptive-pgd `
  --polish constrained-coordinate `
  --polish-steps 25 `
  --polish-top-k 10
```

## Metrics to Add

Add to `summary.csv`, `per_class.csv`, and `per_sample_results.csv` where applicable:

```text
config_id
attack_objective
target_loss
kappa
lambda_latent_l2
repair_in_loop
active_dim_mode
active_dim_topk
active_dims
n_restarts
attempts_per_sample
benign_target_success_rate
benign_target_success_joint_rate
benign_target_success_given_joint_valid
raw_attack_to_benign_rate_pre_repair
post_repair_benign_rate
survives_repair_and_valid_rate
final_asr_valid
best_benign_margin_mean
best_benign_margin_median
best_benign_margin_p10
best_benign_margin_p90
near_miss_rate_margin_gt_-0.5
near_miss_rate_margin_gt_-0.1
selected_restart_label_counts
selected_restart_mean
successful_joint_valid_count
joint_validity_rate
idsr
mean_l2_input
mean_l2_latent
```

Near-miss rates are important because they tell whether a change moves failures toward the Benign boundary even before target success improves.

## Required Evaluation Output

For every `(model x source category x config)` combination, emit:

- Raw attack success: pre-repair attack-to-Benign rate.
- Repair survival: among raw Benign predictions, the rate that remains Benign after repair.
- Valid survival: among raw Benign predictions, the rate that is protocol-valid, mask-valid, raw G1-G8 valid, and joint-valid after repair.
- Final `ASR_valid`: predicted `Benign` and protocol-valid after the full pipeline, with joint validity reported next to it.
- Mean latent L2 perturbation.
- Mean input-space L2 perturbation.
- Attempts per sample for multi-start honesty.
- Selected restart labels and counts.
- Per-class active dimensions used.
- Seed and exact config flags.

The headline metric should be:

```text
final_asr_valid = attack_to_benign AND protocol_valid after the full repair/validation path
```

Joint validity should remain adjacent to the headline:

```text
joint_valid = protocol_valid AND mask_valid AND raw_g1g8_valid
```

Do not report only aggregate results. The thesis table must include per-category breakdowns so hard classes such as `Mirai` cannot be hidden by easier categories such as `Spoofing` or `Recon`.

## Funnel Reporting

Emit a funnel table and funnel plot showing where samples drop out:

```text
selected_correct_source_samples
decoded_attack_candidates
raw_attack_to_benign
post_repair_still_benign
protocol_valid
mask_compliant
raw_g1g8_valid
joint_valid
final_valid_benign
```

Required outputs:

- `funnel_by_model_category.csv`
- `funnel_by_model_category.png` or `.pdf`
- `kappa_sweep_valid_asr.csv` for kappa sweeps
- `kappa_sweep_valid_asr.png` or `.pdf`
- Updated `summary.csv`
- Updated `per_class.csv`
- Updated `per_sample_results.csv`
- Run-local `config_snapshot.json`
- Run-local or root manifest update containing all P3 flags and seed values

Keep PGD and C&W baselines in the same table for the 0% contrast:

```text
baseline_pgd
baseline_cw
targeted_pgd_ce
targeted_pgd_cw_margin
targeted_cw_margin
```

## Failure Analysis Plots

Recommended plots:

- Per-source-class Benign margin histograms before and after attack.
- Best Benign margin by restart type.
- Joint target success by source class and model.
- Joint validity by source class and epsilon.
- Target success vs input L2 scatter.
- Target success vs latent L2 scatter.
- Near-miss count by source class.

These plots can explain whether the bottleneck is optimization, budget, validity, or true source/target manifold separation.

## Acceptance Criteria

A change is worth keeping if it improves one of these without harming validity:

- Primary: higher joint target success.
- Secondary: higher target success given joint validity.
- Secondary: better best Benign margin on failed samples.
- Guardrail: joint validity should not drop materially.
- Guardrail: IDSR should remain comparable unless explicitly studying out-of-distribution targeted attacks.
- Guardrail: per-class results should not hide a collapse in one source class behind gains in another.

Suggested minimum meaningful improvement:

- Absolute +1.0 percentage point joint target success on at least three models, or
- Absolute +2.0 percentage points on one difficult model/source group, or
- A large near-miss margin improvement that justifies another optimization pass.

## Thesis Positioning

If these improvements only modestly increase success, that is still useful. The thesis argument can be:

1. Input-space attacks achieve high raw ASR but fail physical/semantic validity.
2. Latent VAE-constrained attacks preserve validity and achieve non-trivial non-targeted ASR.
3. Targeted-Benign latent attacks are much harder because they require valid source-like traffic to cross specifically into the Benign decision region.
4. Stronger target-aware objectives improve search, but low success reveals that valid source manifolds and Benign decision regions have limited overlap under realistic constraints.

This is a stronger story than simply chasing high raw ASR.

## Recommended Implementation Order

1. Orient by reading the current attack, decode, repair, validator, and metrics code.
2. Summarize current wiring and exact hook points before code edits.
3. Implement Upgrade 1: Benign-targeted CW-style latent loss behind flags.
4. Implement Upgrade 2: kappa sweep utility and valid_ASR-vs-kappa outputs.
5. Run upgrades 1-2 only and review results before proceeding.
6. After approval, implement Upgrade 3: repair-in-the-loop behind `--repair-in-loop`.
7. Implement Upgrade 4: active latent dimension restriction.
8. Implement Upgrade 5: multi-start best-of-N latent initialization with attempts logging.
9. Implement Upgrade 6: per-class attack tuning.
10. Consider shared CVAE only after the required P3 upgrades are measured.

## Workflow Gate

Do not implement all upgrades in one blind pass.

First checkpoint:

- Upgrade 1 complete.
- Upgrade 2 complete.
- Baseline CE/old objective still runnable.
- Kappa sweep run with fixed seed.
- CSV and plot produced.
- Per-model and per-category valid_ASR-vs-kappa reviewed.
- No validator, mask, protocol allowlist, repair, preprocessing, scaler, split, or schema changes.

Proceed to upgrades 3-6 only after this checkpoint is reviewed.

## Source Links

- On-manifold adversarial examples, David Stutz: https://davidstutz.de/on-manifold-adversarial-examples/
- Carlini and Wagner targeted attacks: https://arxiv.org/abs/1608.04644
- AutoAttack/APGD motivation: https://arxiv.org/abs/2003.01690
- Generative latent search for unrestricted adversarial examples: https://arxiv.org/abs/1805.07894
- Constrained adaptive attack for tabular data: https://arxiv.org/abs/2406.00775
- Constrained adaptive attacks and robust training for tabular data: https://arxiv.org/abs/2311.04503
- Adversarial realism for IoT intrusion detection: https://link.springer.com/article/10.1007/s12243-023-00953-y
