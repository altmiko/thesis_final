# Attack Rerun Report - 2026-06-02 Fix Pass

Workspace: `D:\thesis_final`
Branch/head: `version2` / `e95464a`
Log root: `outputs\attack_rerun_logs_20260602_014531_fixes`

## Fixes Understood

`fix.md` marks the VAE/latent attack fixes as resolved for:

- `decode_to_39` mode validation.
- VAE checkpoint loading guard in `AttackRouter.get_vae`.
- Targeted latent-CW objective and success test.
- Latent-CW default restart jitter.
- `encode` guard requiring registered protocol references.
- Latent-CW hard-decode consistency for inner best-delta tracking.
- Dead `get_vae` hparam arity branch removal.

Git note: these fixes are present in the working tree but not committed in Git at the time of rerun.

## Verification

- `python -m compileall -q src tests`: passed.
- `python -m pytest ...`: not run because `pytest` is not installed in the active conda env.
- CUDA device used: NVIDIA GeForce RTX 4080 SUPER.

## Rerun Inventory

| Pipeline | Status | Main output |
| --- | --- | --- |
| Input-space FGSM/PGD/CW for binary, 8-class, 34-class | Complete | `results\attacks\attack_summary.csv` |
| 8-class MLP/LSTM/CNN-LSTM restart attacks | Complete | `results\attacks\attack_summary_8class_models_restarts.csv` |
| Raw-space validity/shock analysis | Complete | `results\attacks\shock_table.csv`, `results\attacks\violation_breakdown.csv` |
| Canonical all-model Gaussian latent/input rerun | Complete | `outputs\latent_attacks\all_models_rerun_20260602_015314_seed42` |
| Gaussian VAE latent-only rerun | Complete | `outputs\latent_attacks\new_vae_attacks_gaussian_anticollapse_beta05_freebits01_20260529_173512_20260602_020809_seed42` |
| Laplace VAE latent-only rerun | Complete | `outputs\latent_attacks\new_vae_attacks_laplace_rerun_20260529_20260602_022105_seed42` |
| Targeted-benign Gaussian/Laplace PGD | Complete | `outputs\latent_attacks\targeted_benign_pgd_*_20260602_*_seed42` |
| Bootstrap/per-category/McNemar stats | Complete | `stats\` under the all-model, Gaussian, and Laplace rerun dirs |
| Non-targeted sample export | Complete | `outputs\latent_attacks\sample_exports\new_vae_attack_samples_20260602_023834` |
| Targeted-benign sample export | Complete | `outputs\latent_attacks\sample_exports\targeted_benign_attack_samples_20260602_023831` |
| Phase 0/1/2/3/4/A scripts | Complete | `outputs\latent_attacks\phase*_20260602_*_seed42` |
| Constrained-input PGD/CW baselines | Complete | `outputs\latent_attacks\constrained_input_baselines_20260602_024104_seed42` |

## Baseline Input-Space Attacks

Source: `results\attacks\attack_summary.csv`

| Task | Clean acc | Best FGSM ASR | Best PGD ASR | CW ASR |
| --- | ---: | ---: | ---: | ---: |
| Binary | 96.77% | 28.13% | 37.80% | 13.99% |
| 8-class | 84.57% | 43.06% | 74.69% | 27.75% |
| 34-class | 74.85% | 69.21% | 88.42% | 42.12% |

## Validity Analysis

Source: `results\attacks\shock_table.csv`, `results\attacks\violation_breakdown.csv`

- Clean validity: 100%.
- FGSM/PGD valid ASR remains 0.00% for binary, 8-class, and 34-class.
- CW valid ASR remains effectively zero: binary CW has 1 valid successful sample, ASR_valid 0.0021%; 8-class and 34-class CW are 0.00%.
- Top violation rates: `R_protocol_valid` 92.21%, `R_binary_SSH` 90.61%, `R_binary_ICMP` 90.47%, `R_binary_HTTP` 90.36%.

## Gaussian Latent/Input Rerun

Source: `outputs\latent_attacks\all_models_rerun_20260602_015314_seed42\summary.csv`

| Model | Latent-PGD ASR / Joint | Latent-CW ASR / Joint | Input-PGD ASR / Joint | Input-CW ASR / Joint |
| --- | ---: | ---: | ---: | ---: |
| CNN | 32.00% / 94.20% | 33.20% / 95.20% | 91.20% / 0.00% | 88.40% / 0.00% |
| CNN-LSTM | 40.50% / 94.00% | 41.17% / 96.17% | 95.00% / 0.00% | 97.33% / 0.00% |
| DualPath | 36.57% / 85.71% | 31.71% / 87.71% | 95.14% / 0.00% | 80.57% / 0.00% |
| LSTM | 37.29% / 87.14% | 39.71% / 90.29% | 97.14% / 0.00% | 95.71% / 0.00% |
| MLP | 38.29% / 87.43% | 37.00% / 88.43% | 95.43% / 0.00% | 89.86% / 0.00% |

McNemar PGD vs CW: only DualPath significant (`p=0.012359`).

## New VAE Latent-Only Reruns

Sources:

- Gaussian: `outputs\latent_attacks\new_vae_attacks_gaussian_anticollapse_beta05_freebits01_20260529_173512_20260602_020809_seed42\summary.csv`
- Laplace: `outputs\latent_attacks\new_vae_attacks_laplace_rerun_20260529_20260602_022105_seed42\summary.csv`

### Gaussian

| Model | Latent-PGD ASR / Joint | Latent-CW ASR / Joint |
| --- | ---: | ---: |
| CNN | 32.00% / 94.20% | 33.20% / 95.20% |
| CNN-LSTM | 40.50% / 94.00% | 41.17% / 96.17% |
| DualPath | 36.57% / 85.71% | 31.71% / 87.71% |
| LSTM | 37.29% / 87.14% | 39.71% / 90.29% |
| MLP | 38.29% / 87.43% | 37.00% / 88.43% |

### Laplace

| Model | Latent-PGD ASR / Joint | Latent-CW ASR / Joint |
| --- | ---: | ---: |
| CNN | 31.40% / 93.80% | 31.00% / 94.60% |
| CNN-LSTM | 37.33% / 82.33% | 40.00% / 81.33% |
| DualPath | 34.57% / 80.71% | 31.86% / 82.86% |
| LSTM | 34.86% / 84.86% | 39.00% / 86.00% |
| MLP | 34.29% / 83.43% | 39.00% / 84.14% |

Laplace McNemar significant for CNN-LSTM, DualPath, LSTM, and MLP; CNN not significant.

## Targeted-Benign PGD

Sources:

- Gaussian: `outputs\latent_attacks\targeted_benign_pgd_gaussian_anticollapse_beta05_freebits01_20260529_173512_20260602_023404_seed42\summary.csv`
- Laplace: `outputs\latent_attacks\targeted_benign_pgd_laplace_rerun_20260529_20260602_023539_seed42\summary.csv`

| Model | Gaussian target / joint-target | Laplace target / joint-target |
| --- | ---: | ---: |
| CNN | 5.00% / 5.00% | 6.20% / 5.80% |
| CNN-LSTM | 3.83% / 3.83% | 3.50% / 3.17% |
| DualPath | 4.43% / 4.14% | 4.00% / 3.57% |
| LSTM | 5.29% / 4.86% | 4.86% / 4.29% |
| MLP | 4.14% / 3.43% | 4.14% / 3.43% |

The targeted sample export selected 20 examples, all target-successful and joint-valid.

## Constrained-Input Baselines

Source: `outputs\latent_attacks\constrained_input_baselines_20260602_024104_seed42\summary.csv`

All constrained-input rows have 100% protocol validity, 100% mask compliance, 100% raw G1-G8 validity, and 100% joint validity. ASR is:

| Model | CInput-PGD ASR | CInput-CW ASR |
| --- | ---: | ---: |
| CNN | 51.60% | 69.00% |
| CNN-LSTM | 65.33% | 75.50% |
| DualPath | 53.57% | 76.71% |
| LSTM | 56.57% | 86.71% |
| MLP | 41.57% | 71.00% |

## Phase Checkpoints

| Phase | Output | Key result |
| --- | --- | --- |
| Phase 0 | `outputs\latent_attacks\phase0_20260602_023929_seed42` | Clean self-outlier rates near 4.59-6.06% by class. |
| Phase 1 | `outputs\latent_attacks\phase1_20260602_023939_seed42` | Mask round-trip pass, noisy mask compliance pass, protocol repair pass. |
| Phase 2 | `outputs\latent_attacks\phase2_20260602_023945_seed42` | Latent-PGD ASR 16.86%, protocol/mask 100%, IDSR 88.43%. |
| Phase 3 | `outputs\latent_attacks\phase3_20260602_023958_seed42` | Latent-CW ASR 21.43%, protocol/mask 100%, IDSR 88.86%. |
| Phase 4 | `outputs\latent_attacks\phase4_20260602_024026_seed42` | Input-PGD ASR 95.43%, Input-CW ASR 89.86%, both 0% joint-valid. |
| Phase A | `outputs\latent_attacks\phaseA_20260602_024038_seed42` | BruteForce p95 reconstruction error 1.9422; DoS p95 0.9988. |

## Bottom Line

The fix pass did not break the core thesis pattern:

- Unconstrained input attacks still achieve high raw ASR but collapse under validity filtering.
- Gaussian and Laplace VAE latent attacks retain nontrivial ASR while preserving protocol/mask validity.
- Targeted-benign attacks remain low-success but produce joint-valid Benign-targeted samples.
- Constrained-input PGD/CW now produce high valid ASR with 100% joint validity in this runner, making them a strong benchmark/ablation to discuss separately from unconstrained input-space attacks.
