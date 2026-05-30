# Attack Rerun Report - 2026-05-31

Workspace: `D:\thesis_final`

Rerun device: CUDA, seed 42 unless noted by the script default.

Log root: `outputs/attack_rerun_logs_20260531_002940`

## Executive Summary

All primary attack pipelines, targeted-benign attacks, statistical post-processing, sample exports, and the older phase/checkpoint attack scripts were rerun.

Main finding: unconstrained input-space attacks still produce high raw ASR, but validity filtering collapses those attacks to zero or near-zero valid ASR. The latent VAE-constrained attacks preserve protocol/mask validity and retain non-trivial ASR. The newer Gaussian and Laplace VAE reruns materially improve latent attack ASR over the older latent rerun, while targeted benign attacks remain low-success but do produce joint-valid Benign-targeted examples.

## Final Rerun Inventory

| Pipeline | Final status | Main output |
| --- | --- | --- |
| Input-space FGSM/PGD/CW for binary, 8-class, 34-class | Complete | `results/attacks/attack_summary.csv` |
| 8-class MLP/LSTM/CNN-LSTM restart attacks | Complete | `results/attacks/attack_summary_8class_models_restarts.csv` |
| Validity/shock analysis | Complete | `results/attacks/shock_table.csv`, `results/attacks/violation_breakdown.csv` |
| Original all-model latent/input rerun | Complete | `outputs/latent_attacks/all_models_rerun_20260531_005623_seed42` |
| New Gaussian VAE latent rerun | Complete | `outputs/latent_attacks/new_vae_attacks_gaussian_anticollapse_beta05_freebits01_20260529_173512_20260531_010241_seed42` |
| New Laplace VAE latent rerun | Complete | `outputs/latent_attacks/new_vae_attacks_laplace_rerun_20260529_20260531_012924_seed42` |
| Targeted-benign Gaussian/Laplace attacks | Complete | `outputs/latent_attacks/targeted_benign_pgd_*_20260531_*_seed42` |
| Bootstrap/per-class/McNemar statistics | Complete | `stats/` folders under each latent rerun |
| Non-targeted sample export | Complete | `outputs/latent_attacks/sample_exports/new_vae_attack_samples_20260531_014204` |
| Targeted-benign sample export | Complete | `outputs/latent_attacks/sample_exports/targeted_benign_attack_samples_20260531_014355` |
| Phase 0/1/2/3/4/A checkpoint scripts | Complete | `outputs/latent_attacks/phase*_20260531_*_seed42` |

Earlier attempts hit path, RNN, parser, and CUDA-memory issues. The final reruns completed after the fixes listed below.

## Rerun Fixes Applied

| File | Fix |
| --- | --- |
| `src/attack/run_attacks.py` | Corrected repo root resolution from `src` to repo root. |
| `src/attack/run_attacks_8class_models.py` | Corrected repo root resolution from `src` to repo root. |
| `src/evaluation/run_validity_analysis.py` | Corrected repo root resolution from `src` to repo root. |
| `src/evaluation/validity_analysis.py` | Updated attack filename parsing for restart suffixes such as `_r10`. |
| `src/attack/adversarial_attacks.py` | Enabled torchattacks training mode for CUDA RNN modules to avoid CuDNN RNN backward failures. |
| `src/attack/run_all_models_attack_rerun.py` | Batched correct-sample selection to avoid large CUDA allocations. |

## Baseline Input-Space Attacks

Source: `results/attacks/attack_summary.csv`

| Task | Clean acc | FGSM 0.05 | FGSM 0.10 | FGSM 0.30 | PGD 0.05 | PGD 0.10 | PGD 0.30 | CW |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Binary | 96.77% | 2.30% | 4.85% | 28.13% | 2.60% | 5.83% | 37.81% | 13.99% |
| 8-class | 84.57% | 12.25% | 18.29% | 43.06% | 13.83% | 23.93% | 74.73% | 27.75% |
| 34-class | 74.85% | 26.08% | 34.84% | 69.21% | 29.74% | 41.30% | 88.33% | 42.12% |

## 8-Class Neural Baseline Restarts

Source: `results/attacks/attack_summary_8class_models_restarts.csv`

| Model | Clean acc | FGSM 0.05 | FGSM 0.10 | FGSM 0.30 | PGD 0.05 | PGD 0.10 | PGD 0.30 | CW |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MLP | 84.75% | 12.22% | 18.34% | 42.68% | 14.19% | 26.18% | 77.59% | 28.02% |
| LSTM | 84.79% | 17.90% | 30.83% | 61.14% | 20.42% | 34.71% | 76.45% | 27.89% |
| CNN-LSTM | 84.17% | 13.39% | 27.01% | 41.53% | 22.09% | 53.83% | 94.25% | 43.04% |

## Validity Analysis

Source: `results/attacks/shock_table.csv`

Clean validity was 100.00%. For the main binary/8-class/34-class input attacks, FGSM and PGD had 0.00% valid ASR after validity filtering. CW was also effectively zero: binary CW produced 1 valid successful sample out of 48,383 originally correct samples, or 0.0021% valid ASR.

| Task | Attack family | Raw ASR range | Adv validity rate | Valid ASR |
| --- | --- | ---: | ---: | ---: |
| Binary | FGSM | 2.30-28.13% | 19.28% | 0.00% |
| Binary | PGD | 2.60-37.81% | 0.00% | 0.00% |
| Binary | CW | 13.99% | 64.34% | 0.0021% |
| 8-class | FGSM | 12.25-43.06% | 0.00% | 0.00% |
| 8-class | PGD | 13.83-74.73% | 0.00% | 0.00% |
| 8-class | CW | 27.75% | 35.02% | 0.00% |
| 34-class | FGSM | 26.08-69.21% | 0.00% | 0.00% |
| 34-class | PGD | 29.74-88.33% | 0.00% | 0.00% |
| 34-class | CW | 42.12% | 38.40% | 0.00% |

Top violation rates across the full processed adversarial set (`1,791,400` samples):

| Rule | Violation rate |
| --- | ---: |
| `R_protocol_valid` | 92.21% |
| `R_binary_SSH` | 90.61% |
| `R_binary_ICMP` | 90.47% |
| `R_binary_HTTP` | 90.36% |
| `R_binary_IPv` | 90.22% |
| `R_binary_ARP` | 90.12% |
| `R_binary_HTTPS` | 90.10% |

## Original Latent/Input All-Model Rerun

Source: `outputs/latent_attacks/all_models_rerun_20260531_005623_seed42/summary.csv`

`JVR` means the joint validity rate emitted by the rerun script. `IDSR` means the in-distribution rate from the Mahalanobis detector.

| Model | Latent-PGD ASR / JVR / IDSR | Latent-CW ASR / JVR / IDSR | Input-PGD ASR / JVR | Input-CW ASR / JVR |
| --- | ---: | ---: | ---: | ---: |
| CNN | 20.00% / 78.60% / 89.60% | 18.20% / 84.60% / 85.60% | 90.80% / 0.00% | 88.40% / 0.00% |
| CNN-LSTM | 16.50% / 80.67% / 89.00% | 22.83% / 89.67% / 92.67% | 95.00% / 0.00% | 97.33% / 0.00% |
| DualPath | 15.57% / 70.43% / 79.43% | 20.29% / 76.57% / 90.14% | 94.43% / 0.00% | 80.57% / 0.00% |
| LSTM | 17.29% / 70.71% / 82.86% | 18.57% / 77.14% / 90.29% | 96.57% / 0.00% | 95.71% / 0.00% |
| MLP | 16.86% / 72.29% / 88.43% | 21.43% / 76.86% / 88.86% | 95.29% / 0.00% | 89.86% / 0.00% |

McNemar PGD-vs-CW significance for this rerun:

| Model | p-value | Significant |
| --- | ---: | --- |
| CNN | 0.280713 | No |
| CNN-LSTM | 0.0000801 | Yes |
| DualPath | 0.001027 | Yes |
| LSTM | 0.342404 | No |
| MLP | 0.003119 | Yes |

## New VAE Latent Reruns

Sources:

- Gaussian: `outputs/latent_attacks/new_vae_attacks_gaussian_anticollapse_beta05_freebits01_20260529_173512_20260531_010241_seed42/summary.csv`
- Laplace: `outputs/latent_attacks/new_vae_attacks_laplace_rerun_20260529_20260531_012924_seed42/summary.csv`

### Gaussian VAE

| Model | Latent-PGD ASR / JVR | Latent-CW ASR / JVR |
| --- | ---: | ---: |
| CNN | 32.00% / 94.20% | 33.20% / 95.20% |
| CNN-LSTM | 40.50% / 94.00% | 41.17% / 96.17% |
| DualPath | 36.57% / 85.71% | 31.71% / 87.71% |
| LSTM | 37.29% / 87.14% | 39.71% / 90.29% |
| MLP | 38.29% / 87.43% | 37.00% / 88.43% |

McNemar significance: only DualPath was significant (`p=0.012359`); CNN, CNN-LSTM, LSTM, and MLP were not significant.

### Laplace VAE

| Model | Latent-PGD ASR / JVR | Latent-CW ASR / JVR |
| --- | ---: | ---: |
| CNN | 31.40% / 93.80% | 31.00% / 94.60% |
| CNN-LSTM | 37.33% / 82.33% | 40.00% / 81.33% |
| DualPath | 34.57% / 80.71% | 31.86% / 82.86% |
| LSTM | 34.86% / 84.86% | 39.00% / 86.00% |
| MLP | 34.29% / 83.43% | 39.00% / 84.14% |

McNemar significance: CNN was not significant (`p=0.877371`). CNN-LSTM (`p=0.033895`), DualPath (`p=0.048182`), LSTM (`p=0.000891`), and MLP (`p=0.000377`) were significant.

## Targeted-Benign Attacks

Sources:

- Gaussian: `outputs/latent_attacks/targeted_benign_pgd_gaussian_anticollapse_beta05_freebits01_20260529_173512_20260531_011821_seed42/summary.csv`
- Laplace: `outputs/latent_attacks/targeted_benign_pgd_laplace_rerun_20260529_20260531_011950_seed42/summary.csv`

### Gaussian Targeted-Benign

| Model | Target success | Joint target success | JVR | IDSR | Mean L2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| CNN | 5.00% | 5.00% | 98.40% | 91.60% | 4.1489 |
| CNN-LSTM | 3.83% | 3.83% | 96.83% | 93.83% | 4.8747 |
| DualPath | 4.43% | 4.14% | 85.14% | 91.43% | 8.4570 |
| LSTM | 5.29% | 4.86% | 89.29% | 93.43% | 10.0490 |
| MLP | 4.14% | 3.43% | 86.29% | 92.43% | 9.5753 |

### Laplace Targeted-Benign

| Model | Target success | Joint target success | JVR | IDSR | Mean L2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| CNN | 6.20% | 5.80% | 97.80% | 94.80% | 3.2264 |
| CNN-LSTM | 3.50% | 3.17% | 96.00% | 93.50% | 3.4220 |
| DualPath | 4.00% | 3.57% | 91.86% | 94.14% | 3.5341 |
| LSTM | 4.86% | 4.29% | 90.86% | 93.57% | 4.3241 |
| MLP | 4.14% | 3.43% | 91.86% | 93.14% | 3.5867 |

The targeted-benign sample export selected 20 examples, all target-successful and joint-valid:

`outputs/latent_attacks/sample_exports/targeted_benign_attack_samples_20260531_014355`

## Phase Checkpoint Reruns

These are older checkpoint scripts, but they were rerun for completeness.

| Phase | Output | Key result |
| --- | --- | --- |
| Phase 0 latent infrastructure | `outputs/latent_attacks/phase0_20260531_012656_seed42` | Clean Mahalanobis self-outlier rates stayed near 4.59-6.06% by class. |
| Phase 1 sanity checks | `outputs/latent_attacks/phase1_20260531_012725_seed42` | Mask round-trip pass, noisy mask compliance pass, protocol repair pass. |
| Phase 2 latent PGD vs MLP-3L | `outputs/latent_attacks/phase2_20260531_012753_seed42` | Overall ASR 16.86%, protocol 100.00%, mask 100.00%, IDSR 88.43%. |
| Phase 3 latent C&W vs MLP-3L | `outputs/latent_attacks/phase3_20260531_012807_seed42` | Overall ASR 21.43%, protocol 100.00%, mask 100.00%, IDSR 88.86%. |
| Phase 4 input baselines vs MLP-3L | `outputs/latent_attacks/phase4_20260531_012834_seed42` | Input-PGD ASR 95.43% and Input-C&W ASR 89.86%, both with 0.00% joint validity. |
| Phase A VAE diagnostics | `outputs/latent_attacks/phaseA_20260531_012849_seed42` | BruteForce p95 reconstruction error 1.9422; DoS p95 0.9988; DoS collapsed dims confirmed. |

## Sample Exports

| Export | Rows | Path |
| --- | ---: | --- |
| New VAE non-targeted attacks | 40 sample rows, 1,560 feature rows | `outputs/latent_attacks/sample_exports/new_vae_attack_samples_20260531_014204` |
| Targeted-benign attacks | 20 sample rows, 780 feature rows | `outputs/latent_attacks/sample_exports/targeted_benign_attack_samples_20260531_014355` |

## Important Notes

- Final reruns completed successfully. A few earlier logs show failed attempts; the corresponding `*_retry*` logs or later numbered logs are the successful ones.
- PyTorch `torch.load(weights_only=False)` and CUDA deterministic warnings appeared during some runs. They were nonfatal; later checkpoint reruns used warning filters so PowerShell would not mark successful runs as failed.
- The validity result is the main robustness distinction: input-space attacks can flip predictions easily but generally produce impossible traffic, while latent constrained attacks keep protocol and mask constraints intact.
- Generated rerun artifacts are large and many output folders are ignored by git. The key tracked result files changed under `results/attacks/`, `figures/`, and `tables/`.
