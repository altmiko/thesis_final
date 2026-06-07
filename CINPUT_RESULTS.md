# CINPUT Attack Results

This document consolidates the constrained-input (CINPUT) attack outputs found in the workspace. The canonical full run is:

`outputs\latent_attacks\constrained_input_baselines_20260602_032651_seed42`

It is the only CINPUT run here that includes both untargeted attacks and target-benign attacks across the evaluated models.

## Result Sources

| Artifact | Role |
| --- | --- |
| `outputs\latent_attacks\constrained_input_baselines_20260602_032651_seed42\summary.csv` | Canonical model-by-attack summary, 20 rows. |
| `outputs\latent_attacks\constrained_input_baselines_20260602_032651_seed42\per_class.csv` | Canonical source-class breakdown, 128 rows. |
| `outputs\latent_attacks\constrained_input_baselines_20260602_032651_seed42\per_sample_results.csv` | Per-sample metadata, 12,800 rows. |
| `outputs\latent_attacks\constrained_input_baselines_20260602_032651_seed42\skipped_cells.csv` | Source-class cells skipped because no correctly classified test samples were available. |
| `outputs\latent_attacks\constrained_input_baselines_20260602_032651_seed42\all_results.json` | JSON copy of summary and detailed records. |
| `outputs\latent_attacks\sample_exports\cinput_sample_summary.csv` | Selected inverse-transformed sample summaries, 8 rows. |
| `outputs\latent_attacks\sample_exports\cinput_featurewise_samples.csv` | Selected inverse-transformed feature tables, 312 rows. |
| `outputs\latent_attacks\sample_exports\cinput_target_benign_sample_summary.csv` | Target-benign selected sample summaries, 4 rows. |
| `outputs\latent_attacks\sample_exports\cinput_target_benign_featurewise_samples.csv` | Target-benign selected feature tables, 156 rows. |
| `c-attack_inverse_result.md` | Existing rendered inverse-transformed selected CINPUT sample exhibit. |
| `cinput_target-benign.md` | Existing rendered target-benign subset of the selected sample exhibit. |

## Run Inventory

| Run | Scope | Summary rows | Per-class rows | Per-sample rows | Notes |
| --- | --- | ---: | ---: | ---: | --- |
| `constrained_input_baselines_20260602_000036_seed42` | MLP smoke run, 5 samples per source class | 2 | 14 | 70 | Partial sanity run. |
| `constrained_input_baselines_20260602_000307_seed42` | Untargeted full-size run | 10 | 64 | 6,400 | Same summary and per-class hashes as `024104`. |
| `constrained_input_baselines_20260602_024104_seed42` | Untargeted full-size rerun | 10 | 64 | 6,400 | Referenced in `ATTACK_RERUN_REPORT_20260602_FIXES.md`. |
| `constrained_input_baselines_20260602_032651_seed42` | Untargeted plus target-benign full run | 20 | 128 | 12,800 | Canonical consolidated run for this file. |

## Setup

Sample setting: up to 100 correctly classified test samples per non-benign source class, skipping zero-available model/class cells.

Constraint set: `full+physics`. Final sample validity is measured after the differentiable constraint projection, not by raw postprocess forcing.

VAE run tag: `gaussian_anticollapse_beta05_freebits01_20260529_173512`.

| Attack | Parameters |
| --- | --- |
| `cinput-pgd` | epsilon=0.5, alpha=0.05, steps=40, random_start=True, targeted=False, constraint_projection=`full+physics` |
| `cinput-cw` | lambda_conf=1.0, kappa=0.0, iterations=200, learning_rate=0.01, convergence_threshold=1e-05, targeted=False, constraint_projection=`full+physics` |
| `cinput-pgd-target-benign` | epsilon=0.5, alpha=0.05, steps=40, random_start=True, targeted=True, target_class=`Benign`, constraint_projection=`full+physics` |
| `cinput-cw-target-benign` | lambda_conf=1.0, kappa=0.0, iterations=200, learning_rate=0.01, convergence_threshold=1e-05, targeted=True, target_class=`Benign`, constraint_projection=`full+physics` |

## Main Takeaways

- Every canonical CINPUT attack cell has 100.00% protocol validity, mask compliance, raw G1-G8 validity, and joint validity.
- Untargeted CINPUT-CW is the strongest CINPUT baseline overall: 2,439 successful joint-valid attacks out of 3,200 evaluated samples, or 76.22% weighted ASR.
- Untargeted CINPUT-PGD reaches 1,683 successful joint-valid attacks out of 3,200 evaluated samples, or 52.59% weighted ASR.
- Target-benign attacks are much harder: CINPUT-PGD target-benign reaches 4.16% weighted ASR, while CINPUT-CW target-benign reaches 10.78%.
- The strongest target-benign model/attack cell is DualPath with `cinput-cw-target-benign`: 161 successes out of 700 samples, or 23.00% ASR.
- Invalid ASR is 0.00% in the canonical run because successful CINPUT examples remain joint-valid under the measured constraints.

## Attack-Level Aggregate

Weighted ASR is computed from total successful joint-valid samples divided by total evaluated samples across models.

| Attack | N | Success | Weighted ASR | Model ASR range | Mean L2 Input |
| --- | ---: | ---: | ---: | ---: | ---: |
| `cinput-pgd` | 3,200 | 1,683 | 52.59% | 41.57%-65.83% | 13.0846 |
| `cinput-cw` | 3,200 | 2,439 | 76.22% | 69.00%-86.71% | 6.3893 |
| `cinput-pgd-target-benign` | 3,200 | 133 | 4.16% | 1.43%-11.43% | 18.3629 |
| `cinput-cw-target-benign` | 3,200 | 345 | 10.78% | 5.17%-23.00% | 15.8405 |

## Thesis-Style Summary Tables

| Attack family | Model | CINPUT PGD ASR | CINPUT PGD JVR | CINPUT C&W ASR | CINPUT C&W JVR |
| --- | --- | ---: | ---: | ---: | ---: |
| CINPUT | CNN | 54.00% | 100.00% | 69.00% | 100.00% |
| CINPUT | CNN-LSTM | 65.83% | 100.00% | 75.50% | 100.00% |
| CINPUT | DualPath | 47.86% | 100.00% | 76.71% | 100.00% |
| CINPUT | LSTM | 56.00% | 100.00% | 86.71% | 100.00% |
| CINPUT | MLP | 41.57% | 100.00% | 71.00% | 100.00% |

Table C.1: Non-targeted constrained-input PGD and C&W attack success rate and joint validity rate across classifiers.

| Attack family | Model | n | Target Success | ASRvalid | Joint Validity | IDSR | Mean L2 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CINPUT-PGD | CNN | 500 | 3.60% | 3.60% | 100.00% | 21.60% | 9.9877 |
| CINPUT-PGD | CNN-LSTM | 600 | 2.00% | 2.00% | 100.00% | 35.67% | 14.0350 |
| CINPUT-PGD | DualPath | 700 | 11.43% | 11.43% | 100.00% | 34.14% | 24.2250 |
| CINPUT-PGD | LSTM | 700 | 1.86% | 1.86% | 100.00% | 31.29% | 28.8169 |
| CINPUT-PGD | MLP | 700 | 1.43% | 1.43% | 100.00% | 40.43% | 14.7499 |

Table C.2: Targeted-benign constrained-input PGD target success and joint-valid target success across classifiers.

| Attack family | Model | n | Target Success | ASRvalid | Joint Validity | IDSR | Mean L2 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CINPUT-C&W | CNN | 500 | 12.80% | 12.80% | 100.00% | 26.60% | 6.1235 |
| CINPUT-C&W | CNN-LSTM | 600 | 5.17% | 5.17% | 100.00% | 41.67% | 9.9187 |
| CINPUT-C&W | DualPath | 700 | 23.00% | 23.00% | 100.00% | 36.29% | 21.4351 |
| CINPUT-C&W | LSTM | 700 | 7.43% | 7.43% | 100.00% | 35.43% | 25.1920 |
| CINPUT-C&W | MLP | 700 | 5.29% | 5.29% | 100.00% | 46.86% | 16.5334 |

Table C.3: Targeted-benign constrained-input C&W target success and joint-valid target success across classifiers.

## Model-by-Attack Summary

| Model | Attack | Goal | Target | N | Success | ASR | Valid ASR | Joint Valid | IDSR | L2 Input |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CNN | cinput-pgd | untargeted |  | 500 | 270 | 54.00% | 54.00% | 100.00% | 24.20% | 8.7226 |
| CNN | cinput-cw | untargeted |  | 500 | 345 | 69.00% | 69.00% | 100.00% | 57.00% | 3.6432 |
| CNN | cinput-pgd-target-benign | target-benign | Benign | 500 | 18 | 3.60% | 3.60% | 100.00% | 21.60% | 9.9877 |
| CNN | cinput-cw-target-benign | target-benign | Benign | 500 | 64 | 12.80% | 12.80% | 100.00% | 26.60% | 6.1235 |
| CNN-LSTM | cinput-pgd | untargeted |  | 600 | 395 | 65.83% | 65.83% | 100.00% | 40.83% | 11.0819 |
| CNN-LSTM | cinput-cw | untargeted |  | 600 | 453 | 75.50% | 75.50% | 100.00% | 63.17% | 4.8290 |
| CNN-LSTM | cinput-pgd-target-benign | target-benign | Benign | 600 | 12 | 2.00% | 2.00% | 100.00% | 35.67% | 14.0350 |
| CNN-LSTM | cinput-cw-target-benign | target-benign | Benign | 600 | 31 | 5.17% | 5.17% | 100.00% | 41.67% | 9.9187 |
| DualPath | cinput-pgd | untargeted |  | 700 | 335 | 47.86% | 47.86% | 100.00% | 34.29% | 17.8387 |
| DualPath | cinput-cw | untargeted |  | 700 | 537 | 76.71% | 76.71% | 100.00% | 61.29% | 8.3239 |
| DualPath | cinput-pgd-target-benign | target-benign | Benign | 700 | 80 | 11.43% | 11.43% | 100.00% | 34.14% | 24.2250 |
| DualPath | cinput-cw-target-benign | target-benign | Benign | 700 | 161 | 23.00% | 23.00% | 100.00% | 36.29% | 21.4351 |
| LSTM | cinput-pgd | untargeted |  | 700 | 392 | 56.00% | 56.00% | 100.00% | 35.29% | 13.5592 |
| LSTM | cinput-cw | untargeted |  | 700 | 607 | 86.71% | 86.71% | 100.00% | 59.14% | 8.8288 |
| LSTM | cinput-pgd-target-benign | target-benign | Benign | 700 | 13 | 1.86% | 1.86% | 100.00% | 31.29% | 28.8169 |
| LSTM | cinput-cw-target-benign | target-benign | Benign | 700 | 52 | 7.43% | 7.43% | 100.00% | 35.43% | 25.1920 |
| MLP | cinput-pgd | untargeted |  | 700 | 291 | 41.57% | 41.57% | 100.00% | 37.57% | 14.2206 |
| MLP | cinput-cw | untargeted |  | 700 | 497 | 71.00% | 71.00% | 100.00% | 61.71% | 6.3215 |
| MLP | cinput-pgd-target-benign | target-benign | Benign | 700 | 10 | 1.43% | 1.43% | 100.00% | 40.43% | 14.7499 |
| MLP | cinput-cw-target-benign | target-benign | Benign | 700 | 37 | 5.29% | 5.29% | 100.00% | 46.86% | 16.5334 |

## Skipped Cells

| Model | Model tag | Source class | Reason |
| --- | --- | --- | --- |
| CNN | cnn | BruteForce | no_correctly_classified_samples |
| CNN | cnn | Web | no_correctly_classified_samples |
| CNN-LSTM | serial | BruteForce | no_correctly_classified_samples |

## Per-Source-Class Breakdown

| Model | Attack | Source class | N | Success | ASR | Valid ASR | Joint Valid | IDSR | L2 Input |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CNN | cinput-pgd | DDoS | 100 | 32 | 32.00% | 32.00% | 100.00% | 20.00% | 1.7952 |
| CNN | cinput-pgd | DoS | 100 | 75 | 75.00% | 75.00% | 100.00% | 27.00% | 1.5329 |
| CNN | cinput-pgd | Mirai | 100 | 21 | 21.00% | 21.00% | 100.00% | 11.00% | 1.5885 |
| CNN | cinput-pgd | Recon | 100 | 71 | 71.00% | 71.00% | 100.00% | 41.00% | 9.3822 |
| CNN | cinput-pgd | Spoofing | 100 | 71 | 71.00% | 71.00% | 100.00% | 22.00% | 29.3140 |
| CNN | cinput-cw | DDoS | 100 | 56 | 56.00% | 56.00% | 100.00% | 51.00% | 1.6449 |
| CNN | cinput-cw | DoS | 100 | 66 | 66.00% | 66.00% | 100.00% | 84.00% | 0.5768 |
| CNN | cinput-cw | Mirai | 100 | 73 | 73.00% | 73.00% | 100.00% | 16.00% | 1.2071 |
| CNN | cinput-cw | Recon | 100 | 68 | 68.00% | 68.00% | 100.00% | 75.00% | 3.2248 |
| CNN | cinput-cw | Spoofing | 100 | 82 | 82.00% | 82.00% | 100.00% | 59.00% | 11.5623 |
| CNN | cinput-pgd-target-benign | DDoS | 100 | 0 | 0.00% | 0.00% | 100.00% | 18.00% | 1.8380 |
| CNN | cinput-pgd-target-benign | DoS | 100 | 0 | 0.00% | 0.00% | 100.00% | 11.00% | 1.5813 |
| CNN | cinput-pgd-target-benign | Mirai | 100 | 0 | 0.00% | 0.00% | 100.00% | 2.00% | 1.3389 |
| CNN | cinput-pgd-target-benign | Recon | 100 | 8 | 8.00% | 8.00% | 100.00% | 41.00% | 12.1357 |
| CNN | cinput-pgd-target-benign | Spoofing | 100 | 10 | 10.00% | 10.00% | 100.00% | 36.00% | 33.0445 |
| CNN | cinput-cw-target-benign | DDoS | 100 | 14 | 14.00% | 14.00% | 100.00% | 7.00% | 3.9162 |
| CNN | cinput-cw-target-benign | DoS | 100 | 17 | 17.00% | 17.00% | 100.00% | 2.00% | 3.1053 |
| CNN | cinput-cw-target-benign | Mirai | 100 | 0 | 0.00% | 0.00% | 100.00% | 0.00% | 1.8152 |
| CNN | cinput-cw-target-benign | Recon | 100 | 16 | 16.00% | 16.00% | 100.00% | 72.00% | 3.5253 |
| CNN | cinput-cw-target-benign | Spoofing | 100 | 17 | 17.00% | 17.00% | 100.00% | 52.00% | 18.2556 |
| CNN-LSTM | cinput-pgd | DDoS | 100 | 52 | 52.00% | 52.00% | 100.00% | 19.00% | 1.7511 |
| CNN-LSTM | cinput-pgd | DoS | 100 | 84 | 84.00% | 84.00% | 100.00% | 57.00% | 1.3418 |
| CNN-LSTM | cinput-pgd | Mirai | 100 | 28 | 28.00% | 28.00% | 100.00% | 2.00% | 1.4905 |
| CNN-LSTM | cinput-pgd | Recon | 100 | 71 | 71.00% | 71.00% | 100.00% | 36.00% | 18.9600 |
| CNN-LSTM | cinput-pgd | Spoofing | 100 | 77 | 77.00% | 77.00% | 100.00% | 32.00% | 20.1939 |
| CNN-LSTM | cinput-pgd | Web | 100 | 83 | 83.00% | 83.00% | 100.00% | 99.00% | 22.7543 |
| CNN-LSTM | cinput-cw | DDoS | 100 | 66 | 66.00% | 66.00% | 100.00% | 31.00% | 1.4922 |
| CNN-LSTM | cinput-cw | DoS | 100 | 83 | 83.00% | 83.00% | 100.00% | 86.00% | 0.4825 |
| CNN-LSTM | cinput-cw | Mirai | 100 | 48 | 48.00% | 48.00% | 100.00% | 6.00% | 1.5481 |
| CNN-LSTM | cinput-cw | Recon | 100 | 76 | 76.00% | 76.00% | 100.00% | 73.00% | 11.6553 |
| CNN-LSTM | cinput-cw | Spoofing | 100 | 95 | 95.00% | 95.00% | 100.00% | 84.00% | 13.4401 |
| CNN-LSTM | cinput-cw | Web | 100 | 85 | 85.00% | 85.00% | 100.00% | 99.00% | 0.3559 |
| CNN-LSTM | cinput-pgd-target-benign | DDoS | 100 | 0 | 0.00% | 0.00% | 100.00% | 25.00% | 1.8300 |
| CNN-LSTM | cinput-pgd-target-benign | DoS | 100 | 0 | 0.00% | 0.00% | 100.00% | 17.00% | 1.5802 |
| CNN-LSTM | cinput-pgd-target-benign | Mirai | 100 | 0 | 0.00% | 0.00% | 100.00% | 0.00% | 1.4041 |
| CNN-LSTM | cinput-pgd-target-benign | Recon | 100 | 0 | 0.00% | 0.00% | 100.00% | 47.00% | 14.6339 |
| CNN-LSTM | cinput-pgd-target-benign | Spoofing | 100 | 5 | 5.00% | 5.00% | 100.00% | 26.00% | 29.5146 |
| CNN-LSTM | cinput-pgd-target-benign | Web | 100 | 7 | 7.00% | 7.00% | 100.00% | 99.00% | 35.2470 |
| CNN-LSTM | cinput-cw-target-benign | DDoS | 100 | 1 | 1.00% | 1.00% | 100.00% | 12.00% | 7.9790 |
| CNN-LSTM | cinput-cw-target-benign | DoS | 100 | 2 | 2.00% | 2.00% | 100.00% | 3.00% | 1.3783 |
| CNN-LSTM | cinput-cw-target-benign | Mirai | 100 | 0 | 0.00% | 0.00% | 100.00% | 0.00% | 1.6808 |
| CNN-LSTM | cinput-cw-target-benign | Recon | 100 | 4 | 4.00% | 4.00% | 100.00% | 74.00% | 10.8940 |
| CNN-LSTM | cinput-cw-target-benign | Spoofing | 100 | 24 | 24.00% | 24.00% | 100.00% | 62.00% | 15.6722 |
| CNN-LSTM | cinput-cw-target-benign | Web | 100 | 0 | 0.00% | 0.00% | 100.00% | 99.00% | 21.9079 |
| DualPath | cinput-pgd | BruteForce | 100 | 74 | 74.00% | 74.00% | 100.00% | 48.00% | 17.9053 |
| DualPath | cinput-pgd | DDoS | 100 | 38 | 38.00% | 38.00% | 100.00% | 18.00% | 1.8391 |
| DualPath | cinput-pgd | DoS | 100 | 82 | 82.00% | 82.00% | 100.00% | 22.00% | 1.8587 |
| DualPath | cinput-pgd | Mirai | 100 | 14 | 14.00% | 14.00% | 100.00% | 2.00% | 1.6876 |
| DualPath | cinput-pgd | Recon | 100 | 78 | 78.00% | 78.00% | 100.00% | 26.00% | 14.8345 |
| DualPath | cinput-pgd | Spoofing | 100 | 49 | 49.00% | 49.00% | 100.00% | 24.00% | 46.0238 |
| DualPath | cinput-pgd | Web | 100 | 0 | 0.00% | 0.00% | 100.00% | 100.00% | 40.7220 |
| DualPath | cinput-cw | BruteForce | 100 | 100 | 100.00% | 100.00% | 100.00% | 57.00% | 14.0389 |
| DualPath | cinput-cw | DDoS | 100 | 70 | 70.00% | 70.00% | 100.00% | 46.00% | 1.4921 |
| DualPath | cinput-cw | DoS | 100 | 96 | 96.00% | 96.00% | 100.00% | 82.00% | 0.8306 |
| DualPath | cinput-cw | Mirai | 100 | 100 | 100.00% | 100.00% | 100.00% | 2.00% | 1.6433 |
| DualPath | cinput-cw | Recon | 100 | 90 | 90.00% | 90.00% | 100.00% | 75.00% | 15.7356 |
| DualPath | cinput-cw | Spoofing | 100 | 81 | 81.00% | 81.00% | 100.00% | 67.00% | 24.0575 |
| DualPath | cinput-cw | Web | 100 | 0 | 0.00% | 0.00% | 100.00% | 100.00% | 0.4692 |
| DualPath | cinput-pgd-target-benign | BruteForce | 100 | 39 | 39.00% | 39.00% | 100.00% | 15.00% | 30.8517 |
| DualPath | cinput-pgd-target-benign | DDoS | 100 | 0 | 0.00% | 0.00% | 100.00% | 31.00% | 1.8439 |
| DualPath | cinput-pgd-target-benign | DoS | 100 | 0 | 0.00% | 0.00% | 100.00% | 32.00% | 1.5663 |
| DualPath | cinput-pgd-target-benign | Mirai | 100 | 0 | 0.00% | 0.00% | 100.00% | 2.00% | 2.7055 |
| DualPath | cinput-pgd-target-benign | Recon | 100 | 33 | 33.00% | 33.00% | 100.00% | 42.00% | 20.9301 |
| DualPath | cinput-pgd-target-benign | Spoofing | 100 | 8 | 8.00% | 8.00% | 100.00% | 17.00% | 68.4927 |
| DualPath | cinput-pgd-target-benign | Web | 100 | 0 | 0.00% | 0.00% | 100.00% | 100.00% | 43.1849 |
| DualPath | cinput-cw-target-benign | BruteForce | 100 | 59 | 59.00% | 59.00% | 100.00% | 13.00% | 25.5290 |
| DualPath | cinput-cw-target-benign | DDoS | 100 | 4 | 4.00% | 4.00% | 100.00% | 7.00% | 7.4180 |
| DualPath | cinput-cw-target-benign | DoS | 100 | 6 | 6.00% | 6.00% | 100.00% | 28.00% | 11.5478 |
| DualPath | cinput-cw-target-benign | Mirai | 100 | 11 | 11.00% | 11.00% | 100.00% | 0.00% | 2.9936 |
| DualPath | cinput-cw-target-benign | Recon | 100 | 53 | 53.00% | 53.00% | 100.00% | 73.00% | 21.8097 |
| DualPath | cinput-cw-target-benign | Spoofing | 100 | 28 | 28.00% | 28.00% | 100.00% | 33.00% | 56.2454 |
| DualPath | cinput-cw-target-benign | Web | 100 | 0 | 0.00% | 0.00% | 100.00% | 100.00% | 24.5021 |
| LSTM | cinput-pgd | BruteForce | 100 | 59 | 59.00% | 59.00% | 100.00% | 74.00% | 8.0596 |
| LSTM | cinput-pgd | DDoS | 100 | 28 | 28.00% | 28.00% | 100.00% | 17.00% | 1.9823 |
| LSTM | cinput-pgd | DoS | 100 | 90 | 90.00% | 90.00% | 100.00% | 16.00% | 1.7483 |
| LSTM | cinput-pgd | Mirai | 100 | 17 | 17.00% | 17.00% | 100.00% | 1.00% | 1.4994 |
| LSTM | cinput-pgd | Recon | 100 | 71 | 71.00% | 71.00% | 100.00% | 27.00% | 12.5220 |
| LSTM | cinput-pgd | Spoofing | 100 | 64 | 64.00% | 64.00% | 100.00% | 14.00% | 27.4798 |
| LSTM | cinput-pgd | Web | 100 | 63 | 63.00% | 63.00% | 100.00% | 98.00% | 41.6232 |
| LSTM | cinput-cw | BruteForce | 100 | 100 | 100.00% | 100.00% | 100.00% | 76.00% | 5.7122 |
| LSTM | cinput-cw | DDoS | 100 | 72 | 72.00% | 72.00% | 100.00% | 27.00% | 1.9142 |
| LSTM | cinput-cw | DoS | 100 | 94 | 94.00% | 94.00% | 100.00% | 85.00% | 0.5663 |
| LSTM | cinput-cw | Mirai | 100 | 100 | 100.00% | 100.00% | 100.00% | 2.00% | 4.4389 |
| LSTM | cinput-cw | Recon | 100 | 62 | 62.00% | 62.00% | 100.00% | 75.00% | 18.7324 |
| LSTM | cinput-cw | Spoofing | 100 | 79 | 79.00% | 79.00% | 100.00% | 53.00% | 25.8163 |
| LSTM | cinput-cw | Web | 100 | 100 | 100.00% | 100.00% | 100.00% | 96.00% | 4.6216 |
| LSTM | cinput-pgd-target-benign | BruteForce | 100 | 9 | 9.00% | 9.00% | 100.00% | 32.00% | 14.8761 |
| LSTM | cinput-pgd-target-benign | DDoS | 100 | 0 | 0.00% | 0.00% | 100.00% | 24.00% | 1.9201 |
| LSTM | cinput-pgd-target-benign | DoS | 100 | 0 | 0.00% | 0.00% | 100.00% | 4.00% | 1.7247 |
| LSTM | cinput-pgd-target-benign | Mirai | 100 | 0 | 0.00% | 0.00% | 100.00% | 1.00% | 1.5594 |
| LSTM | cinput-pgd-target-benign | Recon | 100 | 2 | 2.00% | 2.00% | 100.00% | 42.00% | 27.6329 |
| LSTM | cinput-pgd-target-benign | Spoofing | 100 | 2 | 2.00% | 2.00% | 100.00% | 20.00% | 66.5429 |
| LSTM | cinput-pgd-target-benign | Web | 100 | 0 | 0.00% | 0.00% | 100.00% | 96.00% | 87.4618 |
| LSTM | cinput-cw-target-benign | BruteForce | 100 | 23 | 23.00% | 23.00% | 100.00% | 25.00% | 15.9497 |
| LSTM | cinput-cw-target-benign | DDoS | 100 | 10 | 10.00% | 10.00% | 100.00% | 8.00% | 6.8287 |
| LSTM | cinput-cw-target-benign | DoS | 100 | 0 | 0.00% | 0.00% | 100.00% | 0.00% | 3.1284 |
| LSTM | cinput-cw-target-benign | Mirai | 100 | 1 | 1.00% | 1.00% | 100.00% | 0.00% | 2.3123 |
| LSTM | cinput-cw-target-benign | Recon | 100 | 5 | 5.00% | 5.00% | 100.00% | 70.00% | 19.6544 |
| LSTM | cinput-cw-target-benign | Spoofing | 100 | 13 | 13.00% | 13.00% | 100.00% | 49.00% | 51.9603 |
| LSTM | cinput-cw-target-benign | Web | 100 | 0 | 0.00% | 0.00% | 100.00% | 96.00% | 76.5104 |
| MLP | cinput-pgd | BruteForce | 100 | 25 | 25.00% | 25.00% | 100.00% | 63.00% | 11.5177 |
| MLP | cinput-pgd | DDoS | 100 | 21 | 21.00% | 21.00% | 100.00% | 23.00% | 1.8358 |
| MLP | cinput-pgd | DoS | 100 | 82 | 82.00% | 82.00% | 100.00% | 23.00% | 1.7591 |
| MLP | cinput-pgd | Mirai | 100 | 17 | 17.00% | 17.00% | 100.00% | 1.00% | 1.6157 |
| MLP | cinput-pgd | Recon | 100 | 55 | 55.00% | 55.00% | 100.00% | 40.00% | 19.9454 |
| MLP | cinput-pgd | Spoofing | 100 | 59 | 59.00% | 59.00% | 100.00% | 23.00% | 22.1779 |
| MLP | cinput-pgd | Web | 100 | 32 | 32.00% | 32.00% | 100.00% | 90.00% | 40.6929 |
| MLP | cinput-cw | BruteForce | 100 | 76 | 76.00% | 76.00% | 100.00% | 88.00% | 5.5020 |
| MLP | cinput-cw | DDoS | 100 | 55 | 55.00% | 55.00% | 100.00% | 36.00% | 1.9927 |
| MLP | cinput-cw | DoS | 100 | 86 | 86.00% | 86.00% | 100.00% | 79.00% | 0.5909 |
| MLP | cinput-cw | Mirai | 100 | 99 | 99.00% | 99.00% | 100.00% | 2.00% | 1.6498 |
| MLP | cinput-cw | Recon | 100 | 77 | 77.00% | 77.00% | 100.00% | 72.00% | 8.8215 |
| MLP | cinput-cw | Spoofing | 100 | 76 | 76.00% | 76.00% | 100.00% | 63.00% | 12.8721 |
| MLP | cinput-cw | Web | 100 | 28 | 28.00% | 28.00% | 100.00% | 92.00% | 12.8217 |
| MLP | cinput-pgd-target-benign | BruteForce | 100 | 0 | 0.00% | 0.00% | 100.00% | 74.00% | 11.7218 |
| MLP | cinput-pgd-target-benign | DDoS | 100 | 0 | 0.00% | 0.00% | 100.00% | 20.00% | 1.9388 |
| MLP | cinput-pgd-target-benign | DoS | 100 | 0 | 0.00% | 0.00% | 100.00% | 13.00% | 1.5932 |
| MLP | cinput-pgd-target-benign | Mirai | 100 | 0 | 0.00% | 0.00% | 100.00% | 0.00% | 1.5657 |
| MLP | cinput-pgd-target-benign | Recon | 100 | 3 | 3.00% | 3.00% | 100.00% | 45.00% | 16.7423 |
| MLP | cinput-pgd-target-benign | Spoofing | 100 | 7 | 7.00% | 7.00% | 100.00% | 38.00% | 33.9848 |
| MLP | cinput-pgd-target-benign | Web | 100 | 0 | 0.00% | 0.00% | 100.00% | 93.00% | 35.7029 |
| MLP | cinput-cw-target-benign | BruteForce | 100 | 0 | 0.00% | 0.00% | 100.00% | 82.00% | 10.2142 |
| MLP | cinput-cw-target-benign | DDoS | 100 | 1 | 1.00% | 1.00% | 100.00% | 9.00% | 20.2889 |
| MLP | cinput-cw-target-benign | DoS | 100 | 0 | 0.00% | 0.00% | 100.00% | 12.00% | 6.3343 |
| MLP | cinput-cw-target-benign | Mirai | 100 | 0 | 0.00% | 0.00% | 100.00% | 2.00% | 1.9860 |
| MLP | cinput-cw-target-benign | Recon | 100 | 14 | 14.00% | 14.00% | 100.00% | 75.00% | 9.5525 |
| MLP | cinput-cw-target-benign | Spoofing | 100 | 21 | 21.00% | 21.00% | 100.00% | 56.00% | 20.5359 |
| MLP | cinput-cw-target-benign | Web | 100 | 1 | 1.00% | 1.00% | 100.00% | 92.00% | 46.8217 |

## Selected Inverse-Transformed Samples

The constrained-input runner saves full per-sample metadata, but not the full adversarial vectors for every sample. The selected inverse-transformed exhibits were regenerated from successful, joint-valid sample IDs, then exported under `outputs\latent_attacks\sample_exports`.

| Attack | Goal | Target | Model | Rank | Sample | Source -> After | Success | Joint Valid | L2 scaled | Top raw feature changes |
| --- | --- | --- | --- | ---: | ---: | --- | --- | --- | ---: | --- |
| cinput-pgd | untargeted |  | CNN-LSTM | 1 | 2 | DoS -> DDoS | True | True | 0.947775 | `Rate` 12375.9; `rst_flag_number` 0.3; `ack_count` 2.1; `ack_flag_number` 0.3; `syn_count` 0.3 |
| cinput-pgd | untargeted |  | MLP | 2 | 2 | DoS -> Recon | True | True | 1.034925 | `Header_Length` 6.04; `Number` -44; `rst_flag_number` 0.3; `fin_flag_number` 0.3; `ack_count` 2.1 |
| cinput-cw | untargeted |  | CNN | 1 | 12 | DoS -> DDoS | True | True | 0.018577 | `Rate` -333.004; `Number` 1; `Std` 0.836268; `Header_Length` 0.0404013; `Tot sum` 125.877 |
| cinput-cw | untargeted |  | MLP | 2 | 2 | DoS -> DDoS | True | True | 0.092210 | `psh_flag_number` 0.0862692; `Min` -0.138229; `Header_Length` -0.165438; `Number` 1; `syn_flag_number` 0.00906075 |
| cinput-pgd-target-benign | target-benign | Benign | MLP | 1 | 531 | Recon -> Benign | True | True | 0.815731 | `IAT` -0.000463576; `fin_count` -0.3; `ack_count` -2.1; `rst_count` 0.3; `ack_flag_number` -0.3 |
| cinput-pgd-target-benign | target-benign | Benign | CNN | 2 | 77 | Recon -> Benign | True | True | 1.034376 | `Header_Length` 6.04; `IAT` 0.000463576; `Min` 2.00317; `fin_count` -0.3; `ack_flag_number` -0.3 |
| cinput-cw-target-benign | target-benign | Benign | MLP | 1 | 27 | Spoofing -> Benign | True | True | 0.465821 | `Tot sum` 18163.2; `Variance` -9703.28; `Number` 13; `Max` 74.3638; `Header_Length` -0.419894 |
| cinput-cw-target-benign | target-benign | Benign | CNN-LSTM | 2 | 130 | Recon -> Benign | True | True | 0.485937 | `ack_flag_number` -0.299566; `rst_count` -0.245858; `ack_count` -1.16774; `Min` -0.807606; `Std` -28.0849 |

Full selected feature-level sample rows are in:

- `outputs\latent_attacks\sample_exports\cinput_featurewise_samples.csv`
- `outputs\latent_attacks\sample_exports\cinput_target_benign_featurewise_samples.csv`
- `c-attack_inverse_result.md`
- `cinput_target-benign.md`

## Raw Data Notes

- `per_sample_results.csv` contains all 12,800 canonical per-sample result records: `sample_id`, labels, predicted label, source class, success flags, validity flags, attack type, attack goal, target class, model name, and model tag.
- The per-sample records do not contain full adversarial vectors. The inverse-transformed sample exhibits were regenerated separately for selected successful joint-valid rows.
- The selected target-benign markdown is a strict subset of the combined inverse-transformed selected-sample markdown.
