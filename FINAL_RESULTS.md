# Final Results For Thesis Report

This file consolidates the result artifacts found in the repository. Each table/result includes the source location so it can be traced back to the original output.

## Coverage Notes

- ROC AUC / AUROC: no stored ROC AUC metric was found in the repository artifacts. Searches for `roc_auc`, `auroc`, `auc_score`, `roc auc`, and related terms did not find baseline ROC AUC outputs. Baseline reports provide accuracy, precision, recall, F1, support, and confusion matrices.
- Primary current validation results are in `results/validation/` and show 100% validity over the processed dataset. The older `tables/T4_clean_validity.*` files are historical/superseded and report a lower validity audit.
- Main test-set size for full baseline reports: 885,988 samples.
- Classifier tasks: binary, 8-class category, and 34-class fine-grained classification.

## Executive Thesis Results

| Area | Main Result | Source |
|---|---:|---|
| Processed rows | 4,429,940 total rows after cleaning | `config/run_manifest.json` |
| Split sizes | train 3,100,958; validation 442,994; test 885,988 | `config/run_manifest.json` |
| Feature count | 39 features | `config/run_manifest.json`, `tables/T1_feature_schema.csv` |
| Current dataset validity | 4,429,940 / 4,429,940 valid, 100.00% | `results/validation/full_dataset_validation_report.txt` |
| Best binary accuracy | DualPath, 96.78% | `results/all_models_all_tasks_summary.json` |
| Best binary macro F1 | MLP, 78.73% | `results/all_models_all_tasks_summary.json` |
| Best 8-class accuracy | DualPath, 84.59% | `results/all_models_all_tasks_summary.json` |
| Best 8-class macro F1 | LSTM, 64.70% | `results/all_models_all_tasks_summary.json` |
| Best 34-class accuracy | LSTM, 75.58% | `results/all_models_all_tasks_summary.json` |
| Best 34-class macro F1 | LSTM, 57.47% | `results/all_models_all_tasks_summary.json` |
| Strongest input attack, binary | PGD eps=0.30, ASR raw 37.81%, valid ASR 0.00% | `results/attacks/attack_summary.csv`, `results/attacks/shock_table.csv` |
| Strongest input attack, 8-class | PGD eps=0.30, ASR raw 74.73%, valid ASR 0.00% | `results/attacks/attack_summary.csv`, `results/attacks/shock_table.csv` |
| Strongest input attack, 34-class | PGD eps=0.30, ASR raw 88.33%, valid ASR 0.00% | `results/attacks/attack_summary.csv`, `results/attacks/shock_table.csv` |
| Latest improved Gaussian latent attacks | Non-targeted latent ASR about 31.71% to 41.17%, joint validity 85.71% to 96.17% | `outputs/latent_attacks/all_models_rerun_20260531_184251_seed42/summary.csv` |
| Improved Gaussian VAE training | All 8 classes: 100% pre/post validity, 0 collapsed dimensions | `results/vae/gaussian_anticollapse_beta05_freebits01_20260529_173512/summary.csv` |
| Laplace VAE training | All 8 classes: 100% pre/post validity, 0 collapsed dimensions | `results/vae/laplace_rerun_20260529/summary.csv` |
| Targeted benign latent attack success | Gaussian joint target success 3.43% to 5.00%; Laplace 3.17% to 5.80% | `outputs/latent_attacks/targeted_benign_*_seed42/summary.csv` |

## Dataset And Validation

### Run Manifest

Source: `config/run_manifest.json`

| Field | Value |
|---|---:|
| Timestamp | 2026-04-19T21:20:07.683345 |
| Schema | Modified Schema A, 39 features |
| Mode | SAMPLE |
| Majority sample cap | 200,000 |
| Medium sample cap | 200,000 |
| Minority sample cap | none |
| Rare sample cap | none |
| Rows loaded from parquet | 4,429,940 |
| Rows after cleaning | 4,429,940 |
| Train rows | 3,100,958 |
| Validation rows | 442,994 |
| Test rows | 885,988 |
| Number of features | 39 |
| 34-class labels | 34 |
| 8-class labels | 8 |
| Binary labels | 2 |
| Random seed | 42 |

### Category Distribution

Source: `tables/T2b_category_counts.csv`

| Category | Count | Percentage | Sub-attacks |
|---|---:|---:|---:|
| DDoS | 2,049,917 | 46.2741% | 12 |
| DoS | 668,775 | 15.0967% | 4 |
| Mirai | 599,960 | 13.5433% | 3 |
| Recon | 503,528 | 11.3665% | 5 |
| Spoofing | 371,451 | 8.3850% | 2 |
| Benign | 199,989 | 4.5145% | 1 |
| Web | 23,798 | 0.5372% | 6 |
| BruteForce | 12,522 | 0.2827% | 1 |

### Current Full Dataset Validity

Source: `results/validation/full_dataset_validation_report.txt`

| Split | Shape | Valid Rows | Validity |
|---|---:|---:|---:|
| Train | (3,100,958, 39) | 3,100,958 / 3,100,958 | 100.00% |
| Validation | (442,994, 39) | 442,994 / 442,994 | 100.00% |
| Test | (885,988, 39) | 885,988 / 885,988 | 100.00% |
| Total | 4,429,940 rows | 4,429,940 / 4,429,940 | 100.00% |

Additional validation artifacts:

| Artifact | Detail | Location |
|---|---|---|
| Per-class validity | 102 split/class rows, all validity rates 1.0 | `results/validation/per_class_validity.csv` |
| Per-rule breakdown | 49 rules, all failure rates 0.0 | `results/validation/per_rule_breakdown.csv` |
| Synthetic corruption tests | 9/9 tests passed | `results/validation/synthetic_corruption_tests.csv` |
| Clean data validity JSON | validity_rate 1.0 over 4,429,940 rows | `data/processed/clean_data_validity_report.json` |
| Processed data validity JSON | validity_rate 1.0 over 4,429,940 rows | `data/processed/processed_data_validity_report.json` |
| Historical clean-validity audit | older/superseded audit with 81.3724% validity | `tables/T4_clean_validity.md`, `tables/T4_clean_validity.csv` |

## Baseline Classifier Results

### All Models, All Tasks

Source: `results/all_models_all_tasks_summary.json`

| Task | Model | Accuracy | Macro F1 | Weighted F1 |
|---|---|---:|---:|---:|
| binary | MLP | 96.72% | 78.73% | 96.53% |
| binary | CNN | 96.59% | 76.40% | 96.26% |
| binary | LSTM | 96.74% | 78.40% | 96.51% |
| binary | Serial CNN-LSTM | 96.58% | 74.98% | 96.13% |
| binary | DualPath CNN-LSTM | 96.78% | 78.45% | 96.53% |
| 8class | MLP | 84.49% | 64.69% | 83.97% |
| 8class | CNN | 83.49% | 59.07% | 82.88% |
| 8class | LSTM | 84.57% | 64.70% | 84.13% |
| 8class | Serial CNN-LSTM | 83.94% | 60.69% | 83.58% |
| 8class | DualPath CNN-LSTM | 84.59% | 63.99% | 83.91% |
| 34class | MLP | 74.95% | 56.79% | 73.23% |
| 34class | CNN | 70.24% | 52.39% | 68.88% |
| 34class | LSTM | 75.58% | 57.47% | 74.16% |
| 34class | Serial CNN-LSTM | 74.44% | 56.28% | 72.93% |
| 34class | DualPath CNN-LSTM | 75.35% | 57.12% | 73.81% |

### Best Baseline By Metric

Source: `results/all_models_all_tasks_summary.json`

| Task | Best Accuracy | Best Macro F1 | Best Weighted F1 |
|---|---|---|---|
| binary | DualPath CNN-LSTM, 96.78% | MLP, 78.73% | DualPath CNN-LSTM, 96.53% |
| 8class | DualPath CNN-LSTM, 84.59% | LSTM, 64.70% | LSTM, 84.13% |
| 34class | LSTM, 75.58% | LSTM, 57.47% | LSTM, 74.16% |

### Baseline Detailed Reports

| Artifact | Detail | Location |
|---|---|---|
| Full classification reports | 15 JSON and 15 TXT reports with precision, recall, F1, support, and confusion matrices for all model/task pairs | `results/*_classification_report.json`, `results/*_classification_report.txt` |
| MLP all-task summary | Earlier/single-model MLP run: binary 96.724% acc / 78.725% macro F1; 8-class 84.402% / 64.528%; 34-class 72.991% / 55.237% | `results/mlp_all_tasks_summary.json` |
| Preliminary baseline logs | Small 8-category experiments, train/val/test 40,000/10,000/10,000, 2 epochs | `logs/baselines/run_20260422_184119/`, `logs/baselines/run_20260422_185341/` |
| Additional full-ish baseline logs | Includes test metrics, for example MLP 8-class accuracy 75.91% in that log run | `logs/baselines/run_20260422_184207/metrics_test.json` |

## Input-Space Attack Results

### Main Attack Summary

Source: `results/attacks/attack_summary.csv`

| Task | Attack | Epsilon | Clean Acc | ASR Raw | Flipped | Mean L_inf | Mean L2 |
|---|---|---:|---:|---:|---:|---:|---:|
| binary | FGSM | 0.05 | 96.77% | 2.30% | 1,111 | 0.040 | 0.252 |
| binary | FGSM | 0.10 | 96.77% | 4.85% | 2,345 | 0.081 | 0.504 |
| binary | FGSM | 0.30 | 96.77% | 28.13% | 13,611 | 0.242 | 1.512 |
| binary | PGD | 0.05 | 96.77% | 2.60% | 1,259 | 0.050 | 0.285 |
| binary | PGD | 0.10 | 96.77% | 5.83% | 2,820 | 0.100 | 0.566 |
| binary | PGD | 0.30 | 96.77% | 37.81% | 18,294 | 0.299 | 1.656 |
| binary | CW | N/A | 96.77% | 13.99% | 6,771 | 0.093 | 0.270 |
| 8class | FGSM | 0.05 | 84.57% | 12.25% | 5,178 | 0.050 | 0.312 |
| 8class | FGSM | 0.10 | 84.57% | 18.29% | 7,732 | 0.100 | 0.624 |
| 8class | FGSM | 0.30 | 84.57% | 43.06% | 18,208 | 0.300 | 1.873 |
| 8class | PGD | 0.05 | 84.57% | 13.83% | 5,847 | 0.050 | 0.305 |
| 8class | PGD | 0.10 | 84.57% | 23.93% | 10,119 | 0.100 | 0.601 |
| 8class | PGD | 0.30 | 84.57% | 74.73% | 31,602 | 0.300 | 1.744 |
| 8class | CW | N/A | 84.57% | 27.75% | 11,733 | 0.202 | 0.520 |
| 34class | FGSM | 0.05 | 74.85% | 26.08% | 9,760 | 0.050 | 0.312 |
| 34class | FGSM | 0.10 | 74.85% | 34.84% | 13,040 | 0.100 | 0.624 |
| 34class | FGSM | 0.30 | 74.85% | 69.21% | 25,902 | 0.300 | 1.873 |
| 34class | PGD | 0.05 | 74.85% | 29.74% | 11,131 | 0.050 | 0.305 |
| 34class | PGD | 0.10 | 74.85% | 41.30% | 15,456 | 0.100 | 0.599 |
| 34class | PGD | 0.30 | 74.85% | 88.33% | 33,060 | 0.300 | 1.741 |
| 34class | CW | N/A | 74.85% | 42.12% | 15,766 | 0.116 | 0.297 |

### Validity Shock Table Highlights

Source: `results/attacks/shock_table.csv`

| Task / Attack | ASR Raw | Clean Validity | Adversarial Validity | ASR Valid | Successful Valid |
|---|---:|---:|---:|---:|---:|
| binary PGD eps=0.30 | 37.81% | 100.00% | 0.00% | 0.00% | 0 |
| 8class PGD eps=0.30 | 74.73% | 100.00% | 0.00% | 0.00% | 0 |
| 34class PGD eps=0.30 | 88.33% | 100.00% | 0.00% | 0.00% | 0 |
| binary CW | 13.99% | 100.00% | very low | 0.0021% | 1 |

Main interpretation: input-space attacks can achieve high raw attack success, but the generated samples overwhelmingly violate protocol/physics constraints, collapsing valid attack success to zero or near zero.

### Top Constraint Violations

Source: `results/attacks/violation_breakdown.csv`

| Rule | Violation Rate |
|---|---:|
| R_protocol_valid | 92.2109% |
| R_binary_SSH | 90.6092% |
| R_binary_ICMP | 90.4691% |
| R_binary_HTTP | 90.3580% |
| R_binary_IPv | 90.2241% |
| R_binary_ARP | 90.1218% |
| R_binary_HTTPS | 90.0998% |
| R_binary_UDP | 90.0380% |
| R_binary_LLC | 90.0301% |
| R_binary_DNS | 89.7414% |
| R_binary_TCP | 89.4336% |
| R_var_eq_std_sq | 83.1088% |

Total adversarial samples in this breakdown: 1,791,400.

### 8-Class Multi-Model Attack Rerun

Source: `results/attacks/attack_summary_8class_models_restarts.csv`

| Model | Attack | Epsilon | Clean Acc | ASR Raw | Flipped | Restarts | Mean L2 |
|---|---|---:|---:|---:|---:|---:|---:|
| MLP | FGSM | 0.05 | 84.75% | 12.22% | 2,589 | 1 | 0.312 |
| MLP | FGSM | 0.10 | 84.75% | 18.51% | 3,922 | 1 | 0.624 |
| MLP | FGSM | 0.30 | 84.75% | 43.54% | 9,227 | 1 | 1.873 |
| MLP | PGD | 0.05 | 84.75% | 14.43% | 3,058 | 1 | 0.307 |
| MLP | PGD | 0.10 | 84.75% | 23.68% | 5,018 | 1 | 0.602 |
| MLP | PGD | 0.30 | 84.75% | 75.19% | 15,934 | 5 | 1.651 |
| MLP | CW | N/A | 84.75% | 27.33% | 5,793 | 5 | 0.513 |
| CNN | FGSM/PGD/CW | mixed | see source | see source | see source | see source | see source |
| LSTM | FGSM/PGD/CW | mixed | see source | see source | see source | see source | see source |
| DualPath | FGSM/PGD/CW | mixed | see source | see source | see source | see source | see source |
| CNN-LSTM | PGD | 0.30 | 84.17% | 94.25% | 19,833 | 10 | 1.257 |
| CNN-LSTM | CW | N/A | 84.17% | 43.04% | 9,056 | 10 | 0.419 |

The full 21-row table is available in the source CSV.

### Other Input Attack Artifacts

| Artifact | Detail | Location |
|---|---|---|
| Compact attack validity summary | 7 rows, 11 columns | `results/attacks/attack_summaries.csv` |
| Restart attack summary | 21 rows, 18 columns, includes perturbation norm statistics | `results/attacks/attack_restart_summary_8class.csv` |
| Thesis multimetric bundle | 20 model/attack rows with ASR, ASR_valid, protocol_valid, mask_valid, IDSR, L2_mean | `results/attacks/thesis_bundle.multimetric.csv` |
| Category ASR bundle | 160 rows; input PGD valid category ASR values are 0.0, latent entries are nonzero | `results/attacks/thesis_bundle.category_asr.csv` |
| Feature perturbations | 819 featurewise raw-delta rows | `results/attacks/feature_perturbation_8class.csv` |
| Full impossible-traffic exhibits | 4,196-line exhibit file | `results/attacks/sample_exhibits.txt` |
| Compact impossible-traffic exhibits | 390-line compact exhibit file | `results/attacks/compact_exhibits.txt` |
| Exhibit feature rows | 1,328 feature rows | `results/attacks/sample_exhibits_all.csv` |
| Impossible traffic table packs | Binary, 8-class, 34-class, and slide-pack CSV/MD artifacts | `tables/impossible_traffic_*` |

## Latent / VAE Attack Results

### Original / Older All-Model Latent Attack Rerun

Source: `outputs/latent_attacks/all_models_rerun_20260531_005623_seed42/summary.csv`

| Model | Attack | ASR | Joint Validity | IDSR |
|---|---|---:|---:|---:|
| CNN | latent-pgd | 20.00% | 78.60% | 89.60% |
| CNN | latent-cw | 18.20% | 84.60% | 85.60% |
| CNN | input-pgd | 90.80% | 0.00% | see source |
| CNN | input-cw | 88.40% | 0.00% | see source |
| CNN-LSTM | latent-pgd | 16.50% | 80.67% | see source |
| CNN-LSTM | latent-cw | 22.83% | 89.67% | see source |
| CNN-LSTM | input-pgd | 95.00% | 0.00% | see source |
| CNN-LSTM | input-cw | 97.33% | 0.00% | see source |
| DualPath | latent-pgd | 15.57% | 70.43% | see source |
| DualPath | latent-cw | 20.29% | 76.57% | see source |
| LSTM | latent-pgd | 17.29% | 70.71% | see source |
| LSTM | latent-cw | 18.57% | 77.14% | see source |
| MLP | latent-pgd | 16.86% | 72.29% | 88.43% |
| MLP | latent-cw | 21.43% | 76.86% | 88.86% |
| MLP | input-pgd | 95.29% | 0.00% | see source |
| MLP | input-cw | 89.86% | 0.00% | see source |

### Latest Improved Gaussian All-Model Rerun

Source: `outputs/latent_attacks/all_models_rerun_20260531_184251_seed42/summary.csv`

| Model | Attack | ASR | Joint Validity | Note |
|---|---|---:|---:|---|
| CNN | latent-pgd | 32.00% | 94.20% | improved Gaussian anti-collapse VAE |
| CNN | latent-cw | 33.20% | 95.20% | improved Gaussian anti-collapse VAE |
| CNN | input-pgd | 91.20% | 0.00% | invalid input-space baseline |
| CNN | input-cw | 88.40% | 0.00% | invalid input-space baseline |
| CNN-LSTM | latent-pgd | 40.50% | 94.00% | improved Gaussian anti-collapse VAE |
| CNN-LSTM | latent-cw | 41.17% | 96.17% | improved Gaussian anti-collapse VAE |
| CNN-LSTM | input-pgd | 95.00% | 0.00% | invalid input-space baseline |
| CNN-LSTM | input-cw | 97.33% | 0.00% | invalid input-space baseline |
| DualPath | latent-pgd | 36.57% | 85.71% | improved Gaussian anti-collapse VAE |
| DualPath | latent-cw | 31.71% | 87.71% | improved Gaussian anti-collapse VAE |
| DualPath | input-pgd | 95.14% | 0.00% | invalid input-space baseline |
| DualPath | input-cw | 80.57% | 0.00% | invalid input-space baseline |
| LSTM | latent-pgd | 37.29% | 87.14% | improved Gaussian anti-collapse VAE |
| LSTM | latent-cw | 39.71% | 90.29% | improved Gaussian anti-collapse VAE |
| LSTM | input-pgd | 97.14% | 0.00% | invalid input-space baseline |
| LSTM | input-cw | 95.71% | 0.00% | invalid input-space baseline |
| MLP | latent-pgd | 38.29% | 87.43% | improved Gaussian anti-collapse VAE |
| MLP | latent-cw | 37.00% | 88.43% | improved Gaussian anti-collapse VAE |
| MLP | input-pgd | 95.43% | 0.00% | invalid input-space baseline |
| MLP | input-cw | 89.86% | 0.00% | invalid input-space baseline |

### Improved Gaussian Non-Targeted Latent-Only Attacks

Source: `outputs/latent_attacks/new_vae_attacks_gaussian_anticollapse_beta05_freebits01_20260529_173512_20260531_010241_seed42/summary.csv`

| Model | latent-pgd ASR | latent-pgd Joint Validity | latent-cw ASR | latent-cw Joint Validity |
|---|---:|---:|---:|---:|
| CNN | 32.00% | 94.20% | 33.20% | 95.20% |
| CNN-LSTM | 40.50% | 94.00% | 41.17% | 96.17% |
| DualPath | 36.57% | 85.71% | 31.71% | 87.71% |
| LSTM | 37.29% | 87.14% | 39.71% | 90.29% |
| MLP | 38.29% | 87.43% | 37.00% | 88.43% |

Companion files:

| Artifact | Detail | Location |
|---|---|---|
| Per-class attack results | 64 rows, 20 columns | `outputs/latent_attacks/new_vae_attacks_gaussian_anticollapse_beta05_freebits01_20260529_173512_20260531_010241_seed42/per_class.csv` |
| Bootstrap stats | 10 rows | `outputs/latent_attacks/new_vae_attacks_gaussian_anticollapse_beta05_freebits01_20260529_173512_20260531_010241_seed42/stats/bootstrap_ci.csv` |
| Per-category stats | 10 rows | `outputs/latent_attacks/new_vae_attacks_gaussian_anticollapse_beta05_freebits01_20260529_173512_20260531_010241_seed42/stats/per_category_pivot.csv` |
| McNemar tests | 5 rows | `outputs/latent_attacks/new_vae_attacks_gaussian_anticollapse_beta05_freebits01_20260529_173512_20260531_010241_seed42/stats/mcnemar_tests.csv` |

### Laplace Non-Targeted Latent-Only Attacks

Source: `outputs/latent_attacks/new_vae_attacks_laplace_rerun_20260529_20260531_012924_seed42/summary.csv`

| Model | latent-pgd ASR | latent-pgd Joint Validity | latent-cw ASR | latent-cw Joint Validity |
|---|---:|---:|---:|---:|
| CNN | 31.40% | 93.80% | 31.00% | 94.60% |
| CNN-LSTM | 37.33% | 82.33% | 40.00% | 81.33% |
| DualPath | 34.57% | 80.71% | 31.86% | 82.86% |
| LSTM | 34.86% | 84.86% | 39.00% | 86.00% |
| MLP | 34.29% | 83.43% | 39.00% | 84.14% |

Companion files:

| Artifact | Detail | Location |
|---|---|---|
| Per-class attack results | 64 rows, 20 columns | `outputs/latent_attacks/new_vae_attacks_laplace_rerun_20260529_20260531_012924_seed42/per_class.csv` |
| Bootstrap stats | 10 rows | `outputs/latent_attacks/new_vae_attacks_laplace_rerun_20260529_20260531_012924_seed42/stats/bootstrap_ci.csv` |
| Per-category stats | 10 rows | `outputs/latent_attacks/new_vae_attacks_laplace_rerun_20260529_20260531_012924_seed42/stats/per_category_pivot.csv` |
| McNemar tests | 5 rows | `outputs/latent_attacks/new_vae_attacks_laplace_rerun_20260529_20260531_012924_seed42/stats/mcnemar_tests.csv` |

### Targeted Benign Latent Attacks

Gaussian source: `outputs/latent_attacks/targeted_benign_pgd_gaussian_anticollapse_beta05_freebits01_20260529_173512_20260531_011821_seed42/summary.csv`

| Model | n | Target Success | Joint Target Success | Joint Validity | IDSR | Mean Input L2 |
|---|---:|---:|---:|---:|---:|---:|
| CNN | 500 | 5.00% | 5.00% | 98.40% | 91.60% | 4.149 |
| CNN-LSTM | 600 | 3.83% | 3.83% | 96.83% | 93.83% | 4.875 |
| DualPath | 700 | 4.43% | 4.14% | 85.14% | 91.43% | 8.457 |
| LSTM | 700 | 5.29% | 4.86% | 89.29% | 93.43% | 10.049 |
| MLP | 700 | 4.14% | 3.43% | 86.29% | 92.43% | 9.575 |

Laplace source: `outputs/latent_attacks/targeted_benign_pgd_laplace_rerun_20260529_20260531_011950_seed42/summary.csv`

| Model | n | Target Success | Joint Target Success | Joint Validity | IDSR | Mean Input L2 |
|---|---:|---:|---:|---:|---:|---:|
| CNN | 500 | 6.20% | 5.80% | 97.80% | 94.80% | 3.226 |
| CNN-LSTM | 600 | 3.50% | 3.17% | 96.00% | 93.50% | 3.422 |
| DualPath | 700 | 4.00% | 3.57% | 91.86% | 94.14% | 3.534 |
| LSTM | 700 | 4.86% | 4.29% | 90.86% | 93.57% | 4.324 |
| MLP | 700 | 4.14% | 3.43% | 91.86% | 93.14% | 3.587 |

### Phase Script Outputs

| Phase | Key Result | Location |
|---|---|---|
| Phase 2 latent PGD vs MLP-3L | Overall ASR 16.86%, protocol 100%, mask 100%, IDSR 88.43%; class ASR: BruteForce 13%, DDoS 8%, DoS 27%, Mirai 0%, Recon 12%, Spoofing 22%, Web 36% | `outputs/latent_attacks/phase2_20260531_012753_seed42/summary.log` |
| Phase 3 latent C&W vs MLP-3L | Overall ASR 21.43%, protocol 100%, mask 100%, IDSR 88.86%; class ASR: BruteForce 27%, DDoS 16%, DoS 20%, Mirai 1%, Recon 6%, Spoofing 58%, Web 22% | `outputs/latent_attacks/phase3_20260531_012807_seed42/summary.log` |
| Phase 4 input baselines vs MLP-3L | Input-PGD about 95.43% ASR and input-CW about 89.86% ASR; all joint validity rates 0 | `outputs/latent_attacks/phase4_20260531_012834_seed42/summary.log` |
| Phase A VAE diagnostics | BruteForce reconstruction error p95 1.9422, DoS p95 0.9988; dominant top-10 error feature is Variance | `outputs/latent_attacks/phaseA_20260531_012849_seed42/summary.log` |

### Latent Attack Output Inventory

| Directory | Main Contents |
|---|---|
| `outputs/latent_attacks/all_models_rerun_20260524_183653_seed42/` | summary.csv 20 rows, 13 columns; per_class.csv empty; all_results.json |
| `outputs/latent_attacks/all_models_rerun_20260524_184016_seed42/` | summary.csv 20 rows; per_class.csv 128 rows |
| `outputs/latent_attacks/all_models_rerun_20260524_185905_seed42/` | summary.csv 20 rows; per_class.csv 128 rows; bootstrap, per-category, McNemar stats |
| `outputs/latent_attacks/all_models_rerun_20260531_005623_seed42/` | summary.csv 20 rows, 18 columns; per_class.csv 128 rows, 19 columns; stats |
| `outputs/latent_attacks/all_models_rerun_20260531_184251_seed42/` | summary.csv 20 rows, 19 columns; per_class.csv 128 rows, 20 columns; all_results.json |
| `outputs/latent_attacks/new_vae_attacks_gaussian_anticollapse_beta05_freebits01_20260529_173512_20260531_010241_seed42/` | improved Gaussian non-targeted latent attacks; summary 10 rows, per_class 64 rows, stats |
| `outputs/latent_attacks/new_vae_attacks_laplace_rerun_20260529_20260531_012924_seed42/` | Laplace non-targeted latent attacks; summary 10 rows, per_class 64 rows, stats |
| `outputs/latent_attacks/targeted_benign_pgd_gaussian_anticollapse_beta05_freebits01_20260529_173512_20260531_011821_seed42/` | targeted benign Gaussian PGD; summary 5 rows, per_class 32 rows |
| `outputs/latent_attacks/targeted_benign_pgd_laplace_rerun_20260529_20260531_011950_seed42/` | targeted benign Laplace PGD; summary 5 rows, per_class 32 rows |

## VAE Training And Improved VAE Results

### Original VAE Training Summary

Source: `results/vae/summary.csv`

| Class | n_train | n_val | Best Val Loss | Final KL | Collapsed Dims | Uncond Pre Valid | Cond Pre Valid | Protocol Acc |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Benign | 139,992 | 19,999 | 130.668 | 15.490 | 3 | 0.0% | 0.0% | 99.44% |
| BruteForce | 8,766 | 1,252 | 11.291 | 9.360 | 1 | 72.1% | 58.8% | 98.32% |
| DDoS | 1,434,940 | 204,993 | -31.930 | 16.897 | 7 | 41.3% | 30.2% | 99.90% |
| DoS | 468,144 | 66,876 | -36.077 | 12.858 | 8 | 0.0% | 0.0% | 99.99% |
| Mirai | 419,972 | 59,996 | -42.285 | 10.638 | 7 | 93.9% | 58.9% | 99.91% |
| Recon | 352,470 | 50,353 | -24.945 | 15.663 | 7 | 94.0% | 87.5% | 99.71% |
| Spoofing | 260,015 | 37,145 | -8.122 | 20.118 | 4 | 0.0% | 0.0% | 99.89% |
| Web | 16,659 | 2,380 | 23.071 | 9.230 | 9 | 0.0% | 0.0% | 99.54% |

Postprocessed validity in this run was 100% for all rows, but repair rates were nonzero/high. The pre-postprocess validity values are the more honest decoder-validity metric for this original VAE.

### Gaussian Rerun

Source: `results/vae/gaussian_rerun_20260529_082314/summary.csv`

| Result | Detail |
|---|---|
| Validity | All classes have 100% unconditional and conditional pre-validity |
| Protocol accuracy | 98.32% to 99.98% |
| Collapsed dimensions | Still high: Benign 10, BruteForce 2, DDoS 9, DoS 11, Mirai 10, Recon 9, Spoofing 10, Web 6 |

### Laplace Rerun

Source: `results/vae/laplace_rerun_20260529/summary.csv`

| Result | Detail |
|---|---|
| Validity | All classes have 100% unconditional and conditional pre-validity |
| Collapsed dimensions | 0 for all 8 classes |
| Protocol accuracy | 97.04% to 99.97% |
| Best validation losses | Mostly about -51 to -103, with Web -56.211 |

### Improved Gaussian Anti-Collapse VAE

Source: `results/vae/gaussian_anticollapse_beta05_freebits01_20260529_173512/summary.csv`

| Result | Detail |
|---|---|
| Validity | All classes have 100% unconditional and conditional pre-validity |
| Collapsed dimensions | 0 for all 8 classes |
| Protocol accuracy | 97.36% to 99.97% |
| Final KL range | 9.200 to 21.293 |
| Thesis use | This is the main improved Gaussian VAE used for the latest latent attack rerun |

### VAE Experiment Tables

| Artifact | Detail | Location |
|---|---|---|
| Mirai experiments | 6 experiments, all pre/post validity 100%, protocol accuracy 99.83% to 99.97%; best/lower val loss: `structured_beta025_constraint005`, val loss -45.128091, final KL 23.477237, collapsed dims 3, worst feature Time_To_Live | `results/vae/experiments/mirai_experiments.csv` |
| Web experiments | Baseline had poor pre-validity: 0.1% unconditional and 0.9% conditional; structured variants reached 100% pre/post validity | `results/vae/experiments/web_experiments.csv` |
| Web structured example | `wide24_beta050_protocol3_structured_constraint005`: val loss -19.882939, final KL 19.060429, collapsed dims 10, protocol 98.24% | `results/vae/experiments/web_experiments.csv` |
| Web high-KL example | `wide24_beta025_protocol4_structured_constraint005`: val loss 101.660561, final KL 1707.17452, protocol 96.51% | `results/vae/experiments/web_experiments.csv` |

### VAE Physics And Audit Results

| Artifact | Detail | Location |
|---|---|---|
| Physics retest summary | Post all-rules mostly 1.0; exceptions include DDoS unconditional_post 0.999, Mirai conditional_post 0.989, Recon conditional_post 0.989 and unconditional_post 0.996 | `results/vae/physics_retest_summary.json` |
| Physics calibration | Clean pass rates for P2/P4/P5 by class; P4 low for DDoS 0.4166, DoS 0.3804, Mirai 0.7026 | `results/vae/physics_calibration.json` |
| Clean dataset physics audit | 4,429,940 samples; P2 1.0, P4 0.9971103446, P5 1.0, all-rules 0.9971103446 | `results/vae/physics_clean_dataset_audit.json` |
| Mirai audit | latent mu global mean 0.00778, mu std 0.63048; low-std dims [1,2,3,4,6,9,15]; generated protocols 17:0.41 and 47:0.59; benign through Mirai binary reconstructed attack rate 0.957 | `results/vae/mirai_audit.json` |
| Mirai failure breakdown | original Mirai unconditional validity 0.717 and conditional validity 0.519; main failure was G6 variance/std consistency | `results/vae/mirai_failure_breakdown.json` |

## Sample Exhibits And Cherry-Picked Outputs

### Non-Targeted Latent Attack Samples

Source: `tables/T7_cherrypicked_featurewise_attack_samples.md`

| Result | Detail |
|---|---|
| Export source | `outputs/latent_attacks/sample_exports/new_vae_attack_samples_20260529_234415` |
| Selection rule | Two samples for each VAE run/classifier/latent attack; best two selected by smallest scaled L2 |
| Validity | All 40 candidate samples regenerated successes and joint-valid |
| Main feature pattern | Latent attacks flip IDS while preserving protocol validity and mask; largest changes concentrate on IAT, Header_Length, Min, Std, Rate |
| Best CNN-LSTM latent-CW Laplace example | sample 150, DDoS to DoS, scaled L2 0.0788, latent L2 0.1579 |
| Best CNN-LSTM latent-PGD Laplace example | sample 150, DDoS to DoS, scaled L2 0.2649 |
| Top feature counts in selected top-5 changes | IAT 19, Header_Length 14, Min 13, Std 10, Rate 7, Max 7, ack_count 7, Tot size 6 |

### Targeted Benign Attack Samples

Sources: `tables/T8_cherrypicked_targeted_benign_attack_samples.md`, `tables/T8_cherrypicked_targeted_benign_attack_samples.csv`

| Result | Detail |
|---|---|
| Export source | `outputs/latent_attacks/sample_exports/targeted_benign_attack_samples_20260530_002152` |
| Selected samples | 10 selected targeted benign samples |
| Validity | All selected samples have target_success=True and joint_valid=True |
| Best overall example | LSTM Laplace sample 86334, BruteForce to Benign, scaled L2 0.3830, latent L2 1.6643 |
| Top feature counts | Min 10, Header_Length 9, Variance 7, IAT 6, ack_count 6, AVG 4, Std 3, Tot size 2 |
| Full featurewise table | 390 rows | `tables/T8_cherrypicked_targeted_benign_featurewise.csv` |

## Statistical And Figure Outputs

### Distributional And Statistical Tests

| Artifact | Detail | Location |
|---|---|---|
| KS/Wasserstein results | 39 feature rows, 10 columns; many frozen features have latent KS 0.0 and p=1.0 while input attacks significantly alter protocol/binary features | `thesis_figures/ks_wasserstein_results.csv` |
| Statistical summary | 157 rows, 4 columns; includes Spearman latentDelta_vs_ASRValid = -0.7380952381, p = 0.0365527611 | `thesis_figures/statistical_summary.csv` |
| Benign overlay metrics | 21 rows, 10 columns with JS divergence, MMD, Wasserstein PCA, correlation difference, log-likelihood metrics | `thesis_figures/benign_overlay_fidelity/benign_overlay_metrics_summary.csv` |

### Main Figure Files

| Figure / Group | Location |
|---|---|
| ASR raw vs valid plot | `figures/asr_raw_vs_valid.png` |
| Class distribution | `figures/F1_class_distribution.pdf` |
| Feature distributions | `figures/F2_feature_distributions.pdf` |
| Correlation heatmap | `figures/F3_correlation_heatmap.pdf` |
| PCA by category and loadings | `figures/F4a_pca_by_category.pdf`, `figures/F4b_pca_loadings.pdf` |
| t-SNE by category | `figures/F5_tsne_by_category.pdf` |
| UMAP Mirai vs Benign | `figures/F6b_umap_mirai_vs_benign.pdf` |
| Clustering dendrogram and CH scores | `figures/F7a_dendrogram.pdf`, `figures/F7b_ch_scores.pdf` |
| Thesis category ASR heatmap | `thesis_figures/category_asr_heatmap.pdf` |
| Thesis multimodel radar | `thesis_figures/radar_multimodel.pdf` |
| Perturbation heatmap | `thesis_figures/perturbation_heatmap.pdf` |
| L2 distortion CDF | `thesis_figures/l2_distortion_cdf.pdf` |
| L2 by category boxplot | `thesis_figures/l2_by_category_boxplot.pdf` |
| Latent cluster spread vs ASR | `thesis_figures/latent_cluster_spread_vs_asr.pdf` |
| Latent interpolation validity | `thesis_figures/latent_interpolation_validity.pdf` |
| VAE reconstruction plots | `thesis_figures/vae_reconstruction_error_hist.pdf`, `thesis_figures/vae_reconstruction_mae_by_feature.pdf`, `thesis_figures/vae_reconstruction_scatter.pdf` |
| Latent/input PCA, t-SNE, UMAP comparisons | `thesis_figures/` |
| Benign overlay KDE/heatmaps | `thesis_figures/benign_overlay_fidelity/` |

## Source Map / Artifact Inventory

### Tables Directory

| Artifact | Detail |
|---|---|
| `tables/T0_imbalance_audit.csv` | 34 rows, 7 columns |
| `tables/T1_feature_schema.csv` | 39 rows, 11 columns |
| `tables/T2a_class_counts.csv` | 34 rows, 4 columns |
| `tables/T2b_category_counts.csv` | 8 rows, 4 columns |
| `tables/T3_summary_stats.csv` | 39 rows, 13 columns |
| `tables/T4_clean_validity.csv` | 48 rows, 4 columns; historical/superseded validity audit |
| `tables/T5_sample_rows_wide.csv` | 20 rows, 40 columns |
| `tables/T5_sample_rows_transposed.md` | 43 lines |
| `tables/T6_feature_categorization.csv` | 21 rows, 2 columns |
| `tables/F3_correlation_matrix.csv` | 39 rows, 40 columns |
| `tables/F4_pca_loadings.csv` | 39 rows, 3 columns |
| `tables/impossible_traffic_binary_pgd.csv` | 189 rows, 11 columns |
| `tables/impossible_traffic_binary_pgd_slide.md/.csv` | 15 rows, 9 columns in CSV |
| `tables/impossible_traffic_8class_pgd.csv` | 187 rows, 11 columns |
| `tables/impossible_traffic_8class_pgd_slide.md/.csv` | 15 rows, 9 columns in CSV |
| `tables/impossible_traffic_34class_pgd.csv` | 185 rows, 11 columns |
| `tables/impossible_traffic_34class_pgd_slide.md/.csv` | 15 rows, 9 columns in CSV |
| `tables/impossible_traffic_slide_pack.csv` | 45 rows, 9 columns |
| `tables/supervisor_onepager.md` | 63-line older briefing; use newer attack outputs for final numbers |
| `tables/T7_cherrypicked_featurewise_attack_samples.md` | 66-line sample exhibit table |
| `tables/T8_cherrypicked_targeted_benign_attack_samples.csv` | 10 rows, 13 columns |
| `tables/T8_cherrypicked_targeted_benign_attack_samples.md` | 47-line targeted sample table |
| `tables/T8_cherrypicked_targeted_benign_featurewise.csv` | 390 rows, 30 columns |

### Main Result Directories

| Directory | Contents |
|---|---|
| `results/` | Baseline classification reports, all-model summaries, attack summaries, validation outputs, VAE summaries |
| `results/attacks/` | Input-space attack outputs, shock table, violation breakdown, thesis attack bundles, impossible-traffic exhibits |
| `results/validation/` | Current full dataset validation reports and rule/per-class breakdowns |
| `results/vae/` | Original VAE, improved Gaussian, Laplace, physics, calibration, and audit outputs |
| `outputs/latent_attacks/` | Latent attack reruns, improved VAE attack results, targeted benign attacks, per-class files, bootstrap/McNemar stats |
| `tables/` | EDA tables, sample rows, thesis exhibit tables, impossible traffic tables |
| `figures/` | EDA plots and original attack visualizations |
| `thesis_figures/` | Final thesis plots, statistical summaries, distributional fidelity outputs |
| `logs/baselines/` | Training/evaluation logs for earlier baseline runs |

### Report Files

| File | Detail |
|---|---|
| `ATTACK_RERUN_REPORT_20260531.md` | 205-line report summarizing 2026-05-31 rerun status and conclusions |
| `P2_THESIS_REPORT.md` | Long working thesis/report draft with many embedded results |
| `IMPROVEMENTS.md` | Notes for improvements/defense preparation |
| `runner_guide.md`, `guide.md`, `codex.md` | Guides/context files; not primary metric sources |

