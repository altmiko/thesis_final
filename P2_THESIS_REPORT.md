# Phase 2 Thesis Methodology Report Log

Generated from workspace: `D:\thesis_final`

Generated on: 2026-05-29

Updated on: 2026-05-31 with restart-aware latent attack explanation and beta05 rerun results.

Purpose: this file is a detailed working log for the Phase 2 LaTeX methodology chapter. It records the current data pipeline, preprocessing decisions, model design, VAE/generative framework, attack-validation outputs, and artifact inventory found in the workspace.

Important scope note: I inspected the current repository state, including source code, configs, manifests, tables, figures, model checkpoints, validation reports, attack outputs, VAE outputs, and run folders. Binary artifacts such as `.pt`, `.npy`, `.npz`, `.pkl`, `.pdf`, and `.png` are logged by role, shape/count, and output directory rather than printed byte-for-byte. Text, CSV, JSON, Markdown, YAML, and Python files were used as the main sources for methodology and decision rationale.

Git/workspace note: the working tree already contains modified and untracked files. This report treats those files as thesis artifacts and does not assume they should be reverted.

---

# 3. Methodology

## 3.1 Design Overview or Methodology Overview

The thesis workspace implements a full experimental pipeline for studying adversarial robustness of machine-learning based network intrusion detection on the CICIoT2023 dataset. The central methodology is:

1. Collect and stage a large CICIoT2023 flow dataset.
2. Build a 39-feature schema and an 8-category attack taxonomy from the original 34 labels.
3. Clean, transform, split, scale, validate, and hash the processed dataset.
4. Train baseline IDS classifiers for binary, 8-class, and 34-class prediction.
5. Generate adversarial attacks in two regimes:
   - unconstrained input/scaled feature attacks, which intentionally show what happens when feature-space constraints are ignored;
   - VAE/latent-space attacks, which attempt to stay on a valid data manifold.
6. Inverse-transform adversarial examples into raw feature space and evaluate whether they are physically/domain-valid traffic.
7. Compare `ASR_raw` against `ASR_valid` and use the gap as the main thesis evidence.

The core thesis claim supported by the current artifacts is:

> Raw adversarial success on scaled tabular IDS features can be very high, but many apparently successful input-space attacks become invalid when checked against raw network-domain constraints. A validity-aware attack framework is therefore necessary for realistic IDS robustness evaluation.

The methodology deliberately separates predictive accuracy from physical validity. A classifier can be fooled in scaled feature space, but that does not mean the perturbed row corresponds to a plausible network flow. This is why the validator and the inverse-transform step are first-class components rather than afterthoughts.

### Methodology Components and Source Files

The main package layout is:

| Component | Path | Role |
|---|---|---|
| Feature schema and category map | `src/preprocessing/feature_groups.py` | Defines the 39 features, 34-to-8 label mapping, binary/integer/mutable/frozen feature groups, and base perturbation mask. |
| Preprocessing pipeline | `src/preprocessing/pipeline.py` | Loads staged parquet data, cleans, validates, encodes labels, stratifies splits, fits `RobustScaler`, saves arrays, masks, weights, hashes, and manifest. |
| NetDiffuser feature grouping | `src/preprocessing/netdiffuser_categorization.py` | Uses absolute Spearman correlation, hierarchical clustering, and CH-score search to partition features into discrete and relative groups. |
| Domain validator | `src/attack/validator.py` | Raw-space rules for non-negativity, protocol validity, binary features, protocol indicators, Min/AVG/Max, variance/std consistency, TTL, and packet counts. |
| Classifier models | `src/classifiers/models.py` | Implements MLP, CNN, LSTM, serial CNN-LSTM, attention serial CNN-LSTM, and DualPath IDS. |
| Baseline training | `src/classifiers/baseline_experiments.py` | Trains/evaluates classifier baselines and records the loss-weighting decision. |
| Input-space attacks | `src/attack/adversarial_attacks.py`, `src/attack/run_attacks.py` | Runs FGSM, PGD, and CW without image-style [0,1] clamping to expose unconstrained tabular attack behavior. |
| Validity evaluation | `src/evaluation/validity_analysis.py`, `src/evaluation/run_validity_analysis.py`, `src/evaluation/validate_full_dataset.py` | Inverse-transforms scaled arrays and computes raw validity, `ASR_valid`, rule violations, and exhibits. |
| VAE subsystem | `src/vae/*` | Per-category beta-VAE models, mixed-type decoder, training, diagnostics, latent geometry, fidelity analysis, and physics checks. |
| Latent attacks | `src/attack/latent_infra.py`, `src/attack/latent_pgd.py`, `src/attack/latent_cw.py`, `src/attack/latent_restarts.py`, `src/attack/run_all_models_attack_rerun.py`, `src/attack/run_targeted_benign_latent_pgd.py` | Performs VAE latent attacks under protocol and perturbation-mask governance, including restart-aware latent search. |
| Thesis visualizations | `src/thesis_visualizations.py`, `src/plot_distributional_fidelity.py`, `src/plot_benign_overlay_fidelity.py` | Generates distribution, latent-space, perturbation, and statistical figures for thesis use. |

### Current Workspace Artifact Scale

At inspection time the workspace contained 770 non-git/non-cache files:

| Top-level path | Files | Approx bytes | Meaning |
|---|---:|---:|---|
| `data` | 54 | 11,528,327,804 | Raw CICIoT2023 CSV, processed arrays, encoders, scaler, masks, reports, reduced top-24 feature artifacts. |
| `results` | 182 | 131,216,589 | Classification reports, attack outputs, validation reports, VAE summaries/diagnostics. |
| `outputs` | 233 | 11,925,954 | Latent attack run folders, config snapshots, summaries, per-sample outputs, GMM priors. |
| `models` | 46 | 8,472,229 | Classifier and per-class VAE checkpoints. |
| `checkpoints` | 18 | 6,098,501 | Older CVAE checkpoint runs and metadata. |
| `thesis_figures` | 42 | 5,137,445 | Final thesis figures for VAE, latent attacks, distributional fidelity, correlations, and category ASR. |
| `figures` | 22 | 3,404,754 | EDA and appendix figures. |
| `logs` | 66 | 2,823,121 | Baseline logs, histories, confusion matrices, and EDA logs. |
| `src` | 64 | 802,649 | Source code. |
| `tables` | 26 | 174,866 | EDA, validity, impossible-traffic, feature-selection, and slide-ready tables. |
| Root docs/configs | 14 | 115,956 | Guidance docs, progress notes, run manifests, validator notes. |

Most common file types:

| Extension | Count | Main role |
|---|---:|---|
| `.json` | 166 | Manifests, metrics, reports, configs, diagnostics. |
| `.npz` | 155 | Attack artifacts, t-SNE/UMAP coordinates, thesis bundles. |
| `.csv` | 83 | Tables, summaries, confusion matrices, attack stats. |
| `.pt` | 64 | PyTorch model checkpoints. |
| `.py` | 64 | Source code. |
| `.png` | 47 | Curves and generated figures. |
| `.pdf` | 44 | Thesis/EDA figures. |
| `.npy` | 31 | Processed feature and label arrays. |
| `.log` | 28 | Execution logs. |
| `.pkl` | 24 | Encoders, scaler, latent GMM priors. |

---

## 3.1.1 Data Collection / Input

### Dataset Source

The input dataset in this workspace is:

`data/ciciot2023/ciciot2023_base.csv`

The CSV header is:

```text
Header_Length,Protocol Type,Time_To_Live,Rate,fin_flag_number,syn_flag_number,rst_flag_number,psh_flag_number,ack_flag_number,ece_flag_number,cwr_flag_number,ack_count,syn_count,fin_count,rst_count,HTTP,HTTPS,DNS,Telnet,SMTP,SSH,IRC,TCP,UDP,DHCP,ARP,ICMP,IGMP,IPv,LLC,Tot sum,Min,Max,AVG,Std,Tot size,IAT,Number,Variance,Label
```

This gives 39 numeric input features plus one target column, `Label`.

The raw profiling log (`logs/eda.log`) records:

| Raw dataset fact | Value |
|---|---:|
| CSV files | 1 |
| Total raw file size in log | 9.41 GB |
| File size observed by current filesystem | 10,105,457,570 bytes |
| Raw rows profiled | 45,019,243 |
| Unique 34-class labels | 34 |
| Raw imbalance ratio, largest class / smallest class | 5,763.6x |
| Staged analysis table | `data/processed/raw_loaded.parquet` |
| Staged working rows | 4,429,940 |

The staged parquet and processed arrays use a sampled/capped version of the raw dataset, not the full 45 million rows. The preprocessing manifest records `mode: SAMPLE` and sample caps of 200,000 for majority and medium classes, while minority and rare classes are kept. This is a practical and methodological decision: it keeps experiments computationally feasible while preserving rare attack types rather than discarding them.

### Feature Schema

The schema authority is `src/preprocessing/feature_groups.py`. It defines this project as "Modified Schema A (39 features)".

Important schema details:

| Group | Features |
|---|---|
| Label column | `Label` |
| Total features | 39 |
| Binary indicators | `HTTP`, `HTTPS`, `DNS`, `Telnet`, `SMTP`, `SSH`, `IRC`, `TCP`, `UDP`, `DHCP`, `ARP`, `ICMP`, `IGMP`, `IPv`, `LLC` |
| Integer-valued features | TCP flag numbers, `ack_count`, `syn_count`, `fin_count`, `rst_count`, `Number` |
| Immutable features | `Protocol Type`, `TCP`, `UDP`, `ICMP` |
| Quasi-immutable features | application/protocol indicators and `Time_To_Live` |
| Mutable features | flow rates, header length, packet-size aggregates, flag counts, `IAT`, `Number`, `Variance` |

The schema includes `Time_To_Live` and `IGMP`, which are important because protocol checks later use `IGMP` and TTL range validation.

### Label Hierarchy

The raw dataset contains 34 labels. The thesis pipeline maps those 34 labels into 8 categories:

| 8-class category | Source labels |
|---|---|
| `Benign` | `BENIGN` |
| `BruteForce` | `DICTIONARYBRUTEFORCE` |
| `DDoS` | 12 DDoS variants such as `DDOS-ICMP_FLOOD`, `DDOS-UDP_FLOOD`, `DDOS-SYN_FLOOD`, etc. |
| `DoS` | `DOS-UDP_FLOOD`, `DOS-TCP_FLOOD`, `DOS-SYN_FLOOD`, `DOS-HTTP_FLOOD` |
| `Mirai` | `MIRAI-GREETH_FLOOD`, `MIRAI-UDPPLAIN`, `MIRAI-GREIP_FLOOD` |
| `Recon` | `RECON-PINGSWEEP`, `RECON-OSSCAN`, `RECON-PORTSCAN`, `RECON-HOSTDISCOVERY`, `VULNERABILITYSCAN` |
| `Spoofing` | `MITM-ARPSPOOFING`, `DNS_SPOOFING` |
| `Web` | `BROWSERHIJACKING`, `BACKDOOR_MALWARE`, `XSS`, `SQLINJECTION`, `COMMANDINJECTION`, `UPLOADING_ATTACK` |

Reasons for keeping all three label framings:

- Binary classification answers the coarse IDS question: benign vs attack.
- 8-class classification gives a practical attack-family view and stabilizes rare classes.
- 34-class classification tests fine-grained detection but is harder because several classes are extremely rare.
- The VAE/latent attack framework uses the 8-class view because category-level manifolds are more trainable than one VAE per very rare 34-class label.

### Processed Category Counts

The processed working dataset contains:

| Category | Count | Percent |
|---|---:|---:|
| DDoS | 2,049,917 | 46.2741% |
| DoS | 668,775 | 15.0967% |
| Mirai | 599,960 | 13.5433% |
| Recon | 503,528 | 11.3665% |
| Spoofing | 371,451 | 8.3850% |
| Benign | 199,989 | 4.5145% |
| Web | 23,798 | 0.5372% |
| BruteForce | 12,522 | 0.2827% |
| Total | 4,429,940 | 100.0000% |

The strongest remaining imbalance is at the category and 34-class level. BruteForce and Web are much smaller than DDoS. This is why the report should not rely only on accuracy; macro-F1, class breakdowns, validity-by-class, and per-category attack tables are necessary.

---

## 3.1.2 Data Preprocessing and Analysis

The preprocessing pipeline is implemented mainly in `src/preprocessing/pipeline.py`, with EDA tables/figures in `src/evaluation/eda_tables.py`, `src/evaluation/eda_figures_part1.py`, and `src/evaluation/eda_figures_part2.py`.

The complete preprocessing sequence is:

1. Load `data/processed/raw_loaded.parquet`.
2. Remove infinite and missing values.
3. Drop timestamp columns if present.
4. Clip extreme feature values at the 99.99th percentile.
5. Round integer-valued features.
6. Round binary features to `{0, 1}`.
7. Validate raw-space cleaned data with `validate_batch`.
8. Encode labels for 34-class, 8-class, and binary tasks.
9. Create a 70/10/20 stratified split.
10. Fit `RobustScaler` on the training split only.
11. Transform train/validation/test feature arrays.
12. Save class weights for reference.
13. Build a perturbation mask using mutability policy, NetDiffuser categorization, and near-zero-IQR governance.
14. Save arrays, encoders, scaler, masks, reports, hashes, and run manifest.

### Why This Preprocessing Design Was Used

The dataset is tabular network-flow data with mixed feature types:

- continuous traffic-rate and size features;
- integer packet counts and flag counts;
- binary protocol/application indicators;
- protocol IDs with known allowed values;
- cross-feature rules such as `Min <= AVG <= Max` and `Variance ~= Std^2`.

Generic tabular preprocessing is not enough because an adversarial example can be numerically valid for a neural network and still impossible as network traffic. Therefore the pipeline preserves a raw-space validator and saves the scaler so every scaled adversarial sample can be inverse-transformed and checked.

### EDA Outputs

EDA generated the following tables:

| Table | Path | Purpose |
|---|---|---|
| T0 | `tables/T0_imbalance_audit.*` | Raw class imbalance audit and thesis strategy by rarity tier. |
| T1 | `tables/T1_feature_schema.*` | Feature schema, type, description, ranges, and basic stats. |
| T2a | `tables/T2a_class_counts.*` | 34-class counts. |
| T2b | `tables/T2b_category_counts.*` | 8-category counts. |
| T3 | `tables/T3_summary_stats.*` | Feature summary statistics, skewness, kurtosis, zero rate, unique rate. |
| T4 | `tables/T4_clean_validity.*` | Older clean-data validity audit; see artifact-drift note below. |
| T5 | `tables/T5_sample_rows_wide.csv`, `tables/T5_sample_rows_transposed.md` | Representative raw rows. |
| T6 | `tables/T6_feature_categorization.*` | NetDiffuser discrete/relative feature categorization. |

EDA generated the following figures:

| Figure | Path | Purpose |
|---|---|---|
| F1 | `figures/F1_class_distribution.pdf` | 34-class and 8-category class imbalance. |
| F2 | `figures/F2_feature_distributions.pdf` | Log-binned feature histograms by selected classes. |
| F3 | `figures/F3_correlation_heatmap.pdf` | Spearman absolute correlation heatmap. |
| F4a/F4b | `figures/F4a_pca_by_category.pdf`, `figures/F4b_pca_loadings.pdf` | PCA projection and loadings. |
| F5 | `figures/F5_tsne_by_category.pdf` | t-SNE category projection. |
| F6b | `figures/F6b_umap_mirai_vs_benign.pdf` | UMAP zoom for Mirai vs Benign. |
| F7a/F7b | `figures/F7a_dendrogram.pdf`, `figures/F7b_ch_scores.pdf` | NetDiffuser clustering and cut-height search. |
| Appendix F6 | `figures/appendix/F6_umap_by_category.pdf` | Global UMAP moved to appendix. |

Reasons for these analyses:

- Class distribution plots justify sampling, macro metrics, and category-level aggregation.
- Feature histograms reveal heavy tails and motivate robust scaling.
- Correlation/NetDiffuser analysis informs which features should be treated as independent/discrete versus relational/relative in perturbation policy.
- PCA/t-SNE/UMAP projections check whether attack families overlap and whether per-category modeling is justified.

---

## 3.1.2.1 Data Cleaning

### Infinite and Missing Values

The preprocessing pipeline executes:

```text
replace([inf, -inf], nan).dropna()
```

The run manifest records:

| Stage | Rows |
|---|---:|
| Loaded parquet | 4,429,940 |
| After cleaning | 4,429,940 |

No rows were lost in the final preprocessing run. This is important: downstream differences are not caused by missing-value removal.

### Timestamp Removal

The pipeline drops `ts` or `Timestamp` if either exists. The current 39-feature header does not include a timestamp column. The rule remains in the code for schema resilience.

Reason:

- Timestamp fields can leak collection order or capture-specific artifacts.
- Most IDS models should generalize from traffic features, not from capture chronology.
- Dropping timestamps prevents a model from learning accidental time ordering instead of attack behavior.

### Quantile Clipping

The pipeline clips feature values above the 99.99th percentile. Most non-negative features are clipped to `[0, upper_99.99]`. A signed-feature exception exists for `Covariance`, although `Covariance` is not part of the current 39-feature schema.

The run manifest recorded 18 clipped features. Top clipped features by count:

| Feature | Rows clipped | Upper clip |
|---|---:|---:|
| DNS | 443 | 0.770061 |
| Tot sum | 443 | 175,125.1464 |
| Std | 443 | 4,679.6291 |
| IAT | 443 | 0.121926 |
| Variance | 443 | 21,898,928.7670 |
| AVG | 432 | 3,686.0 |
| Tot size | 432 | 3,686.0 |
| Min | 424 | 2,858.0 |
| Max | 374 | 15,994.0 |
| ARP | 366 | 0.9 |
| Rate | 358 | 998,643.8095 |
| cwr_flag_number | 311 | 0.1 |

Reason for 99.99th percentile clipping:

- CICIoT flow features are heavy-tailed. `Rate`, `Tot sum`, `Variance`, `IAT`, and packet-size features can contain extreme values that dominate means, variances, PCA, neural gradients, and visualizations.
- Clipping only the top 0.01% is conservative: it suppresses pathological extremes without flattening the real distribution.
- The saved clipping log makes the operation auditable and reproducible.
- Clipping before `RobustScaler` prevents rare huge values from generating unstable inverse-transform behavior during adversarial evaluation.

Why not remove those rows instead:

- Removing all high-value rows would disproportionately remove real high-volume attack traffic.
- Clipping preserves sample count and class distribution while limiting the leverage of extreme tails.

### Integer and Binary Rounding

The pipeline rounds integer features and clips them to non-negative values. Binary features are rounded and clipped to `{0, 1}`.

Reason:

- Neural models consume floats, but these columns have discrete semantics.
- If flag or protocol-indicator columns remain soft/noisy, the validator can reject otherwise legitimate flows.
- Rounding creates a clean, domain-compliant baseline so later invalidity can be attributed to adversarial perturbation rather than preprocessing noise.

### Clean-Data Validation

The current processed validation report (`data/processed/processed_data_validity_report.json`) records:

| Validation fact | Value |
|---|---:|
| Samples checked | 4,429,940 |
| Valid samples | 4,429,940 |
| Overall validity | 100.00% |
| Validator rules | 49 |
| Non-zero rule failures | 0 |

The full validation run (`results/validation/full_dataset_validation_report.txt`) independently confirms:

| Split | Shape | Valid samples |
|---|---|---:|
| Train | `(3,100,958, 39)` | 3,100,958 / 3,100,958 |
| Validation | `(442,994, 39)` | 442,994 / 442,994 |
| Test | `(885,988, 39)` | 885,988 / 885,988 |
| Total | 4,429,940 | 4,429,940 / 4,429,940 |

The same report confirms 9/9 synthetic corruption tests passed. The synthetic tests intentionally break protocol IDs, binary fields, non-negativity, Min/Max ordering, variance/std consistency, and packet-count rules. This matters because it shows the validator accepts clean data but rejects designed corruptions.

### Validator Rules

The raw-space validator checks:

| Rule family | Examples |
|---|---|
| Non-negativity | `Header_Length`, `Rate`, `Tot sum`, `Min`, `Max`, `AVG`, `Std`, `IAT`, `Number`, `Variance`, flags/counts. |
| Protocol allowlist | `Protocol Type` must be integer-like and in `{0, 1, 2, 6, 17, 47}`. |
| Binary indicators | application/protocol flags must be integer-like and in `{0, 1}`. |
| Protocol-indicator consistency | if `TCP=1`, protocol must be 6; if `UDP=1`, protocol must be 17; if `ICMP=1`, protocol must be 1; if `IGMP=1`, protocol must be 2. |
| Statistical ordering | `Min <= AVG <= Max`. |
| Variance consistency | `Variance ~= Std^2` within 5% tolerance. |
| TTL range | `Time_To_Live` must be in `[0, 255]`. |
| Packet-count validity | `Number >= 1` and integer-like. |

Reason for raw-space validation:

- Scaled values are not physically interpretable.
- A small scaled perturbation can become a huge raw perturbation on high-IQR features.
- A classifier prediction alone cannot prove that the adversarial row corresponds to realizable traffic.
- The thesis needs `ASR_valid`, not only `ASR_raw`.

### Artifact Drift Note for T4

`tables/T4_clean_validity.md` reports an older clean-data validity rate of 81.3724%, mostly due to protocol-related violations. The current authoritative outputs are:

- `data/processed/processed_data_validity_report.json`
- `results/validation/full_dataset_validation_report.txt`
- `data/processed/clean_data_validity_report.json`

These current reports show 100% validity. The interpretation is that T4 was generated before later preprocessing/validator alignment and was not regenerated after the final validity pass. In the LaTeX report, cite the current processed/full validation reports as final, and mention the older T4 only if discussing development history.

---

## 3.1.2.2 Data Transformation

### Label Encoding

The pipeline uses `LabelEncoder` to create:

| Target file | Task | Classes |
|---|---|---:|
| `y_train.npy`, `y_val.npy`, `y_test.npy` | 34-class | 34 |
| `y_train_cat.npy`, `y_val_cat.npy`, `y_test_cat.npy` | 8-class category | 8 |
| `y_train_bin.npy`, `y_val_bin.npy`, `y_test_bin.npy` | binary | 2 |

Encoders and label names are saved:

- `data/processed/label_encoder.pkl`
- `data/processed/category_encoder.pkl`
- `data/processed/class_names.json`
- `data/processed/category_names.json`
- `data/processed/class_to_category.json`

Reason:

- Saved encoders ensure that metrics, confusion matrices, attacks, and exhibits decode labels consistently.
- Multi-task arrays allow binary, category, and fine-grained results to share the same feature split.

### Stratified 70/10/20 Split

The pipeline creates splits by indices so all label variants remain aligned.

| Split | Rows | Percent |
|---|---:|---:|
| Train | 3,100,958 | 70% |
| Validation | 442,994 | 10% |
| Test | 885,988 | 20% |

Reason for stratification:

- CICIoT2023 is highly imbalanced.
- Rare classes such as `UPLOADING_ATTACK`, `RECON-PINGSWEEP`, and Web attacks must remain represented in train/validation/test.
- A non-stratified split could accidentally remove rare labels from validation or test, invalidating macro-F1 and per-class attack conclusions.

### RobustScaler

The pipeline fits:

```text
RobustScaler()
```

on `X_train` only, then transforms validation and test.

Saved scaler:

`data/processed/scaler.pkl`

Observed scaler statistics:

| Scaler statistic | Value |
|---|---:|
| Center min | 0.0 |
| Center max | 6,510.4683 |
| Center mean | 337.3625 |
| Scale/IQR min | 0.000927 |
| Scale/IQR max | 48,939.0 |
| Scale/IQR mean | 3,087.6852 |

Largest IQR features:

| Feature | IQR/scale |
|---|---:|
| Tot sum | 48,939 |
| Variance | 44,315.3 |
| Rate | 24,751.9 |
| Max | 1,014 |
| AVG | 518 |
| Tot size | 518 |
| Std | 210.512 |
| Number | 90 |

Smallest IQR features:

| Feature | IQR/scale |
|---|---:|
| IAT | 0.000927 |
| Many binary/integer indicators | 1.0 fallback scale |

Reasons for using `RobustScaler`:

- The data is heavy-tailed. `Rate`, `IAT`, `Tot sum`, `Max`, and `Variance` have extreme outliers and strong skewness.
- Standard scaling uses mean and standard deviation, which are unstable under heavy tails. A small number of extreme flows can dominate the mean/std and distort most samples.
- Min-max scaling would compress the majority of samples into a small interval because the raw maxima are very large. It would also make adversarial epsilon values depend heavily on rare extremes.
- Robust scaling uses median and IQR, so it is less sensitive to outliers while still putting heterogeneous features onto a more comparable scale for neural training.
- Fitting on train only prevents validation/test leakage.
- Attack epsilon values become interpretable as perturbations in robust-scaled feature units rather than raw units with incompatible scales.

Important caveat:

- Robust scaling does not guarantee domain validity. A scaled perturbation of 0.30 can still become a large raw change for high-IQR features. That is why inverse-transform validation is mandatory.

### Near-Zero-IQR Governance

The pipeline identifies near-zero-IQR features using:

```text
NEAR_ZERO_IQR_THRESHOLD = 1e-6
RARE_SIGNAL_NONZERO_THRESHOLD = 1e-3
```

Current report:

| Near-zero kind | Count | Policy |
|---|---:|---|
| Constant | 7 | Auto-freeze |
| Rare signal | 3 | Allow |
| Concentrated | 13 | Manual review |
| Total near-zero-IQR features | 23 | Mixed policy |

Auto-frozen features:

```text
DHCP, IGMP, IRC, SMTP, Telnet, cwr_flag_number, ece_flag_number
```

Manual concentrated features allowed as mutable but capped to partial perturbation:

```text
ack_flag_number, fin_count, fin_flag_number, psh_flag_number,
rst_count, rst_flag_number, syn_flag_number
```

Reason:

- Near-zero-IQR features are dangerous for attacks because their scaled representation can exaggerate small raw changes or create non-interpretable epsilon behavior.
- Constant features should be frozen because changing them creates values never observed in training.
- Rare-signal features are not automatically frozen because rare events may be meaningful attack indicators.
- Concentrated but attacker-controllable flags/counts are manually reviewed instead of blindly frozen.

### Perturbation Mask

Saved mask:

`data/processed/perturbation_mask.npy`

Current mask counts:

| Mask value | Meaning | Feature count |
|---:|---|---:|
| 0.0 | Frozen | 19 |
| 0.3 | Partial/capped | 9 |
| 1.0 | Fully perturbable | 11 |

Fully perturbable features:

```text
Header_Length, Rate, Tot sum, Min, Max, AVG, Std, Tot size, IAT, Number, Variance
```

Partial features:

```text
fin_flag_number, syn_flag_number, rst_flag_number, psh_flag_number,
ack_flag_number, ack_count, syn_count, fin_count, rst_count
```

Frozen features include protocol, TTL, application/protocol indicators, and near-constant flags:

```text
Protocol Type, Time_To_Live, ece_flag_number, cwr_flag_number,
HTTP, HTTPS, DNS, Telnet, SMTP, SSH, IRC, TCP, UDP, DHCP, ARP,
ICMP, IGMP, IPv, LLC
```

Reason:

- Protocol and protocol-indicator features represent network-stack or application-layer identities. Arbitrarily changing them can create impossible traffic.
- Packet-size/rate/aggregate features are more plausibly attacker-influenced and are therefore allowed.
- Flag/count features are attacker-influenced but discrete and concentrated, so they are capped.

### NetDiffuser Feature Categorization

`src/preprocessing/netdiffuser_categorization.py` implements a NetDiffuser-style partition:

- absolute Spearman correlation matrix;
- distance `sqrt(2 * (1 - abs(corr)))`;
- average-linkage hierarchical clustering;
- cut-height grid from 0.1 to 1.0;
- Calinski-Harabasz score;
- preference for non-trivial local maxima.

Current output:

| Metric | Value |
|---|---:|
| Discrete features | 21 |
| Relative features | 18 |
| Best cut height | 0.272727 |
| Best index | 19 |
| Best cut is non-trivial | true |

Reason:

- Feature perturbability should not be only manually assigned.
- Correlation-based grouping helps distinguish singleton/discrete-like features from relational feature clusters.
- The constrained cut-height search improves publication stability and avoids degenerate cluster choices.

### Class Weights

The preprocessing pipeline saves balanced class weights:

- `class_weights_34.npy`
- `class_weights_8.npy`
- `class_weights_2.npy`
- `class_weights_named.json`

However, the classifier training summary records:

```text
loss_weighting: excluded
Class weights excluded because stratified sampling already balances the training set; applying original-distribution weights would over-penalize majority classes in the balanced sample.
```

Reason:

- Class weights are useful metadata and can be used in later controlled experiments.
- In the current baseline experiments, stratified/capped sampling already changes the training distribution.
- Re-applying original-distribution weights could double-count imbalance and distort training.

### Data Augmentation Decision

The preprocessing final log states:

```text
Augmentation applied: NONE (SMOTE/ADASYN skipped by design)
```

Reason:

- SMOTE/ADASYN can create fractional protocol/binary/integer values.
- Synthetic interpolation may violate `Min <= AVG <= Max`, protocol consistency, and `Variance ~= Std^2`.
- For a thesis about adversarial validity, adding potentially invalid synthetic training rows would weaken the clean-data invariant.

---

## 3.1.2.3 Data Integration

The integration design connects raw data, processed arrays, model outputs, attack outputs, and validation artifacts through saved encoders, scaler, masks, hashes, and manifests.

### Integrated Artifacts

| Artifact | Path | Integrated use |
|---|---|---|
| Raw source | `data/ciciot2023/ciciot2023_base.csv` | Original CICIoT2023 input. |
| Staged parquet | `data/processed/raw_loaded.parquet` | Shared EDA/preprocessing source. |
| Processed arrays | `data/processed/X_*.npy`, `y_*.npy` | Classifier, VAE, attack, and validation input. |
| Encoders | `data/processed/*encoder.pkl`, `class_names.json`, `category_names.json` | Consistent label decoding. |
| Scaler | `data/processed/scaler.pkl` | Train/val/test scaling and attack inverse transform. |
| Perturbation mask | `data/processed/perturbation_mask.npy` | Governs constrained/latent attacks and exhibit analysis. |
| Run manifest | `config/run_manifest.json` | Reproducibility metadata, package versions, seeds, content hashes, clipping, split sizes. |
| VAE manifest | `vae_run_manifest.json` | VAE checkpoints, protocol partition, validity diagnostics. |
| Attack summaries | `results/attacks/*.csv`, `outputs/latent_attacks/*/summary.csv` | ASR and validity comparison. |
| Validation reports | `results/validation/*` | Clean processed data validation. |
| Thesis figures | `thesis_figures/*` | Final visual evidence. |

### Why Integration Is Central

The thesis requires moving between scaled ML space and raw network space:

```text
raw data -> cleaned raw features -> train-only RobustScaler -> scaled model arrays
scaled adversarial example -> inverse_transform -> raw validator -> ASR_valid
```

The saved scaler is therefore not just a preprocessing artifact; it is part of the evaluation definition. Without it, an adversarial `.npz` file cannot be judged for physical validity.

### Content Hashes

The manifest stores SHA-256 hashes for:

- `X_train`
- `X_val`
- `X_test`
- scaler parameters
- label encoder classes

Reason:

- Hashing guards against silent drift between models, attacks, and data arrays.
- It makes it possible to show that classifier, VAE, attack, and validation outputs are tied to a specific processed dataset.

---

## 3.1.2.4 Data Reduction

The project uses several forms of reduction. They serve different purposes.

### Raw-to-Working Dataset Reduction

Raw dataset profile:

| Source | Rows |
|---|---:|
| Raw CSV | 45,019,243 |
| Processed/staged working dataset | 4,429,940 |

Reason:

- Full CICIoT2023 scale is too large for repeated neural training, attack generation, VAE experiments, t-SNE/UMAP, and validity analysis in a thesis workflow.
- Majority classes are extremely over-represented. Capping majority/medium classes reduces computational burden while keeping rare classes.
- Rare and minority classes are preserved rather than discarded.

### Train/Validation/Test Reduction

The 70/10/20 split reduces data per experimental stage:

| Split | Rows | Use |
|---|---:|---|
| Train | 3,100,958 | Fit classifiers, scaler, VAE training. |
| Validation | 442,994 | Early stopping, diagnostics, GMM priors. |
| Test | 885,988 | Final classifier metrics, attacks, validation. |

### Feature Reduction Artifact: Top-24 Set

There is a separate reduced feature artifact directory:

`data/processed_bestfeat_top24/`

It contains a 24-feature representation selected by ANOVA F-score for the 8-category task.

The report records:

| Feature selection fact | Value |
|---|---|
| Method | ANOVA F-score |
| Task | 8-category |
| Base feature count | 39 |
| Selected feature count | 24 |
| Sample size | 200,000 |
| Interaction base | 10 |

Top selected features/terms include:

1. `sq(Number)`
2. `abs(Number)`
3. `Number`
4. `Header_Length*Protocol Type`
5. `abs(Protocol Type)`
6. `Protocol Type`
7. `sq(Protocol Type)`
8. `Header_Length`
9. `abs(Time_To_Live)`
10. `IAT`

Reason:

- The top-24 set is useful as an auxiliary experiment or ablation.
- The main methodology still uses the full 39-feature schema because the validity validator and protocol/mask policies are defined over the full schema.

### Visualization Sampling

EDA dimensionality reductions sample per class:

- PCA plots use up to 3,000 per class.
- t-SNE uses up to 2,000 per class.
- UMAP uses up to 10,000 per class.
- NetDiffuser uses up to 50,000 rows.

Reason:

- t-SNE/UMAP are computationally expensive at millions of rows.
- Stratified visualization sampling preserves class/category visibility.
- Visual outputs are diagnostic, not training data.

### Attack Evaluation Sampling

Several attack scripts select stratified or correctly-classified subsets, for example:

- main attacks: sampled test subsets;
- all-model latent rerun: up to 100 correctly classified test samples per non-benign source class;
- targeted benign latent PGD: up to 100 correctly classified test samples per non-benign source class, 5 restarts.

Reason:

- Attack generation is expensive.
- Only clean-correct samples should be used for ASR, because an attack cannot claim success on a sample the model already misclassified.
- Equal per-source-class sampling improves comparability across classes.

---

## 3.1.2.5 Summary of Processed Data

### Final Processed Array Shapes

Main 39-feature processed arrays:

| File | Shape | Dtype |
|---|---|---|
| `X_train.npy` | `(3,100,958, 39)` | `float32` |
| `X_val.npy` | `(442,994, 39)` | `float32` |
| `X_test.npy` | `(885,988, 39)` | `float32` |
| `y_train.npy` | `(3,100,958,)` | `int32` |
| `y_val.npy` | `(442,994,)` | `int32` |
| `y_test.npy` | `(885,988,)` | `int32` |
| `y_train_cat.npy` | `(3,100,958,)` | `int32` |
| `y_val_cat.npy` | `(442,994,)` | `int32` |
| `y_test_cat.npy` | `(885,988,)` | `int32` |
| `y_train_bin.npy` | `(3,100,958,)` | `int32` |
| `y_val_bin.npy` | `(442,994,)` | `int32` |
| `y_test_bin.npy` | `(885,988,)` | `int32` |

Reduced top-24 arrays:

| File | Shape | Dtype |
|---|---|---|
| `X_train.npy` | `(3,100,958, 24)` | `float32` |
| `X_val.npy` | `(442,994, 24)` | `float32` |
| `X_test.npy` | `(885,988, 24)` | `float32` |

### Category Counts by Split

| Category | Train | Val | Test | Total |
|---|---:|---:|---:|---:|
| Benign | 139,992 | 19,999 | 39,998 | 199,989 |
| BruteForce | 8,766 | 1,252 | 2,504 | 12,522 |
| DDoS | 1,434,940 | 204,993 | 409,984 | 2,049,917 |
| DoS | 468,144 | 66,876 | 133,755 | 668,775 |
| Mirai | 419,972 | 59,996 | 119,992 | 599,960 |
| Recon | 352,470 | 50,353 | 100,705 | 503,528 |
| Spoofing | 260,015 | 37,145 | 74,291 | 371,451 |
| Web | 16,659 | 2,380 | 4,759 | 23,798 |

### Final Clean-Data Invariant

The most important processed-data invariant is:

```text
All 4,429,940 processed rows pass all 49 domain-validity rules after inverse transform.
```

This invariant supports the thesis evaluation because clean data is not being unfairly rejected. When adversarial data fails validation, the failure is caused by the perturbation process.

---

## 3.2 Design (Model) Specification

## 3.2.1 Overview

The model design has three layers:

1. Predictive IDS classifiers.
2. Unconstrained input-space attacks for baseline adversarial vulnerability.
3. VAE-based latent-space framework for validity-aware/manifold-aware adversarial generation.

### Predictive IDS Classifiers

The classifier suite includes:

| Model | Purpose |
|---|---|
| MLP | Simple dense baseline for tabular features. |
| CNN | 1D convolution over ordered feature vector to capture local feature patterns. |
| LSTM | Sequence-style baseline using features as a single-step sequence; extensible to real temporal sequences. |
| Serial CNN-LSTM | CNN feature extraction followed by LSTM sequence modeling. |
| Attention Serial CNN-LSTM | Serial CNN-LSTM with attention over LSTM outputs. |
| DualPath IDS | Parallel CNN and LSTM branches fused with attention. This is the main IDS architecture proposal in `src/classifiers/models.py`. |

The current saved all-model summary reports these results:

| Task | Best accuracy in saved summary | Best macro-F1 in saved summary | Notes |
|---|---:|---:|---|
| Binary | DualPath, 96.78% | MLP, 0.7873 | All models near 96.6%-96.8% accuracy. |
| 8-class | DualPath, 84.59% | LSTM, 0.6470 | MLP, LSTM, and DualPath are close. |
| 34-class | LSTM, 75.58% | LSTM, 0.5747 | Fine-grained task is hardest. |

The exact all-model metrics are:

| Task | Model | Accuracy | Macro-F1 | Weighted-F1 |
|---|---|---:|---:|---:|
| Binary | MLP | 0.967244 | 0.787254 | 0.965270 |
| Binary | CNN | 0.965941 | 0.764048 | 0.962600 |
| Binary | LSTM | 0.967418 | 0.784028 | 0.965074 |
| Binary | Serial CNN-LSTM | 0.965825 | 0.749818 | 0.961296 |
| Binary | DualPath | 0.967772 | 0.784518 | 0.965291 |
| 8-class | MLP | 0.844897 | 0.646890 | 0.839675 |
| 8-class | CNN | 0.834941 | 0.590679 | 0.828780 |
| 8-class | LSTM | 0.845683 | 0.646977 | 0.841349 |
| 8-class | Serial CNN-LSTM | 0.839351 | 0.606939 | 0.835841 |
| 8-class | DualPath | 0.845856 | 0.639857 | 0.839126 |
| 34-class | MLP | 0.749502 | 0.567880 | 0.732268 |
| 34-class | CNN | 0.702391 | 0.523916 | 0.688783 |
| 34-class | LSTM | 0.755759 | 0.574735 | 0.741649 |
| 34-class | Serial CNN-LSTM | 0.744404 | 0.562846 | 0.729296 |
| 34-class | DualPath | 0.753501 | 0.571154 | 0.738125 |

### Attack/Evaluation Models

The adversarial framework has two comparison groups:

| Group | Description | Reason |
|---|---|---|
| Input-space attacks | FGSM, PGD, CW directly perturb scaled features. | Shows how standard tabular attacks can create high `ASR_raw` but invalid traffic. |
| Latent-space attacks | PGD/CW in the VAE latent space with protocol/mask governance. | Tests whether attacks can remain closer to the learned data manifold and feature constraints. |

The key evaluation metrics are:

| Metric | Definition |
|---|---|
| `ASR_raw` | Fraction of clean-correct samples flipped by the attack, without validity filtering. |
| Validity rate | Fraction of adversarial samples passing domain validator and/or protocol/mask checks. |
| `ASR_valid` | Fraction of clean-correct samples that are both flipped and valid. |
| Gap | `ASR_raw - ASR_valid`; this quantifies how much raw attack success is inflated by invalid examples. |
| Protocol validity | Whether protocol features remain valid/consistent. |
| Mask compliance | Whether frozen/partial/full perturbation policy is obeyed. |
| IDSR | In-distribution success/rate estimate from latent outlier checks. |

---

## 3.2.2 Architecture and Training

### Baseline Classifier Architectures

#### MLP

The MLP architecture is:

```text
Input(39) -> Dense(128) -> ReLU -> Dropout(0.3)
          -> Dense(64)  -> ReLU -> Dropout(0.3)
          -> Dense(num_classes)
```

Reason:

- It is a standard tabular baseline.
- It provides a simple reference point for the more structured CNN/LSTM/DualPath models.
- It is fast to train and attack, which makes it useful for sanity checks.

#### CNN

The CNN treats the 39-feature vector as a 1D signal:

```text
Input(batch, 39)
-> reshape(batch, 1, 39)
-> Conv1d(1 -> 32, kernel=3, padding=1)
-> ReLU
-> Conv1d(32 -> 64, kernel=3, padding=1)
-> ReLU
-> AdaptiveMaxPool1d(1)
-> Dense(64)
-> ReLU
-> Dropout(0.3)
-> Dense(num_classes)
```

Reason:

- Adjacent engineered features may contain local relationships such as flags/counts or packet-size aggregates.
- CNNs can learn local patterns with fewer assumptions than fully dense layers.

#### LSTM

The LSTM treats the feature vector as a single-step sequence:

```text
Input(batch, 39) -> reshape(batch, 1, 39)
-> bidirectional LSTM(hidden=64)
-> Dense(64)
-> ReLU
-> Dropout(0.3)
-> Dense(num_classes)
```

Reason:

- It creates a sequence-compatible baseline and can later extend to true time-series traffic windows.
- Bidirectionality allows the model to use both directions of the feature representation.

#### Serial CNN-LSTM

Serial CNN-LSTM:

```text
Input -> Conv1d -> Conv1d -> transpose as sequence -> bidirectional LSTM -> Dense -> classifier
```

Reason:

- This is a common hybrid design.
- It tests whether local convolutional patterns followed by sequential aggregation improve IDS classification.
- The thesis notes a possible information bottleneck because the LSTM only receives CNN-transformed features.

#### Attention Serial CNN-LSTM

This variant adds attention over LSTM outputs:

```text
Input -> CNN -> LSTM -> attention over hidden states -> Dense -> classifier
```

Reason:

- Attention can identify which transformed feature positions matter most.
- It is a baseline for comparing branch-level attention in DualPath IDS.

#### DualPath IDS

DualPath IDS processes the same input in parallel:

```text
                 -> CNN branch  -> h_cnn  -
Input(39)                                      -> attention fusion -> classifier
                 -> LSTM branch -> h_lstm -
```

Reason:

- CNN and LSTM branches capture different inductive biases.
- Branch-level attention allows the model to combine local feature patterns and sequence-style representations.
- It avoids the serial information bottleneck where one representation must pass through the other.

### Classifier Training

`src/classifiers/baseline_experiments.py` records:

- plain `CrossEntropyLoss`;
- Adam optimizer;
- `ReduceLROnPlateau`;
- early stopping on validation loss;
- batch size default 2048;
- 5 epochs in the main script defaults;
- saved histories, metrics, confusion matrices, reports, checkpoints.

Reasons:

- Cross-entropy is appropriate for multi-class classification.
- Adam is stable for tabular neural baselines.
- Scheduler and early stopping reduce overfitting and wasted training.
- Plain cross-entropy avoids double-counting class imbalance after stratified/capped sampling.

### VAE Architecture: MixedInputBetaVAE

The VAE subsystem is centered on `src/vae/model.py`.

Model description:

```text
Encoder input:
  39-dimensional scaled feature vector,
  with Protocol Type handled via protocol embedding.

Encoder:
  continuous features + independent binaries + pseudo-binaries + protocol embedding
  -> MLP hidden layers
  -> mu and logvar for z

Latent:
  z dimension = 16

Decoder:
  shared MLP body
  -> continuous mean/logvar head
  -> independent binary head
  -> protocol classification head
  -> optional pseudo-binary head

Output adapter:
  reconstructs full 39-dimensional scaled vector;
  derives TCP/UDP/ICMP/IGMP from decoded protocol argmax.
```

Default VAE hyperparameters from `src/vae/config.py`:

| Parameter | Value |
|---|---|
| Classes | Benign, BruteForce, DDoS, DoS, Mirai, Recon, Spoofing, Web |
| Latent dim | 16 for every class |
| Beta target | 1.0 |
| Protocol embedding dim | 4 |
| Encoder hidden | `[128, 64]` |
| Decoder hidden | `[64, 128]` |
| Max epochs | 200 |
| Batch size | 512 |
| Learning rate | 0.001 |
| Weight decay | 0.00001 |
| Early stop patience | 10 |
| Grad clip | 5.0 |
| Protocol loss weight | 2.0 |
| Constraint loss weight | 0.1 |
| Structured continuous decoder | true |
| Structured physics decoder | false by default, true in `configs/vae_physics_retrain.json` |

Reason for mixed-input VAE:

- CICIoT features are not homogeneous continuous variables.
- Protocol IDs, binary flags, integer counts, and continuous statistics need different decoder treatment.
- Derived binaries (`TCP`, `UDP`, `ICMP`, `IGMP`) should follow decoded protocol, not be independently generated in conflict with it.

Reason for per-category VAEs:

- A single global VAE would have to model very different distributions for Benign, DDoS, DoS, Mirai, Recon, Spoofing, Web, and BruteForce.
- Per-category models make the manifold more coherent.
- 34-class per-label VAEs would be unstable for rare labels such as `UPLOADING_ATTACK`, `RECON-PINGSWEEP`, and `BACKDOOR_MALWARE`.

### VAE Loss

`src/vae/losses.py` implements a beta-VAE ELBO with additional terms:

```text
loss =
  continuous reconstruction
  + independent binary reconstruction
  + protocol_loss_weight * protocol cross-entropy
  + beta * KL
  + constraint_loss_weight * raw-space constraint loss
  + physics_constraint_loss_weight * physics constraint loss
  + optional raw relative continuous loss
  + optional pseudo-binary reconstruction
```

The beta scheduler linearly warms beta from 0 to the target value.

Reasons:

- KL warmup reduces posterior collapse risk early in training.
- Protocol cross-entropy is weighted because wrong protocol decoding causes multiple downstream binary/protocol validity failures.
- Raw-space constraints encourage generated samples to obey domain rules before postprocessing.
- Physics constraint options allow stronger versions of packet-size consistency when needed.

### VAE Training Outputs

Current VAE summary:

| Class | latent_dim | n_train | n_val | best_val_loss | final_kl | collapsed_dims | pre validity uncond | post validity uncond | protocol acc |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Benign | 16 | 139,992 | 19,999 | 130.66812 | 15.490193 | 3 | 0.0% | 100.0% | 99.44% |
| BruteForce | 16 | 8,766 | 1,252 | 11.290511 | 9.360052 | 1 | 72.1% | 100.0% | 98.32% |
| DDoS | 16 | 1,434,940 | 204,993 | -31.92959 | 16.89655 | 7 | 41.3% | 100.0% | 99.90% |
| DoS | 16 | 468,144 | 66,876 | -36.077427 | 12.857692 | 8 | 0.0% | 100.0% | 99.99% |
| Mirai | 16 | 419,972 | 59,996 | -42.285265 | 10.638453 | 7 | 93.9% | 100.0% | 99.91% |
| Recon | 16 | 352,470 | 50,353 | -24.945076 | 15.663484 | 7 | 94.0% | 100.0% | 99.71% |
| Spoofing | 16 | 260,015 | 37,145 | -8.122171 | 20.118106 | 4 | 0.0% | 100.0% | 99.89% |
| Web | 16 | 16,659 | 2,380 | 23.070889 | 9.229854 | 9 | 0.0% | 100.0% | 99.54% |

Important interpretation:

- `*_validity_pre_pct` is the honest decoder validity immediately after inverse transform.
- `*_validity_pct` is after `raw_postprocess()`.
- `*_repair_pct` is the fraction rescued by postprocess.

Therefore, postprocessed 100% validity should not be overstated as pure decoder quality. It is valid for generating constraint-compliant samples, but pre-postprocess validity is the honest measure of how naturally the decoder satisfies rules.

### VAE Checkpoint Manifest

`vae_run_manifest.json` records checkpoints for all 8 classes:

| Class | Checkpoint |
|---|---|
| Benign | `models/vae/vae_class_0_Benign.pt` |
| BruteForce | `models/vae/vae_class_1_BruteForce.pt` |
| DDoS | `models/vae/vae_class_2_DDoS.pt` |
| DoS | `models/vae/vae_class_3_DoS.pt` |
| Mirai | `models/vae/vae_class_4_Mirai.pt` |
| Recon | `models/vae/vae_class_5_Recon.pt` |
| Spoofing | `models/vae/vae_class_6_Spoofing.pt` |
| Web | `models/vae/vae_class_7_Web.pt` |

The VAE protocol allowlist is:

```text
0, 1, 2, 6, 17, 47
```

The manifest notes that protocol `2` (IGMP) is absent from training data but retained in the allowlist. This is conservative: the protocol validator knows IGMP is allowed even if it is rare or absent in the sampled training split.

---

## 3.2.3 Model Insights and Framework Proposal

### Main Insight 1: Clean Data Is Valid

The full processed dataset validates at 100%. This is essential because the thesis argument depends on showing that invalidity is introduced by attacks, not by the dataset itself.

### Main Insight 2: Raw Input-Space ASR Can Be Misleading

The main unconstrained attack summary shows:

| Framing | Attack | eps | `ASR_raw` | Validity rate | `ASR_valid` |
|---|---|---:|---:|---:|---:|
| Binary | PGD | 0.30 | 37.8% | 0.0% | 0.0% |
| 8-class | PGD | 0.30 | 74.6% | 0.0% | 0.0% |
| 34-class | PGD | 0.30 | 88.3% | 0.0% | 0.0% |

This supports the thesis claim that ignoring domain constraints inflates adversarial threat estimates.

The impossible-traffic exhibits show typical invalid changes:

- protocol values become non-integer or outside `{0, 1, 2, 6, 17, 47}`;
- binary protocol/application indicators become fractional or negative;
- `Tot sum`, `Rate`, `Max`, `AVG`, `Std`, or `Variance` become negative;
- `Min > Max`;
- `AVG` leaves `[Min, Max]`;
- `Variance` no longer matches `Std^2`;
- frozen features change even though they should not.

### Main Insight 3: Latent Attacks Trade Raw ASR for Validity

The latest all-model latent/input rerun (`outputs/latent_attacks/all_models_rerun_20260524_185905_seed42/summary.csv`) compares latent and input attacks across MLP, CNN, LSTM, CNN-LSTM, and DualPath.

| Model | Attack | n | ASR overall | ASR valid only | Protocol validity | Mask compliance | Joint validity | IDSR |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| CNN | latent-PGD | 500 | 25.00% | 25.00% | 100.00% | 100.00% | 100.00% | 89.80% |
| CNN | latent-CW | 500 | 25.80% | 25.80% | 100.00% | 100.00% | 100.00% | 89.60% |
| CNN | input-PGD | 500 | 92.00% | 0.00% | 22.80% | 0.00% | 0.00% | n/a |
| CNN | input-CW | 500 | 88.40% | 0.00% | 47.00% | 0.00% | 0.00% | n/a |
| CNN-LSTM | latent-PGD | 600 | 17.50% | 17.50% | 100.00% | 100.00% | 100.00% | 90.33% |
| CNN-LSTM | latent-CW | 600 | 17.17% | 17.17% | 100.00% | 100.00% | 100.00% | 92.33% |
| CNN-LSTM | input-PGD | 600 | 95.50% | 0.00% | 23.33% | 0.00% | 0.00% | n/a |
| CNN-LSTM | input-CW | 600 | 97.33% | 0.00% | 88.17% | 0.00% | 0.00% | n/a |
| DualPath | latent-PGD | 700 | 23.57% | 23.57% | 100.00% | 100.00% | 100.00% | 79.29% |
| DualPath | latent-CW | 700 | 25.43% | 25.43% | 100.00% | 100.00% | 100.00% | 92.43% |
| DualPath | input-PGD | 700 | 95.00% | 0.00% | 22.29% | 0.00% | 0.00% | n/a |
| DualPath | input-CW | 700 | 80.57% | 0.00% | 68.14% | 0.00% | 0.00% | n/a |
| LSTM | latent-PGD | 700 | 26.86% | 26.86% | 100.00% | 100.00% | 100.00% | 84.29% |
| LSTM | latent-CW | 700 | 25.00% | 25.00% | 100.00% | 100.00% | 100.00% | 92.14% |
| LSTM | input-PGD | 700 | 96.71% | 0.00% | 28.29% | 0.00% | 0.00% | n/a |
| LSTM | input-CW | 700 | 95.71% | 0.00% | 59.57% | 0.00% | 0.00% | n/a |
| MLP | latent-PGD | 700 | 22.71% | 22.71% | 100.00% | 100.00% | 100.00% | 90.29% |
| MLP | latent-CW | 700 | 25.14% | 25.14% | 100.00% | 100.00% | 100.00% | 91.14% |
| MLP | input-PGD | 700 | 95.00% | 0.00% | 26.14% | 0.00% | 0.00% | n/a |
| MLP | input-CW | 700 | 89.86% | 0.00% | 60.14% | 0.00% | 0.00% | n/a |

Interpretation:

- Input attacks achieve very high `ASR_raw`, but their `ASR_valid` collapses to 0% under mask/protocol validity.
- Latent attacks have lower ASR but preserve protocol validity and perturbation-mask compliance.
- This gives the thesis a more realistic adversarial framework: success should be counted only when the example is both evasive and valid.

### Why Restart-Aware Latent Attacks Improve ASR

The restart-aware rerun uses the beta05 Gaussian VAE line:

`outputs/latent_attacks/new_vae_attacks_gaussian_anticollapse_beta05_freebits01_20260529_173512_20260530_234841_seed42/summary.csv`

This run keeps the same validity-aware philosophy but makes the latent attack search stronger. Instead of running each attack once from a single latent starting point, each sample is attacked from five starts:

```text
encoded, jitter, gmm, jitter, gmm
```

The attack is still performed in VAE latent space. For a clean scaled traffic row `x`, the class-specific VAE encoder maps the row to a latent code:

```text
x_original -> encoder -> z_orig
```

Latent PGD or latent CW then searches for a nearby latent code `z_adv` such that the decoded row fools the classifier:

```text
z_adv -> decoder -> x_adv -> classifier prediction changes
```

The important change is that the optimizer is not forced to start only from `z_orig`. It is given several plausible initial latent positions, and the best candidate is selected per sample after all restarts finish.

The three restart types have different roles:

| Restart type | Starting point | Purpose |
|---|---|---|
| `encoded` | `z_start = z_orig` | Local attack from the original encoded point. This is the most conservative start and tends to preserve similarity to the original sample. |
| `jitter` | `z_start = z_orig + Uniform(-epsilon, epsilon)` | Random local exploration around the original latent point. This helps avoid one unlucky gradient path. |
| `gmm` | `z_start` sampled from a class-specific latent GMM and clipped to the class radius | Manifold-guided exploration from regions where real validation samples from the same class tend to live. |

A GMM is a Gaussian Mixture Model. In this pipeline it is fitted in latent space, not raw feature space. For each 8-class category, validation examples from that category are encoded through the corresponding per-class VAE:

```text
X_val[class] -> encoder -> latent posterior means z_mu
```

Then a 5-component Bayesian Gaussian mixture is fitted to those latent posterior means:

```text
p(z | class) = w1 N(mu1, Sigma1)
             + w2 N(mu2, Sigma2)
             + ...
             + w5 N(mu5, Sigma5)
```

The mixture weights `w_k`, means `mu_k`, and covariance matrices `Sigma_k` approximate where that class lives in the VAE latent space. During an attack, a `gmm` restart samples from this learned class distribution. This is different from blind random noise: random jitter explores a small box around the current sample, while the GMM restart proposes latent points that are statistically typical for the class manifold learned from validation data.

The GMM priors are cached under `outputs/latent_gmm_priors/`. They are not target-class classifiers and they do not use test labels to optimize the attack. They are only a density model over the source class's VAE latent codes.

For latent PGD, the search remains budgeted. After each PGD step, the latent code is projected back into the class-specific `L_inf` ball:

```text
z_adv in [z_orig - epsilon_class, z_orig + epsilon_class]
```

The class-specific epsilon values remove a one-size-fits-all bottleneck:

| Class | Epsilon |
|---|---:|
| Benign | 0.3 |
| BruteForce | 0.3 |
| DDoS | 0.8 |
| DoS | 0.8 |
| Mirai | 0.8 |
| Recon | 0.5 |
| Spoofing | 0.5 |
| Web | 1.0 |

The attack also uses `alpha = 0.1 * epsilon_class` by default, so larger class budgets receive proportionally larger PGD steps. Adaptive PGD then halves the step size when the loss stops improving, which reduces oscillation near a classifier boundary.

For latent CW, the same restart pool is used as initialization. The restart seeds are clipped to the class radius before optimization, but CW then optimizes its own latent L2-style objective. This is why CW also benefits from GMM starts even though it is not a projected-gradient attack.

After all restarts are evaluated, candidate selection is per sample, not per batch. The selection rule is:

1. Prefer a candidate that flips the classifier.
2. If both candidates tie on attack success, prefer the one that is jointly valid.
3. If both tie on success and validity, prefer lower input-space L2 distance.
4. If still tied, prefer the candidate with the stronger classifier loss objective.

This explains the ASR improvement. A single-start attack can fail because it begins in a poor local region of latent space. Restart-aware latent search tries several doors into the classifier decision boundary. The GMM restarts are especially useful because they begin from plausible class-manifold regions rather than arbitrary latent noise.

The beta05 Gaussian rerun shows the effect clearly:

| Model | Attack | Previous ASR | Restart-aware ASR | Raw/Joint validity | Mean selected restart |
|---|---|---:|---:|---:|---:|
| CNN | latent-PGD | 16.80% | 32.00% | 94.20% | 1.62 |
| CNN | latent-CW | 16.40% | 33.20% | 95.20% | 1.74 |
| CNN-LSTM | latent-PGD | 21.33% | 40.50% | 94.00% | 1.83 |
| CNN-LSTM | latent-CW | 22.33% | 41.17% | 96.17% | 2.04 |
| DualPath | latent-PGD | 17.00% | 36.57% | 85.71% | 1.99 |
| DualPath | latent-CW | 16.86% | 31.71% | 87.71% | 2.07 |
| LSTM | latent-PGD | 21.71% | 37.29% | 87.14% | 1.95 |
| LSTM | latent-CW | 22.86% | 39.71% | 90.29% | 2.09 |
| MLP | latent-PGD | 19.86% | 38.29% | 87.43% | 1.95 |
| MLP | latent-CW | 18.43% | 37.00% | 88.43% | 2.20 |

The ASR increase is therefore an optimization effect: more starts, more informed starts, class-specific latent radii, and adaptive PGD. It is not caused by relaxing the attack-success definition. `ASR_overall` is still counted as a classifier label flip. The new raw/joint validity rate is stricter than the earlier latent summary because it includes full raw G1-G8 validation in addition to protocol and perturbation-mask checks. This is why the new joint-validity column is no longer automatically 100%.

### Statistical Comparison of Latent PGD vs Latent CW

The latest McNemar summary reports no significant difference between latent-PGD and latent-CW at the saved sample sizes:

| Model | p-value | Interpretation |
|---|---:|---|
| CNN | 0.635256 | Not significantly different |
| CNN-LSTM | 0.871131 | Not significantly different |
| DualPath | 0.208413 | Not significantly different |
| LSTM | 0.136641 | Not significantly different |
| MLP | 0.061116 | Not significantly different, close to threshold |

Reason this matters:

- Both latent attacks are useful; the difference is not only attack algorithm but validity-preserving attack space.
- The thesis can focus on the framework distinction (input-space invalid vs latent-space valid) rather than overclaiming PGD/CW superiority.

### Targeted Benign Latent PGD

The latest targeted run:

`outputs/latent_attacks/targeted_benign_pgd_20260529_065027_seed42/summary.csv`

uses:

| Parameter | Value |
|---|---|
| Target class | Benign |
| Attack | targeted benign latent PGD |
| Epsilon | 0.5 |
| Alpha | 0.05 |
| Steps | 40 |
| Restarts | 5 |
| Restart strategy | encoded + jitter + GMM |
| GMM components | 5 |
| Source classes | all non-benign 8-class categories |
| Samples per source class | up to 100 correctly classified |

Summary:

| Model | n | Benign target success | Joint target success | Raw G1-G8 validity | Protocol validity | Mask compliance |
|---|---:|---:|---:|---:|---:|---:|
| CNN | 500 | 6.40% | 4.00% | 59.80% | 100.00% | 100.00% |
| CNN-LSTM | 600 | 3.83% | 2.50% | 49.67% | 100.00% | 100.00% |
| DualPath | 700 | 5.57% | 3.57% | 39.14% | 100.00% | 100.00% |
| LSTM | 700 | 3.71% | 2.43% | 50.00% | 100.00% | 100.00% |
| MLP | 700 | 4.29% | 3.14% | 48.86% | 100.00% | 100.00% |

Interpretation:

- Targeting Benign is much harder than untargeted evasion.
- Protocol and mask validity remain perfect by construction, but raw G1-G8 validity is not uniformly perfect.
- This is a useful Phase 2 result because it shows the framework can ask stricter adversarial questions than "any wrong label."

### Proposed Framework

The final framework proposal for the thesis should be:

1. Start with clean, validator-compliant data.
2. Train IDS classifiers under binary, category, and fine-grained label framings.
3. Evaluate standard input-space attacks only as a diagnostic baseline.
4. Always inverse-transform adversarial examples to raw feature space.
5. Count attack success twice:
   - `ASR_raw`, before validity filtering;
   - `ASR_valid`, after protocol/mask/domain validation.
6. Use impossible-traffic exhibits to explain why invalid attacks should not be counted as realistic evasion.
7. Use per-category beta-VAEs to move adversarial search into a learned traffic manifold.
8. Report both decoder honesty (`validity_pre_postprocess`) and repaired validity (`validity_postprocess`).
9. Use statistical summaries, latent geometry, and distributional fidelity figures to judge whether generated attacks are plausible, not only evasive.

This framework is stronger than a standard adversarial-attack benchmark because it rejects examples that only fool the classifier by leaving the traffic domain.

---

# Appendix A: Key Output and Artifact Log

## Data Artifacts

| Path | Status / meaning |
|---|---|
| `data/ciciot2023/ciciot2023_base.csv` | Raw CICIoT2023 CSV, 39 features + `Label`, about 10.1 GB on disk. |
| `data/processed/raw_loaded.parquet` | Staged 4,429,940-row working dataset. |
| `data/processed/X_train.npy`, `X_val.npy`, `X_test.npy` | Main scaled feature arrays, 39 features. |
| `data/processed/y_*.npy` | 34-class, 8-class, and binary labels. |
| `data/processed/scaler.pkl` | Train-fitted `RobustScaler`. |
| `data/processed/label_encoder.pkl`, `category_encoder.pkl` | Saved label decoders. |
| `data/processed/perturbation_mask.npy` | 39-feature perturbation policy mask. |
| `data/processed/netdiffuser_categorization.json` | Discrete/relative feature categorization. |
| `data/processed/processed_data_validity_report.json` | Current processed data validity: 100%. |
| `data/processed/near_zero_iqr_features.json` | Near-zero-IQR feature governance. |
| `data/processed/content_hashes.json` | Dataset/scaler/encoder hashes. |
| `data/processed_bestfeat_top24/*` | Reduced 24-feature artifact set and selection report. |

## Tables

| Path | Role |
|---|---|
| `tables/T0_imbalance_audit.*` | Raw class imbalance and thesis strategy by rarity tier. |
| `tables/T1_feature_schema.*` | Schema, feature descriptions, ranges, stats. |
| `tables/T2a_class_counts.*` | 34-class counts. |
| `tables/T2b_category_counts.*` | 8-category counts. |
| `tables/T3_summary_stats.*` | Feature summary statistics. |
| `tables/T4_clean_validity.*` | Older validity audit; superseded by current validation reports. |
| `tables/T5_sample_rows_wide.csv` | Sample rows. |
| `tables/T5_sample_rows_transposed.md` | Sample rows in thesis-friendly transposed view. |
| `tables/T6_feature_categorization.*` | NetDiffuser grouping. |
| `tables/F3_correlation_matrix.csv` | Spearman correlation matrix. |
| `tables/F4_pca_loadings.csv` | PCA loadings. |
| `tables/impossible_traffic_*` | Invalid adversarial examples and slide-ready tables. |
| `tables/supervisor_onepager.md` | Short supervisor brief summarizing ASR collapse under validity. |

## Main Result Artifacts

| Path | Role |
|---|---|
| `results/all_models_all_tasks_summary.json` | Main classifier summary across binary, 8-class, 34-class tasks. |
| `results/*classification_report.*` | Per-model classification reports. |
| `results/attacks/attack_summary.csv` | Raw attack metrics for baseline attacks. |
| `results/attacks/attack_summaries.csv` | Validity-filtered attack summaries. |
| `results/attacks/shock_table.csv` | Raw-vs-valid thesis table. |
| `results/attacks/violation_breakdown.csv` | Rule violation breakdown for adversarial examples. |
| `results/attacks/thesis_bundle.*` | Final bundle and CSV summaries for thesis visualizations. |
| `results/validation/full_dataset_validation_report.txt` | Full clean-data validation report. |
| `results/validation/per_rule_breakdown.csv` | Rule-by-rule clean-data validation. |
| `results/validation/per_class_validity.csv` | Per-class clean-data validation. |
| `results/validation/synthetic_corruption_tests.csv` | Validator corruption tests. |
| `results/vae/summary.csv`, `summary.md` | VAE training/diagnostics summary. |
| `results/vae/diagnostics_*.json` | Per-category VAE diagnostics. |
| `results/vae/training_log*.txt` | VAE training logs. |

## Model Checkpoints

| Path | Role |
|---|---|
| `models/mlp_*.pt` | MLP checkpoints for binary, 8-class, 34-class. |
| `models/cnn_*.pt` | CNN checkpoints. |
| `models/lstm_*.pt` | LSTM checkpoints. |
| `models/serial_*.pt` | Serial CNN-LSTM checkpoints. |
| `models/dualpath_*.pt` | DualPath checkpoints. |
| `models/vae/vae_class_*_*.pt` | Per-category VAE checkpoints. |
| `checkpoints/cvae/*` | Earlier CVAE checkpoints and metadata. |

## Figure Artifacts

| Folder | Role |
|---|---|
| `figures/` | EDA figures and attack-validity figure. |
| `figures/appendix/` | Appendix UMAP and perturbation figures. |
| `thesis_figures/` | Final thesis figures for VAE/latent attacks/fidelity/statistics. |
| `thesis_figures/benign_overlay_fidelity/` | Per-category benign overlay fidelity outputs. |
| `results/vae/*.png` | VAE training curves. |

## Latent Attack Run Folders

| Run folder | Files | Role |
|---|---:|---|
| `phase0_20260524_025133_seed42` | 2 | Early latent infrastructure/mask smoke run. |
| `phase0_20260524_025504_seed42` | 2 | Follow-up phase0 latent infrastructure run. |
| `phase1_20260524_025838_seed42` | 2 | Early phase1 sanity check. |
| `phase1_20260524_025856_seed42` | 3 | Phase1 results and log. |
| `phase1_20260524_025934_seed42` | 3 | Phase1 rerun. |
| `phaseA_20260524_032553_seed42` | 4 | VAE failure diagnostics and BruteForce mu histograms. |
| `phaseBCD_*` | 3 each | Iterative VAE/attack diagnostics. |
| `phase1_fix_final_20260524_054835_seed42` | 3 | Final phase1 fix run. |
| `dos_targeted_search_20260524_055015_seed42` | 3 | DoS targeted search. |
| `phase2_20260524_055937_seed42` | 10 | Latent PGD, 7 class `.npz` files plus summary. |
| `phase3_20260524_180113_seed42` | 10 | Latent CW, 7 class `.npz` files plus summary. |
| `phase4_20260524_182644_seed42` | 17 | Input PGD/CW baseline comparison. |
| `all_models_rerun_20260524_185905_seed42` | 21 | Main all-model latent/input rerun with stats. |
| `new_vae_attacks_gaussian_anticollapse_beta05_freebits01_20260529_173512_20260530_234841_seed42` | 8 | Restart-aware beta05 latent PGD/CW rerun with GMM restarts, class-specific epsilon, raw G1-G8 validity, and improved ASR. |
| `targeted_benign_pgd_20260529_065027_seed42` | 39 | Latest targeted benign latent PGD run across five classifiers. |
| `latent_gmm_priors/` | 35 pkl files | Cached Bayesian/GMM latent priors per class, checkpoint hash, and sample cap. |

---

# Appendix B: Known Inconsistencies and How to Cite Them

1. `tables/T4_clean_validity.md` reports 81.3724% clean validity, but current validation reports show 100%.
   - Use current reports for final thesis claims.
   - Treat T4 as an older EDA artifact unless it is regenerated.

2. `logs/preprocessing.log` is currently empty.
   - Use `config/run_manifest.json`, `data/processed/processed_data_validity_report.json`, and saved arrays as authoritative preprocessing evidence.

3. Some older docs (`guide.md`, possibly older prompts/notes) mention older flat source paths.
   - Current source code is organized under `src/preprocessing`, `src/classifiers`, `src/attack`, `src/evaluation`, and `src/vae`.

4. Some baseline log folders under `logs/baselines/` are small sampled/smoke runs.
   - Use `results/all_models_all_tasks_summary.json` for the main classifier metrics in this report.

5. VAE postprocessed validity is 100% for all classes, but pre-postprocess validity varies strongly.
   - In thesis prose, call pre-postprocess validity the honest decoder metric and postprocess validity the repaired validity metric.

---

# Appendix C: LaTeX Conversion Notes

Suggested final LaTeX section mapping:

| Markdown section | LaTeX target |
|---|---|
| `3.1 Design Overview or Methodology Overview` | `\section{Methodology}` or `\subsection{Design Overview}` |
| `3.1.1 Data Collection / Input` | `\subsubsection{Data Collection and Input}` |
| `3.1.2 Data Preprocessing and Analysis` | `\subsubsection{Data Preprocessing and Analysis}` |
| `3.1.2.1 Data Cleaning` | `\paragraph{Data Cleaning}` |
| `3.1.2.2 Data Transformation` | `\paragraph{Data Transformation}` |
| `3.1.2.3 Data Integration` | `\paragraph{Data Integration}` |
| `3.1.2.4 Data Reduction` | `\paragraph{Data Reduction}` |
| `3.1.2.5 Summary of Processed Data` | `\paragraph{Summary of Processed Data}` |
| `3.2 Design (Model) Specification` | `\subsection{Design Specification}` |
| `3.2.1 Overview` | `\subsubsection{Overview}` |
| `3.2.2 Architecture and Training` | `\subsubsection{Architecture and Training}` |
| `3.2.3 Model Insights and Framework Proposal` | `\subsubsection{Model Insights and Framework Proposal}` |

Recommended figures/tables to cite:

- Dataset imbalance: `figures/F1_class_distribution.pdf`
- Feature distributions: `figures/F2_feature_distributions.pdf`
- Correlation/NetDiffuser: `figures/F3_correlation_heatmap.pdf`, `figures/F7a_dendrogram.pdf`, `tables/T6_feature_categorization.csv`
- Validity collapse: `figures/asr_raw_vs_valid.png`, `results/attacks/shock_table.csv`
- Latent/input comparison: `thesis_figures/radar_multimodel.pdf`, `thesis_figures/category_asr_heatmap.pdf`, `results/attacks/thesis_bundle.multimetric.csv`
- Distributional fidelity: `thesis_figures/distributional_fidelity.png`, `thesis_figures/distributional_fidelity_pgd.png`, `thesis_figures/kde_top10_features.pdf`
- Latent geometry: PCA/t-SNE/UMAP PDFs in `thesis_figures/`
- VAE reconstruction: `thesis_figures/vae_reconstruction_error_hist.pdf`, `thesis_figures/vae_reconstruction_mae_by_feature.pdf`, `thesis_figures/vae_reconstruction_scatter.pdf`

---

# Appendix D: Decision Rationale Summary

| Decision | Reason |
|---|---|
| Use CICIoT2023 | Large modern IoT intrusion dataset with diverse attack classes and severe imbalance, suitable for IDS robustness study. |
| Use 39-feature schema | Matches actual CSV header and validator; avoids relying on mismatched older schema variants. |
| Map 34 labels to 8 categories | Stabilizes rare classes and supports category-level VAE training while retaining 34-class fine-grained evaluation. |
| Cap majority/medium classes | Makes repeated training/attacks feasible while preserving minority/rare labels. |
| Drop inf/nan | Neural models and scalers require finite input. |
| Clip at 99.99th percentile | Controls extreme heavy-tail leverage without deleting real attack rows. |
| Round integer/binary features | Restores domain semantics before model training and validation. |
| Validate clean processed data | Establishes that invalidity later is attack-induced. |
| Use train-only `RobustScaler` | Handles heavy tails better than StandardScaler/MinMaxScaler and avoids leakage. |
| Save class weights but exclude from baseline loss | Stratified/capped training already changes imbalance; weights remain for reference. |
| Skip SMOTE/ADASYN | Synthetic interpolation would likely violate binary, integer, protocol, and cross-feature constraints. |
| Save perturbation mask | Makes mutability assumptions explicit and auditable. |
| Freeze protocol/application indicators | These are not freely mutable without changing traffic semantics. |
| Use partial caps for flags/counts | They are attacker-influenced but discrete/concentrated. |
| Keep raw-space validator | Required to compute realistic `ASR_valid`. |
| Compare input vs latent attacks | Separates high raw ASR from physically plausible evasion. |
| Use per-category beta-VAEs | Attack families have different manifolds; 34-class VAEs are too sparse for rare labels. |
| Treat postprocess validity carefully | It measures repaired validity, not pure decoder validity. |
| Use bootstrap/McNemar/statistical summaries | Avoids overclaiming from raw point estimates. |
