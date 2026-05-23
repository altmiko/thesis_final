# Codex Knowledge Base

## Purpose

This repository is a bachelor's thesis workspace for studying VAE-based, on-manifold adversarial attacks against ML-based network intrusion detection on the CICIoT2023 dataset. The core thesis contrast is:

- `ASR_raw`: attack success rate without domain constraints
- `ASR_valid`: attack success rate after inverse-transforming adversarial examples and validating them in raw feature space

The repo has moved beyond the original EDA/preprocessing-only phase. It now contains:

- full EDA and preprocessing code
- baseline classifier training code
- unconstrained adversarial attack runners
- post-attack validity analysis
- a newer per-category VAE training and diagnostics pipeline

## Environment And Execution Defaults

- Repo root: `D:/thesis_final`
- Preferred Python: `C:/Users/T2530985/.conda/envs/thesis/python.exe`
- Conda env: `thesis`
- Default seed: `42`
- GPU-aware scripts usually accept `--device cuda`
- For future script execution, use Git Bash rather than PowerShell shell syntax

Canonical Git Bash pattern:

```bash
/c/Users/T2530985/.conda/envs/thesis/python.exe src/path/to/script.py
```

Windows invocation from PowerShell when needed:

```powershell
& 'C:\Program Files\Git\bin\bash.exe' -lc 'cd /d/thesis_final && /c/Users/T2530985/.conda/envs/thesis/python.exe src/path/to/script.py'
```

## Current Repository Shape

The project was reorganized into subpackages. `REORGANIZATION.md` documents the move from flat `src/` files into:

- `src/preprocessing/`
- `src/classifiers/`
- `src/attack/`
- `src/evaluation/`
- `src/vae/`

Important: `CLAUDE.md` still describes the older flat-file layout in several places, so the current code should be read from the package paths above.

## High-Level Pipeline

Current workflow:

1. Raw dataset intake
2. EDA tables and figures
3. Preprocessing into train/val/test arrays
4. Baseline classifier training for binary, 8-class, and 34-class tasks
5. Unconstrained adversarial attacks in scaled feature space
6. Inverse-transform + validator-based domain validity analysis
7. Per-category VAE training and diagnostics

## Dataset And Schema

- Raw source file: `data/ciciot2023/ciciot2023_base.csv`
- Size: about 10.1 GB in this workspace snapshot
- Header shows 39 features plus `Label`
- Row count observed from file scan: about 45,019,244 rows
- Main staged analysis table: `data/processed/raw_loaded.parquet`
- Working sampled dataset size throughout the thesis pipeline: `4,429,940` rows

Schema in `src/preprocessing/feature_groups.py`:

- 39 features
- label column: `Label`
- dataset labels are uppercase at 34-class level
- `CATEGORY_MAP` reduces 34 classes to 8 categories:
  - `Benign`
  - `BruteForce`
  - `DDoS`
  - `DoS`
  - `Mirai`
  - `Recon`
  - `Spoofing`
  - `Web`

Feature governance encoded there:

- `BINARY_FEATURES`
- `INTEGER_FEATURES`
- `IMMUTABLE_FEATURES`
- `QUASI_IMMUTABLE_FEATURES`
- `MUTABLE_FEATURES`
- `FULL_PERTURBABLE_OVERRIDE_FEATURES`
- near-zero-IQR freeze policy constants
- base `PERTURBATION_MASK`

Notable schema detail:

- `Time_To_Live` and `IGMP` are present
- feature count is 39, not the older larger schema variants

## Source Code Map

### `src/preprocessing/feature_groups.py`

Central schema authority. Defines feature names, label column, category map, mutability policy, and perturbation-mask defaults. Almost every other part of the repo depends on this module.

### `src/attack/validator.py`

Raw-space domain validator. Main API: `validate_batch(X, feature_names)`.

Rules cover:

- non-negativity
- protocol validity
- binary constraints
- protocol-indicator consistency
- `Min <= AVG <= Max`
- `Variance ~= Std^2`
- TTL range
- packet count positivity/integrality

This validator is the backbone of `ASR_valid`.

### `src/preprocessing/netdiffuser_categorization.py`

Implements NetDiffuser-style feature partitioning using:

- absolute Spearman correlation
- hierarchical clustering
- CH-score search over cut heights from `0.1` to `1.0`
- preference for nontrivial local maxima

Outputs discrete vs relative feature groups plus diagnostic curves and correlation matrices.

### `src/preprocessing/pipeline.py`

Current preprocessing pipeline. It is a script-style file rather than a function-based module.

What it does:

- loads `data/processed/raw_loaded.parquet`
- drops inf/nan
- clips features at the `99.99` percentile
- rounds integer and binary-like features
- runs post-clean validation and writes `processed_data_validity_report.json`
- label-encodes 34-class and 8-class targets
- derives binary target
- stratified split `70/10/20`
- fits `RobustScaler` on train only
- writes scaled arrays, encoders, weights, perturbation mask, content hashes, near-zero-IQR report, and run manifest

Important implementation detail:

- paths are still hardcoded to `D:/thesis_final/...`

### `src/classifiers/models.py`

Contains the baseline architectures:

- `SimpleMLP`
- `CNNOnly`
- `LSTMOnly`
- `SerialCNNLSTM`
- `AttentionSerialCNNLSTM`
- `DualPathIDS`
- helper branches/attention classes

Factory:

- `get_model(model_type, num_features, num_classes)`

### `src/classifiers/baseline_experiments.py`

Trains baseline classifiers on processed arrays.

Behavior:

- audits processed data first
- trains sequentially for binary, 8-class, and 34-class tasks
- uses plain `CrossEntropyLoss`
- explicitly records the decision to exclude class weights
- uses Adam, validation monitoring, scheduler, and early stopping
- saves metrics, confusion matrices, histories, and leaderboard outputs under `logs/baselines/`

### `src/attack/adversarial_attacks.py`

Attack helpers and custom tabular attack variants.

Key points:

- custom `TabularFGSM`, `TabularPGD`, and `TabularCW`
- removes image-style `[0,1]` clamping
- loads classifier checkpoints and infers model type from filename
- `run_attack()` returns clean/adversarial features plus predictions
- `run_attack_with_restarts()` exists for restart-based evaluation
- perturbation mask is validated for shape but still not applied

That last point is deliberate and consistent with the thesis claim: attacks are intentionally unconstrained in scaled space.

### `src/attack/run_attacks.py`

Runs the main attack suite for the MLP checkpoints across:

- binary
- 8-class
- 34-class

Attack grid:

- FGSM with `eps` `0.05`, `0.10`, `0.30`
- PGD with `eps` `0.05`, `0.10`, `0.30`
- CW

Outputs go to `results/attacks/`.

### `src/attack/run_attacks_8class_models.py`

Runs 8-class attacks across multiple model families, mainly for cross-architecture comparison and restart analysis.

### `src/evaluation/validity_analysis.py`

Core utilities for:

- parsing attack filenames
- inverse-transforming scaled artifacts back to raw space
- validating adversarial examples
- computing `ASR_valid`
- generating impossible-traffic exhibits

### `src/evaluation/run_validity_analysis.py`

Post-attack analysis driver.

Outputs:

- `results/attacks/shock_table.csv`
- `results/attacks/violation_breakdown.csv`
- `figures/asr_raw_vs_valid.png`
- `tables/impossible_traffic_{model}_{attack}.csv`

### `src/evaluation/validate_full_dataset.py`

Runs chunked validation on the processed dataset splits and produces:

- `results/validation/full_dataset_validation_report.txt`
- `results/validation/per_rule_breakdown.csv`
- `results/validation/per_class_validity.csv`
- `results/validation/synthetic_corruption_tests.csv`

### Other evaluation utilities

- `analyze_attack_restarts.py`: summarizes restart experiments and perturbation stats
- `plot_attack_restarts.py`: plots restart-analysis figures
- `compact_exhibits.py`: compact adversarial report assembly
- `sample_exhibit.py`: supervisor-ready sample outputs
- `delta_report.py`: per-feature delta HTML reporting
- `export_slide_exhibits.py`: slide-ready exhibit extraction
- `eda_tables.py`, `eda_figures_part1.py`, `eda_figures_part2.py`: EDA generation scripts

### `src/vae/`

This package is the newer VAE subsystem.

Files and roles:

- `config.py`: class list and default hyperparameters for per-class beta-VAEs
- `schema.py`: protocol-aware partitioning, scaler-aware raw/scaled conversion, postprocess rules
- `dataset.py`: per-class dataset wrapper
- `model.py`: `MixedInputBetaVAE`
- `losses.py`: ELBO and beta scheduler
- `diagnostics.py`: posterior collapse, reconstruction, protocol accuracy, and validity diagnostics
- `train.py`: single-class training loop
- `train_all.py`: orchestrates multi-class training, diagnostics, manifests, and gates
- `_rediag.py`: reruns diagnostics on existing checkpoints

Notable VAE design detail from `schema.py`:

- protocol is treated specially
- TCP/UDP/ICMP/IGMP binaries are derived from protocol rather than modeled independently in the same way as other indicators
- `raw_postprocess()` enforces validator-aligned constraints after inverse transform

## Data, Models, And Artifact Layout

### `data/processed/`

Key files observed:

- `X_train.npy`, `X_val.npy`, `X_test.npy`
- `y_train.npy`, `y_val.npy`, `y_test.npy`
- `y_train_cat.npy`, `y_val_cat.npy`, `y_test_cat.npy`
- `y_train_bin.npy`, `y_val_bin.npy`, `y_test_bin.npy`
- `scaler.pkl`
- `label_encoder.pkl`
- `category_encoder.pkl`
- `class_names.json`
- `category_names.json`
- `class_to_category.json`
- `class_weights_34.npy`, `class_weights_8.npy`, `class_weights_2.npy`
- `class_weights_named.json`
- `perturbation_mask.npy`
- `content_hashes.json`
- `near_zero_iqr_features.json`
- `clean_data_validity_report.json`
- `processed_data_validity_report.json`
- `netdiffuser_categorization.json`

### `data/processed_bestfeat_top24/`

Separate feature-selection artifact set for a 24-feature reduced representation. `feature_selection_report.json` shows:

- method: `ANOVA F-score`
- task: `8-category`
- selected feature count: `24`
- very strong importance for `Number`, `Protocol Type`, `Header_Length`, `Time_To_Live`, `IAT`, `HTTPS`, and `Tot sum`

### `models/`

Classifier checkpoints present for:

- `mlp`
- `cnn`
- `lstm`
- `serial`
- `dualpath`

Each exists for:

- binary
- 8-class
- 34-class

### `models/vae/`

Per-class VAE checkpoints exist for:

- `Benign`
- `BruteForce`
- `DDoS`
- `DoS`
- `Mirai`
- `Recon`
- `Spoofing`

No `Web` VAE checkpoint is present in `models/vae/` in this snapshot, even though the VAE config class list includes it.

### `results/attacks/`

Contains:

- `.npz` attack artifacts
- attack summary CSVs
- restart summary CSVs
- compact exhibit text outputs
- sample exhibits
- shock/violation tables

### `results/validation/`

Contains processed-dataset validation outputs, including the full text report and rule/category breakdowns.

### `results/vae/`

Contains:

- VAE diagnostics JSON per class
- training logs
- training curves PNGs

### `logs/baselines/`

Contains run-specific folders with:

- `run_config.json`
- `history.json`
- `metrics_test.json`
- `confusion_matrix_test.csv`
- `best_model.pt`
- leaderboard files

## Key Current Metrics

### Baseline classifier summary

From `results/all_models_all_tasks_summary.json`:

- Binary accuracy is about `96.6%` to `96.8%`
- 8-class accuracy is about `83.5%` to `84.6%`
- 34-class accuracy is about `70.2%` to `75.6%`
- LSTM is the best 34-class model in this summary
- DualPath is competitive but not dominant in the saved summary

### Main attack summary

From `results/attacks/attack_summary.csv`:

- Binary PGD `eps=0.30` has `ASR_raw` about `37.8%`
- 8-class PGD `eps=0.30` has `ASR_raw` about `74.6%`
- 34-class PGD `eps=0.30` has `ASR_raw` about `88.3%`

This strongly supports the thesis setup: raw attack success can become very high before validity filtering.

### Processed-data validity

From `results/validation/full_dataset_validation_report.txt`:

- all processed train/val/test rows validate at `100%`
- all `49` rules pass across all `4,429,940` processed samples

This is a crucial invariant: validation is intended to fail primarily on adversarial or corrupted raw-space samples, not on the final processed dataset.

### VAE manifest status

From `vae_run_manifest.json`:

- protocol allowlist is `[0, 1, 2, 6, 17, 47]`
- training data did not include protocol `2` even though the allowlist keeps it
- VAE checkpoints are recorded for 7 classes
- some diagnostics are weak:
  - `Benign`, `BruteForce`, `DDoS`, and `DoS` currently show `0.0` unconditional and conditional validity in the manifest
  - `Mirai`, `Recon`, and `Spoofing` have nonzero but still limited validity
- protocol prediction accuracy is generally high, often above `95%`

The VAE subsystem appears active and partially successful, but not yet uniformly strong on validity metrics.

## Important Inconsistencies To Remember

There is artifact drift across the repo.

Examples:

- `data/processed/clean_data_validity_report.json` currently says `100%` validity
- `tables/T4_clean_validity.md` reports `81.3724%` overall validity with large protocol-related violations
- `guide.md` still describes older commands and older file locations such as `src/preprocessing.py`
- `CLAUDE.md` still documents the pre-reorganization source tree

Working interpretation:

- current code and the latest processed-data validation outputs indicate the cleaned/processed dataset is fully validator-compliant
- some tables/docs were generated from earlier validator logic or earlier preprocessing passes and were not regenerated afterward

## Important File-Level Notes

- `src/preprocessing/pipeline.py` executes top-level code on import, so treat it as a script
- many scripts still use hardcoded `D:/thesis_final/...` paths
- `src/attack/adversarial_attacks.py` explicitly leaves the perturbation mask unapplied
- `src/vae/train_all.py` expects package-style imports under `src/`
- `configs/cvae.yaml` is newer than the older thesis docs and targets checkpoint output in `checkpoints/cvae`
- `checkpoints/cvae/` contains multiple timestamped runs with per-run `config.yaml` and `metadata.json`

## Practical Command Map

Prefer these current script paths:

- EDA tables: `src/evaluation/eda_tables.py`
- EDA figures 1: `src/evaluation/eda_figures_part1.py`
- EDA figures 2: `src/evaluation/eda_figures_part2.py`
- preprocessing: `src/preprocessing/pipeline.py`
- baseline classifiers: `src/classifiers/baseline_experiments.py`
- main attacks: `src/attack/run_attacks.py`
- 8-class multi-model attacks: `src/attack/run_attacks_8class_models.py`
- full validation: `src/evaluation/validate_full_dataset.py`
- validity analysis: `src/evaluation/run_validity_analysis.py`
- VAE orchestrator: `src/vae/train_all.py`

Git Bash examples:

```bash
cd /d/thesis_final
/c/Users/T2530985/.conda/envs/thesis/python.exe src/preprocessing/pipeline.py
/c/Users/T2530985/.conda/envs/thesis/python.exe src/classifiers/baseline_experiments.py
/c/Users/T2530985/.conda/envs/thesis/python.exe src/attack/run_attacks.py --device cuda --sample-size 50000
/c/Users/T2530985/.conda/envs/thesis/python.exe src/evaluation/run_validity_analysis.py
/c/Users/T2530985/.conda/envs/thesis/python.exe src/vae/train_all.py --device cuda
```

## Recommended Mental Model For Future Work

- `feature_groups.py` defines the world
- `validator.py` defines domain validity
- preprocessing creates a fully valid, scaled modeling dataset
- baseline classifiers operate entirely in scaled feature space
- attacks are intentionally unconstrained in that scaled space
- evaluation only becomes thesis-relevant after inverse-transforming attacks back to raw space
- the VAE branch is a separate generative track intended to move attacks back toward the data manifold

## Current Workspace State

Git status at inspection time showed:

- active uncommitted reorganization-related changes
- moved source files already staged or tracked in new package paths
- additional untracked directories such as `configs/`, `.vscode/`, `src/vae/`, and artifact folders

That means future edits should assume the tree may be intentionally mid-refactor and should avoid relying on older flat `src/*.py` paths unless compatibility shims are added.
