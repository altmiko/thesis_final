# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Project Overview

Bachelor's thesis codebase studying **VAE-based on-manifold adversarial attacks against network intrusion detection systems (NIDS)**, evaluated on the **CICIoT2023 dataset** (34 attack classes, 8 categories, ~4.4M rows after stratified sampling from a 9.4GB source CSV). The core thesis finding is the gap between raw attack success rate (ASR_raw) and domain-validity-constrained ASR (ASR_valid).

## Environment

- **Python**: 3.10 via conda env `thesis` (`C:\Users\T2530985\.conda\envs\thesis\python.exe`)
- **Dependencies** (no requirements.txt): `numpy pandas scikit-learn matplotlib seaborn torch torchattacks umap-learn scipy joblib pyarrow`
- **GPU**: Scripts accept `--device cuda` where applicable
- **Global seed**: 42

## Commands

All scripts run from repo root (`D:/thesis_final`):

```bash
# EDA (must run in order; part2 produces netdiffuser_categorization.json needed by preprocessing)
python src/eda_tables.py
python src/eda_figures_part1.py
python src/eda_figures_part2.py

# Preprocessing (requires netdiffuser_categorization.json from eda_figures_part2)
python src/preprocessing.py

# Training (15 models: 5 architectures × 3 tasks)
python src/baseline_experiments.py

# Attacks
python src/run_attacks.py --device cuda --sample-size 50000
python src/run_attacks_8class_models.py --device cuda --num-restarts 10

# Post-attack analysis
python src/validate_full_dataset.py
python src/run_validity_analysis.py
python src/compact_exhibits.py --device cuda
python src/sample_exhibit.py
python src/analyze_attack_restarts.py
python src/plot_attack_restarts.py
python src/delta_report.py --n 500 --output results/delta_report.html --device cuda
python src/export_slide_exhibits.py
```

## Architecture

### Pipeline Flow

```
Raw CSV → eda_tables/figures → preprocessing.py → baseline_experiments.py → run_attacks.py → validity analysis
```

### Source Files

**Foundation layer** (imported, not run directly):
- `feature_groups.py` — Central schema: 39 feature names, CATEGORY_MAP (34→8), mutable/immutable/binary/integer feature sets, PERTURBATION_MASK (1.0/0.3/0.0), near-zero-IQR governance. Imported by nearly everything.
- `validator.py` — Domain validity checker (rules G1–G8: non-negativity, protocol validity, binary constraints, Min≤AVG≤Max, Var=Std², TTL range, packet counts). Operates in **raw space only** — always inverse-transform before validating.
- `adversarial_attacks.py` — Tabular-adapted FGSM/PGD/CW (no [0,1] clamping), `run_attack()`, `run_attack_with_restarts()`, `compute_attack_metrics()`.
- `netdiffuser_categorization.py` — Spearman correlation → hierarchical clustering → CH-score feature partitioning into discrete/relative groups.
- `validity_analysis.py` — Inverse-transform + validate adversarial examples, compute ASR_valid, generate impossible-traffic exhibits.

**Models** (`models.py`): SimpleMLP, CNNOnly, LSTMOnly, SerialCNNLSTM, AttentionSerialCNNLSTM, **DualPathIDS** (proposed architecture: parallel CNN+LSTM → FusionAttention → classifier). Factory: `get_model(model_type, num_features, num_classes)`.

### Key Data Paths

- `data/ciciot2023/ciciot2023_base.csv` — Raw source (~10GB, never load fully into memory)
- `data/processed/` — ML-ready arrays (`X/y_{train,val,test}.npy`), `scaler.pkl`, encoders, `perturbation_mask.npy`
- `data/processed/raw_loaded.parquet` — 4.4M-row stratified sample, EDA entrypoint
- `models/` — 15 `.pt` checkpoints: `{mlp,cnn,lstm,serial,dualpath}_{binary,8class,34class}.pt`
- `results/attacks/` — Attack NPZ artifacts + CSV summaries

## Development Guidelines

- **Large data**: The source CSV is ~10GB. Use chunking or `head -n 10` to inspect. EDA scripts read from `raw_loaded.parquet` (4.4M rows), not the raw CSV.
- **Label casing**: Dataset uses ALL UPPERCASE labels (e.g., `DDOS-ICMP_FLOOD`). `CATEGORY_MAP` keys are uppercase.
- **Schema**: "Modified Schema A" — 39 features. Includes `Time_To_Live`, `IGMP`; excludes Magnitude, Radius, flow_duration, Duration, Srate, Drate, urg_count.
- **Perturbation mask is intentionally unconstrained in attacks**: `run_attack()` has a `perturbation_mask` parameter but it's marked TODO. The unconstrained attack design is deliberate — the ASR_raw vs ASR_valid gap is the thesis contribution.
- **No SMOTE/oversampling**: By design, to avoid contaminating the VAE manifold.
- **Hardcoded paths**: EDA and preprocessing scripts use hardcoded `D:/thesis_final/`. Attack/model scripts use `Path(__file__).resolve().parents[1]`.
- **Training config**: Adam(lr=1e-3), ReduceLROnPlateau(patience=3), early stopping(patience=5), 5 epochs max, batch=2048, plain CrossEntropyLoss (no class weights — stratified sampling handles imbalance).

## Reference Documents

- `guide.md` — Master workspace map
- `work.md` — Phase tracker
- `validator_explained.md` — Detailed validator rule documentation
- `eda_preprocessing_prompt.md` — Original spec for EDA+preprocessing pipeline
