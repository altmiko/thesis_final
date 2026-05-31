# Design Spec: XGBoost & Random Forest Tree Baselines

**Date:** 2026-06-01
**Status:** Approved scope (training + metrics reporting). Transfer-attack robustness deferred.
**Author:** Implementation session

## 1. Goal

Add two non-neural classifiers — **Random Forest** (scikit-learn) and **XGBoost** —
as additional baselines alongside the existing five neural architectures
(`mlp`, `cnn`, `lstm`, `serial`, `dualpath`). They are trained and evaluated on
the same three CICIoT2023 framings (binary / 8class / 34class) and folded into the
same comparison tables and ROC-AUC plots, so the thesis can report tree-model
baselines apples-to-apples with the neural nets.

### In scope
- Train RF + XGBoost for all three tasks with light validation-set tuning.
- Save model checkpoints + per-model classification reports in the existing report schema.
- Extend the evaluation script so tree models appear in the unified baseline tables and ROC plots.

### Out of scope (deferred to a later pass)
- Transfer-attack / adversarial robustness evaluation of the tree models.
- Wiring tree models into `run_attacks.py` (gradient attacks require differentiability).
- Any change to neural-net training (`baseline_experiments.py`) or to `models.py`.

## 2. Context (current pipeline)

- **Processed arrays** (`data/processed/`): `X_{train,val,test}.npy` are
  `float32`, scaled, 39 features. Sizes: train **3,100,958**, val **442,994**,
  test **885,988** rows. Per-task labels: `y_{split}_bin.npy` (2 classes),
  `y_{split}_cat.npy` (8 classes), `y_{split}.npy` (34 classes); all integer,
  contiguous `0..K-1`. Class-name artifacts: `category_names.json` (8),
  `class_names.json` (34).
- **NN training** (`src/classifiers/baseline_experiments.py`): trains 5 archs × 3
  tasks. Per model it writes `models/{model}_{task}.pt`,
  `results/{model}_{task}_classification_report.{json,txt}`, and a combined
  `results/all_models_all_tasks_summary.json`. **No class weights** (stratified
  sampling rationale, captured in `CLASS_WEIGHT_DECISION`). Seed 42.
- **NN evaluation** (`src/classifiers/review_baselines.py`): loads `.pt`
  checkpoints via `get_model`, computes accuracy / precision / recall / F1 +
  ROC-AUC, and writes `results/baseline_review/baseline_metrics_summary.{csv,md}`,
  `baseline_per_class_metrics.csv`, `json/{task}_{model}_metrics.json`, and
  `roc_curves/{task}_{model}_roc_auc.png`. `MODEL_TYPES` is the 5-NN tuple.
- **Libraries**: scikit-learn 1.7.2 and joblib 1.5.3 are installed.
  **`xgboost` is NOT installed** — must be added to the `thesis` conda env.

### NN classification-report JSON schema (to be matched by trees)
```
model, task, num_classes, loss_weighting, loss_weighting_decision,
provided_class_weights_for_reference, test_loss, accuracy, macro_f1,
weighted_f1, per_class_f1{ "0":.., ... }, classification_report{ sklearn dict }
```

## 3. Dependency change

Install XGBoost into the `thesis` conda env:

```
C:/Users/T2530985/.conda/envs/thesis/python.exe -m pip install xgboost
```

Target a 2.x release (uses the `device="cuda"` + `tree_method="hist"` API and
sklearn-wrapper `early_stopping_rounds`). The implementation must degrade
gracefully if the GPU is unavailable (fall back to CPU `hist`).

## 4. Component 1 — `src/classifiers/tree_baselines.py` (new)

Standalone training script (run as `python -m src.classifiers.tree_baselines`),
following the import/rooting conventions already used in `review_baselines.py`
(`ROOT = Path(__file__).resolve().parents[2]`, add to `sys.path`).

### 4.1 Responsibilities
For each `task ∈ {binary, 8class, 34class}` × `model ∈ {rf, xgb}`:
1. Load `X_train/val/test` (mmap) and the task's label arrays.
2. **Light validation-set tuning**: fit each grid config on `X_train`/`y_train`,
   score on the validation split by **macro-F1**, keep the best config's fitted model.
   The selected model is **not** refit on train+val (mirrors NN model-selection,
   which uses val for selection only and never trains on it — keeps the test split
   equally clean across families).
3. Evaluate the selected model on the test split.
4. Persist artifacts (Section 4.4).

### 4.2 Model configuration (no class weights, seed 42)
**Random Forest** (`sklearn.ensemble.RandomForestClassifier`):
- Fixed: `random_state=42`, `n_jobs=-1`, `class_weight=None`,
  `max_features="sqrt"`, `max_samples=0.5` (bootstrap fraction to bound memory on
  3.1M rows; configurable via `--rf-max-samples`).
- Grid: `n_estimators ∈ {200, 400}`, `max_depth ∈ {20, 40}` → 4 combos.

**XGBoost** (`xgboost.XGBClassifier`):
- Fixed: `random_state=42`, `n_jobs=-1`, `tree_method="hist"`,
  `device` = `"cuda"` if `--device cuda` and available else `"cpu"`,
  `objective="multi:softprob"` (multiclass) / `"binary:logistic"` (binary),
  `n_estimators=600` with `early_stopping_rounds=30` using the validation split as
  `eval_set`. No `scale_pos_weight` (no class weighting).
- Grid: `max_depth ∈ {6, 10}`, `learning_rate ∈ {0.1, 0.3}` → 4 combos.
- XGBoost requires labels `0..K-1` (already satisfied).

Grids are intentionally small to bound runtime on 3.1M rows.

### 4.3 Selection metric
Validation **macro-F1** (`sklearn.metrics.f1_score(..., average="macro")`),
chosen for robustness to the heavy class imbalance in the 8class/34class framings.
Recorded per config in the run log and in the summary JSON
(`tuning.candidates` + `tuning.selected`).

### 4.4 Outputs (parity with NN baselines)
- `models/{rf,xgb}_{task}.pkl` — joblib-serialized fitted estimator.
- `results/{rf,xgb}_{task}_classification_report.{json,txt}` — **same schema** as
  the NN reports. Field notes:
  - `loss_weighting = "excluded"`, reusing the existing `CLASS_WEIGHT_DECISION`
    string for consistency.
  - `provided_class_weights_for_reference`: load the existing
    `class_weights_{2,8,34}.npy` and store as-is (kept only for reference, as in NN reports).
  - `test_loss`: multiclass log-loss computed from `predict_proba`
    (`sklearn.metrics.log_loss`) — the tree analogue of the NN cross-entropy, so
    the column is populated rather than null.
  - `accuracy`, `macro_f1`, `weighted_f1`, `per_class_f1`, `classification_report`
    computed exactly as in `baseline_experiments.py`
    (`labels=np.arange(num_classes)`, `zero_division=0`).
  - Additional tree-only keys appended (do not break the shared schema):
    `selected_hyperparameters`, `val_macro_f1`, `tuning` (candidate scores).
- `results/tree_models_all_tasks_summary.json` — mirrors
  `all_models_all_tasks_summary.json` structure: `{ models:["rf","xgb"], tasks:{ task:{ model:{accuracy,macro_f1,weighted_f1} } } }`.

### 4.5 CLI
```
--processed-dir   default data/processed
--models-dir      default models
--results-dir     default results
--models          csv subset of {rf,xgb} or "all" (default all)
--tasks           csv subset of {binary,8class,34class} or "all" (default all)
--device          {auto,cpu,cuda} (affects XGBoost only; default auto)
--rf-max-samples  float bootstrap fraction (default 0.5)
--limit-samples   optional int cap for smoke tests
--seed            default 42
```

## 5. Component 2 — `src/classifiers/review_baselines.py` (edit)

Minimal, well-contained changes that let tree models flow through the existing
metric/ROC/CSV machinery unchanged.

### 5.1 Model registry
- Extend `MODEL_TYPES` to include `rf`, `xgb` (and update the `--models`
  validation set). Add a `is_tree_model(model_type) -> bool` helper.
- `MODEL_DISPLAY_NAMES` gains `rf -> "RandomForest"`, `xgb -> "XGBoost"`.

### 5.2 Loading
- `load_model`: branch on family.
  - NN (existing): build via `get_model`, load `.pt` state dict, move to device, `eval()`.
  - Tree: `joblib.load(models_dir / f"{model_type}_{task}.pkl")`; return the estimator.
- Tree checkpoints are `.pkl` (not `.pt`); keep the missing-file error message.

### 5.3 Prediction dispatch
- `predict_probabilities`: branch on family.
  - NN (existing): batched `softmax(logits)`.
  - Tree: batched `estimator.predict_proba(X_batch)` on NumPy (no torch/device).
  - **Class-column alignment (defensive):** build the probability matrix indexed
    by `estimator.classes_`; reindex/scatter into a full `(n, num_classes)` matrix
    over `0..K-1`, filling absent classes with 0. (Stratified data makes missing
    classes unlikely, but guard anyway so ROC/argmax stay correct.)
- Everything downstream — `compute_binary_roc`, `compute_multiclass_roc`,
  `classification_report`, per-class rows, CSV/MD/JSON writers, ROC plotting — is
  **reused unchanged**. Tree rows land in `baseline_metrics_summary.{csv,md}`,
  `baseline_per_class_metrics.csv`, `json/{task}_{model}_metrics.json`, and
  `roc_curves/{task}_{model}_roc_auc.png` with the same naming.

## 6. Data flow

```
data/processed/* ──► tree_baselines.py ──► models/{rf,xgb}_{task}.pkl
                                       ├─► results/{rf,xgb}_{task}_classification_report.{json,txt}
                                       └─► results/tree_models_all_tasks_summary.json

models/*.{pt,pkl} ──► review_baselines.py (extended) ──► results/baseline_review/*
                                                          (NN + RF + XGB in one table + ROC plots)
```

## 7. Error handling / edge cases
- **xgboost import**: fail with a clear "install xgboost into the thesis env"
  message; skip `xgb` gracefully if `--models rf` only.
- **GPU unavailable**: XGBoost falls back to CPU `hist` automatically.
- **predict_proba column alignment**: reindex to `0..K-1` (Section 5.3).
- **RF memory/runtime on 3.1M rows**: bounded by `max_samples` + capped
  `max_depth`; CPU-only (RF has no GPU path). Document expected wall-clock in the
  run log.
- **Schema parity**: tree report JSON must contain all NN keys so any downstream
  reader (and `review_baselines`) treats them uniformly; tree-only keys are additive.
- **Reproducibility**: seed 42 set for NumPy and passed as `random_state`.

## 8. Testing / verification
- **Smoke run**: `--limit-samples 50000 --tasks binary --models rf,xgb` completes
  end-to-end (train → save → reports).
- **Sanity assertions**: binary clean test accuracy plausibly high (e.g. > 0.90);
  `predict_proba` rows sum to ~1; report `accuracy` matches the
  `classification_report["accuracy"]` field.
- **Schema check**: load a tree report JSON and assert its key set ⊇ the NN report
  key set.
- **Integration check**: run `review_baselines.py --models rf,xgb --tasks binary
  --limit-samples 50000` and confirm rows appear in
  `baseline_metrics_summary.csv` and a ROC PNG is produced.

## 9. Open question for reviewer
- Selection metric is set to **validation macro-F1**. Confirm this is preferred
  over validation accuracy / weighted-F1 for the tuning step.
