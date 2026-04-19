import pandas as pd
import numpy as np
import json
import sys
import os
import hashlib
import pickle
import subprocess
import logging
from datetime import datetime

np.random.seed(42)
import random
random.seed(42)

sys.path.insert(0, 'D:/thesis_final/src')
from feature_groups import (FEATURE_NAMES, BINARY_FEATURES, INTEGER_FEATURES,
                            MUTABLE_FEATURES, CATEGORY_MAP,
                            NEAR_ZERO_IQR_THRESHOLD,
                            RARE_SIGNAL_NONZERO_THRESHOLD,
                            NEAR_ZERO_FREEZE_POLICY,
                            MANUAL_CONCENTRATED_DECISIONS)
from validator import validate_batch

PROC_DIR = 'D:/thesis_final/data/processed'
os.makedirs(PROC_DIR, exist_ok=True)
os.makedirs('D:/thesis_final/logs', exist_ok=True)

FREEZE_POLICY = NEAR_ZERO_FREEZE_POLICY

logging.basicConfig(
    filename='D:/thesis_final/logs/preprocessing.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='w',
)


def log(msg: str) -> None:
    print(msg)
    logging.info(msg)

df = pd.read_parquet(f'{PROC_DIR}/raw_loaded.parquet')
log(f"Loaded: {df.shape}")

# ── 4.1 Clean ────────────────────────────────────────────────────────
log("\n=== Phase 4.1: Cleaning ===")

# Already did inf/nan removal during loading, but ensure
before = len(df)
df = df.replace([np.inf, -np.inf], np.nan).dropna()
log(f"Inf/NaN: {before} -> {len(df)} (dropped {before - len(df)})")

# Drop timestamp column if present
for col in ['ts', 'Timestamp']:
    if col in df.columns:
        df = df.drop(columns=[col])
        log(f"Dropped column: {col}")

# Clip to [0, 99.99th percentile] for non-negative features
# Covariance can be negative and should not be lower-clipped at zero.
SIGNED_FEATURES = {'Covariance'}
clip_log = {}
for feat in FEATURE_NAMES:
    if feat in df.columns:
        upper = df[feat].quantile(0.9999)
        original_max = float(df[feat].max())
        clipped = (df[feat] > upper).sum()
        if clipped > 0:
            if feat in SIGNED_FEATURES:
                df[feat] = df[feat].clip(upper=upper)
            else:
                df[feat] = df[feat].clip(lower=0, upper=upper)
            clip_log[feat] = {
                'upper_clip': float(upper),
                'original_max': original_max,
                'n_clipped': int(clipped),
            }

log(f"Clipped features: {len(clip_log)}")
for feat, info in sorted(clip_log.items(), key=lambda x: -x[1]['n_clipped'])[:10]:
    log(f"  {feat}: {info['n_clipped']} rows clipped at {info['upper_clip']:.2f}")

# Round integer features
for feat in INTEGER_FEATURES:
    if feat in df.columns:
        df[feat] = df[feat].round().clip(lower=0).astype(np.float32)

# Round binary features to {0, 1}
for feat in BINARY_FEATURES:
    if feat in df.columns:
        df[feat] = df[feat].round().clip(0, 1).astype(np.float32)

after_clean = len(df)
log(f"After cleaning: {after_clean:,} rows")

# Validate cleaned (post-processing) data before any split.
log("\n=== Phase 4.1b: Post-clean validation ===")
processed_validation = validate_batch(df[FEATURE_NAMES].values, FEATURE_NAMES)
processed_rates = processed_validation.per_rule_violation_rate()
processed_report = {
    'validity_rate': processed_validation.validity_rate,
    'n_samples': processed_validation.n_samples,
    'n_valid': int(processed_validation.overall_valid.sum()),
    'per_rule_violation_rates': {k: float(v) for k, v in processed_rates.items()},
}
with open(f'{PROC_DIR}/processed_data_validity_report.json', 'w') as f:
    json.dump(processed_report, f, indent=2)
log(f"Processed data validity: {processed_validation.validity_rate:.4%}")

# ── 4.2 Label encoding ──────────────────────────────────────────────
log("\n=== Phase 4.2: Label encoding ===")
from sklearn.preprocessing import LabelEncoder

le_34 = LabelEncoder()
y_34 = le_34.fit_transform(df['Label'].values)
log(f"34-class label encoder: {len(le_34.classes_)} classes")

# Category-level encoder
category_series = df['Label'].map(CATEGORY_MAP)
missing_category_count = int(category_series.isna().sum())
log(f"Category map missing rows: {missing_category_count}")
assert category_series.notna().all(), \
    f"CATEGORY_MAP missed labels: {df.loc[category_series.isna(), 'Label'].unique()}"

le_cat = LabelEncoder()
y_cat = le_cat.fit_transform(category_series.values)
log(f"8-class category encoder: {len(le_cat.classes_)} categories")

# Binary encoder
benign_labels = [k for k, v in CATEGORY_MAP.items() if v == 'Benign']
assert len(benign_labels) > 0, "CATEGORY_MAP has no Benign label mapping"
y_bin = (~df['Label'].isin(benign_labels)).astype(np.int32).values
log(f"Binary: {(y_bin==0).sum():,} benign, {(y_bin==1).sum():,} attack")

# Save encoders
with open(f'{PROC_DIR}/label_encoder.pkl', 'wb') as f:
    pickle.dump(le_34, f)
with open(f'{PROC_DIR}/category_encoder.pkl', 'wb') as f:
    pickle.dump(le_cat, f)
with open(f'{PROC_DIR}/class_names.json', 'w') as f:
    json.dump(list(le_34.classes_), f, indent=2)
with open(f'{PROC_DIR}/category_names.json', 'w') as f:
    json.dump(list(le_cat.classes_), f, indent=2)

# ── 4.3 Stratified split ────────────────────────────────────────────
log("\n=== Phase 4.3: Stratified split (70/10/20) ===")
from sklearn.model_selection import train_test_split

X = df[FEATURE_NAMES].values.astype(np.float32)

# Split by indices so all target variants (34/8/2 class) stay aligned.
indices = np.arange(len(df))
idx_trainval, idx_test = train_test_split(
    indices, test_size=0.20, stratify=y_34, random_state=42)
idx_train, idx_val = train_test_split(
    idx_trainval, test_size=0.125, stratify=y_34[idx_trainval], random_state=42)

X_train = X[idx_train]
X_val = X[idx_val]
X_test = X[idx_test]

y_train = y_34[idx_train].astype(np.int32)
y_val = y_34[idx_val].astype(np.int32)
y_test = y_34[idx_test].astype(np.int32)

y_train_cat = y_cat[idx_train].astype(np.int32)
y_val_cat = y_cat[idx_val].astype(np.int32)
y_test_cat = y_cat[idx_test].astype(np.int32)

y_train_bin = y_bin[idx_train].astype(np.int32)
y_val_bin = y_bin[idx_val].astype(np.int32)
y_test_bin = y_bin[idx_test].astype(np.int32)

log(f"Train: {len(X_train):,}  Val: {len(X_val):,}  Test: {len(X_test):,}")
log(f"Ratios: {len(X_train)/len(X)*100:.1f}% / {len(X_val)/len(X)*100:.1f}% / {len(X_test)/len(X)*100:.1f}%")

# Verify every class has >= 1 sample in each split
for split_name, split_y in [('train', y_train), ('val', y_val), ('test', y_test)]:
    unique_classes = np.unique(split_y)
    log(f"  {split_name}: {len(unique_classes)} classes present")
    if len(unique_classes) < len(le_34.classes_):
        missing = set(range(len(le_34.classes_))) - set(unique_classes)
        for m in missing:
            log(f"    WARNING: class {le_34.classes_[m]} missing from {split_name}")

    counts = pd.Series(split_y).value_counts()
    rare = counts[counts < 10]
    if len(rare) > 0:
        for cls_idx, n_samples in rare.items():
            log(f"    WARNING: {le_34.classes_[cls_idx]} has only {n_samples} samples in {split_name}")

# ── 4.4 Scaling ──────────────────────────────────────────────────────
log("\n=== Phase 4.4: RobustScaler (fit on train only) ===")
from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train).astype(np.float32)
X_val_scaled = scaler.transform(X_val).astype(np.float32)
X_test_scaled = scaler.transform(X_test).astype(np.float32)

train_iqr = np.percentile(X_train, 75, axis=0) - np.percentile(X_train, 25, axis=0)
df_train = df.iloc[idx_train]
near_zero_iqr = [
    (feat, float(iqr), float(scale))
    for feat, iqr, scale in zip(FEATURE_NAMES, train_iqr, scaler.scale_)
    if iqr < NEAR_ZERO_IQR_THRESHOLD
]

near_zero_features = []
for feat, iqr, scale in near_zero_iqr:
    col = df_train[feat]
    n_unique_train = int(col.nunique(dropna=False))
    max_value_train = float(col.max())
    non_zero_fraction = float((col != 0).mean())

    if n_unique_train == 1:
        kind = 'constant'
    elif non_zero_fraction < RARE_SIGNAL_NONZERO_THRESHOLD:
        kind = 'rare_signal'
    else:
        kind = 'concentrated'

    near_zero_features.append({
        'feature': feat,
        'train_iqr': iqr,
        'scaler_scale': scale,
        'n_unique_train': n_unique_train,
        'max_value_train': max_value_train,
        'non_zero_fraction': non_zero_fraction,
        'kind': kind,
        'policy_action': FREEZE_POLICY[kind],
    })

kind_counts = {
    'constant': sum(1 for x in near_zero_features if x['kind'] == 'constant'),
    'rare_signal': sum(1 for x in near_zero_features if x['kind'] == 'rare_signal'),
    'concentrated': sum(1 for x in near_zero_features if x['kind'] == 'concentrated'),
}

near_zero_iqr_report = {
    'threshold': NEAR_ZERO_IQR_THRESHOLD,
    'rare_signal_non_zero_threshold': RARE_SIGNAL_NONZERO_THRESHOLD,
    'freeze_policy': FREEZE_POLICY,
    'n_features': len(near_zero_features),
    'counts_by_kind': kind_counts,
    'features': near_zero_features,
}
with open(f'{PROC_DIR}/near_zero_iqr_features.json', 'w') as f:
    json.dump(near_zero_iqr_report, f, indent=2)
log(f"Near-zero IQR report saved: {PROC_DIR}/near_zero_iqr_features.json")

log(f"Scaler center (median): min={scaler.center_.min():.4f}, max={scaler.center_.max():.4f}, mean={scaler.center_.mean():.4f}")
log(f"Scaler scale (IQR):     min={scaler.scale_.min():.6f}, max={scaler.scale_.max():.4f}, mean={scaler.scale_.mean():.4f}")
if near_zero_iqr:
    log(f"WARNING: {len(near_zero_iqr)} features have train IQR ~= 0 (near-constant in train):")
    for item in near_zero_features:
        log(
            f"  {item['feature']}: train_iqr={item['train_iqr']:.2e}, "
            f"scale={item['scaler_scale']:.2e}, non_zero_frac={item['non_zero_fraction']:.2e}, "
            f"kind={item['kind']}, action={item['policy_action']}"
        )
    log("  Review near_zero_iqr_features.json before final attack-budget decisions.")

with open(f'{PROC_DIR}/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# ── 4.5 Save arrays ─────────────────────────────────────────────────
log("\n=== Phase 4.5: Saving arrays ===")
arrays = {
    'X_train': X_train_scaled, 'X_val': X_val_scaled, 'X_test': X_test_scaled,
    'y_train': y_train, 'y_val': y_val, 'y_test': y_test,
    'y_train_cat': y_train_cat, 'y_val_cat': y_val_cat, 'y_test_cat': y_test_cat,
    'y_train_bin': y_train_bin, 'y_val_bin': y_val_bin, 'y_test_bin': y_test_bin,
}
for name, arr in arrays.items():
    np.save(f'{PROC_DIR}/{name}.npy', arr)
    log(f"  {name}.npy: shape={arr.shape}, dtype={arr.dtype}")

# ── 4.6 Class weights ───────────────────────────────────────────────
log("\n=== Phase 4.6: Class weights ===")
from sklearn.utils.class_weight import compute_class_weight

classes_in_train_34 = np.unique(y_train)
classes_in_train_cat = np.unique(y_train_cat)
classes_in_train_bin = np.unique(y_train_bin)

w34 = compute_class_weight('balanced', classes=classes_in_train_34, y=y_train).astype(np.float32)
w8 = compute_class_weight('balanced', classes=classes_in_train_cat, y=y_train_cat).astype(np.float32)
w2 = compute_class_weight('balanced', classes=classes_in_train_bin, y=y_train_bin).astype(np.float32)

if len(classes_in_train_34) < len(le_34.classes_):
    missing_34 = [le_34.classes_[i] for i in range(len(le_34.classes_)) if i not in set(classes_in_train_34.tolist())]
    log(f"WARNING: Missing 34-class labels in train split for class-weight computation: {missing_34}")
if len(classes_in_train_cat) < len(le_cat.classes_):
    missing_8 = [le_cat.classes_[i] for i in range(len(le_cat.classes_)) if i not in set(classes_in_train_cat.tolist())]
    log(f"WARNING: Missing 8-class labels in train split for class-weight computation: {missing_8}")
assert np.array_equal(classes_in_train_bin, np.array([0, 1])), \
    "Binary labels not in expected {0=benign, 1=attack} order"

np.save(f'{PROC_DIR}/class_weights_34.npy', w34)
np.save(f'{PROC_DIR}/class_weights_8.npy', w8)
np.save(f'{PROC_DIR}/class_weights_2.npy', w2)

# Named weights JSON
binary_weight_by_class = {int(c): float(w2[i]) for i, c in enumerate(classes_in_train_bin)}
weights_named = {
    '34class': {
        le_34.classes_[int(c)]: float(w34[i])
        for i, c in enumerate(classes_in_train_34)
    },
    '8class': {
        le_cat.classes_[int(c)]: float(w8[i])
        for i, c in enumerate(classes_in_train_cat)
    },
    'binary': {
        'benign': binary_weight_by_class[0],
        'attack': binary_weight_by_class[1],
    },
}
with open(f'{PROC_DIR}/class_weights_named.json', 'w') as f:
    json.dump(weights_named, f, indent=2)

# Log top/bottom weights
w34_sorted = sorted(zip(le_34.classes_, w34), key=lambda x: -x[1])
log("Top 5 highest weights (34-class):")
for name, w in w34_sorted[:5]:
    log(f"  {name}: {w:.4f}")
log("Top 5 lowest weights (34-class):")
for name, w in w34_sorted[-5:]:
    log(f"  {name}: {w:.4f}")
log(f"Weight ranges: 34-class=[{w34.min():.3f}, {w34.max():.3f}], "
    f"8-class=[{w8.min():.3f}, {w8.max():.3f}], "
    f"binary=[{w2.min():.3f}, {w2.max():.3f}]")

# ── 4.7 Perturbation mask ───────────────────────────────────────────
log("\n=== Phase 4.7: Perturbation mask ===")
try:
    with open(f'{PROC_DIR}/netdiffuser_categorization.json') as f:
        nd_result = json.load(f)
    discrete_set = set(nd_result['discrete'])
    relative_set = set(nd_result['relative'])
except FileNotFoundError:
    raise FileNotFoundError(
        "netdiffuser_categorization.json not found — re-run categorization "
        "before building perturbation mask. Refusing to proceed."
    )

with open(f'{PROC_DIR}/near_zero_iqr_features.json') as f:
    near_zero_loaded = json.load(f)

near_zero_by_feature = {x['feature']: x for x in near_zero_loaded.get('features', [])}
auto_freeze_features = {
    x['feature']
    for x in near_zero_loaded.get('features', [])
    if x.get('policy_action') == 'auto_freeze'
}

manual_review_mutable = [
    x['feature']
    for x in near_zero_loaded.get('features', [])
    if x.get('policy_action') == 'manual' and x['feature'] in set(MUTABLE_FEATURES)
]

unresolved_manual_review = [
    feat for feat in manual_review_mutable
    if feat not in MANUAL_CONCENTRATED_DECISIONS
]

if unresolved_manual_review:
    raise RuntimeError(
        "Near-zero-IQR features require manual review before preprocessing can continue: "
        f"{unresolved_manual_review}. Add explicit decisions in MANUAL_CONCENTRATED_DECISIONS."
    )

for feat in manual_review_mutable:
    decision = MANUAL_CONCENTRATED_DECISIONS[feat]
    if decision == 'force_freeze':
        auto_freeze_features.add(feat)
    elif decision == 'allow_mutable':
        pass
    else:
        raise ValueError(
            f"Invalid decision '{decision}' for feature '{feat}'. "
            "Use 'allow_mutable' or 'force_freeze'."
        )

mutable_set = set(MUTABLE_FEATURES)
cap_partial_features = {
    feat
    for feat, item in near_zero_by_feature.items()
    if feat in mutable_set
    and item.get('kind') in ('rare_signal', 'concentrated')
    and feat not in auto_freeze_features
}

mask = np.zeros(len(FEATURE_NAMES), dtype=np.float32)
n_full, n_partial, n_frozen = 0, 0, 0
n_auto_frozen = 0
n_capped_partial = 0
for i, feat in enumerate(FEATURE_NAMES):
    if feat in auto_freeze_features:
        mask[i] = 0.0
        n_frozen += 1
        n_auto_frozen += 1
    elif feat in cap_partial_features:
        mask[i] = 0.3
        n_partial += 1
        n_capped_partial += 1
    elif feat in mutable_set and feat in discrete_set:
        mask[i] = 1.0
        n_full += 1
    elif feat in mutable_set and feat in relative_set:
        mask[i] = 0.3
        n_partial += 1
    else:
        n_frozen += 1

for feat, item in near_zero_by_feature.items():
    if item.get('kind') in ('rare_signal', 'concentrated') and feat in mutable_set:
        log(
            f"  WARNING: '{feat}' has near-zero IQR ({item['scaler_scale']:.2e}) "
            f"kind={item['kind']} max={item['max_value_train']:.4f}. "
            "Keeping mutable; scaled-epsilon interpretation may be unreliable."
        )

np.save(f'{PROC_DIR}/perturbation_mask.npy', mask)
log(
    f"  Full (1.0): {n_full}, Partial (0.3): {n_partial}, "
    f"Frozen (0.0): {n_frozen} [auto-frozen: {n_auto_frozen}, capped-to-partial: {n_capped_partial}]"
)

# ── 4.8 Content hashes ──────────────────────────────────────────────
log("\n=== Phase 4.8: Content hashes ===")
content_hashes = {}
for name in ['X_train', 'X_val', 'X_test']:
    fpath = f'{PROC_DIR}/{name}.npy'
    with open(fpath, 'rb') as f:
        h = hashlib.sha256(f.read()).hexdigest()
    content_hashes[name] = h

content_hashes['scaler'] = hashlib.sha256(
    scaler.center_.tobytes() + scaler.scale_.tobytes()).hexdigest()
content_hashes['label_encoder'] = hashlib.sha256(
    le_34.classes_.astype(str).tobytes()).hexdigest()

with open(f'{PROC_DIR}/content_hashes.json', 'w') as f:
    json.dump(content_hashes, f, indent=2)
log("  Content hashes saved.")

# ── 4.9 Run manifest ────────────────────────────────────────────────
log("\n=== Phase 4.9: Run manifest ===")
try:
    git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                        stderr=subprocess.DEVNULL).decode().strip()
except Exception:
    git_hash = "not-a-git-repo"

import sklearn, matplotlib, scipy
try:
    import umap
    umap_version = umap.__version__
except Exception:
    umap_version = "unknown"

manifest = {
    'timestamp': datetime.now().isoformat(),
    'git_commit': git_hash,
    'python_version': sys.version,
    'package_versions': {
        'numpy': np.__version__,
        'pandas': pd.__version__,
        'sklearn': sklearn.__version__,
        'matplotlib': matplotlib.__version__,
        'scipy': scipy.__version__,
        'umap-learn': umap_version,
    },
    'seeds': {'numpy': 42, 'random': 42, 'sklearn_random_state': 42},
    'schema': 'Modified Schema A (39 features)',
    'mode': 'SAMPLE',
    'sample_cap': {'majority': 200000, 'medium': 200000, 'minority': None, 'rare': None},
    'rows': {
        'loaded_parquet': int(before),
        'after_clean': int(after_clean),
        'train': len(X_train),
        'val': len(X_val),
        'test': len(X_test),
    },
    'n_features': len(FEATURE_NAMES),
    'n_classes': {'34class': len(le_34.classes_), '8class': len(le_cat.classes_), 'binary': 2},
    'content_hashes': content_hashes,
    'clip_log': clip_log,
    'near_zero_iqr': {
        'threshold': NEAR_ZERO_IQR_THRESHOLD,
        'rare_signal_non_zero_threshold': RARE_SIGNAL_NONZERO_THRESHOLD,
        'freeze_policy': FREEZE_POLICY,
        'counts_by_kind': kind_counts,
        'auto_frozen_features': sorted(list(auto_freeze_features)),
        'capped_partial_features': sorted(list(cap_partial_features)),
        'manual_review_features': sorted(manual_review_mutable),
        'manual_decisions': {
            feat: MANUAL_CONCENTRATED_DECISIONS[feat]
            for feat in sorted(manual_review_mutable)
        },
        'decision_timestamp': datetime.now().isoformat(),
    },
}

os.makedirs('D:/thesis_final/config', exist_ok=True)
with open('D:/thesis_final/config/run_manifest.json', 'w') as f:
    json.dump(manifest, f, indent=2)
log("  Run manifest saved.")

# ── 4.10 Final sanity report ────────────────────────────────────────
# Load processed validity report (post-clean, pre-split)
try:
    with open(f'{PROC_DIR}/processed_data_validity_report.json') as f:
        processed_validity = json.load(f)
    processed_validity_rate = processed_validity['validity_rate']
    processed_worst_rule = max(processed_validity['per_rule_violation_rates'].items(), key=lambda x: x[1])
except Exception:
    processed_validity_rate = -1
    processed_worst_rule = ('unknown', -1)

try:
    with open(f'{PROC_DIR}/clean_data_validity_report.json') as f:
        preclean_validity = json.load(f)
    preclean_validity_rate = preclean_validity['validity_rate']
except Exception:
    preclean_validity_rate = -1

# IQR analysis
iqr_sorted = sorted(zip(FEATURE_NAMES, scaler.scale_), key=lambda x: -x[1])

log("\n" + "=" * 68)
log("     CICIoT2023 PREPROCESSING COMPLETE")
log("=" * 68)
log(f"Schema:           Modified Schema A (39 features)")
log(f"Mode:             SAMPLE (three-tier caps)")
log(f"Rows loaded:      {before:,}")
log(f"After cleaning:   {after_clean:,}")
log(f"Train/Val/Test:   {len(X_train):,} / {len(X_val):,} / {len(X_test):,}  (70/10/20 stratified)")
log(f"Features:         {len(FEATURE_NAMES)}")
log(f"Classes:          34 (full) | 8 (category) | 2 (binary)")
log("")
log(f"Scaler IQR:       min={scaler.scale_.min():.6f}, max={scaler.scale_.max():.4f}, mean={scaler.scale_.mean():.4f}")
log(f"Top-5 largest IQRs:")
for feat, iqr in iqr_sorted[:5]:
    log(f"  {feat}={iqr:.4f}")
log(f"Top-5 smallest IQRs (risk zone for PGD):")
for feat, iqr in iqr_sorted[-5:]:
    log(f"  {feat}={iqr:.6f}")
log("")
if preclean_validity_rate >= 0:
    log(f"Pre-clean validity (EDA report): {preclean_validity_rate:.4%} overall")
log(f"Processed data validity: {processed_validity_rate:.4%} overall")
log(f"  worst rule: {processed_worst_rule[0]} at {processed_worst_rule[1]:.4%}")
log("")
log(f"Class weights (34-class):")
log(f"  highest 3: {', '.join(f'{n}={w:.4f}' for n, w in w34_sorted[:3])}")
log(f"  lowest  3: {', '.join(f'{n}={w:.4f}' for n, w in w34_sorted[-3:])}")
log(f"  weight range: [{w34.min():.4f}, {w34.max():.4f}]  (ratio: {w34.max()/w34.min():.1f}x)")
log("")
log(f"Perturbation mask: {n_full} features at 1.0, {n_partial} at 0.3, {n_frozen} frozen")
log("")
log("Augmentation applied: NONE (SMOTE/ADASYN skipped by design)")
log("Imbalance handling: class-weighted loss vectors saved for downstream use.")
log("")
log("All artifacts saved. Content hashes committed to run_manifest.json.")
log("=" * 68)
