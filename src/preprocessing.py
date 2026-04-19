import pandas as pd
import numpy as np
import json
import sys
import os
import hashlib
import pickle
import subprocess
from datetime import datetime

np.random.seed(42)
import random
random.seed(42)

sys.path.insert(0, 'D:/thesis_final/src')
from feature_groups import (FEATURE_NAMES, BINARY_FEATURES, INTEGER_FEATURES,
                            MUTABLE_FEATURES, CATEGORY_MAP)

PROC_DIR = 'D:/thesis_final/data/processed'
os.makedirs(PROC_DIR, exist_ok=True)

df = pd.read_parquet(f'{PROC_DIR}/raw_loaded.parquet')
print(f"Loaded: {df.shape}")

# ── 4.1 Clean ────────────────────────────────────────────────────────
print("\n=== Phase 4.1: Cleaning ===")

# Already did inf/nan removal during loading, but ensure
before = len(df)
df = df.replace([np.inf, -np.inf], np.nan).dropna()
print(f"Inf/NaN: {before} -> {len(df)} (dropped {before - len(df)})")

# Drop timestamp column if present
for col in ['ts', 'Timestamp']:
    if col in df.columns:
        df = df.drop(columns=[col])
        print(f"Dropped column: {col}")

# Clip to [0, 99.99th percentile] for non-negative features
# Covariance can be negative, but we don't have it in this schema
clip_log = {}
for feat in FEATURE_NAMES:
    if feat in df.columns:
        upper = df[feat].quantile(0.9999)
        clipped = (df[feat] > upper).sum()
        if clipped > 0:
            df[feat] = df[feat].clip(lower=0, upper=upper)
            clip_log[feat] = {'upper_clip': float(upper), 'n_clipped': int(clipped)}

print(f"Clipped features: {len(clip_log)}")
for feat, info in sorted(clip_log.items(), key=lambda x: -x[1]['n_clipped'])[:10]:
    print(f"  {feat}: {info['n_clipped']} rows clipped at {info['upper_clip']:.2f}")

# Round integer features
for feat in INTEGER_FEATURES:
    if feat in df.columns:
        df[feat] = df[feat].round().clip(lower=0).astype(np.float64)

# Round binary features to {0, 1}
for feat in BINARY_FEATURES:
    if feat in df.columns:
        df[feat] = df[feat].round().clip(0, 1).astype(np.float64)

after_clean = len(df)
print(f"After cleaning: {after_clean:,} rows")

# ── 4.2 Label encoding ──────────────────────────────────────────────
print("\n=== Phase 4.2: Label encoding ===")
from sklearn.preprocessing import LabelEncoder

le_34 = LabelEncoder()
y_34 = le_34.fit_transform(df['Label'].values)
print(f"34-class label encoder: {len(le_34.classes_)} classes")

# Category-level encoder
df['category'] = df['Label'].map(CATEGORY_MAP)
le_cat = LabelEncoder()
y_cat = le_cat.fit_transform(df['category'].values)
print(f"8-class category encoder: {len(le_cat.classes_)} categories")

# Binary encoder
y_bin = (df['Label'] != 'BENIGN').astype(np.int32).values
print(f"Binary: {(y_bin==0).sum():,} benign, {(y_bin==1).sum():,} attack")

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
print("\n=== Phase 4.3: Stratified split (70/10/20) ===")
from sklearn.model_selection import train_test_split

X = df[FEATURE_NAMES].values.astype(np.float32)

X_trainval, X_test, y34_trainval, y34_test = train_test_split(
    X, y_34, test_size=0.20, stratify=y_34, random_state=42)

# Actually need proper indices
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

print(f"Train: {len(X_train):,}  Val: {len(X_val):,}  Test: {len(X_test):,}")
print(f"Ratios: {len(X_train)/len(X)*100:.1f}% / {len(X_val)/len(X)*100:.1f}% / {len(X_test)/len(X)*100:.1f}%")

# Verify every class has >= 1 sample in each split
for split_name, split_y in [('train', y_train), ('val', y_val), ('test', y_test)]:
    unique_classes = np.unique(split_y)
    print(f"  {split_name}: {len(unique_classes)} classes present")
    if len(unique_classes) < len(le_34.classes_):
        missing = set(range(len(le_34.classes_))) - set(unique_classes)
        for m in missing:
            print(f"    WARNING: class {le_34.classes_[m]} missing from {split_name}")

# ── 4.4 Scaling ──────────────────────────────────────────────────────
print("\n=== Phase 4.4: RobustScaler (fit on train only) ===")
from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train).astype(np.float32)
X_val_scaled = scaler.transform(X_val).astype(np.float32)
X_test_scaled = scaler.transform(X_test).astype(np.float32)

print(f"Scaler center (median): min={scaler.center_.min():.4f}, max={scaler.center_.max():.4f}, mean={scaler.center_.mean():.4f}")
print(f"Scaler scale (IQR):     min={scaler.scale_.min():.6f}, max={scaler.scale_.max():.4f}, mean={scaler.scale_.mean():.4f}")

with open(f'{PROC_DIR}/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# ── 4.5 Save arrays ─────────────────────────────────────────────────
print("\n=== Phase 4.5: Saving arrays ===")
arrays = {
    'X_train': X_train_scaled, 'X_val': X_val_scaled, 'X_test': X_test_scaled,
    'y_train': y_train, 'y_val': y_val, 'y_test': y_test,
    'y_train_cat': y_train_cat, 'y_val_cat': y_val_cat, 'y_test_cat': y_test_cat,
    'y_train_bin': y_train_bin, 'y_val_bin': y_val_bin, 'y_test_bin': y_test_bin,
}
for name, arr in arrays.items():
    np.save(f'{PROC_DIR}/{name}.npy', arr)
    print(f"  {name}.npy: shape={arr.shape}, dtype={arr.dtype}")

# ── 4.6 Class weights ───────────────────────────────────────────────
print("\n=== Phase 4.6: Class weights ===")
from sklearn.utils.class_weight import compute_class_weight

w34 = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train).astype(np.float32)
w8 = compute_class_weight('balanced', classes=np.unique(y_train_cat), y=y_train_cat).astype(np.float32)
w2 = compute_class_weight('balanced', classes=np.unique(y_train_bin), y=y_train_bin).astype(np.float32)

np.save(f'{PROC_DIR}/class_weights_34.npy', w34)
np.save(f'{PROC_DIR}/class_weights_8.npy', w8)
np.save(f'{PROC_DIR}/class_weights_2.npy', w2)

# Named weights JSON
weights_named = {
    '34class': {le_34.classes_[i]: float(w34[i]) for i in range(len(w34))},
    '8class': {le_cat.classes_[i]: float(w8[i]) for i in range(len(w8))},
    'binary': {'benign': float(w2[0]), 'attack': float(w2[1])},
}
with open(f'{PROC_DIR}/class_weights_named.json', 'w') as f:
    json.dump(weights_named, f, indent=2)

# Log top/bottom weights
w34_sorted = sorted(zip(le_34.classes_, w34), key=lambda x: -x[1])
print("Top 5 highest weights (34-class):")
for name, w in w34_sorted[:5]:
    print(f"  {name}: {w:.4f}")
print("Top 5 lowest weights (34-class):")
for name, w in w34_sorted[-5:]:
    print(f"  {name}: {w:.4f}")

# ── 4.7 Perturbation mask ───────────────────────────────────────────
print("\n=== Phase 4.7: Perturbation mask ===")
try:
    with open(f'{PROC_DIR}/netdiffuser_categorization.json') as f:
        nd_result = json.load(f)
    discrete_set = set(nd_result['discrete'])
    relative_set = set(nd_result['relative'])
except FileNotFoundError:
    print("  WARNING: NetDiffuser results not found, using basic mask")
    discrete_set = set(FEATURE_NAMES)
    relative_set = set()

mutable_set = set(MUTABLE_FEATURES)
mask = np.zeros(len(FEATURE_NAMES), dtype=np.float32)
n_full, n_partial, n_frozen = 0, 0, 0
for i, feat in enumerate(FEATURE_NAMES):
    if feat in mutable_set and feat in discrete_set:
        mask[i] = 1.0
        n_full += 1
    elif feat in mutable_set and feat in relative_set:
        mask[i] = 0.3
        n_partial += 1
    else:
        n_frozen += 1

np.save(f'{PROC_DIR}/perturbation_mask.npy', mask)
print(f"  Full (1.0): {n_full}, Partial (0.3): {n_partial}, Frozen (0.0): {n_frozen}")

# ── 4.8 Content hashes ──────────────────────────────────────────────
print("\n=== Phase 4.8: Content hashes ===")
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
print("  Content hashes saved.")

# ── 4.9 Run manifest ────────────────────────────────────────────────
print("\n=== Phase 4.9: Run manifest ===")
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
        'loaded': int(before),
        'after_clean': int(after_clean),
        'train': len(X_train),
        'val': len(X_val),
        'test': len(X_test),
    },
    'n_features': len(FEATURE_NAMES),
    'n_classes': {'34class': len(le_34.classes_), '8class': len(le_cat.classes_), 'binary': 2},
    'content_hashes': content_hashes,
    'clip_log': clip_log,
}

os.makedirs('D:/thesis_final/config', exist_ok=True)
with open('D:/thesis_final/config/run_manifest.json', 'w') as f:
    json.dump(manifest, f, indent=2)
print("  Run manifest saved.")

# ── 4.10 Final sanity report ────────────────────────────────────────
# Load validity report
try:
    with open(f'{PROC_DIR}/clean_data_validity_report.json') as f:
        validity = json.load(f)
    validity_rate = validity['validity_rate']
    worst_rule = max(validity['per_rule_violation_rates'].items(), key=lambda x: x[1])
except Exception:
    validity_rate = -1
    worst_rule = ('unknown', -1)

# IQR analysis
iqr_sorted = sorted(zip(FEATURE_NAMES, scaler.scale_), key=lambda x: -x[1])

print("\n" + "=" * 68)
print("     CICIoT2023 PREPROCESSING COMPLETE")
print("=" * 68)
print(f"Schema:           Modified Schema A (39 features)")
print(f"Mode:             SAMPLE (three-tier caps)")
print(f"Rows loaded:      {before:,}")
print(f"After cleaning:   {after_clean:,}")
print(f"Train/Val/Test:   {len(X_train):,} / {len(X_val):,} / {len(X_test):,}  (70/10/20 stratified)")
print(f"Features:         {len(FEATURE_NAMES)}")
print(f"Classes:          34 (full) | 8 (category) | 2 (binary)")
print()
print(f"Scaler IQR:       min={scaler.scale_.min():.6f}, max={scaler.scale_.max():.4f}, mean={scaler.scale_.mean():.4f}")
print(f"Top-5 largest IQRs:")
for feat, iqr in iqr_sorted[:5]:
    print(f"  {feat}={iqr:.4f}")
print(f"Top-5 smallest IQRs (risk zone for PGD):")
for feat, iqr in iqr_sorted[-5:]:
    print(f"  {feat}={iqr:.6f}")
print()
print(f"Clean data validity: {validity_rate:.4%} overall")
print(f"  worst rule: {worst_rule[0]} at {worst_rule[1]:.4%}")
print()
print(f"Class weights (34-class):")
print(f"  highest 3: {', '.join(f'{n}={w:.4f}' for n, w in w34_sorted[:3])}")
print(f"  lowest  3: {', '.join(f'{n}={w:.4f}' for n, w in w34_sorted[-3:])}")
print(f"  weight range: [{w34.min():.4f}, {w34.max():.4f}]  (ratio: {w34.max()/w34.min():.1f}x)")
print()
print(f"Perturbation mask: {n_full} features at 1.0, {n_partial} at 0.3, {n_frozen} frozen")
print()
print("Augmentation applied: NONE (SMOTE/ADASYN skipped by design)")
print("Imbalance handling: class-weighted loss vectors saved for downstream use.")
print()
print("All artifacts saved. Content hashes committed to run_manifest.json.")
print("=" * 68)
