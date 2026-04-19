# Claude Code Prompt — CICIoT2023 Full EDA + Preprocessing Pipeline

> **How to use this file.** Paste the entire contents below (everything inside the outer fence, or just the raw markdown text if you prefer) into Claude Code as your task. The prompt is self-contained: it does not reference any other documents. Claude Code will execute every phase autonomously, stopping only at two explicit decision points.

---

## TASK CONTEXT

I am writing a bachelor's thesis on VAE-based on-manifold adversarial attacks against ML-based Network Intrusion Detection Systems, evaluated on the **CICIoT2023** dataset (Neto et al., Sensors 2023). The dataset contains ~46.7M network-flow records across 34 classes (1 benign + 33 attack types grouped into 7 categories: DDoS, DoS, Recon, Web-Based, BruteForce, Spoofing, Mirai). Your task is to build the full EDA and preprocessing pipeline — every figure, table, and processed artifact I will need for the rest of the thesis.

You will produce:
1. **Exploratory Data Analysis**: sample row printouts, class distribution, feature schemas, summary statistics, KDE distributions, correlation heatmaps, **PCA + t-SNE + UMAP** projections, NetDiffuser-style feature categorization, and a clean-data validity audit.
2. **Preprocessing pipeline**: load → clean → stratified split → RobustScaler fit → save reproducible artifacts with content hashes.

## HARD CONSTRAINTS (violating these is a bug)

- **Do not modify files in `data/raw/`.** Treat as read-only.
- **Seed 42 everywhere**: `numpy.random.seed(42)`, `random.seed(42)`, `torch.manual_seed(42)` if torch is used, `PYTHONHASHSEED=42`, and `random_state=42` in every sklearn call.
- **Fit `RobustScaler` on the training split only.** Never fit on val/test.
- **The validator operates in ORIGINAL (unscaled) feature space only.** Never validate scaled values.
- **Figures saved as PDF** at 150 DPI under `figures/`. Tables saved as **both** CSV and Markdown under `tables/`.
- **Class labels are case-sensitive.** Some CICIoT2023 releases use `DDoS-ICMP_Flood`, others use `ddos-icmp_flood`. Normalize to the original casing from the first CSV; document any normalization.

## DECISION POINTS (stop and ask me)

You will pause and ask me only at these two points:

1. **After Phase 0** (schema detection): tell me which schema the CSVs match (A, B, or neither) and await confirmation before proceeding.
2. **After Phase 1** (class counting + imbalance audit): present the profile report + T0 imbalance audit, and ask three questions together:
   - **Mode FULL** (full dataset, 32+ GB RAM) or **Mode SAMPLE** (three-tier capping, recommended)?
   - If SAMPLE, confirm the three-tier cap (majority=200K, medium/minority=full up to 200K, rare=full no cap).
   - Confirm the default imbalance-handling policy: YES to class-weighted loss (computed in Phase 4), NO to SMOTE / ADASYN / synthetic oversampling (these would contaminate the manifold your VAE is supposed to model).

Everywhere else, proceed autonomously.

---

## PHASE 0 — SCAFFOLDING + SCHEMA DETECTION

### 0.1 Directory structure

Create if not present:

```
data/raw/                 # my CSVs — read-only
data/processed/           # your outputs
figures/                  # all PDFs
tables/                   # all CSV + MD tables
src/
  feature_groups.py
  validator.py
  eda.py
  preprocessing.py
  netdiffuser_categorization.py
config/
  run_manifest.json
logs/
  eda.log
  preprocessing.log
```

### 0.2 Auto-detect schema

Read just the header row of **one** CSV in `data/raw/`. The CICIoT2023 dataset ships with two feature-name variants in circulation; detect which one we have.

**Schema A — Paper schema (Neto et al. 2023, Table 4).** 47 columns including `ts` + `label`. Key distinguishing features:
- Has `ack_count`, `syn_count`, `fin_count`, `urg_count`, `rst_count`
- Has `Number` (packet count in flow)
- Does NOT have `Number of Bytes`, `Number of Bits`, `Auto Correlation`, `numberofpkts`

**Schema B — Community release schema.** ~47 columns including `label`. Key distinguishing features:
- Has `numberofpkts`, `Number of Bytes`, `Number of Bits`, `Auto Correlation`
- Has `urg_flag_number` (not `urg_count`)
- Does NOT have `ack_count`, `syn_count`, `fin_count`, `urg_count`, `rst_count`

**Action:** compute set-differences between the detected columns and each schema. Report:

```
Detected columns: N
Matches Schema A (paper):   missing=[...], extra=[...]
Matches Schema B (community): missing=[...], extra=[...]
Best match: A | B | UNKNOWN
```

**STOP and ask me to confirm before continuing.** If neither matches closely, print the full column list and ask which schema to use.

### 0.3 Commit chosen schema to `src/feature_groups.py`

Based on my confirmation, write ONE of these exact specifications to `src/feature_groups.py`. Include feature categorization used later by the validator and perturbation mask.

**Common across both schemas** (copy these into `feature_groups.py` exactly):

```python
# Immutable: defined by the network stack; attacker cannot change
IMMUTABLE_FEATURES = ['Protocol Type', 'TCP', 'UDP', 'ICMP']

# Quasi-immutable: application-layer protocol indicators
QUASI_IMMUTABLE_FEATURES = ['HTTP', 'HTTPS', 'DNS', 'Telnet', 'SMTP', 'SSH',
                             'IRC', 'DHCP', 'ARP', 'IPv', 'LLC']

# Mutable: attacker can influence via crafting packets
BASE_MUTABLE = ['flow_duration', 'Duration', 'Rate', 'Srate', 'Drate',
                'fin_flag_number', 'syn_flag_number', 'rst_flag_number',
                'psh_flag_number', 'ack_flag_number', 'ece_flag_number',
                'cwr_flag_number', 'Tot sum', 'Min', 'Max', 'AVG', 'Std',
                'Tot size', 'IAT', 'Covariance', 'Magnitude', 'Radius',
                'Weight', 'Header_Length', 'Variance']

BINARY_FEATURES = ['HTTP', 'HTTPS', 'DNS', 'Telnet', 'SMTP', 'SSH', 'IRC',
                   'TCP', 'UDP', 'DHCP', 'ARP', 'ICMP', 'IPv', 'LLC']
```

**If Schema A** (paper), add:

```python
FEATURE_NAMES = [
    'flow_duration', 'Header_Length', 'Protocol Type', 'Duration', 'Rate',
    'Srate', 'Drate', 'fin_flag_number', 'syn_flag_number', 'rst_flag_number',
    'psh_flag_number', 'ack_flag_number', 'ece_flag_number', 'cwr_flag_number',
    'ack_count', 'syn_count', 'fin_count', 'urg_count', 'rst_count',
    'HTTP', 'HTTPS', 'DNS', 'Telnet', 'SMTP', 'SSH', 'IRC',
    'TCP', 'UDP', 'DHCP', 'ARP', 'ICMP', 'IPv', 'LLC',
    'Tot sum', 'Min', 'Max', 'AVG', 'Std', 'Tot size', 'IAT', 'Number',
    'Magnitude', 'Radius', 'Covariance', 'Variance', 'Weight',
]
INTEGER_FEATURES = ['fin_flag_number', 'syn_flag_number', 'rst_flag_number',
                    'psh_flag_number', 'ack_flag_number', 'ece_flag_number',
                    'cwr_flag_number',
                    'ack_count', 'syn_count', 'fin_count', 'urg_count',
                    'rst_count', 'Number']
MUTABLE_FEATURES = BASE_MUTABLE + ['ack_count', 'syn_count', 'fin_count',
                                    'urg_count', 'rst_count', 'Number']
```

**If Schema B** (community), add:

```python
FEATURE_NAMES = [
    'flow_duration', 'Header_Length', 'Protocol Type', 'Duration', 'Rate',
    'Srate', 'Drate', 'fin_flag_number', 'syn_flag_number', 'rst_flag_number',
    'psh_flag_number', 'ack_flag_number', 'ece_flag_number', 'cwr_flag_number',
    'urg_flag_number',
    'HTTP', 'HTTPS', 'DNS', 'Telnet', 'SMTP', 'SSH', 'IRC',
    'TCP', 'UDP', 'DHCP', 'ARP', 'ICMP', 'IPv', 'LLC',
    'Tot sum', 'Min', 'Max', 'AVG', 'Std', 'Tot size', 'IAT', 'numberofpkts',
    'Number of Bytes', 'Number of Bits', 'Variance', 'Covariance',
    'Auto Correlation', 'Magnitude', 'Radius', 'Weight',
]
INTEGER_FEATURES = ['fin_flag_number', 'syn_flag_number', 'rst_flag_number',
                    'psh_flag_number', 'ack_flag_number', 'ece_flag_number',
                    'cwr_flag_number', 'urg_flag_number', 'numberofpkts']
MUTABLE_FEATURES = BASE_MUTABLE + ['urg_flag_number', 'numberofpkts',
                                    'Number of Bytes', 'Number of Bits',
                                    'Auto Correlation']
```

Then add a helper at the bottom:

```python
import numpy as np

def get_feature_indices(feats):
    return [FEATURE_NAMES.index(f) for f in feats if f in FEATURE_NAMES]

PERTURBATION_MASK = np.zeros(len(FEATURE_NAMES), dtype=np.float32)
for i in get_feature_indices(MUTABLE_FEATURES):
    PERTURBATION_MASK[i] = 1.0
```

Verify: `assert len(set(FEATURE_NAMES)) == len(FEATURE_NAMES)` (no duplicates) and `assert len(FEATURE_NAMES) in {45, 46}`.

---

## PHASE 1 — ROW COUNTING (DATASET PROFILING)

### 1.1 Enumerate CSVs

List every `*.csv` in `data/raw/` recursively. Compute and record the MD5 hash and row count of each (chunk-read to avoid loading full files).

### 1.2 Class distribution scan

Stream through all CSVs with `pd.read_csv(path, chunksize=500_000, usecols=['label'])` and tally per-class row counts. This should complete in minutes even on 46M rows.

### 1.3 Produce preliminary report

Print to the console **and** write to `logs/eda.log`:

```
================ CICIoT2023 PROFILE ================
Files:        <n> CSVs, total size <X> GB
Total rows:   <N>
Unique classes: <K>
Per-class counts (sorted desc):
  DDoS-ICMP_Flood: <n>  (<pct>%)
  DDoS-UDP_Flood:  <n>  (<pct>%)
  ...
  Uploading_Attack: <n>  (<pct>%)

Category rollups:
  DDoS:       <n>   (from 12 attacks)
  DoS:        <n>   (from 4 attacks)
  Mirai:      <n>   (from 3 attacks)
  Benign:     <n>   (from 1 class)
  Spoofing:   <n>   (from 2 attacks)
  Recon:      <n>   (from 5 attacks)
  Web:        <n>   (from 6 attacks)
  BruteForce: <n>   (from 1 attack)

Imbalance ratio (max_class / min_class): <r>
=====================================================
```

### 1.4 Imbalance audit — T0 (produced NOW, before EDA proceeds)

Before any sampling decision, produce `tables/T0_imbalance_audit.{csv,md}`. One row per 34-class label, sorted by count asc (rarest first). Columns:

| Column | Meaning |
|---|---|
| `class` | e.g. `Uploading_Attack` |
| `category` | from `CATEGORY_MAP` in 3.3 |
| `count` | total rows |
| `pct_total` | count / total rows |
| `imbalance_ratio_vs_max` | count / max_class_count (smaller = rarer) |
| `tier` | see below |
| `thesis_strategy` | see below |

Assign **`tier`** by count:

- `majority`: count ≥ 1,000,000 (DDoS floods, large DoS, Mirai, Benign)
- `medium`: 100,000 ≤ count < 1,000,000
- `minority`: 10,000 ≤ count < 100,000
- `rare`: count < 10,000  (includes `Uploading_Attack` at ~1,252, `Recon-PingSweep` at ~2,262, and all Web attacks)

Assign **`thesis_strategy`** based on tier:

- `majority`  → "cap at 200K for baseline; cap at 200K for VAE training"
- `medium`    → "keep full; use class-weighted loss in baseline"
- `minority`  → "keep full; class-weighted loss; VAE trainable"
- `rare`      → "keep full; class-weighted loss; VAE at CATEGORY level (not per-class); flag reconstruction quality"

Also print a summary line:

```
Rare-class concerns: <k> classes below 10K rows. Per-class VAE training for
these is NOT RECOMMENDED — train VAEs at the category level for Web and
BruteForce, and evaluate attacks per-attack using those category-level VAEs.
```

### 1.5 STOP and ask me

Print both the profile from 1.3 and the T0 audit from 1.4, then ask:

1. "Proceed in **Mode FULL** or **Mode SAMPLE**?"
   - Mode FULL: load all ~46M rows. Requires 32+ GB RAM. Most rigorous for baseline metrics.
   - Mode SAMPLE: apply per-class caps. Recommended default for a thesis.
2. If SAMPLE, confirm the **three-tier cap strategy**:
   - Majority classes: cap at **200,000 rows** each.
   - Medium + Minority classes: cap at **min(count, 200,000)** — effectively kept whole.
   - Rare classes: **keep ALL rows**, no cap.

   Reasoning: total loaded rows ≈ 200K × ~10 majority classes + ~600K medium + ~120K minority + ~30K rare ≈ **~2.8M rows**. Tractable for VAE training and downstream experiments, preserves rare-class signal entirely.
3. "Apply any class-weighted loss / minority-class augmentation?" Recommend:
   - **YES to class-weighted loss** (computed in Phase 4).
   - **NO to SMOTE / ADASYN / Gaussian noise augmentation.** Rationale: these methods generate synthetic tabular samples that *violate protocol constraints* (fractional binary features, impossible Protocol Type values, Min > Max). If we apply them before VAE training, we contaminate the manifold that our on-manifold adversarial attacks are supposed to stay within, undermining the thesis's central claim. Real-data-only is required.
   - **NO to oversampling by duplication.** Increases training cost without adding information and risks overfitting on duplicated minorities.

Do not proceed until I confirm all three.

---

## PHASE 2 — FEATURE MODULES

### 2.1 Build `src/validator.py`

Validator operates on original (unscaled) feature-space DataFrames. It returns a `ValidationResult` object with per-rule boolean arrays. Implement exactly these rule groups. **Every rule must use a `col_exists` guard so the same validator works on Schema A and Schema B.**

```python
# src/validator.py
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List

@dataclass
class ValidationResult:
    n_samples: int
    violations_per_rule: Dict[str, np.ndarray]

    @property
    def overall_valid(self) -> np.ndarray:
        if not self.violations_per_rule:
            return np.ones(self.n_samples, dtype=bool)
        V = np.stack(list(self.violations_per_rule.values()), axis=1)
        return ~V.any(axis=1)

    @property
    def validity_rate(self) -> float:
        return float(self.overall_valid.mean())

    def per_rule_violation_rate(self) -> Dict[str, float]:
        return {k: float(v.mean()) for k, v in self.violations_per_rule.items()}

    def summary(self) -> str:
        rates = self.per_rule_violation_rate()
        out = [f"Validity rate: {self.validity_rate:.4%} "
               f"({int(self.overall_valid.sum())}/{self.n_samples})"]
        out.append("Per-rule violation rates (non-zero only):")
        for k, r in sorted(rates.items(), key=lambda x: -x[1]):
            if r > 0:
                out.append(f"  {k}: {r:.4%}")
        return "\n".join(out)


def validate_batch(X: np.ndarray, feature_names: List[str]) -> ValidationResult:
    df = pd.DataFrame(X, columns=feature_names)
    n = len(df)
    V = {}

    def has(c): return c in df.columns
    def C(c): return df[c].values if has(c) else None

    # G1 — Non-negativity
    for c in ['flow_duration','Header_Length','Rate','Srate','Drate',
              'Tot sum','Min','Max','AVG','Std','Tot size','IAT',
              'Number','numberofpkts','Number of Bytes','Number of Bits',
              'Variance','Weight','Magnitude','Radius']:
        if has(c):
            V[f'R_nonneg_{c}'] = df[c].values < 0
    for c in ['fin_flag_number','syn_flag_number','rst_flag_number',
              'psh_flag_number','ack_flag_number','ece_flag_number',
              'cwr_flag_number','urg_flag_number',
              'ack_count','syn_count','fin_count','urg_count','rst_count']:
        if has(c):
            V[f'R_nonneg_{c}'] = df[c].values < 0

    # G2 — Protocol Type in valid IP protocol numbers
    if has('Protocol Type'):
        proto = np.round(df['Protocol Type'].values).astype(int)
        V['R_protocol_valid'] = ~np.isin(proto, [0, 1, 6, 17])

    # G3 — Binary features ∈ {0, 1}
    for c in ['HTTP','HTTPS','DNS','Telnet','SMTP','SSH','IRC',
              'TCP','UDP','DHCP','ARP','ICMP','IPv','LLC']:
        if has(c):
            V[f'R_binary_{c}'] = ~np.isin(np.round(df[c].values).astype(int), [0, 1])

    # G4 — Protocol ↔ transport-layer indicator consistency
    if has('Protocol Type') and has('TCP'):
        p = np.round(df['Protocol Type'].values).astype(int)
        t = np.round(df['TCP'].values).astype(int)
        V['R_proto_tcp']  = ((p == 6)  & (t != 1)) | ((p != 6)  & (t != 0))
    if has('Protocol Type') and has('UDP'):
        p = np.round(df['Protocol Type'].values).astype(int)
        u = np.round(df['UDP'].values).astype(int)
        V['R_proto_udp']  = ((p == 17) & (u != 1)) | ((p != 17) & (u != 0))
    if has('Protocol Type') and has('ICMP'):
        p = np.round(df['Protocol Type'].values).astype(int)
        i = np.round(df['ICMP'].values).astype(int)
        V['R_proto_icmp'] = ((p == 1)  & (i != 1)) | ((p != 1)  & (i != 0))

    # G5 — Statistical ordering Min ≤ AVG ≤ Max
    if has('Min') and has('Max'):
        V['R_min_leq_max'] = df['Min'].values > df['Max'].values
    if has('Min') and has('AVG') and has('Max'):
        V['R_avg_in_range'] = (df['AVG'].values < df['Min'].values) | \
                              (df['AVG'].values > df['Max'].values)

    # G6 — Variance = Std²  (5% tolerance)
    if has('Std') and has('Variance'):
        exp = df['Std'].values ** 2
        err = np.abs(df['Variance'].values - exp) / (exp + 1e-8)
        V['R_var_eq_std_sq'] = err > 0.05

    # G7 — Bits = 8 × Bytes  (Schema B only; 1% tolerance)
    if has('Number of Bytes') and has('Number of Bits'):
        exp = df['Number of Bytes'].values * 8
        err = np.abs(df['Number of Bits'].values - exp) / (exp + 1e-8)
        V['R_bits_eq_8bytes'] = err > 0.01

    # G8 — TTL / Duration range [0, 255]
    if has('Duration'):
        V['R_ttl_range'] = (df['Duration'].values < 0) | \
                           (df['Duration'].values > 255)

    # G9 — Packet count positive integer
    pkt_col = 'Number' if has('Number') else ('numberofpkts' if has('numberofpkts') else None)
    if pkt_col:
        V['R_pkts_positive'] = df[pkt_col].values < 1
        V['R_pkts_integer']  = np.abs(df[pkt_col].values -
                                       np.round(df[pkt_col].values)) > 0.5

    return ValidationResult(n_samples=n, violations_per_rule=V)
```

### 2.2 Build `src/netdiffuser_categorization.py`

Implement NetDiffuser Algorithm 1 (feature categorization into Discrete vs Relative via hierarchical clustering of correlation distances with Calinski-Harabasz cut selection):

```python
# src/netdiffuser_categorization.py
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import calinski_harabasz_score


def categorize_features(df: pd.DataFrame, feature_cols: list,
                        method='spearman', h_grid_points=30):
    """
    NetDiffuser Algorithm 1: partition features into Discrete vs Relative.
    Returns: dict with 'discrete', 'relative', 'linkage_matrix', 'best_cut',
             'ch_scores', 'correlation_matrix'.
    """
    # 1. Pairwise correlations → distance matrix
    corr = df[feature_cols].corr(method=method).abs().values
    np.fill_diagonal(corr, 1.0)
    dist = np.sqrt(np.maximum(2 * (1 - corr), 0.0))

    # 2. Agglomerative clustering
    # Use condensed form; distance matrix is symmetric
    from scipy.spatial.distance import squareform
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method='average')

    # 3. Grid over cut heights; pick h that maximizes ΔCH
    h_min, h_max = float(Z[:, 2].min()), float(Z[:, 2].max())
    h_grid = np.linspace(h_min, h_max, h_grid_points)

    ch_scores = []
    data = df[feature_cols].values  # rows=samples, cols=features
    features_as_rows = data.T  # NetDiffuser clusters features, not samples
    for h in h_grid:
        labels = fcluster(Z, t=h, criterion='distance')
        if len(set(labels)) < 2 or len(set(labels)) >= len(feature_cols):
            ch_scores.append(np.nan)
        else:
            try:
                ch = calinski_harabasz_score(features_as_rows, labels)
            except Exception:
                ch = np.nan
            ch_scores.append(ch)
    ch_scores = np.array(ch_scores)

    # Pick cut by largest successive improvement in CH
    diffs = np.diff(ch_scores)
    valid = ~np.isnan(diffs)
    if not valid.any():
        best_i = int(np.nanargmax(ch_scores))
    else:
        best_i = int(np.nanargmax(np.where(valid, diffs, -np.inf)))
    best_h = float(h_grid[best_i])
    labels = fcluster(Z, t=best_h, criterion='distance')

    # Features in clusters of size > 1 are Relative; singletons are Discrete
    label_sizes = pd.Series(labels).value_counts().to_dict()
    relative = [feature_cols[i] for i, L in enumerate(labels) if label_sizes[L] > 1]
    discrete = [feature_cols[i] for i, L in enumerate(labels) if label_sizes[L] == 1]

    return {
        'discrete': discrete,
        'relative': relative,
        'linkage_matrix': Z,
        'best_cut': best_h,
        'ch_scores': ch_scores.tolist(),
        'h_grid': h_grid.tolist(),
        'correlation_matrix': corr.tolist(),
    }
```

---

## PHASE 3 — EXPLORATORY DATA ANALYSIS

This phase operates on the **full dataset** in Mode FULL, or the **stratified sample** in Mode SAMPLE.

### 3.1 Load data

**Mode FULL:** concat all CSVs using pandas with `low_memory=False`. If memory is tight, load per-CSV and concat lazily. Apply `.replace([np.inf, -np.inf], np.nan).dropna()` once at the end. Log the row count before/after.

**Mode SAMPLE — three-tier strategy:** for each class, determine the cap from its tier in T0:

```python
CAPS_BY_TIER = {
    'majority': 200_000,
    'medium':   200_000,   # effectively keeps full class since medium ≤ 1M but ≥100K;
                           # the cap matters only for medium classes > 200K
    'minority': None,      # None = keep full
    'rare':     None,      # None = keep full
}
```

Implementation: two-pass over CSVs.

**Pass 1** (fast, label-only): for each CSV, read only the `label` column and record, per class, the list of `(csv_path, row_index_within_file)` positions. This builds a global mapping `class → list of (path, local_idx)`.

**Pass 2** (actual loading): for each class, if its tier has `None` cap, select all positions; otherwise take a deterministic random sample of size `cap` using `numpy.random.default_rng(seed=42)`. Group the selected positions by CSV path, read each CSV once, and slice the needed rows. Concat all slices into the final DataFrame.

Verify after loading:
- Majority classes have exactly 200,000 rows (± rows lost to NaN/Inf cleaning).
- Rare classes have their full original count.
- Print per-class "loaded vs original" counts.

Always coerce `label` to string, strip whitespace. Do not lowercase (the paper's class names are case-sensitive like `DDoS-ICMP_Flood`).

Save the loaded DataFrame to `data/processed/raw_loaded.parquet` (parquet, not CSV — 10× smaller and faster to reload).

### 3.2 T1 — Feature schema table

For each feature in `FEATURE_NAMES`, output columns: Name, Type, Description, Domain Range, Mean, Std, Min, 25%, 50%, 75%, Max.

Types:
- `binary` for features in `BINARY_FEATURES`
- `integer` for features in `INTEGER_FEATURES`
- `categorical` for `Protocol Type`
- `float` otherwise

Descriptions: use the CICIoT2023 paper Table 4 descriptions. For the 5 count features in Schema A or the extra 4 features in Schema B, use these:
- `ack_count`: number of packets with ACK flag set in the same flow
- `syn_count`: number of packets with SYN flag set in the same flow
- `fin_count`, `urg_count`, `rst_count`: analogous
- `Number of Bytes`: total bytes in flow
- `Number of Bits`: 8 × Number of Bytes
- `Auto Correlation`: autocorrelation of inter-arrival times
- `numberofpkts`: total packet count in flow

Save to `tables/T1_feature_schema.csv` and `tables/T1_feature_schema.md`.

### 3.3 T2 — Class counts

Two tables:

**T2a — Per-class** (34 rows sorted by count desc): class name, category, count, percentage.

**T2b — Per-category** (8 rows: 7 attack categories + Benign): category, count, percentage, number of sub-attacks.

Canonical category mapping:

```python
CATEGORY_MAP = {
    # DDoS (12)
    'DDoS-ICMP_Flood': 'DDoS', 'DDoS-UDP_Flood': 'DDoS',
    'DDoS-TCP_Flood': 'DDoS', 'DDoS-PSHACK_Flood': 'DDoS',
    'DDoS-SYN_Flood': 'DDoS', 'DDoS-RSTFINFlood': 'DDoS',
    'DDoS-SynonymousIP_Flood': 'DDoS', 'DDoS-UDP_Fragmentation': 'DDoS',
    'DDoS-ACK_Fragmentation': 'DDoS', 'DDoS-ICMP_Fragmentation': 'DDoS',
    'DDoS-HTTP_Flood': 'DDoS', 'DDoS-SlowLoris': 'DDoS',
    # DoS (4)
    'DoS-UDP_Flood': 'DoS', 'DoS-TCP_Flood': 'DoS',
    'DoS-SYN_Flood': 'DoS', 'DoS-HTTP_Flood': 'DoS',
    # Mirai (3)
    'Mirai-greeth_flood': 'Mirai', 'Mirai-udpplain': 'Mirai',
    'Mirai-greip_flood': 'Mirai',
    # Benign
    'BenignTraffic': 'Benign',
    # Spoofing (2)
    'MITM-ArpSpoofing': 'Spoofing', 'DNS_Spoofing': 'Spoofing',
    # Recon (5)
    'Recon-PingSweep': 'Recon', 'Recon-OSScan': 'Recon',
    'Recon-PortScan': 'Recon', 'Recon-HostDiscovery': 'Recon',
    'VulnerabilityScan': 'Recon',
    # Web (6)
    'BrowserHijacking': 'Web', 'Backdoor_Malware': 'Web',
    'XSS': 'Web', 'SqlInjection': 'Web',
    'CommandInjection': 'Web', 'Uploading_Attack': 'Web',
    # BruteForce (1)
    'DictionaryBruteForce': 'BruteForce',
}
```

If your class strings differ (casing/spacing), normalize and log the mapping. Save mapping to `data/processed/class_to_category.json`.

### 3.4 T3 — Summary statistics table

Full per-feature describe (count, mean, std, min, 25/50/75%, max) + skewness + kurtosis + % zeros + % unique. Save `tables/T3_summary_stats.{csv,md}`.

### 3.5 T4 — Clean-data validity audit (CRITICAL)

Run the validator on the full/sampled loaded dataset. Expected outcome: **every rule is satisfied by ≥99% of clean data.** This is your ground-truth check — both on the validator and on the dataset integrity.

Output `tables/T4_clean_validity.{csv,md}` with columns: `rule_name`, `violation_rate`, `n_violations`, `interpretation`.

If any rule shows violation rate > 1% on clean data:
- Do NOT silently proceed.
- Log a `WARNING` with the rule name and sample of violating rows (up to 10).
- Add a note in T4 under "Known Dataset Quirks" explaining the discrepancy.
- The most likely culprit is `R_var_eq_std_sq` or `R_bits_eq_8bytes` due to float precision in the original feature extraction; if violation rate is <5% for those two, tag as benign.

### 3.6 T5 — Sample rows (printable)

For 4 classes (`BenignTraffic`, `DDoS-ICMP_Flood`, `Recon-PortScan`, `Mirai-greeth_flood`), sample 5 rows each uniformly at random (seed=42). Show **all features** in original units (not scaled). Transpose the view for readability: features as rows, samples as columns.

Save as:
- `tables/T5_sample_rows_wide.csv` (standard rows × features)
- `tables/T5_sample_rows_transposed.md` (features × samples, for the thesis)

### 3.7 F1 — Class distribution bar chart

Two-panel horizontal bar chart, log-scaled x-axis:
- Panel A: 34 classes sorted descending, colored by category.
- Panel B: 8 categories sorted descending.

Annotate bars with the raw count. Save `figures/F1_class_distribution.pdf`.

### 3.8 F2 — Feature KDE overlays (benign vs attack)

Select 6 key features:
- `flow_duration`, `Rate`, `IAT`, `AVG`, `Tot size`
- Schema-appropriate packet count: `Number` or `numberofpkts`

2×3 grid. Each subplot overlays KDE for: BenignTraffic, DDoS-ICMP_Flood, DoS-SYN_Flood, Recon-PortScan, Mirai-greeth_flood. Clip at 99th percentile for readability. Log-scale x-axis where beneficial (flow_duration, Rate, Tot size).

Save `figures/F2_feature_distributions.pdf`.

### 3.9 F3 — Correlation heatmap

Spearman absolute correlation across all numeric features. Mask upper triangle. Colormap `YlOrRd`. No annotations (too dense). Row/column labels with small font. Annotate feature groups with colored sidebar (binary/integer/float).

Save `figures/F3_correlation_heatmap.pdf` and the raw matrix as `tables/F3_correlation_matrix.csv`.

### 3.10 F4 — PCA projection (2D)

Use `sklearn.decomposition.PCA(n_components=2)` on RobustScaler-scaled features (fit scaler on a held-out split — see Phase 4, do not leak). For visualization, sample min(class_count, 3000) per class. Color points by category.

Also compute and log:
- Variance explained by PC1, PC2
- Top 5 feature loadings per component (for Schema interpretation in thesis)

Save `figures/F4a_pca_by_category.pdf`, `figures/F4b_pca_loadings.pdf`, and `tables/F4_pca_loadings.csv`.

### 3.11 F5 — t-SNE projection (2D)

On a stratified sample of **min(class_count, 2000) per class** (t-SNE does not scale beyond ~30-50k points). Pre-reduce to 50 PCA components for speed, then `TSNE(n_components=2, perplexity=40, learning_rate='auto', init='pca', random_state=42, n_jobs=-1)`.

Color by category. Save `figures/F5_tsne_by_category.pdf`. Save 2D coordinates + labels to `data/processed/tsne_coords.npz`.

### 3.12 F6 — UMAP projection (2D)

On a stratified sample of **min(class_count, 10000) per class** (UMAP scales much better). `umap.UMAP(n_neighbors=30, min_dist=0.1, metric='euclidean', random_state=42, n_jobs=-1)`.

Color by category. Save `figures/F6_umap_by_category.pdf` and `data/processed/umap_coords.npz`.

Also produce a second UMAP plot zoomed to benign + Mirai only (they cluster together in the latent space per my thesis narrative; this plot will preview that finding). Save as `figures/F6b_umap_mirai_vs_benign.pdf`.

> **If `umap-learn` is not installed, run `pip install umap-learn`. Do not silently skip this figure.**

### 3.13 F7 — NetDiffuser feature categorization

Run `categorize_features()` from Phase 2.2 on the training split's features (fit later in Phase 4; for now use the stratified sample).

Produce:
- `figures/F7a_dendrogram.pdf`: hierarchical clustering dendrogram with the chosen cut height marked as a horizontal line. Color leaves by Discrete (green) vs Relative (red).
- `figures/F7b_ch_scores.pdf`: CH score vs cut height, with the chosen cut marked.
- `data/processed/netdiffuser_categorization.json`: `{discrete: [...], relative: [...], best_cut: ..., ch_scores: [...]}`
- `tables/T6_feature_categorization.md`: two-column table of Discrete features vs Relative features.

### 3.14 Final EDA sanity print

```
================ EDA COMPLETE ================
Schema:       A (paper) | B (community)
Rows used:    <N>  (Mode: FULL | SAMPLE)
Figures generated: 7 (F1, F2, F3, F4a/b, F5, F6a/b, F7a/b)
Tables generated:  6 (T1, T2a/b, T3, T4, T5, T6)
Clean data validity: <%> overall (see T4 for details)
NetDiffuser: <d> Discrete, <r> Relative features
t-SNE sample: <n> points | UMAP sample: <n> points
===============================================
```

---

## PHASE 4 — PREPROCESSING PIPELINE

All outputs go to `data/processed/`. All operations are deterministic given seed 42.

### 4.1 Clean

On the full loaded DataFrame:

1. `df.replace([np.inf, -np.inf], np.nan)` → `df.dropna()`. Log rows dropped.
2. Per numeric feature, clip to `[0, 99.99th percentile]` (except features that can legitimately be negative — `Covariance`). Log clip counts.
3. Round integer features (`INTEGER_FEATURES`) to nearest int then clip to ≥0.
4. Round binary features to {0, 1}.
5. Drop `ts` / `Timestamp` column if present (it is not a feature, only an ordering key).

### 4.2 Label encoding

Use `sklearn.preprocessing.LabelEncoder` fit on the full label column. Save as `data/processed/label_encoder.pkl`. Also save the class name list in order (`data/processed/class_names.json`) for downstream reference.

Additionally, create a **category-level encoder** using `CATEGORY_MAP` from 3.3 (8 classes: 7 attack categories + Benign). Save as `data/processed/category_encoder.pkl` and `data/processed/category_names.json`.

This lets downstream experiments choose between 34-class, 8-class, and binary (benign vs any-attack) framings.

### 4.3 Stratified split

Two-stage, all with seed=42:

1. `train_test_split(X, y, test_size=0.20, stratify=y_34class, random_state=42)` → produces `X_trainval`, `X_test`.
2. `train_test_split(X_trainval, y_trainval, test_size=0.125, stratify=y_trainval_34class, random_state=42)` → produces `X_train`, `X_val`. (0.125 × 0.8 = 0.10 of total.)

Final split: **70 / 10 / 20** of total rows.

Verify: every class has ≥1 sample in each of train/val/test. If any class has <10 rows total, log a warning (may be `Uploading_Attack` which has only ~1252 rows — still OK).

### 4.4 Scaling

`RobustScaler().fit(X_train)`. Apply to train, val, test.

Log scaler diagnostics:
- `scaler.center_` (median vector): min/max/mean
- `scaler.scale_` (IQR vector): min/max/mean

Save `data/processed/scaler.pkl`.

### 4.5 Save arrays

All as `.npy` (float32 for X, int32 for y), under `data/processed/`:

```
X_train.npy, X_val.npy, X_test.npy
y_train.npy, y_val.npy, y_test.npy           # 34-class labels
y_train_cat.npy, y_val_cat.npy, y_test_cat.npy  # 8-class labels
y_train_bin.npy, y_val_bin.npy, y_test_bin.npy  # binary: 0=benign, 1=attack
```

Binary label: `1 if class != 'BenignTraffic' else 0`.

### 4.6 Class weights for baseline classifier training

Imbalance in CICIoT2023 is severe (ratio ≈ 5,700× between largest and smallest class). For the downstream baseline classifiers (MLP, CNN, LightGBM), compute per-class weights now so we don't recompute during every training run.

Compute **three** weight vectors — one per label framing — using `sklearn.utils.class_weight.compute_class_weight('balanced', ...)` fitted on the **training split only** (never on the full dataset; that would leak from val/test):

```python
from sklearn.utils.class_weight import compute_class_weight

# 34-class
w34 = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
# 8-class (category)
w8  = compute_class_weight('balanced', classes=np.unique(y_train_cat), y=y_train_cat)
# 2-class (binary)
w2  = compute_class_weight('balanced', classes=np.unique(y_train_bin), y=y_train_bin)
```

Save all three as `.npy` under `data/processed/`:
- `class_weights_34.npy` (float32, shape `(34,)`)
- `class_weights_8.npy` (float32, shape `(8,)`)
- `class_weights_2.npy` (float32, shape `(2,)`)

Also save a JSON with explicit name→weight mapping for readability:

```
data/processed/class_weights_named.json:
{
  "34class": {"DDoS-ICMP_Flood": 0.234, ..., "Uploading_Attack": 187.3},
  "8class":  {"DDoS": 0.412, ..., "BruteForce": 12.7},
  "binary":  {"BenignTraffic": 22.1, "attack": 0.52}
}
```

Log the top 5 highest and lowest weights in the 34-class vector. The highest should correspond to rare-tier classes from T0.

**Important note for downstream consumers:** these weights are intended for `CrossEntropyLoss(weight=torch.tensor(w))` in PyTorch or `class_weight=dict(enumerate(w))` / `sample_weight` in sklearn/LightGBM. Do NOT apply them to the VAE loss in Phase G — the per-class VAEs are trained on single-class data and class weighting is irrelevant there.

### 4.7 Perturbation mask

Build using `MUTABLE_FEATURES` from `feature_groups.py` AND the NetDiffuser `discrete` list from Phase 3.13:
- Mutable AND Discrete → mask value 1.0 (fully perturbable)
- Mutable AND Relative → mask value 0.3 (perturbable but dependency-heavy)
- Not Mutable → mask value 0.0

Save as `data/processed/perturbation_mask.npy` (shape `(n_features,)`, float32).

### 4.8 Content hashes

Compute SHA-256 of:
- `X_train.npy`, `X_val.npy`, `X_test.npy` (raw bytes)
- `scaler.center_.tobytes() + scaler.scale_.tobytes()`
- `label_encoder.classes_.tobytes()`

Save all to `data/processed/content_hashes.json`.

### 4.9 Write `config/run_manifest.json`

Include:
- ISO timestamp
- Git commit hash: `subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()` (catch CalledProcessError and use `"not-a-git-repo"` as fallback)
- Python version, torch/numpy/pandas/sklearn/umap-learn versions
- Seeds used (all 42)
- Schema used (A or B)
- Mode used (FULL or SAMPLE) and sample cap if applicable
- Number of rows: loaded, after clean, per split
- Per-file MD5 hashes of raw CSVs (from Phase 1)
- Content hashes from 4.7
- Paths of all output artifacts

### 4.10 Final sanity report

```
================ CICIoT2023 PREPROCESSING COMPLETE ================
Schema:           A | B
Mode:             FULL | SAMPLE (three-tier caps)
Rows loaded:      <N>
  majority tier:  <n_maj> rows from <k_maj> classes (capped at 200K each)
  medium tier:    <n_med> rows from <k_med> classes
  minority tier:  <n_min> rows from <k_min> classes
  rare tier:      <n_rare> rows from <k_rare> classes (kept in full)
After cleaning:   <N'>   (dropped <d>)
Train/Val/Test:   <a> / <b> / <c>  (70/10/20 stratified)
Features:         <k>
Classes:          34 (full) | 8 (category) | 2 (binary)

Scaler IQR:       min=<min>, max=<max>, mean=<mean>
Top-5 largest IQRs: <feat1>=<iqr>, <feat2>=<iqr>, ...
Top-5 smallest IQRs (risk zone for PGD): <feat>=<iqr>, ...

Clean data validity: <%> overall
  worst rule: <rule> at <%>

Class weights (34-class):
  highest 3: <rare_class_1>=<w>, <rare_class_2>=<w>, <rare_class_3>=<w>
  lowest  3: <majority_1>=<w>, <majority_2>=<w>, <majority_3>=<w>
  weight range: [<min>, <max>]  (ratio: <max/min>×)

Perturbation mask: <n_full> features at 1.0, <n_partial> at 0.3,
                   <n_frozen> frozen (incl. <immutables>)

Augmentation applied: NONE (SMOTE/ADASYN skipped by design — would
                            contaminate the VAE's target manifold).
Imbalance handling: class-weighted loss vectors saved for downstream use.

All artifacts saved. Content hashes committed to run_manifest.json.
====================================================================
```

---

## STRICT ORDERING

You must do phases in order: 0 → 1 → STOP → 2 → 3 → 4 → DONE. Do not skip ahead. If a phase fails, fix the cause; do not proceed with degraded output.

## WHAT "DONE" LOOKS LIKE

- 7 figures in `figures/` (F1, F2, F3, F4a/b, F5, F6a/b, F7a/b — count pairs as one figure): all PDF, 150 DPI.
- 7 tables in `tables/` (T0 imbalance audit, T1 schema, T2a/b class counts, T3 summary stats, T4 clean validity, T5 sample rows, T6 feature categorization), both CSV and MD.
- 14 arrays in `data/processed/` (X_train/val/test, y × 3 framings × 3 splits, perturbation_mask, class_weights × 3 framings).
- `scaler.pkl`, `label_encoder.pkl`, `category_encoder.pkl`.
- `content_hashes.json`, `run_manifest.json`, `netdiffuser_categorization.json`, `clean_data_validity_report.json`, `class_weights_named.json`, `class_to_category.json`.
- `src/feature_groups.py`, `src/validator.py`, `src/netdiffuser_categorization.py`.
- Log files in `logs/`.

When every artifact above exists and the final sanity report prints cleanly, the task is complete. Do not generate a summary document — the run manifest and sanity report are the summary.

## LIBRARIES YOU MAY NEED

```bash
pip install pandas numpy scikit-learn matplotlib seaborn umap-learn scipy joblib pyarrow
```

Use `pyarrow` for parquet. Do not use `fastparquet`.

---

## BEGIN WITH PHASE 0. When you hit the Phase 1 checkpoint, print the profile and ask me.
