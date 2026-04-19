# EDA & Preprocessing Pipeline — Work Log

## Phase 0 — Scaffolding + Schema Detection
**Status: IN PROGRESS**

### 0.1 Directory Structure
- [x] Created: `data/raw/`, `data/processed/`, `figures/`, `tables/`, `src/`, `config/`, `logs/`

### 0.2 Schema Detection
- **Data file**: `data/ciciot2023/ciciot2023_base.csv` (single file, 9.41 GB, ~45M rows)
- **Detected columns**: 40 (39 features + `Label`)
- **Label column**: `Label` (capital L), values appear UPPERCASE (e.g., `MIRAI-GREIP_FLOOD`)
- **Best match**: Schema A (paper) — partial match
  - Has Schema A-exclusive features: `ack_count`, `syn_count`, `fin_count`, `rst_count`, `Number`
  - Missing from Schema A (9): `flow_duration`, `Duration`, `Srate`, `Drate`, `urg_count`, `Magnitude`, `Radius`, `Covariance`, `Weight`
  - Extra vs Schema A (2): `Time_To_Live`, `IGMP`
  - Label column casing: `Label` (detected) vs `label` (Schema A)
- **Conclusion**: NEITHER schema matches exactly. This is a **reduced variant** of Schema A.

### AWAITING USER CONFIRMATION on schema before proceeding.

---

## Phase 1 — Row Counting
**Status: NOT STARTED**

## Phase 2 — Feature Modules
**Status: NOT STARTED**

## Phase 3 — EDA
**Status: DONE**

## Phase 4 — Preprocessing
**Status: DONE**
