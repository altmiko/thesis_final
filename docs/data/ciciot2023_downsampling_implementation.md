# Downsampling Strategy — Implementation Log

Implementation of `downsampling_strategy.md` (Rafsan Rahman, BRAC CSE thesis).
This log records **what was built, why, and how it was verified**, so the
architecture and every non-obvious decision is legible after the fact.

> Spec = `downsampling_strategy.md`. Section refs below (§N) point into it.
> One-line summary of the method: *intra-class clustering-based undersampling
> (selection variant) with cluster-proportional-with-floor allocation, applied
> train-only after a forward-chaining shard-level temporal split, keeping real
> rows and rare classes whole.*

---

## 0. Environment & ground-truth facts established before coding

| Fact | Value | Source |
|---|---|---|
| Python env | `C:/Users/user6/.local/share/mamba/envs/thesis/python.exe` (3.11, np 2.4.4, sklearn 1.9.0, torch 2.5.1+cu121, CUDA available) | probed |
| RAM | 68.5 GB total / ~58 GB free | psutil |
| Full labelled source | `data/processed/ciciot2023_labeled_full.parquet` — 46,775,660 rows, **309 row groups (1 shard per row group)**, ZSTD | pyarrow |
| Parquet columns | 39 features + `Label` (34-class UPPER) + `category` (8-class) + `source_csv_filename` (rel. shard path) + `source_folder` | pyarrow schema |
| Row order | shard-by-shard in folder-walk order; **preserved** (pure concatenator build) → block splits are meaningful (§4.3) | build script + row-group probe |
| Continuous features for clustering | **23** = 39 − 15 `BINARY_FEATURES` − `Protocol Type` (matches §5.3's "23 RobustScaler'd ones") | computed |
| Schema source of truth | `src/preprocessing/feature_groups.py` (`FEATURE_NAMES`, `BINARY_FEATURES`, `CATEGORY_MAP`, near-zero governance) — imported, not duplicated. Promoted out of the now-deleted `old/` during cleanup. | build script comment |

**Downstream artifact contract** (union of every filename any live consumer in
`src/vae`, `src/classifiers`, `src/attack`, `src/evaluation`, `src/thesis_eval`,
`runner.py` loads from `data/processed/`), which the new pipeline MUST reproduce:

- Arrays: `X_{train,val,test}.npy` (N×39 float32, RobustScaler model-space),
  `y_{train,val,test}.npy` (34-class int), `y_{...}_cat.npy` (8-class),
  `y_{...}_bin.npy` (binary), `class_weights_{34,8,2}.npy`, `perturbation_mask.npy` (len 39).
- Pickles: `scaler.pkl` (RobustScaler, loadable by pickle **and** joblib),
  `label_encoder.pkl` (34-class), `category_encoder.pkl` (8-class).
- JSON: `class_names.json` (34), `category_names.json` (8),
  `near_zero_iqr_features.json`, `netdiffuser_categorization.json`
  (keys `discrete`/`relative`), `class_to_category.json`.

`X_test.npy`/`X_val.npy` are large (≈1.4 GB / 0.7 GB) because val/test are kept
at **natural proportions** (never sampled) — this is the point (§4.1).

---

## 1. Corrections carried from spec §12 that shaped the code

The spec's own changelog fixes three factual errors from its previous revision.
These directly changed the code:

1. **Shards are byte-size `tcpdump` slices, not capture sessions** → the split is
   a *forward-chaining temporal holdout*, **not** `StratifiedGroupKFold` (which
   assigns groups randomly and can train-on-future/test-on-past = data snooping).
2. **`source_csv_filename` is a contiguous time segment**; its numeric suffix is
   a wall-clock sequence number → natural-numeric shard order = time order.
3. **The SMALL/LARGE pool split** in the sampler — the old `max(floor, raw)`
   allocation was unsatisfiable when a cluster is smaller than the floor.

---

## 2. Modules built (Foundation phase — DONE)

### `config/paths.py` (NEW)
Central path + constant module (spec §7.1–7.3 mandate: no hardcoded absolute
paths; the old pipeline was pinned to `D:/thesis_final`). Exposes `REPO_ROOT`,
`PROCESSED_DIR`, `LABELED_PARQUET`, `LABELED_MANIFEST`, `DIAGNOSTICS_DIR`,
`RUN_MANIFEST`, global `SEED = 42`, and helpers `processed(name)` / `ensure_dirs()`.

### `src/preprocessing/ciciot2023/splitter.py` (NEW — spec §7.2)
Leakage-critical split protocol. Pure (numpy + stdlib only):
- `natkey(path)` — natural-numeric key. **Deviation from spec §4.2, documented
  and justified:** the spec's literal `natkey` glues the `.pcap.csv` extension
  onto the base shard's first token, so the suffix-less base file (`tcpdump`
  sequence 0, the *earliest* segment) wrongly sorts *after* sequence 1. We strip
  the trailing alphabetic extension first, so the base sorts first — strictly
  more faithful to the spec's stated intent ("natural-numeric order IS temporal
  order", §1.3). Verified by unit test.
- `forward_chain_shards(shards, val_frac, test_frac)` — ≥3-shard classes; latest
  shards → test (§4.2).
- `block_split_single_shard(n_rows, ...)` / `block_split_two_way(...)` —
  contiguous, order-preserving row cuts for 1-shard / 2-shard classes (§4.3).
- `plan_class_split(label, shards, shard_rows, ...)` — dispatch on shard count
  (≥3 → forward-chain, 2 → hybrid, 1 → block), returns a `ClassSplitPlan`.
- `assert_forward_chaining(plan)` — leakage guard (§6.5): no test row precedes a
  train row within a class; no shard in two splits.

### `src/preprocessing/ciciot2023/sampler.py` (NEW/rewritten — spec §7.1, §5.1)
Intra-class cluster-proportional-floor selection. Dataset-agnostic (continuous
matrix, N, k, floor passed in → reusable for the two ablation datasets, §10).
- `proportional_floor_allocations(cluster_sizes, target_n, floor)` implements the
  corrected **SMALL/LARGE pool split**: clusters ≤ floor are taken whole and
  removed from the proportional pool; the rest share the remaining budget via a
  **bounded largest-remainder water-fill** (`_bounded_proportional`) clamped to
  `[floor, size]`, so `sum(alloc) == target_n` exactly when feasible.
  `cap_not_binding=True` flags the step-5 `break` path (floors ≥ budget) — the
  floor guarantee wins over the cap and a warning is logged (§5.1).
- `cluster_proportional_floor_sample(...)` — MiniBatchKMeans (seeded) + seeded
  within-cluster shuffle (`random_within`, default) or `nearest_centroid`
  (§5.2). A row is kept iff its cluster assignment + shuffle position < alloc.
- `random_floor_sample(...)` — honest plain-seeded fallback for a class the
  multi-modality diagnostic finds unimodal (§6.2), so we don't claim structure
  preservation we didn't get.

### `src/preprocessing/ciciot2023/tests/` (NEW — spec §7.1/§7.2 checklists)
`test_sampler.py` + `test_splitter.py`, **14 tests, all passing**. Covers the two
spec-mandated sampler cases (`size[c] < floor` → exact sum, no infinite loop;
`N` near total → `cap_not_binding` fires without corrupting counts), determinism,
rare-mode survival, and every split protocol + the forward-chaining guard.

**Verification:** `pytest src/preprocessing/ciciot2023/tests/ -q` → `14 passed`.

---

## 3. Diagnostics (spec §6.2 / §6.3 / §11 step 2 — DONE)

`python -m src.preprocessing.ciciot2023.reports diagnostics` (module `reports.py`).
Per majority category: sample ≤50k train rows, cluster on the 23 scaled
continuous features for `k ∈ {10,15,20}`, measure cluster-size skew (Gini) and
cluster→subtype purity, plot both.

**Result — all six majority categories are multi-modal, so all use
cluster-proportional-floor (none falls back to random):**

| category | n_train | n_sub | chosen k | Gini | subtype purity | verdict |
|---|---:|---:|---:|---:|---:|---|
| DDoS | 24,271,179 | 12 | 20 | 0.57 | 0.66 | multimodal |
| DoS | 5,286,875 | 4 | 20 | 0.55 | 0.71 | multimodal |
| Mirai | 1,904,377 | 3 | 10 | 0.55 | 0.69 | multimodal |
| Recon | 483,369 | 5 | 15 | 0.55 | 0.62 | multimodal |
| Spoofing | 341,958 | 2 | 15 | 0.50 | 0.67 | multimodal |
| Benign | 657,907 | 1 | 10 | 0.46 | 1.00¹ | multimodal |

¹ trivially pure (one sub-label), but **multi-modal in feature space** (Gini
0.46) — answering spec §6.2's explicit open question about Benign.

`k` per category = the grid value maximising cluster→subtype purity (ties k to
sub-attack structure, spec §6.4). Chosen parameters live in the constants block
of **`pipeline.py`** (`CAP_PER_CATEGORY`, `K_PER_CATEGORY`, `FLOOR`, …): cap =
200,000 per majority category; floor =
500 (≫ latent dim 16 and 39-dim Σ rank, yet `max k·floor = 10k ≪ 200k` so the
cap stays binding); rare `Web`/`BruteForce` kept whole; `random_within` pick.
Evidence: `data/processed/downsampling_diagnostics/multimodality.json` +
`clustersize_<CAT>.png` + `crosstab_<CAT>{.png,_k*.csv}`.

## 4. Pipeline (spec §4.1 / §7.3 — DONE)

`src/preprocessing/ciciot2023/pipeline.py` (run: `python -m src.preprocessing.ciciot2023.pipeline`).
`pipeline.py` is the core module: it holds the parameter constants, the shared
parquet-IO helpers (`load_metadata`/`compute_split`/`load_features`), and the
cleaning/scaling/sampling/artifact logic. Order exactly per §4.1:

1. Load full parquet (46,775,660 rows, in-memory — 68 GB host).
2. **Split** (`pipeline.compute_split` → `splitter`), before scaling/sampling.
   Forward-chaining guard asserted for every class.
3. **Clean**: lower-clip 0 + **train-only** 99.99-percentile upper clip
   (leakage-safe; the spec sketches clip under "clean" but a whole-dataset
   percentile would leak test extremes, so we take the bound from train alone),
   round integer + binary features.
4. **Fit RobustScaler on train only**, transform all splits.
5. **Sample** majority-category train rows via cluster-proportional-floor;
   `BruteForce`/`Web` pass through whole; val/test untouched.
6. Class weights from the subsampled train; encoders; NetDiffuser
   discrete/relative (from train); near-zero-IQR report; perturbation mask
   (§4.7 logic ported verbatim); `run_manifest.json` with all §9 tracked items.

**Result:** train 32,972,198 → **1,226,533** (6×200k caps + 9,146 BruteForce +
17,387 Web), val 5,555,150, test 8,248,312. Split protocols: 17 forward-chain,
2 two-shard hybrid, 15 single-shard block. Perturbation mask: 12 full / 8
partial / 19 frozen. Runtime ≈ 214 s.

**Artifact contract** (verified against every downstream consumer): all
`X_/y_*` arrays, `class_weights_{34,8,2}`, `scaler.pkl` (pickle+joblib),
`label_encoder.pkl`, `category_encoder.pkl` (classes = Benign…Web, matching
`src/vae/config.py`), `class_names/category_names/class_to_category/
near_zero_iqr_features/netdiffuser_categorization.json`, `perturbation_mask.npy`
— plus new `train_kept_indices.npy` (audit) and `run_manifest.json`.

### Decisions / deviations forced during the run (all documented, none silent)
- **`natkey` extension-strip** (splitter): base shard sorts first — §2 above.
- **Train-only clip bound** instead of whole-dataset — leakage-safe (§4.1).
- **`Min`/`Number` → `allow_mutable`** in `MANUAL_CONCENTRATED_DECISIONS`
  (`feature_groups.py`): both became near-zero-IQR "concentrated" under the
  DDoS-dominated full train (Q25==Q75) but genuinely vary (nunique 1360/99) and
  are already in `FULL_PERTURBABLE_OVERRIDE_FEATURES`; kept mutable, consistent
  with every other concentrated mutable feature. The mask now differs from the
  old sample-based mask because the full train has 27 near-zero-IQR features vs
  the stratified sample's fewer — a legitimate consequence of using the full
  labelled parquet.
- **`netdiffuser_categorization.py` `.values.copy()`** — one-token robustness
  fix; numpy 2.x returns a read-only view that broke `np.fill_diagonal`.

### Leakage guards (spec §6.5 — DONE)
`python -m src.preprocessing.ciciot2023.reports verify` recomputes the split and asserts, in code:
no shard in two splits; every class in all three splits; no test row precedes a
train row within a class; val/test counts equal the natural totals (sampler
never touched them); train total matches the manifest. **All pass.**

## 5. Evidence figures (spec §7.4 — DONE)

`python -m src.preprocessing.ciciot2023.reports evidence`. Produces
`per_split_class_counts.{csv,png}`, `prepost_overlap_<CAT>.png` (PCA-2D
manifold-support overlap of full vs kept + per-feature within-cluster fidelity),
`prepost_within_cluster_<CAT>.png` (pre/post histograms in the largest clusters),
and `fidelity_summary.json`. The cluster-size + crosstab figures come from the
`diagnostics` mode.

**Framing correction (important, recorded so the panel narrative is honest):**
cluster-proportional-**floor intentionally up-weights rare modes**, so the
*whole-class marginal* median/IQR shift by design — that is mode preservation,
not a fidelity loss. The correct fidelity claim is (a) **manifold-support
overlap** (PCA scatter: kept covers the same region) and (b) **within-cluster
shape preservation** (the pick inside a cluster is uniform random). Measured
within-cluster median shift, normalised by each cluster's own IQR, is small
(mean 0.009–0.05; max ≤0.33 for DDoS/DoS/Mirai/Spoofing/Benign, ≤1.0 for one
Recon cluster) — confirming the random within-cluster pick loses nothing.

## 6. Sensitivity study (spec §7.5 — DONE)

`python -m src.preprocessing.ciciot2023.reports sensitivity --device cuda --epochs 4`. Trains the four
gradient-attackable classifiers (`mlp,cnn,lstm,dualpath`) on the 17
shard-splittable classes twice — **row-level shuffled** vs **forward-chaining
temporal** split — holding cleaning, scaler, per-class train cap (20k), and
epochs identical, so the split's temporal discipline is the only variable.

**Result — measured cost of temporal leakage (test accuracy):**

| model | forward-chain | shuffled | inflation |
|---|---:|---:|---:|
| mlp | 0.6942 | 0.7256 | +0.0314 |
| cnn | 0.6166 | 0.6536 | +0.0370 |
| lstm | 0.7220 | 0.7500 | +0.0280 |
| dualpath | 0.6742 | 0.7382 | +0.0640 |
| **mean** | | | **+0.0401** |

So a leakage-blind (shuffled) split inflates accuracy by **≈4 points** on
average. Thesis wording (spec §7.5): temporal leakage inflated accuracy by ~4
points on the classes where the comparison was possible; treat the 15
single-shard classes' figures as an upper bound. (`sensitivity_study.json`.)

> Windows note: `torch` MUST be imported **after** `pyarrow` (the script does
> this) — the reverse order crashes with a DLL access violation.

## 7. File layout (consolidated)

The pipeline lives in **7 code modules** (+ tests), all under `src/preprocessing/`.
The dataset-specific runnable scripts sit in the `ciciot2023/` subpackage; the
repo-wide schema library (`feature_groups`, `netdiffuser_categorization`) stays at
the `src/preprocessing/` top level:

| file | role |
|---|---|
| `config/paths.py` | paths + global `SEED` (spec §7.1–7.3 mandate a path module) |
| `src/preprocessing/feature_groups.py` | schema source of truth (39 features, `CATEGORY_MAP`, mutability/near-zero governance) — imported repo-wide |
| `src/preprocessing/netdiffuser_categorization.py` | Spearman→hierarchical→CH feature partition (discrete/relative) |
| `src/preprocessing/ciciot2023/splitter.py` | pure split primitives + forward-chaining guard (unit-tested) |
| `src/preprocessing/ciciot2023/sampler.py` | pure cluster-proportional-floor sampler (unit-tested) |
| `src/preprocessing/ciciot2023/pipeline.py` | **core**: param constants + parquet IO + clean/scale/sample/artifacts |
| `src/preprocessing/ciciot2023/reports.py` | `diagnostics` / `verify` / `evidence` / `sensitivity` modes (§6–§7 reports) |
| `src/preprocessing/ciciot2023/tests/` | `test_sampler.py`, `test_splitter.py` (14 tests) |

> **`old/` removed (cleanup).** The archived `src/preprocessing/old/` held the
> superseded pre-reorg pipeline. Its two still-live modules — `feature_groups.py`
> and `netdiffuser_categorization.py` — were **promoted** to `src/preprocessing/`
> (which also fixes the many `from preprocessing.feature_groups import …` callers
> in `src/attack` and `src/evaluation` that expected them there); the dead files
> (`old/pipeline.py`, `old/sampler.py`, the three `build_*`/`export_*` scripts)
> were deleted. The superseded outputs in `data/processed/old/` (~2.9 GB) were
> removed too — they are fully regenerated in `data/processed/`.
>
> Earlier this revision also folded `dataset.py`, `sampling_params.py`, and four
> separate report scripts into `pipeline.py` (IO + params) and `reports.py` (the
> four `--mode`s). `splitter.py`/`sampler.py` stay separate because the unit
> tests target them directly.

## 8. How to reproduce (order = spec §11)
```
python -m src.preprocessing.ciciot2023.reports diagnostics   # §6.2/§6.3 → params (evidence)
python -m src.preprocessing.ciciot2023.pipeline              # §4.1 → arrays + manifest
python -m src.preprocessing.ciciot2023.reports verify        # §6.5 leakage guards
python -m src.preprocessing.ciciot2023.reports evidence      # §7.4 fidelity figures
python -m src.preprocessing.ciciot2023.reports sensitivity   # §7.5 leakage bound (GPU)
pytest src/preprocessing/ciciot2023/tests/ -q                # §7.1/§7.2 unit tests
```
All outputs land in `data/processed/` and `data/processed/downsampling_diagnostics/`.
Downstream (VAEs, attacks) now consume the corrected arrays unchanged (spec §11
step 6): retrain/re-run those scripts on the regenerated `data/processed/`.
