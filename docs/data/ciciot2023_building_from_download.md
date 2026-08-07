# CICIoT2023 Dataset — Labelling, Export, and Provenance

This document is the single source of truth for how the CICIoT2023 raw CSV
shards were turned into the labelled artifacts the thesis pipeline consumes.
It covers three separate, auditable stages:

1. **Parquet build** — read 309 vendor shards, attach labels, stream to one
   labelled parquet (`src/preprocessing/build_labeled_dataset.py`).
2. **CSV export** — project the parquet back to one full labelled CSV that
   matches the historical `ciciot2023_base.csv` shape
   (`src/preprocessing/export_labeled_csv.py`).
3. **Verification** — invariants asserted at write time plus post-hoc
   cross-checks proving no rows, no columns, and no within-shard order were
   destroyed.

All operations sit at the top of the pipeline described in
`downsampling_strategy.md` §4:

```
RAW CSV  →  *this document*  →  clean  →  shard-holdout split (§11)  →  scaler fit  →  sample
```

Everything downstream (clip, round, validator, split, scaler, cluster
sampling, per-class VAE training) stays in `src/preprocessing/pipeline.py`
and later modules so cleaning logic is never applied twice with two
different thresholds.

**Outputs currently on disk:**

| Path | Size | Purpose |
|---|---:|---|
| `data/raw/CICIoT2023_CSV_DOWNLOADED/` | 8.4 GB | vendor source (untouched) |
| `data/processed/ciciot2023_labeled_full.parquet` | 594 MB | labelled parquet, 43 cols |
| `data/processed/ciciot2023_labeled_full_manifest.json` | 24 KB | build provenance + row-count breakdowns |
| `data/raw/ciciot2023_full/ciciot2023_base.csv` | 7.0 GB | labelled CSV, 40 cols (historical shape) |

---

## 1. Environment

Runs against the `thesis` mamba env (miniforge). The parquet writer needs
`pyarrow`; the CSV export needs it too (via `pyarrow.parquet.ParquetFile`).

```bash
"C:/Users/user6/.local/share/mamba/envs/thesis/python.exe" -m pip install pyarrow
```

`pyarrow==24.0.0` was installed on 2026-07-09. It is missing from
`environment.yml` even though it is listed in `AGENTS.md` and imported by
the pre-existing `src/preprocessing/old/build_ciciot2023_dataset.py`. Add to
`environment.yml` on the next commit.

Other deps come from `environment.yml` (Python 3.11, pandas 3.0.3, numpy
2.4.4).

---

## 2. Raw Input Structure

```
data/raw/CICIoT2023_CSV_DOWNLOADED/
├── Backdoor_Malware/                 1 shard
├── Benign_Final/                     4 shards
├── BrowserHijacking/                 1 shard
├── CommandInjection/                 1 shard
├── DDoS-ACK_Fragmentation/          13 shards
├── DDoS-HTTP_Flood/                  1 shard
├── DDoS-ICMP_Flood/                 27 shards
├── DDoS-ICMP_Fragmentation/         20 shards
├── DDoS-PSHACK_FLOOD/               16 shards
├── DDoS-RSTFINFLOOD/                16 shards
├── DDoS-SYN_Flood/                  16 shards
├── DDoS-SlowLoris/                   1 shard
├── DDoS-SynonymousIP_Flood/         14 shards
├── DDoS-TCP_Flood/                  18 shards
├── DDoS-UDP_Flood/                  21 shards
├── DDoS-UDP_Fragmentation/          13 shards
├── DNS_Spoofing/                     1 shard
├── DictionaryBruteForce/             1 shard
├── DoS-HTTP_Flood/                   2 shards
├── DoS-SYN_Flood/                    8 shards
├── DoS-TCP_Flood/                   11 shards
├── DoS-UDP_Flood/                   17 shards
├── MITM-ArpSpoofing/                 2 shards
├── Mirai-greeth_flood/              29 shards
├── Mirai-greip_flood/               22 shards
├── Mirai-udpplain/                  25 shards
├── Recon-HostDiscovery/              1 shard
├── Recon-OSScan/                     1 shard
├── Recon-PingSweep/                  1 shard
├── Recon-PortScan/                   1 shard
├── SqlInjection/                     1 shard
├── Uploading_Attack/                 1 shard
├── VulnerabilityScan/                1 shard
├── XSS/                              1 shard
└── README_CSV.pdf                   (ignored — not a directory)
```

- **34 class folders**, **309 CSV shards** total.
- Every shard's first line is identical to the canonical 39-feature header
  and none carries a `Label` column — the class name lives only in the
  parent folder name.
- Raw disk footprint: **8.4 GB** total (largest folder: `DDoS-ICMP_Flood/`
  at 1.3 GB across 27 shards).
- `README_CSV.pdf` in the root is skipped automatically
  (`entry.is_dir()` gate).

---

## 3. Folder → Label → Category Mapping

The 34 folder names are mixed-case (`DDoS-ICMP_Flood`, `MITM-ArpSpoofing`,
`Benign_Final`), but `CATEGORY_MAP` in `feature_groups.py` keys on the
uppercased CICIoT convention. `FOLDER_TO_LABEL` (reused from
`src/preprocessing/build_ciciot2023_dataset.py`) is the single place the
translation is defined:

| Folder | Label (`Label`) | Category |
|---|---|---|
| `Backdoor_Malware` | `BACKDOOR_MALWARE` | Web |
| `Benign_Final` | `BENIGN` | Benign |
| `BrowserHijacking` | `BROWSERHIJACKING` | Web |
| `CommandInjection` | `COMMANDINJECTION` | Web |
| `DDoS-ACK_Fragmentation` | `DDOS-ACK_FRAGMENTATION` | DDoS |
| `DDoS-HTTP_Flood` | `DDOS-HTTP_FLOOD` | DDoS |
| `DDoS-ICMP_Flood` | `DDOS-ICMP_FLOOD` | DDoS |
| `DDoS-ICMP_Fragmentation` | `DDOS-ICMP_FRAGMENTATION` | DDoS |
| `DDoS-PSHACK_FLOOD` | `DDOS-PSHACK_FLOOD` | DDoS |
| `DDoS-RSTFINFLOOD` | `DDOS-RSTFINFLOOD` | DDoS |
| `DDoS-SlowLoris` | `DDOS-SLOWLORIS` | DDoS |
| `DDoS-SynonymousIP_Flood` | `DDOS-SYNONYMOUSIP_FLOOD` | DDoS |
| `DDoS-SYN_Flood` | `DDOS-SYN_FLOOD` | DDoS |
| `DDoS-TCP_Flood` | `DDOS-TCP_FLOOD` | DDoS |
| `DDoS-UDP_Flood` | `DDOS-UDP_FLOOD` | DDoS |
| `DDoS-UDP_Fragmentation` | `DDOS-UDP_FRAGMENTATION` | DDoS |
| `DictionaryBruteForce` | `DICTIONARYBRUTEFORCE` | BruteForce |
| `DNS_Spoofing` | `DNS_SPOOFING` | Spoofing |
| `DoS-HTTP_Flood` | `DOS-HTTP_FLOOD` | DoS |
| `DoS-SYN_Flood` | `DOS-SYN_FLOOD` | DoS |
| `DoS-TCP_Flood` | `DOS-TCP_FLOOD` | DoS |
| `DoS-UDP_Flood` | `DOS-UDP_FLOOD` | DoS |
| `Mirai-greeth_flood` | `MIRAI-GREETH_FLOOD` | Mirai |
| `Mirai-greip_flood` | `MIRAI-GREIP_FLOOD` | Mirai |
| `Mirai-udpplain` | `MIRAI-UDPPLAIN` | Mirai |
| `MITM-ArpSpoofing` | `MITM-ARPSPOOFING` | Spoofing |
| `Recon-HostDiscovery` | `RECON-HOSTDISCOVERY` | Recon |
| `Recon-OSScan` | `RECON-OSSCAN` | Recon |
| `Recon-PingSweep` | `RECON-PINGSWEEP` | Recon |
| `Recon-PortScan` | `RECON-PORTSCAN` | Recon |
| `SqlInjection` | `SQLINJECTION` | Web |
| `Uploading_Attack` | `UPLOADING_ATTACK` | Web |
| `VulnerabilityScan` | `VULNERABILITYSCAN` | Recon |
| `XSS` | `XSS` | Web |

- `Benign_Final → BENIGN` is the one non-obvious rename (the CIC convention
  drops the `_Final` suffix; the CSV subdirectory keeps it).
- All 34 folders resolve into `CATEGORY_MAP` — an unknown-folder guard in
  `resolve_shards()` raises immediately if that ever stops being true.

---

## 4. Feature Schema

The 39 features come from `src/preprocessing/feature_groups.py::FEATURE_NAMES`
— the single source of truth reused by every downstream module
(preprocessing, VAE, attacks, validator, evaluation). Order matters because
the parquet writer records this order in metadata and downstream code
indexes by position.

```
 1. Header_Length      2. Protocol Type      3. Time_To_Live
 4. Rate               5. fin_flag_number    6. syn_flag_number
 7. rst_flag_number    8. psh_flag_number    9. ack_flag_number
10. ece_flag_number   11. cwr_flag_number   12. ack_count
13. syn_count         14. fin_count         15. rst_count
16. HTTP              17. HTTPS             18. DNS
19. Telnet            20. SMTP              21. SSH
22. IRC               23. TCP               24. UDP
25. DHCP              26. ARP               27. ICMP
28. IGMP              29. IPv               30. LLC
31. Tot sum           32. Min               33. Max
34. AVG               35. Std               36. Tot size
37. IAT               38. Number            39. Variance
```

### 4.1 Where the 39-count comes from — the vendor's ship, not a codebase choice

The 39-column shape is **not** a modelling decision made in this codebase.
CIC's official CSV distribution of CICIoT2023 ships exactly these 39
features per flow; the vendor ran CICFlowMeter, then dropped several
columns before publishing the CSV mirror. Every shard on disk has this
exact header — spot-checked on shards from `Benign_Final`, `DDoS-ICMP_Flood`,
`Mirai-udpplain`, `XSS`, and `DictionaryBruteForce`.

The name **"Modified Schema A"** (used by `feature_groups.py`,
`P2_THESIS_REPORT.md` §3.1.1, and older narrative docs) is a
paper-derived label for this specific 39-column subset. From the
annotation in the archived `src/preprocessing/old/feature_groups.py`:

- **Has vs bare Schema A:** the flag/count features `ack_count`,
  `syn_count`, `fin_count`, `rst_count`, `Number`.
- **Missing vs full Schema A** (CICFlowMeter's default output):
  `flow_duration`, `Duration`, `Srate`, `Drate`, `urg_count`,
  `Magnitude`, `Radius`, `Covariance`, `Weight`.

### 4.2 Consequences of the omissions

The missing features cannot be recovered from the shipped CSVs alone —
they require re-running CICFlowMeter on the original PCAP archives
(hundreds of GB), and CIC's exact CICFlowMeter version/flags would need
matching to avoid inconsistent values between the vendor's 39 columns and
regenerated columns.

Impact on the thesis:

| Missing feature | IDS signal | Thesis impact |
|---|---|---|
| `flow_duration` / `Duration` | attacks are often much shorter than benign flows; losing this hurts classifier accuracy | negligible — the 49-rule validator is defined over the 39-col schema, and the `ASR_raw` vs `ASR_valid` gap is a property of the validator, not of feature cardinality |
| `Srate` / `Drate` (source-/destination-side rates) | asymmetric rate is a strong DoS indicator; `Rate` in this schema is the aggregate | mild — small classifier accuracy hit, no argument change |
| `urg_count` | rare in modern traffic; no significant attack family uses URG | none |
| `Magnitude` / `Radius` | algebraic composites of `Min`/`Max`/`AVG`/`Std`/`Variance` | none — VAE will learn any equivalent internal representation |
| `Covariance` / `Weight` | rarely-used auxiliary features | none |

### 4.3 Defense-safe methodology wording

> The CIC-shipped CICIoT2023 CSV distribution contains 39 features per
> flow, corresponding to a subset of the 46-feature CICFlowMeter output
> described in Neto et al. 2023. The 49-rule domain validator is defined
> over the 39-column schema, and the ASR_raw vs ASR_valid gap that is the
> thesis's central metric is a property of the validator's constraint
> set, not of feature cardinality.

### 4.4 Value types

Every value in the raw CSVs is numeric. Booleans/protocol indicators are
already 0/1 floats — not integers — so the `float32` cast at the
labelling stage is not lossy.

---

## 5. Cleaning Policy at the Labelling Stage

The only cleaning applied here is the minimum required to prevent parquet
nulls and float NaN propagation downstream:

```python
chunk = chunk.replace([np.inf, -np.inf], np.nan).dropna()
chunk = chunk.astype({feature: "float32" for feature in FEATURE_NAMES})
```

Everything else is deferred (see next table). Rationale:

- The 99.99th-percentile clip in `pipeline.py` is computed on the whole
  loaded dataset in one pass. Applying a preliminary clip here with a
  different threshold would either weaken it (double-clipping compresses
  tails) or contradict it. Zero clipping here keeps clip logic in exactly
  one place.
- Rounding integer/binary features at this stage would silently repair a
  potentially-invalid raw sample before the domain validator has ever seen
  the pristine version. The clean-data validity report loses meaning if
  cleaning is done before validation.
- Downstream code (`pipeline.py`, EDA scripts, VAE dataset builder) always
  runs its own `dropna()` — this NaN drop here is defensive, not
  authoritative.

Deferred operations:

| Operation | Handled by |
|---|---|
| 99.99th-percentile clip on non-negative features | `src/preprocessing/pipeline.py` §4.1 |
| Rounding integer/binary features to their canonical shape | `src/preprocessing/pipeline.py` §4.1 |
| Domain validity check (49 rules, G1–G8) | `src/attack/validator.py` invoked by `pipeline.py` §4.1b |
| Group-disjoint train/val/test split on `source_csv_filename` | `pipeline.py` §4.3 (currently row-level stratified; upgrade to `StratifiedGroupKFold` per `downsampling_strategy.md` §4.1 is a pending task) |
| `RobustScaler` fit on train only | `pipeline.py` §4.4 |
| Intra-class cluster-based undersampling | `src/preprocessing/sampler.py::cluster_proportional_floor_sample` invoked by `build_ciciot2023_dataset.py` |
| Perturbation-mask derivation | `pipeline.py` §4.7 |

### NaN/inf actually dropped

**1,040 rows out of 46,776,700 (0.002%)** contained at least one NaN or
±inf value across the 39 features. These were dropped silently; the
manifest records the count under `dropped_nan_inf`.

---

## 6. Stage 1: Labelled Parquet Build

**Script:** `src/preprocessing/build_labeled_dataset.py`

### 6.1 Algorithm

1. `resolve_shards(raw_dir)` — walks every class folder under `raw_dir` and
   resolves label + category. Refuses on unknown folder or on a folder
   whose derived label is not in `CATEGORY_MAP`.
2. `iter_shard_chunks(shard, chunk_size=500_000)` — pandas `read_csv` in
   chunks. Verifies each chunk has the full 39-column schema and refuses
   any shard that already carries a `Label` column.
3. `process_shard(...)` — drop NaN/inf, cast to float32, attach `Label`,
   `category`, `source_csv_filename`, `source_folder`, and append to the
   growing parquet via a single `pq.ParquetWriter` reused across shards.
4. `build(...)` — after all shards are processed, assert the row-count
   invariants and write the JSON manifest.

### 6.2 Parquet output schema (43 columns)

| # | Column | Arrow Type | Source |
|---:|---|---|---|
| 1..39 | 39 features | `float` (float32) | CSV row values, cast |
| 40 | `Label` | `dictionary<string, int8>` | folder → `FOLDER_TO_LABEL` |
| 41 | `category` | `dictionary<string, int8>` | Label → `CATEGORY_MAP` |
| 42 | `source_csv_filename` | `large_string` | shard path relative to raw root |
| 43 | `source_folder` | `large_string` | shard's parent folder name |

- `Label` and `category` are pandas `Categorical` on the pandas side, so
  pyarrow stores them dict-encoded with `int8` indices (34 and 8 unique
  values respectively). Storage cost per row: one byte, not one string.
- Column order is asserted before every write
  (`process_shard()`'s `RuntimeError` on drift). A future rename of a
  metadata column cannot silently swap positions.

### 6.3 Row-group layout

- **309 row groups** — one per source CSV shard. Every shard fits in one
  `chunk_size=500_000` window (largest shard is
  `VulnerabilityScan.pcap.csv` at 373,344 rows), so the writer flushes
  exactly one row group per shard.
- This makes shard-holdout splitting (§11) trivially efficient — filter
  by `source_csv_filename` and pyarrow prunes at the row-group level.
- Row-group sizes range 1,252 (`Uploading_Attack.pcap.csv`) to 373,344
  (`VulnerabilityScan.pcap.csv`).

---

## 7. Why the Parquet Is ≈ 594 MB, Not 9–11 GB

Common confusion: the raw CSV footprint is 8.4 GB, so a labelled version
should be at least the same size, right? No — the parquet is a **typed
columnar binary format**, and three compression mechanisms stack.

### 7.1 CSV text → float32 binary (before any encoding)

CSV writes `111.800003,0.0,17.0,` — ~20 bytes for 3 numbers. Float32
stores the same 3 values as 12 bytes. This alone is a ~2× shrink and
accounts for the CSV → 10 GB uncompressed columnar transition.

Uncompressed float32 for the 39 feature columns:

```
46,775,660 rows × 39 cols × 4 B  =  7.30 GB
```

Plus metadata columns (`Label` + `category` dict-encoded to `int8`,
`source_csv_filename` ≈ 40 B/row, `source_folder` ≈ 20 B/row):

```
46,775,660 × (1 + 1 + 40 + 20) B  ≈  2.8 GB
```

**Uncompressed columnar total ≈ 10 GB** — right in the 9–11 GB expectation
range.

### 7.2 Parquet built-in encodings before ZSTD

Parquet's "uncompressed size" as reported by the file metadata is already
*after* dictionary/RLE/delta encoding. Per-column footprint over all 309
row groups (from `pq.ParquetFile.metadata.row_group(rg).column(c)`):

| Column | Uncompressed | Compressed | Ratio |
|---|---:|---:|---:|
| `IAT` | 124.57 MB | 118.06 MB | 1.1× |
| `Rate` | 124.21 MB | 119.66 MB | 1.0× |
| `Variance` | 65.30 MB | 37.92 MB | 1.7× |
| `Std` | 65.29 MB | 37.68 MB | 1.7× |
| `Tot sum` | 52.46 MB | 27.54 MB | 1.9× |
| `AVG` | 52.12 MB | 27.90 MB | 1.9× |
| `Tot size` | 52.12 MB | 27.90 MB | 1.9× |
| `Time_To_Live` | 49.07 MB | 23.00 MB | 2.1× |
| … (continuous features, similar) | | | |
| `Protocol Type` | 527 KB | 411 KB | 1.3× |
| `SSH` / `IRC` / `SMTP` / `Telnet` / `IGMP` / `ece_flag_number` / `cwr_flag_number` | ~300 KB each | ~250 KB | 1.1× |
| `source_csv_filename` | 329 KB | 351 KB | 0.9× |
| `Label` | 188 KB | 213 KB | 0.9× |
| `category` | 129 KB | 154 KB | 0.8× |
| `source_folder` | 189 KB | 214 KB | 0.9× |
| **TOTAL** | **923.52 MB** | **565.48 MB** | **1.6×** |

Look at the low-cardinality columns: `Label` uses **188 KB for
46.7M rows** — 0.004 B/row. Dict-encoded to `int8` + RLE means each of
the 309 row groups just stores "value X repeated N times". Same for
`source_csv_filename` (329 KB): each row group has exactly one value.

`Protocol Type` takes only 527 KB across 46.7M rows because it's one of 6
distinct integers per row group. The 15 binary indicator columns are
almost incompressible-past-a-point because they're mostly `0.0` or `0.1`.

### 7.3 ZSTD on top

Continuous-value columns like `IAT` and `Rate` are already almost
incompressible (unique float32 values every row) so they dominate the
final size — together they take 238 MB, ≈ 40% of the whole file.

### 7.4 Compression summary

$$
\underbrace{8.4\,\text{GB}}_{\text{CSV text}} \;\to\;
\underbrace{\approx 10\,\text{GB}}_{\text{true uncompressed columnar}} \;\to\;
\underbrace{923.5\,\text{MB}}_{\text{parquet built-in encodings}} \;\to\;
\underbrace{565.5\,\text{MB}}_{\text{column chunks + ZSTD}} \;\to\;
\underbrace{594\,\text{MB}}_{\text{on disk (+metadata/footer)}}
$$

$$
\text{Overall CSV} \to \text{Parquet} \approx 14\times
$$

That's textbook for typed columnar formats on NIDS-style tabular data
with lots of low-cardinality flag columns.

---

## 8. Stage 2: Labelled CSV Export

**Script:** `src/preprocessing/old/export_labeled_csv.py` (archived under `old/`).

Streams the labelled parquet row-group by row-group and appends every row
to a single CSV at `data/raw/ciciot2023_full/ciciot2023_base.csv`. The filename
matches the historical convention documented in `codex.md` and
`P2_THESIS_REPORT.md` §3.1.1.

### 8.1 Column set modes

| Mode | Cols | Contents | Size |
|---|---:|---|---:|
| `historical` (default) | 40 | 39 features + `Label` | **7.0 GB** |
| `extended` | 43 | + `category`, `source_csv_filename`, `source_folder` | ~11 GB |

Default is `historical` because every pre-existing script that read
`ciciot2023_base.csv` expects the classic 40-column shape. The extra
metadata is opt-in via `--columns extended`.

### 8.2 Float precision

`float_format='%.10g'` — 10 significant digits.

- Float32 has ~7.2 decimal digits of precision, so 10 digits guarantee
  exact round-trip (`float32(x) == float32(read_csv('%.10g' % float32(x)))`)
  in every case.
- Matches the ~9-digit noise range visible in the raw shards (e.g.,
  `"473.600006"` — that's float32 imprecision showing through).
- Skipping this and letting pandas call `str(x)` inflates size by ~35%
  for zero information gain, because pandas 3.0's default float
  repr uses more digits than float32 can distinguish.

Effect: `210.500000` in the source shards becomes `210.5` in the export,
`473.600006` stays `473.600006`, and zero-columns (`0.000000000`) collapse
to `0`. Aggregate saving is ≈ 15% versus the source CSV formatting.

### 8.3 Streaming design

- Peak RAM = one row group ≤ 373,344 rows ≈ 63 MB.
- Truncates the output file up-front (`write_text("")`) so a crash mid-run
  never leaves a stale suffix from a prior write.
- Per-row-group progress: `[write] rg N/309 rows=… file=… rate=… eta=…`.
- Row-count invariant asserted at the end: raises
  `RuntimeError("Row-count drift: wrote X, expected Y")` if anything
  went missing.

### 8.4 Refusal to overwrite

Both build stages refuse to overwrite an existing output unless `--force`
is passed. Full-dataset writes are expensive (~4 min parquet, ~17 min CSV
on this workstation) and silently overwriting a prior artifact is not
worth doing by accident.

---

## 9. Attack Structure Preservation

This section addresses the thesis-critical question: **does the parquet
preserve everything that per-category VAE training, IDSR/Mahalanobis
class-manifold analysis, and shard-holdout splitting rely on?** Answer:
yes.

### 9.1 What "attack structure" means concretely

Five aspects the thesis downstream pipeline depends on:

| Aspect | What could break it |
|---|---|
| **Within-shard row order** | shuffling rows during read → destroys wall-clock temporal structure inside the capture segment |
| **Shard boundaries** | interleaving rows from different shards → destroys the tcpdump-C-segment grouping needed for temporal holdout |
| **Shard identity** | dropping the source filename → can't recover natsort order or do shard-level holdout |
| **Per-class completeness** | filtering / sub-sampling / aggregating → shrinks or narrows the class manifold that per-class VAEs learn |
| **Sub-mode representation** | any of the above → the SYN/UDP/ICMP sub-attacks inside DDoS get statistically erased (`downsampling_strategy.md` §1's panel objection) |

### 9.2 What the build actually did — traced through the code

`build_labeled_ciciot2023_dataset.py` acts as a **pure concatenator with
labels appended**. No shuffles, no aggregates, no filters (beyond NaN-drop).

1. `resolve_shards()` → `sorted(iterdir())` for folders,
   `sorted(rglob("*.csv"))` for shards. Each shard resolved to its label +
   category via `FOLDER_TO_LABEL` + `CATEGORY_MAP`.
2. `iter_shard_chunks(shard, chunk_size=500_000)` → `pd.read_csv(chunksize=…)`.
   Pandas reads chunks **sequentially**, preserving row order within the CSV.
3. Cleaning is `chunk.replace([inf, -inf], nan).dropna()` — `dropna`
   preserves relative order for the surviving rows (only introduces at
   most 1,040 gaps in total across the whole dataset, 0.002%).
4. Cast to float32 — matches the source CSVs' native precision (values
   like `473.600006` are float32-rounded text already).
5. Attach `Label`, `category`, `source_csv_filename`, `source_folder` —
   never reorder.
6. `pa.Table.from_pandas(chunk, preserve_index=False)` →
   `writer.write_table(table)`. PyArrow writes rows in the given order.
   Since every shard fits in one `chunk_size` window, each shard produces
   exactly one row group.

### 9.3 What is verifiably preserved

| Structure | Preserved? | Evidence |
|---|---|---|
| Row order within each shard | **byte-identical** modulo NaN drops | see live check below |
| Shard identity | **exact** | `source_csv_filename` on every row |
| Shard → row-group mapping | **1:1** | 309 row groups = 309 CSV shards |
| Rows from shard A never mixed with shard B | **guaranteed** | writer flushes one shard's data before opening the next |
| All rows kept per class | **46,775,660 / 46,776,700** | 1,040 NaN dropped (0.002%), everything else intact |
| Sub-modes inside each class | **untouched** | no filtering / aggregation / summarizing anywhere in the code path |

### 9.4 Byte-identical proof (Backdoor_Malware shard)

Run this against the current artifacts to reproduce:

```python
import pandas as pd
import pyarrow.parquet as pq
from src.preprocessing.old.feature_groups import FEATURE_NAMES

raw = pd.read_csv("data/raw/CICIoT2023_CSV_DOWNLOADED/Backdoor_Malware/Backdoor_Malware.pcap.csv")
raw = raw.replace([float("inf"), float("-inf")], float("nan")).dropna()
raw = raw.astype({f: "float32" for f in FEATURE_NAMES}).reset_index(drop=True)

pf = pq.ParquetFile("data/processed/ciciot2023_labeled_full.parquet")
rg0 = pf.read_row_group(0).to_pandas()[FEATURE_NAMES].reset_index(drop=True)

assert len(raw) == len(rg0) == 3_218
for col in FEATURE_NAMES:
    assert (raw[col].values == rg0[col].values).all(), col
print("byte-identical across 3,218 rows × 39 features")
```

Verified live during the build. Zero drift across 3,218 × 39 = 125,502
feature values; first and last rows compared verbatim.

### 9.5 Cross-shard order is temporally meaningful (correction to an earlier framing)

`sorted(rglob("*.csv"))` yields lexicographic order — for a multi-shard
folder, that produces:

```
DDoS-ICMP_Flood.pcap.csv       (base — segment 0)
DDoS-ICMP_Flood1.pcap.csv
DDoS-ICMP_Flood10.pcap.csv     ← "10" sorts before "2"
DDoS-ICMP_Flood11.pcap.csv
…
DDoS-ICMP_Flood2.pcap.csv      ← natural-numeric order would put this earlier
DDoS-ICMP_Flood20.pcap.csv
…
```

**The numeric suffix is a `tcpdump -C` sequence number encoding wall-clock
order within one continuous capture** — the CIC pipeline is documented in
§11 below, and the temporal direction is verified: **when** a folder has
a distinct sub-budget remainder (a shard significantly smaller than the
rest), it is the highest-numbered one — DDoS-PSHACK_FLOOD `15` at 78,907
rows, Mirai-udpplain `24` at 15,088, DDoS-UDP_Flood `20` at 79,330. This
is only possible if suffix N + 1 was still being filled when the capture
ended. Folders whose capture happened to end at a byte-budget boundary
(Mirai-greeth_flood, DDoS-ICMP_Flood) have no distinct remainder and
every shard sits inside the tight cluster; the direction argument still
holds transitively from the folders that do have a remainder.

Lexicographic sorting therefore **does** scramble cross-shard temporal
order: segment 10 lands in the parquet before segment 2 for every
multi-shard attack folder. An earlier version of this document dismissed
this as "not destroying structure" on the assumption that each shard was
an independent capture. That assumption was wrong.

What is still preserved (§9.4 stands):

- Within-shard row order — byte-identical to the source CSV.
- Shard identity — `source_csv_filename` on every row.
- 1:1 shard → row-group mapping.

What is scrambled:

- Concatenation order across shards of the same attack. If a downstream
  reader walks parquet row groups `[0, 1, 2, …, 308]` expecting wall-clock
  progression, they will see segments in the order 0, 1, 10, 11, …, 19,
  2, 20, …, 29, 3, 4, … within each multi-shard class.

Recovery is a one-liner because `source_csv_filename` was preserved
verbatim:

```python
import re
df.sort_values(
    "source_csv_filename",
    key=lambda col: col.map(
        lambda p: [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", p)]
    ),
)
```

Or install `natsort` and use `natsort.natsort_keygen()`.

**Consequences for the pipeline:**

- Per-category VAE training, KMeans-based undersampling, and
  cluster-vs-KDE fidelity checks operate over the feature-space class
  manifold. They are order-invariant, so the scramble is harmless there.
- Any **temporal-holdout split** on `source_csv_filename` (see §11) MUST
  natsort the file names before assigning shards to
  `{train, val, test}`; otherwise the split walks wall-clock time in
  scrambled order and the holdout is meaningless.
- If a temporal-analysis stage is planned in the parquet itself
  (e.g., streaming reads that assume in-order row groups), swap
  `resolve_shards()`'s `sorted(...)` for a natsort-keyed sort in
  `build_labeled_ciciot2023_dataset.py` and rerun the builder. One-line
  change; row-group content is unaffected.

### 9.6 Panel-safe defense sentence

> The labelled parquet is a lossless concatenation of the 309 vendor
> shards with class labels appended: rows within each shard are preserved
> in their original order byte-for-byte (verified per-position on the
> Backdoor_Malware shard), and every shard occupies exactly one row group
> whose `source_csv_filename` identifies it. No downsampling, filtering,
> or reordering is applied at the labelling stage — those all live in
> later, explicitly-scoped stages (`pipeline.py` for cleaning + split,
> `sampler.py` for intra-class cluster-based undersampling). The only
> rows removed are 1,040 with NaN or ±inf values (0.002%), which would
> corrupt any float32 downstream and are always dropped by convention.

---

## 10. Final Row Counts

### 10.1 By 8-category (per-class VAE partitioning)

| # | Category | Rows | % | Sub-labels |
|---:|---|---:|---:|---:|
| 1 | DDoS | 33,983,922 | 72.65 | 12 |
| 2 | DoS | 7,844,894 | 16.77 | 4 |
| 3 | Mirai | 2,633,870 | 5.63 | 3 |
| 4 | Benign | 1,098,126 | 2.35 | 1 |
| 5 | Recon | 690,521 | 1.48 | 5 |
| 6 | Spoofing | 486,435 | 1.04 | 2 |
| 7 | Web | 24,828 | 0.05 | 6 |
| 8 | BruteForce | 13,064 | 0.03 | 1 |
| | **Total** | **46,775,660** | **100.00** | **34** |

- **Max/min imbalance ratio (categories):** 33,983,922 / 13,064 ≈ **2,601 ×**.
- **Web and BruteForce are the "rare" categories** —
  `downsampling_strategy.md` §3.3 mandates they be kept whole (no
  clustering, no capping) at every downstream stage.

### 10.2 By 34-class label

Ordered by row count within each category:

**DDoS (12 labels)**

| Label | Rows | % of total |
|---|---:|---:|
| DDOS-ICMP_FLOOD | 7,200,436 | 15.394 |
| DDOS-UDP_FLOOD | 5,412,169 | 11.570 |
| DDOS-TCP_FLOOD | 4,497,546 | 9.615 |
| DDOS-PSHACK_FLOOD | 4,094,727 | 8.754 |
| DDOS-SYN_FLOOD | 4,059,097 | 8.678 |
| DDOS-RSTFINFLOOD | 4,045,248 | 8.648 |
| DDOS-SYNONYMOUSIP_FLOOD | 3,598,100 | 7.692 |
| DDOS-ICMP_FRAGMENTATION | 452,444 | 0.967 |
| DDOS-UDP_FRAGMENTATION | 286,895 | 0.613 |
| DDOS-ACK_FRAGMENTATION | 285,045 | 0.609 |
| DDOS-HTTP_FLOOD | 28,790 | 0.062 |
| DDOS-SLOWLORIS | 23,425 | 0.050 |

**DoS (4 labels)**

| Label | Rows | % of total |
|---|---:|---:|
| DOS-UDP_FLOOD | 3,072,883 | 6.569 |
| DOS-TCP_FLOOD | 2,671,363 | 5.711 |
| DOS-SYN_FLOOD | 2,028,791 | 4.337 |
| DOS-HTTP_FLOOD | 71,857 | 0.154 |

**Mirai (3 labels)**

| Label | Rows | % of total |
|---|---:|---:|
| MIRAI-GREETH_FLOOD | 991,774 | 2.120 |
| MIRAI-UDPPLAIN | 890,507 | 1.904 |
| MIRAI-GREIP_FLOOD | 751,589 | 1.607 |

**Benign (1 label)**

| Label | Rows | % of total |
|---|---:|---:|
| BENIGN | 1,098,126 | 2.348 |

**Recon (5 labels)**

| Label | Rows | % of total |
|---|---:|---:|
| VULNERABILITYSCAN | 373,344 | 0.798 |
| RECON-HOSTDISCOVERY | 134,377 | 0.287 |
| RECON-OSSCAN | 98,255 | 0.210 |
| RECON-PORTSCAN | 82,283 | 0.176 |
| RECON-PINGSWEEP | 2,262 | 0.005 |

**Spoofing (2 labels)**

| Label | Rows | % of total |
|---|---:|---:|
| MITM-ARPSPOOFING | 307,542 | 0.657 |
| DNS_SPOOFING | 178,893 | 0.382 |

**Web (6 labels)**

| Label | Rows | % of total |
|---|---:|---:|
| BROWSERHIJACKING | 5,859 | 0.013 |
| COMMANDINJECTION | 5,409 | 0.012 |
| SQLINJECTION | 5,244 | 0.011 |
| XSS | 3,846 | 0.008 |
| BACKDOOR_MALWARE | 3,218 | 0.007 |
| UPLOADING_ATTACK | 1,252 | 0.003 |

**BruteForce (1 label)**

| Label | Rows | % of total |
|---|---:|---:|
| DICTIONARYBRUTEFORCE | 13,064 | 0.028 |

- **Max/min imbalance ratio (34-class):** DDOS-ICMP_FLOOD /
  UPLOADING_ATTACK ≈ **5,751 ×**. This matches the historical figure
  recorded in `logs/eda.log` for the older CSV mirror (5,763 ×),
  confirming the sampled dataset is faithful to CICIoT2023's native
  distribution and not an accidentally-truncated snapshot.

### 10.3 Files per folder

| Folder | Rows | Files | Note |
|---|---:|---:|---|
| Backdoor_Malware | 3,218 | 1 | 🚨 no shard-level holdout possible |
| Benign_Final | 1,098,126 | 4 | ok, splittable |
| BrowserHijacking | 5,859 | 1 | 🚨 |
| CommandInjection | 5,409 | 1 | 🚨 |
| DDoS-ACK_Fragmentation | 285,045 | 13 | ok |
| DDoS-HTTP_Flood | 28,790 | 1 | 🚨 |
| DDoS-ICMP_Flood | 7,200,436 | 27 | ok |
| DDoS-ICMP_Fragmentation | 452,444 | 20 | ok |
| DDoS-PSHACK_FLOOD | 4,094,727 | 16 | ok |
| DDoS-RSTFINFLOOD | 4,045,248 | 16 | ok |
| DDoS-SYN_Flood | 4,059,097 | 16 | ok |
| DDoS-SlowLoris | 23,425 | 1 | 🚨 |
| DDoS-SynonymousIP_Flood | 3,598,100 | 14 | ok |
| DDoS-TCP_Flood | 4,497,546 | 18 | ok |
| DDoS-UDP_Flood | 5,412,169 | 21 | ok |
| DDoS-UDP_Fragmentation | 286,895 | 13 | ok |
| DNS_Spoofing | 178,893 | 1 | 🚨 |
| DictionaryBruteForce | 13,064 | 1 | 🚨 (also rare) |
| DoS-HTTP_Flood | 71,857 | 2 | tight |
| DoS-SYN_Flood | 2,028,791 | 8 | ok |
| DoS-TCP_Flood | 2,671,363 | 11 | ok |
| DoS-UDP_Flood | 3,072,883 | 17 | ok |
| MITM-ArpSpoofing | 307,542 | 2 | tight |
| Mirai-greeth_flood | 991,774 | 29 | ok |
| Mirai-greip_flood | 751,589 | 22 | ok |
| Mirai-udpplain | 890,507 | 25 | ok |
| Recon-HostDiscovery | 134,377 | 1 | 🚨 |
| Recon-OSScan | 98,255 | 1 | 🚨 |
| Recon-PingSweep | 2,262 | 1 | 🚨 |
| Recon-PortScan | 82,283 | 1 | 🚨 |
| SqlInjection | 5,244 | 1 | 🚨 |
| Uploading_Attack | 1,252 | 1 | 🚨 |
| VulnerabilityScan | 373,344 | 1 | 🚨 |
| XSS | 3,846 | 1 | 🚨 |

---

## 11. Splitting Implication — Temporal Holdout, Not Group-Disjoint

### 11.1 How CIC produced the shards

The CIC-documented preprocessing pipeline for CICIoT2023 is:

```
per-attack PCAPs  →  mergecap  →  one merged capture per attack
                                                    │
                                                    ▼
                                            tcpdump  -C  (split by BYTE budget)
                                                    │
                                                    ▼
                                            {base, 1, 2, …, N}.pcap segments
                                                    │
                                                    ▼
                                            DPKT flow-feature extraction
                                                    │
                                                    ▼
                                            one CSV per pcap segment
```

**Consequence:** each `.pcap.csv` shard is a contiguous temporal segment
of ONE merged capture, not an independent session. Per-file row counts
confirm this directly:

| Folder | Shards | Tight cluster | Spread | Remainder |
|---|---:|---|---:|---|
| DDoS-PSHACK_FLOOD | 16 | 15/16 in 265,547–269,060 rows | 1.3% | `15` at 78,907 rows |
| Mirai-udpplain | 25 | 24/25 in 36,211–36,982 rows | 2.1% | `24` at 15,088 rows |
| DDoS-UDP_Flood | 21 | 20/21 in 261,897–268,391 rows | 2.4% | `20` at 79,330 rows |
| Mirai-greeth_flood | 29 | 29/29 in 33,593–34,752 rows | 3.3% | (none — clean roll-over) |
| DDoS-ICMP_Flood | 27 | 27/27 in 249,337–269,201 rows | 7.4% | (none) |

The two folders converge on **different per-file row counts** because the
`tcpdump -C` budget is in bytes, not rows: DDoS floods sit at ~267K
rows/shard while Mirai variants sit at ~35K rows/shard, ≈ 7.6× different
because Mirai flow records are longer (more per-flow state) so fewer fit
in the same byte budget.

### 11.2 The one anomaly — `Benign_Final`

`Benign_Final` contains 4 shards
(`BenignTraffic{,1,2,3}.pcap.csv`) with row counts 129,824 / 295,565 /
310,395 / 362,342. That spread does not match a single-capture-with-byte-
budget signature; it looks like 4 **independent** benign captures
concatenated into one folder. Benign is therefore the only class where
shard identity really is capture-level identity — everywhere else, shard
identity is temporal-segment identity within one capture.

### 11.3 What "splitting on shard identity" actually is

Because attack classes are single-capture-per-attack (single-run design),
group-disjoint splitting at the capture level is **unattainable**: there
is only one group per attack. What we can do is **temporal holdout** on
the numeric suffix — for a class with shards `[0, 1, …, N]`, assign
earlier segments to train and later segments to val/test:

- Same source distribution: it's all the same capture, so class-mix is
  identical across segments.
- Real temporal separation: the val/test rows come from packets that
  arrived strictly later in wall-clock time than the training rows.
- **NOT** capture-level group-disjointness: an attacker who could observe
  the training capture would trivially predict the test capture, because
  it's a continuation of the same session.

This is a weaker leakage guarantee than the classical "different-capture"
holdout the `downsampling_strategy.md` §4.1 rule was written for, but it
is the strongest available given the dataset's single-run-per-attack
design. Disclose the framing explicitly in the thesis methodology.

### 11.4 The 15 single-shard classes are still stranded

Even under a temporal-holdout framing, the 15 classes whose entire
capture is a single file (see 🚨 rows in §10.3) admit no shard-level
split at all. For those, the fallback is unavoidable:

- **Row-level stratified split** on `Label` within the single file. This
  introduces mild within-capture leakage (adjacent rows from the same
  attack session appear on both sides of the split) but preserves per-
  class representation across `{train, val, test}`. Disclosed as a
  known limitation.

### 11.5 Recommended split strategy — updated

Bake into the split stage:

- **Temporal shard-holdout** (natsort-ordered `source_csv_filename`
  → cumulative-row bucket) for the 19 multi-shard classes. Earlier
  segments to train, latest to val/test.
- **Row-level stratified** on `Label` for the 15 single-shard classes.
- **Group-disjoint at capture level** for `Benign_Final` only, because
  its 4 shards are 4 independent captures — the leakage argument that
  motivates §4.1 of `downsampling_strategy.md` genuinely applies here.
- **Disclose the hybrid** in the methodology chapter: attack classes get
  temporal holdout on segments of one capture; Benign gets capture-level
  group-disjoint; single-shard rare classes get row-level stratified.

`source_csv_filename` is preserved on every parquet row precisely so
every rule above can be applied downstream with full information.

### 11.6 Panel-safe defense sentence

> The CIC pipeline generates each attack class from one continuous
> capture that `tcpdump -C` slices into fixed-byte segments; shard files
> within an attack folder are therefore contiguous temporal chunks of one
> capture, not independent sessions (verified by within-folder row-count
> uniformity of 1.3–3.3% except for a single remainder). Splitting on
> `source_csv_filename` is a temporal holdout within one capture — a
> weaker leakage guarantee than classical group-disjoint holdout across
> independent captures, but it is the strongest available under CIC's
> single-run-per-attack design. `Benign_Final` is the sole exception: its
> four shards are four independent captures, so real capture-level
> group-disjointness applies there. The 15 single-shard rare classes
> receive row-level stratified splits, disclosed as a limitation.

---

## 12. Current On-Disk Layout

```
E:/Shameem/thesis/data/
├── raw/
│   ├── CICIoT2023_CSV_DOWNLOADED/   8.4 GB   — VENDOR SOURCE (untouched)
│   │   ├── Backdoor_Malware/            — 34 class folders, 309 CSVs total
│   │   ├── Benign_Final/                — each shard: 39 features, NO Label column
│   │   ├── … 32 more folders
│   │   └── README_CSV.pdf
│   │
│   └── ciciot2023_full/         7.0 GB   — LABELLED CSV
│       └── ciciot2023_base.csv          — 40 cols (39 features + Label)
│                                        — 46,775,661 lines (1 header + 46,775,660 rows)
│                                        — MD5: 7e487a151c271550ff33200c59336843
│
└── processed/
    ├── ciciot2023_labeled_full.parquet         594 MB — LABELLED PARQUET (source of truth)
    │                                                    43 cols, 46,775,660 rows,
    │                                                    309 row groups (1 per shard), ZSTD
    ├── ciciot2023_labeled_full_manifest.json    24 KB — provenance + per-{folder,label,category,file} counts
    │
    └── old/                                    2.85 GB — STALE PREPROCESSING OUTPUTS (from before today)
        ├── raw_labeled_full.parquet             510 MB  — older labelled parquet, same 43 cols,
        │                                                  same 46,775,660 rows / 309 groups,
        │                                                  just different ZSTD level (84 MB smaller)
        ├── X_{train,val,test}.npy               ≈2.3 GB — scaled feature arrays (train-fit RobustScaler)
        ├── y_{train,val,test}[,_cat,_bin].npy    ≈180 MB — 34-class / 8-class / binary labels
        ├── scaler.pkl, label_encoder.pkl,
        │   category_encoder.pkl                          — sklearn artifacts
        ├── class_names.json, category_names.json         — encoder class order
        ├── class_weights_{34,8,2}.npy                    — saved but not used in loss
        ├── perturbation_mask.npy                         — 39-value mask (0.0 / 0.3 / 1.0)
        ├── raw_file_inventory.csv                        — per-shard inventory (from prior discovery pass)
        ├── split_strategy.json                           — split assignments
        └── run_manifest.json                             — reproducibility for the old run
```

### 12.1 What is stale

`data/processed/old/*` was built by a previous `pipeline.py` run before
the current parquet existed:

- `old/raw_labeled_full.parquet` is functionally identical to the current
  parquet (same 43 cols, same 46,775,660 rows, same 309 row groups) but
  84 MB smaller due to different ZSTD level. Safe to delete after
  confirming the current parquet is used everywhere.
- `old/X_*.npy`, `old/y_*.npy`, `old/scaler.pkl`, encoders, weights, mask,
  `run_manifest.json` — outputs of the previous `pipeline.py` run (dated
  Jul 8 21:24–21:29). Rerunning `pipeline.py` will regenerate a fresh set
  targeting the current parquet; the old outputs should not be trusted
  alongside a re-run.

### 12.2 What is missing

No `data/processed/raw_loaded.parquet` — that used to be the ~4.4M-row
stratified sample referenced by `pipeline.py` and every EDA script. It is
not present in either `processed/` root or `processed/old/`. Before
running `pipeline.py`, either point it at the new full parquet or produce
a sampled `raw_loaded.parquet` from it first — the current `pipeline.py`
assumes a pre-sampled input.

---

## 13. Provenance Chain

```
data/raw/CICIoT2023_CSV_DOWNLOADED/  (vendor shards, 8.4 GB, unlabelled)
    │
    │  src/preprocessing/build_labeled_ciciot2023_dataset.py
    │      • 500k-row chunks per shard
    │      • folder → FOLDER_TO_LABEL → Label (34-class)
    │      •                          → CATEGORY_MAP → category (8-class)
    │      • drop 1,040 NaN/inf rows (0.002%)
    │      • cast features → float32
    │      • one row group per shard
    ▼
data/processed/ciciot2023_labeled_full.parquet    (43 cols, 594 MB, source of truth)
data/processed/ciciot2023_labeled_full_manifest.json
    │
    │  src/preprocessing/old/export_labeled_csv.py   (archived under old/)
    │      • read row-group-by-row-group
    │      • project to 40-col historical shape (features + Label)
    │      • float_format='%.10g' (round-trip safe for float32)
    ▼
data/raw/ciciot2023_full/ciciot2023_base.csv      (40 cols, 7.0 GB)
```

Every derived artifact matches the parquet exactly at row-count level.

---

## 14. Row-Count Invariants Across Artifacts

| Artifact | Rows | Match |
|---|---:|---|
| raw shards (sum minus 309 headers) | 46,776,700 | source |
| current parquet (`num_rows`) | **46,775,660** | −1,040 NaN drop |
| current CSV (`wc -l` − 1 header) | **46,775,660** | ✓ |
| manifest (`kept_rows`) | **46,775,660** | ✓ |
| manifest (`sum(per_label_rows)`) | **46,775,660** | ✓ |
| manifest (`sum(per_category_rows)`) | **46,775,660** | ✓ |
| manifest (`sum(per_folder_rows)`) | **46,775,660** | ✓ |
| manifest (`sum(per_file_rows)`) | **46,775,660** | ✓ |
| old parquet (`data/processed/old/raw_labeled_full.parquet`) | 46,775,660 | ✓ (same schema, same rows, older ZSTD level) |

Any drift raises `RuntimeError("Row accounting drift: ...")` before the
manifest is written.

---

## 15. Manifest Fields

`data/processed/ciciot2023_labeled_full_manifest.json`:

| Key | Type | Purpose |
|---|---|---|
| `raw_dir` | string | absolute path to the raw CSV root used |
| `output_parquet` | string | absolute path of the labelled parquet |
| `dry_run` | bool | true when `--dry-run` was passed (no parquet written) |
| `chunk_size` | int | pandas read chunksize used |
| `limit_per_file` | int \| null | smoke-test row cap per shard, or null for full |
| `n_files` | int | 309 |
| `n_folders` | int | 34 |
| `total_rows_read` | int | 46,776,700 |
| `kept_rows` | int | 46,775,660 |
| `dropped_nan_inf` | int | 1,040 |
| `feature_names` | list[str] | canonical 39-feature order |
| `n_features` | int | 39 |
| `label_column` | string | `"Label"` |
| `expected_columns` | list[str] | features + label, matches `EXPECTED_COLUMNS` |
| `per_folder_rows` | dict[str, int] | rows kept per class folder |
| `per_label_rows` | dict[str, int] | rows kept per 34-class label |
| `per_category_rows` | dict[str, int] | rows kept per 8-category tag |
| `per_file_rows` | dict[str, int] | rows kept per source shard |
| `folder_to_label` | dict[str, str] | the `FOLDER_TO_LABEL` mapping used |

---

## 16. Reproduction

Full parquet build (≈ 4 min end-to-end):

```bash
cd /e/Shameem/thesis
"C:/Users/user6/.local/share/mamba/envs/thesis/python.exe" \
    -m src.preprocessing.build_labeled_ciciot2023_dataset
```

Full CSV export (≈ 17 min end-to-end):

```bash
"C:/Users/user6/.local/share/mamba/envs/thesis/python.exe" \
    -m src.preprocessing.old.export_labeled_csv
```

Smoke variants (100 rows per shard, no parquet written / first 3 row
groups only, ~130 MB smoke CSV):

```bash
# parquet builder — dry-run counts + manifest only
"C:/Users/user6/.local/share/mamba/envs/thesis/python.exe" \
    -m src.preprocessing.build_labeled_ciciot2023_dataset \
    --limit-per-file 100 --dry-run \
    --manifest data/processed/_smoke_manifest.json

# CSV exporter — 3 row groups
"C:/Users/user6/.local/share/mamba/envs/thesis/python.exe" \
    -m src.preprocessing.old.export_labeled_csv \
    --limit-row-groups 3 \
    --output data/raw/ciciot2023_full/_smoke.csv --force
```

### 16.1 CLI reference

`build_labeled_ciciot2023_dataset.py`:

| Flag | Default | Purpose |
|---|---|---|
| `--raw-dir` | `data/raw/CICIoT2023_CSV_DOWNLOADED` | raw CSV root |
| `--output` | `data/processed/ciciot2023_labeled_full.parquet` | output parquet |
| `--manifest` | `data/processed/ciciot2023_labeled_full_manifest.json` | manifest JSON |
| `--chunk-size` | 500_000 | pandas read chunksize |
| `--limit-per-file` | unlimited | per-shard row cap for smoke tests |
| `--dry-run` | off | write only manifest, not parquet |

`export_labeled_csv.py`:

| Flag | Default | Purpose |
|---|---|---|
| `--parquet` | `data/processed/ciciot2023_labeled_full.parquet` | source parquet |
| `--output` | `data/raw/ciciot2023_full/ciciot2023_base.csv` | output CSV |
| `--columns` | `historical` (40 cols) | column set: `historical` or `extended` (43 cols) |
| `--force` | off | overwrite existing output |
| `--limit-row-groups` | unlimited | first N row groups only (smoke) |
| `--progress-every` | 10 | print progress every N row groups |

Determinism: both scripts do no shuffling and read shards in a
deterministic `sorted(rglob("*.csv"))` order; identical raw inputs always
yield an identical parquet (byte-for-byte given the same pyarrow version)
and an identical CSV (byte-for-byte given the same pandas version).

---

## 17. Verification

The build was verified in three passes:

1. **Discovery invariants** (raised at read time in the builder):
   - Every folder present has a `FOLDER_TO_LABEL` entry (34/34).
   - Every derived Label is present in `CATEGORY_MAP` (34/34).
   - Every CSV chunk has the full 39-column header (checked per chunk).
   - No CSV already carries a `Label` column (would refuse silently
     merging two label sources).

2. **Row-count accounting** (asserted at write time in the builder):
   - `sum(per_label) == sum(per_category) == sum(per_folder) == sum(per_file) == kept_rows`.
   - Manifest `kept_rows` matches parquet `metadata.num_rows`.

3. **Post-write inspection**:
   - `pq.ParquetFile` opens the file cleanly.
   - `num_row_groups == 309` (one per shard).
   - `num_rows == 46,775,660`.
   - `num_columns == 43`.
   - Schema shows 39 float features + `Label`/`category` as
     `dictionary<string, int8>` + `source_csv_filename`/`source_folder`
     as `large_string`.
   - Row-group 0 spot-check confirms `Label = BACKDOOR_MALWARE`,
     `category = Web`,
     `source_csv_filename = Backdoor_Malware/Backdoor_Malware.pcap.csv`
     — matches the alphabetically-first shard as expected.
   - `len(per_label_rows) == 34`, `len(per_category_rows) == 8`.
   - **Byte-identical row-order proof** on the Backdoor_Malware shard
     (§9.4) — 3,218 × 39 feature values compared byte-for-byte with the
     source CSV. Zero drift.

CSV export was verified by loading the full `Label` column with pandas and
cross-checking against `per_label_rows` in the manifest — all 34 label
counts match exactly, total = 46,775,660.

---

## 18. Downstream Enablers

With these artifacts in place, downstream stages can:

- Load the full labelled dataset once with `pd.read_parquet(...)` or
  `pyarrow.parquet.ParquetDataset(...)` (row-group-level streaming works too).
- Trust that `Label` is the exact CICIoT2023 34-class label and `category`
  the corresponding 8-way tag — no folder-name reparsing anywhere else in
  the codebase.
- Use `source_csv_filename` as the split group key for
  `StratifiedGroupKFold` (see §11) or, when needed, fall back to row-level
  stratified splits for the 15 single-shard classes.
- Run any per-category filter (e.g., for the 8 per-class β-VAEs) with a
  simple `pf.read_row_group(rg, filters=[("category", "=", "Mirai")])` —
  because `category` is dictionary-encoded, row-group pruning is efficient.

The next stage is the clean step in `pipeline.py` §4.1 (percentile clip,
integer/binary rounding, then the raw-space domain validator over the 49
rules).

---

## 19. Known Environment Drift

- `pyarrow` is not in `environment.yml` but is required by the labelling
  builder and the CSV exporter. It was installed manually during this
  build (`pyarrow==24.0.0`). **Add to `environment.yml` on next commit.**
- `AGENTS.md` cites a `C:\Users\T2530985\.conda\envs\thesis\python.exe`
  path that does not exist on this workstation; the active env here is
  `C:\Users\user6\.local\share\mamba\envs\thesis\python.exe` (miniforge +
  mamba). Behavior is otherwise identical.
- **Folder rename** — the CSV shard root was renamed from `CICIoT2023_CSV/`
  to `CICIoT2023_CSV_DOWNLOADED/`, and the labelled-CSV directory from
  `ciciot2023/` to `ciciot2023_full/`. The currently-active build script,
  this doc, `AGENTS.md`, and `knowledge.md` reference the new names;
  older narrative docs (`guide.md`, `codex.md`, `P2_THESIS_REPORT.md`)
  still reference `data/ciciot2023/ciciot2023_base.csv` or
  `data/raw/CICIoT2023_CSV/` — treat those as historical.
- **Preprocessing archival** — `src/preprocessing/feature_groups.py`,
  `pipeline.py`, `netdiffuser_categorization.py`, `sampler.py`,
  `build_ciciot2023_dataset.py`, the previous `build_labeled_dataset.py`,
  and `export_labeled_csv.py` were all moved to
  `src/preprocessing/old/`. The only currently-active file at
  `src/preprocessing/` is `build_labeled_ciciot2023_dataset.py`
  (the renamed labelling builder), which imports its schema constants
  from `src.preprocessing.old.feature_groups` and inlines the
  `FOLDER_TO_LABEL` mapping to avoid pulling in the rest of the
  archived pipeline.
