# Downsampling Strategy for CICIoT2023 (and ablation datasets)

**Thesis:** Adversarial robustness of DL-based NIDS · validity-aware attacks · MIBVAE on CICIoT2023
**Author:** Rafsan Rahman · BRAC University CSE
**Purpose of this doc:** Single source of truth for *how* the dataset is downsampled, *why* this method (not the alternatives), how the train/val/test split is constructed, what to verify before committing, and what to track. Written to be directly defensible at the panel and reusable for the two ablation datasets.

**Revision note (this version):** Corrects three factual errors in the previous revision, all traceable to a mistaken assumption about how CIC produced the CSV shards. See §1 and the changelog in §12. The changes are not cosmetic — they alter the split protocol and the name of the leakage guarantee the thesis claims.

> **One-line summary:** Use **intra-class clustering-based undersampling (selection variant)** with **cluster-proportional-with-floor** allocation, applied **train-only after a forward-chaining shard-level temporal split**, keeping **real rows** (never synthetic centroids) and keeping **rare classes whole**.

---

## 0. The problem in one paragraph

CICIoT2023 ships **46,775,660 labelled flows** across **309 CSV shards** in **34 folders** (34 subtypes → 8 macro-categories). Imbalance is severe: DDoS holds 72.65% of rows, BruteForce 0.03% — a **2,601×** category ratio, and **5,751×** at the 34-class level (`DDOS-ICMP_FLOOD` vs `UPLOADING_ATTACK`). We must reduce volume so that (a) 8 per-class VAEs train in reasonable time, (b) the attack families × targeted/untargeted × 3 datasets grid stays tractable, and (c) per-sample iterative black-box attacks are feasible. **But** the headline thesis claims are *fidelity* and *on-manifold validity*, so any reduction that distorts a class's internal distribution directly threatens the core result. The P2 panel correctly flagged that **random undersampling preserves volume but not structure**. This doc fixes that, and separately fixes the split protocol that the previous revision got wrong.

---

## 1. What a "shard" actually is (corrects the previous revision)

The previous revision described the raw data as *"169 Spark `part-*.csv` shards"* and treated each shard as an **independent capture session**, which licensed a group-disjoint split on `source_csv_filename`. **Both claims are wrong.**

### 1.1 The vendor's documented pipeline

From CIC's own dataset page for CICIoT2023:

> "We used Mergecap to merge multiple .pcap files, PySpark to handle the data, TCPDump to split the .pcap files in multiple smaller files, and DPKT to extract features."

Read the order: **merge, then split.** Per-attack PCAPs are merged into one continuous stream, then `tcpdump` chops that stream into smaller files — by **file size**, not by session. Flow features are extracted afterward. The 169 figure refers to CIC's PCAP trace count, not the CSV shard count; the CSV distribution contains **309** shards.

### 1.2 Direct empirical confirmation

If shards were separate attack runs, rows-per-shard would vary with the natural variance of an operator starting and stopping a script. If they are byte-size splits, rows-per-shard will be near-constant within a folder, with a short remainder. Measured from `per_file_rows` in the build manifest:

| Folder | Shards | Rows/shard (full files) | Spread | Remainder |
|---|---:|---|---:|---:|
| `DDoS-PSHACK_FLOOD` | 16 | 265,547 – 269,060 | **1.3%** | 78,907 |
| `DDoS-SYN_Flood` | 16 | 260,187 – 268,420 | **3.1%** | 137,253 |
| `Mirai-udpplain` | 25 | 36,211 – 36,982 | **2.1%** | 15,088 |

Fifteen files landing within 1.3% of each other is not an experimental coincidence — it is a fixed byte budget. The two folders converge on *different* per-file row counts (~267k vs ~36.5k) because Mirai's UDP payloads are larger, so fewer flows fit in the same bytes. This is textbook `tcpdump -C <size>` behaviour.

**Anomaly, noted for completeness:** `DDoS-SYN_Flood4.pcap.csv` holds 191,154 rows — a short file in an *interior* position. Most likely a seam where `mergecap` joined two underlying captures and the byte counter reset. It is not exploitable (it exists in one high-volume folder that was never the leakage problem) and is recorded here only so that a reader running the same check knows it was seen.

### 1.3 Consequences

1. **`source_csv_filename` indexes a contiguous temporal segment**, not an independent capture. A "group" is a time window.
2. **The numeric suffix is a `tcpdump` sequence number** encoding wall-clock order. Natural-numeric order *is* temporal order. (Lexicographic order interleaves it: `Flood1` < `Flood10` < `Flood2`.)
3. **A shard-level split buys temporal separation, not environmental separation.** Same attacker, same victim, same testbed, same run — only a different minute of it.
4. **`StratifiedGroupKFold` is now actively wrong.** It assigns groups *randomly*, so it can place minute 40 in train and minute 20 in test. Training on the future to predict the past is a data-snooping violation (Arp et al., P3) hiding inside what looked like a leakage fix.

The split protocol in §4 is rewritten accordingly.

### 1.4 What leakage remains, and why it is irreducible

Three layers. Only the first is reachable.

- **Layer 1 — temporal proximity.** Adjacent flows in one capture are near-identical. A shard-level (or block-level) temporal holdout separates them. **This we fix.**
- **Layer 2 — single testbed run per attack.** Every `DDOS-ICMP_FLOOD` row in the dataset originates from one attacker, one victim, one network, one afternoon. No partition of that data separates a model from the IP range, the TTL distribution, or the packet-size fingerprint. Arp et al. name this exactly (P4, False Causality): a NIDS may learn to detect a network region rather than an attack pattern. The only fix is a second, independent capture. **It does not exist.**
- **Layer 3 — 15 of 34 classes live in exactly one shard.** Even the block-level fallback of §4.3 only approximates independence.

Layers 2 and 3 are properties of the dataset, not of this pipeline. Every published result on CICIoT2023 carries the same defect. The thesis must state this rather than claim a guarantee it cannot deliver — see §4.5 and §8.

---

## 2. Why random undersampling failed the panel (the theory)

A macro-class is **multi-modal**: DDoS = SYN flood + UDP flood + ICMP flood + HTTP flood + fragmentation variants (12 sub-labels), each forming dense region(s) in feature space, of wildly different sizes. Random undersampling draws uniformly across the whole class, so it keeps each sub-mode **in proportion to its current size** and **silently deletes small modes**.

Concrete failure, using real DDoS counts: `DDOS-ICMP_FLOOD` has 7,200,436 rows and `DDOS-SLOWLORIS` has 23,425 — a 307× ratio *inside one category*. Sampling 200K uniformly from DDoS keeps ~42K ICMP-flood rows and ~138 SlowLoris rows. The rare mode is effectively gone.

**Why this is worse for *this* thesis than for a generic classifier paper:**

1. **Per-class VAE manifold** — each VAE learns one category's manifold. If RUS truncates DDoS to mostly-ICMP-flood, the VAE learns a *narrowed* manifold, and "on-manifold" attacks are on-manifold of a distribution we accidentally clipped, not of real DDoS.
2. **IDSR / Mahalanobis reference** — IDSR uses class centre μ and covariance Σ. Deleting a mode shifts μ and Σ, so the fidelity metric measures against a distorted reference.
3. **Reproducibility** — different seeds keep different rare modes; results wobble on minority classes (Web n=24,828; BruteForce n=13,064).

**Takeaway:** random is *harmless within a homogeneous group* and *harmful across heterogeneous groups*. The fix is to make sampling **mode-aware**, so the "random" part only ever acts inside a homogeneous cluster.

---

## 3. The undersampling landscape (literature) and fit for NIDS

Two fundamental families — the distinction is decisive here:

- **Prototype selection** → keeps a subset of *real original rows*.
- **Prototype generation** → *manufactures new rows* (e.g. cluster centroids). **Disqualifying for us** — synthetic points fail the 49-rule validator (fractional flags, broken `Min ≤ AVG ≤ Max`, impossible protocol combinations) and corrupt the VAE manifold. Same reason we reject SMOTE/ADASYN.

### 3.1 Method-by-method verdict

| Method | Family | Keeps real rows? | Intra/inter-class | Scales to millions? | Verdict |
|---|---|---|---|---|---|
| Random undersampling (RUS) | selection | yes | intra | yes | **Weak** — erases sub-modes (the panel's objection) |
| Tomek Links | selection (boundary) | yes | inter | no (k-NN) | **Poor** — deforms class, cross-class info, ~O(n²) |
| ENN / NCR | selection (boundary) | yes | inter | no | **Poor** — boundary cleaner, doesn't scale |
| NearMiss-1/2/3 | selection (boundary) | yes | inter | no | **Poor** — selects by distance to *other* classes |
| One-Sided Selection | selection (boundary) | yes | inter | no | **Poor** — same family as Tomek |
| Instance Hardness Threshold | selection | yes | needs classifier | partial | **Poor** — needs trained model, indirect |
| **Cluster Centroids** | **generation** | **NO** | intra | yes | **DISQUALIFYING** — synthetic rows break the validator |
| **Cluster-based selection (proportional + floor)** | **selection** | **yes** | **intra** | **yes (MiniBatchKMeans)** | **BEST — chosen method** |

### 3.2 Why boundary cleaners are the wrong tool here

They are **inter-class, boundary-focused** — their job is to reshape the decision boundary *between* classes by deleting majority points near minorities. Our problem is **per-class** reduction for per-class VAEs: shrink DDoS while keeping DDoS's *own internal shape*, not carve away its overlap with Benign.

- **Cross-class contamination:** they decide what to keep based on *other classes' positions*. The imbalanced-learning literature flags proximity-based removal as relying on out-of-training information.
- **They don't scale:** all are k-NN based (≈O(n²)). Running them on 33.9M DDoS rows is infeasible; NIDS papers that use them already subsampled to ~100K first.

### 3.3 Why intra-class cluster-selection is right

- **Intra-class** → one class at a time, ignoring others → no cross-class information, no boundary deformation.
- **Mode-aware** → proportional-with-floor keeps every dense region.
- **Real rows** → selection variant, validator-safe, manifold-faithful.
- **Scales** → MiniBatchKMeans is ~linear in n.
- **Published precedent** → not invented here; cite Lin et al. (2017), Ebrahimi-Shahabadi et al. (2021), Yen & Lee (2009).

---

## 4. The split protocol (LEAKAGE-CRITICAL — rewritten)

### 4.1 Pipeline order

```
data/processed/ciciot2023_labeled_full.parquet   (46,775,660 rows, 309 shards)
  → clean (NaN/inf drop, 99.99% clip, int-floor, zero-floor)   [row-level only, no cross-row info]
  → FORWARD-CHAINING TEMPORAL SPLIT                            [§4.2 / §4.3]
        → train segments | val segments | test segments
  → FIT RobustScaler on TRAIN ONLY
  → transform train/val/test with the train scaler
  → CLUSTER + SAMPLE majorities, TRAIN ONLY                    [§5, this doc's method]
  → compute class weights from the (subsampled) train set
```

**Why this exact order:**

- **Split before scaling** — else the scaler sees the test distribution. Leakage.
- **Split before sampling** — sampling reads global class counts; doing it first bleeds whole-dataset information into how train is built.
- **Cluster on TRAIN ONLY** — clustering the full dataset then splitting lets cluster structure encode test-set information.
- **Fit scaler before sampling** — RobustScaler's median/IQR is most faithful on the full natural train distribution.
- **val/test are NOT sampled** — keep them at natural proportions so headline ASR/IDSR are measured against a realistic class mix.

### 4.2 Forward-chaining shard split (19 classes with ≥3 shards)

Because shard suffixes are `tcpdump` sequence numbers, natural-numeric order is wall-clock order. Assign the **latest** segments to test.

```python
import re

def natkey(path: str):
    """Natural-numeric sort key. 'Flood2' sorts before 'Flood10'."""
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", path)]

def forward_chain_shards(shards, val_frac=0.10, test_frac=0.20):
    """shards: list of source_csv_filename for ONE class. Returns 3 lists."""
    s = sorted(shards, key=natkey)
    n = len(s)
    n_test = max(1, round(n * test_frac))
    n_val  = max(1, round(n * val_frac))
    assert n - n_test - n_val >= 1, "too few shards for a 3-way shard split"
    return s[: n - n_val - n_test], s[n - n_val - n_test : n - n_test], s[n - n_test :]
```

Applies to the 19 classes below (shard counts from the build manifest):

`Benign_Final` (4), `DDoS-ACK_Fragmentation` (13), `DDoS-ICMP_Flood` (27), `DDoS-ICMP_Fragmentation` (20), `DDoS-PSHACK_FLOOD` (16), `DDoS-RSTFINFLOOD` (16), `DDoS-SYN_Flood` (16), `DDoS-SynonymousIP_Flood` (14), `DDoS-TCP_Flood` (18), `DDoS-UDP_Flood` (21), `DDoS-UDP_Fragmentation` (13), `DoS-SYN_Flood` (8), `DoS-TCP_Flood` (11), `DoS-UDP_Flood` (17), `Mirai-greeth_flood` (29), `Mirai-greip_flood` (22), `Mirai-udpplain` (25)

— that is 17 with ≥3 shards. Two classes have exactly **2** shards and need §4.3's hybrid: `DoS-HTTP_Flood` (2), `MITM-ArpSpoofing` (2).

### 4.3 Contiguous-block fallback (15 single-shard classes + 2 two-shard classes)

For a class living in one shard, a shard-level holdout is logically impossible without stranding the entire class in one split. The fallback is to perform, at finer granularity, **the same operation `tcpdump` already performed at the folder level**: cut the row-ordered shard into contiguous temporal blocks.

```python
def block_split_single_shard(n_rows, val_frac=0.10, test_frac=0.20):
    """Contiguous, order-preserving. Returns three index ranges."""
    n_test = int(n_rows * test_frac)
    n_val  = int(n_rows * val_frac)
    n_train = n_rows - n_val - n_test
    return (slice(0, n_train),
            slice(n_train, n_train + n_val),
            slice(n_train + n_val, n_rows))
```

Rule table:

| Shards in class | Protocol |
|---:|---|
| ≥ 3 | forward-chaining shard split (§4.2) |
| 2 | later shard → **test**; earlier shard → contiguous block split into **train / val** |
| 1 | contiguous block split into **train / val / test** (§4.3) |

**This is only valid because row order was preserved.** The parquet build is a pure concatenator: no shuffles, no aggregates, verified byte-for-byte on the `Backdoor_Malware` shard (provenance doc §9.4). Shuffle at build time and blocks become meaningless.

Applies to: `Backdoor_Malware`, `BrowserHijacking`, `CommandInjection`, `DDoS-HTTP_Flood`, `DDoS-SlowLoris`, `DNS_Spoofing`, `DictionaryBruteForce`, `Recon-HostDiscovery`, `Recon-OSScan`, `Recon-PingSweep`, `Recon-PortScan`, `SqlInjection`, `Uploading_Attack`, `VulnerabilityScan`, `XSS` (single-shard) plus `DoS-HTTP_Flood` and `MITM-ArpSpoofing` (two-shard).

### 4.4 What this protocol guarantees, and what it does not

**Guarantees:**
- No shard spans two splits, for the 17 classes where that is meaningful.
- No test row precedes any training row, within any class. (Forward-chaining; no data snooping.)
- Every class is present in all three splits.

**Does not guarantee:**
- Capture-level independence. Impossible (§1.4, Layer 2).
- That block boundaries within a single shard correspond to anything physical. They are a proxy for temporal separation, weaker than a true shard boundary and much weaker than a true capture boundary.

### 4.5 Wording for the thesis

> The 309 CSV shards are not independent capture sessions. CIC's documented pipeline merges each attack's PCAPs (`mergecap`), splits the merged stream by file size (`tcpdump`), then extracts flow features (`DPKT`). Per-file row counts confirm this directly: within `DDoS-PSHACK_FLOOD`, fifteen of sixteen shards fall in 265,547–269,060 rows (1.3% spread) with a single 78,907-row remainder; within `Mirai-udpplain`, twenty-four of twenty-five fall in 36,211–36,982 rows (2.1% spread) with a 15,088-row remainder. The two folders converge on different per-shard row counts because the split budget is in bytes, not rows.
>
> Consequently `source_csv_filename` indexes a **contiguous temporal segment** of one capture, and the numeric suffix is a `tcpdump` sequence number encoding wall-clock order. We therefore split on shard identity as a **forward-chaining temporal holdout** — test segments are drawn from the latest portion of each attack's capture, and no flow from a training segment appears in test. We do **not** claim capture-level independence, which the dataset's single-run-per-attack design makes unattainable (cf. Arp et al., P4). For the fifteen classes occupying a single shard, we apply a contiguous-block temporal split within that shard, a finer-grained instance of the partitioning the vendor already performed.
>
> The residual leakage inherent to this design inflates absolute classifier accuracy. The thesis's central finding — the collapse from ASR_raw to ASR_valid under domain-constraint enforcement — is a **ratio** computed against a fixed classifier under fixed conditions, and is invariant to that inflation.

That last paragraph is the load-bearing one. It converts an unfixable dataset flaw into a demonstration that you understand your own evaluation better than the objection does.

---

## 5. The chosen sampling method (precise specification)

**Name to use in the thesis:** *"intra-class clustering-based undersampling (selection variant) with cluster-proportional-with-floor allocation."*

### 5.1 Per majority class, the algorithm

```
INPUT:  rows of one majority class (TRAIN split only, scaled continuous features)
        N     = cap for this class
        k     = number of clusters
        floor = min rows kept per cluster
        seed  = fixed global seed

1. labels = MiniBatchKMeans(n_clusters=k, random_state=seed).fit_predict(X_cont)

2. Partition clusters into two pools:
       SMALL  = { c : size[c] <= floor }      # cannot satisfy the floor -> take whole
       LARGE  = { c : size[c] >  floor }

3. alloc[c] = size[c]                for c in SMALL          # take all
   remaining = N - sum(alloc[SMALL])
   assert remaining >= 0, "N too small: floors on SMALL clusters already exceed the cap"

4. For c in LARGE:
       raw[c]   = remaining * size[c] / sum(size[LARGE])
       alloc[c] = clamp(raw[c], lo=floor, hi=size[c])

5. Iteratively repair sum(alloc[LARGE]) == remaining:
       while |sum(alloc[LARGE]) - remaining| > 0:
           FREE = { c in LARGE : floor < alloc[c] < size[c] }   # clusters not pinned
           if FREE is empty: break                             # cannot repair; log + accept
           delta = remaining - sum(alloc[LARGE])
           distribute delta across FREE proportional to size[c]
           re-clamp each alloc[c] to [floor, size[c]]

6. For each cluster c:
       idx   = indices where labels == c
       perm  = seeded_shuffle(idx, seed)
       keep += perm[:alloc[c]]

7. return rows[keep]
```

**Step 2–3 exist because the previous revision's pseudocode was unsatisfiable.** It asserted both `alloc[c] >= floor` and `alloc[c] <= size[c]`, which have no solution when `size[c] < floor` — a tiny cluster smaller than the floor. The old renormalisation loop would either spin or silently return `sum(alloc) != N`. Small clusters are now taken whole and removed from the renormalisation pool. **Write a unit test for `size[c] < floor` before trusting the sampler.**

Step 5's `break` case (every LARGE cluster pinned at its floor or its size) is reachable when `N` is close to the class's total size. Log it loudly; it means the cap is not actually reducing anything and `N` should be lowered or the class left uncapped.

### 5.2 Which individual row is kept? (the mechanical answer the panel wants)

A row is kept **iff** (1) KMeans assigned it to cluster `c`, **and** (2) its position in cluster `c`'s seeded shuffle is `< alloc[c]`.

Within a cluster the pick is a **seeded uniform shuffle** — safe because a cluster is *by construction* a region of near-identical rows, so dropping one of two near-twins loses nothing. Structure preservation comes entirely from **allocation** (how many rows each cluster gets), not from the within-cluster pick.

- Reproducible: same seed → same KMeans → same shuffle → same rows, every run, every machine.
- Optional deterministic variant: keep the `alloc[c]` rows **nearest the centroid** (Lin 2017's nearest-neighbour-of-centroid selection). **Caveat:** this shrinks within-cluster variance and narrows the manifold. Prefer seeded random-within-cluster unless a reviewer specifically demands determinism beyond the seed.

### 5.3 Rules that are non-negotiable

- **Never** use centroid replacement (prototype generation) — synthetic rows break the validator.
- **Never** apply SMOTE/ADASYN — same reason (already a stated strength of the thesis).
- **Keep rare categories whole** — Web (24,828) and BruteForce (13,064) enter at 100%, never clustered or capped.
- **Cap only** DDoS, DoS, Mirai, Recon, Spoofing, Benign. Check Benign's train-split count first — at 1,098,126 total it may sit near its cap already.
- **Cluster on continuous features only** (the 23 RobustScaler'd ones). The 11 binary, 4 protocol-derived binary, and 1 categorical feature distort Euclidean KMeans.
- **Cluster on the train split only.** Never on the full dataset.

---

## 6. What to CHECK BEFORE implementing

> Do not write the sampler until §6.2 is answered. It determines whether the method is justified *for each class*.

### 6.1 File ↔ class coupling — RESOLVED, no diagnostic needed

The previous revision proposed a three-way diagnostic ("mixed / one-class-ish / rare class in 1–2 files"). It is unnecessary: **every shard is single-class by construction**, because the label is derived from the parent folder name and no shard carries a `Label` column. Coupling is total, for all 34 classes. The decision tree resolves immediately to the third branch.

The operative facts, already established:

- **19 classes** have ≥2 shards; **17** of those have ≥3 and take the forward-chaining shard split (§4.2).
- **2 classes** have exactly 2 shards and take the hybrid (§4.3).
- **15 classes** have exactly 1 shard and take the contiguous-block split (§4.3).

Record this in the run manifest. Do not re-derive it.

### 6.2 Multi-modality — THE diagnostic that matters

This is what the entire method rests on. If cluster sizes come back roughly uniform, proportional-with-floor ≈ random, and the panel's original objection stands unanswered.

- [ ] **Per majority class**, sample ~50K train rows, run `MiniBatchKMeans(k ∈ {10,15,20})` on scaled continuous features, plot the **cluster-size distribution**.
- [ ] **Highly skewed sizes** → multi-modal → proportional-with-floor genuinely preserves structure. This is the win; claim it.
- [ ] **Roughly uniform sizes** → unimodal → proportional-with-floor ≈ random. **Say so honestly** and cap that class with plain seeded sampling. Do not claim structure preservation you did not get.
- [ ] **Decide per class.** DDoS has 12 sub-labels and is near-certain to be multi-modal. **Benign has one sub-label and 1.1M rows — nobody knows whether it is multi-modal.** If it is not, clustering it is theatre.

### 6.3 Cluster ↔ subtype crosstab — MANDATORY, not optional

The previous revision marked this optional. It is the single strongest piece of evidence available to you.

```python
pd.crosstab(cluster_labels, subtype_labels, normalize="index")
```

If clusters recover the known sub-attack structure — cluster 3 is ~all `DDOS-SYN_FLOOD`, cluster 7 is ~all `DDOS-ICMP_FRAGMENTATION` — you have converted *"I clustered and hoped"* into *"clustering recovered the ground-truth sub-attack partition; here is the contingency table."* That is a figure in the thesis and a one-sentence answer at the defense.

If clusters **don't** align with subtypes, that is also worth knowing before you build on them. Either way: one `crosstab`, run it.

### 6.4 Parameter sizing

- [ ] **N (cap):** driven by per-class VAE training time on the research machine (RTX 4070 Ti Super, 16 GB VRAM). Cap = VAE train-set size. Start at 150–250K.
- [ ] **k:** tie to the known sub-attack count per category (DDoS 12, DoS 4, Mirai 3, Recon 5, Spoofing 2, Benign 1). Do not pick arbitrarily. Run k ∈ {10,15,20} and confirm the retained distribution is stable.
- [ ] **floor:** tie to "minimum rows for a stable per-class VAE and non-degenerate Σ." Comfortably above the latent dimension (16). Justify the number in the thesis, do not assert it.

### 6.5 Leakage guards (assert in code, not in prose)

- [ ] Scaler and KMeans are fit on **train only**.
- [ ] No `source_csv_filename` appears in two splits.
- [ ] For block-split classes, no row index appears in two splits, and `max(train_idx) < min(val_idx) < min(test_idx)`.
- [ ] val/test are **untouched** by the sampler.
- [ ] For shard-split classes, `max(natkey(train_shard)) < min(natkey(test_shard))` — forward-chaining actually holds.

---

## 7. What to IMPLEMENT

### 7.1 Core module (`src/preprocessing/sampler.py`)

- [ ] `cluster_proportional_floor_sample(X_cont, N, k, floor, seed) -> kept_indices` — pure, deterministic, returns indices into the input.
- [ ] Allocation per §5.1, **including the SMALL/LARGE pool split**.
- [ ] Unit test: `size[c] < floor` for at least one cluster; assert `sum(alloc) == N` and no infinite loop.
- [ ] Unit test: `N` close to total size; assert the step-5 `break` path logs and does not silently corrupt counts.
- [ ] Seeded `MiniBatchKMeans` + seeded within-cluster shuffle.
- [ ] Toggle `selection_mode ∈ {"random_within", "nearest_centroid"}` (default `random_within`).
- [ ] Rare-class passthrough below a configurable threshold.
- [ ] **Dataset-agnostic:** continuous-feature list, cap, k, floor all passed in → reusable for the two ablation datasets.

### 7.2 Split module (`src/preprocessing/splitter.py`) — NEW

The previous revision assumed `StratifiedGroupKFold` would do this. It will not (§1.3).

- [ ] `natkey(path)` natural-numeric sort key.
- [ ] `forward_chain_shards(shards, val_frac, test_frac)` for ≥3-shard classes.
- [ ] `block_split_single_shard(n_rows, val_frac, test_frac)` for 1-shard classes.
- [ ] Two-shard hybrid per the §4.3 rule table.
- [ ] Dispatch on `len(shards)` per class; emit a per-class record of which protocol was used.
- [ ] Assertions from §6.5.

### 7.3 Pipeline wiring (`src/preprocessing/pipeline.py`)

`pipeline.py` and `sampler.py` are currently archived under `src/preprocessing/old/`, and the intermediate `raw_loaded.parquet` they expected no longer exists. Both must be unarchived and rewired against `config/paths.py` and `data/processed/ciciot2023_labeled_full.parquet` before any of this runs.

- [ ] Driven by `config/paths.py` — no hardcoded paths.
- [ ] Global `seed` for numpy / sklearn / torch.
- [ ] Order enforced as in §4.1 (split → scaler fit → sample train → weights).
- [ ] Persist `run_manifest.json`: seeds, caps, k, floor, per-class split protocol, per-cluster kept counts, artefact hashes.

### 7.4 Diagnostics / evidence figures (for the defense)

- [ ] **Cluster-size distribution per majority class** — justifies the method (or honestly caveats it).
- [ ] **Cluster × subtype crosstab** (§6.3) — the strongest single figure.
- [ ] **Pre vs post within-cluster feature distributions** — shows sampling preserves shape.
- [ ] **Pre vs post whole-class distribution overlap** (UMAP or per-feature KDE) — visual proof of retained fidelity.
- [ ] **Per-split per-class counts table** — honesty about the split protocol's effect on proportions.

### 7.5 The sensitivity study (one extra training run, high value)

Train the four gradient-attackable classifiers twice on the 17 shard-splittable classes: once with a **row-level shuffled** split, once with the **forward-chaining** split. The accuracy delta is the measured cost of temporal leakage.

Then state: *for the fifteen single-shard classes we could not construct a shard-level holdout; on classes where the comparison was possible, temporal leakage inflated accuracy by ~X points, so treat those fifteen classes' figures as an upper bound.*

That converts an unfixable weakness into a **quantified** limitation. One extra training run, zero new code.

---

## 8. Panel-facing answers (rehearse these)

**Q: "How did you downsample, which technique, and why?"**

1. **What:** "I cap each majority class for tractability — the per-class VAEs and iterative attacks don't need millions of rows — while keeping the rare categories, Web and BruteForce, whole."
2. **How:** "Not random undersampling. Intra-class clustering-based undersampling, selection variant: I cluster each majority class in feature space and allocate the retention budget proportional to cluster size with a per-cluster floor, so every dense mode survives instead of small sub-attacks being statistically erased. The crosstab shows the clusters recover the known sub-attack labels."
3. **Why this, not the alternatives:** "It keeps real rows — I avoid Cluster *Centroids* and SMOTE/ADASYN because generated points violate my 49-rule domain validator and corrupt the manifold the VAE learns. I avoid Tomek/ENN/NearMiss because those are inter-class boundary cleaners: they deform each class's distribution, select using other classes' positions, and don't scale to 34 million DDoS rows. My goal is per-class fidelity, not boundary sharpening. It's published methodology — Lin et al. 2017, Ebrahimi-Shahabadi et al. 2021 — and I cluster on the training split only."

**Q: "Did your splits leak across capture files?"**

> "That question assumes each CSV is a separate capture, which is how the dataset is usually described but not how it was built. CIC merged each attack's PCAPs, then split the merged stream by file size with `tcpdump`. I verified this from the per-file row counts — fifteen of sixteen PSHACK shards land within 1.3% of each other, with one short remainder. So a shard is a contiguous time window, not an independent session. I split forward in time: the latest shards go to test, and no training flow post-dates a test flow. For the fifteen classes in a single shard I apply the same cut at block level inside the file. I don't claim capture-level independence — the dataset ran each attack once, so it isn't available to anyone using CICIoT2023. What I do claim is that my headline metric is a ratio against a fixed classifier and is invariant to the residual inflation."

**Anticipated follow-ups:**

- *"KMeans assumes convex clusters; traffic isn't convex."* → "KMeans is a pragmatic mode-finder. The goal isn't perfect clustering, only that no dense region is dropped — the floor guarantees coverage even where boundaries are imperfect. The subtype crosstab shows the recovered partition is close enough to the ground-truth sub-attacks to serve that purpose."
- *"Why k=15, why that floor?"* → sub-attack count per category and stable-Σ row counts (§6.4); show the k-sensitivity check.
- *"If the within-cluster pick is random, how do you know nothing important was dropped inside a cluster?"* → "A cluster is a homogeneous region by construction, so the post-sampling within-cluster distribution matches the pre-sampling one — here is the pre/post overlap figure."
- *"Why not `StratifiedGroupKFold`?"* → "It assigns groups randomly, which for temporally-ordered shards means it can train on minute 40 and test on minute 20. That's data snooping (Arp et al. P3). Forward-chaining gives the same disjointness guarantee without inverting the arrow of time."

---

## 9. What to TRACK (every run)

| Item | Why it matters |
|---|---|
| seed (global) | reproducibility |
| per-class cap N | VAE train size; comparability across datasets |
| k per class | structure granularity; sensitivity reporting |
| floor per class | rare-mode survival guarantee |
| per-cluster kept counts | proof the allocation behaved as designed |
| clusters hitting the SMALL pool | proof the `size[c] < floor` path was exercised |
| step-5 `break` events | warns the cap is not actually reducing the class |
| per-class kept totals (train) | matches intended caps; rare classes = 100% |
| val/test counts (untouched) | proof of eval-distribution fidelity |
| split protocol per class (shard / hybrid / block) | defensibility of the leakage story |
| forward-chaining assertion result | proof no test row precedes a train row |
| multi-modality result per class | justifies (or honestly caveats) the method |
| cluster × subtype crosstab | the defense's strongest figure |
| artefact hashes | audit trail |

---

## 10. Reusing this for the ablation datasets

Same **procedure**, dataset-specific **parameters**:

- **Re-derive the shard semantics first.** Do not assume the new dataset's files are sessions. Check per-file row counts for the fixed-budget signature (§1.2). If files *are* genuine independent captures, a true group-disjoint split is available and should be used — and the difference from CICIoT2023 is itself worth a sentence in the thesis.
- Re-run §6.2 multi-modality and §6.3 crosstab per class.
- Re-pick k/floor/N per dataset. Do **not** force CICIoT2023's caps onto a smaller dataset.
- Map each dataset's features to its own continuous-feature list for clustering.
- Keep the *method* identical (intra-class cluster-selection, proportional-with-floor, train-only, real rows, rare-classes-whole) so cross-dataset comparison is apples-to-apples.
- State explicitly: "identical sampling procedure, dataset-tuned parameters; split protocol adapted to each dataset's shard semantics."

---

## 11. Execution order (defense in ~3 months)

Nothing downstream is meaningful until the split is correct, and the sampler should not be written until §6.2 says it is justified.

1. **Fix the split protocol** — implement `splitter.py` per §4.2/§4.3 with the §6.5 assertions. Half a day. Changes what every downstream number means.
2. **Run §6.2 multi-modality + §6.3 crosstab** per majority class. One day. Either validates the method or tells you to stop.
3. **Unarchive and rewire** `pipeline.py` and `sampler.py` against `config/paths.py` and the labelled parquet. Resolve the missing `raw_loaded.parquet` dependency (§7.3).
4. **Regenerate** `X_*.npy` / `y_*.npy` / scaler / encoders under the corrected split.
5. **Sensitivity study** (§7.5) — one extra training run.
6. **Retrain VAEs and re-run attacks** on the corrected splits.

---

## 12. Changelog vs. previous revision

| # | Was | Now | Why |
|---:|---|---|---|
| 1 | "169 Spark `part-*.csv` shards" | **309 `tcpdump`-split CSV shards** | 169 is CIC's PCAP count; the CSV distribution has 309 files, named `<Attack>N.pcap.csv` |
| 2 | "DDoS ~46% vs BruteForce ~0.28%" | **DDoS 72.65% vs BruteForce 0.03%** | prior figures came from a stratified sample, not the full labelled parquet |
| 3 | Shard = independent capture session | **Shard = contiguous byte-slice of a merged capture** | CIC's documented `mergecap → tcpdump → DPKT` pipeline; confirmed by 1.3–3.1% row-count spread within folders |
| 4 | Group-disjoint split via `StratifiedGroupKFold` | **Forward-chaining temporal split, natural-numeric shard order** | `StratifiedGroupKFold` assigns groups randomly → can train on later time than it tests on → data snooping (Arp P3) |
| 5 | "← fixes prior leakage" | **"← reduces temporal leakage; capture-level leakage is irreducible"** | Layers 2 and 3 of §1.4 are properties of the dataset |
| 6 | §5.1 file↔class coupling diagnostic | **Deleted; coupling is total by construction** | label derives from folder name; every shard is single-class |
| 7 | `alloc[c] = max(floor, raw[c])` with `alloc[c] <= size[c]` | **SMALL/LARGE pool split** | unsatisfiable when `size[c] < floor`; old loop spins or silently breaks the cap |
| 8 | Cluster×subtype crosstab "optional" | **Mandatory** | strongest available evidence for the panel |
| 9 | "RTX 4080S, 64GB" | **RTX 4070 Ti Super, 16 GB VRAM** | actual research machine |
| 10 | — | **§7.5 sensitivity study added** | converts irreducible leakage into a quantified bound |

---

## 13. References

> Verify each citation's bibliographic details against the source before adding to the thesis `.bib`.

**Dataset & its construction (cite for §1):**
- Neto, E.C.P., Dadkhah, S., Ferreira, R., Zohourian, A., Lu, R., Ghorbani, A.A. (2023). *CICIoT2023: A real-time dataset and benchmark for large-scale attacks in IoT environment.* Sensors 23(13):5941. DOI 10.3390/s23135941.
- Canadian Institute for Cybersecurity, *IoT Dataset 2023* dataset page, University of New Brunswick. — **cite this for the `mergecap` → `tcpdump` → `DPKT` pipeline description.** This is the primary evidence for §1.1; the Sensors paper does not state the split granularity as plainly.

**Methodology hygiene (cite for §4 and §1.4):**
- Arp, D., Quiring, E., Pendlebury, F., Warnecke, A., Pierazzi, F., Wressnegger, C., Cavallaro, L., Rieck, K. (2022). *Dos and Don'ts of Machine Learning in Computer Security.* USENIX Security. — **P3 (Data Snooping)** for why random group assignment on temporally-ordered shards is unsound; **P4 (False Causality / spurious correlations)** for why a single-testbed capture lets a NIDS learn a network region rather than an attack pattern. Load-bearing citation for the "irreducible leakage" argument.
- Imbalanced-learning leakage reviews (e.g. "Don't push the button! Exploring data leakage risks in ML") — proximity-based resampling can rely on out-of-training data; resample inside train only.

**Core sampling method:**
- Lin, W.-C., Tsai, C.-F., Hu, Y.-H., Jhang, J.-S. (2017). *Clustering-based undersampling in class-imbalanced data.* Information Sciences. — k-means undersampling; centroid-replacement vs nearest-neighbour-of-centroid (selection); selection variant outperforms RUS.
- Ebrahimi-Shahabadi, M.S., Tabrizchi, H., Kuchaki Rafsanjani, M. (2021). *A combination of clustering-based under-sampling with ensemble methods for solving imbalanced class problem in intelligent systems.* Technological Forecasting & Social Change. — cluster-proportional sampling weights preserving in-cluster distribution; closest precedent to proportional-with-floor.
- Yen, S.-J., Lee, Y.-S. (2009). *Cluster-based under-sampling approaches for imbalanced data distributions.* Expert Systems with Applications. — foundational.

**Method taxonomy / what was rejected:**
- *Benchmark of Data Preprocessing Methods for Imbalanced Classification* (arXiv:2303.03094). — classifies Cluster Centroids as **prototype generation**; use to justify rejecting it.
- *CLIMB: Class-imbalanced Learning Benchmark on Tabular Data* (arXiv:2505.17451).
- Tomek (1976); Wilson (1972) ENN; Mani & Zhang (2003) NearMiss-1/2/3.

**NIDS-specific resampling (situates the work):**
- Abdelkhalek, A., Mashaly, M. (2023). *Addressing the class imbalance problem in NIDS using data resampling and deep learning.* J. Supercomputing. DOI 10.1007/s11227-023-05073-x.
- *A detailed study of resampling algorithms for cyberattack classification* (PMC11041950).
- *Enhanced intrusion detection system IoT … FFNN* (Sci. Reports, 2025, s41598-025-20047-0). — CICIoT2023 with random over+under sampling; the baseline being improved on.
- *Towards unbalanced multiclass intrusion detection with hybrid sampling* (Applied Soft Computing, 2024). — note its "evaluate on a distinct authentic test set" rule, which matches our val/test-untouched constraint.

---

*End of document.*