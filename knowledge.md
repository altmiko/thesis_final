# Project Knowledge: Mixed-Input Beta-VAE for Valid Adversarial Attacks on CICIoT2023 NIDS

Bachelor's thesis workspace (Rafsan Rahman, BRAC University CSE). Studies whether tabular
adversarial attacks on ML-based network intrusion detection can be forced to remain
**domain-valid** by generating them **on-manifold** through per-category β-VAEs.

Repo root: `E:/Shameem/thesis` (older docs use `D:/thesis_final`; treat as the same
project — hardcoded absolute paths remain in a few scripts).

## 1. Thesis Claim (the one sentence to defend)

> Conventional tabular adversarial attacks (FGSM/PGD/CW) achieve very high raw attack
> success rate (`ASR_raw`) against NIDS classifiers, but almost none of those adversarial
> examples correspond to physically realizable network traffic once inverse-transformed
> and checked against domain rules. A Mixed-Input β-VAE trained per attack category, plus
> latent-space PGD/CW with a perturbation-mask + protocol governance, generates attacks
> that keep meaningful `ASR_valid` — that is, evade the IDS **and** pass the 49-rule
> domain validator.

Core metric contrast:

| Metric | Meaning |
|---|---|
| `ASR_raw` | fraction of clean-correct samples the classifier misclassifies after attack |
| Validity rate | fraction of adversarial samples passing raw-space domain rules |
| `ASR_valid` | `ASR_raw AND valid` — the honest IDS-robustness number |
| Gap | `ASR_raw - ASR_valid` — the inflation caused by ignoring domain validity |

## 2. Environment

- Python 3.10/3.11 conda env `thesis`
  - Windows path: `C:\Users\T2530985\.conda\envs\thesis\python.exe`
  - Alt: `C:\Users\user6\.local\share\mamba\envs\thesis` (see `environment.yml`)
- Deps (`requirements.txt`, `environment.yml`): `numpy=2.4`, `pandas=3.0`,
  `scikit-learn=1.9`, `torch=2.5.1+cu121`, `torchvision=0.20.1+cu121`, `torchattacks=3.5.1`,
  `xgboost=3.2`, `lightgbm` (conda), `adversarial-robustness-toolbox=1.20.1`,
  `matplotlib=3.11`, `seaborn=0.13.2`, `scipy=1.17.1`, `PyYAML`, `joblib`.
- Global seed: **42**. GPU-aware scripts take `--device cuda`.
- Windows shell: prefer **Git Bash** over PowerShell for chained commands; harness scripts
  used `python -m src.<pkg>.<module>` (see fix.md #1–#3).

## 3. Pipeline at a Glance

```
raw CSV (~10 GB, 45M rows, 34 labels, 39 features + Label)
    │
    ├── src/evaluation/eda_tables.py           → T1–T5, class_to_category.json
    ├── src/evaluation/eda_figures_part1.py    → F1–F3, F3 correlation matrix
    ├── src/evaluation/eda_figures_part2.py    → F4–F7, T6, netdiffuser_categorization.json
    │       (calls src/preprocessing/netdiffuser_categorization.py)
    ▼
data/processed/raw_loaded.parquet      (4,429,940 stratified sampled rows)
    │
    ▼  src/preprocessing/pipeline.py
        clean → 99.99% clip → int/bin round → validate → LabelEncoder (34/8/2)
        stratified 70/10/20 → RobustScaler on train → build perturbation_mask
        sha256 hashes + config/run_manifest.json
    ▼
data/processed/X_{train,val,test}.npy + y_*(_cat|_bin).npy + scaler.pkl + encoders
    │
    ├── src/classifiers/baseline_experiments.py → 15 checkpoints (5 archs × {bin,8,34})
    │       models/{mlp,cnn,lstm,serial,dualpath}_{binary,8class,34class}.pt
    │
    ├── src/vae/train_all.py → 8 per-category MixedInputBetaVAEs
    │       models/vae/vae_class_{0..7}_{Benign..Web}.pt
    │       results/vae/diagnostics_<class>.json, training_log*.txt, curves .png
    │       vae_run_manifest.json (root) + results/vae/<run_tag>/vae_run_manifest.json
    │
    ├── input-space attacks
    │       src/attack/run_attacks.py         (MLP × {bin,8,34})
    │       src/attack/run_attacks_8class_models.py (all archs on 8-class)
    │       src/attack/adversarial_attacks.py (TabularFGSM/PGD/CW, run_attack, restarts)
    │
    ├── latent-space attacks
    │       src/attack/latent_infra.py  (AttackRouter, PerturbationMask, ProtocolValidator,
    │                                    MahalanobisOutlierDetector, decoder-residual)
    │       src/attack/latent_pgd.py    (CE + CW-margin objective, adaptive PGD, restarts)
    │       src/attack/latent_cw.py     (targeted/untargeted CW in latent space)
    │       src/attack/latent_gmm.py    (BayesianGMM latent priors, cached)
    │       src/attack/latent_restarts.py (encoded/jitter/gmm restart mixer,
    │                                       class-specific ε defaults)
    │       Runners: run_all_models_attack_rerun.py (canonical),
    │                run_new_vae_attack_rerun.py (latent-only, tagged),
    │                run_targeted_benign_latent_pgd.py, ...kappa_sweep, ...cw
    │
    └── evaluation
            src/evaluation/validity_analysis.py, run_validity_analysis.py
            src/evaluation/validate_full_dataset.py
            src/attack/build_thesis_bundle.py, src/thesis_visualizations.py
            src/plot_distributional_fidelity.py, src/plot_benign_overlay_fidelity.py
            src/thesis_eval/fidelity_tables.py, realism_table.py, metrics/fidelity.py
```

## 4. Dataset & Schema

- Source: `data/ciciot2023/ciciot2023_base.csv` — 10.1 GB, ~45,019,244 rows,
  **34 attack labels** (uppercase, e.g. `DDOS-ICMP_FLOOD`).
- Staged working table: `data/processed/raw_loaded.parquet` — **4,429,940 rows** after
  the SAMPLE-mode capping in the manifest (`majority=medium=200000` cap, minority/rare
  kept in full).
- Splits: **train 3,100,958 / val 442,994 / test 885,988** (stratified on 34-class label).
- Category counts (8-class) in the sampled dataset:

  | Category | Count | % |
  |---|---:|---:|
  | DDoS | 2,049,917 | 46.27 |
  | DoS | 668,775 | 15.10 |
  | Mirai | 599,960 | 13.54 |
  | Recon | 503,528 | 11.37 |
  | Spoofing | 371,451 | 8.39 |
  | Benign | 199,989 | 4.51 |
  | Web | 23,798 | 0.54 |
  | BruteForce | 12,522 | 0.28 |

### 4.1 Feature schema — 39 features (vendor's CSV mirror, "Modified Schema A")

Ordered exactly as the CSV minus `Label`
(`src/preprocessing/old/feature_groups.py::FEATURE_NAMES`):

```
Header_Length, Protocol Type, Time_To_Live, Rate,
fin_flag_number, syn_flag_number, rst_flag_number, psh_flag_number,
ack_flag_number, ece_flag_number, cwr_flag_number,
ack_count, syn_count, fin_count, rst_count,
HTTP, HTTPS, DNS, Telnet, SMTP, SSH, IRC,
TCP, UDP, DHCP, ARP, ICMP, IGMP, IPv, LLC,
Tot sum, Min, Max, AVG, Std, Tot size, IAT, Number, Variance
```

Notable: **includes `Time_To_Live` and `IGMP`**. The vendor's CSV mirror already
omits `Magnitude`, `Radius`, `flow_duration`, `Duration`, `Srate`, `Drate`,
`urg_count`, `Covariance`, `Weight` — these cannot be recovered without
re-running CICFlowMeter on the original PCAPs. Not a codebase decision; it's
what CIC ships. Full rationale + defense-safe wording in `docs/data/ciciot.md` §4.

### 4.2 Feature governance groups

| Group | Members | Role |
|---|---|---|
| `IMMUTABLE_FEATURES` | `Protocol Type`, `TCP`, `UDP`, `ICMP` | never perturbed |
| `QUASI_IMMUTABLE_FEATURES` | app-layer indicators + `Time_To_Live` | frozen by default |
| `BINARY_FEATURES` | 15 protocol/app indicators (HTTP..LLC) | round to {0,1} |
| `INTEGER_FEATURES` | 12 flag/count cols + `Number` | round, clip ≥0 |
| `MUTABLE_FEATURES` | rates, header, size aggregates, flags/counts, IAT, Number, Variance | attacker-controllable |
| `FULL_PERTURBABLE_OVERRIDE_FEATURES` | Header_Length, Rate, Tot sum, Min, Max, AVG, Std, Tot size, IAT, Number, Variance | ε=1.0 in mask |

### 4.3 Perturbation mask (`data/processed/perturbation_mask.npy`)

Encoded per-feature policy, from `pipeline.py` §4.7 combining `MUTABLE_FEATURES`,
`FULL_PERTURBABLE_OVERRIDE_FEATURES`, `MANUAL_CONCENTRATED_DECISIONS`, near-zero-IQR
policy, and NetDiffuser discrete/relative split:

| Mask value | Meaning | Count | Features |
|---:|---|---:|---|
| 1.0 | Full perturbation | **11** | Header_Length, Rate, Tot sum, Min, Max, AVG, Std, Tot size, IAT, Number, Variance |
| 0.3 | Partial (± delta) | **9** | fin/syn/rst/psh/ack `_flag_number`, ack/syn/fin/rst `_count` |
| 0.0 | Frozen | **19** | Protocol Type, TTL, ece/cwr flags, HTTP, HTTPS, DNS, Telnet, SMTP, SSH, IRC, TCP, UDP, DHCP, ARP, ICMP, IGMP, IPv, LLC |

Auto-frozen (near-zero IQR, constant kind): DHCP, IGMP, IRC, SMTP, Telnet,
cwr_flag_number, ece_flag_number.
Manual concentrated → `allow_mutable`: `fin_flag_number, syn_flag_number, rst_flag_number,
psh_flag_number, ack_flag_number, fin_count, rst_count`.

NetDiffuser categorization result: **21 discrete / 18 relative**, best cut height
0.272727, non-trivial local max (CH-score).

### 4.4 Category mapping (34 → 8)

`Benign, BruteForce, DDoS (12), DoS (4), Mirai (3), Recon (5), Spoofing (2), Web (6)`.
Labels are ALL UPPERCASE in the CSV; keys of `CATEGORY_MAP` must match.

## 5. Domain Validator (raw-space, 49 rules)

`src/attack/validator.py::validate_batch(X, feature_names)` returns
`ValidationResult(n_samples, violations_per_rule)` with `.overall_valid`,
`.validity_rate`, `.per_rule_violation_rate()`.

Rule families (G1–G8):

- **G1 Non-negativity** — `Header_Length, Rate, Time_To_Live, Tot sum, Min, Max, AVG, Std, Tot size, IAT, Number, Variance` and all flag/count columns; tolerance `FLOAT_TOL = 0.01`.
- **G2 Protocol allowlist** — `Protocol Type ∈ {0 HOPOPT, 1 ICMP, 2 IGMP, 6 TCP, 17 UDP, 47 GRE}` and integer-like within tol.
- **G3 Binary constraints** — all 15 binary indicators integer-like and `∈ {0,1}`.
- **G4 Protocol-indicator consistency** — indicator-implies-protocol only: `TCP=1 ⇒ Proto=6`, `UDP=1 ⇒ Proto=17`, `ICMP=1 ⇒ Proto=1`, `IGMP=1 ⇒ Proto=2`. (Weaker "protocol implies indicator" direction is not enforced.)
- **G5 Statistical ordering** — `Min ≤ Max`, `Min ≤ AVG ≤ Max` within tol.
- **G6 Variance–std consistency** — `|Variance − Std²|` must be small in both absolute (`VAR_ABS_TOL=0.01`) and relative (`VAR_REL_TOL=0.05`) terms.
- **G7 TTL range** — `Time_To_Live ∈ [0, 255]`.
- **G8 Packet count** — `Number ≥ 1` and integer-like.

**Reports:**

- `data/processed/clean_data_validity_report.json` — pre-clean (raw) audit.
- `data/processed/processed_data_validity_report.json` — post-clean, currently **100%**.
- `results/validation/full_dataset_validation_report.txt` — all 4,429,940 processed rows
  pass all 49 rules across train/val/test. Also confirms 9/9 synthetic corruption tests.
- Drift alert: `tables/T4_clean_validity.md` shows an older 81.37% figure (produced before
  final validator/preprocessing alignment). Always cite the JSON reports as authoritative.

Validation is intentionally done in **raw space** — every attack artifact is
`scaler.inverse_transform`-ed first (`validity_analysis.py::inverse_transform_results`).

## 6. Baseline IDS Classifiers (`src/classifiers/`)

`models.py` factory: `get_model('mlp'|'cnn'|'lstm'|'serial'|'dualpath'|'attn_serial',
num_features, num_classes)`.

| Model | Shape recipe |
|---|---|
| `SimpleMLP` | Dense(128)→ReLU→Drop→Dense(64)→ReLU→Drop→Dense(K) |
| `CNNOnly` | reshape→Conv1d(1→32)→Conv1d(→64)→AdaptiveMaxPool→Dense(64)→Dense(K) |
| `LSTMOnly` | reshape (seq_len=1)→BiLSTM(64)→Dense(64)→Dense(K) |
| `SerialCNNLSTM` | Conv1d×2→transpose→BiLSTM→Dense→Dense(K) — deliberately has the info bottleneck |
| `AttentionSerialCNNLSTM` | Serial + softmax-attention over LSTM steps |
| **`DualPathIDS`** | CNNBranch ∥ LSTMBranch → `FusionAttention` (branch or feature) → Dense→Dense(K). Thesis-proposed architecture. |

Training (`baseline_experiments.py`):
- 15 models = 5 archs × {binary, 8class, 34class}.
- Adam(lr=1e-3), plain `CrossEntropyLoss` **without class weights** (justified: stratified
  sampling already balances; original-distribution weights would double-penalize).
- `ReduceLROnPlateau(patience=3)`, early stop patience 5, batch 2048, ≤5 epochs.
- Outputs under `logs/baselines/<run_id>/` and `results/all_models_all_tasks_summary.json`.

**Baseline results (`results/all_models_all_tasks_summary.json`):**

| Task | Best Acc | Best Macro-F1 | Notes |
|---|---:|---:|---|
| Binary | DualPath 96.78% | MLP 0.787 | all ~96.6–96.8 |
| 8-class | DualPath 84.59% | LSTM 0.647 | MLP/LSTM/DualPath close |
| 34-class | LSTM 75.58% | LSTM 0.575 | fine-grained is hardest |

## 7. Input-Space Attacks (`src/attack/adversarial_attacks.py`)

Tabular variants of torchattacks with **image-style `[0, 1]` clamping removed**:

- `TabularFGSM`, `TabularPGD` (keeps ε-ball projection), `TabularCW` (no tanh cage).
- `run_attack(...)` orchestrates loading (`load_model` infers arch from filename), batches
  attack, returns clean/adversarial arrays and predictions.
- `run_attack_with_restarts(...)` re-uses the same primitive with multiple random starts.
- **`perturbation_mask` param is intentionally not applied** — see `fix.md` #4. The
  unconstrained-in-scaled-space attack is the thesis contrast case (proves `ASR_raw`
  balloons when domain constraints are ignored). Callers who pass a mask silently get an
  unconstrained attack — considered a UX bug, not a correctness one.
- Runners:
  - `run_attacks.py` — MLP × {binary, 8-class, 34-class}, ε ∈ {0.05, 0.10, 0.30}, plus CW.
  - `run_attacks_8class_models.py` — all 5 archs on 8-class, restart experiments.
- Outputs: `results/attacks/*.npz`, `attack_summary.csv`, restart CSVs.

**Headline numbers (`results/attacks/attack_summary.csv`):**

| Framing | Attack | ε | `ASR_raw` | Validity | `ASR_valid` |
|---|---|---:|---:|---:|---:|
| Binary | PGD | 0.30 | 37.8% | 0% | 0% |
| 8-class | PGD | 0.30 | 74.6% | 0% | 0% |
| 34-class | PGD | 0.30 | 88.3% | 0% | 0% |

Typical invalid changes (see `tables/impossible_traffic_*` exhibits): protocol becomes
non-integer or outside allowlist; binary indicators go fractional/negative; aggregate
statistics go negative; `Min > Max`; `AVG ∉ [Min, Max]`; `Variance ≠ Std²`; frozen
features move.

## 8. MixedInputBetaVAE (the core generative contribution)

Files: `src/vae/{model.py, schema.py, losses.py, dataset.py, config.py, train.py,
train_all.py, diagnostics.py}`. Retrain variants: `retrain_phase1_fix{,_v3..v6}.py`.

**Design intent** — CICIoT2023 features are heterogeneous (continuous stats, protocol IDs,
binary indicators, integer counts, cross-feature constraints). A single continuous VAE
head cannot decode a semantically valid 39-vector, so `MixedInputBetaVAE` uses **four
decoder heads** and enforces derived binaries by construction.

### 8.1 Feature partition (`schema.get_partition()`)

Per canonical 39-feature order:

- `continuous_idx` — the 23 continuous/aggregate/integer-numeric columns.
- `protocol_idx = [1]` — `Protocol Type`, treated categorically.
- `independent_binary_idx` — 11 binary indicators **not** determined by protocol
  (HTTP, HTTPS, DNS, Telnet, SMTP, SSH, IRC, DHCP, ARP, IPv, LLC).
- `derived_binary_idx` — 4 binaries **derived deterministically from decoded protocol
  argmax**: TCP(proto=6), UDP(proto=17), ICMP(proto=1), IGMP(proto=2).

`PROTOCOL_ALLOWLIST = [0, 1, 2, 6, 17, 47]` (HOPOPT, ICMP, IGMP, TCP, UDP, GRE).
IGMP (2) is present in the allowlist but was absent from training data.

### 8.2 Encoder

Input 39-dim scaled vector →

1. split into `x_continuous`, `x_ind_bin`, `x_pseudo`;
2. recover protocol index from `x[:, protocol_idx]` via nearest-neighbor to precomputed
   scaled reference values (buffer `ref_proto_scaled`) — **no CPU sync**;
3. embed protocol (`nn.Embedding(6, protocol_embed_dim=4)`);
4. concat all → MLP body `[128, 64]` → `(μ, logvar)` for `z ∈ ℝ¹⁶`;
5. `logvar` clamped to `latent_logvar_bounds = (−6.0, 6.0)`.

At `eval()` the reparameterisation returns `μ` deterministically (no noise).

### 8.3 Decoder + output adapter

Shared MLP body `[64, 128]` from `z` produces:

- `head_continuous_mu`, `head_continuous_logvar` (per-feature),
- `head_binary` — 11 logits for independent binaries,
- `head_protocol` — 6-class softmax over allowlist,
- optional `head_pseudo` (sigmoid, default 0-dim).

`decode_to_39(z, scaler, mode)`:

- `mode="soft"` — sigmoid on binaries (differentiable, used inside attack loops).
- `mode="hard"` — thresholded binaries (used for reconstruction metrics / final decode).
- Derived binaries come from `derive_binaries_from_protocol_index(argmax(head_protocol))`
  — TCP/UDP/ICMP/IGMP are always mutually consistent with the emitted protocol.
- Continuous head can be run through `_structure_continuous_raw`:
  - `use_structured_continuous_decoder=True` (default in `config.py`) — enforces
    non-negativity, TTL sigmoid×255, `Min ≤ AVG ≤ Max` via softplus gaps, integer packet
    counts via straight-through, `Variance = Std²`.
  - `use_structured_physics_decoder=True` — additionally caps `Std ≤ 0.5(Max−Min)`,
    zeros dispersion for singleton flows, and enforces `Tot sum ≈ N·AVG`, `Tot size = AVG`
    (P2/P4/P5 rules).
- Protocol reference buffers must be initialised once via
  `register_protocol_references(scaler)` before `encode()` runs — else the constructor's
  zero placeholders would embed every sample as protocol index 0 (guarded by
  `RuntimeError`).

### 8.4 β-VAE loss (`losses.compute_elbo`)

```
loss = recon_continuous
     + recon_independent_binary
     + protocol_loss_weight · CE(protocol_logits, protocol_idx)
     + β · KL(N(μ,σ²) ∥ N(0,I))       — with free_bits allowance
     + constraint_loss_weight · G1/G5/G6/G7/G8 raw-space penalties
     + physics_constraint_loss_weight · P2/P4/P5 physics penalties
     + optional raw_relative continuous L1 (with tail-focus quantile) + pseudo-binary BCE
```

Two continuous likelihoods behind `continuous_likelihood`:

- `"gaussian"` — heteroscedastic Gaussian NLL, L2-style.
- `"laplace"` — heteroscedastic Laplace NLL, L1-style, robust to Rate/IAT/Tot sum tails.

Both use the same decoder mean (`continuous_mu`), so switching only changes the
reconstruction penalty shape — see `likelihood_spec.md`.

`BetaScheduler` — linear warmup from 0 to `beta_target` over `beta_warmup_epochs`
(default 10). **Free bits** `free_bits_lambda = 0.1` gives each latent dim a KL allowance
before it is penalized — this is the anti-collapse pair with `beta_target = 0.5`.

### 8.5 Configuration (`vae/config.py`)

- 8 per-category VAEs (one per 8-class label). Sklearn alphabetical ID mapping:
  `Benign=0, BruteForce=1, DDoS=2, DoS=3, Mirai=4, Recon=5, Spoofing=6, Web=7`.
- Latent dim 16 for every class; protocol embed dim 4; encoder [128, 64] / decoder [64, 128].
- Training: Adam(lr=1e-3, wd=1e-5), batch 512, up to 200 epochs, early stop patience 10,
  grad clip 5.0, `protocol_loss_weight=2.0`, `constraint_loss_weight=0.1`,
  `physics_constraint_loss_weight=0.0` (default).
- `use_structured_continuous_decoder=True`, `structured_std_floor=0.01`.
- Default likelihood `gaussian`; `retrain_phase1_fix_v6.py` ports the laplace variant.

### 8.6 `raw_postprocess()` (schema.py)

Non-differentiable safety net applied **after** `scaler.inverse_transform` on generated
raw samples. Enforces G1 non-neg, G7 TTL clamp, G5 Min/Max swap + AVG clamp, G6 `Var=Std²`,
G8 Number = round≥1, G3 binary round. It repairs the very rules the validator checks, so
**postprocessed validity is often 100% while pre-postprocess validity is much lower** —
thesis text must distinguish "honest decoder validity" from "repaired validity"
(see P2 report §3.2.2 VAE Training Outputs).

### 8.7 Training results (`vae_run_manifest.json`, `results/vae/summary.csv`)

| Class | latent_dim | val_loss | final_kl | collapsed dims | pre-repair validity | post-repair validity | protocol acc |
|---|---:|---:|---:|---:|---:|---:|---:|
| Benign | 16 | 130.67 | 15.49 | 3 | 0.0% | 100% | 99.44% |
| BruteForce | 16 | 11.29 | 9.36 | 1 | 72.1% | 100% | 98.32% |
| DDoS | 16 | −31.93 | 16.90 | 7 | 41.3% | 100% | 99.90% |
| DoS | 16 | −36.08 | 12.86 | 8 | 0.0% | 100% | 99.99% |
| Mirai | 16 | −42.29 | 10.64 | 7 | 93.9% | 100% | 99.91% |
| Recon | 16 | −24.95 | 15.66 | 7 | 94.0% | 100% | 99.71% |
| Spoofing | 16 | −8.12 | 20.12 | 4 | 0.0% | 100% | 99.89% |
| Web | 16 | 23.07 | 9.23 | 9 | 0.0% | 100% | 99.54% |

Collapse counts are high — free-bits is the counter-measure; the `retrain_phase1_fix_v*`
scripts iterate on this. Multiple named runs live under `results/vae/<tag>/`, canonical
tag is **`gaussian_anticollapse_beta05_freebits01_20260529_173512`**.

## 9. Latent-Space Attacks

### 9.1 Shared infrastructure — `src/attack/latent_infra.py`

- `AttackRunLogger` — every run writes `outputs/latent_attacks/<phase>_<ts>_seed<N>/`
  with `config_snapshot.json` + `summary.log`. Directories are never overwritten
  (`mkdir(exist_ok=False)`).
- `PerturbationMask.from_preprocessing_artifacts()` — reads
  `data/processed/near_zero_iqr_features.json` + `netdiffuser_categorization.json`, builds
  `full_indices / partial_indices / frozen_indices` with `partial_lower=partial_upper=±0.3`.
  Methods: `apply(x_adv, x_original)` clamps partial deltas and reimposes frozen columns;
  `verify(...)` audits compliance.
- `ProtocolValidator` — scaled→raw protocol via `scaled_to_raw_protocol` (RobustScaler
  buffer trick to avoid cross-column contamination), checks allowlist membership.
- `AttackRouter` — VAE + classifier cache. Loads each checkpoint with strict-except-buffer
  policy (protocol reference buffers are allowed to be missing, everything else must
  match). Default classifier map: `mlp_8class.pt`, `cnn_8class.pt`, and (stale)
  `lightgbm_8class.pkl` — see `fix.md` #5, tree-model aliases still absent.
- `MahalanobisOutlierDetector` — per-class Mahalanobis on encoded validation μ's over
  active latent dims (collapsed dims dropped). Sweeps a ridge grid to calibrate the
  nominal 95% χ² threshold to actually-observed clean-outlier rate ≈ 0.05. Used to compute
  **IDSR** (in-distribution success rate).
- `apply_decoder_residual(...)` — critical trick: attacks compute
  `x_delta = decode(z_adv) − decode(z_orig)` and apply `x_original + x_delta` under
  mask, so decoder reconstruction bias cancels out. Then `reimpose_protocol_features`
  re-injects the original Protocol Type + TCP/UDP/ICMP/IGMP columns from the clean sample.

### 9.2 Latent PGD (`src/attack/latent_pgd.py`)

`latent_pgd_attack(vae, classifier, mask, x_original, y_true, scaler, epsilon, alpha,
num_steps, ...)`

- Encodes `z_orig = vae.encode(x_original).μ`, caches soft+hard reference decodes.
- Optimizes over `z_adv` with `L∞` projection to `[z_orig ± ε]`.
- Each inner step: `decode_to_39(soft) → apply_decoder_residual → mask/protocol reimpose
  → classifier logits → objective`, then `z_adv += α · sign(∇z L)`.
- **Objectives**:
  - `target_loss="ce"` — cross-entropy (untargeted) or `−CE` (targeted).
  - `target_loss="cw-margin"` — Carlini-Wagner style
    `−[max(max_other − target_logit + κ, 0) + λ_l2 · ‖z_adv−z_orig‖₂]`.
- **Adaptive PGD** — every `checkpoint_interval=10` steps, if best-loss hasn't improved
  for `rho·interval` steps, halve `α` down to `min_alpha=1e-4`.
- **Restarts** — supports precomputed `z_initializers` `(R, batch, latent_dim)` or random
  starts. Best-per-sample selection priority: attack success first, joint validity second,
  lower input-L2 third, higher objective last. Returns per-sample `selected_restart` and
  `best_success_mask`.
- Zero-budget shortcut: `epsilon ≤ 0` passes through the clean sample under mask.

### 9.3 Latent CW (`src/attack/latent_cw.py`)

Adam over `δ = z − z_orig` minimizing `‖x_adv − x_original‖₂ + λ_conf · max(loss_margin +
κ, 0)`. Restart seeds are clipped to the class radius before optimization; CW then
optimizes freely without a hard ε projection.

### 9.4 Restart infrastructure (`latent_restarts.py`, `latent_gmm.py`)

- `DEFAULT_NUM_RESTARTS = 5`, `DEFAULT_RESTART_STRATEGY = "encoded+jitter+gmm"` — with 5
  restarts the label pattern is `[encoded, jitter, gmm, jitter, gmm]`.
- Class-specific L∞ radii (in latent space, applies to the L∞ projection and CW init):

  | Class | ε |
  |---|---:|
  | Benign | 0.3 |
  | BruteForce | 0.3 |
  | DDoS | 0.8 |
  | DoS | 0.8 |
  | Mirai | 0.8 |
  | Recon | 0.5 |
  | Spoofing | 0.5 |
  | Web | 1.0 |

  Default PGD `alpha = 0.1 · epsilon_class`. Override via
  `--epsilon-by-class`, `--alpha-ratio`, `--cw-init-radius-by-class`.
- `LatentGMMPrior` — fits a `BayesianGaussianMixture(components=5, ...)` on **validation**
  μ's per class, cached to `outputs/latent_gmm_priors/<class>__<vae_sha>__<sample_tag>.pkl`.
  GMM sampled points serve as target-**source**-class-aware restarts. Not the classifier
  target; not trained with test labels.
- `strategy_uses_gmm(...)` gates whether a run must fit/load the priors.

### 9.5 Runners (`src/attack/`)

Canonical rerun (see `runner_guide.md`):

```powershell
python src/attack/run_all_models_attack_rerun.py `
  --vae-run-tag gaussian_anticollapse_beta05_freebits01_20260529_173512 `
  --models all --attacks all --samples-per-class 100 `
  --num-restarts 5 --restart-strategy encoded+jitter+gmm --adaptive-pgd
```

- `run_all_models_attack_rerun.py` — latent-PGD + latent-CW + input-PGD + input-CW across
  MLP, CNN, LSTM, CNN-LSTM (`serial`), DualPath. Writes `summary.csv`, `per_class.csv`,
  `per_sample_results.csv`, `all_results.json`, `summary.md`, `config.json` per run dir.
- `run_new_vae_attack_rerun.py` — latent-only, tagged VAE, results named by run tag.
- `run_targeted_benign_latent_pgd.py` — targeted (`target_class = Benign`), 5 restarts,
  full CW-margin loss, class-conditional GMM priors.
- `run_targeted_benign_new_vae_methods.py` — wrapper: runs the targeted PGD for both
  `gaussian` and `laplace` VAE run tags.
- `run_targeted_benign_kappa_sweep.py` — κ ∈ {0, 5, 10, 20, 40} sweep utility (P3 spec).
- `run_targeted_benign_latent_cw.py` — CW analogue.
- Phase runners kept for reproducibility: `run_phase0_latent_infra.py`,
  `run_phase1_sanity_checks.py`, `run_phase2_latent_pgd.py`, `run_phase3_latent_cw.py`,
  `run_phase4_input_baselines.py`, `run_phaseA_vae_failure_diagnostics.py`.
- `run_constrained_input_baselines.py` + `constrained_input_baselines.py` — apply the
  VAE's structured-decoder rules (`VAEConstraintProjection`) directly to scaled inputs, no
  VAE encoder involved. Useful ablation isolating "the constraints" from "the manifold".
- Export scripts: `export_targeted_benign_attack_samples.py`,
  `export_constrained_input_attack_samples.py`, `export_new_vae_attack_samples.py`.
- Analysis: `run_attack_statistics.py`, `statistical_analysis.py`,
  `build_thesis_bundle.py`.

### 9.6 Metric definitions (used across summaries)

| Column | Meaning |
|---|---|
| `asr_overall` | classifier misclassification rate over attacked samples |
| `asr_valid_only` | asr among jointly valid samples |
| `protocol_validity_rate` | protocol allowlist + indicator consistency passes |
| `mask_compliance_rate` | frozen features unchanged, partial ∈ ±δ |
| `raw_g1g8_validity_rate` | all 49 rules pass after inverse-transform |
| `joint_validity_rate` | `protocol_valid & mask_valid & raw_g1g8_valid` |
| `successful_joint_valid_count` | numerator of jointly-valid attacks |
| `idsr` | 1 − Mahalanobis outlier rate over source-class VAE latent |
| `mean_l2_input` / `mean_l2_latent` | perturbation magnitudes |
| `selected_restart_mean`, `restart_success_counts`, `restart_labels` | multi-start telemetry |
| `attempts_per_sample` | per-sample restart count (targeted-benign runs) |

### 9.7 Latent-vs-input headline (`all_models_rerun_20260524_185905_seed42`)

| Model | Attack | n | `asr_overall` | proto valid | mask ok | joint valid | IDSR |
|---|---|---:|---:|---:|---:|---:|---:|
| CNN | latent-PGD | 500 | 25.0% | 100% | 100% | 100% | 89.8% |
| CNN | input-PGD | 500 | 92.0% | 22.8% | 0% | 0% | n/a |
| CNN-LSTM | latent-PGD | 600 | 17.5% | 100% | 100% | 100% | 90.3% |
| CNN-LSTM | input-PGD | 600 | 95.5% | 23.3% | 0% | 0% | n/a |
| DualPath | latent-PGD | 700 | 23.6% | 100% | 100% | 100% | 79.3% |
| DualPath | input-PGD | 700 | 95.0% | 22.3% | 0% | 0% | n/a |
| LSTM | latent-PGD | 700 | 26.9% | 100% | 100% | 100% | 84.3% |
| LSTM | input-PGD | 700 | 96.7% | 28.3% | 0% | 0% | n/a |
| MLP | latent-PGD | 700 | 22.7% | 100% | 100% | 100% | 90.3% |
| MLP | input-PGD | 700 | 95.0% | 26.1% | 0% | 0% | n/a |

Restart-aware β05 rerun (`new_vae_attacks_...beta05_freebits01_...`) roughly **doubles**
latent ASR (e.g., CNN 25.0% → 32.0%; CNN-LSTM latent-CW 22.3% → 41.2%) while
`raw_g1g8_validity_rate` stays 85–96% depending on class.

Targeted-Benign latent PGD (harder problem — cross into a specific class while staying
valid) currently reports **3–6% joint target success** across models — this is the P3
improvement backlog (VAE_IMPROVEMENTS_P3.md).

McNemar test on the restart-aware run: no significant difference between latent-PGD and
latent-CW at these sample sizes (all `p > 0.05`, closest is MLP at `p = 0.061`).

## 10. Evaluation Modules

### `src/evaluation/`

- `validity_analysis.py` — filename parser, inverse-transform, `validate_batch` wrapper,
  `compute_asr_valid`, `generate_impossible_traffic_exhibit` (which rules broken + human
  description).
- `run_validity_analysis.py` — driver for input-attack `.npz` artifacts. Emits
  `results/attacks/shock_table.csv`, `violation_breakdown.csv`,
  `figures/asr_raw_vs_valid.png`, `tables/impossible_traffic_<model>_<attack>.csv`.
- `validate_full_dataset.py` — chunked validation of `X_{train,val,test}.npy` → 
  `results/validation/full_dataset_validation_report.txt`, per-rule, per-class, and 9-way
  synthetic corruption tests. Currently 100% valid across all 4,429,940 rows.
- `sample_exhibit.py` — supervisor-ready adversarial-sample tables.
- `compact_exhibits.py` — dense one-page adversarial reports (also uses raw-space validity
  + rule breakdown).
- `delta_report.py` — per-feature Δ HTML report (`--n 500 --device cuda`).
- `export_slide_exhibits.py` — slide-ready subset.
- `analyze_attack_restarts.py`, `plot_attack_restarts.py`,
  `plot_target_benign_asr_valid.py` — restart telemetry + κ-vs-valid-ASR plots.
- `compute_he_idsr.py` — high-effort IDSR computation (heaviest evaluation).
- `eda_tables.py`, `eda_figures_part1.py`, `eda_figures_part2.py` — kept here (moved from
  the old flat `src/` layout).

### `src/thesis_eval/`

- `metrics/fidelity.py` — reusable distribution-fidelity metrics.
- `fidelity_tables.py`, `realism_table.py` — per-feature Wasserstein, MMD, KS, χ² and
  PC-1 Wasserstein tables (`.trash_p2/wasserstein_perfeature.csv`,
  `.trash_p2/wasserstein_pc1.csv`, `realism_table.tex` in the trash tree are outputs).

### Root-level plot helpers

- `src/thesis_visualizations.py`, `src/plot_distributional_fidelity.py`,
  `src/plot_benign_overlay_fidelity.py` — figures under `thesis_figures/`.
- `generate_eda_figures.py` (repo root) — legacy EDA figure generator.

## 11. Directory Layout Reference

```
E:/Shameem/thesis/
├── src/
│   ├── preprocessing/          feature_groups, netdiffuser_categorization, pipeline,
│   │                            sampler (intra-class cluster undersampling helper),
│   │                            build_ciciot2023_dataset (raw-CSV → parquet)
│   ├── classifiers/            models, baseline_experiments, tree_baselines,
│   │                            review_baselines
│   ├── vae/                    schema, model, losses, dataset, config, train, train_all,
│   │                            diagnostics, latent_geometry, reconstruction_accuracy,
│   │                            physics_validator, calibrate_physics_validator,
│   │                            inspect_benign_reconstruction, analyze_collapsed_perturbable,
│   │                            experiment_runner, _rediag,
│   │                            retrain_phase1_fix{,_v3..v6}
│   ├── attack/                 adversarial_attacks, latent_{infra,pgd,cw,gmm,restarts},
│   │                            input_baselines, constrained_input_baselines,
│   │                            validator (raw-space G1–G8), run_attacks,
│   │                            run_attacks_8class_models, run_phase0..4,
│   │                            run_phaseA_vae_failure_diagnostics,
│   │                            run_all_models_attack_rerun (canonical),
│   │                            run_new_vae_attack_rerun,
│   │                            run_targeted_benign_{latent_pgd,latent_cw,
│   │                                                new_vae_methods,kappa_sweep},
│   │                            run_constrained_input_baselines, build_thesis_bundle,
│   │                            run_attack_statistics, statistical_analysis,
│   │                            export_{targeted_benign,constrained_input,new_vae}_...
│   ├── evaluation/             validity_analysis, run_validity_analysis,
│   │                            validate_full_dataset, sample_exhibit,
│   │                            compact_exhibits, delta_report, compute_he_idsr,
│   │                            analyze_attack_restarts, plot_attack_restarts,
│   │                            plot_target_benign_asr_valid, export_slide_exhibits,
│   │                            eda_tables, eda_figures_part{1,2}
│   ├── thesis_eval/            fidelity_tables, realism_table, metrics/fidelity
│   ├── thesis_visualizations.py
│   ├── plot_distributional_fidelity.py
│   └── plot_benign_overlay_fidelity.py
├── configs/
│   └── cvae.yaml               (older CVAE run config, separate from src/vae/config.py)
├── checkpoints/
│   └── cvae/<timestamped runs>/{config.yaml, metadata.json, ...}
├── data/
│   ├── ciciot2023/ciciot2023_base.csv   (10 GB raw)
│   ├── raw/                             (reserved, empty)
│   └── processed/
│       ├── raw_loaded.parquet           (4.4M rows staged)
│       ├── X_{train,val,test}.npy       float32 (N, 39)
│       ├── y_{train,val,test}.npy       int32 34-class
│       ├── y_*_cat.npy                  int32 8-class
│       ├── y_*_bin.npy                  int32 binary
│       ├── scaler.pkl                   RobustScaler (train-fit only)
│       ├── label_encoder.pkl, category_encoder.pkl
│       ├── class_names.json, category_names.json, class_to_category.json
│       ├── class_weights_{34,8,2}.npy   + class_weights_named.json (saved, NOT used in loss)
│       ├── perturbation_mask.npy        (39,) ∈ {0, 0.3, 1.0}
│       ├── content_hashes.json          SHA-256 of arrays + scaler + encoder
│       ├── {clean,processed}_data_validity_report.json
│       ├── netdiffuser_categorization.json  (21 discrete / 18 relative)
│       ├── near_zero_iqr_features.json  (freeze policy per feature)
│       ├── tsne_coords.npz, umap_coords.npz
│   └── processed_bestfeat_top24/        ANOVA F-score top-24 auxiliary set
├── models/
│   ├── {mlp,cnn,lstm,serial,dualpath}_{binary,8class,34class}.pt   (15 files)
│   └── vae/vae_class_{0..7}_{Benign,BruteForce,DDoS,DoS,Mirai,Recon,Spoofing,Web}.pt
├── results/
│   ├── all_models_all_tasks_summary.json
│   ├── attacks/                  input attack .npz + summary CSVs + exhibits
│   ├── validation/               full-dataset validation reports
│   └── vae/                      per-class diagnostics + training curves + tagged runs
├── outputs/
│   ├── latent_attacks/<phase>_<ts>_seed<N>/{summary.csv, per_class.csv,
│   │                                        per_sample_results.csv, config_snapshot.json,
│   │                                        summary.log, all_results.json, *.npz}
│   └── latent_gmm_priors/                Bayesian GMM caches per class × VAE sha × sample-cap
├── config/run_manifest.json      preprocessing reproducibility manifest
├── vae_run_manifest.json         root VAE checkpoint index (with per-class validity)
├── figures/                      EDA figures (F1..F7 + appendix)
├── thesis_figures/               final thesis figures
├── tables/                       T0..T6 + impossible_traffic + supervisor_onepager.md
├── logs/                         eda.log, class_counts.json, baselines/, preprocessing.log
└── docs (root)                   AGENTS.md, guide.md, codex.md, P2_THESIS_REPORT.md,
                                   VAE_IMPROVEMENTS_P3.md, IMPROVEMENTS.md, fix.md,
                                   likelihood_spec.md, validator_explained.md,
                                   downsampling_strategy.md, runner_guide.md,
                                   eda_preprocessing_prompt.md
```

## 12. Documentation Cross-Reference

| Document | Purpose |
|---|---|
| `AGENTS.md` | Autoloaded agent context; short repo pointer |
| `guide.md` | Original workspace map (EDA + preprocessing centric; **partly stale** — some paths reference the pre-reorg flat layout) |
| `codex.md` | Fuller repo shape after reorg; matches current subpackage layout |
| `P2_THESIS_REPORT.md` | Full Phase 2 methodology log; exhaustive; tracks classifier/VAE/attack metrics |
| `VAE_IMPROVEMENTS_P3.md` | P3 backlog for targeted-Benign latent attacks; upgrade 1–6 sequence + experiment ladder |
| `IMPROVEMENTS.md` | Defense-facing question bank + thesis-doc improvement checklist |
| `fix.md` | Code audit 2026-06-02: sys.path, import conventions, silent-mask no-op, AttackRouter aliasing, etc. |
| `likelihood_spec.md` | Terminology fix: `N(0,I)` = latent prior, Gaussian/Laplace = decoder likelihoods, GMM = restart-only empirical prior. Cite verbatim to avoid mislabeling. |
| `validator_explained.md` | Detailed G1–G8 rule documentation |
| `downsampling_strategy.md` | Intra-class cluster-based undersampling (selection variant) — the defense-ready explanation of the sampler (`src/preprocessing/sampler.py`) |
| `runner_guide.md` | Canonical rerun commands (all-model, latent-only, targeted, smoke tests, per-class ε overrides) |
| `eda_preprocessing_prompt.md` | Original EDA + preprocessing prompt spec |

## 13. Known Drift / Inconsistencies (always cite the correct source)

1. `tables/T4_clean_validity.md` shows 81.37% overall validity from an earlier validator
   pass. Current source of truth: `data/processed/processed_data_validity_report.json`
   and `results/validation/full_dataset_validation_report.txt` — **100%**.
2. `guide.md` still references the pre-reorg flat layout (`src/preprocessing.py`,
   `src/eda_tables.py`, etc.). Current paths are `src/<subpackage>/<module>.py`.
3. `CLAUDE.md` (not in repo but referenced by AGENTS.md) documents old commands. Use
   `codex.md` §"Practical Command Map" and `runner_guide.md`.
4. `logs/preprocessing.log` is empty — treat `config/run_manifest.json` and the JSON
   validation reports as authoritative.
5. `configs/cvae.yaml` targets an older CVAE (data_space raw, encoder [256,128]) unrelated
   to the current `MixedInputBetaVAE` config; do **not** confuse the two — main VAE
   defaults live in `src/vae/config.py`.
6. `AttackRouter` still ships a `lightgbm_8class.pkl` default path and no tree aliases;
   `models/lightgbm_8class.pkl` does not exist (the project uses XGBoost). fix.md #5.
7. `src/attack/adversarial_attacks.py::run_attack` validates a `perturbation_mask`
   argument's shape but **never applies it** — deliberate for the raw-vs-valid contrast,
   but silent to callers. fix.md #4.
8. Two incompatible import conventions coexist: `from src.X…` (needs repo root on
   `sys.path`, used by preprocessing/classifiers/evaluation) vs. `from vae.X…` /
   `from attack.X…` (needs `src/` on `sys.path`, used by all of `vae/`, most of
   `attack/`). Every entry-point script patches `sys.path` accordingly. fix.md #3.
9. Postprocessed VAE validity of 100% must be reported honestly next to the
   pre-postprocess number (see §8.6 and P2 report Appendix B).

## 14. Common Workflows

### 14.1 Fresh reproduction from parquet

```bash
python -m src.evaluation.eda_tables
python -m src.evaluation.eda_figures_part1
python -m src.evaluation.eda_figures_part2       # writes netdiffuser_categorization.json
python -m src.preprocessing.pipeline             # 70/10/20, scaler, mask, manifest
python -m src.classifiers.baseline_experiments   # 15 checkpoints
python src/vae/train_all.py --device cuda        # 8 VAEs + diagnostics
```

### 14.2 Canonical attack rerun

```bash
python src/attack/run_all_models_attack_rerun.py \
  --vae-run-tag gaussian_anticollapse_beta05_freebits01_20260529_173512 \
  --models all --attacks all --samples-per-class 100 \
  --num-restarts 5 --restart-strategy encoded+jitter+gmm --adaptive-pgd
```

### 14.3 Targeted-Benign (P3 backlog surface)

```bash
python src/attack/run_targeted_benign_new_vae_methods.py \
  --methods gaussian,laplace --models all --samples-per-class 100 \
  --num-restarts 5 --restart-strategy encoded+jitter+gmm --force-refit-gmm
```

### 14.4 Post-attack validity + exhibits

```bash
python -m src.evaluation.validate_full_dataset
python -m src.evaluation.run_validity_analysis      # shock_table + violation_breakdown
python -m src.evaluation.compact_exhibits --device cuda
python -m src.evaluation.sample_exhibit
python -m src.evaluation.analyze_attack_restarts
python -m src.evaluation.plot_attack_restarts
python -m src.evaluation.delta_report --n 500 --output results/delta_report.html --device cuda
python -m src.evaluation.export_slide_exhibits
```

### 14.5 Smoke test a code change (CPU, 1 sample per class)

```bash
python src/attack/run_all_models_attack_rerun.py \
  --device cpu --models mlp --attacks latent-pgd,latent-cw \
  --samples-per-class 1 --selection-batch-size 4096 \
  --num-restarts 2 --restart-strategy encoded+jitter \
  --num-steps 1 --num-iterations 1 \
  --vae-run-tag gaussian_anticollapse_beta05_freebits01_20260529_173512
```

Signal it worked: run dir contains a `summary.csv` with `restart_labels != ["encoded"]`
and `raw_g1g8_validity_rate` populated.

## 15. Mental Model for Future Work

1. `feature_groups.py` defines what a network flow **is** in this project — every other
   module reads schema/mask/category from there. Change it and everything downstream
   breaks silently unless hashes are rechecked.
2. `attack/validator.py` defines what **valid** means. Never relax it to inflate ASR.
3. `preprocessing/pipeline.py` produces the invariant: **clean processed data validates
   100%**. Attacks are the only intended source of invalidity.
4. Classifiers live entirely in scaled feature space. They are the target, not the story.
5. Input-space attacks are the pathological baseline — high `ASR_raw`, ~0% `ASR_valid`.
6. `MixedInputBetaVAE` provides a decoded feature space where four constraints
   (protocol identity, protocol-derived binaries, mask, structured continuous head)
   are enforced **by construction**, so latent attacks trade some ASR for validity.
7. The latent PGD/CW loop uses `x_original + (decode(z_adv) − decode(z_orig))` under
   mask + protocol reimpose — that is what buys 100% mask/protocol compliance without
   defeating the attack.
8. Restarts (`encoded+jitter+gmm`) + class-specific ε + adaptive PGD are the current
   levers for making latent attacks stronger without touching validity.
9. Targeted-Benign is the harder open question — the P3 doc lays out kappa sweeps,
   repair-in-the-loop, active-latent-dim restriction, target-aware restarts, and per-class
   tuning as the ranked backlog before considering a shared CVAE.
10. Two numbers must always travel together in the thesis: `ASR_raw` (or `ASR_overall`)
    and `ASR_valid` (or `successful_joint_valid_count / n`). Reporting one without the
    other loses the entire contribution.
