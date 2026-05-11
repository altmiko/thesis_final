# VAE-Based Constrained Adversarial Attack — Phase Plan

## Discrepancies Found and Resolved (Phase 0)

| # | Issue | Prompt | Reality | Resolution |
|---|-------|--------|---------|------------|
| D1 | Feature count | 37 | **39** (Modified Schema A) | **RESOLVED**: Use 39 features. CVAE encoder = 39-dim. |
| D2 | Classifier naming | `mlp_3l`, `cnn_1d`, `lightgbm` | `mlp`, `cnn`, `lstm`, `serial`, `dualpath` — no LightGBM | **RESOLVED**: Use all 5 differentiable classifiers. Each is an independent attack target; 5×5 transferability matrix. |
| D3 | Binary accuracy | ≥ 97% | Best: dualpath 96.78% | **RESOLVED**: Proceed as-is. |
| D4 | 8-class accuracy | ≥ 95% | Best: dualpath 84.59% | **RESOLVED**: Proceed as-is. Classifiers are adequate attack targets. |
| D5 | Perturbation mask source | N/A | Code (binary 0/1) vs disk (three-tier 0.0/0.3/1.0) | **RESOLVED**: On-disk `perturbation_mask.npy` is authoritative (11 full / 9 partial / 19 frozen). |
| D6 | Three-tier sampling | "When manifest specifies" | No tier definitions in manifest | **RESOLVED**: Skip tiered sampling. Use all available attack samples. |
| D7 | Directory naming | `configs/`, `checkpoints/`, `baselines/` | `config/`, `models/`, no `baselines/` | **RESOLVED**: Use actual paths on disk. |

---

## Phase 0 — Environment Discovery and Branching

**Goal**: Verify the locked preprocessing state, inventory classifier checkpoints, confirm feature ordering and perturbation mask, report discrepancies.

### Steps
1. Read `config/run_manifest.json` — extract content hashes, schema, feature counts.
2. Verify content hashes by re-hashing `X_train.npy`, `X_val.npy`, `X_test.npy`, `scaler.pkl`, `label_encoder.pkl` using the same method as `preprocessing.py` (raw file bytes for .npy, scaler center+scale bytes, encoder classes bytes).
3. Inventory model checkpoints in `models/` — list all `.pt` files, map to architectures and tasks.
4. Extract the 39-feature ordering from `feature_groups.py:FEATURE_NAMES`.
5. Read `data/processed/perturbation_mask.npy` — report the three-tier breakdown (full/partial/frozen) with feature names.
6. Flag the code vs disk mask discrepancy.
7. Report classifier accuracy from `results/all_models_all_tasks_summary.json`.
8. Print one-page environment summary. **Stop and ask for confirmation on all discrepancies** before proceeding.

### Decision required from user
- Accept 39 features (not 37)?
- Which classifiers to use as attack targets (given no LightGBM, and 8-class accuracy ≈ 84.5%)?
- Which perturbation mask is authoritative — on-disk three-tier or code binary?
- Provide three-tier sampling tier definitions, or skip tiered sampling?

---

## Phase 1 — Mixed-Input Conditional VAE (CVAE) Architecture

**Goal**: Build and train a class-conditional VAE modeling the joint distribution of 7 attack classes.

### Files to create
- `src/vae/cvae.py` — CVAE model (encoder, decoder with continuous + binary heads)
- `src/vae/train.py` — training loop with β-annealing, checkpointing, early stopping
- `src/vae/dataset.py` — dataset wrapper emitting (x, c) pairs using feature-type metadata
- `configs/cvae.yaml` — all hyperparameters (latent_dim, lr, batch_size, β schedule, layer sizes)

### Architecture summary (39 features confirmed)
- **Encoder**: [x (39-dim), c (7-dim one-hot)] → MLP → (μ, log σ²), latent z ∈ ℝ^16
- **Decoder**: [z, c] → MLP → two heads:
  - Continuous head (Gaussian NLL) for flow statistics
  - Binary head (BCE with sigmoid) for protocol indicators
  - Decision per-column: if >5% of values are non-{0,1}, use continuous head with sigmoid squash
- **Loss**: ELBO = Gaussian_NLL + BCE + β·KL, with β linearly annealed 0→1 over first 30% of training
- **Optimizer**: AdamW, lr=1e-3, cosine decay, batch=512
- **Early stopping**: patience 10 on val ELBO
- **Data scope**: attack samples only (7 classes), existing train/val/test splits

### Deliverables
- Best checkpoint at `checkpoints/cvae/{run_id}/best.pt`
- Training report: ELBO decomposition, KL per latent dim, per-class reconstruction error

---

## Phase 2 — Attack Generation: Latent-Space Search for Evasion

**Goal**: Implement constrained adversarial attack via gradient-based latent space search through the trained CVAE.

### Files to create
- `src/attack/latent_search.py` — iterative latent-space attack (K=50 steps, Adam on z)
- `src/attack/constraints.py` — perturbation mask enforcement, protocol projection, binary rounding, range clamps
- `src/attack/generate.py` — top-level script: load CVAE + classifier(s), iterate test pool, save outputs
- `tests/test_constraints.py` — unit tests for constraint enforcement

### Algorithm (per sample x_src of source class c_src → target = benign)
1. Encode x_src → μ_src; initialize z = μ_src
2. For k = 1..K:
   - Decode (z, c_src) → x̂
   - Apply perturbation mask: frozen → copy from x_src; partial → clip to x_src ± δ; full → unconstrained
   - Project protocol field to {0,1,2,6,17,47}; reject + resample if too far
   - Round true-binary indicators to {0,1}
   - Clamp non-negative features
   - Score with classifier; compute L = -log p(benign | x̂)
   - Backprop through decoder to get ∂L/∂z; Adam step on z
   - Early stop if classifier predicts benign with margin ≥ τ and all constraints pass
3. Return final x̂ + success flag

### Multi-classifier evaluation (all 5 differentiable)
- Each of the 5 classifiers (mlp, cnn, lstm, serial, dualpath) is an independent attack target
- For each attack target, evaluate transferability by scoring adversarial samples on all 4 remaining classifiers
- Result: 5×5 transferability matrix per source class (25 ESR values per class)

### Outputs
- `outputs/adversarial/{run_id}/{src_class}.parquet`

---

## Phase 3 — Evaluation and Reporting

**Goal**: Comprehensive evaluation comparing CVAE-based attack to gradient baselines.

### Metrics per (source class → benign)
1. **Evasion Success Rate (ESR)**: fraction classified as benign per classifier
2. **Constraint Validity Rate**: fraction passing full validator chain (should be 100%)
3. **Perturbation Budget**: L2 and L∞ on Full/Partial subsets; Frozen subset sanity check (should be 0)
4. **Manifold Fidelity**: CVAE reconstruction error of adversarial vs clean samples
5. **Transferability Matrix**: ESR across classifier pairs (if >1 classifier)
6. **Gradient Baseline Comparison**: side-by-side ESR + validity if `baselines/pgd/` and `baselines/cw/` outputs exist

### Outputs
- `reports/phase2_evaluation_{run_id}.md` — tables in markdown
- `reports/phase2_evaluation_{run_id}.json` — raw numbers

---

## Working Style Invariants

- One phase at a time; plan → show → implement
- Show diffs before modifying existing files
- Content-hash every checkpoint, generated dataset, config into `run_manifest.json`
- Structured changelog at end of each phase
- Flag inconsistencies immediately — never silently reconcile
- Cite paper anchors in code comments (VAE-TabAttack, NetDiffuser, SAAE, Grini 2025)
