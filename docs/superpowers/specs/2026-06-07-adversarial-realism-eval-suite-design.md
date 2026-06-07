# Adversarial-Realism Evaluation Suite (`thesis_eval`) — Design

**Date:** 2026-06-07
**Status:** Approved design, pending spec review → writing-plans
**Owner:** Shameem Hussain (MIBVAE NIDS thesis)

## 1. Purpose & central claim

Build a single, reproducible CLI pipeline that quantifies and visualizes the
**realism, detectability, and effectiveness** of latent-space adversarial
examples produced by the per-class Mixed-Input β-VAE (MIBVAE) against five
ML-based NIDS classifiers, benchmarked against gradient baselines (PGD, CW).

The figures must support this claim:

> Latent-space attacks achieve high classifier-evasive **and** protocol-valid
> ASR (ASR_valid), remain on the data manifold (low Mahalanobis distance,
> indistinguishable from clean traffic), and generalize across classifier
> architectures — while gradient baselines achieve ~0% valid ASR and are
> trivially caught by a Mahalanobis detector.

## 2. Scope decisions (locked with user 2026-06-07)

1. **Array gap → build + run.** The canonical untargeted latent runs and the
   latent-CW runs saved only CSV outcomes (no per-sample arrays). We build an
   `export` stage that re-runs the missing latent attacks with `npz` array
   export, then populate every figure with real numbers. (We run it.)
2. **Latent family → both, side by side.** Treat *targeted-benign* and
   *untargeted source-class* latent attacks as distinct populations throughout.
3. **Canonical MIBVAE set →**
   `gaussian_anticollapse_beta05_freebits01_20260529_173512` (Gaussian
   likelihood, anti-collapse; matches the Mahalanobis Gaussian assumption).
4. **Layout →** new package `src/thesis_eval/`, outputs under
   `results/thesis_eval/`.

## 3. Discovered artifacts (ground truth, do not assume)

| Concern | Path |
| --- | --- |
| Classifiers (5 NN) | `models/{mlp,cnn,lstm,serial,dualpath}_8class.pt` (serial = CNN-LSTM) |
| Classifier factory | `src/classifiers/models.py` (`get_model`) |
| MIBVAE per-class checkpoints | `models/vae/vae_class_{0..7}_<Class>_gaussian_anticollapse_beta05_freebits01_20260529_173512.pt` |
| VAE model / encode-decode | `src/vae/model.py` (`MixedInputBetaVAE`) |
| Attack router + encode/decode + cov helpers | `src/attack/latent_infra.py` |
| Latent attacks | `src/attack/latent_pgd.py`, `latent_cw.py`, `latent_gmm.py` |
| Validator ("49-rule" = G1–G8 atomic sub-rules) | `src/attack/validator.py` (`validate_batch`) |
| Class taxonomy | `src/vae/config.py` `CLASSES = [Benign,BruteForce,DDoS,DoS,Mirai,Recon,Spoofing,Web]`; `SOURCE_CLASSES` = all but Benign |
| Data splits / scaler / mask | `data/processed/{X,y}_{train,val,test}.npy`, `scaler.pkl`, `perturbation_mask.npy`, `class_names.json` |
| Precomputed clean coords | `data/processed/{tsne,umap}_coords.npz` |
| **Gradient AE arrays** (X_adv,X_clean,y_*; no latent z) | `results/attacks/attack_8class_{pgd,cw}*.npz` |
| **Targeted-benign latent AE arrays** (full: x_orig,x_adv,z_orig,z_adv,*_valid,success,benign_margin) | `outputs/latent_attacks/kappa_sweeps/**/<model>_targeted_benign_pgd_<Class>.npz` |
| **Untargeted latent runs** (CSV outcomes only — **no arrays**) | `outputs/latent_attacks/new_vae_attacks_*/per_sample_results.csv` |
| Existing bootstrap CIs + McNemar | `outputs/latent_attacks/new_vae_attacks_*/stats/{bootstrap_summary,mcnemar_*}.csv` |
| VAE diagnostics (F13/F14 source) | `results/vae/diagnostics_<Class>.json` (`posterior_collapse.per_dim_kl`, `collapsed_dim_count`, `collapse_threshold`, `per_feature_recon`), `results/vae/reconstruction_accuracy.json` |
| VAE loss curves | `results/vae/curves_<Class>.png` (raster only — numeric per-epoch arrays may be absent) |
| Existing GMM latent priors (k=5) | `outputs/latent_gmm_priors/latent_gmm_<Class>_val_k5_*.pkl` |

## 4. Architecture — bundle-centric, two-phase

**Phase 1 — ingest/export → canonical AE bundles.** Normalize all three array
sources into one store with identical schema. **Phase 2 — pure post-processing.**
All metrics/figures read only bundles + cached intermediates; no GPU, no attack
re-runs needed to regenerate figures.

### 4.1 AE bundle schema (`results/thesis_eval/bundles/<population>__<model>__<class>.npz`)
Keys (all per-sample, aligned): `x_orig (N,39) float32`, `x_adv (N,39)`,
`z_orig (N,16)` (NaN-filled for gradient pops if not re-encodable here),
`z_adv (N,16)` (NaN for gradient until encoded in maha stage),
`y_true (N,) int64`, `y_pred_clean (N,)`, `y_pred_adv (N,)`,
`success (N,) bool` (classifier-evaded), `protocol_valid`, `mask_valid`,
`raw_g1g8_valid`, `joint_valid` (all bool). Plus scalar meta: `population`,
`model`, `source_class`, `attack_type`, `vae_tag`, `seed`.

`joint_valid` is recomputed in-suite via `validator.validate_batch` on the
**inverse-transformed** `x_adv` (raw space), never trusted blindly from source.

### 4.2 Populations registry (8)
`clean_benign`, `clean_malicious`, `latentPGD_untgt`, `latentCW_untgt`,
`latentPGD_tgtBenign`, `latentCW_tgtBenign`, `PGD`, `CW`.
Clean populations are derived per source-class from the test split.

## 5. Module layout

```
src/thesis_eval/
  __main__.py, cli.py     # --stage {all,export,attack,maha,fidelity,geometry,vae,report}
                          # --cov-mode {tied,per-class,both}  --seed 42  --n-boot 1000  --device
  config.py               # discovered paths, CANONICAL_VAE_TAG, CLASS_ORDER, POPULATIONS, PALETTE refs
  palette.py              # Okabe-Ito colorblind map (6 base colors by method); targeted/untargeted = linestyle/hatch
  manifest.py             # run_manifest.json (seed, lib versions, artifact hash, cov-mode, vae tag)
  io_utils.py             # save_figure() -> pdf + png(300dpi) + sibling data/*.csv; fail-loud path checks
  data/
    artifacts.py          # load splits, scaler, mask, feature_names, 5 classifiers, 8 per-class VAEs
    bundles.py            # AEBundle dataclass + unified loader/writer
    export_latent.py      # DUMPER: run latent PGD/CW (untgt + tgt-benign) with npz export
    adapters.py           # gradient-npz + targeted-benign kappa-npz -> AEBundle
  metrics/
    asr.py                # ASR_raw, ASR_valid, validity rate, mean/median L2; per model/attack/class
    bootstrap.py          # reusable bootstrap 95% CI (n=1000, seeded)
    mahalanobis.py        # fit mu_c, Sigma (tied + per-class); stable inverse + cond number; M(x)
    detector.py           # LR on Maha score(s); ROC + AUC per attack (with bootstrap CI)
    idsr.py               # IDSR = valid & classifier-evaded & detector-evaded
    fidelity.py           # per-feature Wasserstein, JS divergence, |corr_real-corr_gen|, NN distance
    stats_tests.py        # McNemar latent-PGD vs latent-CW
  figures/
    layer1_effectiveness.py  # F1 table(csv+tex), F2 grouped bar, F3 heatmap
    layer2_mahalanobis.py    # F4 KDE, F5 ROC, F6 scatter, F7 IDSR bar
    layer3_fidelity.py       # F8 Wasserstein, F9 PCA-KDE+JS, F10 corr heatmap
    layer4_geometry.py       # F11 t-SNE+UMAP, F12 NN-distance hist
    layer5_vae.py            # F13 KL/active-units, F14 recon by class, F15 loss curves
    report.py                # emits FIGURES.md
```

Outputs: `results/thesis_eval/{figures,data,bundles,cache}/`, `run_manifest.json`, `FIGURES.md`.

## 6. Conventions (enforced in `io_utils`/`palette`)

- Every figure → `.pdf` (vector) **and** `.png` (300 dpi); underlying numbers →
  `results/thesis_eval/data/<fig>.csv`. Figures regenerable from CSV alone.
- Colorblind-safe **Okabe-Ito** palette; one fixed color per method
  {clean_benign, clean_malicious, latent_PGD, latent_CW, PGD, CW}; targeted vs
  untargeted distinguished by linestyle/hatch/marker, never by reusing a color.
- Canonical **class order everywhere**: `[DoS, DDoS, Mirai, BruteForce, Recon,
  Web, Spoofing]`.
- All distributional/ASR/AUC numbers reported with **bootstrap 95% CI**
  (n=1000, seeded). CI width verified empirically, never hardcoded.
- Global seed set & logged; library versions + artifact hash in `run_manifest.json`.
- **No silent fallbacks:** missing checkpoint/array/covariance → raise with the
  expected path. Figures whose arrays aren't yet exported carry
  `# TODO(needs export stage)` rather than fabricated values.

## 7. Metric definitions

1. **ASR_valid** = fraction simultaneously (a) classifier-evasive and (b) valid
   per `validate_batch` on inverse-transformed `x_adv`. Also report raw ASR,
   validity rate, mean/median feature-space L2. Per attack × classifier × class.
2. **Mahalanobis score (Lee et al., 2018)** in MIBVAE latent space.
   `M(x) = min_c (z − μ_c)ᵀ Σ⁻¹ (z − μ_c)`, μ_c & Σ fit on **clean training
   latents only**. Two covariance modes (CLI flag, both produced for appendix):
   *tied* pooled-within-class Σ; *per-class* Σ_c. Numerically stable inverse:
   Ledoit-Wolf shrinkage → pseudo-inverse fallback; **condition number logged**.
   **Space realization (chosen default):** per-class VAEs do not share one latent
   space, so M is computed **inside each source class's own VAE_c latent space**,
   with the `min_c` taken over **{class-c center, benign center}** (optionally the
   5 existing GMM modes per class as additional centers). Gradient AEs (no latent
   z) are **encoded through VAE_c** so PGD/CW and latent-AE scores are comparable.
3. **Mahalanobis detector** = logistic regression on the Maha score(s),
   AE-vs-clean; ROC + AUC per attack type, with bootstrap CI. Claim direction:
   latent-AE AUC → 0.5, PGD/CW AUC → 1.0.
4. **IDSR** (numbered, citable): `IDSR = Pr[valid ∧ classifier-evaded ∧
   detector-evaded]`, the strictly harder bar than ASR_valid. "detector-evaded"
   = Maha score below operating threshold (clean 95th percentile, logged). Per
   attack × class. Computed fresh in-suite (distinct from the repo CSV `idsr`).
5. **Fidelity / diversity / originality.** Per-feature 1st-order **Wasserstein**
   (real vs generated) and **JS divergence**; **|Corr_real − Corr_gen|**;
   nearest-neighbor distance (AE→nearest real vs real→real).

## 8. Figures (narrative order preserved)

**Layer 1 — effectiveness.** F1 ASR_valid table (CSV + LaTeX), rows {PGD, CW,
latent-PGD, latent-CW}, cols {raw ASR, ASR_valid, validity rate, mean L2} with
bootstrap CIs, per class — carries "gradient = 0% valid ASR". F2 grouped bar of
ASR_valid per class (latent vs PGD/CW), Mirai≈0 callout → collapsed-latent
explanation. F3 architecture-invariance heatmap (5 classifiers × ASR_valid).

**Layer 2 — realism (core).** F4 overlaid Maha-score KDE/hist {clean_malicious,
clean_benign, valid latent-AEs, PGD/CW}. F5 detector ROC per attack + AUC table.
F6 scatter feature-L2 (x) vs Maha (y), colored by valid/invalid & attack. F7
IDSR grouped bar per class, latent vs baselines.

**Layer 3 — distributional fidelity (NetDiffuser-style).** F8 horizontal bar of
per-feature Wasserstein, sorted, one panel per class. F9 PCA-projection KDE
density grid, JS annotated per subplot. F10 |Corr_real − Corr_gen| heatmap per
class.

**Layer 4 — latent geometry.** F11 t-SNE **and** UMAP latent overlays (valid
latent-AEs inside benign cluster, PGD/CW outside) + dedicated Mirai
collapsed-region panel. F12 NN-distance histogram, AE→real vs real→real.

**Layer 5 — VAE diagnostics.** F13 per-dim KL / active-units bar (collapse
threshold marked) from `diagnostics_*.json`. F14 reconstruction fidelity by
class from `reconstruction_accuracy.json`. F15 heterogeneous β-VAE loss curves
(total/recon/KL with β-anneal schedule).

## 9. Statistical rigor

- **McNemar** latent-PGD vs latent-CW evasion outcomes; statistic + p-value
  supporting "manifold geometry, not the optimizer, is the binding constraint."
- All ROC AUCs and ASR figures get bootstrap CIs.

## 10. CLI & deliverables

- `python -m thesis_eval --stage {all|export|attack|maha|fidelity|geometry|vae|report}`
  with `--cov-mode`, `--seed`, `--n-boot`, `--device`. `export` populates
  bundles; remaining stages are pure post-processing.
- Auto-generated `FIGURES.md` mapping every metric → its figure/table.
- `run_manifest.json` (seed, lib versions, artifact hash, cov-mode, VAE tag).
- Modular: one module per metric, one per figure layer. No monolith.

## 11. Open implementation decisions (chosen defaults, revisitable)

- **D1 Maha space** — per-class VAE_c space + `min` over {class, benign} centers;
  gradient AEs encoded through VAE_c. (Section 7.2.)
- **D2 F15 loss curves** — if numeric per-epoch loss arrays were not saved (only
  `curves_*.png`), F15 is marked `# TODO` pending a short re-log step in `export`.
- **D3 Sample sizes** — match the existing 100-correctly-classified-per-class
  convention for latent runs; subsample gradient AEs to the same per-class N for
  comparable CIs.

## 12. Out of scope

34-class / binary tasks (suite targets the 8-class models). No re-preprocessing,
no SMOTE, no VAE retraining. Tree models (RF/XGB) excluded — claim is about NN
architecture invariance.
