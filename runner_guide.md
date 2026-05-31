# Attack Runner Guide

This guide shows the commands to rerun the tagged beta05 VAE attacks and explains what each runner does.

Run commands from the repository root:

```powershell
cd D:\thesis_final
```

## Recommended Full Rerun

This is the main command for the canonical all-model rerun. It runs latent PGD, latent CW, input PGD, and input CW across all five 8-class classifiers.

```powershell
python src\attack\run_all_models_attack_rerun.py `
  --vae-run-tag gaussian_anticollapse_beta05_freebits01_20260529_173512 `
  --models all `
  --attacks all `
  --samples-per-class 100 `
  --num-restarts 5 `
  --restart-strategy encoded+jitter+gmm `
  --adaptive-pgd
```

What it does:

- Loads the canonical beta05 Gaussian VAE run from `results\vae\gaussian_anticollapse_beta05_freebits01_20260529_173512`.
- Uses the tagged VAE manifest instead of the stale root manifest.
- Attacks all neural 8-class classifiers: MLP, CNN, LSTM, CNN-LSTM, and DualPath.
- Samples up to 100 correctly classified test examples per non-benign source class for each model.
- Runs the latent attacks with five restart seeds per sample.
- Runs the input-space attacks as a baseline comparison.
- Writes results under `outputs\latent_attacks\all_models_rerun_<timestamp>_seed42`.

The important output files are:

- `summary.csv`: one row per model and attack.
- `per_class.csv`: one row per model, attack, and source class.
- `per_sample_results.csv`: selected adversarial result for each attacked sample.
- `all_results.json`: complete machine-readable payload.
- `summary.md`: readable summary table.
- `config.json`: exact run configuration, including VAE tag, restart settings, epsilon map, and model list.

## Latent-Only Rerun

Use this when you only want the VAE-constrained attacks.

```powershell
python src\attack\run_all_models_attack_rerun.py `
  --vae-run-tag gaussian_anticollapse_beta05_freebits01_20260529_173512 `
  --models all `
  --attacks latent-pgd,latent-cw `
  --samples-per-class 100 `
  --num-restarts 5 `
  --restart-strategy encoded+jitter+gmm `
  --adaptive-pgd
```

This is usually the cleanest command for thesis comparisons because it avoids rerunning the input-space baselines.

## New-VAE Latent Runner

`run_new_vae_attack_rerun.py` is the latent-only tagged VAE runner. It does not run input PGD or input CW.

```powershell
python src\attack\run_new_vae_attack_rerun.py `
  --vae-run-tag gaussian_anticollapse_beta05_freebits01_20260529_173512 `
  --models all `
  --attacks latent-pgd,latent-cw `
  --samples-per-class 100 `
  --num-restarts 5 `
  --restart-strategy encoded+jitter+gmm `
  --adaptive-pgd
```

Use this runner when you want output directories named by the VAE run tag, for example:

```text
outputs\latent_attacks\new_vae_attacks_gaussian_anticollapse_beta05_freebits01_20260529_173512_<timestamp>_seed42
```

Use `run_all_models_attack_rerun.py` when you want latent and input attacks in the same rerun table.

## Quick Smoke Tests

Use a smoke test before a long CUDA run if you changed code or want to verify the command shape.

All-model runner, tiny CPU smoke:

```powershell
python src\attack\run_all_models_attack_rerun.py `
  --device cpu `
  --models mlp `
  --attacks latent-pgd,latent-cw `
  --samples-per-class 1 `
  --selection-batch-size 4096 `
  --num-restarts 2 `
  --restart-strategy encoded+jitter `
  --num-steps 1 `
  --num-iterations 1 `
  --vae-run-tag gaussian_anticollapse_beta05_freebits01_20260529_173512
```

New-VAE runner, tiny CPU smoke:

```powershell
python src\attack\run_new_vae_attack_rerun.py `
  --device cpu `
  --models mlp `
  --attacks latent-pgd,latent-cw `
  --samples-per-class 1 `
  --selection-batch-size 4096 `
  --num-restarts 2 `
  --restart-strategy encoded+jitter `
  --num-steps 1 `
  --num-iterations 1 `
  --vae-run-tag gaussian_anticollapse_beta05_freebits01_20260529_173512
```

A good smoke result should produce a new output directory and summary files with these columns:

```text
vae_run_tag
raw_g1g8_validity_rate
successful_joint_valid_count
selected_restart_mean
restart_success_counts
restart_labels
```

If `restart_labels` is only `["encoded"]`, the run did not use the restart-aware attack setting.

## Restart Settings

The default restart strategy is:

```text
encoded+jitter+gmm
```

With `--num-restarts 5`, the restart labels are normally:

```text
encoded, jitter, gmm, jitter, gmm
```

The restart types mean:

- `encoded`: starts from the VAE encoder mean for the original sample.
- `jitter`: starts near the encoded point by adding random latent noise.
- `gmm`: starts from a class-specific Gaussian mixture model fitted in latent space.

For GMM restarts, the runner fits or loads class priors from:

```text
latent_gmm_priors\
```

The GMM defaults are:

```text
--gmm-split val
--gmm-components 5
--gmm-fit-max-samples 50000
```

Use this if you want to refit the GMM priors instead of using cached ones:

```powershell
python src\attack\run_all_models_attack_rerun.py `
  --vae-run-tag gaussian_anticollapse_beta05_freebits01_20260529_173512 `
  --models all `
  --attacks latent-pgd,latent-cw `
  --samples-per-class 100 `
  --num-restarts 5 `
  --restart-strategy encoded+jitter+gmm `
  --force-refit-gmm
```

## Class-Specific Latent Budgets

The latent PGD and latent CW restart seeds use class-specific default radii:

```text
Benign=0.3
BruteForce=0.3
DDoS=0.8
DoS=0.8
Mirai=0.8
Recon=0.5
Spoofing=0.5
Web=1.0
```

PGD alpha defaults to:

```text
alpha = class_epsilon * 0.1
```

Override the epsilon map like this:

```powershell
python src\attack\run_all_models_attack_rerun.py `
  --vae-run-tag gaussian_anticollapse_beta05_freebits01_20260529_173512 `
  --models all `
  --attacks latent-pgd,latent-cw `
  --samples-per-class 100 `
  --epsilon-by-class "BruteForce=0.3,DDoS=0.8,DoS=0.8,Mirai=0.8,Recon=0.5,Spoofing=0.5,Web=1.0" `
  --alpha-ratio 0.1 `
  --num-restarts 5 `
  --restart-strategy encoded+jitter+gmm
```

For latent CW, restart seeds are clipped to the class radius before optimization. CW's L2 objective is then allowed to optimize without a hard PGD-style epsilon projection. Override the CW seed radius separately with:

```powershell
--cw-init-radius-by-class "DDoS=0.8,DoS=0.8,Mirai=0.8,Recon=0.5,Spoofing=0.5,Web=1.0"
```

## Adaptive PGD

Adaptive PGD is on by default. These are the defaults:

```text
--adaptive-pgd
--checkpoint-interval 10
--rho 0.75
--min-alpha 1e-4
```

It tracks the classifier objective during latent PGD. If progress stalls at checkpoints, the step size is halved down to `min_alpha`. This helps prevent PGD from bouncing around the same local region.

Disable it for an ablation:

```powershell
python src\attack\run_all_models_attack_rerun.py `
  --vae-run-tag gaussian_anticollapse_beta05_freebits01_20260529_173512 `
  --models all `
  --attacks latent-pgd `
  --samples-per-class 100 `
  --num-restarts 5 `
  --restart-strategy encoded+jitter+gmm `
  --no-adaptive-pgd
```

## Candidate Selection

For latent attacks, each restart produces a candidate adversarial sample. The runner picks the best candidate per original sample using this priority:

1. Attack success first.
2. Joint validity second.
3. Lower input-space L2 distance third.
4. Higher classifier objective last.

This is why the restart-aware runner can report better ASR than an encoded-only run: it tries several plausible latent starts and keeps the best candidate for each sample.

## Validity Metrics

The main validity columns are:

- `protocol_validity_rate`: protocol and protocol-indicator checks pass.
- `mask_compliance_rate`: immutable features remain unchanged according to the perturbation mask.
- `raw_g1g8_validity_rate`: raw-space G1-G8 physics checks pass after inverse scaling.
- `joint_validity_rate`: `protocol_valid & mask_valid & raw_g1g8_valid`.
- `successful_joint_valid_count`: number of samples that are both successful attacks and jointly valid.
- `asr_overall`: untargeted misclassification rate over all attacked samples.
- `asr_valid_only`: attack success rate among jointly valid selected samples.
- `idsr`: in-distribution success rate proxy from the latent Mahalanobis detector.

The thesis-safe attack success number is usually `asr_overall`, with validity reported beside it through `joint_validity_rate` and `successful_joint_valid_count`.

## Common Pitfall

The earlier low-ASR rerun had only encoded restarts. In `summary.csv`, that looks like:

```text
restart_labels = ["encoded"]
selected_restart_mean = 0.0
```

For the restart-aware beta05 rerun, expect restart labels like:

```text
restart_labels = ["encoded","jitter","gmm","jitter","gmm"]
```

If the labels are encoded-only, rerun with:

```powershell
--num-restarts 5 --restart-strategy encoded+jitter+gmm
```
