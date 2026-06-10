# Wasserstein and MMD Fidelity Results

These tables compare each adversarial population with the class-matched clean-malicious test reference in the model's scaled feature space. They are the NetDiffuser-comparable Realism/Fidelity metrics: 1st-order Wasserstein distance and Gaussian-kernel MMD². For both metrics, lower values mean closer to clean malicious traffic. Jensen-Shannon divergence is retained only in the appendix because it depends on histogram binning.

## Table 1 — Wasserstein (PC1)

| Attack class | PGD | CW | Latent-PGD | Latent-CW |
| --- | ---: | ---: | ---: | ---: |
| DoS | 0.856009 [0.462827, 3.711643] | 0.664421 [0.250294, 3.405598] | 13.095340 [6.785163, 19.461568] | 13.039748 [6.253714, 19.738026] |
| DDoS | 0.553621 [0.518364, 3.435330] | 0.407248 [0.368687, 3.279620] | 5.593191 [2.397625, 15.112029] | 5.565624 [2.395410, 15.504825] |
| Mirai | 0.353992 [0.428022, 3.270258] | 0.216565 [0.257859, 3.022196] | 9.081776 [8.356142, 14.431196] | 4.222997 [3.150322, 11.010708] |
| BruteForce | 2.497210 [1.830850, 3.930199] | 2.499695 [1.818618, 3.933607] | 3.945479 [2.925734, 5.168915] | 3.991558 [2.853428, 5.317559] |
| Recon | 0.424221 [0.349087, 2.023137] | 0.422183 [0.338228, 2.041037] | 3.008632 [2.121892, 4.738900] | 3.245106 [2.282198, 4.879854] |
| Web | 2.871992 [1.425961, 10.366411] | 2.785901 [1.306440, 10.282205] | 124.736666 [103.196669, 148.258522] | 148.676524 [123.004335, 174.274473] |
| Spoofing | 1.761133 [0.626584, 5.402942] | 1.696775 [0.509190, 5.769984] | 20.568295 [5.741873, 40.240762] | 17.390986 [4.185552, 34.231639] |

Cells show point estimate [bootstrap 95% CI]. PCA is fitted once per class on the full clean-malicious reference in all 39 RobustScaler-transformed features, then reused for every method.

## Table 2 — Wasserstein (mean per-feature, 39 features)

| Attack class | PGD | CW | Latent-PGD | Latent-CW |
| --- | ---: | ---: | ---: | ---: |
| DoS | 0.276799 | 0.037909 | 0.524990 | 0.527217 |
| DDoS | 0.271350 | 0.076939 | 0.372217 | 0.420811 |
| Mirai | 0.281276 | 0.161872 | 0.275830 | 0.172299 |
| BruteForce | 0.443693 | 0.213449 | 0.648629 | 0.683687 |
| Recon | 0.281522 | 0.073371 | 0.282286 | 0.287395 |
| Web | 0.432679 | 0.216665 | 4.631914 | 5.284076 |
| Spoofing | 0.298501 | 0.114665 | 0.790707 | 0.697416 |

Each cell is the arithmetic mean of the 39 parameter-free one-dimensional empirical-CDF Wasserstein distances. The complete per-feature vectors are in `wasserstein_perfeature.csv`.

## Table 3 — MMD² (full feature space)

| Attack class | PGD | CW | Latent-PGD | Latent-CW |
| --- | ---: | ---: | ---: | ---: |
| DoS | 0.009039 [-0.007436, 0.088356] | -0.000676 [-0.008407, 0.063295] | 0.068977 [0.014704, 0.177863] | 0.069422 [0.014211, 0.186465] |
| DDoS | -0.008228 [-0.007121, 0.029610] | -0.000466 [-0.006009, 0.047376] | 0.010699 [-0.004478, 0.068119] | 0.009433 [-0.003907, 0.066226] |
| Mirai | 0.034759 [-0.007252, 0.140347] | 0.002774 [-0.007651, 0.069618] | 0.044449 [0.014445, 0.122766] | -0.001264 [-0.006755, 0.055373] |
| BruteForce | -0.006038 [-0.004173, 0.018392] | -0.006548 [-0.004965, 0.017374] | 0.037357 [0.017748, 0.083905] | 0.037371 [0.018667, 0.082067] |
| Recon | -0.000058 [-0.001890, 0.032761] | -0.002868 [-0.003724, 0.026453] | -0.003738 [-0.002940, 0.020811] | -0.003662 [-0.003690, 0.026857] |
| Web | -0.004511 [-0.004774, 0.023832] | -0.004463 [-0.004688, 0.024160] | 0.493775 [0.338358, 0.686191] | 0.547875 [0.370991, 0.759990] |
| Spoofing | -0.006785 [-0.004970, 0.016030] | -0.001559 [-0.003513, 0.031311] | 0.009997 [0.001633, 0.045277] | 0.007891 [-0.000803, 0.042620] |

Cells show unbiased MMD² point estimate [bootstrap 95% CI]. The Gaussian RBF kernel is `exp(-||x-y||² / (2 sigma²))`; sigma is the pair-specific median of non-zero pooled pairwise Euclidean distances. Exact bandwidths are recorded per row in `mmd_fullspace.csv` and the run manifest. Observed sigma range: 17.328334 to 97.153659. An unbiased MMD² estimate may be slightly negative near equality.

## cInput Population Status

### Supplementary Table S1 — cInput Wasserstein (PC1)

| Attack class | cInput CPGD | cInput CCW |
| --- | ---: | ---: |
| DoS | N/A (# TODO: canonical cInput run has outcomes but no full 39-feature adversarial vectors; the eight-sample exhibit is not used) | N/A (# TODO: canonical cInput run has outcomes but no full 39-feature adversarial vectors; the eight-sample exhibit is not used) |
| DDoS | N/A (# TODO: canonical cInput run has outcomes but no full 39-feature adversarial vectors; the eight-sample exhibit is not used) | N/A (# TODO: canonical cInput run has outcomes but no full 39-feature adversarial vectors; the eight-sample exhibit is not used) |
| Mirai | N/A (# TODO: canonical cInput run has outcomes but no full 39-feature adversarial vectors; the eight-sample exhibit is not used) | N/A (# TODO: canonical cInput run has outcomes but no full 39-feature adversarial vectors; the eight-sample exhibit is not used) |
| BruteForce | N/A (# TODO: canonical cInput run has outcomes but no full 39-feature adversarial vectors; the eight-sample exhibit is not used) | N/A (# TODO: canonical cInput run has outcomes but no full 39-feature adversarial vectors; the eight-sample exhibit is not used) |
| Recon | N/A (# TODO: canonical cInput run has outcomes but no full 39-feature adversarial vectors; the eight-sample exhibit is not used) | N/A (# TODO: canonical cInput run has outcomes but no full 39-feature adversarial vectors; the eight-sample exhibit is not used) |
| Web | N/A (# TODO: canonical cInput run has outcomes but no full 39-feature adversarial vectors; the eight-sample exhibit is not used) | N/A (# TODO: canonical cInput run has outcomes but no full 39-feature adversarial vectors; the eight-sample exhibit is not used) |
| Spoofing | N/A (# TODO: canonical cInput run has outcomes but no full 39-feature adversarial vectors; the eight-sample exhibit is not used) | N/A (# TODO: canonical cInput run has outcomes but no full 39-feature adversarial vectors; the eight-sample exhibit is not used) |

### Supplementary Table S2 — cInput mean per-feature Wasserstein

| Attack class | cInput CPGD | cInput CCW |
| --- | ---: | ---: |
| DoS | N/A (# TODO: canonical cInput run has outcomes but no full 39-feature adversarial vectors; the eight-sample exhibit is not used) | N/A (# TODO: canonical cInput run has outcomes but no full 39-feature adversarial vectors; the eight-sample exhibit is not used) |
| DDoS | N/A (# TODO: canonical cInput run has outcomes but no full 39-feature adversarial vectors; the eight-sample exhibit is not used) | N/A (# TODO: canonical cInput run has outcomes but no full 39-feature adversarial vectors; the eight-sample exhibit is not used) |
| Mirai | N/A (# TODO: canonical cInput run has outcomes but no full 39-feature adversarial vectors; the eight-sample exhibit is not used) | N/A (# TODO: canonical cInput run has outcomes but no full 39-feature adversarial vectors; the eight-sample exhibit is not used) |
| BruteForce | N/A (# TODO: canonical cInput run has outcomes but no full 39-feature adversarial vectors; the eight-sample exhibit is not used) | N/A (# TODO: canonical cInput run has outcomes but no full 39-feature adversarial vectors; the eight-sample exhibit is not used) |
| Recon | N/A (# TODO: canonical cInput run has outcomes but no full 39-feature adversarial vectors; the eight-sample exhibit is not used) | N/A (# TODO: canonical cInput run has outcomes but no full 39-feature adversarial vectors; the eight-sample exhibit is not used) |
| Web | N/A (# TODO: canonical cInput run has outcomes but no full 39-feature adversarial vectors; the eight-sample exhibit is not used) | N/A (# TODO: canonical cInput run has outcomes but no full 39-feature adversarial vectors; the eight-sample exhibit is not used) |
| Spoofing | N/A (# TODO: canonical cInput run has outcomes but no full 39-feature adversarial vectors; the eight-sample exhibit is not used) | N/A (# TODO: canonical cInput run has outcomes but no full 39-feature adversarial vectors; the eight-sample exhibit is not used) |

### Supplementary Table S3 — cInput MMD²

| Attack class | cInput CPGD | cInput CCW |
| --- | ---: | ---: |
| DoS | N/A (# TODO: canonical cInput run has outcomes but no full 39-feature adversarial vectors; the eight-sample exhibit is not used) | N/A (# TODO: canonical cInput run has outcomes but no full 39-feature adversarial vectors; the eight-sample exhibit is not used) |
| DDoS | N/A (# TODO: canonical cInput run has outcomes but no full 39-feature adversarial vectors; the eight-sample exhibit is not used) | N/A (# TODO: canonical cInput run has outcomes but no full 39-feature adversarial vectors; the eight-sample exhibit is not used) |
| Mirai | N/A (# TODO: canonical cInput run has outcomes but no full 39-feature adversarial vectors; the eight-sample exhibit is not used) | N/A (# TODO: canonical cInput run has outcomes but no full 39-feature adversarial vectors; the eight-sample exhibit is not used) |
| BruteForce | N/A (# TODO: canonical cInput run has outcomes but no full 39-feature adversarial vectors; the eight-sample exhibit is not used) | N/A (# TODO: canonical cInput run has outcomes but no full 39-feature adversarial vectors; the eight-sample exhibit is not used) |
| Recon | N/A (# TODO: canonical cInput run has outcomes but no full 39-feature adversarial vectors; the eight-sample exhibit is not used) | N/A (# TODO: canonical cInput run has outcomes but no full 39-feature adversarial vectors; the eight-sample exhibit is not used) |
| Web | N/A (# TODO: canonical cInput run has outcomes but no full 39-feature adversarial vectors; the eight-sample exhibit is not used) | N/A (# TODO: canonical cInput run has outcomes but no full 39-feature adversarial vectors; the eight-sample exhibit is not used) |
| Spoofing | N/A (# TODO: canonical cInput run has outcomes but no full 39-feature adversarial vectors; the eight-sample exhibit is not used) | N/A (# TODO: canonical cInput run has outcomes but no full 39-feature adversarial vectors; the eight-sample exhibit is not used) |

The canonical cInput run is not silently omitted. Its outcome tables exist, but the run did not persist full adversarial feature vectors, so distributional distances are marked `N/A` / `# TODO` rather than estimated from the eight-sample exhibit.

## Appendix Table A1 — JS divergence (PC1)

| Attack class | PGD | CW | Latent-PGD | Latent-CW |
| --- | ---: | ---: | ---: | ---: |
| DoS | 0.001758 | 0.000778 | 0.118862 | 0.106281 |
| DDoS | 0.000291 | 0.000260 | 0.037791 | 0.037561 |
| Mirai | 0.005470 | 0.001341 | 0.505070 | 0.257772 |
| BruteForce | 0.012471 | 0.009996 | 0.013580 | 0.032295 |
| Recon | 0.000833 | 0.000766 | 0.015475 | 0.018768 |
| Web | 0.011320 | 0.010684 | 0.535377 | 0.488455 |
| Spoofing | 0.002187 | 0.000972 | 0.083018 | 0.057700 |

Supplementary only. JS divergence uses 40 equal-width bins over the pooled PC1 range and is therefore bin-sensitive.

### Appendix Table A2 — cInput JS divergence (PC1)

| Attack class | cInput CPGD | cInput CCW |
| --- | ---: | ---: |
| DoS | N/A (# TODO: canonical cInput run has outcomes but no full 39-feature adversarial vectors; the eight-sample exhibit is not used) | N/A (# TODO: canonical cInput run has outcomes but no full 39-feature adversarial vectors; the eight-sample exhibit is not used) |
| DDoS | N/A (# TODO: canonical cInput run has outcomes but no full 39-feature adversarial vectors; the eight-sample exhibit is not used) | N/A (# TODO: canonical cInput run has outcomes but no full 39-feature adversarial vectors; the eight-sample exhibit is not used) |
| Mirai | N/A (# TODO: canonical cInput run has outcomes but no full 39-feature adversarial vectors; the eight-sample exhibit is not used) | N/A (# TODO: canonical cInput run has outcomes but no full 39-feature adversarial vectors; the eight-sample exhibit is not used) |
| BruteForce | N/A (# TODO: canonical cInput run has outcomes but no full 39-feature adversarial vectors; the eight-sample exhibit is not used) | N/A (# TODO: canonical cInput run has outcomes but no full 39-feature adversarial vectors; the eight-sample exhibit is not used) |
| Recon | N/A (# TODO: canonical cInput run has outcomes but no full 39-feature adversarial vectors; the eight-sample exhibit is not used) | N/A (# TODO: canonical cInput run has outcomes but no full 39-feature adversarial vectors; the eight-sample exhibit is not used) |
| Web | N/A (# TODO: canonical cInput run has outcomes but no full 39-feature adversarial vectors; the eight-sample exhibit is not used) | N/A (# TODO: canonical cInput run has outcomes but no full 39-feature adversarial vectors; the eight-sample exhibit is not used) |
| Spoofing | N/A (# TODO: canonical cInput run has outcomes but no full 39-feature adversarial vectors; the eight-sample exhibit is not used) | N/A (# TODO: canonical cInput run has outcomes but no full 39-feature adversarial vectors; the eight-sample exhibit is not used) |

## Reading the tables

A low projection distance is not sufficient evidence of realistic traffic. Input-space PGD/CW can overlap the clean distribution on PC1 while still reaching 0% protocol-validity, so distributional closeness alone overstates realism. Read these tables alongside the validity-rate table in `results/attacks/thesis_bundle.multimetric.csv` and the Mahalanobis analysis in the `thesis_eval` realism suite.

## Run manifest

The full machine-readable manifest is `wasserstein_mmd_run_manifest.json`.

```json
{
  "artifact_hash": "57e6f8b9af16161bdb0b4741e0c6da87622515fbf8ff39cf8a67af6ae8b43df4",
  "js_bins": 40,
  "library_versions": {
    "joblib": "1.5.3",
    "numpy": "2.0.1",
    "pandas": "2.3.3",
    "python": "3.10.20",
    "scikit-learn": "1.7.2",
    "scipy": "1.15.3"
  },
  "mmd_subsample_n": 100,
  "n_boot": 1000,
  "scaling": "39-feature RobustScaler-transformed model space (data/processed/scaler.pkl); no additional feature scaling",
  "seed": 42
}
```
