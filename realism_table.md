# Unified Realism Table

Class-grouped method sub-rows are used for readability. This table is the source of truth for chapter prose and on-figure annotations; downstream consumers should read `realism_table.csv` rather than recompute its metrics.

| Attack class | Method | W (PC1), 95% CI | W (mean per-feature) | Validity % | Mahalanobis^2 (mean) |
| --- | --- | ---: | ---: | ---: | ---: |
| DoS | PGD | 0.856 (0.463-3.712) | 0.277 | 0.00 | 48.638 |
|  | CW | 0.664 (0.250-3.406) | 0.038 | 47.69 | 10.117 |
|  | Latent-PGD | 13.095 (6.785-19.462) | 0.525 | 100.00 | 118.018 |
|  | Latent-CW | 13.040 (6.254-19.738) | 0.527 | 100.00 | 14.698 |
| DDoS | PGD | 0.554 (0.518-3.435) | 0.271 | 0.00 | 100.290 |
|  | CW | 0.407 (0.369-3.280) | 0.077 | 43.95 | 23.545 |
|  | Latent-PGD | 5.593 (2.398-15.112) | 0.372 | 100.00 | 1673.429 |
|  | Latent-CW | 5.566 (2.395-15.505) | 0.421 | 100.00 | 15.362 |
| Mirai† | PGD | 0.354 (0.428-3.270) | 0.281 | 0.00 | 115.430 |
|  | CW | 0.217 (0.258-3.022) | 0.162 | 0.75 | 25.708 |
|  | Latent-PGD | 9.082 (8.356-14.431) | 0.276 | 100.00 | 2464.059 |
|  | Latent-CW | 4.223 (3.150-11.011) | 0.172 | 100.00 | 9.654 |
| BruteForce | PGD | 2.497 (1.831-3.930) | 0.444 | 0.00 | 26.610 |
|  | CW | 2.500 (1.819-3.934) | 0.213 | 86.52 | 14.168 |
|  | Latent-PGD | 3.945 (2.926-5.169) | 0.649 | 100.00 | 66.271 |
|  | Latent-CW | 3.992 (2.853-5.318) | 0.684 | 100.00 | 26.941 |
| Recon | PGD | 0.424 (0.349-2.023) | 0.282 | 0.00 | 69.999 |
|  | CW | 0.422 (0.338-2.041) | 0.073 | 24.71 | 17.918 |
|  | Latent-PGD | 3.009 (2.122-4.739) | 0.282 | 100.00 | 602.366 |
|  | Latent-CW | 3.245 (2.282-4.880) | 0.287 | 100.00 | 8.757 |
| Web‡ | PGD | 2.872 (1.426-10.366) | 0.433 | 0.00 | 33.702 |
|  | CW | 2.786 (1.306-10.282) | 0.217 | 96.28 | 15.471 |
|  | Latent-PGD | 124.737 (103.197-148.259) | 4.632 | 100.00 | 584.710 |
|  | Latent-CW | 148.677 (123.004-174.274) | 5.284 | 100.00 | 40.257 |
| Spoofing | PGD | 1.761 (0.627-5.403) | 0.299 | 0.00 | 47.191 |
|  | CW | 1.697 (0.509-5.770) | 0.115 | 20.73 | 17.087 |
|  | Latent-PGD | 20.568 (5.742-40.241) | 0.791 | 100.00 | 192.286 |
|  | Latent-CW | 17.391 (4.186-34.232) | 0.697 | 100.00 | 11.215 |

**Footnotes**

1. **W (PC1)** reports the persisted point estimate with its bootstrap 95% CI. **W (mean per-feature)** is the arithmetic mean of the 39 persisted feature distances; the complete vector remains in `wasserstein_perfeature.csv`.
2. **Validity %** is class-resolved: each cell is the mean of the persisted `protocol_valid` vector in the exact bundle used for that method and class. Input attacks use the shared bundle and latent attacks use the MLP bundle. The older method-level multimetric CSV is retained only as an aggregate audit source.
3. **Mahalanobis^2 (mean)** is the mean persisted squared Mahalanobis score, minimized over the clean-malicious and clean-Benign centers. A single tied covariance is fitted per source class and shared by all four methods. It now uses always-on Ledoit-Wolf shrinkage; empirical and regularized condition numbers are recorded in `realism_table.csv` and `F6_covariance_diagnostics.csv`. The `shared`/`mlp` labels describe bundle provenance, not different covariance models.
4. † **Mirai non-realism case:** collapsed-latent-dimension pathology. ‡ **Web non-realism case:** out-of-distribution generation, including unusually large latent mean-per-feature Wasserstein distances. These rows must not be read as healthy novelty.
5. A large latent Wasserstein distance is healthy novelty only when validity is high **and** squared Mahalanobis remains on-manifold. For Mirai/Web, large distance instead indicates collapse/OOD.
6. All class-specific latent populations used for Wasserstein and Mahalanobis are n=100. Bootstrap CIs are therefore wide; cross-class differences within overlapping CIs should not be over-interpreted.
7. MMD and JS are appendix-only: MMD uses n=100 and is noisy, so it is not suitable for ranking; JS is bin-sensitive. See `Wasserstein_MMD_results.md`.

**Interpretation guard:** shrinkage substantially reduces the largest Latent-PGD scores, especially Recon, but does not remove the separation from Latent-CW. The evidence therefore supports an artifact-amplified, not artifact-only, Latent-PGD off-manifold effect.
