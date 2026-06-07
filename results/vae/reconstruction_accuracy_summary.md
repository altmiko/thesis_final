# VAE Reconstruction Accuracy

Deterministic posterior-mean reconstruction (`z = mu`, no sampling) on the validation split. Continuous features compared in raw space; binary/protocol compared as exact categorical matches.

| class_name | n_val | continuous_median_r2 | continuous_frac_r2_ge_0p5 | continuous_median_nrmse | binary_mean_accuracy | protocol_accuracy | categorical_exact_match |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Benign | 19999 | 0.9886 | 0.6316 | 0.1067 | 0.9931 | 0.9992 | 0.9030 |
| BruteForce | 1252 | 0.8462 | 0.6000 | 0.3910 | 0.9763 | 0.9736 | 0.6909 |
| DDoS | 204993 | 0.9954 | 0.6190 | 0.0678 | 0.9991 | 0.9996 | 0.9866 |
| DoS | 66876 | -0.0251 | 0.3333 | 1.0125 | 0.9971 | 0.9996 | 0.9569 |
| Mirai | 59996 | 0.9156 | 0.7222 | 0.2903 | 0.9999 | 0.9990 | 0.9982 |
| Recon | 50353 | 0.9967 | 0.6190 | 0.0578 | 0.9984 | 0.9997 | 0.9768 |
| Spoofing | 37145 | 0.9918 | 0.5238 | 0.0903 | 0.9980 | 0.9985 | 0.9729 |
| Web | 2380 | 0.9388 | 0.5789 | 0.2474 | 0.9841 | 0.9950 | 0.7945 |
| OVERALL (weighted) | 442994 | 0.8293 | 0.5822 | 0.2450 | 0.9983 | 0.9993 | 0.9758 |

- **continuous_median_r2**: median coefficient of determination across non-degenerate continuous features (1.0 = perfect). Median, not mean, because a few near-constant features (tiny variance) give hugely negative R^2 that dominates the mean — see `continuous_mean_r2` in the JSON for the raw mean.
- **continuous_frac_r2_ge_0p5**: fraction of continuous features reconstructed with R^2 >= 0.5.
- **continuous_median_nrmse**: median RMSE normalised by each feature's std (lower is better).
- **binary_mean_accuracy**: mean exact-bit accuracy over 11 independent + 4 derived binaries.
- **protocol_accuracy**: protocol-type top-1 reconstruction accuracy.
- **categorical_exact_match**: fraction of samples with all 11 independent bits AND protocol correct.