# VAE Reconstruction Accuracy

Deterministic posterior-mean reconstruction (`z = mu`, no sampling) on the validation split. Continuous features compared in raw space; binary/protocol compared as exact categorical matches.

| class_name | n_val | continuous_median_r2 | continuous_frac_r2_ge_0p5 | continuous_median_nrmse | binary_mean_accuracy | protocol_accuracy | categorical_exact_match |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Benign | 19999 | -0.0188 | 0.3158 | 1.0094 | 0.9933 | 0.9944 | 0.9112 |
| BruteForce | 1252 | 0.1139 | 0.4500 | 0.9393 | 0.9786 | 0.9832 | 0.7093 |
| DDoS | 204993 | 0.9227 | 0.7143 | 0.2780 | 0.9993 | 0.9990 | 0.9905 |
| DoS | 66876 | -0.0396 | 0.2857 | 1.0196 | 0.9991 | 0.9999 | 0.9871 |
| Mirai | 59996 | 0.7162 | 0.6111 | 0.5324 | 0.9998 | 0.9991 | 0.9980 |
| Recon | 50353 | 0.8901 | 0.5714 | 0.3315 | 0.9975 | 0.9971 | 0.9684 |
| Spoofing | 37145 | -0.0005 | 0.3810 | 1.0003 | 0.9976 | 0.9989 | 0.9666 |
| Web | 2380 | -0.0004 | 0.3684 | 1.0002 | 0.9851 | 0.9954 | 0.7987 |
| OVERALL (weighted) | 442994 | 0.6186 | 0.5708 | 0.5298 | 0.9986 | 0.9986 | 0.9811 |

- **continuous_median_r2**: median coefficient of determination across non-degenerate continuous features (1.0 = perfect). Median, not mean, because a few near-constant features (tiny variance) give hugely negative R^2 that dominates the mean — see `continuous_mean_r2` in the JSON for the raw mean.
- **continuous_frac_r2_ge_0p5**: fraction of continuous features reconstructed with R^2 >= 0.5.
- **continuous_median_nrmse**: median RMSE normalised by each feature's std (lower is better).
- **binary_mean_accuracy**: mean exact-bit accuracy over 11 independent + 4 derived binaries.
- **protocol_accuracy**: protocol-type top-1 reconstruction accuracy.
- **categorical_exact_match**: fraction of samples with all 11 independent bits AND protocol correct.