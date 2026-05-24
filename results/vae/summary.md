# VAE Training Summary

Important: treat `*_validity_pre_pct` as the honest decoder metric. `*_validity_pre_pct` is validity in raw space immediately after `scaler.inverse_transform`, before any repair. `*_validity_pct` is validity after `raw_postprocess()`, and `*_repair_pct` is the fraction of samples rescued by that repair step.

| Class | latent_dim | n_train | n_val | best_val_loss | final_kl | collapsed_dims | unconditional_validity_pre_pct | unconditional_validity_pct | unconditional_repair_pct | conditional_validity_pre_pct | conditional_validity_pct | conditional_repair_pct | protocol_accuracy_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Benign | 16 | 139992 | 19999 | 130.66812 | 15.490193 | 3 | 0.0 | 100.0 | 100.0 | 0.0 | 100.0 | 100.0 | 99.44 |
| BruteForce | 16 | 8766 | 1252 | 11.290511 | 9.360052 | 1 | 72.1 | 100.0 | 27.9 | 58.8 | 100.0 | 41.2 | 98.32 |
| DDoS | 16 | 1434940 | 204993 | -31.92959 | 16.89655 | 7 | 41.3 | 100.0 | 58.7 | 30.2 | 100.0 | 69.8 | 99.9 |
| DoS | 16 | 468144 | 66876 | -36.077427 | 12.857692 | 8 | 0.0 | 100.0 | 100.0 | 0.0 | 100.0 | 100.0 | 99.99 |
| Mirai | 16 | 419972 | 59996 | -42.285265 | 10.638453 | 7 | 93.9 | 100.0 | 6.1 | 58.9 | 100.0 | 41.1 | 99.91 |
| Recon | 16 | 352470 | 50353 | -24.945076 | 15.663484 | 7 | 94.0 | 100.0 | 6.0 | 87.5 | 100.0 | 12.5 | 99.71 |
| Spoofing | 16 | 260015 | 37145 | -8.122171 | 20.118106 | 4 | 0.0 | 100.0 | 100.0 | 0.0 | 100.0 | 100.0 | 99.89 |
| Web | 16 | 16659 | 2380 | 23.070889 | 9.229854 | 9 | 0.0 | 100.0 | 100.0 | 0.0 | 100.0 | 100.0 | 99.54 |
