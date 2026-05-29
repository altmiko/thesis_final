# VAE Training Summary

Important: treat `*_validity_pre_pct` as the honest decoder metric. `*_validity_pre_pct` is validity in raw space immediately after `scaler.inverse_transform`, before any repair. `*_validity_pct` is validity after `raw_postprocess()`, and `*_repair_pct` is the fraction of samples rescued by that repair step.

| Class | latent_dim | n_train | n_val | best_val_loss | final_kl | collapsed_dims | unconditional_validity_pre_pct | unconditional_validity_pct | unconditional_repair_pct | conditional_validity_pre_pct | conditional_validity_pct | conditional_repair_pct | protocol_accuracy_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Benign | 16 | 139992 | 19999 | -77.579765 | 18.657723 | 0 | 100.0 | 100.0 | 0.0 | 100.0 | 100.0 | 0.0 | 99.77 |
| BruteForce | 16 | 8766 | 1252 | -51.751846 | 14.355189 | 0 | 100.0 | 100.0 | 0.0 | 100.0 | 100.0 | 0.0 | 97.04 |
| DDoS | 16 | 1434940 | 204993 | -94.985857 | 13.164973 | 0 | 100.0 | 100.0 | 0.0 | 100.0 | 100.0 | 0.0 | 99.96 |
| DoS | 16 | 468144 | 66876 | -95.110155 | 12.33546 | 0 | 100.0 | 100.0 | 0.0 | 100.0 | 100.0 | 0.0 | 99.97 |
| Mirai | 16 | 419972 | 59996 | -103.135416 | 13.773723 | 0 | 100.0 | 100.0 | 0.0 | 100.0 | 100.0 | 0.0 | 99.91 |
| Recon | 16 | 352470 | 50353 | -93.057014 | 23.882347 | 0 | 100.0 | 100.0 | 0.0 | 100.0 | 100.0 | 0.0 | 99.79 |
| Spoofing | 16 | 260015 | 37145 | -74.96384 | 19.310262 | 0 | 100.0 | 100.0 | 0.0 | 100.0 | 100.0 | 0.0 | 99.86 |
| Web | 16 | 16659 | 2380 | -56.210917 | 13.619513 | 0 | 100.0 | 100.0 | 0.0 | 100.0 | 100.0 | 0.0 | 97.94 |
