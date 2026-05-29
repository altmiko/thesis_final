# VAE Training Summary

Important: treat `*_validity_pre_pct` as the honest decoder metric. `*_validity_pre_pct` is validity in raw space immediately after `scaler.inverse_transform`, before any repair. `*_validity_pct` is validity after `raw_postprocess()`, and `*_repair_pct` is the fraction of samples rescued by that repair step.

| Class | latent_dim | n_train | n_val | best_val_loss | final_kl | collapsed_dims | unconditional_validity_pre_pct | unconditional_validity_pct | unconditional_repair_pct | conditional_validity_pre_pct | conditional_validity_pct | conditional_repair_pct | protocol_accuracy_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Benign | 16 | 139992 | 19999 | -32.994794 | 19.247165 | 0 | 100.0 | 100.0 | 0.0 | 100.0 | 100.0 | 0.0 | 99.92 |
| BruteForce | 16 | 8766 | 1252 | -10.940266 | 19.160202 | 0 | 100.0 | 100.0 | 0.0 | 100.0 | 100.0 | 0.0 | 97.36 |
| DDoS | 16 | 1434940 | 204993 | -33.777347 | 10.941663 | 0 | 100.0 | 100.0 | 0.0 | 100.0 | 100.0 | 0.0 | 99.96 |
| DoS | 16 | 468144 | 66876 | -28.509051 | 9.989531 | 0 | 100.0 | 100.0 | 0.0 | 100.0 | 100.0 | 0.0 | 99.96 |
| Mirai | 16 | 419972 | 59996 | -43.191619 | 9.199627 | 0 | 100.0 | 100.0 | 0.0 | 100.0 | 100.0 | 0.0 | 99.9 |
| Recon | 16 | 352470 | 50353 | -32.756611 | 18.579984 | 0 | 100.0 | 100.0 | 0.0 | 100.0 | 100.0 | 0.0 | 99.97 |
| Spoofing | 16 | 260015 | 37145 | -30.490987 | 21.292602 | 0 | 100.0 | 100.0 | 0.0 | 100.0 | 100.0 | 0.0 | 99.85 |
| Web | 16 | 16659 | 2380 | -20.940318 | 19.573603 | 0 | 100.0 | 100.0 | 0.0 | 100.0 | 100.0 | 0.0 | 99.5 |
