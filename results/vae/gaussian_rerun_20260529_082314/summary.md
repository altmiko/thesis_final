# VAE Training Summary

Important: treat `*_validity_pre_pct` as the honest decoder metric. `*_validity_pre_pct` is validity in raw space immediately after `scaler.inverse_transform`, before any repair. `*_validity_pct` is validity after `raw_postprocess()`, and `*_repair_pct` is the fraction of samples rescued by that repair step.

| Class | latent_dim | n_train | n_val | best_val_loss | final_kl | collapsed_dims | unconditional_validity_pre_pct | unconditional_validity_pct | unconditional_repair_pct | conditional_validity_pre_pct | conditional_validity_pct | conditional_repair_pct | protocol_accuracy_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Benign | 16 | 139992 | 19999 | -24.49935 | 11.766454 | 10 | 100.0 | 100.0 | 0.0 | 100.0 | 100.0 | 0.0 | 99.72 |
| BruteForce | 16 | 8766 | 1252 | -11.260128 | 11.256556 | 2 | 100.0 | 100.0 | 0.0 | 100.0 | 100.0 | 0.0 | 98.32 |
| DDoS | 16 | 1434940 | 204993 | -33.750586 | 6.914749 | 9 | 100.0 | 100.0 | 0.0 | 100.0 | 100.0 | 0.0 | 99.98 |
| DoS | 16 | 468144 | 66876 | -25.159345 | 5.608587 | 11 | 100.0 | 100.0 | 0.0 | 100.0 | 100.0 | 0.0 | 99.97 |
| Mirai | 16 | 419972 | 59996 | -41.332634 | 5.704934 | 10 | 100.0 | 100.0 | 0.0 | 100.0 | 100.0 | 0.0 | 99.95 |
| Recon | 16 | 352470 | 50353 | -21.293546 | 11.88818 | 9 | 100.0 | 100.0 | 0.0 | 100.0 | 100.0 | 0.0 | 99.65 |
| Spoofing | 16 | 260015 | 37145 | -23.624822 | 12.287034 | 10 | 100.0 | 100.0 | 0.0 | 100.0 | 100.0 | 0.0 | 99.92 |
| Web | 16 | 16659 | 2380 | -15.101175 | 12.04826 | 6 | 100.0 | 100.0 | 0.0 | 100.0 | 100.0 | 0.0 | 99.75 |
