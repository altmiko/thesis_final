# T8. Cherry-Picked Targeted Benign Latent Attack Samples

Source export: `outputs/latent_attacks/sample_exports/targeted_benign_attack_samples_20260530_002152`.

Selection rule: from the targeted-to-Benign Gaussian and Laplace VAE attack exports, keep only samples with `target_success=True` and `joint_valid=True`; for each classifier, select the two lowest scaled input-space L2 examples across both VAE priors. Latent L2 and sample id are tie-breakers. The selected VAE prior is retained in the table.

## Main Finding

The best targeted-benign examples are all valid attacks that move an attack-family sample to the `Benign` prediction while preserving protocol validity, perturbation-mask compliance, and raw G1-G8 validity. Laplace supplies the most compact overall case: LSTM flips a BruteForce sample to Benign at scaled L2 0.3830. Across the selected rows, the strongest feature changes concentrate on packet-size summaries and timing/count features, especially `Min`, `Header_Length`, `Variance`, `IAT`, `ack_count`, and `Std`.

## Selected Samples

| Model | VAE prior | Sample | Source -> After | Target success | Joint valid | Scaled L2 | Latent L2 | Top raw feature changes |
| --- | --- | ---: | --- | --- | --- | ---: | ---: | --- |
| CNN | laplace | 99 | Recon -> Benign | True | True | 0.7172 | 1.5656 | `Min` 3.4931; `IAT` 0.000335039; `Tot size` 64.3042; `AVG` 64.1677; `Header_Length` 0.923021 |
| CNN | laplace | 821 | Spoofing -> Benign | True | True | 1.1551 | 1.6502 | `Min` 5.96418; `IAT` 0.000472626; `Header_Length` 2.68718; `ack_count` 1.31466; `Max` 28.466 |
| CNN-LSTM | gaussian | 389 | Spoofing -> Benign | True | True | 0.8678 | 1.7604 | `Min` 4.33166; `Header_Length` 4.86153; `Std` -36.842; `Variance` -6426.96; `ack_count` 0.620348 |
| CNN-LSTM | gaussian | 80 | Spoofing -> Benign | True | True | 1.0814 | 1.9468 | `Min` 4.89556; `Variance` 18754.2; `Header_Length` 4.28543; `IAT` -0.00023637; `ack_count` 1.40574 |
| DualPath | laplace | 1485 | Spoofing -> Benign | True | True | 0.9631 | 1.7589 | `Min` -3.91738; `Header_Length` 6.81462; `ack_count` 2.1; `Std` -31.493; `Variance` -6271.85 |
| DualPath | gaussian | 1077 | Spoofing -> Benign | True | True | 1.3796 | 1.9468 | `IAT` 0.00096856; `Min` -4.20552; `Variance` 14813.9; `Header_Length` -3.37359; `AVG` 115.003 |
| LSTM | laplace | 86334 | BruteForce -> Benign | True | True | 0.3830 | 1.6643 | `Min` 1.9099; `Header_Length` 2.33374; `ack_count` 0.557618; `ack_flag_number` 0.0321867; `AVG` 9.37953 |
| LSTM | gaussian | 940 | Recon -> Benign | True | True | 1.3400 | 1.9059 | `Rate` -18303.6; `IAT` 0.000671636; `Tot size` 334.728; `Variance` -22309.7; `Min` 1.22102 |
| MLP | laplace | 1485 | Spoofing -> Benign | True | True | 0.9846 | 1.6874 | `Min` -4.28387; `Header_Length` 6.31994; `ack_count` 2.1; `Std` -31.7984; `Variance` -6322.96 |
| MLP | gaussian | 1077 | Spoofing -> Benign | True | True | 1.5722 | 1.6248 | `IAT` 0.0012027; `Min` -4.25676; `Header_Length` -3.75767; `Variance` -12290.1; `AVG` 102.217 |

## Featurewise Pattern

| Feature | Times appearing in selected top-5 changes | Interpretation |
| --- | ---: | --- |
| `Min` | 10 | Packet-size lower-bound movement is the most common benign-target route. |
| `Header_Length` | 9 | Header-size shifts help several Spoofing and BruteForce samples cross the decision boundary. |
| `Variance` | 7 | Traffic dispersion changes are prominent in Spoofing -> Benign flips. |
| `IAT` | 6 | Timing movement remains a reusable latent attack direction. |
| `ack_count` | 6 | TCP acknowledgement-count changes appear in compact Laplace examples. |
| `AVG` | 4 | Average packet-size movement supports several benign-target flips. |
| `Std` | 3 | Spread changes often accompany variance adjustments. |
| `Tot size` | 2 | Total-size changes appear in compact CNN and CNN-LSTM examples. |

## Thesis-Ready Paragraph

For the targeted benign case study, I selected the two lowest-L2 joint-valid successful samples per classifier across the Gaussian and Laplace VAE priors. All selected examples change the classifier prediction to `Benign` while retaining protocol, perturbation-mask, and raw-domain validity. The most compact example is a Laplace-prior LSTM attack on sample 86334, flipping `BruteForce` to `Benign` with scaled L2 0.3830. The selected perturbations are concentrated in behavioral aggregate features such as `Min`, `Header_Length`, `Variance`, `IAT`, `ack_count`, and `Std`, rather than changing immutable protocol or service indicators. This supports the interpretation that the latent attack can produce validity-preserving evasions that resemble benign traffic under the trained IDS decision boundary.

## Output Files

- `tables/T8_cherrypicked_targeted_benign_attack_samples.csv` contains the selected sample-level rows.
- `tables/T8_cherrypicked_targeted_benign_featurewise.csv` contains all 39 inverse-transformed feature rows for each selected sample.
