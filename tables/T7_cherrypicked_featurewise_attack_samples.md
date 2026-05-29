# T7. Cherry-Picked Featurewise Latent Attack Samples

Source export: `outputs/latent_attacks/sample_exports/new_vae_attack_samples_20260529_234415`.

Selection rule: the export provides two samples for each VAE run, classifier, and latent attack. For this thesis table, the Gaussian and Laplace VAE reruns were treated as candidate generators, and the best two samples were selected for each classifier x attack method. All 40 candidates were regenerated successes and all were joint-valid; therefore the ranking criterion was smallest scaled input-space L2, with latent L2 used as the tie-breaker. The selected VAE prior is retained in the table.

## Main Finding

The strongest featurewise exhibits show that latent-space attacks can flip the IDS prediction while preserving protocol validity and perturbation-mask compliance. The most economical examples are concentrated in the Laplace rerun for CNN and CNN-LSTM, where DDoS samples are flipped to DoS with very small scaled L2 distances. For MLP, LSTM, and DualPath, the selected cases mostly originate from BruteForce and move to DDoS, DoS, Spoofing, or Mirai while keeping protocol/service indicators mostly stable.

Across the selected attacks, the dominant changes are not arbitrary protocol flips. They concentrate on timing and aggregate traffic statistics, especially `IAT`, `Header_Length`, `Min`, `Std`, and `Rate`. This supports the thesis claim that the latent attack space produces more coherent adversarial traffic than unconstrained input-space perturbations.

## Selected Samples

Top feature changes are reported as raw deltas. Features were chosen by absolute scaled delta, so a small raw timing change can appear when it is large relative to that feature's scale.

| Model | Attack | VAE prior | Sample | Source -> After | Scaled L2 | Latent L2 | Top feature changes |
| --- | --- | --- | ---: | --- | ---: | ---: | --- |
| CNN | latent-CW | laplace | 81 | DDoS -> DoS | 0.2885 | 0.5618 | `Rate` -7139; `IAT` +6.17e-06; `Header_Length` -0.00103; `Tot size` +0.00762 |
| CNN | latent-CW | laplace | 7 | DDoS -> DoS | 0.3001 | 0.0104 | `syn_count` -0.300; `Header_Length` -0.0787; `Max` +1.18; `Tot sum` -9.58 |
| CNN | latent-PGD | laplace | 150 | DDoS -> DoS | 0.3167 | 2.0000 | `Rate` -7549; `Header_Length` +0.807; `Max` +53.2; `IAT` +8.12e-06 |
| CNN | latent-PGD | laplace | 81 | DDoS -> DoS | 0.3184 | 2.0000 | `Rate` -7596; `Header_Length` +0.804; `Max` +53.0; `IAT` +7.01e-06 |
| CNN-LSTM | latent-CW | laplace | 150 | DDoS -> DoS | 0.0788 | 0.1579 | `Rate` -1950; `IAT` +1.52e-06; `Tot size` -0.0106; `Header_Length` +0.000125 |
| CNN-LSTM | latent-CW | laplace | 81 | DDoS -> DoS | 0.1485 | 0.2925 | `Rate` -3674; `IAT` +2.82e-06; `Tot size` -0.0233; `Header_Length` -0.000425 |
| CNN-LSTM | latent-PGD | laplace | 150 | DDoS -> DoS | 0.2649 | 1.5958 | `Rate` -6556; `IAT` +6.62e-06; `Header_Length` -0.00481; `Tot sum` +9.47 |
| CNN-LSTM | latent-PGD | laplace | 81 | DDoS -> DoS | 0.2683 | 1.7988 | `Rate` -6639; `IAT` +6.40e-06; `Header_Length` -0.0179; `Tot size` +0.0499 |
| DualPath | latent-CW | gaussian | 20502 | BruteForce -> DDoS | 0.5689 | 0.4630 | `Min` -2.94; `IAT` +0.000238; `Std` +17.6; `ack_count` +0.483 |
| DualPath | latent-CW | laplace | 25476 | BruteForce -> DDoS | 2.4270 | 1.1678 | `IAT` -0.00222; `Min` -1.87; `Std` -37.0; `ack_count` -0.919 |
| DualPath | latent-PGD | laplace | 25476 | BruteForce -> DDoS | 2.7514 | 1.7765 | `IAT` -0.00253; `Min` -1.59; `Std` -37.8; `ack_count` -0.765 |
| DualPath | latent-PGD | gaussian | 20502 | BruteForce -> Mirai | 5.4012 | 1.9491 | `IAT` +0.00487; `Min` -6.99; `Std` +58.1; `ack_count` +1.82 |
| LSTM | latent-CW | laplace | 25476 | BruteForce -> DoS | 2.2564 | 1.0511 | `IAT` -0.00206; `Min` -1.77; `Std` -30.8; `ack_count` -0.929 |
| LSTM | latent-CW | gaussian | 11336 | BruteForce -> DoS | 3.7413 | 0.6433 | `IAT` -0.00345; `syn_count` -0.238; `ack_flag_number` -0.145; `rst_count` -0.124 |
| LSTM | latent-PGD | gaussian | 20502 | BruteForce -> Mirai | 2.0609 | 1.9653 | `IAT` -0.00138; `Min` -8.03; `Std` +68.1; `Header_Length` -3.33 |
| LSTM | latent-PGD | laplace | 25476 | BruteForce -> DoS | 2.7854 | 1.8423 | `IAT` -0.00256; `Min` -1.72; `Std` -37.2; `ack_count` -0.842 |
| MLP | latent-CW | gaussian | 2195 | BruteForce -> DDoS | 3.4556 | 1.2909 | `IAT` +0.00312; `Min` -3.27; `Variance` -14750; `Std` -64.8 |
| MLP | latent-CW | gaussian | 24321 | BruteForce -> DDoS | 11.3310 | 1.1943 | `IAT` +0.0105; `Min` -2.65; `rst_count` +0.300; `syn_count` +0.300 |
| MLP | latent-PGD | laplace | 34409 | BruteForce -> Spoofing | 0.6248 | 1.8302 | `IAT` -0.000381; `Min` -2.25; `Header_Length` -2.73; `ack_count` -0.997 |
| MLP | latent-PGD | gaussian | 20502 | BruteForce -> Spoofing | 1.8696 | 1.9235 | `Min` -8.07; `IAT` -0.00111; `Std` +67.1; `Header_Length` -3.17 |

## Featurewise Pattern

| Feature | Times appearing in selected top-5 changes | Interpretation |
| --- | ---: | --- |
| `IAT` | 19 | Timing is the most reusable route for valid latent movement. |
| `Header_Length` | 14 | Header-size shifts often accompany DDoS -> DoS flips. |
| `Min` | 13 | Packet-size lower-bound changes are common in BruteForce flips. |
| `Std` | 10 | Dispersion changes help move BruteForce samples toward high-volume attack labels. |
| `Rate` | 7 | Rate reductions dominate the CNN and CNN-LSTM DDoS -> DoS examples. |
| `Max` | 7 | Upper packet-size changes appear mainly in DDoS examples. |
| `ack_count` | 7 | Count-level TCP behavior changes support BruteForce reclassification. |
| `Tot size` | 6 | Total-size adjustments appear in the lowest-L2 CNN-LSTM cases. |

## Model-Level Reading

CNN and CNN-LSTM: the cleanest samples are all Laplace-prior DDoS -> DoS flips. CNN-LSTM is the most compact exhibit set, with the best CW sample at scaled L2 0.0788 and the best PGD sample at scaled L2 0.2649.

DualPath: both attack methods select BruteForce-origin samples. The best CW example is especially compact, flipping BruteForce -> DDoS with scaled L2 0.5689.

LSTM: the selected cases are BruteForce-origin flips to DoS or Mirai. The feature changes concentrate on `IAT`, `Min`, `Std`, and `ack_count`, suggesting a stable behavioral route across PGD and CW.

MLP: PGD gives a compact BruteForce -> Spoofing exhibit at scaled L2 0.6248, while CW requires larger input-space movement but remains valid and successful.

## Thesis-Ready Paragraph

For the featurewise case study, I selected the two lowest-L2 regenerated successes for each classifier and latent attack method across the Gaussian and Laplace VAE reruns. All selected samples remained protocol-valid and mask-compliant. The most compact examples occur for CNN-LSTM, where latent-CW flips DDoS to DoS with scaled L2 0.0788 and latent-PGD does so with scaled L2 0.2649. Across models, the perturbations concentrate on timing and aggregate traffic statistics such as `IAT`, `Header_Length`, `Min`, `Std`, and `Rate`, rather than indiscriminately changing protocol or service indicators. This featurewise behavior supports the central interpretation that latent attacks produce coherent, validity-preserving adversarial examples, in contrast to unconstrained input-space attacks that often achieve high raw ASR by leaving the feasible traffic manifold.
