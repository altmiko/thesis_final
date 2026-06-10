# He-IDSR and Domain-Valid Attack Success Report

VAE run: `gaussian_anticollapse_beta05_freebits01_20260529_173512`

## Definitions

- `E_i`: attack-goal success for sample `i`. For untargeted attacks this is misclassification; for target-to-Benign attacks this is prediction as Benign.
- `V_i`: joint domain validity (protocol validity, perturbation-mask compliance, and raw G1-G8 validity where evaluated).
- `O_i`: Mahalanobis outlier flag after re-encoding the final adversarial sample with its source-class VAE.
- `ASR_raw = mean(E_i)`.
- `ASR_valid = mean(E_i * V_i)`. This is the thesis joint valid-success rate, not the conditional `asr_valid_only` field in the rerun summaries.
- `Mahalanobis ID rate = mean(1 - O_i)`.
- `He-IDSR = mean(E_i * (1 - O_i))`.

## Scope

- Five classifiers: MLP, CNN, LSTM, CNN-LSTM, and DualPath.
- Seven malicious source classes. Benign is a target class, not an attack source class.
- Nine attack configurations: latent PGD/CW, unconstrained input PGD/CW, targeted latent PGD, constrained input PGD/CW, and their target-to-Benign variants.
- Cells without any correctly classified source samples are omitted, matching the original runs.

## Overall Results

| Family | Attack | Goal | N | ASR raw | ASR valid | Mahalanobis ID | He-IDSR |
|---|---|---|---|---|---|---|---|
| latent | latent-pgd | untargeted | 3200 | 37.12% | 33.22% | 90.16% | 32.38% |
| latent | latent-cw | untargeted | 3200 | 36.62% | 33.22% | 91.09% | 32.09% |
| unconstrained-input | input-pgd | untargeted | 3200 | 95.00% | 0.00% | 25.37% | 25.28% |
| unconstrained-input | input-cw | untargeted | 3200 | 90.28% | 0.00% | 74.44% | 67.72% |
| latent | targeted-benign-latent-pgd | target-benign | 3200 | 4.53% | 4.22% | 92.56% | 4.00% |
| constrained-input | cinput-pgd | untargeted | 3200 | 52.59% | 52.59% | 34.88% | 17.94% |
| constrained-input | cinput-cw | untargeted | 3200 | 76.22% | 76.22% | 60.59% | 45.38% |
| constrained-input | cinput-pgd-target-benign | target-benign | 3200 | 4.16% | 4.16% | 33.22% | 1.91% |
| constrained-input | cinput-cw-target-benign | target-benign | 3200 | 10.78% | 10.78% | 37.91% | 5.69% |

## Key Findings

- Latent PGD and latent CW have similar joint performance: He-IDSR is `32.38%` and `32.09%`, while ASR_valid is `33.22%` for both.
- Unconstrained input PGD and CW achieve high raw evasion (`95.00%` and `90.28%`) but `0.00%` ASR_valid because the generated samples fail the attack pipeline's joint domain constraints.
- Mahalanobis membership alone is not equivalent to domain validity: unconstrained input CW has `67.72%` He-IDSR despite `0.00%` ASR_valid.
- Constrained input CW has the strongest ASR_valid at `76.22%`, but its He-IDSR is lower at `45.38%` because many successful samples are Mahalanobis outliers.
- Target-to-Benign attacks remain weak: targeted latent PGD reaches `4.00%` He-IDSR, constrained PGD `1.91%`, and constrained CW `5.69%`.

## Results by Classifier

| Attack | Classifier | N | ASR raw | ASR valid | Mahalanobis ID | He-IDSR |
|---|---|---|---|---|---|---|
| latent-pgd | MLP | 700 | 38.29% | 33.57% | 90.14% | 33.86% |
| latent-cw | MLP | 700 | 37.00% | 33.29% | 90.86% | 33.14% |
| input-pgd | MLP | 700 | 95.43% | 0.00% | 23.71% | 23.57% |
| input-cw | MLP | 700 | 89.86% | 0.00% | 72.00% | 65.86% |
| latent-pgd | CNN | 500 | 32.00% | 29.20% | 88.80% | 27.00% |
| latent-cw | CNN | 500 | 33.20% | 31.40% | 91.20% | 28.80% |
| input-pgd | CNN | 500 | 91.20% | 0.00% | 8.80% | 8.40% |
| input-cw | CNN | 500 | 88.40% | 0.00% | 70.60% | 63.00% |
| latent-pgd | LSTM | 700 | 37.29% | 34.43% | 90.86% | 33.00% |
| latent-cw | LSTM | 700 | 39.71% | 36.00% | 90.71% | 35.00% |
| input-pgd | LSTM | 700 | 97.14% | 0.00% | 31.43% | 31.43% |
| input-cw | LSTM | 700 | 95.71% | 0.00% | 73.43% | 71.00% |
| latent-pgd | CNN-LSTM | 600 | 40.50% | 36.33% | 89.17% | 34.17% |
| latent-cw | CNN-LSTM | 600 | 41.17% | 38.83% | 90.83% | 35.67% |
| input-pgd | CNN-LSTM | 600 | 95.00% | 0.00% | 28.50% | 28.50% |
| input-cw | CNN-LSTM | 600 | 97.33% | 0.00% | 84.67% | 82.50% |
| latent-pgd | DualPath | 700 | 36.57% | 31.86% | 91.29% | 32.57% |
| latent-cw | DualPath | 700 | 31.71% | 26.86% | 91.86% | 27.43% |
| input-pgd | DualPath | 700 | 95.14% | 0.00% | 30.14% | 30.14% |
| input-cw | DualPath | 700 | 80.57% | 0.00% | 71.86% | 57.00% |
| targeted-benign-latent-pgd | MLP | 700 | 4.14% | 3.43% | 92.43% | 4.00% |
| targeted-benign-latent-pgd | CNN | 500 | 5.00% | 5.00% | 91.60% | 4.80% |
| targeted-benign-latent-pgd | LSTM | 700 | 5.29% | 4.86% | 93.43% | 4.57% |
| targeted-benign-latent-pgd | CNN-LSTM | 600 | 3.83% | 3.83% | 93.83% | 3.67% |
| targeted-benign-latent-pgd | DualPath | 700 | 4.43% | 4.14% | 91.43% | 3.14% |
| cinput-pgd | MLP | 700 | 41.57% | 41.57% | 37.57% | 12.57% |
| cinput-cw | MLP | 700 | 71.00% | 71.00% | 61.71% | 39.57% |
| cinput-pgd-target-benign | MLP | 700 | 1.43% | 1.43% | 40.43% | 1.29% |
| cinput-cw-target-benign | MLP | 700 | 5.29% | 5.29% | 46.86% | 5.14% |
| cinput-pgd | CNN | 500 | 54.00% | 54.00% | 24.20% | 13.00% |
| cinput-cw | CNN | 500 | 69.00% | 69.00% | 57.00% | 38.80% |
| cinput-pgd-target-benign | CNN | 500 | 3.60% | 3.60% | 21.60% | 2.40% |
| cinput-cw-target-benign | CNN | 500 | 12.80% | 12.80% | 26.60% | 6.20% |
| cinput-pgd | LSTM | 700 | 56.00% | 56.00% | 35.29% | 21.43% |
| cinput-cw | LSTM | 700 | 86.71% | 86.71% | 59.14% | 52.14% |
| cinput-pgd-target-benign | LSTM | 700 | 1.86% | 1.86% | 31.29% | 0.43% |
| cinput-cw-target-benign | LSTM | 700 | 7.43% | 7.43% | 35.43% | 2.14% |
| cinput-pgd | CNN-LSTM | 600 | 65.83% | 65.83% | 40.83% | 31.50% |
| cinput-cw | CNN-LSTM | 600 | 75.50% | 75.50% | 63.17% | 52.17% |
| cinput-pgd-target-benign | CNN-LSTM | 600 | 2.00% | 2.00% | 35.67% | 1.33% |
| cinput-cw-target-benign | CNN-LSTM | 600 | 5.17% | 5.17% | 41.67% | 4.00% |
| cinput-pgd | DualPath | 700 | 47.86% | 47.86% | 34.29% | 11.71% |
| cinput-cw | DualPath | 700 | 76.71% | 76.71% | 61.29% | 43.29% |
| cinput-pgd-target-benign | DualPath | 700 | 11.43% | 11.43% | 34.14% | 4.14% |
| cinput-cw-target-benign | DualPath | 700 | 23.00% | 23.00% | 36.29% | 10.86% |

## Detailed Results by Source Class

### `latent-pgd`

| Classifier | Class | N | ASR raw | ASR valid | Mahalanobis ID | He-IDSR |
|---|---|---|---|---|---|---|
| MLP | BruteForce | 100 | 16.00% | 4.00% | 93.00% | 16.00% |
| MLP | DDoS | 100 | 23.00% | 17.00% | 77.00% | 9.00% |
| MLP | DoS | 100 | 79.00% | 75.00% | 90.00% | 71.00% |
| MLP | Mirai | 100 | 1.00% | 1.00% | 92.00% | 0.00% |
| MLP | Recon | 100 | 13.00% | 12.00% | 98.00% | 12.00% |
| MLP | Spoofing | 100 | 37.00% | 30.00% | 88.00% | 36.00% |
| MLP | Web | 100 | 99.00% | 96.00% | 93.00% | 93.00% |
| CNN | DDoS | 100 | 23.00% | 18.00% | 82.00% | 14.00% |
| CNN | DoS | 100 | 66.00% | 61.00% | 88.00% | 57.00% |
| CNN | Mirai | 100 | 8.00% | 7.00% | 90.00% | 3.00% |
| CNN | Recon | 100 | 20.00% | 20.00% | 98.00% | 19.00% |
| CNN | Spoofing | 100 | 43.00% | 40.00% | 86.00% | 42.00% |
| LSTM | BruteForce | 100 | 7.00% | 2.00% | 95.00% | 4.00% |
| LSTM | DDoS | 100 | 26.00% | 19.00% | 74.00% | 9.00% |
| LSTM | DoS | 100 | 72.00% | 71.00% | 91.00% | 66.00% |
| LSTM | Mirai | 100 | 1.00% | 1.00% | 90.00% | 0.00% |
| LSTM | Recon | 100 | 8.00% | 7.00% | 95.00% | 7.00% |
| LSTM | Spoofing | 100 | 47.00% | 41.00% | 91.00% | 45.00% |
| LSTM | Web | 100 | 100.00% | 100.00% | 100.00% | 100.00% |
| CNN-LSTM | DDoS | 100 | 32.00% | 23.00% | 65.00% | 7.00% |
| CNN-LSTM | DoS | 100 | 53.00% | 48.00% | 90.00% | 47.00% |
| CNN-LSTM | Mirai | 100 | 3.00% | 3.00% | 93.00% | 0.00% |
| CNN-LSTM | Recon | 100 | 9.00% | 9.00% | 96.00% | 8.00% |
| CNN-LSTM | Spoofing | 100 | 46.00% | 37.00% | 92.00% | 44.00% |
| CNN-LSTM | Web | 100 | 100.00% | 98.00% | 99.00% | 99.00% |
| DualPath | BruteForce | 100 | 18.00% | 3.00% | 92.00% | 14.00% |
| DualPath | DDoS | 100 | 20.00% | 14.00% | 80.00% | 12.00% |
| DualPath | DoS | 100 | 70.00% | 68.00% | 85.00% | 62.00% |
| DualPath | Mirai | 100 | 2.00% | 1.00% | 96.00% | 0.00% |
| DualPath | Recon | 100 | 9.00% | 8.00% | 98.00% | 9.00% |
| DualPath | Spoofing | 100 | 44.00% | 36.00% | 88.00% | 38.00% |
| DualPath | Web | 100 | 93.00% | 93.00% | 100.00% | 93.00% |

### `latent-cw`

| Classifier | Class | N | ASR raw | ASR valid | Mahalanobis ID | He-IDSR |
|---|---|---|---|---|---|---|
| MLP | BruteForce | 100 | 36.00% | 20.00% | 90.00% | 36.00% |
| MLP | DDoS | 100 | 22.00% | 20.00% | 79.00% | 9.00% |
| MLP | DoS | 100 | 78.00% | 76.00% | 89.00% | 70.00% |
| MLP | Mirai | 100 | 1.00% | 1.00% | 92.00% | 0.00% |
| MLP | Recon | 100 | 9.00% | 9.00% | 97.00% | 9.00% |
| MLP | Spoofing | 100 | 61.00% | 56.00% | 96.00% | 60.00% |
| MLP | Web | 100 | 52.00% | 51.00% | 93.00% | 48.00% |
| CNN | DDoS | 100 | 24.00% | 21.00% | 86.00% | 16.00% |
| CNN | DoS | 100 | 57.00% | 55.00% | 90.00% | 53.00% |
| CNN | Mirai | 100 | 8.00% | 8.00% | 91.00% | 3.00% |
| CNN | Recon | 100 | 13.00% | 13.00% | 97.00% | 12.00% |
| CNN | Spoofing | 100 | 64.00% | 60.00% | 92.00% | 60.00% |
| LSTM | BruteForce | 100 | 36.00% | 19.00% | 94.00% | 31.00% |
| LSTM | DDoS | 100 | 24.00% | 21.00% | 79.00% | 12.00% |
| LSTM | DoS | 100 | 70.00% | 69.00% | 88.00% | 63.00% |
| LSTM | Mirai | 100 | 4.00% | 4.00% | 91.00% | 0.00% |
| LSTM | Recon | 100 | 9.00% | 8.00% | 95.00% | 8.00% |
| LSTM | Spoofing | 100 | 70.00% | 66.00% | 88.00% | 66.00% |
| LSTM | Web | 100 | 65.00% | 65.00% | 100.00% | 65.00% |
| CNN-LSTM | DDoS | 100 | 31.00% | 28.00% | 72.00% | 11.00% |
| CNN-LSTM | DoS | 100 | 48.00% | 45.00% | 90.00% | 43.00% |
| CNN-LSTM | Mirai | 100 | 4.00% | 4.00% | 93.00% | 0.00% |
| CNN-LSTM | Recon | 100 | 7.00% | 7.00% | 97.00% | 7.00% |
| CNN-LSTM | Spoofing | 100 | 62.00% | 57.00% | 94.00% | 59.00% |
| CNN-LSTM | Web | 100 | 95.00% | 92.00% | 99.00% | 94.00% |
| DualPath | BruteForce | 100 | 53.00% | 26.00% | 91.00% | 46.00% |
| DualPath | DDoS | 100 | 23.00% | 21.00% | 86.00% | 17.00% |
| DualPath | DoS | 100 | 64.00% | 64.00% | 90.00% | 59.00% |
| DualPath | Mirai | 100 | 6.00% | 6.00% | 89.00% | 0.00% |
| DualPath | Recon | 100 | 9.00% | 8.00% | 98.00% | 9.00% |
| DualPath | Spoofing | 100 | 61.00% | 57.00% | 89.00% | 55.00% |
| DualPath | Web | 100 | 6.00% | 6.00% | 100.00% | 6.00% |

### `input-pgd`

| Classifier | Class | N | ASR raw | ASR valid | Mahalanobis ID | He-IDSR |
|---|---|---|---|---|---|---|
| MLP | BruteForce | 100 | 100.00% | 0.00% | 76.00% | 76.00% |
| MLP | DDoS | 100 | 81.00% | 0.00% | 3.00% | 2.00% |
| MLP | DoS | 100 | 100.00% | 0.00% | 7.00% | 7.00% |
| MLP | Mirai | 100 | 89.00% | 0.00% | 0.00% | 0.00% |
| MLP | Recon | 100 | 100.00% | 0.00% | 1.00% | 1.00% |
| MLP | Spoofing | 100 | 98.00% | 0.00% | 3.00% | 3.00% |
| MLP | Web | 100 | 100.00% | 0.00% | 76.00% | 76.00% |
| CNN | DDoS | 100 | 67.00% | 0.00% | 6.00% | 4.00% |
| CNN | DoS | 100 | 99.00% | 0.00% | 7.00% | 7.00% |
| CNN | Mirai | 100 | 93.00% | 0.00% | 0.00% | 0.00% |
| CNN | Recon | 100 | 100.00% | 0.00% | 20.00% | 20.00% |
| CNN | Spoofing | 100 | 97.00% | 0.00% | 11.00% | 11.00% |
| LSTM | BruteForce | 100 | 100.00% | 0.00% | 95.00% | 95.00% |
| LSTM | DDoS | 100 | 87.00% | 0.00% | 12.00% | 12.00% |
| LSTM | DoS | 100 | 100.00% | 0.00% | 27.00% | 27.00% |
| LSTM | Mirai | 100 | 99.00% | 0.00% | 0.00% | 0.00% |
| LSTM | Recon | 100 | 100.00% | 0.00% | 2.00% | 2.00% |
| LSTM | Spoofing | 100 | 94.00% | 0.00% | 3.00% | 3.00% |
| LSTM | Web | 100 | 100.00% | 0.00% | 81.00% | 81.00% |
| CNN-LSTM | DDoS | 100 | 96.00% | 0.00% | 8.00% | 8.00% |
| CNN-LSTM | DoS | 100 | 100.00% | 0.00% | 25.00% | 25.00% |
| CNN-LSTM | Mirai | 100 | 78.00% | 0.00% | 10.00% | 10.00% |
| CNN-LSTM | Recon | 100 | 100.00% | 0.00% | 15.00% | 15.00% |
| CNN-LSTM | Spoofing | 100 | 96.00% | 0.00% | 17.00% | 17.00% |
| CNN-LSTM | Web | 100 | 100.00% | 0.00% | 96.00% | 96.00% |
| DualPath | BruteForce | 100 | 100.00% | 0.00% | 91.00% | 91.00% |
| DualPath | DDoS | 100 | 83.00% | 0.00% | 1.00% | 1.00% |
| DualPath | DoS | 100 | 100.00% | 0.00% | 11.00% | 11.00% |
| DualPath | Mirai | 100 | 93.00% | 0.00% | 0.00% | 0.00% |
| DualPath | Recon | 100 | 96.00% | 0.00% | 4.00% | 4.00% |
| DualPath | Spoofing | 100 | 94.00% | 0.00% | 4.00% | 4.00% |
| DualPath | Web | 100 | 100.00% | 0.00% | 100.00% | 100.00% |

### `input-cw`

| Classifier | Class | N | ASR raw | ASR valid | Mahalanobis ID | He-IDSR |
|---|---|---|---|---|---|---|
| MLP | BruteForce | 100 | 100.00% | 0.00% | 95.00% | 95.00% |
| MLP | DDoS | 100 | 60.00% | 0.00% | 42.00% | 21.00% |
| MLP | DoS | 100 | 86.00% | 0.00% | 82.00% | 74.00% |
| MLP | Mirai | 100 | 100.00% | 0.00% | 31.00% | 31.00% |
| MLP | Recon | 100 | 90.00% | 0.00% | 82.00% | 74.00% |
| MLP | Spoofing | 100 | 93.00% | 0.00% | 79.00% | 73.00% |
| MLP | Web | 100 | 100.00% | 0.00% | 93.00% | 93.00% |
| CNN | DDoS | 100 | 71.00% | 0.00% | 61.00% | 49.00% |
| CNN | DoS | 100 | 86.00% | 0.00% | 91.00% | 77.00% |
| CNN | Mirai | 100 | 100.00% | 0.00% | 26.00% | 26.00% |
| CNN | Recon | 100 | 87.00% | 0.00% | 91.00% | 81.00% |
| CNN | Spoofing | 100 | 98.00% | 0.00% | 84.00% | 82.00% |
| LSTM | BruteForce | 100 | 100.00% | 0.00% | 96.00% | 96.00% |
| LSTM | DDoS | 100 | 86.00% | 0.00% | 67.00% | 62.00% |
| LSTM | DoS | 100 | 100.00% | 0.00% | 90.00% | 90.00% |
| LSTM | Mirai | 100 | 100.00% | 0.00% | 0.00% | 0.00% |
| LSTM | Recon | 100 | 92.00% | 0.00% | 76.00% | 72.00% |
| LSTM | Spoofing | 100 | 92.00% | 0.00% | 85.00% | 77.00% |
| LSTM | Web | 100 | 100.00% | 0.00% | 100.00% | 100.00% |
| CNN-LSTM | DDoS | 100 | 98.00% | 0.00% | 61.00% | 61.00% |
| CNN-LSTM | DoS | 100 | 94.00% | 0.00% | 94.00% | 88.00% |
| CNN-LSTM | Mirai | 100 | 99.00% | 0.00% | 69.00% | 69.00% |
| CNN-LSTM | Recon | 100 | 94.00% | 0.00% | 97.00% | 91.00% |
| CNN-LSTM | Spoofing | 100 | 99.00% | 0.00% | 88.00% | 87.00% |
| CNN-LSTM | Web | 100 | 100.00% | 0.00% | 99.00% | 99.00% |
| DualPath | BruteForce | 100 | 100.00% | 0.00% | 95.00% | 95.00% |
| DualPath | DDoS | 100 | 84.00% | 0.00% | 58.00% | 58.00% |
| DualPath | DoS | 100 | 100.00% | 0.00% | 90.00% | 90.00% |
| DualPath | Mirai | 100 | 100.00% | 0.00% | 0.00% | 0.00% |
| DualPath | Recon | 100 | 87.00% | 0.00% | 72.00% | 71.00% |
| DualPath | Spoofing | 100 | 91.00% | 0.00% | 88.00% | 83.00% |
| DualPath | Web | 100 | 2.00% | 0.00% | 100.00% | 2.00% |

### `targeted-benign-latent-pgd`

| Classifier | Class | N | ASR raw | ASR valid | Mahalanobis ID | He-IDSR |
|---|---|---|---|---|---|---|
| MLP | BruteForce | 100 | 0.00% | 0.00% | 93.00% | 0.00% |
| MLP | DDoS | 100 | 0.00% | 0.00% | 90.00% | 0.00% |
| MLP | DoS | 100 | 0.00% | 0.00% | 86.00% | 0.00% |
| MLP | Mirai | 100 | 0.00% | 0.00% | 97.00% | 0.00% |
| MLP | Recon | 100 | 6.00% | 5.00% | 98.00% | 6.00% |
| MLP | Spoofing | 100 | 20.00% | 16.00% | 89.00% | 19.00% |
| MLP | Web | 100 | 3.00% | 3.00% | 94.00% | 3.00% |
| CNN | DDoS | 100 | 0.00% | 0.00% | 90.00% | 0.00% |
| CNN | DoS | 100 | 0.00% | 0.00% | 91.00% | 0.00% |
| CNN | Mirai | 100 | 0.00% | 0.00% | 94.00% | 0.00% |
| CNN | Recon | 100 | 10.00% | 10.00% | 97.00% | 10.00% |
| CNN | Spoofing | 100 | 15.00% | 15.00% | 86.00% | 14.00% |
| LSTM | BruteForce | 100 | 4.00% | 3.00% | 92.00% | 0.00% |
| LSTM | DDoS | 100 | 0.00% | 0.00% | 93.00% | 0.00% |
| LSTM | DoS | 100 | 0.00% | 0.00% | 87.00% | 0.00% |
| LSTM | Mirai | 100 | 0.00% | 0.00% | 95.00% | 0.00% |
| LSTM | Recon | 100 | 4.00% | 4.00% | 96.00% | 4.00% |
| LSTM | Spoofing | 100 | 18.00% | 16.00% | 91.00% | 17.00% |
| LSTM | Web | 100 | 11.00% | 11.00% | 100.00% | 11.00% |
| CNN-LSTM | DDoS | 100 | 0.00% | 0.00% | 90.00% | 0.00% |
| CNN-LSTM | DoS | 100 | 0.00% | 0.00% | 92.00% | 0.00% |
| CNN-LSTM | Mirai | 100 | 0.00% | 0.00% | 95.00% | 0.00% |
| CNN-LSTM | Recon | 100 | 3.00% | 3.00% | 97.00% | 3.00% |
| CNN-LSTM | Spoofing | 100 | 20.00% | 20.00% | 90.00% | 19.00% |
| CNN-LSTM | Web | 100 | 0.00% | 0.00% | 99.00% | 0.00% |
| DualPath | BruteForce | 100 | 7.00% | 7.00% | 82.00% | 0.00% |
| DualPath | DDoS | 100 | 0.00% | 0.00% | 89.00% | 0.00% |
| DualPath | DoS | 100 | 0.00% | 0.00% | 85.00% | 0.00% |
| DualPath | Mirai | 100 | 0.00% | 0.00% | 95.00% | 0.00% |
| DualPath | Recon | 100 | 4.00% | 4.00% | 98.00% | 4.00% |
| DualPath | Spoofing | 100 | 20.00% | 18.00% | 91.00% | 18.00% |
| DualPath | Web | 100 | 0.00% | 0.00% | 100.00% | 0.00% |

### `cinput-pgd`

| Classifier | Class | N | ASR raw | ASR valid | Mahalanobis ID | He-IDSR |
|---|---|---|---|---|---|---|
| MLP | BruteForce | 100 | 25.00% | 25.00% | 63.00% | 4.00% |
| MLP | DDoS | 100 | 21.00% | 21.00% | 23.00% | 8.00% |
| MLP | DoS | 100 | 82.00% | 82.00% | 23.00% | 18.00% |
| MLP | Mirai | 100 | 17.00% | 17.00% | 1.00% | 1.00% |
| MLP | Recon | 100 | 55.00% | 55.00% | 40.00% | 18.00% |
| MLP | Spoofing | 100 | 59.00% | 59.00% | 23.00% | 17.00% |
| MLP | Web | 100 | 32.00% | 32.00% | 90.00% | 22.00% |
| CNN | DDoS | 100 | 32.00% | 32.00% | 20.00% | 7.00% |
| CNN | DoS | 100 | 75.00% | 75.00% | 27.00% | 12.00% |
| CNN | Mirai | 100 | 21.00% | 21.00% | 11.00% | 0.00% |
| CNN | Recon | 100 | 71.00% | 71.00% | 41.00% | 26.00% |
| CNN | Spoofing | 100 | 71.00% | 71.00% | 22.00% | 20.00% |
| LSTM | BruteForce | 100 | 59.00% | 59.00% | 74.00% | 36.00% |
| LSTM | DDoS | 100 | 28.00% | 28.00% | 17.00% | 10.00% |
| LSTM | DoS | 100 | 90.00% | 90.00% | 16.00% | 15.00% |
| LSTM | Mirai | 100 | 17.00% | 17.00% | 1.00% | 0.00% |
| LSTM | Recon | 100 | 71.00% | 71.00% | 27.00% | 18.00% |
| LSTM | Spoofing | 100 | 64.00% | 64.00% | 14.00% | 10.00% |
| LSTM | Web | 100 | 63.00% | 63.00% | 98.00% | 61.00% |
| CNN-LSTM | DDoS | 100 | 52.00% | 52.00% | 19.00% | 12.00% |
| CNN-LSTM | DoS | 100 | 84.00% | 84.00% | 57.00% | 48.00% |
| CNN-LSTM | Mirai | 100 | 28.00% | 28.00% | 2.00% | 2.00% |
| CNN-LSTM | Recon | 100 | 71.00% | 71.00% | 36.00% | 18.00% |
| CNN-LSTM | Spoofing | 100 | 77.00% | 77.00% | 32.00% | 27.00% |
| CNN-LSTM | Web | 100 | 83.00% | 83.00% | 99.00% | 82.00% |
| DualPath | BruteForce | 100 | 74.00% | 74.00% | 48.00% | 25.00% |
| DualPath | DDoS | 100 | 38.00% | 38.00% | 18.00% | 6.00% |
| DualPath | DoS | 100 | 82.00% | 82.00% | 22.00% | 18.00% |
| DualPath | Mirai | 100 | 14.00% | 14.00% | 2.00% | 1.00% |
| DualPath | Recon | 100 | 78.00% | 78.00% | 26.00% | 19.00% |
| DualPath | Spoofing | 100 | 49.00% | 49.00% | 24.00% | 13.00% |
| DualPath | Web | 100 | 0.00% | 0.00% | 100.00% | 0.00% |

### `cinput-cw`

| Classifier | Class | N | ASR raw | ASR valid | Mahalanobis ID | He-IDSR |
|---|---|---|---|---|---|---|
| MLP | BruteForce | 100 | 76.00% | 76.00% | 88.00% | 65.00% |
| MLP | DDoS | 100 | 55.00% | 55.00% | 36.00% | 13.00% |
| MLP | DoS | 100 | 86.00% | 86.00% | 79.00% | 73.00% |
| MLP | Mirai | 100 | 99.00% | 99.00% | 2.00% | 2.00% |
| MLP | Recon | 100 | 77.00% | 77.00% | 72.00% | 50.00% |
| MLP | Spoofing | 100 | 76.00% | 76.00% | 63.00% | 53.00% |
| MLP | Web | 100 | 28.00% | 28.00% | 92.00% | 21.00% |
| CNN | DDoS | 100 | 56.00% | 56.00% | 51.00% | 35.00% |
| CNN | DoS | 100 | 66.00% | 66.00% | 84.00% | 55.00% |
| CNN | Mirai | 100 | 73.00% | 73.00% | 16.00% | 2.00% |
| CNN | Recon | 100 | 68.00% | 68.00% | 75.00% | 49.00% |
| CNN | Spoofing | 100 | 82.00% | 82.00% | 59.00% | 53.00% |
| LSTM | BruteForce | 100 | 100.00% | 100.00% | 76.00% | 76.00% |
| LSTM | DDoS | 100 | 72.00% | 72.00% | 27.00% | 21.00% |
| LSTM | DoS | 100 | 94.00% | 94.00% | 85.00% | 83.00% |
| LSTM | Mirai | 100 | 100.00% | 100.00% | 2.00% | 2.00% |
| LSTM | Recon | 100 | 62.00% | 62.00% | 75.00% | 42.00% |
| LSTM | Spoofing | 100 | 79.00% | 79.00% | 53.00% | 45.00% |
| LSTM | Web | 100 | 100.00% | 100.00% | 96.00% | 96.00% |
| CNN-LSTM | DDoS | 100 | 66.00% | 66.00% | 31.00% | 21.00% |
| CNN-LSTM | DoS | 100 | 83.00% | 83.00% | 86.00% | 71.00% |
| CNN-LSTM | Mirai | 100 | 48.00% | 48.00% | 6.00% | 6.00% |
| CNN-LSTM | Recon | 100 | 76.00% | 76.00% | 73.00% | 51.00% |
| CNN-LSTM | Spoofing | 100 | 95.00% | 95.00% | 84.00% | 80.00% |
| CNN-LSTM | Web | 100 | 85.00% | 85.00% | 99.00% | 84.00% |
| DualPath | BruteForce | 100 | 100.00% | 100.00% | 57.00% | 57.00% |
| DualPath | DDoS | 100 | 70.00% | 70.00% | 46.00% | 38.00% |
| DualPath | DoS | 100 | 96.00% | 96.00% | 82.00% | 79.00% |
| DualPath | Mirai | 100 | 100.00% | 100.00% | 2.00% | 2.00% |
| DualPath | Recon | 100 | 90.00% | 90.00% | 75.00% | 68.00% |
| DualPath | Spoofing | 100 | 81.00% | 81.00% | 67.00% | 59.00% |
| DualPath | Web | 100 | 0.00% | 0.00% | 100.00% | 0.00% |

### `cinput-pgd-target-benign`

| Classifier | Class | N | ASR raw | ASR valid | Mahalanobis ID | He-IDSR |
|---|---|---|---|---|---|---|
| MLP | BruteForce | 100 | 0.00% | 0.00% | 74.00% | 0.00% |
| MLP | DDoS | 100 | 0.00% | 0.00% | 20.00% | 0.00% |
| MLP | DoS | 100 | 0.00% | 0.00% | 13.00% | 0.00% |
| MLP | Mirai | 100 | 0.00% | 0.00% | 0.00% | 0.00% |
| MLP | Recon | 100 | 3.00% | 3.00% | 45.00% | 2.00% |
| MLP | Spoofing | 100 | 7.00% | 7.00% | 38.00% | 7.00% |
| MLP | Web | 100 | 0.00% | 0.00% | 93.00% | 0.00% |
| CNN | DDoS | 100 | 0.00% | 0.00% | 18.00% | 0.00% |
| CNN | DoS | 100 | 0.00% | 0.00% | 11.00% | 0.00% |
| CNN | Mirai | 100 | 0.00% | 0.00% | 2.00% | 0.00% |
| CNN | Recon | 100 | 8.00% | 8.00% | 41.00% | 6.00% |
| CNN | Spoofing | 100 | 10.00% | 10.00% | 36.00% | 6.00% |
| LSTM | BruteForce | 100 | 9.00% | 9.00% | 32.00% | 0.00% |
| LSTM | DDoS | 100 | 0.00% | 0.00% | 24.00% | 0.00% |
| LSTM | DoS | 100 | 0.00% | 0.00% | 4.00% | 0.00% |
| LSTM | Mirai | 100 | 0.00% | 0.00% | 1.00% | 0.00% |
| LSTM | Recon | 100 | 2.00% | 2.00% | 42.00% | 2.00% |
| LSTM | Spoofing | 100 | 2.00% | 2.00% | 20.00% | 1.00% |
| LSTM | Web | 100 | 0.00% | 0.00% | 96.00% | 0.00% |
| CNN-LSTM | DDoS | 100 | 0.00% | 0.00% | 25.00% | 0.00% |
| CNN-LSTM | DoS | 100 | 0.00% | 0.00% | 17.00% | 0.00% |
| CNN-LSTM | Mirai | 100 | 0.00% | 0.00% | 0.00% | 0.00% |
| CNN-LSTM | Recon | 100 | 0.00% | 0.00% | 47.00% | 0.00% |
| CNN-LSTM | Spoofing | 100 | 5.00% | 5.00% | 26.00% | 1.00% |
| CNN-LSTM | Web | 100 | 7.00% | 7.00% | 99.00% | 7.00% |
| DualPath | BruteForce | 100 | 39.00% | 39.00% | 15.00% | 0.00% |
| DualPath | DDoS | 100 | 0.00% | 0.00% | 31.00% | 0.00% |
| DualPath | DoS | 100 | 0.00% | 0.00% | 32.00% | 0.00% |
| DualPath | Mirai | 100 | 0.00% | 0.00% | 2.00% | 0.00% |
| DualPath | Recon | 100 | 33.00% | 33.00% | 42.00% | 27.00% |
| DualPath | Spoofing | 100 | 8.00% | 8.00% | 17.00% | 2.00% |
| DualPath | Web | 100 | 0.00% | 0.00% | 100.00% | 0.00% |

### `cinput-cw-target-benign`

| Classifier | Class | N | ASR raw | ASR valid | Mahalanobis ID | He-IDSR |
|---|---|---|---|---|---|---|
| MLP | BruteForce | 100 | 0.00% | 0.00% | 82.00% | 0.00% |
| MLP | DDoS | 100 | 1.00% | 1.00% | 9.00% | 1.00% |
| MLP | DoS | 100 | 0.00% | 0.00% | 12.00% | 0.00% |
| MLP | Mirai | 100 | 0.00% | 0.00% | 2.00% | 0.00% |
| MLP | Recon | 100 | 14.00% | 14.00% | 75.00% | 14.00% |
| MLP | Spoofing | 100 | 21.00% | 21.00% | 56.00% | 20.00% |
| MLP | Web | 100 | 1.00% | 1.00% | 92.00% | 1.00% |
| CNN | DDoS | 100 | 14.00% | 14.00% | 7.00% | 0.00% |
| CNN | DoS | 100 | 17.00% | 17.00% | 2.00% | 2.00% |
| CNN | Mirai | 100 | 0.00% | 0.00% | 0.00% | 0.00% |
| CNN | Recon | 100 | 16.00% | 16.00% | 72.00% | 14.00% |
| CNN | Spoofing | 100 | 17.00% | 17.00% | 52.00% | 15.00% |
| LSTM | BruteForce | 100 | 23.00% | 23.00% | 25.00% | 0.00% |
| LSTM | DDoS | 100 | 10.00% | 10.00% | 8.00% | 0.00% |
| LSTM | DoS | 100 | 0.00% | 0.00% | 0.00% | 0.00% |
| LSTM | Mirai | 100 | 1.00% | 1.00% | 0.00% | 0.00% |
| LSTM | Recon | 100 | 5.00% | 5.00% | 70.00% | 4.00% |
| LSTM | Spoofing | 100 | 13.00% | 13.00% | 49.00% | 11.00% |
| LSTM | Web | 100 | 0.00% | 0.00% | 96.00% | 0.00% |
| CNN-LSTM | DDoS | 100 | 1.00% | 1.00% | 12.00% | 0.00% |
| CNN-LSTM | DoS | 100 | 2.00% | 2.00% | 3.00% | 0.00% |
| CNN-LSTM | Mirai | 100 | 0.00% | 0.00% | 0.00% | 0.00% |
| CNN-LSTM | Recon | 100 | 4.00% | 4.00% | 74.00% | 4.00% |
| CNN-LSTM | Spoofing | 100 | 24.00% | 24.00% | 62.00% | 20.00% |
| CNN-LSTM | Web | 100 | 0.00% | 0.00% | 99.00% | 0.00% |
| DualPath | BruteForce | 100 | 59.00% | 59.00% | 13.00% | 7.00% |
| DualPath | DDoS | 100 | 4.00% | 4.00% | 7.00% | 0.00% |
| DualPath | DoS | 100 | 6.00% | 6.00% | 28.00% | 0.00% |
| DualPath | Mirai | 100 | 11.00% | 11.00% | 0.00% | 0.00% |
| DualPath | Recon | 100 | 53.00% | 53.00% | 73.00% | 49.00% |
| DualPath | Spoofing | 100 | 28.00% | 28.00% | 33.00% | 20.00% |
| DualPath | Web | 100 | 0.00% | 0.00% | 100.00% | 0.00% |

## Reproduction Checks

- Minimum per-cell evasion-mask agreement with saved results: `100.00%`.
- Minimum per-cell joint-validity-mask agreement with saved results: `100.00%`.
- Targeted latent PGD used its saved `x_adv` NPZ artifacts; the other families were rerun because their final adversarial samples or per-sample Mahalanobis flags were not persisted.

## Source Runs

- Untargeted latent and unconstrained input: `D:\thesis_final\outputs\latent_attacks\all_models_rerun_20260602_015314_seed42`
- Constrained input: `D:\thesis_final\outputs\latent_attacks\constrained_input_baselines_20260602_032651_seed42`
- Targeted latent PGD: `D:\thesis_final\outputs\latent_attacks\targeted_benign_pgd_gaussian_anticollapse_beta05_freebits01_20260529_173512_20260602_023404_seed42`

The CSV files in `results/he_idsr/` contain the same results in machine-readable form.
