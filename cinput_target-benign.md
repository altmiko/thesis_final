# Constrained-Input Target-Benign Inverse-Transformed Samples

Source run: `outputs\latent_attacks\constrained_input_baselines_20260602_032651_seed42`.

The feature tables below are regenerated from selected `per_sample_results.csv` sample IDs because the constrained-input runner saves metadata, not the full adversarial vectors. Deltas are adversarial minus clean after `scaler.inverse_transform`.

Selection rule: candidate rows are successful and joint-valid in the source run; up to 4 diverse rows per model are regenerated for each attack; the displayed rows are the lowest scaled-L2 regenerated successful joint-valid examples per attack, with fallback to the best regenerated rows if needed.

## Attack Parameters

| Attack | Parameters |
| --- | --- |
| cinput-pgd-target-benign | epsilon=0.5, alpha=0.05, steps=40, random_start=True, targeted=True, target_class=Benign |
| cinput-cw-target-benign | lambda_conf=1, kappa=0, iterations=200, lr=0.01, convergence_threshold=1.0000e-05, targeted=True, target_class=Benign |

## Sample Summary

| Attack | Goal | Target | Model | Rank | Sample | Source -> After | Goal Success | Joint Valid | Raw G1-G8 | L2 scaled | Top raw feature changes |
| --- | --- | --- | --- | ---: | ---: | --- | --- | --- | --- | ---: | --- |
| cinput-pgd-target-benign | target-benign | Benign | MLP | 1 | 531 | Recon -> Benign | True | True | True | 0.815731 | `IAT` -0.000463576; `fin_count` -0.3; `ack_count` -2.1; `rst_count` 0.3; `ack_flag_number` -0.3 |
| cinput-pgd-target-benign | target-benign | Benign | CNN | 2 | 77 | Recon -> Benign | True | True | True | 1.03438 | `Header_Length` 6.04; `IAT` 0.000463576; `Min` 2.00317; `fin_count` -0.3; `ack_flag_number` -0.3 |
| cinput-cw-target-benign | target-benign | Benign | MLP | 1 | 27 | Spoofing -> Benign | True | True | True | 0.465821 | `Tot sum` 18163.2; `Variance` -9703.28; `Number` 13; `Max` 74.3638; `Header_Length` -0.419894 |
| cinput-cw-target-benign | target-benign | Benign | CNN-LSTM | 2 | 130 | Recon -> Benign | True | True | True | 0.485937 | `ack_flag_number` -0.299566; `rst_count` -0.245858; `ack_count` -1.16774; `Min` -0.807606; `Std` -28.0849 |

## cinput-pgd-target-benign / MLP / sample 531

Goal `target-benign` target `Benign`; source `Recon`; regenerated prediction `Recon` -> `Benign`; goal_success=True; protocol_valid=True; mask_valid=True; raw_g1g8_valid=True; joint_valid=True; scaled L2=0.815731.

| # | Feature | Clean raw | C-input raw | Delta raw | Clean scaled | C-input scaled | Delta scaled |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | Header_Length | 25.6 | 25.8229 | 0.222931 | 0.47351 | 0.491965 | 0.0184546 |
| 1 | Protocol Type | 6 | 6 | 0 | 0 | 0 | 0 |
| 2 | Time_To_Live | 136.9 | 136.9 | 0 | 38.1675 | 38.1675 | 0 |
| 3 | Rate | 103.688 | -0.000354005 | -103.688 | -0.25884 | -0.263029 | -0.0041891 |
| 4 | fin_flag_number | 0 | 0 | 0 | 0 | 0 | 0 |
| 5 | syn_flag_number | 0 | 0 | 0 | 0 | 0 | 0 |
| 6 | rst_flag_number | 0 | 0 | 0 | 0 | 0 | 0 |
| 7 | psh_flag_number | 0 | 0 | 0 | 0 | 0 | 0 |
| 8 | ack_flag_number | 1 | 0.7 | -0.3 | 1 | 0.7 | -0.3 |
| 9 | ece_flag_number | 0 | 0 | 0 | 0 | 0 | 0 |
| 10 | cwr_flag_number | 0 | 0 | 0 | 0 | 0 | 0 |
| 11 | ack_count | 8 | 5.9 | -2.1 | 1.14286 | 0.842857 | -0.3 |
| 12 | syn_count | 0 | 0 | 0 | 0 | 0 | 0 |
| 13 | fin_count | 1 | 0.7 | -0.3 | 1 | 0.7 | -0.3 |
| 14 | rst_count | 0 | 0.3 | 0.3 | 0 | 0.3 | 0.3 |
| 15 | HTTP | 0 | 0 | 0 | 0 | 0 | 0 |
| 16 | HTTPS | 1 | 1 | 0 | 1 | 1 | 0 |
| 17 | DNS | 0 | 0 | 0 | 0 | 0 | 0 |
| 18 | Telnet | 0 | 0 | 0 | 0 | 0 | 0 |
| 19 | SMTP | 0 | 0 | 0 | 0 | 0 | 0 |
| 20 | SSH | 0 | 0 | 0 | 0 | 0 | 0 |
| 21 | IRC | 0 | 0 | 0 | 0 | 0 | 0 |
| 22 | TCP | 1 | 1 | 0 | 0 | 0 | 0 |
| 23 | UDP | 0 | 0 | 0 | 0 | 0 | 0 |
| 24 | DHCP | 0 | 0 | 0 | 0 | 0 | 0 |
| 25 | ARP | 0 | 0 | 0 | 0 | 0 | 0 |
| 26 | ICMP | 0 | 0 | 0 | 0 | 0 | 0 |
| 27 | IGMP | 0 | 0 | 0 | 0 | 0 | 0 |
| 28 | IPv | 1 | 1 | 0 | 0 | 0 | 0 |
| 29 | LLC | 1 | 1 | 0 | 0 | 0 | 0 |
| 30 | Tot sum | 861 | 178.156 | -682.844 | -0.105008 | -0.118961 | -0.013953 |
| 31 | Min | 60 | 59.3855 | -0.614529 | 0 | -0.102421 | -0.102421 |
| 32 | Max | 156 | 59.3855 | -96.6145 | -0.0571992 | -0.15248 | -0.0952806 |
| 33 | AVG | 86.1 | 59.3855 | -26.7145 | 0.00637065 | -0.0452018 | -0.0515724 |
| 34 | Std | 32.1885 | -2.8623e-08 | -32.1885 | 0.13723 | -0.0156761 | -0.152906 |
| 35 | Tot size | 86.1 | 59.3855 | -26.7145 | 0.00637065 | -0.0452018 | -0.0515724 |
| 36 | IAT | 0.0112251 | 0.0107615 | -0.000463576 | 11.9366 | 11.4366 | -0.5 |
| 37 | Number | 10 | 3 | -7 | -1 | -1.07778 | -0.0777777 |
| 38 | Variance | 1036.1 | -2.0227e-07 | -1036.1 | 0.0231344 | -0.000245739 | -0.0233802 |

## cinput-pgd-target-benign / CNN / sample 77

Goal `target-benign` target `Benign`; source `Recon`; regenerated prediction `Recon` -> `Benign`; goal_success=True; protocol_valid=True; mask_valid=True; raw_g1g8_valid=True; joint_valid=True; scaled L2=1.03438.

| # | Feature | Clean raw | C-input raw | Delta raw | Clean scaled | C-input scaled | Delta scaled |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | Header_Length | 27.2 | 33.24 | 6.04 | 0.60596 | 1.10596 | 0.5 |
| 1 | Protocol Type | 6 | 6 | 0 | 0 | 0 | 0 |
| 2 | Time_To_Live | 131 | 131 | 0 | 35.0785 | 35.0785 | 0 |
| 3 | Rate | 366.101 | -0.000354005 | -366.101 | -0.248238 | -0.263029 | -0.0147908 |
| 4 | fin_flag_number | 0 | 0.0754316 | 0.0754316 | 0 | 0.0754316 | 0.0754316 |
| 5 | syn_flag_number | 0 | 0.3 | 0.3 | 0 | 0.3 | 0.3 |
| 6 | rst_flag_number | 0 | 0 | 0 | 0 | 0 | 0 |
| 7 | psh_flag_number | 0 | 0 | 0 | 0 | 0 | 0 |
| 8 | ack_flag_number | 1 | 0.7 | -0.3 | 1 | 0.7 | -0.3 |
| 9 | ece_flag_number | 0 | 0 | 0 | 0 | 0 | 0 |
| 10 | cwr_flag_number | 0 | 0 | 0 | 0 | 0 | 0 |
| 11 | ack_count | 8 | 8.53275 | 0.532753 | 1.14286 | 1.21896 | 0.0761075 |
| 12 | syn_count | 0 | 0 | 0 | 0 | 0 | 0 |
| 13 | fin_count | 1 | 0.7 | -0.3 | 1 | 0.7 | -0.3 |
| 14 | rst_count | 0 | 0 | 0 | 0 | 0 | 0 |
| 15 | HTTP | 0 | 0 | 0 | 0 | 0 | 0 |
| 16 | HTTPS | 1 | 1 | 0 | 1 | 1 | 0 |
| 17 | DNS | 0 | 0 | 0 | 0 | 0 | 0 |
| 18 | Telnet | 0 | 0 | 0 | 0 | 0 | 0 |
| 19 | SMTP | 0 | 0 | 0 | 0 | 0 | 0 |
| 20 | SSH | 0 | 0 | 0 | 0 | 0 | 0 |
| 21 | IRC | 0 | 0 | 0 | 0 | 0 | 0 |
| 22 | TCP | 1 | 1 | 0 | 0 | 0 | 0 |
| 23 | UDP | 0 | 0 | 0 | 0 | 0 | 0 |
| 24 | DHCP | 0 | 0 | 0 | 0 | 0 | 0 |
| 25 | ARP | 0 | 0 | 0 | 0 | 0 | 0 |
| 26 | ICMP | 0 | 0 | 0 | 0 | 0 | 0 |
| 27 | IGMP | 0 | 0 | 0 | 0 | 0 | 0 |
| 28 | IPv | 1 | 1 | 0 | 0 | 0 | 0 |
| 29 | LLC | 1 | 1 | 0 | 0 | 0 | 0 |
| 30 | Tot sum | 1914 | 680.043 | -1233.96 | -0.0834917 | -0.108706 | -0.0252142 |
| 31 | Min | 66 | 68.0032 | 2.00317 | 1 | 1.33386 | 0.333862 |
| 32 | Max | 573 | 457.211 | -115.789 | 0.354043 | 0.239854 | -0.11419 |
| 33 | AVG | 191.4 | 68.0042 | -123.396 | 0.209652 | -0.0285633 | -0.238216 |
| 34 | Std | 171.221 | 194.604 | 23.3836 | 0.797677 | 0.908756 | 0.111079 |
| 35 | Tot size | 191.4 | 68.0042 | -123.396 | 0.209652 | -0.0285633 | -0.238216 |
| 36 | IAT | 0.00273159 | 0.00319516 | 0.000463576 | 2.77575 | 3.27575 | 0.5 |
| 37 | Number | 10 | 10 | 0 | -1 | -1 | 0 |
| 38 | Variance | 29316.5 | 37870.8 | 8554.29 | 0.661297 | 0.854329 | 0.193032 |

## cinput-cw-target-benign / MLP / sample 27

Goal `target-benign` target `Benign`; source `Spoofing`; regenerated prediction `Spoofing` -> `Benign`; goal_success=True; protocol_valid=True; mask_valid=True; raw_g1g8_valid=True; joint_valid=True; scaled L2=0.465821.

| # | Feature | Clean raw | C-input raw | Delta raw | Clean scaled | C-input scaled | Delta scaled |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | Header_Length | 32 | 31.5801 | -0.419894 | 1.00331 | 0.968552 | -0.0347595 |
| 1 | Protocol Type | 6 | 6 | 0 | 0 | 0 | 0 |
| 2 | Time_To_Live | 60.5 | 60.5 | 0 | -1.83246 | -1.83246 | 0 |
| 3 | Rate | 537.546 | 531.241 | -6.30444 | -0.241312 | -0.241566 | -0.000254706 |
| 4 | fin_flag_number | 0 | 0.000431106 | 0.000431106 | 0 | 0.000431106 | 0.000431106 |
| 5 | syn_flag_number | 0 | 0 | 0 | 0 | 0 | 0 |
| 6 | rst_flag_number | 0 | 0 | 0 | 0 | 0 | 0 |
| 7 | psh_flag_number | 0 | 0 | 0 | 0 | 0 | 0 |
| 8 | ack_flag_number | 1 | 0.985307 | -0.0146927 | 1 | 0.985307 | -0.0146927 |
| 9 | ece_flag_number | 0 | 0 | 0 | 0 | 0 | 0 |
| 10 | cwr_flag_number | 0 | 0 | 0 | 0 | 0 | 0 |
| 11 | ack_count | 10 | 9.93902 | -0.0609852 | 1.42857 | 1.41986 | -0.00871217 |
| 12 | syn_count | 0 | 0 | 0 | 0 | 0 | 0 |
| 13 | fin_count | 0 | 0.0238168 | 0.0238168 | 0 | 0.0238168 | 0.0238168 |
| 14 | rst_count | 0 | 0 | 0 | 0 | 0 | 0 |
| 15 | HTTP | 0 | 0 | 0 | 0 | 0 | 0 |
| 16 | HTTPS | 1 | 1 | 0 | 1 | 1 | 0 |
| 17 | DNS | 0 | 0 | 0 | 0 | 0 | 0 |
| 18 | Telnet | 0 | 0 | 0 | 0 | 0 | 0 |
| 19 | SMTP | 0 | 0 | 0 | 0 | 0 | 0 |
| 20 | SSH | 0 | 0 | 0 | 0 | 0 | 0 |
| 21 | IRC | 0 | 0 | 0 | 0 | 0 | 0 |
| 22 | TCP | 1 | 1 | 0 | 0 | 0 | 0 |
| 23 | UDP | 0 | 0 | 0 | 0 | 0 | 0 |
| 24 | DHCP | 0 | 0 | 0 | 0 | 0 | 0 |
| 25 | ARP | 0 | 0 | 0 | 0 | 0 | 0 |
| 26 | ICMP | 0 | 0 | 0 | 0 | 0 | 0 |
| 27 | IGMP | 0 | 0 | 0 | 0 | 0 | 0 |
| 28 | IPv | 1 | 1 | 0 | 0 | 0 | 0 |
| 29 | LLC | 1 | 1 | 0 | 0 | 0 | 0 |
| 30 | Tot sum | 13692 | 31855.2 | 18163.2 | 0.157175 | 0.528314 | 0.371139 |
| 31 | Min | 66 | 65.9423 | -0.0576553 | 1 | 0.990391 | -0.00960922 |
| 32 | Max | 4410 | 4484.36 | 74.3638 | 4.13807 | 4.2114 | 0.0733371 |
| 33 | AVG | 1369.2 | 1385.01 | 15.8067 | 2.4834 | 2.51391 | 0.030515 |
| 34 | Std | 1593.53 | 1590.48 | -3.04744 | 7.55411 | 7.53963 | -0.0144763 |
| 35 | Tot size | 1369.2 | 1385.01 | 15.8067 | 2.4834 | 2.51391 | 0.030515 |
| 36 | IAT | 0.0018893 | 0.00191597 | 2.6669e-05 | 1.86729 | 1.89605 | 0.0287639 |
| 37 | Number | 10 | 23 | 13 | -1 | -0.855556 | 0.144444 |
| 38 | Variance | 2.5393e+06 | 2.5296e+06 | -9703.28 | 57.3014 | 57.0824 | -0.21896 |

## cinput-cw-target-benign / CNN-LSTM / sample 130

Goal `target-benign` target `Benign`; source `Recon`; regenerated prediction `Recon` -> `Benign`; goal_success=True; protocol_valid=True; mask_valid=True; raw_g1g8_valid=True; joint_valid=True; scaled L2=0.485937.

| # | Feature | Clean raw | C-input raw | Delta raw | Clean scaled | C-input scaled | Delta scaled |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | Header_Length | 29.2 | 29.0243 | -0.175674 | 0.771523 | 0.756981 | -0.0145426 |
| 1 | Protocol Type | 6 | 6 | 0 | 0 | 0 | 0 |
| 2 | Time_To_Live | 161 | 161 | 0 | 50.7852 | 50.7852 | 0 |
| 3 | Rate | 247.943 | 1590.58 | 1342.64 | -0.253012 | -0.198768 | 0.054244 |
| 4 | fin_flag_number | 0 | 0.0623815 | 0.0623815 | 0 | 0.0623815 | 0.0623815 |
| 5 | syn_flag_number | 0 | 0 | 0 | 0 | 0 | 0 |
| 6 | rst_flag_number | 0 | 0.0513411 | 0.0513411 | 0 | 0.0513411 | 0.0513411 |
| 7 | psh_flag_number | 0 | 0 | 0 | 0 | 0 | 0 |
| 8 | ack_flag_number | 1 | 0.700434 | -0.299566 | 1 | 0.700434 | -0.299566 |
| 9 | ece_flag_number | 0 | 0 | 0 | 0 | 0 | 0 |
| 10 | cwr_flag_number | 0 | 0 | 0 | 0 | 0 | 0 |
| 11 | ack_count | 8 | 6.83226 | -1.16774 | 1.14286 | 0.976037 | -0.16682 |
| 12 | syn_count | 2 | 1.96539 | -0.0346074 | 2 | 1.96539 | -0.0346074 |
| 13 | fin_count | 0 | 0 | 0 | 0 | 0 | 0 |
| 14 | rst_count | 1 | 0.754142 | -0.245858 | 1 | 0.754142 | -0.245858 |
| 15 | HTTP | 0 | 0 | 0 | 0 | 0 | 0 |
| 16 | HTTPS | 1 | 1 | 0 | 1 | 1 | 0 |
| 17 | DNS | 0 | 0 | 0 | 0 | 0 | 0 |
| 18 | Telnet | 0 | 0 | 0 | 0 | 0 | 0 |
| 19 | SMTP | 0 | 0 | 0 | 0 | 0 | 0 |
| 20 | SSH | 0 | 0 | 0 | 0 | 0 | 0 |
| 21 | IRC | 0 | 0 | 0 | 0 | 0 | 0 |
| 22 | TCP | 1 | 1 | 0 | 0 | 0 | 0 |
| 23 | UDP | 0 | 0 | 0 | 0 | 0 | 0 |
| 24 | DHCP | 0 | 0 | 0 | 0 | 0 | 0 |
| 25 | ARP | 0 | 0 | 0 | 0 | 0 | 0 |
| 26 | ICMP | 0 | 0 | 0 | 0 | 0 | 0 |
| 27 | IGMP | 0 | 0 | 0 | 0 | 0 | 0 |
| 28 | IPv | 1 | 1 | 0 | 0 | 0 | 0 |
| 29 | LLC | 1 | 1 | 0 | 0 | 0 | 0 |
| 30 | Tot sum | 732 | 796.463 | 64.4633 | -0.107644 | -0.106327 | 0.00131722 |
| 31 | Min | 60 | 59.1924 | -0.807606 | 0 | -0.134601 | -0.134601 |
| 32 | Max | 156 | 61.4828 | -94.5172 | -0.0571992 | -0.150411 | -0.0932122 |
| 33 | AVG | 73.2 | 61.2664 | -11.9336 | -0.0185328 | -0.0415706 | -0.0230378 |
| 34 | Std | 29.2301 | 1.1452 | -28.0849 | 0.123176 | -0.010236 | -0.133412 |
| 35 | Tot size | 73.2 | 61.2664 | -11.9336 | -0.0185328 | -0.0415706 | -0.0230378 |
| 36 | IAT | 0.00475659 | 0.00474772 | -8.8747e-06 | 4.95987 | 4.9503 | -0.00957203 |
| 37 | Number | 10 | 13 | 3 | -1 | -0.966667 | 0.0333334 |
| 38 | Variance | 854.4 | 1.31148 | -853.089 | 0.0190343 | -0.000216145 | -0.0192504 |
