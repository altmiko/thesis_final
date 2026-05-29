# Phase 2 Thesis Improvements and Defense Preparation

This file records what the supervisor and thesis defense panel may ask, and what should be added or strengthened in the final Phase 2 thesis report.

The current `P2_THESIS_REPORT.md` is a detailed methodology and artifact log. This file is different: it is a defense-facing improvement checklist. Its purpose is to help convert the detailed technical log into a report that is easier for a supervisor, examiner, or defense panel to evaluate.

## Core Defense Message

The central claim should be stated clearly and repeatedly:

> Conventional tabular adversarial attacks can appear highly successful under raw attack success rate, but many of those attacks become invalid when inverse-transformed and checked against network-domain constraints. Therefore, IDS robustness evaluation must report valid adversarial success, not only raw classifier misclassification.

The report should make this distinction impossible to miss:

| Metric | Meaning |
|---|---|
| `ASR_raw` | Attack success before checking whether the adversarial sample is physically/domain valid. |
| Validity rate | Fraction of adversarial samples that pass network-domain, protocol, and mask constraints. |
| `ASR_valid` | Fraction of adversarial samples that both fool the classifier and remain valid. |
| Gap | `ASR_raw - ASR_valid`; this shows how much raw attack success is inflated by invalid examples. |

## Most Important Missing Item

The biggest missing item is a formal threat model.

The final report should explicitly define:

| Threat-model component | What to specify |
|---|---|
| Attacker goal | Evasion or targeted misclassification, especially attack traffic classified as benign. |
| Attacker knowledge | White-box, gray-box, or assumed access to model gradients. |
| Attacker control | Which network-flow features are mutable, partially mutable, or immutable. |
| Constraints | Protocol validity, binary feature validity, packet/statistical consistency, perturbation mask compliance. |
| Success condition | A sample must fool the model and remain valid after inverse transformation. |
| Evaluation metric | Prefer `ASR_valid` over raw `ASR_raw`. |

Without this, the panel can ask: "What can the attacker actually change?" or "Are these adversarial examples realistic?"

## Likely Supervisor and Defense Panel Questions

### Dataset Questions

1. Why did you choose CICIoT2023?
2. Why is CICIoT2023 suitable for IoT intrusion detection?
3. What are the limitations or weaknesses of CICIoT2023?
4. Did you use the full dataset or a sampled subset?
5. If sampling was used, how do you know the sample is representative?
6. How many original rows were available, and how many were finally used?
7. How many features and labels were used?
8. Why use binary, 8-category, and 34-class tasks?
9. Which classes are most imbalanced?
10. How does class imbalance affect accuracy and macro-F1?

### Preprocessing Questions

1. Why was `RobustScaler` used instead of `StandardScaler` or `MinMaxScaler`?
2. Why was the scaler fitted only on the training split?
3. Why were extreme values clipped at the 99.99th percentile?
4. Why were integer and binary features rounded after cleaning?
5. Why were timestamp-like features removed or ignored?
6. Why not use SMOTE, ADASYN, or random oversampling?
7. Why keep the full 39-feature schema instead of only the selected top 24 features?
8. How were near-zero-IQR features handled?
9. What is the purpose of the perturbation mask?
10. How do you prevent data leakage during preprocessing?

### Validity Questions

1. What does a "valid adversarial example" mean in this thesis?
2. Who defined the 49 validation rules?
3. Are the validation rules complete?
4. Can a sample pass the validator and still be unrealistic?
5. Can a sample fail the validator but still represent possible network behavior?
6. Why is validation done after inverse-transforming from scaled space?
7. Why do input-space PGD/CW attacks show high raw ASR but zero valid ASR?
8. Which validation rules are violated most often by input-space attacks?
9. How do protocol constraints affect attack realism?
10. Why is clean-data validity important before evaluating attacks?

### Model Questions

1. Why use MLP, CNN, LSTM, CNN-LSTM, and DualPath models?
2. Why not use Random Forest, XGBoost, Transformer models, or classical IDS baselines?
3. Which model performed best on binary classification?
4. Which model performed best on 8-category classification?
5. Which model performed best on 34-class classification?
6. Why is macro-F1 important in addition to accuracy?
7. Why can a high-accuracy classifier still be vulnerable?
8. How were hyperparameters chosen?
9. Was early stopping used?
10. Were class weights used? If not, why not?

### VAE and Latent Attack Questions

1. Why use a VAE for adversarial attack generation?
2. Why train class-conditional VAEs instead of one global VAE?
3. Why use latent dimension 16?
4. What does the latent space represent?
5. How do you know the VAE learned realistic traffic structure?
6. Why do latent attacks usually have lower raw ASR than input attacks?
7. Why do latent attacks preserve protocol and mask constraints better?
8. What is the difference between pre-postprocess validity and postprocessed validity?
9. Does postprocessing artificially improve results?
10. How should VAE validity results be interpreted honestly?

### Adversarial Attack Questions

1. What attacks were used?
2. Why use PGD and CW?
3. Are PGD and CW appropriate for tabular IDS data?
4. What is the difference between input-space attacks and latent-space attacks?
5. What epsilon values were used and why?
6. What does attack success mean?
7. Why should `ASR_valid` be treated as more meaningful than `ASR_raw`?
8. Can an attacker modify every feature in real life?
9. What features are immutable or partially mutable?
10. What is the targeted benign attack and why is it important?

### Results and Evidence Questions

1. What is the main result in one sentence?
2. What is the strongest evidence supporting your claim?
3. Which attacks achieved the highest raw ASR?
4. Which attacks achieved the highest valid ASR?
5. Which models were most robust?
6. Which classes were easiest to attack?
7. Which classes were hardest to attack?
8. Are the differences statistically significant?
9. Did you use bootstrap confidence intervals or McNemar tests?
10. Did you run multiple random seeds?

### Limitations Questions

1. Is the validator complete?
2. Are flow-level features directly controllable by a real attacker?
3. Is the result dataset-specific?
4. Would the same method work on another IDS dataset?
5. Does the VAE generate truly realistic flows or only validator-compliant flows?
6. Does postprocessing hide decoder weaknesses?
7. Are the attacks white-box only?
8. Is the sample size large enough for every class?
9. Are rare classes underrepresented?
10. What would you improve with more time?

## Missing or Weak Items to Add to the Final Report

### 1. Research Questions

Add explicit research questions before methodology or at the start of Chapter 3.

Suggested questions:

1. Do conventional input-space adversarial attacks overestimate IDS vulnerability by producing invalid network-flow samples?
2. Can a validity-aware latent attack framework generate adversarial examples that better preserve network-domain constraints?
3. How does attack success change when evaluated using `ASR_valid` instead of raw `ASR_raw`?
4. Which IDS architectures are most affected by validity-aware adversarial attacks?

### 2. Formal Threat Model

Add a dedicated subsection.

Minimum content:

| Item | Thesis choice |
|---|---|
| Threat type | Evasion attack against trained IDS classifiers. |
| Attacker goal | Cause malicious traffic to be misclassified, especially as benign. |
| Attacker access | State whether experiments assume white-box gradients. |
| Feature control | Use mutable, partially mutable, and immutable feature groups. |
| Constraint set | Protocol validity, binary consistency, range constraints, relational rules, perturbation mask. |
| Success definition | Misclassification plus validity. |

### 3. Methodology Diagram

Add at least one full pipeline diagram.

Recommended diagram:

```text
Raw CICIoT2023
    -> cleaning and validation
    -> stratified train/val/test split
    -> train-only RobustScaler
    -> IDS classifier training
    -> input-space attacks
    -> inverse transform
    -> domain validator
    -> ASR_raw vs ASR_valid
```

Recommended latent framework diagram:

```text
Raw CICIoT2023
    -> class-specific training data
    -> class-conditional VAE
    -> latent PGD / latent CW
    -> decoder
    -> raw postprocess / repair
    -> validator
    -> valid adversarial evaluation
```

### 4. Preprocessing Decision Table

Add a table with this structure:

| Decision | Reason | Risk if not done | Artifact |
|---|---|---|---|
| Train-only `RobustScaler` | Handles heavy-tailed features and avoids leakage. | Scaling leakage or unstable perturbations. | `data/processed/scaler.pkl` |
| 99.99 percentile clipping | Limits extreme outliers while preserving most distributional structure. | Rare extreme values dominate scaling and inverse-transform behavior. | `config/run_manifest.json` |
| Stratified split | Preserves class proportions across train/val/test. | Rare classes may disappear from validation/test splits. | `data/processed/*.npy` |
| Domain validator | Separates valid attacks from invalid feature vectors. | Raw ASR may overstate realistic vulnerability. | `src/attack/validator.py` |
| Perturbation mask | Prevents attacks from changing immutable or restricted features. | Unrealistic adversarial examples. | `data/processed/perturbation_mask.npy` |
| No SMOTE/ADASYN | Avoids synthetic rows that may violate domain rules. | Oversampling could create invalid traffic samples. | `results/all_models_all_tasks_summary.json` |

### 5. Metric Definition Table

Add this before presenting results:

| Metric | Definition | Why it matters |
|---|---|---|
| Accuracy | Overall correct predictions. | General model performance. |
| Macro-F1 | Average F1 across classes. | Better for imbalanced classes. |
| `ASR_raw` | Attack success without validity filtering. | Shows apparent attack strength. |
| Validity rate | Fraction of adversarial samples passing constraints. | Shows realism of generated samples. |
| `ASR_valid` | Attack success among valid adversarial examples. | Main robustness metric. |
| Protocol validity | Whether protocol fields remain valid and consistent. | Critical for network realism. |
| Mask compliance | Whether immutable/restricted features were respected. | Measures threat-model compliance. |
| Joint validity | Combined protocol and mask validity. | Useful for latent attack comparison. |
| IDSR | In-distribution success or distance-related score, depending on final definition. | Must be defined clearly if reported. |

Important: define `IDSR` precisely in the final report before using it.

### 6. Main Evidence Table

Add a compact table that directly supports the thesis claim.

Suggested format:

| Attack type | `ASR_raw` | Validity | `ASR_valid` | Interpretation |
|---|---:|---:|---:|---|
| Input PGD | High | Near zero or zero | Zero or near zero | Strong-looking but invalid. |
| Input CW | High | Near zero or zero | Zero or near zero | Strong-looking but invalid. |
| Latent PGD | Moderate | High protocol/mask validity | Nonzero | More realistic attack space. |
| Latent CW | Moderate | High protocol/mask validity | Nonzero | More realistic attack space. |

This should be one of the most important tables in the report.

### 7. Per-Class Attack Analysis

Add per-class results.

Include:

1. Which source classes are easiest to attack.
2. Which source classes are hardest to attack.
3. Whether rare classes behave differently.
4. Whether Web, BruteForce, Mirai, DDoS, DoS, Recon, Spoofing, and Benign show different patterns.

Why this matters:

Panel members often distrust only-average results. Per-class analysis shows that the methodology was not hiding minority-class behavior.

### 8. Confusion Matrices

Add:

1. Best binary model confusion matrix.
2. Best 8-category model confusion matrix.
3. Best 34-class model confusion matrix, if readable.
4. Optional attacked-versus-clean confusion movement.

What to explain:

1. Which classes are confused naturally before attack.
2. Which classes become confused after attack.
3. Whether attacks mainly push malicious classes toward benign or toward other attack categories.

### 9. Ablation Study

Even a small ablation would strengthen the defense.

Useful ablations:

| Ablation | Purpose |
|---|---|
| Raw ASR vs valid ASR | Shows why validity filtering changes the conclusion. |
| Input-space vs latent-space attacks | Shows the value of the proposed framework. |
| PGD vs CW | Shows whether results are attack-specific. |
| With mask vs without mask | Shows importance of feature mutability constraints. |
| Full 39 features vs selected 24 features | Shows whether feature reduction changes conclusions, if results exist. |
| Pre-postprocess VAE validity vs postprocess validity | Shows honest decoder quality versus repaired validity. |

### 10. Statistical Evidence

The report should not rely only on point estimates.

Add or emphasize:

1. Bootstrap confidence intervals for ASR.
2. McNemar test for paired attack comparisons.
3. Number of tested samples per model and attack.
4. Whether differences are statistically significant.
5. Clear statement when results are not statistically significant.

Suggested wording:

> The McNemar tests did not show statistically significant differences between latent PGD and latent CW at the saved sample sizes. Therefore, the thesis does not claim one latent optimizer is universally superior; instead, it claims that the validity-preserving latent attack space changes the interpretation of adversarial robustness.

### 11. Reproducibility Box

Add a short reproducibility table.

| Item | Value to report |
|---|---|
| Dataset | CICIoT2023 |
| Raw file | `data/ciciot2023/ciciot2023_base.csv` |
| Processed rows | 4,429,940 |
| Features | 39 |
| Split | 70/10/20 stratified train/val/test |
| Scaler | `RobustScaler`, fitted on train only |
| Random seed | State seed values used in manifests and attack runs |
| Hardware | CPU/GPU used for major runs |
| Main code | `src/preprocessing`, `src/classifiers`, `src/attack`, `src/vae`, `src/evaluation` |
| Main manifests | `config/run_manifest.json`, `vae_run_manifest.json`, attack `config_snapshot.json` files |

### 12. Limitations Section

Add a serious limitations subsection.

Suggested limitations:

1. The validator is rule-based and may not cover every real network constraint.
2. A validator-passing row is not guaranteed to be executable as real traffic.
3. Some flow-level features may not be directly controllable by an attacker.
4. The main experiments are dataset-specific to CICIoT2023.
5. Rare classes such as Web and BruteForce have much smaller sample sizes.
6. Postprocessing can repair VAE outputs, so pre-postprocess and postprocess validity must be reported separately.
7. White-box attacks may overestimate attacker knowledge.
8. Multiple random seeds would strengthen reliability if not already done.

### 13. Contribution List

Add a concise contribution list.

Suggested contributions:

1. Built a validity-aware CICIoT2023 preprocessing pipeline.
2. Trained and evaluated multiple IDS classifiers across binary, 8-category, and 34-class tasks.
3. Demonstrated that conventional input-space adversarial attacks can inflate robustness risk by producing invalid network-flow samples.
4. Proposed a VAE-based latent attack framework for more validity-aware adversarial generation.
5. Evaluated adversarial examples using both raw attack success and validity-filtered attack success.

## High-Priority Figures to Include

| Figure | Why it matters |
|---|---|
| Methodology pipeline diagram | Gives panel a map of the whole system. |
| Raw ASR vs valid ASR bar chart | Shows the core thesis result visually. |
| Input attack vs latent attack comparison | Shows why the proposed framework matters. |
| Confusion matrix for best 8-class model | Shows baseline model behavior. |
| Per-class attack success chart | Shows attack behavior is not uniform. |
| VAE pre/post validity chart | Prevents overclaiming about decoder quality. |
| Feature importance or feature-selection chart | Helps justify feature behavior and selected features. |

## High-Priority Tables to Include

| Table | Purpose |
|---|---|
| Dataset summary | Rows, features, classes, split counts. |
| Preprocessing decisions | Shows methodological justification. |
| Metric definitions | Prevents confusion over `ASR_raw`, `ASR_valid`, validity, IDSR. |
| Classifier performance | Accuracy and macro-F1 across tasks/models. |
| Attack comparison | Input PGD/CW vs latent PGD/CW. |
| Validity rule summary | Shows what makes examples valid or invalid. |
| Limitation table | Shows honest awareness of scope. |

## Specific Things to Be Careful About

### Do Not Overclaim VAE Validity

The VAE summary includes both pre-postprocess and postprocessed validity.

Use this interpretation:

| Validity type | Interpretation |
|---|---|
| Pre-postprocess validity | Honest decoder validity immediately after inverse transform. |
| Postprocessed validity | Validity after repair or domain postprocessing. |

Do not write that the VAE naturally generates 100% valid samples unless referring specifically to postprocessed outputs.

### Do Not Use Only Accuracy

Because the dataset is imbalanced, accuracy can hide poor minority-class performance.

Always report:

1. Accuracy.
2. Macro-F1.
3. Per-class results where possible.
4. Confusion matrices where possible.

### Do Not Let Raw ASR Become the Main Claim

Raw ASR is useful as a contrast, not as the main robustness conclusion.

The main claim should be based on:

1. Inverse-transformed adversarial samples.
2. Domain validity.
3. Perturbation mask compliance.
4. `ASR_valid`.

### Clarify Artifact Drift

The report must explain that older artifacts may differ from current final outputs.

Known issue:

| Artifact | Issue | How to handle |
|---|---|---|
| `tables/T4_clean_validity.md` | Older clean-validity table reports 81.3724%. | Treat as superseded by current 100% processed/full validation reports. |

Use current authoritative files:

1. `data/processed/processed_data_validity_report.json`
2. `results/validation/full_dataset_validation_report.txt`
3. `config/run_manifest.json`

## Suggested Defense Answers

### If asked: Why RobustScaler?

Network-flow features are heavy-tailed and contain extreme values. `StandardScaler` is sensitive to extreme values because it uses mean and standard deviation, while `MinMaxScaler` can compress most observations if a few large values exist. `RobustScaler` uses median and IQR, so it is more stable for skewed traffic features. It was fitted only on the training split to avoid data leakage.

### If asked: Why not SMOTE?

The thesis focuses on domain-valid network-flow samples. Synthetic oversampling methods such as SMOTE or ADASYN interpolate feature vectors but do not guarantee protocol consistency, integer validity, binary validity, or relational constraints. Adding such samples could weaken the clean-data validity guarantee.

### If asked: Why is ASR_valid needed?

Raw ASR only tells us that a classifier prediction changed. It does not tell us whether the modified feature vector represents plausible network traffic. `ASR_valid` is stricter because it counts only attacks that both fool the classifier and pass domain constraints after inverse transformation.

### If asked: Why use a VAE?

The VAE provides a learned latent space for each traffic class. Instead of perturbing scaled tabular features directly, latent attacks perturb a representation that is decoded back into the data manifold. This usually lowers raw ASR but improves constraint preservation, which is more aligned with realistic adversarial evaluation.

### If asked: Are your adversarial examples real packets?

They are not packet captures. They are flow-level feature vectors checked against network-domain constraints. Therefore, the claim should be stated carefully: the framework improves validity at the feature-flow level, but it does not prove that every adversarial example can be directly executed as real traffic.

### If asked: What is your main contribution?

The main contribution is showing that adversarial robustness claims for tabular IDS can change drastically after domain validity is enforced. The work also proposes and evaluates a VAE-based latent attack framework that better respects feature constraints than unconstrained input-space attacks.

## Final Priority Checklist

Before thesis submission or defense, prioritize:

1. Add a formal threat model.
2. Add metric definitions, especially `ASR_raw`, `ASR_valid`, validity rate, joint validity, and IDSR.
3. Add methodology and latent-framework diagrams.
4. Add the main evidence table comparing input-space and latent-space attacks.
5. Add per-class attack analysis.
6. Add confusion matrices.
7. Add limitations.
8. Add a reproducibility box.
9. Clarify artifact drift around older validity tables.
10. Avoid overclaiming postprocessed VAE validity.

