# Tree Model Summary: Random Forest and XGBoost

This report consolidates the Random Forest and XGBoost outputs for the binary, 8-class category, and 34-class fine-grained CICIoT2023 classification tasks. It is written as a thesis-ready summary of the evaluation results, model architecture, and training parameters.

## Source artifacts

| Artifact | Path |
|---|---|
| Training script | `src/classifiers/tree_baselines.py` |
| Combined tree summary | `results/tree_models_all_tasks_summary.json` |
| Training stdout/log | `results/tree_training_stdout.log` |
| Random Forest Binary report | `results/rf_binary_classification_report.json`, `results/rf_binary_classification_report.txt` |
| Random Forest 8-class category report | `results/rf_8class_classification_report.json`, `results/rf_8class_classification_report.txt` |
| Random Forest 34-class fine-grained report | `results/rf_34class_classification_report.json`, `results/rf_34class_classification_report.txt` |
| XGBoost Binary report | `results/xgb_binary_classification_report.json`, `results/xgb_binary_classification_report.txt` |
| XGBoost 8-class category report | `results/xgb_8class_classification_report.json`, `results/xgb_8class_classification_report.txt` |
| XGBoost 34-class fine-grained report | `results/xgb_34class_classification_report.json`, `results/xgb_34class_classification_report.txt` |
| Baseline review summary with ROC-AUC | `results/baseline_review/baseline_metrics_summary.csv`, `results/baseline_review/baseline_metrics_summary.md` |
| Per-class named metrics | `results/baseline_review/baseline_per_class_metrics.csv` |
| ROC-AUC plots | `results/baseline_review/roc_curves/*_{rf,xgb}_roc_auc.png` |
| Saved estimators | `models/rf_binary.pkl`, `models/rf_8class.pkl`, `models/rf_34class.pkl`, `models/xgb_binary.pkl`, `models/xgb_8class.pkl`, `models/xgb_34class.pkl` |

## Experimental setup

- Dataset representation: processed CICIoT2023 NumPy arrays in `data/processed`.
- Input dimensionality: 39 scaled numeric traffic features per sample.
- Training split: 3,100,958 samples; validation split: 442,994 samples; test split: 885,988 samples.
- Prediction tasks: binary Benign vs Attack, 8-class attack-category classification, and 34-class fine-grained attack-family classification.
- Model selection: validation macro-F1. This objective was used because it weights minority and majority classes equally at the class level, which is important under strong traffic-class imbalance.
- Class weighting: excluded. The stored reason is: "Class weights excluded because stratified sampling already balances the training set; applying original-distribution weights would over-penalize majority classes in the balanced sample." The class-weight arrays were still saved in each JSON report for reference, but no sample weights or class weights were applied during fitting.
- Random seed: 42 for Python, NumPy, scikit-learn Random Forest, and XGBoost estimators.
- XGBoost device resolution during this run: `cuda`; RF is CPU-only. The persisted XGBoost estimators were switched back to CPU for later inference in the review step.

## Model architecture and parameters

### Random Forest

The Random Forest baseline is an ensemble of independently trained decision trees implemented with `sklearn.ensemble.RandomForestClassifier`. Each tree is trained on a bootstrap sample of the training set and predicts a class-probability distribution from class frequencies in its terminal leaves. The ensemble prediction is the average probability distribution across all trees, followed by an argmax decision rule. This architecture captures nonlinear feature interactions without assuming a neural sequence or spatial structure, making it a strong tabular-data baseline for network-flow features.

Fixed RF settings used for every task:

| Parameter | Value | Role |
|---|---:|---|
| `max_features` | `sqrt` | At each split, only sqrt(39) candidate features are considered, reducing correlation among trees. |
| `max_samples` | `0.4` | Each tree uses 40% of the training rows under bootstrap sampling, bounding memory and increasing ensemble diversity. |
| `min_samples_leaf` | `25` | Leaves must contain at least 25 samples, limiting overfitting and controlling tree size. |
| `class_weight` | `None` | No class weighting was applied because the training sample was already stratified/balanced. |
| `n_jobs` | `-1` | Uses all available CPU workers. |
| `random_state` | `42` | Reproducible tree construction and bootstrap sampling. |
| Other sklearn defaults | default | Includes `criterion="gini"`, `bootstrap=True`, and `min_samples_split=2`. |

The tuned RF parameters were `n_estimators` and `max_depth`. The search grid was `(200, 20)`, `(300, 20)`, `(200, 30)`, and `(300, 30)`. The 34-class memory footprint motivated capping depth and using `min_samples_leaf=25`, since each scikit-learn tree stores class-value arrays at every node.

### XGBoost

The XGBoost baseline is a gradient-boosted decision-tree ensemble implemented with `xgboost.XGBClassifier`. Trees are added sequentially; each new tree is fitted to improve the current ensemble according to the task loss. For the binary task, the model optimizes logistic loss and outputs the attack probability. For the multiclass tasks, it optimizes a soft-probability objective and outputs a full class-probability vector. The final prediction is the class with maximum predicted probability.

Fixed XGBoost settings used for every task:

| Parameter | Value | Role |
|---|---:|---|
| `n_estimators` | `600` | Maximum number of boosting rounds. The actual effective number may be lower due to early stopping. |
| `early_stopping_rounds` | `30` | Stops boosting if validation loss does not improve for 30 rounds. |
| `tree_method` | `hist` | Histogram-based tree construction for efficient large-scale tabular training. |
| `device` | `cuda` during training, CPU after saving | GPU acceleration was used when available; saved models were reset to CPU for review inference. |
| `objective` | `binary:logistic` or `multi:softprob` | Binary logistic objective for 2 classes; multiclass probability objective for 8 and 34 classes. |
| `eval_metric` | `logloss` or `mlogloss` | Binary log-loss for the binary task; multiclass log-loss for 8-class and 34-class tasks. |
| `n_jobs` | `-1` | Uses all available CPU workers where applicable. |
| `random_state` | `42` | Reproducible model initialization and training behavior. |
| Class/sample weights | none | No class weights or sample weights were passed. |
| Other XGBoost defaults | default | Parameters such as `subsample`, `colsample_bytree`, and regularization terms were not overridden in the script. |

The tuned XGBoost parameters were `max_depth` and `learning_rate`. The search grid was `(max_depth=6, learning_rate=0.1)`, `(10, 0.1)`, `(6, 0.3)`, and `(10, 0.3)`. Validation macro-F1 selected the best configuration after training each candidate with validation-set early stopping.

## Selected hyperparameters

| Task | Model | Selected hyperparameters | Validation macro-F1 | Test loss |
|---|---|---|---:|---:|
| Binary | Random Forest | `n_estimators=300, max_depth=30` | 0.8135 | 0.0659 |
| Binary | XGBoost | `max_depth=10, learning_rate=0.1` | 0.8339 | 0.0612 |
| 8-class category | Random Forest | `n_estimators=200, max_depth=30` | 0.6645 | 0.3161 |
| 8-class category | XGBoost | `max_depth=10, learning_rate=0.1` | 0.7108 | 0.2948 |
| 34-class fine-grained | Random Forest | `n_estimators=200, max_depth=30` | 0.5982 | 0.5339 |
| 34-class fine-grained | XGBoost | `max_depth=10, learning_rate=0.1` | 0.6307 | 0.4931 |

## Test-set summary

| Task | Model | Accuracy | Macro precision | Macro recall | Macro F1 | Weighted F1 | ROC-AUC |
|---|---|---:|---:|---:|---:|---:|---:|
| Binary | Random Forest | 0.9718 | 0.8651 | 0.7763 | 0.8139 | 0.9698 | 0.9837 |
| Binary | XGBoost | 0.9733 | 0.8592 | 0.8140 | 0.8349 | 0.9724 | 0.9855 |
| 8-class category | Random Forest | 0.8555 | 0.8712 | 0.6400 | 0.6625 | 0.8503 | 0.9801 |
| 8-class category | XGBoost | 0.8614 | 0.8251 | 0.6777 | 0.7094 | 0.8577 | 0.9820 |
| 34-class fine-grained | Random Forest | 0.7719 | 0.7262 | 0.5939 | 0.5985 | 0.7611 | 0.9856 |
| 34-class fine-grained | XGBoost | 0.7822 | 0.7105 | 0.6191 | 0.6311 | 0.7749 | 0.9868 |

## Direct RF vs XGBoost comparison

| Task | Accuracy gain, XGB-RF | Macro-F1 gain, XGB-RF | Weighted-F1 gain, XGB-RF | ROC-AUC gain, XGB-RF |
|---|---:|---:|---:|---:|
| Binary | 0.0015 | 0.0210 | 0.0026 | 0.0019 |
| 8-class category | 0.0059 | 0.0469 | 0.0073 | 0.0019 |
| 34-class fine-grained | 0.0102 | 0.0326 | 0.0138 | 0.0012 |

Across all three tasks, XGBoost achieved the stronger aggregate result. The advantage was smallest for binary classification and largest for the 8-class task in macro-F1, suggesting that boosted trees improved minority-category recovery more than the bagged-tree ensemble. Both models show the expected degradation as the label space becomes more fine-grained: macro-F1 drops from binary to 8-class to 34-class because the classifier must distinguish increasingly similar attack families.

## Hyperparameter search details

| Task | Model | Candidate parameters | Validation macro-F1 | Fit seconds | Selected |
|---|---|---|---:|---:|---|
| Binary | Random Forest | `n_estimators=200, max_depth=20` | 0.8093 | 35.2 |  |
| Binary | Random Forest | `n_estimators=300, max_depth=20` | 0.8097 | 50.4 |  |
| Binary | Random Forest | `n_estimators=200, max_depth=30` | 0.8133 | 33.8 |  |
| Binary | Random Forest | `n_estimators=300, max_depth=30` | 0.8135 | 49.8 | yes |
| Binary | XGBoost | `max_depth=6, learning_rate=0.1` | 0.8295 | 4.8 |  |
| Binary | XGBoost | `max_depth=10, learning_rate=0.1` | 0.8339 | 5.9 | yes |
| Binary | XGBoost | `max_depth=6, learning_rate=0.3` | 0.8320 | 3.6 |  |
| Binary | XGBoost | `max_depth=10, learning_rate=0.3` | 0.8321 | 2.6 |  |
| 8-class category | Random Forest | `n_estimators=200, max_depth=20` | 0.6627 | 48.7 |  |
| 8-class category | Random Forest | `n_estimators=300, max_depth=20` | 0.6629 | 74.2 |  |
| 8-class category | Random Forest | `n_estimators=200, max_depth=30` | 0.6645 | 50.9 | yes |
| 8-class category | Random Forest | `n_estimators=300, max_depth=30` | 0.6644 | 74.5 |  |
| 8-class category | XGBoost | `max_depth=6, learning_rate=0.1` | 0.7036 | 34.6 |  |
| 8-class category | XGBoost | `max_depth=10, learning_rate=0.1` | 0.7108 | 49.6 | yes |
| 8-class category | XGBoost | `max_depth=6, learning_rate=0.3` | 0.7100 | 34.9 |  |
| 8-class category | XGBoost | `max_depth=10, learning_rate=0.3` | 0.7083 | 19.1 |  |
| 34-class fine-grained | Random Forest | `n_estimators=200, max_depth=20` | 0.5916 | 60.1 |  |
| 34-class fine-grained | Random Forest | `n_estimators=300, max_depth=20` | 0.5909 | 88.0 |  |
| 34-class fine-grained | Random Forest | `n_estimators=200, max_depth=30` | 0.5982 | 57.6 | yes |
| 34-class fine-grained | Random Forest | `n_estimators=300, max_depth=30` | 0.5979 | 88.9 |  |
| 34-class fine-grained | XGBoost | `max_depth=6, learning_rate=0.1` | 0.6267 | 239.0 |  |
| 34-class fine-grained | XGBoost | `max_depth=10, learning_rate=0.1` | 0.6307 | 173.4 | yes |
| 34-class fine-grained | XGBoost | `max_depth=6, learning_rate=0.3` | 0.6267 | 122.2 |  |
| 34-class fine-grained | XGBoost | `max_depth=10, learning_rate=0.3` | 0.6242 | 69.1 |  |

## Per-class metrics

### Binary

| Class | Support | RF precision | RF recall | RF F1 | RF OvR AUC | XGB precision | XGB recall | XGB F1 | XGB OvR AUC | Delta F1, XGB-RF |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Benign | 39,998 | 0.7508 | 0.5615 | 0.6425 |  | 0.7354 | 0.6388 | 0.6837 |  | 0.0412 |
| Attack | 845,990 | 0.9795 | 0.9912 | 0.9853 | 0.9837 | 0.9830 | 0.9891 | 0.9861 | 0.9855 | 0.0008 |

### 8-class category

| Class | Support | RF precision | RF recall | RF F1 | RF OvR AUC | XGB precision | XGB recall | XGB F1 | XGB OvR AUC | Delta F1, XGB-RF |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Benign | 39,998 | 0.7007 | 0.6363 | 0.6670 | 0.9835 | 0.6989 | 0.6947 | 0.6968 | 0.9856 | 0.0298 |
| BruteForce | 2,504 | 1.0000 | 0.1170 | 0.2095 | 0.9729 | 0.8267 | 0.2228 | 0.3511 | 0.9758 | 0.1415 |
| DDoS | 409,984 | 0.8766 | 0.9167 | 0.8962 | 0.9759 | 0.8802 | 0.9181 | 0.8987 | 0.9771 | 0.0026 |
| DoS | 133,755 | 0.7045 | 0.6075 | 0.6524 | 0.9537 | 0.7119 | 0.6188 | 0.6621 | 0.9559 | 0.0097 |
| Mirai | 119,992 | 0.9994 | 0.9960 | 0.9977 | 1.0000 | 0.9991 | 0.9977 | 0.9984 | 1.0000 | 0.0008 |
| Recon | 100,705 | 0.7823 | 0.8994 | 0.8367 | 0.9897 | 0.8100 | 0.8884 | 0.8474 | 0.9909 | 0.0107 |
| Spoofing | 74,291 | 0.9304 | 0.8708 | 0.8996 | 0.9951 | 0.9352 | 0.8825 | 0.9081 | 0.9962 | 0.0085 |
| Web | 4,759 | 0.9757 | 0.0761 | 0.1411 | 0.9702 | 0.7384 | 0.1982 | 0.3125 | 0.9745 | 0.1713 |

### 34-class fine-grained

| Class | Support | RF precision | RF recall | RF F1 | RF OvR AUC | XGB precision | XGB recall | XGB F1 | XGB OvR AUC | Delta F1, XGB-RF |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BACKDOOR_MALWARE | 615 | 0.9000 | 0.0146 | 0.0288 | 0.9760 | 0.3511 | 0.0537 | 0.0931 | 0.9801 | 0.0643 |
| BENIGN | 39,998 | 0.5813 | 0.7761 | 0.6647 | 0.9834 | 0.6206 | 0.7791 | 0.6909 | 0.9855 | 0.0262 |
| BROWSERHIJACKING | 1,126 | 0.9640 | 0.0950 | 0.1730 | 0.9748 | 0.6950 | 0.2833 | 0.4025 | 0.9786 | 0.2295 |
| COMMANDINJECTION | 1,034 | 0.9231 | 0.2089 | 0.3407 | 0.9819 | 0.6442 | 0.2592 | 0.3697 | 0.9850 | 0.0290 |
| DDOS-ACK_FRAGMENTATION | 39,996 | 0.9912 | 0.9886 | 0.9899 | 1.0000 | 0.9910 | 0.9902 | 0.9906 | 1.0000 | 0.0007 |
| DDOS-HTTP_FLOOD | 5,519 | 0.8963 | 0.7081 | 0.7912 | 0.9987 | 0.8875 | 0.7733 | 0.8265 | 0.9992 | 0.0353 |
| DDOS-ICMP_FLOOD | 40,000 | 0.9996 | 0.9987 | 0.9991 | 1.0000 | 0.9998 | 0.9989 | 0.9994 | 1.0000 | 0.0002 |
| DDOS-ICMP_FRAGMENTATION | 39,996 | 0.9899 | 0.9850 | 0.9874 | 1.0000 | 0.9874 | 0.9903 | 0.9889 | 1.0000 | 0.0014 |
| DDOS-PSHACK_FLOOD | 40,000 | 0.9995 | 0.9978 | 0.9987 | 1.0000 | 0.9995 | 0.9984 | 0.9990 | 1.0000 | 0.0003 |
| DDOS-RSTFINFLOOD | 40,000 | 1.0000 | 0.9988 | 0.9994 | 1.0000 | 0.9999 | 0.9990 | 0.9994 | 1.0000 | 0.0000 |
| DDOS-SLOWLORIS | 4,480 | 0.7325 | 0.9234 | 0.8169 | 0.9997 | 0.8674 | 0.9391 | 0.9018 | 0.9998 | 0.0849 |
| DDOS-SYNONYMOUSIP_FLOOD | 40,000 | 0.4594 | 0.6367 | 0.5337 | 0.9721 | 0.4579 | 0.6598 | 0.5406 | 0.9723 | 0.0069 |
| DDOS-SYN_FLOOD | 39,999 | 0.4891 | 0.2428 | 0.3245 | 0.9646 | 0.4868 | 0.2669 | 0.3448 | 0.9650 | 0.0203 |
| DDOS-TCP_FLOOD | 39,999 | 0.7002 | 0.5135 | 0.5925 | 0.9862 | 0.6900 | 0.5509 | 0.6127 | 0.9865 | 0.0202 |
| DDOS-UDP_FLOOD | 39,999 | 0.6496 | 0.8688 | 0.7434 | 0.9910 | 0.6982 | 0.7569 | 0.7264 | 0.9912 | -0.0170 |
| DDOS-UDP_FRAGMENTATION | 39,996 | 0.9918 | 0.9883 | 0.9900 | 1.0000 | 0.9900 | 0.9907 | 0.9903 | 1.0000 | 0.0003 |
| DICTIONARYBRUTEFORCE | 2,504 | 0.9920 | 0.1490 | 0.2590 | 0.9727 | 0.7323 | 0.2392 | 0.3606 | 0.9761 | 0.1016 |
| DNS_SPOOFING | 34,293 | 0.7691 | 0.7095 | 0.7381 | 0.9892 | 0.7674 | 0.7380 | 0.7524 | 0.9914 | 0.0143 |
| DOS-HTTP_FLOOD | 13,759 | 0.8734 | 0.9475 | 0.9089 | 0.9994 | 0.9025 | 0.9483 | 0.9248 | 0.9996 | 0.0159 |
| DOS-SYN_FLOOD | 39,999 | 0.5274 | 0.5853 | 0.5549 | 0.9759 | 0.5498 | 0.5540 | 0.5519 | 0.9766 | -0.0029 |
| DOS-TCP_FLOOD | 39,999 | 0.6155 | 0.7779 | 0.6873 | 0.9861 | 0.6264 | 0.7515 | 0.6833 | 0.9865 | -0.0040 |
| DOS-UDP_FLOOD | 39,998 | 0.7993 | 0.5289 | 0.6366 | 0.9909 | 0.7338 | 0.6711 | 0.7010 | 0.9911 | 0.0645 |
| MIRAI-GREETH_FLOOD | 39,997 | 0.9973 | 0.9915 | 0.9944 | 1.0000 | 0.9969 | 0.9949 | 0.9959 | 1.0000 | 0.0015 |
| MIRAI-GREIP_FLOOD | 39,998 | 0.9931 | 0.9960 | 0.9946 | 1.0000 | 0.9971 | 0.9974 | 0.9972 | 1.0000 | 0.0027 |
| MIRAI-UDPPLAIN | 39,997 | 0.9965 | 0.9961 | 0.9963 | 1.0000 | 0.9982 | 0.9975 | 0.9978 | 1.0000 | 0.0015 |
| MITM-ARPSPOOFING | 39,998 | 0.7588 | 0.7191 | 0.7384 | 0.9879 | 0.7861 | 0.7395 | 0.7620 | 0.9900 | 0.0237 |
| RECON-HOSTDISCOVERY | 25,735 | 0.6573 | 0.6916 | 0.6740 | 0.9872 | 0.6757 | 0.7378 | 0.7054 | 0.9901 | 0.0314 |
| RECON-OSSCAN | 18,793 | 0.4980 | 0.0983 | 0.1642 | 0.9629 | 0.4463 | 0.1733 | 0.2497 | 0.9660 | 0.0855 |
| RECON-PINGSWEEP | 432 | 0.0000 | 0.0000 | 0.0000 | 0.9632 | 0.2273 | 0.0231 | 0.0420 | 0.9651 | 0.0420 |
| RECON-PORTSCAN | 15,746 | 0.5222 | 0.2483 | 0.3366 | 0.9644 | 0.5049 | 0.3095 | 0.3837 | 0.9682 | 0.0472 |
| SQLINJECTION | 1,004 | 0.9149 | 0.0428 | 0.0818 | 0.9679 | 0.8125 | 0.0647 | 0.1199 | 0.9692 | 0.0381 |
| UPLOADING_ATTACK | 239 | 0.0000 | 0.0000 | 0.0000 | 0.9774 | 0.2273 | 0.0209 | 0.0383 | 0.9783 | 0.0383 |
| VULNERABILITYSCAN | 39,999 | 0.5084 | 0.7661 | 0.6112 | 0.9784 | 0.5568 | 0.7563 | 0.6414 | 0.9811 | 0.0303 |
| XSS | 741 | 0.0000 | 0.0000 | 0.0000 | 0.9789 | 0.2500 | 0.0432 | 0.0736 | 0.9797 | 0.0736 |

## Thesis interpretation

Random Forest and XGBoost serve as non-neural tabular baselines for the CICIoT2023 feature representation. Random Forest tests whether a bagged ensemble of decorrelated decision trees can separate attack classes using local feature thresholds. XGBoost tests a stronger sequential boosting hypothesis, where each additional tree corrects residual errors from the preceding ensemble. Because both methods operate directly on the 39 engineered/scaled tabular features, their performance is a useful reference point for evaluating whether neural models add value beyond high-capacity tree methods.

The results show that both tree models are competitive with, and in these outputs stronger than, the neural baselines in the baseline-review table. XGBoost is the best tree model on every task: binary macro-F1 is 0.8349, 8-class macro-F1 is 0.7094, and 34-class macro-F1 is 0.6311. Random Forest follows with 0.8139, 0.6625, and 0.5985 macro-F1 respectively. Weighted-F1 remains higher than macro-F1 because the test distribution contains very large attack classes; macro-F1 therefore gives the more conservative view of minority-class behavior.

The per-class metrics reveal where the remaining difficulty lies. High-volume and structurally distinctive classes such as Mirai and several DDoS fragmentation/flood classes obtain near-perfect F1 and ROC-AUC. Small or confusable classes are much harder: examples include Web and BruteForce in the 8-class task, and fine-grained classes such as BACKDOOR_MALWARE, RECON-PINGSWEEP, UPLOADING_ATTACK, and XSS in the 34-class task. XGBoost improves many minority-class F1 values, but not uniformly; this supports reporting macro-F1 and per-class tables rather than relying only on accuracy.

For thesis reporting, the most defensible conclusion is that boosted decision trees are the strongest classical baseline in this experiment. The XGBoost model combines histogram-based scalable tree learning, early stopping on validation loss, and macro-F1 hyperparameter selection, producing the best aggregate performance across binary, category-level, and fine-grained classification. Random Forest remains valuable as a stable bagging baseline, but its capped-depth trees and averaged voting are less effective than boosting for recovering difficult minority classes in the multiclass tasks.

## Notes and caveats

- The JSON reports include `test_loss` from predicted probabilities. During tree training, sklearn emitted a warning for XGBoost log-loss that some probability rows did not sum exactly to one, likely due to device/probability handling. Accuracy, precision, recall, F1, and ROC-AUC are unaffected by that warning and should be emphasized over log-loss for the reported comparison.
- Validation data was used for model selection and XGBoost early stopping. The reported final metrics are from the held-out test split.
- No class weights were used. This makes the tree baselines methodologically aligned with the existing neural baseline setup in this repository.
