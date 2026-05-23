# Repository Reorganization Changelog

No files were deleted. No Git commits were created.

## Moves

- `src/adversarial_attacks.py` -> `src/attack/adversarial_attacks.py`
- `src/validator.py` -> `src/attack/validator.py`
- `src/run_attacks.py` -> `src/attack/run_attacks.py`
- `src/run_attacks_8class_models.py` -> `src/attack/run_attacks_8class_models.py`
- `src/models.py` -> `src/classifiers/models.py`
- `src/baseline_experiments.py` -> `src/classifiers/baseline_experiments.py`
- `src/feature_groups.py` -> `src/preprocessing/feature_groups.py`
- `src/netdiffuser_categorization.py` -> `src/preprocessing/netdiffuser_categorization.py`
- `src/preprocessing.py` -> `src/preprocessing/pipeline.py`
- `src/analyze_attack_restarts.py` -> `src/evaluation/analyze_attack_restarts.py`
- `src/compact_exhibits.py` -> `src/evaluation/compact_exhibits.py`
- `src/delta_report.py` -> `src/evaluation/delta_report.py`
- `src/eda_figures_part1.py` -> `src/evaluation/eda_figures_part1.py`
- `src/eda_figures_part2.py` -> `src/evaluation/eda_figures_part2.py`
- `src/eda_tables.py` -> `src/evaluation/eda_tables.py`
- `src/export_slide_exhibits.py` -> `src/evaluation/export_slide_exhibits.py`
- `src/plot_attack_restarts.py` -> `src/evaluation/plot_attack_restarts.py`
- `src/run_validity_analysis.py` -> `src/evaluation/run_validity_analysis.py`
- `src/sample_exhibit.py` -> `src/evaluation/sample_exhibit.py`
- `src/validate_full_dataset.py` -> `src/evaluation/validate_full_dataset.py`
- `src/validity_analysis.py` -> `src/evaluation/validity_analysis.py`

## Files Already In Target Location

- `src/vae/cvae.py`
- `src/vae/dataset.py`
- `src/vae/train.py`
- `configs/cvae.yaml`

## Added Structure

- `src/attack/__init__.py`
- `src/classifiers/__init__.py`
- `src/preprocessing/__init__.py`
- `src/evaluation/__init__.py`
- `src/vae/__init__.py`
- `data/raw/.gitkeep`
- `data/processed/.gitkeep`
- `checkpoints/.gitkeep`

## Import Updates

- `src/vae/train.py`
  - `from cvae ...` -> `from src.vae.cvae ...`
  - `from dataset ...` -> `from src.vae.dataset ...`
  - `from feature_groups ...` -> `from src.preprocessing.feature_groups ...`
- `src/vae/dataset.py`
  - `from feature_groups ...` -> `from src.preprocessing.feature_groups ...`
- `src/preprocessing/pipeline.py`
  - `from feature_groups ...` -> `from src.preprocessing.feature_groups ...`
  - `from validator ...` -> `from src.attack.validator ...`
- `src/classifiers/baseline_experiments.py`
  - `from models ...` -> `from src.classifiers.models ...`
- `src/attack/adversarial_attacks.py`
  - `from models ...` -> `from src.classifiers.models ...`
- `src/attack/run_attacks.py`
  - `from adversarial_attacks ...` -> `from src.attack.adversarial_attacks ...`
- `src/attack/run_attacks_8class_models.py`
  - `from adversarial_attacks ...` -> `from src.attack.adversarial_attacks ...`
- `src/evaluation/validity_analysis.py`
  - `from validator ...` -> `from src.attack.validator ...`
- `src/evaluation/validate_full_dataset.py`
  - `from feature_groups ...` -> `from src.preprocessing.feature_groups ...`
  - `from validator ...` -> `from src.attack.validator ...`
- `src/evaluation/sample_exhibit.py`
  - `from feature_groups ...` -> `from src.preprocessing.feature_groups ...`
  - `from validator ...` -> `from src.attack.validator ...`
- `src/evaluation/run_validity_analysis.py`
  - `from feature_groups ...` -> `from src.preprocessing.feature_groups ...`
  - `from validity_analysis ...` -> `from src.evaluation.validity_analysis ...`
- `src/evaluation/eda_tables.py`
  - `from feature_groups ...` -> `from src.preprocessing.feature_groups ...`
  - `from validator ...` -> `from src.attack.validator ...`
- `src/evaluation/eda_figures_part2.py`
  - `from feature_groups ...` -> `from src.preprocessing.feature_groups ...`
  - `from netdiffuser_categorization ...` -> `from src.preprocessing.netdiffuser_categorization ...`
- `src/evaluation/eda_figures_part1.py`
  - `from feature_groups ...` -> `from src.preprocessing.feature_groups ...`
- `src/evaluation/delta_report.py`
  - `from adversarial_attacks ...` -> `from src.attack.adversarial_attacks ...`
  - `from feature_groups ...` -> `from src.preprocessing.feature_groups ...`
- `src/evaluation/analyze_attack_restarts.py`
  - `from feature_groups ...` -> `from src.preprocessing.feature_groups ...`
- `src/evaluation/compact_exhibits.py`
  - `from adversarial_attacks ...` -> `from src.attack.adversarial_attacks ...`
  - `from feature_groups ...` -> `from src.preprocessing.feature_groups ...`
  - `from validator ...` -> `from src.attack.validator ...`

## Uncertain / Left In Place

The following config-like or generated files were left in place because they are run/checkpoint metadata or existing generated artifacts rather than source configuration:

- `checkpoints/cvae/**/config.yaml`
- `checkpoints/cvae/**/metadata.json`
- `logs/baselines/**/run_config.json`
- `logs/baselines/**/history.json`
- `logs/baselines/**/metrics_test.json`
- `logs/baselines/**/leaderboard.json`
- `logs/class_counts.json`
- `results/**/*.json`

The following repo/editor metadata and documentation files were also left in place:

- `.claude/settings.json`
- `.claude/settings.local.json`
- `.vscode/settings.json`
- `config/run_manifest.json`
- `eda_preprocessing_prompt.md`
- `guide.md`
- `phase_plan.md`
- `progress.txt`
- `validator_explained.md`
- `work.md`

## Verification

- `python -c "import src.vae.cvae; import src.vae.dataset; import src.classifiers.models; import src.preprocessing.feature_groups; import src.attack.validator; import src.evaluation.validity_analysis; print('subpackage imports ok')"` passed.
- `python -c "import src.attack.adversarial_attacks; print('attack helpers import ok')"` passed.
- `pytest --collect-only` reported no collection failures, but exited with no tests collected.
