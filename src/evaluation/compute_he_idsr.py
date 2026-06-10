from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from attack import run_all_models_attack_rerun as all_models_run  # noqa: E402
from attack import run_constrained_input_baselines as constrained_run  # noqa: E402
from attack.adversarial_attacks import load_model  # noqa: E402
from attack.latent_gmm import LatentGMMPrior, fit_or_load_latent_gmm  # noqa: E402
from attack.latent_infra import (  # noqa: E402
    AttackRouter,
    PerturbationMask,
    ProtocolValidator,
    load_split,
    set_global_seed,
)
from attack.latent_restarts import (  # noqa: E402
    build_latent_restart_initializers,
    class_float_value,
    parse_class_float_map,
    strategy_uses_gmm,
)
from preprocessing.feature_groups import FEATURE_NAMES  # noqa: E402
from vae.config import CLASS_TO_ID, CLASSES  # noqa: E402


GAUSSIAN_RUN_TAG = "gaussian_anticollapse_beta05_freebits01_20260529_173512"
SOURCE_CLASSES = [name for name in CLASSES if name != "Benign"]
DEFAULT_ALL_MODELS_RUN = (
    REPO_ROOT / "outputs" / "latent_attacks" / "all_models_rerun_20260602_015314_seed42"
)
DEFAULT_CONSTRAINED_RUN = (
    REPO_ROOT
    / "outputs"
    / "latent_attacks"
    / "constrained_input_baselines_20260602_032651_seed42"
)
DEFAULT_TARGETED_LATENT_RUN = (
    REPO_ROOT
    / "outputs"
    / "latent_attacks"
    / (
        "targeted_benign_pgd_gaussian_anticollapse_beta05_freebits01_"
        "20260529_173512_20260602_023404_seed42"
    )
)
DEFAULT_TARGETED_LATENT_CW_RUN = (
    REPO_ROOT
    / "outputs"
    / "latent_attacks"
    / (
        "targeted_benign_cw_gaussian_anticollapse_beta05_freebits01_"
        "20260529_173512_20260610_213313_seed42"
    )
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "he_idsr"
DEFAULT_REPORT = REPO_ROOT / "IDSR_REPORT.md"

OUTCOME_COLUMNS = [
    "attack_family",
    "attack_method",
    "attack_goal",
    "classifier",
    "classifier_tag",
    "attack_class",
    "sample_id",
    "evasion",
    "joint_valid",
    "mahalanobis_outlier",
    "saved_evasion",
    "saved_joint_valid",
    "evasion_match",
    "validity_match",
    "source_run",
]


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _as_bool(values: Any) -> np.ndarray:
    series = pd.Series(values)
    if pd.api.types.is_bool_dtype(series):
        return series.to_numpy(dtype=np.bool_)
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map({"true": True, "false": False, "1": True, "0": False})
        .fillna(False)
        .to_numpy(dtype=np.bool_)
    )


def _model_specs(config: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {
            "tag": str(spec["tag"]),
            "label": str(spec["label"]),
            "checkpoint": str(spec["checkpoint"]),
        }
        for spec in config["models"]
    ]


def _saved_cell(
    saved: pd.DataFrame,
    *,
    model_tag: str,
    attack: str | None,
    class_name: str,
) -> pd.DataFrame:
    mask = (saved["model_tag"] == model_tag) & (saved["source_class"] == class_name)
    if attack is not None:
        attack_col = "attack_type" if "attack_type" in saved.columns else "attack"
        mask &= saved[attack_col] == attack
    return saved.loc[mask].copy()


def _outcome_frame(
    *,
    family: str,
    attack: str,
    attack_goal: str,
    model_label: str,
    model_tag: str,
    class_name: str,
    sample_ids: np.ndarray,
    evasion: np.ndarray,
    joint_valid: np.ndarray,
    outlier: np.ndarray,
    saved_evasion: np.ndarray,
    saved_joint_valid: np.ndarray,
    source_run: Path,
) -> pd.DataFrame:
    n = len(sample_ids)
    arrays = [evasion, joint_valid, outlier, saved_evasion, saved_joint_valid]
    if any(len(values) != n for values in arrays):
        raise ValueError(
            f"Length mismatch for {model_tag}/{attack}/{class_name}: "
            f"sample_ids={n}, arrays={[len(values) for values in arrays]}"
        )

    return pd.DataFrame(
        {
            "attack_family": family,
            "attack_method": attack,
            "attack_goal": attack_goal,
            "classifier": model_label,
            "classifier_tag": model_tag,
            "attack_class": class_name,
            "sample_id": sample_ids.astype(np.int64),
            "evasion": evasion.astype(np.bool_),
            "joint_valid": joint_valid.astype(np.bool_),
            "mahalanobis_outlier": outlier.astype(np.bool_),
            "saved_evasion": saved_evasion.astype(np.bool_),
            "saved_joint_valid": saved_joint_valid.astype(np.bool_),
            "evasion_match": (evasion == saved_evasion),
            "validity_match": (joint_valid == saved_joint_valid),
            "source_run": str(source_run),
        },
        columns=OUTCOME_COLUMNS,
    )


def _load_classifier(spec: dict[str, str], device: str) -> torch.nn.Module:
    return load_model(
        model_path=str(REPO_ROOT / "models" / spec["checkpoint"]),
        num_features=len(FEATURE_NAMES),
        num_classes=len(CLASSES),
        device=device,
    )


def _encode_outliers(
    *,
    router: AttackRouter,
    detector: Any,
    class_id: int,
    x_adv: torch.Tensor,
    device: str,
) -> np.ndarray:
    vae = router.get_vae(class_id)
    with torch.no_grad():
        z_reencoded, _ = vae.encode(x_adv.to(device))
        outlier = detector.outlier_mask(class_id, z_reencoded.cpu()).cpu()
    return outlier.numpy().astype(np.bool_)


def _prepare_gaussian_infrastructure(
    *,
    config: dict[str, Any],
    device: str,
    fit_detector: Any,
    load_collapsed_dims: Any,
) -> tuple[AttackRouter, Any, PerturbationMask, ProtocolValidator, dict[str, Any]]:
    manifest_path = Path(config["vae_run"]["manifest_path"])
    diagnostics_dir = Path(config["vae_run"]["diagnostics_dir"])
    manifest = _load_json(manifest_path)
    collapsed_by_class = load_collapsed_dims(
        diagnostics_dir=diagnostics_dir,
        manifest=manifest,
    )

    router = AttackRouter(device=device)
    router.manifest = manifest
    mask = PerturbationMask.from_preprocessing_artifacts()
    protocol_validator = ProtocolValidator(router.scaler)
    detector = fit_detector(
        router,
        device,
        collapsed_by_class=collapsed_by_class,
    )
    return router, detector, mask, protocol_validator, manifest


def _rerun_all_models(
    *,
    run_dir: Path,
    device: str,
) -> tuple[pd.DataFrame, AttackRouter, Any]:
    config = _load_json(run_dir / "config_snapshot.json")
    if config["vae_run"]["run_tag"] != GAUSSIAN_RUN_TAG:
        raise ValueError(f"Expected Gaussian run {GAUSSIAN_RUN_TAG}, got {config['vae_run']['run_tag']}")

    saved = pd.read_csv(run_dir / "per_sample_results.csv")
    seed = int(config["seed"])
    set_global_seed(seed)

    router, detector, mask, protocol_validator, manifest = _prepare_gaussian_infrastructure(
        config=config,
        device=device,
        fit_detector=all_models_run._fit_detector,
        load_collapsed_dims=all_models_run._load_collapsed_dims_for_run,
    )
    split_test = load_split("test")

    attack_config = config["attacks"]
    epsilon_by_class = parse_class_float_map(
        attack_config["latent-pgd"]["epsilon_by_class"],
        default_by_class_name={},
    )
    cw_radius_by_class = parse_class_float_map(
        attack_config["latent-cw"]["cw_init_radius_by_class"],
        default_by_class_name={
            class_name: class_float_value(
                CLASS_TO_ID[class_name],
                epsilon_by_class,
                attack_config["latent-pgd"]["epsilon_fallback"],
            )
            for class_name in CLASSES
        },
    )

    gmm_priors: dict[int, LatentGMMPrior] = {}
    restart_strategy = str(attack_config["latent-pgd"]["restart_strategy"])
    if strategy_uses_gmm(restart_strategy):
        for class_name in SOURCE_CLASSES:
            class_id = CLASS_TO_ID[class_name]
            print(f"[all-models] GMM {class_name}", flush=True)
            gmm_priors[class_id] = fit_or_load_latent_gmm(
                router=router,
                class_id=class_id,
                split_name=str(attack_config["latent-pgd"]["gmm_split"]),
                n_components=int(attack_config["latent-pgd"]["gmm_components"]),
                max_fit_samples=int(attack_config["latent-pgd"]["gmm_fit_max_samples"]),
                seed=seed,
                device=device,
                force_refit=False,
            )

    specs = _model_specs(config)
    classifiers = {spec["tag"]: _load_classifier(spec, device) for spec in specs}
    results: list[pd.DataFrame] = []

    original_row_from_best = all_models_run._row_from_best

    def row_with_outlier(best: dict[str, Any], *, class_id: int, n: int) -> dict[str, Any]:
        row = original_row_from_best(best, class_id=class_id, n=n)
        row["_outlier_mask"] = best["outlier_mask"].cpu().numpy().tolist()
        return row

    all_models_run._row_from_best = row_with_outlier
    try:
        for spec in specs:
            classifier = classifiers[spec["tag"]]
            for attack_name in attack_config["selected"]:
                for class_name in SOURCE_CLASSES:
                    saved_cell = _saved_cell(
                        saved,
                        model_tag=spec["tag"],
                        attack=attack_name,
                        class_name=class_name,
                    )
                    if saved_cell.empty:
                        continue

                    print(
                        f"[all-models] {spec['tag']} {attack_name} {class_name} "
                        f"N={len(saved_cell)}",
                        flush=True,
                    )
                    class_id = CLASS_TO_ID[class_name]
                    sample_ids = saved_cell["sample_id"].to_numpy(dtype=np.int64)
                    x_batch = torch.from_numpy(
                        split_test["X"][sample_ids].astype(np.float32)
                    )
                    y_batch = torch.from_numpy(
                        split_test["y_8"][sample_ids].astype(np.int64)
                    )

                    if attack_name.startswith("latent-"):
                        vae = router.get_vae(class_id)
                        class_epsilon = class_float_value(
                            class_id,
                            epsilon_by_class,
                            float(attack_config["latent-pgd"]["epsilon_fallback"]),
                        )
                        class_alpha = class_epsilon * float(
                            attack_config["latent-pgd"]["alpha_ratio"]
                        )
                        init_radius = (
                            class_float_value(class_id, cw_radius_by_class, class_epsilon)
                            if attack_name == "latent-cw"
                            else class_epsilon
                        )
                        z_initializers, restart_labels = build_latent_restart_initializers(
                            vae=vae,
                            x_batch=x_batch,
                            gmm_prior=gmm_priors.get(class_id),
                            epsilon=init_radius,
                            num_restarts=int(attack_config[attack_name]["num_restarts"]),
                            restart_strategy=str(
                                attack_config[attack_name]["restart_strategy"]
                            ),
                            seed=seed,
                            class_id=class_id,
                            device=device,
                            project_to_epsilon=True,
                        )
                        metrics = all_models_run._evaluate_latent_attack(
                            attack_name=attack_name,
                            class_id=class_id,
                            x_batch=x_batch,
                            y_batch=y_batch,
                            vae=vae,
                            classifier=classifier,
                            mask=mask,
                            protocol_validator=protocol_validator,
                            detector=detector,
                            scaler=router.scaler,
                            device=device,
                            epsilon=class_epsilon,
                            alpha=class_alpha,
                            num_steps=int(attack_config["latent-pgd"]["num_steps"]),
                            random_start=bool(
                                attack_config["latent-pgd"]["random_start"]
                            ),
                            lambda_conf=float(
                                attack_config["latent-cw"]["lambda_conf"]
                            ),
                            kappa=float(attack_config["latent-cw"]["kappa"]),
                            num_iterations=int(
                                attack_config["latent-cw"]["num_iterations"]
                            ),
                            learning_rate=float(
                                attack_config["latent-cw"]["learning_rate"]
                            ),
                            convergence_threshold=float(
                                attack_config["latent-cw"]["convergence_threshold"]
                            ),
                            num_restarts=int(attack_config[attack_name]["num_restarts"]),
                            restart_strategy=str(
                                attack_config[attack_name]["restart_strategy"]
                            ),
                            z_initializers=z_initializers,
                            restart_labels=restart_labels,
                            adaptive_pgd=bool(
                                attack_config["latent-pgd"]["adaptive_pgd"]
                            ),
                            checkpoint_interval=int(
                                attack_config["latent-pgd"]["checkpoint_interval"]
                            ),
                            rho=float(attack_config["latent-pgd"]["rho"]),
                            min_alpha=float(
                                attack_config["latent-pgd"]["min_alpha"]
                            ),
                        )
                        outlier = np.asarray(
                            metrics.pop("_outlier_mask"), dtype=np.bool_
                        )
                    else:
                        captured: dict[str, torch.Tensor] = {}
                        if attack_name == "input-pgd":
                            original_attack = all_models_run.input_pgd_attack

                            def capture_attack(*args: Any, **kwargs: Any) -> Any:
                                output = original_attack(*args, **kwargs)
                                captured["x_adv"] = output[0].detach().cpu()
                                return output

                            all_models_run.input_pgd_attack = capture_attack
                        else:
                            original_attack = all_models_run.input_cw_attack

                            def capture_attack(*args: Any, **kwargs: Any) -> Any:
                                output = original_attack(*args, **kwargs)
                                captured["x_adv"] = output[0].detach().cpu()
                                return output

                            all_models_run.input_cw_attack = capture_attack

                        try:
                            metrics = all_models_run._evaluate_input_attack(
                                attack_name=attack_name,
                                class_id=class_id,
                                x_batch=x_batch,
                                y_batch=y_batch,
                                classifier=classifier,
                                mask=mask,
                                protocol_validator=protocol_validator,
                                device=device,
                                epsilon=float(attack_config["input-pgd"]["epsilon"]),
                                alpha=float(attack_config["input-pgd"]["alpha"]),
                                num_steps=int(attack_config["input-pgd"]["num_steps"]),
                                random_start=bool(
                                    attack_config["input-pgd"]["random_start"]
                                ),
                                lambda_conf=float(
                                    attack_config["input-cw"]["lambda_conf"]
                                ),
                                kappa=float(attack_config["input-cw"]["kappa"]),
                                num_iterations=int(
                                    attack_config["input-cw"]["num_iterations"]
                                ),
                                learning_rate=float(
                                    attack_config["input-cw"]["learning_rate"]
                                ),
                                convergence_threshold=float(
                                    attack_config["input-cw"][
                                        "convergence_threshold"
                                    ]
                                ),
                            )
                        finally:
                            if attack_name == "input-pgd":
                                all_models_run.input_pgd_attack = original_attack
                            else:
                                all_models_run.input_cw_attack = original_attack

                        outlier = _encode_outliers(
                            router=router,
                            detector=detector,
                            class_id=class_id,
                            x_adv=captured["x_adv"],
                            device=device,
                        )

                    evasion = np.asarray(metrics["success_mask"], dtype=np.bool_)
                    joint_valid = np.asarray(
                        metrics["joint_valid_mask"], dtype=np.bool_
                    )
                    results.append(
                        _outcome_frame(
                            family=(
                                "latent"
                                if attack_name.startswith("latent-")
                                else "unconstrained-input"
                            ),
                            attack=attack_name,
                            attack_goal="untargeted",
                            model_label=spec["label"],
                            model_tag=spec["tag"],
                            class_name=class_name,
                            sample_ids=sample_ids,
                            evasion=evasion,
                            joint_valid=joint_valid,
                            outlier=outlier,
                            saved_evasion=_as_bool(saved_cell["success"]),
                            saved_joint_valid=_as_bool(saved_cell["joint_valid"]),
                            source_run=run_dir,
                        )
                    )
    finally:
        all_models_run._row_from_best = original_row_from_best

    return pd.concat(results, ignore_index=True), router, detector


def _rerun_constrained(
    *,
    run_dir: Path,
    device: str,
) -> pd.DataFrame:
    config = _load_json(run_dir / "config_snapshot.json")
    if config["vae_run"]["run_tag"] != GAUSSIAN_RUN_TAG:
        raise ValueError(f"Expected Gaussian run {GAUSSIAN_RUN_TAG}, got {config['vae_run']['run_tag']}")

    saved = pd.read_csv(run_dir / "per_sample_results.csv")
    seed = int(config["seed"])
    set_global_seed(seed)
    router, detector, mask, protocol_validator, _manifest = (
        _prepare_gaussian_infrastructure(
            config=config,
            device=device,
            fit_detector=constrained_run._fit_detector,
            load_collapsed_dims=constrained_run._load_collapsed_dims_for_run,
        )
    )
    projection = constrained_run.VAEConstraintProjection(
        router.scaler,
        mask,
        enable_physics=True,
        device=device,
    )
    split_test = load_split("test")
    specs = _model_specs(config)
    classifiers = {spec["tag"]: _load_classifier(spec, device) for spec in specs}
    attack_config = config["attacks"]
    args = SimpleNamespace(
        input_epsilon=float(attack_config["cinput-pgd"]["epsilon"]),
        input_alpha=float(attack_config["cinput-pgd"]["alpha"]),
        num_steps=int(attack_config["cinput-pgd"]["num_steps"]),
        random_start=bool(attack_config["cinput-pgd"]["random_start"]),
        lambda_conf=float(attack_config["cinput-cw"]["lambda_conf"]),
        kappa=float(attack_config["cinput-cw"]["kappa"]),
        num_iterations=int(attack_config["cinput-cw"]["num_iterations"]),
        learning_rate=float(attack_config["cinput-cw"]["learning_rate"]),
        convergence_threshold=float(
            attack_config["cinput-cw"]["convergence_threshold"]
        ),
    )

    results: list[pd.DataFrame] = []
    original_outlier_mask = detector.outlier_mask
    captured: dict[str, np.ndarray] = {}

    def capture_outlier(class_id: int, z_batch: torch.Tensor) -> torch.Tensor:
        value = original_outlier_mask(class_id, z_batch)
        captured["outlier"] = value.detach().cpu().numpy().astype(np.bool_)
        return value

    detector.outlier_mask = capture_outlier
    for spec in specs:
        classifier = classifiers[spec["tag"]]
        for attack_name in attack_config["selected"]:
            for class_name in SOURCE_CLASSES:
                saved_cell = _saved_cell(
                    saved,
                    model_tag=spec["tag"],
                    attack=attack_name,
                    class_name=class_name,
                )
                if saved_cell.empty:
                    continue

                print(
                    f"[constrained] {spec['tag']} {attack_name} {class_name} "
                    f"N={len(saved_cell)}",
                    flush=True,
                )
                class_id = CLASS_TO_ID[class_name]
                sample_ids = saved_cell["sample_id"].to_numpy(dtype=np.int64)
                x_batch = torch.from_numpy(
                    split_test["X"][sample_ids].astype(np.float32)
                )
                y_batch = torch.from_numpy(
                    split_test["y_8"][sample_ids].astype(np.int64)
                )
                captured.clear()
                metrics = constrained_run._evaluate(
                    attack_name=attack_name,
                    class_id=class_id,
                    x_batch=x_batch,
                    y_batch=y_batch,
                    vae=router.get_vae(class_id),
                    classifier=classifier,
                    projection=projection,
                    mask=mask,
                    protocol_validator=protocol_validator,
                    detector=detector,
                    scaler=router.scaler,
                    device=device,
                    args=args,
                )
                evasion = np.asarray(metrics["success_mask"], dtype=np.bool_)
                joint_valid = np.asarray(
                    metrics["joint_valid_mask"], dtype=np.bool_
                )
                results.append(
                    _outcome_frame(
                        family="constrained-input",
                        attack=attack_name,
                        attack_goal=str(metrics["attack_goal"]),
                        model_label=spec["label"],
                        model_tag=spec["tag"],
                        class_name=class_name,
                        sample_ids=sample_ids,
                        evasion=evasion,
                        joint_valid=joint_valid,
                        outlier=captured["outlier"],
                        saved_evasion=_as_bool(saved_cell["success"]),
                        saved_joint_valid=_as_bool(saved_cell["joint_valid"]),
                        source_run=run_dir,
                    )
                )

    return pd.concat(results, ignore_index=True)


def _evaluate_targeted_latent_artifacts(
    *,
    run_dir: Path,
    router: AttackRouter,
    detector: Any,
    device: str,
) -> pd.DataFrame:
    config = _load_json(run_dir / "config_snapshot.json")
    if config["vae_run"]["run_tag"] != GAUSSIAN_RUN_TAG:
        raise ValueError(f"Expected Gaussian run {GAUSSIAN_RUN_TAG}, got {config['vae_run']['run_tag']}")

    saved = pd.read_csv(run_dir / "per_sample_results.csv")
    specs = _model_specs(config)
    results: list[pd.DataFrame] = []
    for spec in specs:
        for class_name in SOURCE_CLASSES:
            artifact = run_dir / f"{spec['tag']}_targeted_benign_pgd_{class_name}.npz"
            if not artifact.exists():
                continue

            saved_cell = _saved_cell(
                saved,
                model_tag=spec["tag"],
                attack=None,
                class_name=class_name,
            )
            print(
                f"[targeted-latent] {spec['tag']} {class_name} N={len(saved_cell)}",
                flush=True,
            )
            with np.load(artifact, allow_pickle=False) as data:
                x_adv = torch.from_numpy(data["x_adv"].astype(np.float32))
                evasion = data["target_success"].astype(np.bool_)
                joint_valid = data["joint_valid"].astype(np.bool_)

            class_id = CLASS_TO_ID[class_name]
            outlier = _encode_outliers(
                router=router,
                detector=detector,
                class_id=class_id,
                x_adv=x_adv,
                device=device,
            )
            results.append(
                _outcome_frame(
                    family="latent",
                    attack="targeted-benign-latent-pgd",
                    attack_goal="target-benign",
                    model_label=spec["label"],
                    model_tag=spec["tag"],
                    class_name=class_name,
                    sample_ids=saved_cell["sample_id"].to_numpy(dtype=np.int64),
                    evasion=evasion,
                    joint_valid=joint_valid,
                    outlier=outlier,
                    saved_evasion=_as_bool(saved_cell["target_success"]),
                    saved_joint_valid=_as_bool(saved_cell["joint_valid"]),
                    source_run=run_dir,
                )
            )

    return pd.concat(results, ignore_index=True)


def _load_targeted_latent_cw_outcomes(run_dir: Path) -> pd.DataFrame:
    saved = pd.read_csv(run_dir / "per_sample_results.csv")
    required = {
        "sample_id",
        "model",
        "model_tag",
        "source_class",
        "target_success",
        "joint_valid",
        "mahalanobis_outlier",
    }
    missing = required - set(saved.columns)
    if missing:
        raise ValueError(
            f"{run_dir / 'per_sample_results.csv'} is missing columns: "
            f"{sorted(missing)}"
        )

    results: list[pd.DataFrame] = []
    for (model_tag, model_label, class_name), cell in saved.groupby(
        ["model_tag", "model", "source_class"],
        sort=False,
    ):
        evasion = _as_bool(cell["target_success"])
        joint_valid = _as_bool(cell["joint_valid"])
        outlier = _as_bool(cell["mahalanobis_outlier"])
        results.append(
            _outcome_frame(
                family="latent",
                attack="targeted-benign-latent-cw",
                attack_goal="target-benign",
                model_label=str(model_label),
                model_tag=str(model_tag),
                class_name=str(class_name),
                sample_ids=cell["sample_id"].to_numpy(dtype=np.int64),
                evasion=evasion,
                joint_valid=joint_valid,
                outlier=outlier,
                saved_evasion=evasion,
                saved_joint_valid=joint_valid,
                source_run=run_dir,
            )
        )
    return pd.concat(results, ignore_index=True)


def _aggregate(outcomes: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    work = outcomes.copy()
    work["asr_valid_outcome"] = work["evasion"] & work["joint_valid"]
    work["mahalanobis_id"] = ~work["mahalanobis_outlier"]
    work["he_idsr_outcome"] = work["evasion"] & work["mahalanobis_id"]
    return (
        work.groupby(group_cols, sort=False, dropna=False)
        .agg(
            N=("evasion", "size"),
            ASR_raw=("evasion", "mean"),
            ASR_valid=("asr_valid_outcome", "mean"),
            mahalanobis_id_rate=("mahalanobis_id", "mean"),
            he_idsr=("he_idsr_outcome", "mean"),
            evasion_match_rate=("evasion_match", "mean"),
            validity_match_rate=("validity_match", "mean"),
        )
        .reset_index()
    )


def _pct(value: float) -> str:
    return f"{100.0 * float(value):.2f}%"


def _markdown_table(df: pd.DataFrame, columns: list[str]) -> list[str]:
    labels = {
        "attack": "Attack",
        "attack_family": "Family",
        "attack_method": "Attack",
        "attack_goal": "Goal",
        "classifier": "Classifier",
        "classifier_tag": "Classifier",
        "models": "Models included",
        "attack_class": "Class",
        "N": "N",
        "ASR_raw": "ASR raw",
        "ASR_valid": "ASR valid",
        "mahalanobis_id_rate": "Mahalanobis ID",
        "he_idsr": "He-IDSR",
        "evasion_match_rate": "E match",
        "validity_match_rate": "V match",
    }
    lines = [
        "| " + " | ".join(labels.get(column, column) for column in columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    for _, row in df.iterrows():
        values: list[str] = []
        for column in columns:
            value = row[column]
            if column in {
                "ASR_raw",
                "ASR_valid",
                "mahalanobis_id_rate",
                "he_idsr",
                "evasion_match_rate",
                "validity_match_rate",
            }:
                values.append(_pct(float(value)))
            elif column == "N":
                values.append(str(int(value)))
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def _write_report(
    *,
    report_path: Path,
    overall: pd.DataFrame,
    by_classifier: pd.DataFrame,
    by_class: pd.DataFrame,
    all_models_run: Path,
    constrained_run: Path,
    targeted_latent_run: Path,
    targeted_latent_cw_run: Path,
) -> None:
    overall_by_attack = overall.set_index("attack_method")

    def metric(attack: str, column: str) -> str:
        return _pct(float(overall_by_attack.loc[attack, column]))

    target_labels = {
        "targeted-benign-latent-pgd": "Latent PGD",
        "targeted-benign-latent-cw": "Latent CW",
        "cinput-pgd-target-benign": "Constrained input PGD",
        "cinput-cw-target-benign": "Constrained input CW",
    }
    target_average = (
        by_classifier.loc[
            (by_classifier["attack_goal"] == "target-benign")
            & (by_classifier["classifier"] != "DualPath")
            & by_classifier["attack_method"].isin(target_labels),
            ["attack_method", "ASR_valid"],
        ]
        .groupby("attack_method", as_index=False, sort=False)
        .agg(ASR_valid=("ASR_valid", "mean"))
    )
    target_average["attack"] = target_average["attack_method"].map(target_labels)
    target_average["models"] = "MLP, CNN, LSTM, CNN-LSTM"
    target_average = target_average.set_index("attack_method").loc[
        list(target_labels)
    ].reset_index()

    lines = [
        "# He-IDSR and Domain-Valid Attack Success Report",
        "",
        f"VAE run: `{GAUSSIAN_RUN_TAG}`",
        "",
        "## Definitions",
        "",
        "- `E_i`: attack-goal success for sample `i`. For untargeted attacks this is misclassification; for target-to-Benign attacks this is prediction as Benign.",
        "- `V_i`: joint domain validity (protocol validity, perturbation-mask compliance, and raw G1-G8 validity where evaluated).",
        "- `O_i`: Mahalanobis outlier flag after re-encoding the final adversarial sample with its source-class VAE.",
        "- `ASR_raw = mean(E_i)`.",
        "- `ASR_valid = mean(E_i * V_i)`. This is the thesis joint valid-success rate, not the conditional `asr_valid_only` field in the rerun summaries.",
        "- `Mahalanobis ID rate = mean(1 - O_i)`.",
        "- `He-IDSR = mean(E_i * (1 - O_i))`.",
        "",
        "## Scope",
        "",
        "- Five classifiers: MLP, CNN, LSTM, CNN-LSTM, and DualPath.",
        "- Seven malicious source classes. Benign is a target class, not an attack source class.",
        "- Ten attack configurations: latent PGD/CW, unconstrained input PGD/CW, targeted latent PGD/CW, constrained input PGD/CW, and their target-to-Benign variants.",
        "- Cells without any correctly classified source samples are omitted, matching the original runs.",
        "",
        "## Overall Results",
        "",
    ]
    lines.extend(
        _markdown_table(
            overall,
            [
                "attack_family",
                "attack_method",
                "attack_goal",
                "N",
                "ASR_raw",
                "ASR_valid",
                "mahalanobis_id_rate",
                "he_idsr",
            ],
        )
    )
    lines.extend(
        [
            "",
            "## Key Findings",
            "",
            f"- Latent PGD and latent CW have similar joint performance: He-IDSR is `{metric('latent-pgd', 'he_idsr')}` and `{metric('latent-cw', 'he_idsr')}`, while ASR_valid is `{metric('latent-pgd', 'ASR_valid')}` for both.",
            f"- Unconstrained input PGD and CW achieve high raw evasion (`{metric('input-pgd', 'ASR_raw')}` and `{metric('input-cw', 'ASR_raw')}`) but `0.00%` ASR_valid because the generated samples fail the attack pipeline's joint domain constraints.",
            f"- Mahalanobis membership alone is not equivalent to domain validity: unconstrained input CW has `{metric('input-cw', 'he_idsr')}` He-IDSR despite `0.00%` ASR_valid.",
            f"- Constrained input CW has the strongest ASR_valid at `{metric('cinput-cw', 'ASR_valid')}`, but its He-IDSR is lower at `{metric('cinput-cw', 'he_idsr')}` because many successful samples are Mahalanobis outliers.",
            f"- Target-to-Benign attacks remain weak: targeted latent PGD reaches `{metric('targeted-benign-latent-pgd', 'he_idsr')}` He-IDSR, targeted latent CW `{metric('targeted-benign-latent-cw', 'he_idsr')}`, constrained PGD `{metric('cinput-pgd-target-benign', 'he_idsr')}`, and constrained CW `{metric('cinput-cw-target-benign', 'he_idsr')}`.",
            "",
            "## Target-to-Benign Average ASR Valid",
            "",
            "These are unweighted macro-averages across MLP, CNN, LSTM, and CNN-LSTM. DualPath is excluded.",
            "",
        ]
    )
    lines.extend(
        _markdown_table(
            target_average,
            ["attack", "models", "ASR_valid"],
        )
    )
    lines.extend(
        [
            "",
            "### Vertical",
            "",
            "![Vertical target-to-Benign average ASR valid chart](results/he_idsr/target_benign_asr_valid_vertical.png)",
            "",
            "### Horizontal",
            "",
            "![Horizontal target-to-Benign average ASR valid chart](results/he_idsr/target_benign_asr_valid_horizontal.png)",
            "",
            "## Results by Classifier",
            "",
        ]
    )
    lines.extend(
        _markdown_table(
            by_classifier,
            [
                "attack_method",
                "classifier",
                "N",
                "ASR_raw",
                "ASR_valid",
                "mahalanobis_id_rate",
                "he_idsr",
            ],
        )
    )

    lines.extend(["", "## Detailed Results by Source Class", ""])
    for attack_name in by_class["attack_method"].drop_duplicates().tolist():
        lines.extend([f"### `{attack_name}`", ""])
        attack_rows = by_class.loc[by_class["attack_method"] == attack_name]
        lines.extend(
            _markdown_table(
                attack_rows,
                [
                    "classifier",
                    "attack_class",
                    "N",
                    "ASR_raw",
                    "ASR_valid",
                    "mahalanobis_id_rate",
                    "he_idsr",
                ],
            )
        )
        lines.append("")

    min_e_match = float(by_class["evasion_match_rate"].min())
    min_v_match = float(by_class["validity_match_rate"].min())
    lines.extend(
        [
            "## Reproduction Checks",
            "",
            f"- Minimum per-cell evasion-mask agreement with saved results: `{_pct(min_e_match)}`.",
            f"- Minimum per-cell joint-validity-mask agreement with saved results: `{_pct(min_v_match)}`.",
            "- Targeted latent PGD used its saved `x_adv` NPZ artifacts. Targeted latent CW persisted the Mahalanobis flag directly. The older untargeted and constrained families were rerun because their final adversarial samples or per-sample Mahalanobis flags were not persisted.",
            "",
            "## Source Runs",
            "",
            f"- Untargeted latent and unconstrained input: `{all_models_run}`",
            f"- Constrained input: `{constrained_run}`",
            f"- Targeted latent PGD: `{targeted_latent_run}`",
            f"- Targeted latent CW: `{targeted_latent_cw_run}`",
            "",
            "The CSV files in `results/he_idsr/` contain the same results in machine-readable form.",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute joint He-IDSR and thesis ASR_valid for Gaussian VAE attacks."
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--all-models-run", type=Path, default=DEFAULT_ALL_MODELS_RUN)
    parser.add_argument("--constrained-run", type=Path, default=DEFAULT_CONSTRAINED_RUN)
    parser.add_argument(
        "--targeted-latent-run",
        type=Path,
        default=DEFAULT_TARGETED_LATENT_RUN,
    )
    parser.add_argument(
        "--targeted-latent-cw-run",
        type=Path,
        default=DEFAULT_TARGETED_LATENT_CW_RUN,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--merge-targeted-cw-only",
        action="store_true",
        help="Reuse existing he_idsr_per_sample.csv and merge only targeted latent CW.",
    )
    args = parser.parse_args()

    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    targeted_cw_outcomes = _load_targeted_latent_cw_outcomes(
        args.targeted_latent_cw_run
    )
    if args.merge_targeted_cw_only:
        existing_path = args.output_dir / "he_idsr_per_sample.csv"
        if not existing_path.exists():
            raise FileNotFoundError(
                f"--merge-targeted-cw-only requires {existing_path}"
            )
        existing = pd.read_csv(existing_path)
        existing = existing.loc[
            existing["attack_method"] != "targeted-benign-latent-cw"
        ].copy()
        outcomes = pd.concat(
            [existing, targeted_cw_outcomes],
            ignore_index=True,
        )
    else:
        all_outcomes, router, detector = _rerun_all_models(
            run_dir=args.all_models_run,
            device=args.device,
        )
        targeted_outcomes = _evaluate_targeted_latent_artifacts(
            run_dir=args.targeted_latent_run,
            router=router,
            detector=detector,
            device=args.device,
        )
        constrained_outcomes = _rerun_constrained(
            run_dir=args.constrained_run,
            device=args.device,
        )
        outcomes = pd.concat(
            [
                all_outcomes,
                targeted_outcomes,
                targeted_cw_outcomes,
                constrained_outcomes,
            ],
            ignore_index=True,
        )

    by_class = _aggregate(
        outcomes,
        [
            "attack_family",
            "attack_method",
            "attack_goal",
            "classifier",
            "classifier_tag",
            "attack_class",
        ],
    )
    by_classifier = _aggregate(
        outcomes,
        [
            "attack_family",
            "attack_method",
            "attack_goal",
            "classifier",
            "classifier_tag",
        ],
    )
    overall = _aggregate(
        outcomes,
        ["attack_family", "attack_method", "attack_goal"],
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outcomes.to_csv(args.output_dir / "he_idsr_per_sample.csv", index=False)
    by_class.to_csv(args.output_dir / "he_idsr_by_class.csv", index=False)
    by_classifier.to_csv(args.output_dir / "he_idsr_by_classifier.csv", index=False)
    overall.to_csv(args.output_dir / "he_idsr_overall.csv", index=False)
    _write_report(
        report_path=args.report,
        overall=overall,
        by_classifier=by_classifier,
        by_class=by_class,
        all_models_run=args.all_models_run,
        constrained_run=args.constrained_run,
        targeted_latent_run=args.targeted_latent_run,
        targeted_latent_cw_run=args.targeted_latent_cw_run,
    )

    print()
    print("=== He-IDSR overall ===")
    display = overall.copy()
    for column in ["ASR_raw", "ASR_valid", "mahalanobis_id_rate", "he_idsr"]:
        display[column] = display[column].map(_pct)
    print(display.to_string(index=False))
    print(f"\nReport: {args.report}")
    print(f"CSV directory: {args.output_dir}")


if __name__ == "__main__":
    main()
