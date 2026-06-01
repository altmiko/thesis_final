"""
Tree-based baseline training for CICIoT2023 processed arrays.

Adds two non-neural baselines alongside the five neural architectures:
- Random Forest (scikit-learn)
- XGBoost

For each task (binary / 8class / 34class) and each model (rf / xgb), the script:
- Runs a small validation-set hyperparameter search (selection by macro-F1).
- Fits the best config on the training split (val is used for selection only,
  never trained on, mirroring the neural-net baselines).
- Evaluates the selected model on the test split.
- Saves the fitted estimator (joblib .pkl) and a classification report whose
  JSON schema matches the neural-net baseline reports, so both families fold
  into the same downstream comparison tables.

No class weights are used (stratified sampling rationale, same as the NN
baselines). Global seed 42.

Run from repo root:
    python -m src.classifiers.tree_baselines --models all --tasks all --device auto
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# Reuse the exact rationale string from the neural-net baselines for parity.
CLASS_WEIGHT_DECISION = (
    "Class weights excluded because stratified sampling already balances the training set; "
    "applying original-distribution weights would over-penalize majority classes in the balanced sample."
)

TREE_MODEL_TYPES: Tuple[str, ...] = ("rf", "xgb")

# Light tuning grids (deliberately small to bound runtime on 3.1M rows).
# RF depth is capped and paired with min_samples_leaf + max_samples to bound
# memory: scikit-learn stores an (n_classes,)-float value array at every node,
# so for the 34-class task unbounded-depth trees would consume tens of GB each.
RF_GRID: Tuple[Dict[str, int], ...] = (
    {"n_estimators": 200, "max_depth": 20},
    {"n_estimators": 300, "max_depth": 20},
    {"n_estimators": 200, "max_depth": 30},
    {"n_estimators": 300, "max_depth": 30},
)
RF_MIN_SAMPLES_LEAF_DEFAULT = 25
RF_MAX_SAMPLES_DEFAULT = 0.4
XGB_GRID: Tuple[Dict[str, float], ...] = (
    {"max_depth": 6, "learning_rate": 0.1},
    {"max_depth": 10, "learning_rate": 0.1},
    {"max_depth": 6, "learning_rate": 0.3},
    {"max_depth": 10, "learning_rate": 0.3},
)
XGB_N_ESTIMATORS = 600
XGB_EARLY_STOPPING_ROUNDS = 30


@dataclass(frozen=True)
class TaskConfig:
    name: str
    y_train_file: str
    y_val_file: str
    y_test_file: str
    class_weights_file: str
    num_classes: int


TASKS: Tuple[TaskConfig, ...] = (
    TaskConfig(
        name="binary",
        y_train_file="y_train_bin.npy",
        y_val_file="y_val_bin.npy",
        y_test_file="y_test_bin.npy",
        class_weights_file="class_weights_2.npy",
        num_classes=2,
    ),
    TaskConfig(
        name="8class",
        y_train_file="y_train_cat.npy",
        y_val_file="y_val_cat.npy",
        y_test_file="y_test_cat.npy",
        class_weights_file="class_weights_8.npy",
        num_classes=8,
    ),
    TaskConfig(
        name="34class",
        y_train_file="y_train.npy",
        y_val_file="y_val.npy",
        y_test_file="y_test.npy",
        class_weights_file="class_weights_34.npy",
        num_classes=34,
    ),
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def parse_csv_arg(value: str, valid: Sequence[str]) -> List[str]:
    if value.lower() == "all":
        return list(valid)
    requested = [item.strip().lower() for item in value.split(",") if item.strip()]
    unknown = sorted(set(requested) - set(valid))
    if unknown:
        raise ValueError(f"Unknown values {unknown}; valid choices are: {list(valid)}")
    return requested


def resolve_xgb_device(requested: str) -> str:
    """Resolve the XGBoost device string. RF is always CPU."""
    if requested == "cpu":
        return "cpu"
    try:
        import torch

        cuda_ok = bool(torch.cuda.is_available())
    except Exception:
        cuda_ok = False
    if requested == "cuda":
        return "cuda" if cuda_ok else "cpu"
    # auto
    return "cuda" if cuda_ok else "cpu"


def load_array(path: Path, mmap: bool = False) -> np.ndarray:
    if mmap:
        return np.load(path, mmap_mode="r")
    return np.load(path)


def align_proba_columns(
    proba: np.ndarray,
    classes: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    """Scatter a (n, len(classes)) probability matrix into a full (n, num_classes)
    matrix indexed by class label. Columns for classes absent from `classes`
    are filled with zeros. Guards against estimators that did not observe every
    label during fitting."""
    proba = np.asarray(proba, dtype=np.float64)
    class_idx = np.asarray(classes).astype(int)
    n_rows = proba.shape[0]
    full = np.zeros((n_rows, num_classes), dtype=np.float64)
    full[:, class_idx] = proba
    return full


def grid_for(model_type: str) -> List[Dict[str, float]]:
    if model_type == "rf":
        return [dict(p) for p in RF_GRID]
    if model_type == "xgb":
        return [dict(p) for p in XGB_GRID]
    raise ValueError(f"Unknown tree model type: {model_type}")


def make_rf(
    params: Dict[str, float],
    seed: int = 42,
    max_samples: float = RF_MAX_SAMPLES_DEFAULT,
    min_samples_leaf: int = RF_MIN_SAMPLES_LEAF_DEFAULT,
    n_jobs: int = -1,
) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=int(params["n_estimators"]),
        max_depth=int(params["max_depth"]),
        max_features="sqrt",
        max_samples=max_samples,
        min_samples_leaf=int(min_samples_leaf),
        class_weight=None,
        n_jobs=n_jobs,
        random_state=seed,
    )


def make_xgb(params: Dict[str, float], num_classes: int, device: str = "cpu", seed: int = 42):
    from xgboost import XGBClassifier

    objective = "binary:logistic" if num_classes == 2 else "multi:softprob"
    eval_metric = "logloss" if num_classes == 2 else "mlogloss"
    return XGBClassifier(
        n_estimators=XGB_N_ESTIMATORS,
        max_depth=int(params["max_depth"]),
        learning_rate=float(params["learning_rate"]),
        tree_method="hist",
        device=device,
        objective=objective,
        eval_metric=eval_metric,
        early_stopping_rounds=XGB_EARLY_STOPPING_ROUNDS,
        n_jobs=-1,
        random_state=seed,
    )


def fit_estimator(
    model_type: str,
    estimator,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
):
    if model_type == "xgb":
        estimator.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)
    else:
        estimator.fit(x_train, y_train)
    return estimator


def select_best(
    model_type: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    num_classes: int,
    device: str = "cpu",
    seed: int = 42,
    rf_max_samples: float = RF_MAX_SAMPLES_DEFAULT,
    rf_min_samples_leaf: int = RF_MIN_SAMPLES_LEAF_DEFAULT,
    rf_n_jobs: int = -1,
):
    """Fit each grid config on train, score on val by macro-F1, keep the best
    fitted estimator. Returns (best_estimator, best_params, best_val_f1, candidates)."""
    labels = np.arange(num_classes)
    candidates: List[Dict[str, object]] = []
    best: Dict[str, object] | None = None

    for params in grid_for(model_type):
        if model_type == "rf":
            estimator = make_rf(
                params,
                seed=seed,
                max_samples=rf_max_samples,
                min_samples_leaf=rf_min_samples_leaf,
                n_jobs=rf_n_jobs,
            )
        else:
            estimator = make_xgb(params, num_classes=num_classes, device=device, seed=seed)

        t0 = time.time()
        fit_estimator(model_type, estimator, x_train, y_train, x_val, y_val)
        val_pred = estimator.predict(x_val)
        val_f1 = float(f1_score(y_val, val_pred, labels=labels, average="macro", zero_division=0))
        elapsed = float(time.time() - t0)

        record = {"params": params, "val_macro_f1": val_f1, "fit_seconds": elapsed}
        candidates.append(record)
        print(
            f"    [{model_type}] params={params} val_macro_f1={val_f1:.6f} "
            f"({elapsed:.1f}s)"
        )

        if best is None or val_f1 > float(best["val_macro_f1"]):
            best = {"estimator": estimator, "params": params, "val_macro_f1": val_f1}

    assert best is not None
    return best["estimator"], best["params"], float(best["val_macro_f1"]), candidates


def evaluate_on_test(estimator, x_test: np.ndarray, y_test: np.ndarray, num_classes: int) -> Dict[str, object]:
    labels = np.arange(num_classes)
    proba = align_proba_columns(estimator.predict_proba(x_test), estimator.classes_, num_classes)
    y_pred = np.argmax(proba, axis=1)

    accuracy = float(accuracy_score(y_test, y_pred))
    macro_f1 = float(f1_score(y_test, y_pred, labels=labels, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_test, y_pred, labels=labels, average="weighted", zero_division=0))
    per_class_f1 = f1_score(y_test, y_pred, labels=labels, average=None, zero_division=0)
    report_dict = classification_report(
        y_test, y_pred, labels=labels, output_dict=True, zero_division=0
    )
    report_text = classification_report(y_test, y_pred, labels=labels, zero_division=0)

    try:
        test_loss = float(log_loss(y_test, proba, labels=labels))
    except ValueError:
        test_loss = float("nan")

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class_f1": {str(i): float(v) for i, v in enumerate(per_class_f1.tolist())},
        "classification_report": report_dict,
        "classification_report_text": report_text,
        "test_loss": test_loss,
        "y_pred": y_pred,
    }


def build_report_payload(
    model_type: str,
    task_name: str,
    num_classes: int,
    class_weights: np.ndarray,
    metrics: Dict[str, object],
    best_params: Dict[str, float],
    val_macro_f1: float,
    tuning_records: List[Dict[str, object]],
) -> Dict[str, object]:
    """Build a report dict whose key set is a superset of the NN baseline schema."""
    return {
        "model": model_type,
        "task": task_name,
        "num_classes": num_classes,
        "loss_weighting": "excluded",
        "loss_weighting_decision": CLASS_WEIGHT_DECISION,
        "provided_class_weights_for_reference": [float(w) for w in np.asarray(class_weights).tolist()],
        "test_loss": metrics["test_loss"],
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
        "per_class_f1": metrics["per_class_f1"],
        "classification_report": metrics["classification_report"],
        # Tree-only additive keys (do not break the shared schema):
        "selection_metric": "val_macro_f1",
        "selected_hyperparameters": best_params,
        "val_macro_f1": float(val_macro_f1),
        "tuning": tuning_records,
    }


def _subset(arr: np.ndarray, limit: int | None) -> np.ndarray:
    return arr if limit is None else arr[:limit]


def run_task(
    model_type: str,
    task: TaskConfig,
    processed_dir: Path,
    models_dir: Path,
    results_dir: Path,
    device: str,
    seed: int,
    rf_max_samples: float,
    rf_min_samples_leaf: int,
    rf_n_jobs: int,
    limit_samples: int | None,
) -> Dict[str, float]:
    print("\n" + "-" * 90)
    print(f"Starting: task={task.name}, model={model_type}, device={device if model_type=='xgb' else 'cpu'}")

    x_train = np.ascontiguousarray(_subset(load_array(processed_dir / "X_train.npy"), limit_samples), dtype=np.float32)
    x_val = np.ascontiguousarray(_subset(load_array(processed_dir / "X_val.npy"), limit_samples), dtype=np.float32)
    x_test = np.ascontiguousarray(_subset(load_array(processed_dir / "X_test.npy"), limit_samples), dtype=np.float32)
    y_train = np.asarray(_subset(load_array(processed_dir / task.y_train_file), limit_samples), dtype=np.int64)
    y_val = np.asarray(_subset(load_array(processed_dir / task.y_val_file), limit_samples), dtype=np.int64)
    y_test = np.asarray(_subset(load_array(processed_dir / task.y_test_file), limit_samples), dtype=np.int64)
    class_weights = np.load(processed_dir / task.class_weights_file).astype(np.float32, copy=False)

    print(f"X_train shape: {x_train.shape}  num_classes: {task.num_classes}")
    print(f"loss_weighting_decision: {CLASS_WEIGHT_DECISION}")

    estimator, best_params, val_f1, candidates = select_best(
        model_type=model_type,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        num_classes=task.num_classes,
        device=device,
        seed=seed,
        rf_max_samples=rf_max_samples,
        rf_min_samples_leaf=rf_min_samples_leaf,
        rf_n_jobs=rf_n_jobs,
    )
    print(f"  selected params={best_params} (val_macro_f1={val_f1:.6f})")

    metrics = evaluate_on_test(estimator, x_test, y_test, task.num_classes)

    print("\n" + "*" * 90)
    print(f"TEST RESULTS [{task.name}:{model_type}]")
    print(f"test_loss: {metrics['test_loss']:.6f}")
    print(f"accuracy: {metrics['accuracy']:.6f}")
    print(f"macro_f1: {metrics['macro_f1']:.6f}")
    print(f"weighted_f1: {metrics['weighted_f1']:.6f}")
    print("*" * 90)

    payload = build_report_payload(
        model_type=model_type,
        task_name=task.name,
        num_classes=task.num_classes,
        class_weights=class_weights,
        metrics=metrics,
        best_params=best_params,
        val_macro_f1=val_f1,
        tuning_records=candidates,
    )

    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    import joblib

    # XGBoost trains on GPU when available but is persisted for CPU inference so
    # the saved model predicts cleanly in the downstream review step (which feeds
    # CPU NumPy) without a device-mismatch fallback.
    if model_type == "xgb":
        try:
            estimator.set_params(device="cpu")
            estimator.get_booster().set_param({"device": "cpu"})
        except Exception:
            pass

    model_path = models_dir / f"{model_type}_{task.name}.pkl"
    joblib.dump(estimator, model_path)

    json_path = results_dir / f"{model_type}_{task.name}_classification_report.json"
    txt_path = results_dir / f"{model_type}_{task.name}_classification_report.txt"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    with txt_path.open("w", encoding="utf-8") as f:
        f.write(metrics["classification_report_text"])

    print(f"Saved model: {model_path.resolve()}")
    print(f"Saved reports: {json_path.resolve()}")

    return {
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=Path("data") / "processed")
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--models", default="all", help="Comma-separated subset of {rf,xgb} or 'all'.")
    parser.add_argument("--tasks", default="all", help="Comma-separated subset of {binary,8class,34class} or 'all'.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="XGBoost device (RF is CPU-only).")
    parser.add_argument("--rf-max-samples", type=float, default=RF_MAX_SAMPLES_DEFAULT, help="RF bootstrap sample fraction (memory bound).")
    parser.add_argument("--rf-min-samples-leaf", type=int, default=RF_MIN_SAMPLES_LEAF_DEFAULT, help="RF min samples per leaf (memory bound for many-class tasks).")
    parser.add_argument("--rf-n-jobs", type=int, default=-1, help="RF parallel jobs (-1 = all cores; lower to reduce peak memory).")
    parser.add_argument("--limit-samples", type=int, default=None, help="Optional row cap for smoke tests.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    selected_models = parse_csv_arg(args.models, TREE_MODEL_TYPES)
    selected_task_names = parse_csv_arg(args.tasks, [t.name for t in TASKS])
    selected_tasks = [t for t in TASKS if t.name in selected_task_names]
    device = resolve_xgb_device(args.device)

    processed_dir = args.processed_dir
    if not processed_dir.exists():
        raise FileNotFoundError(f"Processed data directory not found: {processed_dir.resolve()}")

    print("=" * 90)
    print("TREE BASELINE TRAINING")
    print(f"models: {selected_models}")
    print(f"tasks: {[t.name for t in selected_tasks]}")
    print(f"xgb_device: {device}")
    print(f"rf_max_samples: {args.rf_max_samples}")
    print(f"rf_min_samples_leaf: {args.rf_min_samples_leaf}")
    print(f"rf_n_jobs: {args.rf_n_jobs}")
    print(f"limit_samples: {args.limit_samples}")
    print(f"seed: {args.seed}")
    print("=" * 90)

    summary: Dict[str, Dict[str, Dict[str, float]]] = {}
    for task in selected_tasks:
        summary[task.name] = {}
        for model_type in selected_models:
            metrics = run_task(
                model_type=model_type,
                task=task,
                processed_dir=processed_dir,
                models_dir=args.models_dir,
                results_dir=args.results_dir,
                device=device,
                seed=args.seed,
                rf_max_samples=args.rf_max_samples,
                rf_min_samples_leaf=args.rf_min_samples_leaf,
                rf_n_jobs=args.rf_n_jobs,
                limit_samples=args.limit_samples,
            )
            summary[task.name][model_type] = metrics

    args.results_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.results_dir / "tree_models_all_tasks_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "loss_weighting": "excluded",
                "loss_weighting_decision": CLASS_WEIGHT_DECISION,
                "selection_metric": "val_macro_f1",
                "models": selected_models,
                "tasks": summary,
            },
            f,
            indent=2,
        )

    print("\nAll tree tasks complete. Summary:")
    for task_name, task_metrics in summary.items():
        for model_name, metrics in task_metrics.items():
            print(
                f"{task_name:7s} {model_name:4s} | acc={metrics['accuracy']:.6f} | "
                f"macro_f1={metrics['macro_f1']:.6f} | weighted_f1={metrics['weighted_f1']:.6f}"
            )
    print(f"Saved summary: {summary_path.resolve()}")


if __name__ == "__main__":
    main()
