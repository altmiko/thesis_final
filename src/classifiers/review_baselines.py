"""
Evaluate saved baseline classifiers on the processed CICIoT2023 test split.

The script loads the checkpoint files produced by the baseline training
pipeline, computes per-model metrics, and writes ROC-AUC plots for each
model/task pair.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score
from sklearn.metrics import auc
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.preprocessing import label_binarize

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.classifiers.models import get_model


MODEL_TYPES: Tuple[str, ...] = ("mlp", "cnn", "lstm", "serial", "dualpath")
MODEL_DISPLAY_NAMES: Dict[str, str] = {
    "mlp": "MLP",
    "cnn": "CNN",
    "lstm": "LSTM",
    "serial": "CNN-LSTM",
    "dualpath": "DualPath",
}


@dataclass(frozen=True)
class TaskConfig:
    name: str
    y_test_file: str
    num_classes: int


TASKS: Tuple[TaskConfig, ...] = (
    TaskConfig(name="binary", y_test_file="y_test_bin.npy", num_classes=2),
    TaskConfig(name="8class", y_test_file="y_test_cat.npy", num_classes=8),
    TaskConfig(name="34class", y_test_file="y_test.npy", num_classes=34),
)


def parse_csv_arg(value: str, valid: Sequence[str]) -> List[str]:
    if value.lower() == "all":
        return list(valid)
    requested = [item.strip().lower() for item in value.split(",") if item.strip()]
    unknown = sorted(set(requested) - set(valid))
    if unknown:
        raise ValueError(f"Unknown values {unknown}; valid choices are: {list(valid)}")
    return requested


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def load_json(path: Path) -> object:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def class_names_for_task(processed_dir: Path, task: TaskConfig) -> List[str]:
    if task.name == "binary":
        return ["Benign", "Attack"]
    if task.name == "8class":
        names = load_json(processed_dir / "category_names.json")
    else:
        names = load_json(processed_dir / "class_names.json")
    if not isinstance(names, list) or len(names) != task.num_classes:
        raise ValueError(f"Invalid class-name artifact for task={task.name}")
    return [str(name) for name in names]


def build_model(model_type: str, num_features: int, num_classes: int) -> torch.nn.Module:
    kwargs = {}
    if model_type == "mlp":
        kwargs["hidden_dims"] = (256, 128, 64)
    return get_model(model_type, num_features=num_features, num_classes=num_classes, **kwargs)


def load_state_dict(path: Path, device: torch.device) -> Dict[str, torch.Tensor]:
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def load_model(
    model_type: str,
    task: TaskConfig,
    num_features: int,
    models_dir: Path,
    device: torch.device,
) -> torch.nn.Module:
    model_path = models_dir / f"{model_type}_{task.name}.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {model_path.resolve()}")

    model = build_model(model_type, num_features=num_features, num_classes=task.num_classes)
    state_dict = load_state_dict(model_path, device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def iter_batches(n_items: int, batch_size: int) -> Iterable[Tuple[int, int]]:
    for start in range(0, n_items, batch_size):
        yield start, min(start + batch_size, n_items)


def predict_probabilities(
    model: torch.nn.Module,
    x_test: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    n_items = int(x_test.shape[0])
    logits_probe = None

    probs: np.ndarray | None = None
    preds: np.ndarray | None = None

    with torch.no_grad():
        for start, end in iter_batches(n_items, batch_size):
            xb_np = np.array(x_test[start:end], dtype=np.float32, copy=True)
            xb = torch.from_numpy(xb_np).to(device, non_blocking=True)
            output = model(xb)
            logits = output[0] if isinstance(output, tuple) else output

            if logits_probe is None:
                logits_probe = logits
                num_classes = int(logits_probe.shape[1])
                probs = np.empty((n_items, num_classes), dtype=np.float32)
                preds = np.empty(n_items, dtype=np.int64)

            batch_probs = torch.softmax(logits, dim=1).detach().cpu().numpy().astype(np.float32)
            probs[start:end] = batch_probs
            preds[start:end] = np.argmax(batch_probs, axis=1)

    if probs is None or preds is None:
        raise ValueError("No predictions were produced.")
    return probs, preds


def compute_multiclass_roc(
    y_true: np.ndarray,
    y_score: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, object]:
    y_bin = label_binarize(y_true, classes=labels)
    fpr_by_class: Dict[int, np.ndarray] = {}
    tpr_by_class: Dict[int, np.ndarray] = {}
    auc_by_class: Dict[int, float] = {}
    mean_fpr = np.linspace(0.0, 1.0, 1000)
    interpolated_tprs: List[np.ndarray] = []

    for class_idx in labels:
        class_idx_int = int(class_idx)
        y_class = y_bin[:, class_idx_int]
        if np.unique(y_class).size < 2:
            auc_by_class[class_idx_int] = float("nan")
            continue
        fpr, tpr, _ = roc_curve(y_class, y_score[:, class_idx_int])
        fpr_by_class[class_idx_int] = fpr
        tpr_by_class[class_idx_int] = tpr
        auc_by_class[class_idx_int] = float(auc(fpr, tpr))
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        interpolated_tprs.append(interp_tpr)

    macro_tpr = np.mean(np.vstack(interpolated_tprs), axis=0) if interpolated_tprs else mean_fpr
    macro_tpr[-1] = 1.0

    return {
        "macro_auc": float(
            roc_auc_score(y_true, y_score, labels=labels, average="macro", multi_class="ovr")
        ),
        "weighted_auc": float(
            roc_auc_score(y_true, y_score, labels=labels, average="weighted", multi_class="ovr")
        ),
        "per_class_auc": auc_by_class,
        "fpr_by_class": fpr_by_class,
        "tpr_by_class": tpr_by_class,
        "macro_fpr": mean_fpr,
        "macro_tpr": macro_tpr,
    }


def compute_binary_roc(y_true: np.ndarray, y_score: np.ndarray) -> Dict[str, object]:
    fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=1)
    return {
        "auc": float(roc_auc_score(y_true, y_score)),
        "fpr": fpr,
        "tpr": tpr,
    }


def plot_roc(
    task: TaskConfig,
    model_type: str,
    class_names: Sequence[str],
    roc_payload: Dict[str, object],
    output_path: Path,
) -> None:
    plt.figure(figsize=(10.5, 7.5))
    model_label = MODEL_DISPLAY_NAMES.get(model_type, model_type)

    if task.num_classes == 2:
        fpr = roc_payload["fpr"]
        tpr = roc_payload["tpr"]
        auc_value = float(roc_payload["auc"])
        plt.plot(fpr, tpr, color="#1f77b4", lw=2.5, label=f"Attack AUC = {auc_value:.4f}")
        title = f"{model_label} ROC-AUC on CICIoT2023 {task.name}"
    else:
        macro_fpr = roc_payload["macro_fpr"]
        macro_tpr = roc_payload["macro_tpr"]
        macro_auc = float(roc_payload["macro_auc"])
        plt.plot(
            macro_fpr,
            macro_tpr,
            color="black",
            lw=3.0,
            label=f"Macro OvR AUC = {macro_auc:.4f}",
        )

        fpr_by_class = roc_payload["fpr_by_class"]
        tpr_by_class = roc_payload["tpr_by_class"]
        auc_by_class = roc_payload["per_class_auc"]
        cmap = plt.get_cmap("tab20")
        for idx in range(task.num_classes):
            if idx not in fpr_by_class:
                continue
            color = cmap(idx % 20)
            label = f"{class_names[idx]} AUC = {float(auc_by_class[idx]):.3f}"
            plt.plot(fpr_by_class[idx], tpr_by_class[idx], lw=1.0, alpha=0.75, color=color, label=label)
        title = f"{model_label} OvR ROC-AUC on CICIoT2023 {task.name}"

    plt.plot([0, 1], [0, 1], color="#7f7f7f", lw=1.2, linestyle="--", label="Chance")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.02])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.grid(True, alpha=0.25)
    if task.num_classes == 34:
        plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=5.5, frameon=False)
    else:
        plt.legend(loc="lower right", fontsize=8, frameon=False)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()


def evaluate_model_task(
    model_type: str,
    task: TaskConfig,
    x_test: np.ndarray,
    y_test: np.ndarray,
    class_names: Sequence[str],
    models_dir: Path,
    output_dir: Path,
    batch_size: int,
    device: torch.device,
    make_plots: bool,
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    start_time = time.time()
    labels = np.arange(task.num_classes)
    model = load_model(model_type, task, int(x_test.shape[1]), models_dir, device)
    probs, y_pred = predict_probabilities(model, x_test, batch_size, device)

    accuracy = float(accuracy_score(y_test, y_pred))
    macro_precision = float(precision_score(y_test, y_pred, labels=labels, average="macro", zero_division=0))
    macro_recall = float(recall_score(y_test, y_pred, labels=labels, average="macro", zero_division=0))
    macro_f1 = float(f1_score(y_test, y_pred, labels=labels, average="macro", zero_division=0))
    weighted_precision = float(
        precision_score(y_test, y_pred, labels=labels, average="weighted", zero_division=0)
    )
    weighted_recall = float(recall_score(y_test, y_pred, labels=labels, average="weighted", zero_division=0))
    weighted_f1 = float(f1_score(y_test, y_pred, labels=labels, average="weighted", zero_division=0))

    per_class_precision, per_class_recall, per_class_f1, per_class_support = precision_recall_fscore_support(
        y_test,
        y_pred,
        labels=labels,
        average=None,
        zero_division=0,
    )

    if task.num_classes == 2:
        roc_payload = compute_binary_roc(y_test, probs[:, 1])
        roc_auc = float(roc_payload["auc"])
        roc_auc_macro = roc_auc
        roc_auc_weighted = roc_auc
        per_class_auc = {1: roc_auc}
    else:
        roc_payload = compute_multiclass_roc(y_test, probs, labels)
        roc_auc = float(roc_payload["macro_auc"])
        roc_auc_macro = float(roc_payload["macro_auc"])
        roc_auc_weighted = float(roc_payload["weighted_auc"])
        per_class_auc = roc_payload["per_class_auc"]

    roc_plot_path = output_dir / "roc_curves" / f"{task.name}_{model_type}_roc_auc.png"
    if make_plots:
        plot_roc(task, model_type, class_names, roc_payload, roc_plot_path)

    report = classification_report(
        y_test,
        y_pred,
        labels=labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    per_class_rows: List[Dict[str, object]] = []
    for idx in range(task.num_classes):
        per_class_rows.append(
            {
                "task": task.name,
                "model": model_type,
                "class_index": idx,
                "class_name": class_names[idx],
                "support": int(per_class_support[idx]),
                "precision": float(per_class_precision[idx]),
                "recall": float(per_class_recall[idx]),
                "f1": float(per_class_f1[idx]),
                "roc_auc_ovr": float(per_class_auc[idx]) if idx in per_class_auc else "",
            }
        )

    elapsed_sec = float(time.time() - start_time)
    summary_row: Dict[str, object] = {
        "task": task.name,
        "model": model_type,
        "model_display": MODEL_DISPLAY_NAMES.get(model_type, model_type),
        "n_test": int(len(y_test)),
        "num_classes": task.num_classes,
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,
        "roc_auc": roc_auc,
        "roc_auc_macro_ovr": roc_auc_macro,
        "roc_auc_weighted_ovr": roc_auc_weighted,
        "roc_plot": str(roc_plot_path),
        "runtime_sec": elapsed_sec,
    }

    detail_payload = {
        "summary": summary_row,
        "classification_report": report,
        "per_class": per_class_rows,
    }
    detail_path = output_dir / "json" / f"{task.name}_{model_type}_metrics.json"
    detail_path.parent.mkdir(parents=True, exist_ok=True)
    with detail_path.open("w", encoding="utf-8") as f:
        json.dump(detail_payload, f, indent=2)

    return summary_row, per_class_rows


def write_csv(path: Path, rows: Sequence[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def format_pct(value: object) -> str:
    return f"{100.0 * float(value):.2f}%"


def write_markdown_summary(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    lines = [
        "# Baseline Classifier Review",
        "",
        "Metrics are computed on the processed CICIoT2023 test split.",
        "",
    ]
    for task_name in [task.name for task in TASKS]:
        task_rows = [row for row in rows if row["task"] == task_name]
        if not task_rows:
            continue
        lines.extend(
            [
                f"## {task_name}",
                "",
                "| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 | ROC-AUC |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in sorted(task_rows, key=lambda item: str(item["model"])):
            lines.append(
                "| {model} | {acc} | {mp} | {mr} | {mf1} | {wf1} | {auc_value} |".format(
                    model=row["model_display"],
                    acc=format_pct(row["accuracy"]),
                    mp=format_pct(row["macro_precision"]),
                    mr=format_pct(row["macro_recall"]),
                    mf1=format_pct(row["macro_f1"]),
                    wf1=format_pct(row["weighted_f1"]),
                    auc_value=f"{float(row['roc_auc']):.4f}",
                )
            )
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=Path("data") / "processed")
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--output-dir", type=Path, default=Path("results") / "baseline_review")
    parser.add_argument("--models", default="all", help="Comma-separated model list or 'all'.")
    parser.add_argument("--tasks", default="all", help="Comma-separated task list or 'all'.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--limit-samples", type=int, default=None, help="Optional smoke-test cap.")
    parser.add_argument("--skip-plots", action="store_true")
    args = parser.parse_args()

    selected_models = parse_csv_arg(args.models, MODEL_TYPES)
    selected_task_names = parse_csv_arg(args.tasks, [task.name for task in TASKS])
    selected_tasks = [task for task in TASKS if task.name in selected_task_names]
    device = resolve_device(args.device)

    processed_dir = args.processed_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    x_test = np.load(processed_dir / "X_test.npy", mmap_mode="r")
    if args.limit_samples is not None:
        x_test = x_test[: args.limit_samples]

    summary_rows: List[Dict[str, object]] = []
    per_class_rows: List[Dict[str, object]] = []

    print(f"Processed data: {processed_dir.resolve()}")
    print(f"Models: {selected_models}")
    print(f"Tasks: {[task.name for task in selected_tasks]}")
    print(f"Device: {device}")
    print(f"Test rows: {x_test.shape[0]:,}; features: {x_test.shape[1]}")

    for task in selected_tasks:
        y_test = np.load(processed_dir / task.y_test_file, mmap_mode="r")
        if args.limit_samples is not None:
            y_test = y_test[: args.limit_samples]
        y_test = np.asarray(y_test, dtype=np.int64)
        class_names = class_names_for_task(processed_dir, task)

        for model_type in selected_models:
            print(f"Evaluating task={task.name} model={model_type}...")
            summary_row, task_class_rows = evaluate_model_task(
                model_type=model_type,
                task=task,
                x_test=x_test,
                y_test=y_test,
                class_names=class_names,
                models_dir=args.models_dir,
                output_dir=output_dir,
                batch_size=args.batch_size,
                device=device,
                make_plots=not args.skip_plots,
            )
            summary_rows.append(summary_row)
            per_class_rows.extend(task_class_rows)
            print(
                "  acc={acc:.4f} macro_f1={macro_f1:.4f} roc_auc={auc_value:.4f}".format(
                    acc=float(summary_row["accuracy"]),
                    macro_f1=float(summary_row["macro_f1"]),
                    auc_value=float(summary_row["roc_auc"]),
                )
            )

    summary_fields = [
        "task",
        "model",
        "model_display",
        "n_test",
        "num_classes",
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_precision",
        "weighted_recall",
        "weighted_f1",
        "roc_auc",
        "roc_auc_macro_ovr",
        "roc_auc_weighted_ovr",
        "roc_plot",
        "runtime_sec",
    ]
    per_class_fields = [
        "task",
        "model",
        "class_index",
        "class_name",
        "support",
        "precision",
        "recall",
        "f1",
        "roc_auc_ovr",
    ]
    write_csv(output_dir / "baseline_metrics_summary.csv", summary_rows, summary_fields)
    write_csv(output_dir / "baseline_per_class_metrics.csv", per_class_rows, per_class_fields)
    write_markdown_summary(output_dir / "baseline_metrics_summary.md", summary_rows)

    manifest = {
        "processed_dir": str(processed_dir.resolve()),
        "models_dir": str(args.models_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "models": selected_models,
        "tasks": [task.name for task in selected_tasks],
        "device": str(device),
        "batch_size": args.batch_size,
        "limit_samples": args.limit_samples,
        "plots": not args.skip_plots,
        "n_test": int(x_test.shape[0]),
        "n_features": int(x_test.shape[1]),
    }
    with (output_dir / "run_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Saved summary: {(output_dir / 'baseline_metrics_summary.csv').resolve()}")
    print(f"Saved per-class metrics: {(output_dir / 'baseline_per_class_metrics.csv').resolve()}")
    print(f"Saved markdown summary: {(output_dir / 'baseline_metrics_summary.md').resolve()}")
    if not args.skip_plots:
        print(f"Saved ROC plots under: {(output_dir / 'roc_curves').resolve()}")


if __name__ == "__main__":
    main()
