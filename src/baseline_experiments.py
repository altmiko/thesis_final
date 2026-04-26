"""
Baseline training pipeline for CICIoT2023 processed arrays.

This script audits the processed arrays and trains baseline models
sequentially for:
1) Binary task (2 classes)
2) Category task (8 classes)
3) Fine-grained class task (34 classes)

Requirements implemented from thesis pipeline checklist:
- Uses preprocessed arrays directly (no extra scaling, no augmentation)
- Uses plain CrossEntropyLoss (no class weights) for each framing
- Uses Adam + ReduceLROnPlateau + early stopping on validation loss
- Prints per-epoch train loss, val loss, val accuracy
- Saves trained models and classification reports
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

from models import get_model


CLASS_WEIGHT_DECISION = (
    "Class weights excluded because stratified sampling already balances the training set; "
    "applying original-distribution weights would over-penalize majority classes in the balanced sample."
)


@dataclass
class TaskConfig:
    name: str
    y_train_file: str
    y_val_file: str
    y_test_file: str
    class_weights_file: str
    num_classes: int


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_array(path: Path, mmap: bool = True) -> np.ndarray:
    if mmap:
        return np.load(path, mmap_mode="r")
    return np.load(path)


def audit_processed_data(processed_dir: Path) -> None:
    print("=" * 90)
    print("DATA AUDIT")
    print("=" * 90)

    files = [
        "X_train.npy",
        "X_val.npy",
        "X_test.npy",
        "y_train.npy",
        "y_val.npy",
        "y_test.npy",
        "y_train_cat.npy",
        "y_val_cat.npy",
        "y_test_cat.npy",
        "y_train_bin.npy",
        "y_val_bin.npy",
        "y_test_bin.npy",
        "class_weights_34.npy",
        "class_weights_8.npy",
        "class_weights_2.npy",
    ]

    arrays: Dict[str, np.ndarray] = {}
    for name in files:
        arrays[name] = load_array(processed_dir / name, mmap=True)

    print("\n[1] Shapes and dtypes")
    for name in files:
        arr = arrays[name]
        print(f"{name:20s} shape={arr.shape} dtype={arr.dtype}")

    print("\n[2] Feature count check")
    x_train = arrays["X_train.npy"]
    print(f"X_train features: {x_train.shape[1]}")
    if x_train.shape[1] != 39:
        print("WARNING: Expected 39 features in X_train.")

    print("\n[3] Label encoding checks")
    y_files = [
        "y_train.npy",
        "y_val.npy",
        "y_test.npy",
        "y_train_cat.npy",
        "y_val_cat.npy",
        "y_test_cat.npy",
        "y_train_bin.npy",
        "y_val_bin.npy",
        "y_test_bin.npy",
    ]
    for name in y_files:
        y = arrays[name]
        unique_vals = np.unique(y)
        contiguous = len(unique_vals) == int(unique_vals.max() - unique_vals.min() + 1)
        print(
            f"{name:20s} unique={len(unique_vals)} min={int(unique_vals.min())} "
            f"max={int(unique_vals.max())} integer={np.issubdtype(y.dtype, np.integer)} "
            f"starts_at_0={bool(unique_vals.min() == 0)} contiguous={contiguous}"
        )

    print("\n[4] Class count checks")
    print(f"34-class unique count: {len(np.unique(arrays['y_train.npy']))}")
    print(f"8-class unique count: {len(np.unique(arrays['y_train_cat.npy']))}")
    print(f"2-class unique count: {len(np.unique(arrays['y_train_bin.npy']))}")

    print("\n[5] Class weights shape checks")
    for name, expected_len in [
        ("class_weights_34.npy", 34),
        ("class_weights_8.npy", 8),
        ("class_weights_2.npy", 2),
    ]:
        w = arrays[name]
        print(
            f"{name:20s} len={len(w)} expected={expected_len} "
            f"min={float(np.min(w)):.6f} max={float(np.max(w)):.6f}"
        )
        if len(w) != expected_len:
            print(f"WARNING: {name} length mismatch.")

    print("\n[6] y_train value_counts (34-class)")
    y_train_34 = arrays["y_train.npy"]
    vals, cnts = np.unique(y_train_34, return_counts=True)
    order = np.argsort(cnts)[::-1]
    total = len(y_train_34)
    for idx in order[:15]:
        cls = int(vals[idx])
        c = int(cnts[idx])
        pct = 100.0 * c / total
        print(f"class={cls:2d} count={c:9d} pct={pct:7.3f}%")
    max_pct = 100.0 * int(cnts[order[0]]) / total
    print(f"dominant class share: {max_pct:.3f}%")

    print("\n[7] Scaling sanity (X_train, first 5 columns)")
    x_train_np = np.asarray(arrays["X_train.npy"], dtype=np.float64)
    for j in range(min(5, x_train_np.shape[1])):
        col = x_train_np[:, j]
        print(
            f"col{j:02d}: mean={float(np.mean(col)):+.6f} std={float(np.std(col)):.6f} "
            f"min={float(np.min(col)):+.6f} max={float(np.max(col)):+.6f}"
        )

    print("\n[8] NaN/Inf checks in X arrays")
    for name in ["X_train.npy", "X_val.npy", "X_test.npy"]:
        x = np.asarray(arrays[name])
        has_nan = bool(np.isnan(x).any())
        has_pos_inf = bool(np.isposinf(x).any())
        has_neg_inf = bool(np.isneginf(x).any())
        print(f"{name:20s} nan={has_nan} +inf={has_pos_inf} -inf={has_neg_inf}")

    print("=" * 90)
    print("END DATA AUDIT")
    print("=" * 90)


def prepare_tensors(x: np.ndarray, y: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
    x_tensor = torch.from_numpy(np.array(x, dtype=np.float32, copy=True))
    y_tensor = torch.from_numpy(np.array(y, dtype=np.int64, copy=True))
    return x_tensor, y_tensor


def make_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    x_tensor, y_tensor = prepare_tensors(x, y)
    ds = TensorDataset(x_tensor, y_tensor)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    total = 0
    all_true: List[np.ndarray] = []
    all_pred: List[np.ndarray] = []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            output = model(xb)
            logits = output[0] if isinstance(output, tuple) else output
            loss = criterion(logits, yb)

            preds = torch.argmax(logits, dim=1)
            n = yb.size(0)
            total += n
            total_loss += loss.item() * n

            all_true.append(yb.detach().cpu().numpy())
            all_pred.append(preds.detach().cpu().numpy())

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    avg_loss = float(total_loss / max(total, 1))
    acc = float(accuracy_score(y_true, y_pred))

    return avg_loss, acc, y_true, y_pred


def train_single_task(
    model_type: str,
    task: TaskConfig,
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    class_weights: np.ndarray,
    device: torch.device,
    epochs: int = 5,
    batch_size: int = 2048,
    lr: float = 1e-3,
    early_stop_patience: int = 5,
) -> Dict[str, float]:
    num_features = int(x_train.shape[1])

    print("\n" + "-" * 90)
    print(f"Starting: task={task.name}, model={model_type}")
    print(f"X_train shape: {x_train.shape}")
    print(f"num_classes: {task.num_classes}")
    print(f"loss_weighting_decision: {CLASS_WEIGHT_DECISION}")
    print(f"device: {device}")
    print("-" * 90)

    model_kwargs = {}
    if model_type == "mlp":
        model_kwargs["hidden_dims"] = (256, 128, 64)
    model = get_model(model_type, num_features=num_features, num_classes=task.num_classes, **model_kwargs).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=3,
        factor=0.5,
    )

    train_loader = make_loader(x_train, y_train, batch_size=batch_size, shuffle=True)
    val_loader = make_loader(x_val, y_val, batch_size=batch_size, shuffle=False)
    test_loader = make_loader(x_test, y_test, batch_size=batch_size, shuffle=False)

    best_state = None
    best_val_loss = float("inf")
    wait = 0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        total = 0
        correct = 0

        for batch_idx, (xb, yb) in enumerate(train_loader, start=1):
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            output = model(xb)
            logits = output[0] if isinstance(output, tuple) else output
            loss = criterion(logits, yb)

            if epoch == 1 and batch_idx == 1:
                print(f"[{task.name}:{model_type}] first batch loss={loss.item():.6f}")

            loss.backward()
            optimizer.step()

            preds = torch.argmax(logits, dim=1)
            n = yb.size(0)
            total += n
            correct += int((preds == yb).sum().item())
            running_loss += loss.item() * n

        train_loss = float(running_loss / max(total, 1))
        train_acc = float(correct / max(total, 1))

        val_loss, val_acc, val_true, val_pred = evaluate(model, val_loader, criterion, device)
        val_macro_f1 = float(f1_score(val_true, val_pred, average="macro", zero_division=0))
        val_weighted_f1 = float(f1_score(val_true, val_pred, average="weighted", zero_division=0))
        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"[{task.name}:{model_type}] epoch {epoch:02d}/{epochs} | "
            f"train_loss={train_loss:.6f} | train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.6f} | val_acc={val_acc:.4f} | "
            f"val_macro_f1={val_macro_f1:.4f} | "
            f"lr={current_lr:.6g}"
        )

        if task.name == "binary" and epoch == 3 and train_acc < 0.80:
            print(
                "DIAGNOSTIC STOP: Binary training accuracy did not exceed 80% by epoch 3. "
                "This suggests a pipeline issue (labels, loss wiring, or dataloader mismatch)."
            )
            break

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= early_stop_patience:
                print(
                    f"[{task.name}:{model_type}] Early stopping at epoch {epoch} "
                    f"(patience={early_stop_patience})."
                )
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss, test_acc, y_true, y_pred = evaluate(model, test_loader, criterion, device)
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    per_class_f1 = f1_score(y_true, y_pred, average=None, labels=np.arange(task.num_classes), zero_division=0)

    print("\n" + "*" * 90)
    print(f"TEST RESULTS [{task.name}:{model_type}]")
    print(f"test_loss: {test_loss:.6f}")
    print(f"accuracy: {test_acc:.6f}")
    print(f"macro_f1: {macro_f1:.6f}")
    print(f"weighted_f1: {weighted_f1:.6f}")
    print("*" * 90)

    report_dict = classification_report(
        y_true,
        y_pred,
        labels=np.arange(task.num_classes),
        output_dict=True,
        zero_division=0,
    )
    report_text = classification_report(
        y_true,
        y_pred,
        labels=np.arange(task.num_classes),
        zero_division=0,
    )

    models_dir = Path("models")
    results_dir = Path("results")
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / f"{model_type}_{task.name}.pt"
    torch.save(model.state_dict(), model_path)

    metrics_payload = {
        "model": model_type,
        "task": task.name,
        "num_classes": task.num_classes,
        "loss_weighting": "excluded",
        "loss_weighting_decision": CLASS_WEIGHT_DECISION,
        "provided_class_weights_for_reference": class_weights.tolist(),
        "test_loss": test_loss,
        "accuracy": test_acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class_f1": {str(i): float(v) for i, v in enumerate(per_class_f1.tolist())},
        "classification_report": report_dict,
    }

    with (results_dir / f"{model_type}_{task.name}_classification_report.json").open("w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    with (results_dir / f"{model_type}_{task.name}_classification_report.txt").open("w", encoding="utf-8") as f:
        f.write(report_text)

    print(f"Saved model: {model_path.resolve()}")
    print(f"Saved reports: {(results_dir / f'{model_type}_{task.name}_classification_report.json').resolve()}")

    return {
        "accuracy": test_acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    }


def main() -> None:
    set_seed(42)

    processed_dir = Path("data") / "processed"
    device = resolve_device()

    if not processed_dir.exists():
        raise FileNotFoundError(f"Processed data directory not found: {processed_dir.resolve()}")

    audit_processed_data(processed_dir)

    x_train = load_array(processed_dir / "X_train.npy", mmap=True)
    x_val = load_array(processed_dir / "X_val.npy", mmap=True)
    x_test = load_array(processed_dir / "X_test.npy", mmap=True)

    tasks = [
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
    ]
    model_types = ["mlp", "cnn", "lstm", "serial", "dualpath"]

    summary = {}

    for task in tasks:
        y_train = load_array(processed_dir / task.y_train_file, mmap=True)
        y_val = load_array(processed_dir / task.y_val_file, mmap=True)
        y_test = load_array(processed_dir / task.y_test_file, mmap=True)
        class_weights = np.load(processed_dir / task.class_weights_file).astype(np.float32, copy=False)

        if len(class_weights) != task.num_classes:
            raise ValueError(
                f"Class weights length mismatch for {task.name}: "
                f"got {len(class_weights)}, expected {task.num_classes}"
            )

        summary[task.name] = {}
        for model_type in model_types:
            metrics = train_single_task(
                model_type=model_type,
                task=task,
                x_train=x_train,
                x_val=x_val,
                x_test=x_test,
                y_train=y_train,
                y_val=y_val,
                y_test=y_test,
                class_weights=class_weights,
                device=device,
                epochs=5,
                batch_size=2048,
                lr=1e-3,
                early_stop_patience=5,
            )
            summary[task.name][model_type] = metrics

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    with (results_dir / "all_models_all_tasks_summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "loss_weighting": "excluded",
                "loss_weighting_decision": CLASS_WEIGHT_DECISION,
                "models": model_types,
                "tasks": summary,
            },
            f,
            indent=2,
        )

    print("\nAll tasks complete. Summary:")
    for task_name, task_metrics in summary.items():
        for model_name, metrics in task_metrics.items():
            print(
                f"{task_name:7s} {model_name:8s} | acc={metrics['accuracy']:.6f} | "
                f"macro_f1={metrics['macro_f1']:.6f} | weighted_f1={metrics['weighted_f1']:.6f}"
            )


if __name__ == "__main__":
    main()
