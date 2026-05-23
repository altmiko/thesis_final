"""
Generate per-feature adversarial delta reports for CICIoT2023 models.

Runs PGD (eps=0.05/0.10/0.30) and CW attacks against MLP, LSTM, and
CNN-LSTM (serial) 8-class checkpoints, computes per-feature aggregate
statistics in raw feature space, and writes a sortable HTML report.
"""

from __future__ import annotations

import argparse
import html
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from src.attack.adversarial_attacks import compute_attack_metrics
from src.attack.adversarial_attacks import load_model
from src.attack.adversarial_attacks import run_attack
from src.preprocessing.feature_groups import FEATURE_NAMES


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"

FLOAT_TOL = 0.01
VAR_STD_REL_TOL = 0.05
TARGET_ATTACK_CLASSES = ("DDoS", "DoS", "Mirai", "Recon")

MODEL_CONFIGS: Sequence[Dict[str, str]] = (
    {"tag": "mlp", "label": "MLP", "checkpoint": "mlp_8class.pt"},
    {"tag": "lstm", "label": "LSTM", "checkpoint": "lstm_8class.pt"},
    {"tag": "serial", "label": "CNN-LSTM", "checkpoint": "serial_8class.pt"},
)

ATTACK_CONFIGS: Sequence[Dict[str, object]] = (
    {"attack_name": "pgd", "attack_label": "PGD eps=0.05", "eps": 0.05},
    {"attack_name": "pgd", "attack_label": "PGD eps=0.10", "eps": 0.10},
    {"attack_name": "pgd", "attack_label": "PGD eps=0.30", "eps": 0.30},
    {"attack_name": "cw", "attack_label": "CW", "eps": None},
)

NONNEG_FEATURES = (
    "Header_Length",
    "Rate",
    "Time_To_Live",
    "Tot sum",
    "Min",
    "Max",
    "AVG",
    "Std",
    "Tot size",
    "IAT",
    "Number",
    "Variance",
    "fin_flag_number",
    "syn_flag_number",
    "rst_flag_number",
    "psh_flag_number",
    "ack_flag_number",
    "ece_flag_number",
    "cwr_flag_number",
    "ack_count",
    "syn_count",
    "fin_count",
    "rst_count",
)

BINARY_FEATURES = (
    "HTTP",
    "HTTPS",
    "DNS",
    "Telnet",
    "SMTP",
    "SSH",
    "IRC",
    "TCP",
    "UDP",
    "DHCP",
    "ARP",
    "ICMP",
    "IGMP",
    "IPv",
    "LLC",
)

PROTOCOL_ONEHOT_FEATURES = ("TCP", "UDP", "ICMP", "IGMP")


def _format_float(v: float) -> str:
    return f"{float(v):.4f}"


def _feature_index_map(feature_names: Sequence[str]) -> Dict[str, int]:
    return {name: idx for idx, name in enumerate(feature_names)}


def _resolve_device(device: str) -> str:
    device_l = device.lower().strip()
    if device_l == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable. Falling back to CPU.")
        return "cpu"
    return device_l


def _load_required_files() -> Dict[str, object]:
    required = [
        DATA_DIR / "X_test.npy",
        DATA_DIR / "y_test_cat.npy",
        DATA_DIR / "category_names.json",
        DATA_DIR / "scaler.pkl",
    ]
    missing = [p for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(str(p) for p in missing))

    for config in MODEL_CONFIGS:
        ckpt = MODELS_DIR / str(config["checkpoint"])
        if not ckpt.exists():
            raise FileNotFoundError(f"Missing model checkpoint: {ckpt}")

    with (DATA_DIR / "category_names.json").open("r", encoding="utf-8") as f:
        category_names = json.load(f)
    if not isinstance(category_names, list):
        raise ValueError("category_names.json must be a JSON list")

    with (DATA_DIR / "scaler.pkl").open("rb") as f:
        scaler = pickle.load(f)

    x_test = np.load(DATA_DIR / "X_test.npy", mmap_mode="r")
    y_test_cat = np.load(DATA_DIR / "y_test_cat.npy")

    return {
        "x_test": x_test,
        "y_test_cat": y_test_cat,
        "category_names": [str(x) for x in category_names],
        "scaler": scaler,
    }


def _stratified_subset_indices(
    y_test_cat: np.ndarray,
    category_names: Sequence[str],
    n: int,
    seed: int,
) -> np.ndarray:
    name_to_idx = {name: i for i, name in enumerate(category_names)}
    missing = [name for name in TARGET_ATTACK_CLASSES if name not in name_to_idx]
    if missing:
        raise ValueError(f"Missing target classes in category names: {missing}")

    target_ids = np.array([name_to_idx[name] for name in TARGET_ATTACK_CLASSES], dtype=np.int64)
    mask = np.isin(y_test_cat, target_ids)
    idx_pool = np.where(mask)[0]

    if n > len(idx_pool):
        raise ValueError(
            f"Requested n={n} but only {len(idx_pool)} samples are available "
            f"for classes {TARGET_ATTACK_CLASSES}"
        )

    y_pool = y_test_cat[idx_pool]
    chosen, _ = train_test_split(
        idx_pool,
        train_size=n,
        random_state=seed,
        stratify=y_pool,
        shuffle=True,
    )
    return np.asarray(chosen, dtype=np.int64)


def _sample_distribution_text(y: np.ndarray, category_names: Sequence[str]) -> str:
    vals, cnts = np.unique(y, return_counts=True)
    parts: List[str] = []
    total = len(y)
    for val, cnt in zip(vals, cnts):
        name = category_names[int(val)] if int(val) < len(category_names) else str(int(val))
        pct = 100.0 * float(cnt) / float(total)
        parts.append(f"{name}: {int(cnt)} ({pct:.1f}%)")
    return ", ".join(parts)


def _evaluate_constraints(
    x_adv_raw: np.ndarray,
    feature_names: Sequence[str],
) -> Tuple[np.ndarray, Dict[int, List[Tuple[str, np.ndarray]]]]:
    idx = _feature_index_map(feature_names)
    n_samples, n_features = x_adv_raw.shape

    per_feature_viol = np.zeros((n_samples, n_features), dtype=bool)
    reason_by_feature: Dict[int, List[Tuple[str, np.ndarray]]] = {}

    def add_rule(mask: np.ndarray, rule_reason: str, involved_feats: Sequence[str]) -> None:
        if mask.dtype != bool:
            mask = mask.astype(bool)
        for feat in involved_feats:
            j = idx.get(feat)
            if j is None:
                continue
            per_feature_viol[:, j] |= mask
            reason_by_feature.setdefault(j, []).append((rule_reason, mask))

    for feat in NONNEG_FEATURES:
        j = idx.get(feat)
        if j is None:
            continue
        mask = x_adv_raw[:, j] < -FLOAT_TOL
        add_rule(mask, "non-negativity", (feat,))

    for feat in BINARY_FEATURES:
        j = idx.get(feat)
        if j is None:
            continue
        raw = x_adv_raw[:, j]
        rounded = np.round(raw)
        not_integer_like = np.abs(raw - rounded) > FLOAT_TOL
        not_binary = ~np.isin(rounded.astype(np.int64), np.array([0, 1], dtype=np.int64))
        mask = not_integer_like | not_binary
        add_rule(mask, "binary {0,1}", (feat,))

    if "Min" in idx and "Max" in idx:
        min_vals = x_adv_raw[:, idx["Min"]]
        max_vals = x_adv_raw[:, idx["Max"]]
        min_gt_max = min_vals > (max_vals + FLOAT_TOL)
        add_rule(min_gt_max, "Min <= Max consistency", ("Min", "Max"))

    if "Std" in idx and "Variance" in idx:
        std_vals = x_adv_raw[:, idx["Std"]]
        var_vals = x_adv_raw[:, idx["Variance"]]
        expected_var = std_vals ** 2
        rel_err = np.abs(var_vals - expected_var) / (expected_var + 1e-8)
        std_var_bad = rel_err > VAR_STD_REL_TOL
        add_rule(std_var_bad, "Std/Variance consistency", ("Std", "Variance"))

    onehot_idx = [idx[name] for name in PROTOCOL_ONEHOT_FEATURES if name in idx]
    if len(onehot_idx) == len(PROTOCOL_ONEHOT_FEATURES):
        rounded = np.round(x_adv_raw[:, onehot_idx]).astype(np.int64)
        onehot_bad = rounded.sum(axis=1) != 1
        add_rule(onehot_bad, "protocol type one-hot sum != 1", PROTOCOL_ONEHOT_FEATURES)

    return per_feature_viol, reason_by_feature


def _build_feature_rows(
    x_clean_raw: np.ndarray,
    x_adv_raw: np.ndarray,
    feature_names: Sequence[str],
    per_feature_viol: np.ndarray,
) -> List[Dict[str, float]]:
    delta = x_adv_raw - x_clean_raw
    mean_orig = np.mean(x_clean_raw, axis=0)
    mean_adv = np.mean(x_adv_raw, axis=0)
    mean_delta = np.mean(delta, axis=0)
    std_delta = np.std(delta, axis=0)
    violation_rate = np.mean(per_feature_viol, axis=0) * 100.0

    rows: List[Dict[str, float]] = []
    for i, feature in enumerate(feature_names):
        rows.append(
            {
                "feature": feature,
                "mean_original": float(mean_orig[i]),
                "mean_perturbed": float(mean_adv[i]),
                "mean_delta": float(mean_delta[i]),
                "std_delta": float(std_delta[i]),
                "violation_rate": float(violation_rate[i]),
            }
        )
    return rows


def _build_violation_examples(
    x_clean_raw: np.ndarray,
    x_adv_raw: np.ndarray,
    feature_rows: Sequence[Dict[str, float]],
    per_feature_viol: np.ndarray,
    reason_by_feature: Dict[int, List[Tuple[str, np.ndarray]]],
    max_examples: int = 3,
) -> List[str]:
    examples: List[str] = []
    ranked = sorted(feature_rows, key=lambda r: (-r["violation_rate"], r["feature"]))
    feature_to_idx = _feature_index_map([row["feature"] for row in feature_rows])

    for row in ranked:
        if row["violation_rate"] <= 0.0:
            continue
        feature_name = str(row["feature"])
        j = feature_to_idx[feature_name]
        sample_idx_candidates = np.where(per_feature_viol[:, j])[0]
        if sample_idx_candidates.size == 0:
            continue

        i = int(sample_idx_candidates[0])
        reason = "constraint"
        for reason_name, mask in reason_by_feature.get(j, []):
            if bool(mask[i]):
                reason = reason_name
                break

        examples.append(
            f"{feature_name}: {_format_float(x_clean_raw[i, j])} -> {_format_float(x_adv_raw[i, j])} "
            f"(violates {reason})"
        )
        if len(examples) >= max_examples:
            break

    return examples


def _render_table_rows(rows: Sequence[Dict[str, float]]) -> str:
    html_rows: List[str] = []
    for row in rows:
        cls = "viol-row" if row["violation_rate"] > 0.0 else ""
        html_rows.append(
            "<tr class=\"{}\">"
            "<td>{}</td>"
            "<td data-sort=\"{:.10f}\">{}</td>"
            "<td data-sort=\"{:.10f}\">{}</td>"
            "<td data-sort=\"{:.10f}\">{}</td>"
            "<td data-sort=\"{:.10f}\">{}</td>"
            "<td data-sort=\"{:.10f}\">{}%</td>"
            "</tr>".format(
                cls,
                html.escape(str(row["feature"])),
                row["mean_original"],
                _format_float(row["mean_original"]),
                row["mean_perturbed"],
                _format_float(row["mean_perturbed"]),
                row["mean_delta"],
                _format_float(row["mean_delta"]),
                row["std_delta"],
                _format_float(row["std_delta"]),
                row["violation_rate"],
                _format_float(row["violation_rate"]),
            )
        )
    return "\n".join(html_rows)


def _render_html_report(
    *,
    n: int,
    sample_distribution: str,
    results: Sequence[Dict[str, object]],
) -> str:
    sections: List[str] = []

    for idx, section in enumerate(results):
        feature_rows = section["feature_rows"]
        table_rows = _render_table_rows(feature_rows)
        examples = section["example_lines"]

        examples_html = ""
        if examples:
            examples_html = "\n".join(f"<li>{html.escape(line)}</li>" for line in examples)
            examples_html = f"<div class=\"examples\"><div class=\"examples-title\">Example violations</div><ul>{examples_html}</ul></div>"

        sections.append(
            f"""
<section class=\"card\">
  <h2>{html.escape(str(section['model_label']))} x {html.escape(str(section['attack_label']))}</h2>
  <div class=\"meta\">
    <div><strong>ASR:</strong> {float(section['asr']) * 100.0:.2f}%</div>
    <div><strong>Samples originally correct:</strong> {int(section['samples_originally_correct'])}</div>
    <div><strong>Samples flipped:</strong> {int(section['samples_flipped'])}</div>
  </div>
  {examples_html}
  <table class=\"sortable\" id=\"tbl-{idx}\">
    <thead>
      <tr>
        <th data-type=\"text\">Feature</th>
        <th data-type=\"num\">Mean Original</th>
        <th data-type=\"num\">Mean Perturbed</th>
        <th data-type=\"num\">Mean Delta</th>
        <th data-type=\"num\">Std Delta</th>
        <th data-type=\"num\">Violation Rate</th>
      </tr>
    </thead>
    <tbody>
      {table_rows}
    </tbody>
  </table>
</section>
"""
        )

    return f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>CICIoT2023 Adversarial Delta Report</title>
  <style>
    :root {{
      --bg: #f6f8fb;
      --card: #ffffff;
      --line: #d9e0ea;
      --text: #132033;
      --muted: #4a6078;
      --accent: #0b6bb5;
      --bad: #ffe6e6;
      --bad-border: #cc2f2f;
    }}
    body {{
      margin: 0;
      background: radial-gradient(circle at top left, #e9f3ff, var(--bg));
      color: var(--text);
      font-family: "Segoe UI", Tahoma, sans-serif;
      line-height: 1.35;
    }}
    .wrap {{
      max-width: 1300px;
      margin: 24px auto;
      padding: 0 16px 24px;
    }}
    .header {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 16px 18px;
      margin-bottom: 18px;
    }}
    .header h1 {{
      margin: 0 0 8px;
      font-size: 1.35rem;
    }}
    .sub {{
      color: var(--muted);
      font-size: 0.95rem;
    }}
    .card {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 14px 16px;
      margin-bottom: 16px;
      overflow-x: auto;
    }}
    .card h2 {{
      margin: 0 0 8px;
      font-size: 1.1rem;
      color: var(--accent);
    }}
    .meta {{
      display: flex;
      flex-wrap: wrap;
      gap: 18px;
      margin-bottom: 10px;
      font-size: 0.92rem;
    }}
    .examples {{
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 8px 10px;
      margin-bottom: 10px;
      background: #fdfefe;
    }}
    .examples-title {{
      font-weight: 700;
      margin-bottom: 6px;
      color: #1b3d63;
    }}
    .examples ul {{
      margin: 0;
      padding-left: 18px;
    }}
    table {{
      border-collapse: collapse;
      width: 100%;
      min-width: 860px;
      font-size: 0.9rem;
    }}
    th, td {{
      border: 1px solid var(--line);
      padding: 7px 8px;
      text-align: left;
      vertical-align: middle;
    }}
    th {{
      background: #f0f5fb;
      cursor: pointer;
      user-select: none;
    }}
    tr:nth-child(even) {{
      background: #fbfdff;
    }}
    tr.viol-row {{
      background: var(--bad);
      border-left: 3px solid var(--bad-border);
    }}
  </style>
</head>
<body>
  <div class=\"wrap\">
    <section class=\"header\">
      <h1>CICIoT2023 Adversarial Delta Report</h1>
      <div class=\"sub\">Sample size: {n} | Stratified classes: DDoS, DoS, Mirai, Recon</div>
      <div class=\"sub\">Sample distribution: {html.escape(sample_distribution)}</div>
      <div class=\"sub\">Victim models: MLP, LSTM, CNN-LSTM | Attacks: PGD (eps=0.05/0.10/0.30), CW</div>
    </section>
    {''.join(sections)}
  </div>
  <script>
    (function() {{
      function getCellValue(row, idx) {{
        var cell = row.children[idx];
        var sort = cell.getAttribute('data-sort');
        return sort !== null ? sort : cell.innerText;
      }}

      function comparator(idx, type, asc) {{
        return function(a, b) {{
          var v1 = getCellValue(asc ? a : b, idx);
          var v2 = getCellValue(asc ? b : a, idx);
          if (type === 'num') {{
            return parseFloat(v1) - parseFloat(v2);
          }}
          return String(v1).localeCompare(String(v2));
        }};
      }}

      document.querySelectorAll('table.sortable th').forEach(function(th) {{
        var table = th.closest('table');
        var tbody = table.querySelector('tbody');
        var idx = Array.prototype.indexOf.call(th.parentNode.children, th);
        var type = th.getAttribute('data-type') || 'text';
        var asc = true;

        th.addEventListener('click', function() {{
          var rows = Array.prototype.slice.call(tbody.querySelectorAll('tr'));
          rows.sort(comparator(idx, type, asc));
          asc = !asc;
          rows.forEach(function(r) {{ tbody.appendChild(r); }});
        }});
      }});
    }})();
  </script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate adversarial per-feature delta HTML report")
    parser.add_argument("--n", type=int, default=500, help="Stratified sample size from test set")
    parser.add_argument("--output", default="delta_report.html", help="Output HTML path")
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pgd-steps", type=int, default=40)
    parser.add_argument("--cw-steps", type=int, default=100)
    parser.add_argument("--cw-lr", type=float, default=0.01)
    parser.add_argument("--cw-c", type=float, default=1.0)
    parser.add_argument("--cw-kappa", type=float, default=0.0)
    args = parser.parse_args()

    device = _resolve_device(args.device)
    loaded = _load_required_files()

    x_test = loaded["x_test"]
    y_test_cat = np.asarray(loaded["y_test_cat"], dtype=np.int64)
    category_names = loaded["category_names"]
    scaler = loaded["scaler"]

    idx = _stratified_subset_indices(
        y_test_cat=y_test_cat,
        category_names=category_names,
        n=int(args.n),
        seed=int(args.seed),
    )

    x_sample_scaled = np.array(x_test[idx], dtype=np.float32, copy=True)
    y_sample = np.array(y_test_cat[idx], dtype=np.int64, copy=True)

    print(f"Sampled N={len(x_sample_scaled)} from target classes {TARGET_ATTACK_CLASSES}")
    print("Distribution:", _sample_distribution_text(y_sample, category_names))

    n_features = x_sample_scaled.shape[1]
    if n_features != len(FEATURE_NAMES):
        raise ValueError(
            f"Feature dimension mismatch: X has {n_features}, FEATURE_NAMES has {len(FEATURE_NAMES)}"
        )

    all_sections: List[Dict[str, object]] = []

    for model_cfg in MODEL_CONFIGS:
        model = load_model(
            model_path=str(MODELS_DIR / str(model_cfg["checkpoint"])),
            num_features=n_features,
            num_classes=8,
            device=device,
        )

        for attack_cfg in ATTACK_CONFIGS:
            attack_name = str(attack_cfg["attack_name"])
            eps = attack_cfg["eps"]
            eps_value = 0.0 if eps is None else float(eps)
            attack_label = str(attack_cfg["attack_label"])

            attack_kwargs: Dict[str, object] = {}
            if attack_name == "pgd":
                attack_kwargs["steps"] = int(args.pgd_steps)
                attack_kwargs["alpha"] = eps_value / 4.0 if eps_value > 0 else 0.01
            elif attack_name == "cw":
                attack_kwargs["steps"] = int(args.cw_steps)
                attack_kwargs["lr"] = float(args.cw_lr)
                attack_kwargs["c"] = float(args.cw_c)
                attack_kwargs["kappa"] = float(args.cw_kappa)

            print(f"Running {model_cfg['label']} + {attack_label}")
            result = run_attack(
                model=model,
                X=x_sample_scaled,
                y=y_sample,
                attack_name=attack_name,
                eps=eps_value,
                batch_size=int(args.batch_size),
                device=device,
                **attack_kwargs,
            )

            metrics = compute_attack_metrics(result)
            x_clean_raw = np.asarray(scaler.inverse_transform(result["X_clean"]), dtype=np.float32)
            x_adv_raw = np.asarray(scaler.inverse_transform(result["X_adv"]), dtype=np.float32)

            per_feature_viol, reason_by_feature = _evaluate_constraints(
                x_adv_raw=x_adv_raw,
                feature_names=FEATURE_NAMES,
            )
            feature_rows = _build_feature_rows(
                x_clean_raw=x_clean_raw,
                x_adv_raw=x_adv_raw,
                feature_names=FEATURE_NAMES,
                per_feature_viol=per_feature_viol,
            )
            example_lines = _build_violation_examples(
                x_clean_raw=x_clean_raw,
                x_adv_raw=x_adv_raw,
                feature_rows=feature_rows,
                per_feature_viol=per_feature_viol,
                reason_by_feature=reason_by_feature,
            )

            all_sections.append(
                {
                    "model_label": model_cfg["label"],
                    "model_tag": model_cfg["tag"],
                    "attack_name": attack_name,
                    "attack_label": attack_label,
                    "eps": eps,
                    "asr": float(metrics["attack_success_rate_raw"]),
                    "samples_originally_correct": int(metrics["samples_originally_correct"]),
                    "samples_flipped": int(metrics["samples_flipped"]),
                    "feature_rows": feature_rows,
                    "example_lines": example_lines,
                }
            )

    html_text = _render_html_report(
        n=len(x_sample_scaled),
        sample_distribution=_sample_distribution_text(y_sample, category_names),
        results=all_sections,
    )

    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = Path.cwd() / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html_text, encoding="utf-8")

    print(f"Wrote report: {out_path}")


if __name__ == "__main__":
    main()
