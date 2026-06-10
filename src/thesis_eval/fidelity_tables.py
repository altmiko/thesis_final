"""Compute and persist Wasserstein, MMD, and supplementary JS fidelity tables."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn

from src.preprocessing.feature_groups import FEATURE_NAMES
from src.thesis_eval.metrics import fidelity


CLASS_ORDER = [
    "DoS",
    "DDoS",
    "Mirai",
    "BruteForce",
    "Recon",
    "Web",
    "Spoofing",
]
CLASS_TO_ID = {
    "BruteForce": 1,
    "DDoS": 2,
    "DoS": 3,
    "Mirai": 4,
    "Recon": 5,
    "Spoofing": 6,
    "Web": 7,
}
METHOD_ORDER = [
    "PGD",
    "CW",
    "Latent-PGD",
    "Latent-CW",
    "cInput CPGD",
    "cInput CCW",
]
HEADLINE_METHODS = METHOD_ORDER[:4]
CINPUT_METHODS = METHOD_ORDER[4:]
SCALING_DESCRIPTION = (
    "39-feature RobustScaler-transformed model space "
    "(data/processed/scaler.pkl); no additional feature scaling"
)

dataframe = pd.DataFrame


@dataclass(frozen=True)
class PopulationVectorStatus:
    available: bool
    path: Path
    reason: str


def _stable_seed(seed: int, *parts: str) -> int:
    payload = "|".join([str(seed), *parts]).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:4], "little")


def _require_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    return path


def inspect_cinput_vectors(repo_root: Path) -> PopulationVectorStatus:
    run_dir = (
        Path(repo_root)
        / "outputs"
        / "latent_attacks"
        / "constrained_input_baselines_20260602_032651_seed42"
    )
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Missing canonical cInput run: {run_dir}")
    per_sample_path = _require_file(
        run_dir / "per_sample_results.csv",
        "canonical cInput per-sample results",
    )
    try:
        attack_types = set(
            pd.read_csv(per_sample_path, usecols=["attack_type"])[
                "attack_type"
            ].astype(str)
        )
    except ValueError as exc:
        raise RuntimeError(
            f"{per_sample_path} is missing required column 'attack_type'"
        ) from exc
    required = {"cinput-pgd", "cinput-cw"}
    missing = sorted(required - attack_types)
    if missing:
        raise RuntimeError(
            "Canonical cInput run is missing required populations: "
            + ", ".join(missing)
        )
    vector_files = sorted(
        path
        for path in run_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".npy", ".npz", ".parquet"}
    )
    if not vector_files:
        return PopulationVectorStatus(
            available=False,
            path=run_dir,
            reason=(
                "canonical cInput run has outcomes but no full 39-feature "
                "adversarial vectors; the eight-sample exhibit is not used"
            ),
        )
    return PopulationVectorStatus(
        available=True,
        path=vector_files[0],
        reason="",
    )


def unavailable_metric_row(
    *,
    attack_class: str,
    method: str,
    metric: str,
    reason: str,
    source_path: str = "",
) -> dict[str, Any]:
    return {
        "attack_class": attack_class,
        "method": method,
        "metric": metric,
        "value": "N/A",
        "ci_low": "N/A",
        "ci_high": "N/A",
        "status": f"# TODO: {reason}",
        "source_path": source_path,
    }


def _subsample_rows(
    values: np.ndarray, *, n: int, seed: int
) -> np.ndarray:
    array = np.asarray(values)
    if array.shape[0] < n:
        raise ValueError(
            f"Need fixed subsample N={n}, found only {array.shape[0]} rows"
        )
    if array.shape[0] == n:
        return np.asarray(array, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.choice(array.shape[0], size=n, replace=False)
    return np.asarray(array[indices], dtype=np.float64)


def _unavailable_rows(
    *,
    attack_class: str,
    method: str,
    reason: str,
    source_path: str,
    feature_names: list[str],
) -> dict[str, list[dict[str, Any]]]:
    common = {
        "attack_class": attack_class,
        "method": method,
        "value": "N/A",
        "status": f"# TODO: {reason}",
        "source_path": source_path,
        "scaling": SCALING_DESCRIPTION,
    }
    pc1 = {
        **common,
        "metric": "wasserstein_pc1",
        "ci_low": "N/A",
        "ci_high": "N/A",
        "n_clean": "N/A",
        "n_attack": "N/A",
        "bootstrap_n_clean": "N/A",
        "bootstrap_n_attack": "N/A",
        "pc1_explained_variance_ratio": "N/A",
    }
    mmd = {
        **common,
        "metric": "mmd2_fullspace",
        "ci_low": "N/A",
        "ci_high": "N/A",
        "bandwidth": "N/A",
        "subsample_n": "N/A",
        "seed": "N/A",
    }
    js = {
        **common,
        "metric": "js_pc1",
        "bins": "N/A",
        "n_clean": "N/A",
        "n_attack": "N/A",
    }
    per_feature = [
        {
            **common,
            "metric": "wasserstein_per_feature",
            "feature_index": index,
            "feature": feature,
            "n_clean": "N/A",
            "n_attack": "N/A",
        }
        for index, feature in enumerate(feature_names)
    ]
    return {
        "wasserstein_pc1": [pc1],
        "wasserstein_perfeature": per_feature,
        "mmd_fullspace": [mmd],
        "js_pc1_appendix": [js],
    }


def compute_class_metrics(
    *,
    attack_class: str,
    clean: np.ndarray,
    populations: dict[str, np.ndarray | None],
    source_paths: dict[str, str],
    unavailable_reasons: dict[str, str],
    feature_names: list[str],
    mmd_n: int,
    n_boot: int,
    bootstrap_n: int,
    js_bins: int,
    seed: int,
) -> dict[str, pd.DataFrame]:
    clean_array = np.asarray(clean, dtype=np.float64)
    if clean_array.ndim != 2 or clean_array.shape[1] != len(feature_names):
        raise ValueError("clean population does not match feature_names")
    pc1 = fidelity.fit_clean_pc1(clean_array)
    clean_pc1 = pc1.transform(clean_array).ravel()
    clean_mmd = _subsample_rows(
        clean_array,
        n=mmd_n,
        seed=_stable_seed(seed, attack_class, "clean", "mmd"),
    )
    rows: dict[str, list[dict[str, Any]]] = {
        "wasserstein_pc1": [],
        "wasserstein_perfeature": [],
        "mmd_fullspace": [],
        "js_pc1_appendix": [],
    }

    for method in METHOD_ORDER:
        generated = populations.get(method)
        source_path = source_paths.get(method, "")
        if generated is None:
            missing = _unavailable_rows(
                attack_class=attack_class,
                method=method,
                reason=unavailable_reasons.get(method, "population unavailable"),
                source_path=source_path,
                feature_names=feature_names,
            )
            for key in rows:
                rows[key].extend(missing[key])
            continue

        generated_array = np.asarray(generated, dtype=np.float64)
        if (
            generated_array.ndim != 2
            or generated_array.shape[1] != len(feature_names)
        ):
            raise ValueError(
                f"{attack_class}/{method} has shape {generated_array.shape}; "
                f"expected (N, {len(feature_names)})"
            )
        generated_pc1 = pc1.transform(generated_array).ravel()
        w_seed = _stable_seed(seed, attack_class, method, "wasserstein")
        w_result = fidelity.bootstrap_wasserstein_1d(
            clean_pc1,
            generated_pc1,
            n_boot=n_boot,
            seed=w_seed,
            max_samples=bootstrap_n,
        )
        rows["wasserstein_pc1"].append(
            {
                "attack_class": attack_class,
                "method": method,
                "metric": "wasserstein_pc1",
                "value": w_result.value,
                "ci_low": w_result.ci_low,
                "ci_high": w_result.ci_high,
                "n_clean": clean_array.shape[0],
                "n_attack": generated_array.shape[0],
                "bootstrap_n_clean": min(
                    clean_array.shape[0], bootstrap_n
                ),
                "bootstrap_n_attack": min(
                    generated_array.shape[0], bootstrap_n
                ),
                "pc1_explained_variance_ratio": float(
                    pc1.explained_variance_ratio_[0]
                ),
                "status": "ok",
                "source_path": source_path,
                "scaling": SCALING_DESCRIPTION,
            }
        )

        per_feature = fidelity.wasserstein_per_feature(
            clean_array, generated_array
        )
        rows["wasserstein_perfeature"].extend(
            {
                "attack_class": attack_class,
                "method": method,
                "metric": "wasserstein_per_feature",
                "feature_index": index,
                "feature": feature,
                "value": float(per_feature[index]),
                "n_clean": clean_array.shape[0],
                "n_attack": generated_array.shape[0],
                "status": "ok",
                "source_path": source_path,
                "scaling": SCALING_DESCRIPTION,
            }
            for index, feature in enumerate(feature_names)
        )

        generated_mmd = _subsample_rows(
            generated_array,
            n=mmd_n,
            seed=_stable_seed(seed, attack_class, method, "mmd-sample"),
        )
        bandwidth = fidelity.median_heuristic_bandwidth(
            clean_mmd, generated_mmd
        )
        mmd_seed = _stable_seed(seed, attack_class, method, "mmd-bootstrap")
        mmd_result = fidelity.bootstrap_mmd2(
            clean_mmd,
            generated_mmd,
            bandwidth=bandwidth,
            n_boot=n_boot,
            seed=mmd_seed,
        )
        rows["mmd_fullspace"].append(
            {
                "attack_class": attack_class,
                "method": method,
                "metric": "mmd2_fullspace",
                "value": mmd_result.value,
                "ci_low": mmd_result.ci_low,
                "ci_high": mmd_result.ci_high,
                "bandwidth": bandwidth,
                "subsample_n": mmd_n,
                "seed": seed,
                "bootstrap_seed": mmd_seed,
                "status": "ok",
                "source_path": source_path,
                "scaling": SCALING_DESCRIPTION,
            }
        )

        rows["js_pc1_appendix"].append(
            {
                "attack_class": attack_class,
                "method": method,
                "metric": "js_pc1",
                "value": fidelity.js_divergence_1d(
                    clean_pc1, generated_pc1, bins=js_bins
                ),
                "bins": js_bins,
                "n_clean": clean_array.shape[0],
                "n_attack": generated_array.shape[0],
                "status": "ok",
                "source_path": source_path,
                "scaling": SCALING_DESCRIPTION,
            }
        )

    return {key: pd.DataFrame(value) for key, value in rows.items()}


def _format_number(value: Any) -> str:
    if value == "N/A" or pd.isna(value):
        return "N/A"
    return f"{float(value):.6f}"


def _format_ci_cell(row: pd.Series) -> str:
    if row.get("status") != "ok":
        return f"N/A ({row.get('status', '# TODO')})"
    return (
        f"{_format_number(row['value'])} "
        f"[{_format_number(row['ci_low'])}, "
        f"{_format_number(row['ci_high'])}]"
    )


def _format_value_cell(row: pd.Series) -> str:
    if row.get("status") != "ok":
        return f"N/A ({row.get('status', '# TODO')})"
    return _format_number(row["value"])


def _markdown_table(
    frame: pd.DataFrame,
    *,
    methods: list[str],
    with_ci: bool,
) -> str:
    lines = [
        "| Attack class | " + " | ".join(methods) + " |",
        "| --- | " + " | ".join(["---:" for _ in methods]) + " |",
    ]
    formatter = _format_ci_cell if with_ci else _format_value_cell
    for attack_class in CLASS_ORDER:
        cells = []
        for method in methods:
            selected = frame[
                (frame["attack_class"] == attack_class)
                & (frame["method"] == method)
            ]
            if selected.empty:
                cells.append("N/A (# TODO: row missing)")
            else:
                cells.append(formatter(selected.iloc[0]))
        lines.append(f"| {attack_class} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _mean_per_feature(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for attack_class in CLASS_ORDER:
        for method in METHOD_ORDER:
            subset = frame[
                (frame["attack_class"] == attack_class)
                & (frame["method"] == method)
            ]
            ok = subset[subset["status"] == "ok"]
            if ok.empty:
                status = (
                    subset.iloc[0]["status"]
                    if not subset.empty
                    else "# TODO: row missing"
                )
                rows.append(
                    {
                        "attack_class": attack_class,
                        "method": method,
                        "value": "N/A",
                        "status": status,
                    }
                )
            else:
                rows.append(
                    {
                        "attack_class": attack_class,
                        "method": method,
                        "value": pd.to_numeric(ok["value"]).mean(),
                        "status": "ok",
                    }
                )
    return pd.DataFrame(rows)


def render_markdown(
    frames: dict[str, pd.DataFrame], manifest: dict[str, Any]
) -> str:
    per_feature_mean = _mean_per_feature(
        frames["wasserstein_perfeature"]
    )
    bandwidths = pd.to_numeric(
        frames["mmd_fullspace"].loc[
            frames["mmd_fullspace"]["status"] == "ok", "bandwidth"
        ],
        errors="coerce",
    ).dropna()
    bandwidth_note = (
        f"Observed sigma range: {bandwidths.min():.6f} to "
        f"{bandwidths.max():.6f}. "
        if not bandwidths.empty
        else ""
    )
    manifest_summary = {
        "seed": manifest["seed"],
        "n_boot": manifest["n_boot"],
        "mmd_subsample_n": manifest["mmd_subsample_n"],
        "js_bins": manifest["js_bins"],
        "scaling": manifest["scaling"],
        "artifact_hash": manifest["artifact_hash"],
        "library_versions": manifest["library_versions"],
    }
    sections = [
        "# Wasserstein and MMD Fidelity Results",
        "",
        (
            "These tables compare each adversarial population with the "
            "class-matched clean-malicious test reference in the model's "
            "scaled feature space. They are the NetDiffuser-comparable "
            "Realism/Fidelity metrics: 1st-order Wasserstein distance and "
            "Gaussian-kernel MMD². For both metrics, lower values mean closer "
            "to clean malicious traffic. Jensen-Shannon divergence is retained "
            "only in the appendix because it depends on histogram binning."
        ),
        "",
        "## Table 1 — Wasserstein (PC1)",
        "",
        _markdown_table(
            frames["wasserstein_pc1"],
            methods=HEADLINE_METHODS,
            with_ci=True,
        ),
        "",
        (
            "Cells show point estimate [bootstrap 95% CI]. PCA is fitted once "
            "per class on the full clean-malicious reference in all 39 "
            "RobustScaler-transformed features, then reused for every method."
        ),
        "",
        "## Table 2 — Wasserstein (mean per-feature, 39 features)",
        "",
        _markdown_table(
            per_feature_mean,
            methods=HEADLINE_METHODS,
            with_ci=False,
        ),
        "",
        (
            "Each cell is the arithmetic mean of the 39 parameter-free "
            "one-dimensional empirical-CDF Wasserstein distances. The complete "
            "per-feature vectors are in `wasserstein_perfeature.csv`."
        ),
        "",
        "## Table 3 — MMD² (full feature space)",
        "",
        _markdown_table(
            frames["mmd_fullspace"],
            methods=HEADLINE_METHODS,
            with_ci=True,
        ),
        "",
        (
            "Cells show unbiased MMD² point estimate [bootstrap 95% CI]. "
            "The Gaussian RBF kernel is "
            "`exp(-||x-y||² / (2 sigma²))`; sigma is the pair-specific median "
            "of non-zero pooled pairwise Euclidean distances. Exact bandwidths "
            f"are recorded per row in `mmd_fullspace.csv` and the run manifest. "
            f"{bandwidth_note}"
            "An unbiased MMD² estimate may be slightly negative near equality."
        ),
        "",
        "## cInput Population Status",
        "",
        "### Supplementary Table S1 — cInput Wasserstein (PC1)",
        "",
        _markdown_table(
            frames["wasserstein_pc1"],
            methods=CINPUT_METHODS,
            with_ci=True,
        ),
        "",
        "### Supplementary Table S2 — cInput mean per-feature Wasserstein",
        "",
        _markdown_table(
            per_feature_mean,
            methods=CINPUT_METHODS,
            with_ci=False,
        ),
        "",
        "### Supplementary Table S3 — cInput MMD²",
        "",
        _markdown_table(
            frames["mmd_fullspace"],
            methods=CINPUT_METHODS,
            with_ci=True,
        ),
        "",
        (
            "The canonical cInput run is not silently omitted. Its outcome "
            "tables exist, but the run did not persist full adversarial feature "
            "vectors, so distributional distances are marked `N/A` / `# TODO` "
            "rather than estimated from the eight-sample exhibit."
        ),
        "",
        "## Appendix Table A1 — JS divergence (PC1)",
        "",
        _markdown_table(
            frames["js_pc1_appendix"],
            methods=HEADLINE_METHODS,
            with_ci=False,
        ),
        "",
        (
            f"Supplementary only. JS divergence uses {manifest['js_bins']} "
            "equal-width bins over the pooled PC1 range and is therefore "
            "bin-sensitive."
        ),
        "",
        "### Appendix Table A2 — cInput JS divergence (PC1)",
        "",
        _markdown_table(
            frames["js_pc1_appendix"],
            methods=CINPUT_METHODS,
            with_ci=False,
        ),
        "",
        "## Reading the tables",
        "",
        (
            "A low projection distance is not sufficient evidence of realistic "
            "traffic. Input-space PGD/CW can overlap the clean distribution on "
            "PC1 while still reaching 0% protocol-validity, so distributional "
            "closeness alone overstates realism. Read these tables alongside "
            "the validity-rate table in "
            "`results/attacks/thesis_bundle.multimetric.csv` and the "
            "Mahalanobis analysis in the `thesis_eval` realism suite."
        ),
        "",
        "## Run manifest",
        "",
        (
            "The full machine-readable manifest is "
            "`wasserstein_mmd_run_manifest.json`."
        ),
        "",
        "```json",
        json.dumps(manifest_summary, indent=2, sort_keys=True),
        "```",
        "",
    ]
    return "\n".join(sections)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _combined_artifact_hash(hashes: dict[str, str]) -> str:
    digest = hashlib.sha256()
    for path, file_hash in sorted(hashes.items()):
        digest.update(path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_hash.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def discover_sources(repo_root: Path) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    processed = root / "data" / "processed"
    sources: dict[str, Any] = {
        "clean_features": _require_file(
            processed / "X_test.npy", "scaled clean test features"
        ),
        "clean_labels": _require_file(
            processed / "y_test_cat.npy", "8-class test labels"
        ),
        "scaler": _require_file(
            processed / "scaler.pkl", "preprocessing scaler"
        ),
        "PGD": _require_file(
            root / "results" / "attacks" / "attack_8class_pgd_0.30.npz",
            "input-space PGD population",
        ),
        "CW": _require_file(
            root / "results" / "attacks" / "attack_8class_cw_0.npz",
            "input-space CW population",
        ),
        "Latent-PGD": {},
        "Latent-CW": {},
    }
    for attack_class in CLASS_ORDER:
        sources["Latent-PGD"][attack_class] = _require_file(
            root
            / "outputs"
            / "latent_attacks"
            / "phase2_20260602_023945_seed42"
            / f"phase2_latent_pgd_{attack_class}.npz",
            f"Latent-PGD population for {attack_class}",
        )
        sources["Latent-CW"][attack_class] = _require_file(
            root
            / "outputs"
            / "latent_attacks"
            / "phase3_20260602_023958_seed42"
            / f"phase3_latent_cw_{attack_class}.npz",
            f"Latent-CW population for {attack_class}",
        )
    sources["cInput"] = inspect_cinput_vectors(root)
    return sources


def print_discovered_sources(sources: dict[str, Any]) -> None:
    print("Discovered population paths:")
    print(f"  Clean features: {sources['clean_features']}")
    print(f"  Clean labels:   {sources['clean_labels']}")
    print(f"  PGD:            {sources['PGD']}")
    print(f"  CW:             {sources['CW']}")
    for method in ("Latent-PGD", "Latent-CW"):
        print(f"  {method}:")
        for attack_class in CLASS_ORDER:
            print(f"    {attack_class}: {sources[method][attack_class]}")
    cinput = sources["cInput"]
    print(
        f"  cInput CPGD/CCW: {cinput.path} "
        f"[{'available' if cinput.available else 'N/A'}]"
    )
    if cinput.reason:
        print(f"    reason: {cinput.reason}")


def print_module_layout() -> None:
    print("Module layout:")
    print("  src/thesis_eval/metrics/fidelity.py")
    print("    pure Wasserstein, PCA, JS, MMD2, and bootstrap estimators")
    print("  src/thesis_eval/fidelity_tables.py")
    print("    fail-loud discovery, computation, CSV/Markdown, manifest")
    print("  tests/thesis_eval/test_fidelity.py")
    print("    estimator, determinism, discovery, and report-contract tests")


def _load_npz_array(path: Path, key: str) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        if key not in data.files:
            raise KeyError(f"{path} is missing required key '{key}'")
        return np.asarray(data[key])


def _source_files(sources: dict[str, Any]) -> list[Path]:
    files = [
        sources["clean_features"],
        sources["clean_labels"],
        sources["scaler"],
        sources["PGD"],
        sources["CW"],
    ]
    for method in ("Latent-PGD", "Latent-CW"):
        files.extend(sources[method].values())
    cinput = sources["cInput"]
    if cinput.path.is_file():
        files.append(cinput.path)
    else:
        files.extend(
            path
            for path in (
                cinput.path / "config_snapshot.json",
                cinput.path / "per_sample_results.csv",
            )
            if path.is_file()
        )
    return files


def _build_manifest(
    *,
    sources: dict[str, Any],
    frames: dict[str, pd.DataFrame],
    seed: int,
    n_boot: int,
    bootstrap_n: int,
    mmd_n: int,
    js_bins: int,
) -> dict[str, Any]:
    root = Path.cwd().resolve()
    artifact_hashes = {
        str(path.resolve().relative_to(root)): _sha256_file(path)
        for path in _source_files(sources)
    }
    bandwidth_rows = frames["mmd_fullspace"][
        frames["mmd_fullspace"]["status"] == "ok"
    ][["attack_class", "method", "bandwidth"]]
    bandwidths = {
        f"{row.attack_class}/{row.method}": float(row.bandwidth)
        for row in bandwidth_rows.itertuples(index=False)
    }
    scaler = joblib.load(sources["scaler"])
    return {
        "seed": seed,
        "n_boot": n_boot,
        "bootstrap_max_n_per_population": bootstrap_n,
        "mmd_subsample_n": mmd_n,
        "js_bins": js_bins,
        "scaling": SCALING_DESCRIPTION,
        "scaler_class": (
            f"{type(scaler).__module__}.{type(scaler).__name__}"
        ),
        "pca": (
            "one component, fitted per class on the full clean-malicious "
            "reference in scaled 39-feature space"
        ),
        "wasserstein": "scipy.stats.wasserstein_distance, no binning",
        "mmd_estimator": "unbiased MMD^2",
        "mmd_kernel": "Gaussian RBF exp(-||x-y||^2 / (2*sigma^2))",
        "mmd_bandwidth_rule": (
            "pair-specific median of non-zero pooled pairwise Euclidean "
            "distances"
        ),
        "mmd_bandwidths": bandwidths,
        "cinput_status": {
            "available": sources["cInput"].available,
            "path": str(sources["cInput"].path),
            "reason": sources["cInput"].reason,
        },
        "artifact_hash": _combined_artifact_hash(artifact_hashes),
        "artifact_hashes": artifact_hashes,
        "library_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit-learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
    }


def compute_repository_metrics(
    *,
    repo_root: Path,
    seed: int,
    n_boot: int,
    bootstrap_n: int,
    mmd_n: int,
    js_bins: int,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    sources = discover_sources(repo_root)
    print_discovered_sources(sources)
    print_module_layout()

    clean_features = np.load(
        sources["clean_features"], mmap_mode="r", allow_pickle=False
    )
    clean_labels = np.load(
        sources["clean_labels"], mmap_mode="r", allow_pickle=False
    )
    with np.load(sources["PGD"], allow_pickle=False) as data:
        pgd_adv = np.asarray(data["X_adv"])
        pgd_labels = np.asarray(data["y_true"])
    with np.load(sources["CW"], allow_pickle=False) as data:
        cw_adv = np.asarray(data["X_adv"])
        cw_labels = np.asarray(data["y_true"])

    collected: dict[str, list[pd.DataFrame]] = {
        "wasserstein_pc1": [],
        "wasserstein_perfeature": [],
        "mmd_fullspace": [],
        "js_pc1_appendix": [],
    }
    cinput = sources["cInput"]
    if cinput.available:
        raise NotImplementedError(
            "A cInput vector artifact was found, but no canonical loader schema "
            f"is defined for {cinput.path}"
        )

    for attack_class in CLASS_ORDER:
        class_id = CLASS_TO_ID[attack_class]
        print(f"Computing {attack_class}...")
        clean = np.asarray(
            clean_features[np.asarray(clean_labels) == class_id],
            dtype=np.float64,
        )
        latent_pgd = _load_npz_array(
            sources["Latent-PGD"][attack_class], "x_adv"
        )
        latent_cw = _load_npz_array(
            sources["Latent-CW"][attack_class], "x_adv"
        )
        populations = {
            "PGD": pgd_adv[pgd_labels == class_id],
            "CW": cw_adv[cw_labels == class_id],
            "Latent-PGD": latent_pgd,
            "Latent-CW": latent_cw,
            "cInput CPGD": None,
            "cInput CCW": None,
        }
        source_paths = {
            "PGD": str(sources["PGD"]),
            "CW": str(sources["CW"]),
            "Latent-PGD": str(sources["Latent-PGD"][attack_class]),
            "Latent-CW": str(sources["Latent-CW"][attack_class]),
            "cInput CPGD": str(cinput.path),
            "cInput CCW": str(cinput.path),
        }
        unavailable_reasons = {
            "cInput CPGD": cinput.reason,
            "cInput CCW": cinput.reason,
        }
        frames = compute_class_metrics(
            attack_class=attack_class,
            clean=clean,
            populations=populations,
            source_paths=source_paths,
            unavailable_reasons=unavailable_reasons,
            feature_names=list(FEATURE_NAMES),
            mmd_n=mmd_n,
            n_boot=n_boot,
            bootstrap_n=bootstrap_n,
            js_bins=js_bins,
            seed=seed,
        )
        for key in collected:
            collected[key].append(frames[key])

    combined = {
        key: pd.concat(value, ignore_index=True)
        for key, value in collected.items()
    }
    manifest = _build_manifest(
        sources=sources,
        frames=combined,
        seed=seed,
        n_boot=n_boot,
        bootstrap_n=bootstrap_n,
        mmd_n=mmd_n,
        js_bins=js_bins,
    )
    return combined, manifest


def write_outputs(
    *,
    output_dir: Path,
    frames: dict[str, pd.DataFrame],
    manifest: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    filenames = {
        "wasserstein_pc1": "wasserstein_pc1.csv",
        "wasserstein_perfeature": "wasserstein_perfeature.csv",
        "mmd_fullspace": "mmd_fullspace.csv",
        "js_pc1_appendix": "js_pc1_appendix.csv",
    }
    for key, filename in filenames.items():
        frames[key].to_csv(output_dir / filename, index=False)
    (output_dir / "Wasserstein_MMD_results.md").write_text(
        render_markdown(frames, manifest),
        encoding="utf-8",
    )
    (output_dir / "wasserstein_mmd_run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute class-wise Wasserstein, MMD2, and JS fidelity tables "
            "from existing attack artifacts."
        )
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    parser.add_argument("--output-dir", type=Path, default=Path.cwd())
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--bootstrap-n", type=int, default=2000)
    parser.add_argument("--mmd-n", type=int, default=100)
    parser.add_argument("--js-bins", type=int, default=40)
    args = parser.parse_args()
    frames, manifest = compute_repository_metrics(
        repo_root=args.repo_root,
        seed=args.seed,
        n_boot=args.n_boot,
        bootstrap_n=args.bootstrap_n,
        mmd_n=args.mmd_n,
        js_bins=args.js_bins,
    )
    write_outputs(
        output_dir=args.output_dir,
        frames=frames,
        manifest=manifest,
    )
    print(f"Wrote fidelity outputs to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
