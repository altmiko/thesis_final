"""Central path + global-constant module for the CICIoT2023 preprocessing pipeline.

Single source of truth for filesystem locations, referenced by
``downsampling_strategy.md`` §7.1–7.3. No script under ``src/preprocessing``
may hardcode an absolute path; every path is derived from :data:`REPO_ROOT`
so the pipeline is portable across machines (the thesis was previously
pinned to ``D:/thesis_final`` — see AGENTS.md).

Layout::

    <REPO_ROOT>/
      config/paths.py                 <- this file
      data/processed/
        ciciot2023_labeled_full.parquet          (46,775,660 rows, 309 shards)
        ciciot2023_labeled_full_manifest.json
        <pipeline outputs: X_*.npy, y_*.npy, scaler.pkl, ...>
      docs/
"""
from __future__ import annotations

from pathlib import Path

# ── Roots ───────────────────────────────────────────────────────────────────
REPO_ROOT: Path = Path(__file__).resolve().parents[1]

DATA_DIR: Path = REPO_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"
DOCS_DIR: Path = REPO_ROOT / "docs"
LOGS_DIR: Path = REPO_ROOT / "logs"
CONFIG_DIR: Path = REPO_ROOT / "config"

# ── Full labelled source (input to the split → scale → sample pipeline) ──────
LABELED_PARQUET: Path = PROCESSED_DIR / "ciciot2023_labeled_full.parquet"
LABELED_MANIFEST: Path = PROCESSED_DIR / "ciciot2023_labeled_full_manifest.json"

# ── Diagnostics / evidence output (§6, §7.4) ─────────────────────────────────
DIAGNOSTICS_DIR: Path = PROCESSED_DIR / "downsampling_diagnostics"

# ── Pipeline run manifest (§7.3) ─────────────────────────────────────────────
RUN_MANIFEST: Path = PROCESSED_DIR / "run_manifest.json"

# ── Global reproducibility seed (AGENTS.md: global seed 42) ──────────────────
SEED: int = 42


def processed(name: str) -> Path:
    """Return ``PROCESSED_DIR / name``; the canonical way to name an artefact."""
    return PROCESSED_DIR / name


def ensure_dirs() -> None:
    """Create the output directories the pipeline writes into."""
    for d in (PROCESSED_DIR, DIAGNOSTICS_DIR, DOCS_DIR, LOGS_DIR):
        d.mkdir(parents=True, exist_ok=True)
