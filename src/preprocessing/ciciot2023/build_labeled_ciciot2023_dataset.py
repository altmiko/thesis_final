"""Build one labelled parquet from the raw CICIoT2023 CSV folders.

Scope — labelling stage of the pipeline described in `downsampling_strategy.md` §4:

    RAW CSV → *this step* → clean → group-disjoint split → scaler fit → sample

The raw distribution ships 34 class folders under `data/raw/CICIoT2023_CSV_DOWNLOADED/`,
each with one or more `part-*.pcap.csv` shards whose header exactly matches
the 39-feature Schema A. **The shards do not carry a Label column.** This
script attaches:

    Label                — 34-class label (uppercase, matches CATEGORY_MAP keys)
    category             — 8-class category (Benign, BruteForce, DDoS, DoS,
                            Mirai, Recon, Spoofing, Web)
    source_csv_filename  — relative path of the source shard; required later
                            for group-disjoint splitting per doc §4.1
    source_folder        — original folder name, kept for auditability

Cleaning done here is deliberately minimal — NaN/inf drop only. Percentile
clipping and integer/binary rounding stay in `pipeline.py` so cleaning is
never done twice with inconsistent thresholds.

Outputs (default):

    data/processed/ciciot2023_labeled_full.parquet          (ZSTD)
    data/processed/ciciot2023_labeled_full_manifest.json

The parquet is streamed row-group by row-group so the ~46M-row full dataset
fits in memory well below 8 GB peak.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Schema constants come from the archived feature_groups module — it is a
# leaf module (no cross-package imports) and remains the single source of
# truth for the 39-column FEATURE_NAMES and the 34→8 CATEGORY_MAP that
# every downstream stage indexes into.
from src.preprocessing.feature_groups import (  # noqa: E402
    CATEGORY_MAP,
    EXPECTED_COLUMNS,
    FEATURE_NAMES,
    LABEL_COLUMN,
)

# Folder-name → 34-class Label mapping. Inlined rather than imported from
# ``old/build_ciciot2023_dataset.py`` because that module pulls in the whole
# archived preprocessing/ML stack (sklearn, pyarrow, sampler, etc.) which
# has its own broken import chain post-reorganisation. All 34 keys were
# copied verbatim from the archived source (git blame trail preserved
# through the ``old/`` directory) and are cross-checked against the disk
# layout in ``docs/data/ciciot.md`` §3.
FOLDER_TO_LABEL: dict[str, str] = {
    "Backdoor_Malware": "BACKDOOR_MALWARE",
    "Benign_Final": "BENIGN",
    "BrowserHijacking": "BROWSERHIJACKING",
    "CommandInjection": "COMMANDINJECTION",
    "DDoS-ACK_Fragmentation": "DDOS-ACK_FRAGMENTATION",
    "DDoS-HTTP_Flood": "DDOS-HTTP_FLOOD",
    "DDoS-ICMP_Flood": "DDOS-ICMP_FLOOD",
    "DDoS-ICMP_Fragmentation": "DDOS-ICMP_FRAGMENTATION",
    "DDoS-PSHACK_FLOOD": "DDOS-PSHACK_FLOOD",
    "DDoS-RSTFINFLOOD": "DDOS-RSTFINFLOOD",
    "DDoS-SlowLoris": "DDOS-SLOWLORIS",
    "DDoS-SynonymousIP_Flood": "DDOS-SYNONYMOUSIP_FLOOD",
    "DDoS-SYN_Flood": "DDOS-SYN_FLOOD",
    "DDoS-TCP_Flood": "DDOS-TCP_FLOOD",
    "DDoS-UDP_Flood": "DDOS-UDP_FLOOD",
    "DDoS-UDP_Fragmentation": "DDOS-UDP_FRAGMENTATION",
    "DictionaryBruteForce": "DICTIONARYBRUTEFORCE",
    "DNS_Spoofing": "DNS_SPOOFING",
    "DoS-HTTP_Flood": "DOS-HTTP_FLOOD",
    "DoS-SYN_Flood": "DOS-SYN_FLOOD",
    "DoS-TCP_Flood": "DOS-TCP_FLOOD",
    "DoS-UDP_Flood": "DOS-UDP_FLOOD",
    "Mirai-greeth_flood": "MIRAI-GREETH_FLOOD",
    "Mirai-greip_flood": "MIRAI-GREIP_FLOOD",
    "Mirai-udpplain": "MIRAI-UDPPLAIN",
    "MITM-ArpSpoofing": "MITM-ARPSPOOFING",
    "Recon-HostDiscovery": "RECON-HOSTDISCOVERY",
    "Recon-OSScan": "RECON-OSSCAN",
    "Recon-PingSweep": "RECON-PINGSWEEP",
    "Recon-PortScan": "RECON-PORTSCAN",
    "SqlInjection": "SQLINJECTION",
    "Uploading_Attack": "UPLOADING_ATTACK",
    "VulnerabilityScan": "VULNERABILITYSCAN",
    "XSS": "XSS",
}
assert len(FOLDER_TO_LABEL) == 34, "FOLDER_TO_LABEL must cover all 34 CICIoT2023 classes"


DEFAULT_RAW_DIR = _REPO_ROOT / "data" / "raw" / "CICIoT2023_CSV_DOWNLOADED"
DEFAULT_OUTPUT = _REPO_ROOT / "data" / "processed" / "ciciot2023_labeled_full.parquet"
DEFAULT_MANIFEST = (
    _REPO_ROOT / "data" / "processed" / "ciciot2023_labeled_full_manifest.json"
)
DEFAULT_CHUNK_SIZE = 500_000  # rows per read; keeps peak RAM ≲ 500MB · 39 · 4B ≈ 80MB


@dataclass
class RawShard:
    """One CSV shard resolved to its class label."""

    path: Path
    source: str  # relative to raw root, used as source_csv_filename downstream
    folder: str
    label: str
    category: str


@dataclass
class Counter:
    """Row tally by folder/label/category, used to build the manifest."""

    total_rows: int = 0
    kept_rows: int = 0
    dropped_nan_inf: int = 0
    per_folder: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    per_label: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    per_category: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    per_file: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    n_files_seen: int = 0


def resolve_shards(raw_dir: Path) -> list[RawShard]:
    """Walk every class folder under ``raw_dir`` and resolve its label.

    Every folder present on disk must appear in FOLDER_TO_LABEL, and the
    resulting 34-class label must appear in CATEGORY_MAP. A silent skip
    of an unknown folder here would mean an entire attack class disappears
    from the labelled dataset with no error — refuse instead.
    """
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw directory does not exist: {raw_dir}")

    shards: list[RawShard] = []
    unknown_folders: list[str] = []
    for entry in sorted(raw_dir.iterdir()):
        if not entry.is_dir():
            continue  # skip README_CSV.pdf and any other loose files
        folder = entry.name
        if folder not in FOLDER_TO_LABEL:
            unknown_folders.append(folder)
            continue
        label = FOLDER_TO_LABEL[folder]
        if label not in CATEGORY_MAP:
            raise ValueError(
                f"Label {label!r} from folder {folder!r} is not in CATEGORY_MAP"
            )
        category = CATEGORY_MAP[label]
        for csv_path in sorted(entry.rglob("*.csv")):
            shards.append(
                RawShard(
                    path=csv_path,
                    source=str(csv_path.relative_to(raw_dir)).replace("\\", "/"),
                    folder=folder,
                    label=label,
                    category=category,
                )
            )

    if unknown_folders:
        raise ValueError(
            "Unknown class folders under raw dir "
            f"(no FOLDER_TO_LABEL entry): {unknown_folders}"
        )
    if not shards:
        raise FileNotFoundError(f"No CSV files found under {raw_dir}")
    return shards


def iter_shard_chunks(
    shard: RawShard,
    chunk_size: int,
    limit_rows: int | None,
) -> Iterable[pd.DataFrame]:
    """Yield validated feature chunks from one CSV shard.

    Raises if the header is missing any of the 39 FEATURE_NAMES so a
    silently-relabelled CSV cannot slip through.
    """
    reader = pd.read_csv(shard.path, chunksize=chunk_size)
    emitted = 0
    for chunk in reader:
        missing = set(FEATURE_NAMES) - set(chunk.columns)
        if missing:
            raise ValueError(
                f"{shard.source} is missing expected columns: {sorted(missing)}"
            )
        # Reject any pre-existing Label column so we never merge two label sources.
        if LABEL_COLUMN in chunk.columns:
            raise ValueError(
                f"{shard.source} already contains a '{LABEL_COLUMN}' column; "
                "this script only labels un-labelled CICIoT2023 shards."
            )
        # Reorder to canonical schema. Extra columns (if any) are silently
        # dropped here; header verification above catches missing ones.
        chunk = chunk[FEATURE_NAMES]
        if limit_rows is not None:
            remaining = limit_rows - emitted
            if remaining <= 0:
                break
            if len(chunk) > remaining:
                chunk = chunk.iloc[:remaining]
        emitted += len(chunk)
        yield chunk


def process_shard(
    shard: RawShard,
    chunk_size: int,
    limit_rows: int | None,
    writer_state: dict,
    output: Path,
    counter: Counter,
    dry_run: bool,
) -> None:
    """Read + label + append one shard to the growing parquet."""
    for chunk in iter_shard_chunks(shard, chunk_size, limit_rows):
        raw_len = len(chunk)
        counter.total_rows += raw_len

        # Minimum cleaning: NaN/inf → drop. Anything else (percentile clip,
        # integer/binary rounding) is deferred to pipeline.py so cleaning
        # is never applied with two different thresholds.
        chunk = chunk.replace([np.inf, -np.inf], np.nan).dropna()
        dropped = raw_len - len(chunk)
        counter.dropped_nan_inf += dropped
        if chunk.empty:
            continue

        # Cast features to float32 to match downstream expectations
        # (pipeline.py stores X_*.npy as float32).
        chunk = chunk.astype({feature: "float32" for feature in FEATURE_NAMES})

        # Attach labelling metadata. All four columns are categorical/string;
        # pyarrow will encode Label + category as dictionary types thanks to
        # `.astype("category")` producing pd.Categorical.
        chunk[LABEL_COLUMN] = shard.label
        chunk["category"] = shard.category
        chunk[LABEL_COLUMN] = chunk[LABEL_COLUMN].astype("category")
        chunk["category"] = chunk["category"].astype("category")
        chunk["source_csv_filename"] = shard.source
        chunk["source_folder"] = shard.folder

        kept = len(chunk)
        counter.kept_rows += kept
        counter.per_folder[shard.folder] += kept
        counter.per_label[shard.label] += kept
        counter.per_category[shard.category] += kept
        counter.per_file[shard.source] += kept

        if dry_run:
            continue

        # Assert final column order — belt and braces so a future rename
        # of one of the metadata columns cannot silently swap positions.
        expected_out = FEATURE_NAMES + [
            LABEL_COLUMN,
            "category",
            "source_csv_filename",
            "source_folder",
        ]
        if list(chunk.columns) != expected_out:
            raise RuntimeError(
                f"Column order drift on {shard.source}: got {list(chunk.columns)}"
            )

        table = pa.Table.from_pandas(chunk, preserve_index=False)
        writer: pq.ParquetWriter | None = writer_state.get("writer")
        if writer is None:
            output.parent.mkdir(parents=True, exist_ok=True)
            writer = pq.ParquetWriter(output, table.schema, compression="zstd")
            writer_state["writer"] = writer
            writer_state["schema"] = table.schema
        else:
            # PyArrow appends must share schema exactly; a mismatch here means
            # a shard produced a different dtype (e.g., all-zero categorical
            # with unseen category value). Cast to the initial schema.
            if not table.schema.equals(writer_state["schema"]):
                table = table.cast(writer_state["schema"], safe=False)
        writer.write_table(table)


def build(
    raw_dir: Path,
    output: Path,
    manifest_path: Path,
    chunk_size: int,
    limit_per_file: int | None,
    dry_run: bool,
) -> Counter:
    """Top-level driver: enumerate shards, stream, write parquet + manifest."""
    shards = resolve_shards(raw_dir)
    print(
        f"[discover] {len(shards)} CSV shards across "
        f"{len({s.folder for s in shards})} class folders"
    )

    counter = Counter()
    writer_state: dict = {"writer": None}
    try:
        for idx, shard in enumerate(shards, start=1):
            print(
                f"[read] {idx:>3}/{len(shards)} "
                f"{shard.source}  → {shard.label} ({shard.category})"
            )
            process_shard(
                shard=shard,
                chunk_size=chunk_size,
                limit_rows=limit_per_file,
                writer_state=writer_state,
                output=output,
                counter=counter,
                dry_run=dry_run,
            )
            counter.n_files_seen += 1
    finally:
        writer = writer_state.get("writer")
        if writer is not None:
            writer.close()

    # Post-write cross-checks: kept rows must equal what we accounted for
    # in the per-label breakdown. If they diverge, a chunk was written but
    # not counted (or vice versa) — surface the drift immediately.
    per_label_sum = sum(counter.per_label.values())
    per_category_sum = sum(counter.per_category.values())
    per_folder_sum = sum(counter.per_folder.values())
    if not (per_label_sum == per_category_sum == per_folder_sum == counter.kept_rows):
        raise RuntimeError(
            "Row accounting drift: "
            f"kept={counter.kept_rows} label_sum={per_label_sum} "
            f"category_sum={per_category_sum} folder_sum={per_folder_sum}"
        )

    manifest = {
        "raw_dir": str(raw_dir),
        "output_parquet": str(output),
        "dry_run": dry_run,
        "chunk_size": chunk_size,
        "limit_per_file": limit_per_file,
        "n_files": counter.n_files_seen,
        "n_folders": len({s.folder for s in shards}),
        "total_rows_read": counter.total_rows,
        "kept_rows": counter.kept_rows,
        "dropped_nan_inf": counter.dropped_nan_inf,
        "feature_names": FEATURE_NAMES,
        "n_features": len(FEATURE_NAMES),
        "label_column": LABEL_COLUMN,
        "expected_columns": EXPECTED_COLUMNS,
        "per_folder_rows": dict(sorted(counter.per_folder.items())),
        "per_label_rows": dict(sorted(counter.per_label.items())),
        "per_category_rows": dict(sorted(counter.per_category.items())),
        "per_file_rows": dict(sorted(counter.per_file.items())),
        "folder_to_label": FOLDER_TO_LABEL,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"[manifest] wrote {manifest_path}")

    return counter


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help=f"Root of CICIoT2023 raw CSVs (default: {DEFAULT_RAW_DIR})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output labelled parquet (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help=f"Output manifest JSON (default: {DEFAULT_MANIFEST})",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=f"Rows per read chunk (default: {DEFAULT_CHUNK_SIZE:,})",
    )
    parser.add_argument(
        "--limit-per-file",
        type=int,
        default=None,
        help="Cap rows read per CSV (smoke test); default: unlimited",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Count rows and write manifest but do not emit the parquet",
    )
    args = parser.parse_args()

    counter = build(
        raw_dir=args.raw_dir,
        output=args.output,
        manifest_path=args.manifest,
        chunk_size=args.chunk_size,
        limit_per_file=args.limit_per_file,
        dry_run=args.dry_run,
    )

    print("")
    print("=" * 68)
    print("     CICIoT2023 LABELLING COMPLETE")
    print("=" * 68)
    print(f"Raw dir:          {args.raw_dir}")
    print(f"Output parquet:   {args.output}{'  [DRY RUN]' if args.dry_run else ''}")
    print(f"Manifest:         {args.manifest}")
    print(f"Files processed:  {counter.n_files_seen}")
    print(f"Rows read:        {counter.total_rows:,}")
    print(f"Rows kept:        {counter.kept_rows:,}")
    print(f"NaN/inf dropped:  {counter.dropped_nan_inf:,}")
    print("")
    print(f"Per-category rows (8-class):")
    for cat, n in sorted(counter.per_category.items(), key=lambda kv: -kv[1]):
        pct = 100.0 * n / counter.kept_rows if counter.kept_rows else 0.0
        print(f"  {cat:<12} {n:>15,}  {pct:>6.2f}%")
    print("")
    print(f"Per-label rows (top 10, 34-class):")
    top_labels = sorted(counter.per_label.items(), key=lambda kv: -kv[1])[:10]
    for label, n in top_labels:
        pct = 100.0 * n / counter.kept_rows if counter.kept_rows else 0.0
        print(f"  {label:<26} {n:>15,}  {pct:>6.2f}%")
    print("=" * 68)


if __name__ == "__main__":
    main()
