# GenAI reference: this file was drafted by me, completed by GenAI, see the lines where it is declared. 
"""
xtract per species fixes from Movebank or other per species corpora, apply the inclusion policy and structural QC, and stream 
retained fixes to the common ingest schema. Invalid coordinates and duplicate losers are removed; speed issues are flagged and 
retained for downstream handling.
"""

from __future__ import annotations

import os

os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import json
import re
import tempfile
from pathlib import Path

import duckdb
import pandas as pd

from animal_movement_fm.data.inclusion import (
    apply_inclusion_policy,
    is_excluded_deployment,
    policy_report,
)
from animal_movement_fm.data.qc import SPEED_CEILINGS, flag_speed, structural_qc

REPO = Path(__file__).resolve().parents[2]
PROCESSED = REPO / "data" / "processed"

# Movebank Tier 2 species/candidates; marine corpora arent extracted here
TIER2_AND_CANDIDATES = (
    "Ciconia ciconia",
    "Larus fuscus",
    "Cervus elaphus",
    "Canis lupus",
    "Ovis dalli",
    "Canis latrans",
    "Lynx rufus",
    "Connochaetes taurinus",
    "Bison bison",
)

# GenAI refined this 
SPECIES_SQL = """
SELECT l.individual_id, l.deployment_id, l.study_id, l.timestamp,
       l.location_lat AS lat, l.location_long AS lon,
       i.taxon_canonical_name AS taxon
FROM read_parquet($locations) l
JOIN read_parquet($individuals) i ON l.individual_id = i.id
WHERE i.taxon_canonical_name = $species
"""


def slug(species: str) -> str:
    """Convert a taxon name to a predictable filename slug."""
    return re.sub(r"[^a-z0-9]+", "_", species.lower()).strip("_")


# optional Argos fields used to rank conflicting duplicate fixes
QUALITY_COL = "location_quality"
ERROR_COLS = ("semi_major_m", "error_radius_m")


def _rank_columns(frame: pd.DataFrame) -> tuple[str | None, str | None]:
    """Return available quality/error columns, or `None`."""
    quality = QUALITY_COL if QUALITY_COL in frame.columns else None
    error = next((c for c in ERROR_COLS if c in frame.columns), None)
    return quality, error


# GenAI reference: assisted implementation of batch QC/accounting; policy and expected behaviour were defined separately
def _clean_batch(
    raw: pd.DataFrame,
    species: str,
    stats: dict,
    deployments: pd.DataFrame | None = None,
    drop_class_z: bool = True,
    quality_col: str | None = None,
    error_col: str | None = None,
) -> pd.DataFrame:
    """Apply inclusion policy and QC to one batch, updating `stats`."""
    if "study_id" in raw.columns and "taxon" in raw.columns:
        kept = apply_inclusion_policy(raw, taxon_col="taxon")
        kept = kept[kept["taxon"] == species]  # A retag moves an animal out of this pool.
    else:
        # Non-Movebank corpora use deployment-level exclusions instead.
        kept = raw
        if "deployment_id" in kept.columns:
            drop_dep = kept["deployment_id"].map(is_excluded_deployment).to_numpy(dtype=bool)
            stats["fixes_excluded_deployment"] = (
                stats.get("fixes_excluded_deployment", 0) + int(drop_dep.sum())
            )
            kept = kept.loc[~drop_dep]
    stats["fixes_after_inclusion"] = stats.get("fixes_after_inclusion", 0) + len(kept)
    if kept.empty:
        return kept

    # Argos class Z is explicitly invalid, unlike lower-quality A/B/0 fixes.
    if drop_class_z and "location_quality" in kept.columns:
        is_z = kept["location_quality"].astype(str).str.upper().eq("Z")
        stats["fixes_class_z"] = stats.get("fixes_class_z", 0) + int(is_z.sum())
        kept = kept.loc[~is_z]
        if kept.empty:
            return kept

    # Detect ranking columns per batch so Argos conflicts can use quality information.
    auto_quality, auto_error = _rank_columns(kept)
    flagged, qc = structural_qc(
        kept,
        species=species,
        deployments=deployments,
        quality_col=quality_col if quality_col is not None else auto_quality,
        error_col=error_col if error_col is not None else auto_error,
    )
    if species in SPEED_CEILINGS:
        # Keep speed counts in the QC audit as well as the batch stats.
        flagged = flag_speed(flagged, SPEED_CEILINGS[species], report=qc)
        stats["speed_over_ceiling"] = stats.get("speed_over_ceiling", 0) + int(
            flagged["qc_over_ceiling"].sum()
        )
        stats["speed_spikes"] = stats.get("speed_spikes", 0) + int(flagged["qc_spike"].sum())

    drop = (
        flagged["qc_invalid_coord"] | flagged["qc_invalid_time"] | flagged["qc_duplicate"]
    )
    # Counts accumulate across batches; settings do not.
    settings = {"notes", "ceiling_multiplier"}
    for key, value in qc.as_dict().items():
        if key in settings:
            stats.setdefault(key, value)
        elif isinstance(value, (int, float)):
            stats[key] = stats.get(key, 0) + value
        else:
            stats[key] = value
    stats["fixes_dropped_by_qc"] = stats.get("fixes_dropped_by_qc", 0) + int(drop.sum())
    return flagged.loc[~drop]


def _stable_dtypes(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep batch schemas stable before streaming to one parquet file."""
    out = frame.copy()
    for column in out.columns:
        kind = out[column].dtype.kind
        if kind not in "biufM":  # not bool/int/unsigned/float/datetime
            out[column] = out[column].astype("string")
    return out


def extract(
    con: duckdb.DuckDBPyConnection,
    species: str,
    paths: dict[str, str],
    out_path: Path,
    batch: int = 60,
    source_parquet: Path | None = None,
    deployments: pd.DataFrame | None = None,
    drop_class_z: bool = True,
    quality_col: str | None = None,
    error_col: str | None = None,
) -> tuple[bool, dict]:
    """Extract one species in individual batches, apply inclusion/QC, and record audit counts."""
    stats: dict[str, object] = {"species": species}

    if source_parquet is not None:
        # per species non Movebank input skips the full locations scan
        ids = con.execute(
            f"SELECT DISTINCT individual_id FROM read_parquet('{source_parquet}') ORDER BY 1"
        ).df()["individual_id"].tolist()
        stats["individuals_raw"] = len(ids)
        if not ids:
            return False, stats
        return _process(con, str(source_parquet), ids, species, batch, stats, out_path,
                        deployments, drop_class_z, quality_col, error_col)

    # materialise once; otherwise every batch re scans the full locations parquet
    with tempfile.TemporaryDirectory() as tmp:
        scratch = str(Path(tmp) / "species.parquet")
        con.execute(
            f"COPY ({SPECIES_SQL}) TO '{scratch}' (FORMAT PARQUET)",
            {**paths, "species": species},
        )
        ids = con.execute(
            f"SELECT DISTINCT individual_id FROM '{scratch}' ORDER BY individual_id"
        ).df()["individual_id"].tolist()
        stats["individuals_raw"] = len(ids)
        if not ids:
            return False, stats
        return _process(con, scratch, ids, species, batch, stats, out_path,
                        deployments, drop_class_z, quality_col, error_col)


# GenAI reference: completed the implementation of the streamed parquet batch writer
def _process(con, scratch, ids, species, batch, stats, out_path: Path,
             deployments=None, drop_class_z: bool = True,
             quality_col: str | None = None, error_col: str | None = None):
    """Clean batches and stream them directly to parquet to avoid whole-species memory use."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    writer, raw_total, kept_total, seen = None, 0, 0, set()
    try:
        for start in range(0, len(ids), batch):
            chunk = ids[start : start + batch]
            raw = con.execute(
                f"SELECT * FROM '{scratch}' WHERE individual_id IN (SELECT unnest($ids))",
                {"ids": chunk},
            ).df()
            raw_total += len(raw)
            cleaned = _clean_batch(raw, species, stats, deployments, drop_class_z,
                                   quality_col, error_col)
            if not cleaned.empty:
                cleaned = _stable_dtypes(cleaned)
                table = pa.Table.from_pandas(cleaned, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(out_path, table.schema)
                writer.write_table(table)
                kept_total += len(cleaned)
                seen.update(cleaned["individual_id"].unique().tolist())
            del raw, cleaned
    finally:
        if writer is not None:
            writer.close()

    stats["fixes_raw"] = raw_total
    stats["fixes_out"] = kept_total
    stats["individuals_out"] = len(seen)
    return kept_total > 0, stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path.home() / "moveformer-data")
    parser.add_argument("--species", nargs="*", help="taxon names to extract")
    parser.add_argument("--tier2", action="store_true",
                        help=f"extract the {len(TIER2_AND_CANDIDATES)} Movebank Tier-2 species and candidates")
    parser.add_argument("--out-dir", type=Path, default=PROCESSED)
    parser.add_argument("--source", type=Path, default=None,
                        help="a per-species fixes parquet from a non-Movebank corpus "
                             "(RAATD, the NES archive); skips the locations_gps scan")
    parser.add_argument("--deployments", type=Path, default=None,
                        help="deployment metadata csv with deploy_on/off timestamps")
    parser.add_argument("--quality-col", default=None,
                        help="column holding the Argos location class, used to rank conflicting "
                             f"duplicates; detected as {QUALITY_COL!r} when present")
    parser.add_argument("--error-col", default=None,
                        help="column holding the error radius or ellipse axis, the tie-break "
                             f"within one class; detected from {ERROR_COLS} when present")
    parser.add_argument("--keep-class-z", action="store_true",
                        help="retain Argos class Z, the service's own invalid-location flag")
    parser.add_argument("--batch", type=int, default=60,
                        help="individuals per batch; lower it if memory is tight")
    args = parser.parse_args()

    species = list(TIER2_AND_CANDIDATES) if args.tier2 else (args.species or [])
    if not species:
        raise SystemExit("give --species or --tier2")

    paths = {
        "locations": str(args.data_root / "locations_gps.parquet"),
        "individuals": str(args.data_root / "individuals.parquet"),
    }
    if args.source is None:
        for p in paths.values():
            if not Path(p).exists():
                raise SystemExit(f"FAIL: missing {p}")

    con = duckdb.connect()
    con.execute("PRAGMA threads=8; PRAGMA memory_limit='10GB'")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for name in species:
        path = args.out_dir / f"{slug(name)}_fixes.parquet"
        deployments = None
        if args.deployments is not None:
            deployments = pd.read_csv(args.deployments, low_memory=False)
        wrote, stats = extract(
            con, name, paths, path, batch=args.batch,
            source_parquet=args.source, deployments=deployments,
            drop_class_z=not args.keep_class_z,
            quality_col=args.quality_col, error_col=args.error_col,
        )
        if not wrote:
            print(f"{name:<24} no fixes after policy; nothing written")
            summaries.append(stats)
            continue
        stats["output"] = str(path)
        summaries.append(stats)
        print(
            f"{name:<24} {stats['fixes_raw']:>10,} raw -> {stats['fixes_out']:>10,} kept  "
            f"({stats['individuals_out']:>4} individuals, "
            f"{stats['fixes_dropped_by_qc']:,} dropped by QC)"
        )

    # merge summaries because species may be extracted in separate runs
    summary_path = args.out_dir / "species_fixes_summary.json"
    existing = {}
    if summary_path.exists():
        try:
            existing = {s["species"]: s for s in json.loads(summary_path.read_text())}
        except (json.JSONDecodeError, KeyError, TypeError):
            print(f"warning: could not parse {summary_path}; starting a fresh summary")
    existing.update({s["species"]: s for s in summaries})
    merged = sorted(existing.values(), key=lambda s: s["species"])
    summary_path.write_text(json.dumps(merged, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"\nwrote -> {summary_path} ({len(merged)} species)")


if __name__ == "__main__":
    main()
