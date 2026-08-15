# GenAI reference: it completed the main method, and part of the sql query from my skeleton file (what I had done thus far). 
# outlined that fixes are binned to 6h, splits into consecutive runs per individual, and turned window nto window counts under 
# both context lengths etc. The guards re derive numbers on every run and I validated the correctness of this. 
"""
Audit Movebank species pools for taxon label contamination by study, apply the inclusion policy, and compare the corrected 
and uncorrected eligibility counts. Window counts use the historical audit rule for comparability with earlier results.
"""

from __future__ import annotations

import os

os.environ["OMP_NUM_THREADS"] = "1"

import argparse
from datetime import date
from pathlib import Path

import duckdb
import pandas as pd

from animal_movement_fm.data.inclusion import apply_inclusion_policy, policy_report

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results"

BIN_SECONDS = 21600

# expected audit counts, keyed by (species, pool, context)
GUARDS: dict[tuple[str, str, int], dict[str, int]] = {
    # original audit counts.
    ("Canis lupus", "uncorrected", 24): {"ge1": 172, "ge25": 147, "ge100": 108},
    ("Cervus elaphus", "uncorrected", 24): {"ge1": 324, "ge25": 322, "ge100": 320},
    ("Rangifer tarandus", "uncorrected", 24): {"ge1": 173, "ge25": 161, "ge100": 143},
    ("Larus fuscus", "uncorrected", 24): {"ge1": 427, "ge25": 391, "ge100": 341},
    ("Ciconia ciconia", "uncorrected", 24): {"ge1": 900, "ge25": 886, "ge100": 790},
    # Cleaned wolf count.
    ("Canis lupus", "clean", 24): {"ge100": 91},
    # Cleaned C=12 counts.
    ("Canis lupus", "clean", 12): {"ge1": 119, "ge25": 110},
    ("Larus fuscus", "clean", 12): {"ge1": 352, "ge25": 330},
}

# species included in the standard audit sweep
SWEEP = (
    "Ciconia ciconia",
    "Larus fuscus",
    "Larus argentatus",
    "Anser albifrons",
    "Anser caerulescens",
    "Cathartes aura",
    "Haematopus ostralegus",
    "Cervus elaphus",
    "Rangifer tarandus",
    "Canis lupus",
    "Bos taurus",
)

WINDOW_SQL = f"""
WITH inds AS (
    SELECT id, taxon_canonical_name AS taxon
    FROM read_parquet($individuals)
    WHERE taxon_canonical_name IN (SELECT unnest($species))
),
bins AS (
    SELECT DISTINCT l.study_id, l.individual_id, i.taxon,
           CAST(floor(epoch(l.timestamp) / {BIN_SECONDS}) AS BIGINT) AS bin
    FROM read_parquet($locations) l
    JOIN inds i ON l.individual_id = i.id
),
runs AS (
    SELECT study_id, individual_id, taxon, count(*) AS length
    FROM (
        SELECT study_id, individual_id, taxon, bin,
               bin - row_number() OVER (PARTITION BY individual_id ORDER BY bin) AS run_id
        FROM bins
    )
    GROUP BY study_id, individual_id, taxon, run_id
)
SELECT study_id, individual_id, taxon,
       sum(greatest(length - 24, 0)) AS windows_c24,
       sum(greatest(length - 12, 0)) AS windows_c12,
       sum(length) AS bins
FROM runs
GROUP BY study_id, individual_id, taxon
"""


def gate_counts(frame: pd.DataFrame, column: str) -> dict[str, object]:
    """Count individuals clearing each window floor and the A7 F=25 gate."""
    windows = frame[column].to_numpy()
    ge1 = int((windows >= 1).sum())
    ge25 = int((windows >= 25).sum())
    return {
        "individuals": len(frame),
        "ge1": ge1,
        "ge25": ge25,
        "ge50": int((windows >= 50).sum()),
        "ge100": int((windows >= 100).sum()),
        # A7 gate; >=20 evaluable and >=40 trainable individuals
        "gate1_F25": bool(ge25 >= 20 and ge1 >= 40),
    }


def by_study(frame: pd.DataFrame, species: str) -> pd.DataFrame:
    """Summarise one species pool by study."""
    pool = frame[frame["taxon"] == species]
    if pool.empty:
        return pd.DataFrame()
    return (
        pool.groupby("study_id")
        .agg(
            individuals=("individual_id", "size"),
            ge25_c12=("windows_c12", lambda w: int((w >= 25).sum())),
            bins=("bins", "sum"),
        )
        .reset_index()
        .sort_values("individuals", ascending=False)
    )


def check_guards(species: str, pool: str, context: int, counts: dict[str, object]) -> list[str]:
    """Return any mismatches against registered audit counts."""
    expected = GUARDS.get((species, pool, context))
    if not expected:
        return []
    return [
        f"{species} [{pool}, C={context}] {key} = {int(counts[key])}, expected {want}"
        for key, want in expected.items()
        if int(counts[key]) != want
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path.home() / "moveformer-data",
        help="directory holding locations_gps.parquet and individuals.parquet",
    )
    parser.add_argument("--species", nargs="*", help="species to audit")
    parser.add_argument(
        "--sweep", action="store_true", help=f"audit all {len(SWEEP)} headline candidates"
    )
    parser.add_argument(
        "--gate-sweep",
        type=int,
        metavar="N",
        help="audit every species with at least N individuals; use 40 for the A7 gate-1 minimum",
    )
    parser.add_argument(
        "--studies", action="store_true", help="also print the per-study breakdown of each pool"
    )
    args = parser.parse_args()

    locations = args.data_root / "locations_gps.parquet"
    individuals = args.data_root / "individuals.parquet"
    for path in (locations, individuals):
        if not path.exists():
            raise SystemExit(f"FAIL: missing {path}")

    con = duckdb.connect()
    con.execute("PRAGMA threads=8; PRAGMA memory_limit='10GB'")

    if args.gate_sweep:
        species = con.execute(
            """
            SELECT i.taxon_canonical_name AS taxon
            FROM read_parquet($locations) l
            JOIN read_parquet($individuals) i ON l.individual_id = i.id
            WHERE i.taxon_canonical_name IS NOT NULL
            GROUP BY 1
            HAVING count(DISTINCT l.individual_id) >= $floor
            ORDER BY count(DISTINCT l.individual_id) DESC
            """,
            {
                "locations": str(locations),
                "individuals": str(individuals),
                "floor": args.gate_sweep,
            },
        ).df()["taxon"].tolist()
        print(f"gate sweep: {len(species)} species with >= {args.gate_sweep} individuals")
    elif args.sweep:
        species = list(SWEEP)
    else:
        species = args.species or ["Canis lupus"]
    per_individual = con.execute(
        WINDOW_SQL,
        {"individuals": str(individuals), "locations": str(locations), "species": species},
    ).df()

    report = policy_report(per_individual, taxon_col="taxon")
    if not report.empty:
        print("\n=== studies affected by the inclusion policy ===")
        print(report.to_string(index=False))

    corrected = apply_inclusion_policy(per_individual, taxon_col="taxon")

    rows, failures = [], []
    for name in species:
        for pool, frame in (("uncorrected", per_individual), ("clean", corrected)):
            subset = frame[frame["taxon"] == name]
            if subset.empty:
                continue
            for context, column in ((24, "windows_c24"), (12, "windows_c12")):
                counts = gate_counts(subset, column)
                rows.append({"species": name, "pool": pool, "context": context, **counts})
                failures.extend(check_guards(name, pool, context, counts))
        if args.studies:
            studies = by_study(per_individual, name)
            if not studies.empty:
                print(f"\n=== {name}: individuals by study ===")
                print(studies.to_string(index=False))

    summary = pd.DataFrame(rows)
    print("\n=== pool summary ===")
    print(summary.to_string(index=False))

    # Show only pools changed by the inclusion policy.
    changed = (
        summary.pivot_table(index=["species", "context"], columns="pool", values="individuals")
        .reset_index()
        .query("uncorrected != clean")
        if {"uncorrected", "clean"} <= set(summary["pool"])
        else pd.DataFrame()
    )
    print("\n=== pools changed by the inclusion policy ===")
    print("none" if changed.empty else changed.to_string(index=False))

    if failures:
        raise SystemExit("FAIL: guard mismatch\n  " + "\n  ".join(failures))
    guarded = sum(1 for key in GUARDS if key[0] in species)
    print(f"\nguards passed: {guarded} checked")

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"pool_audit_{date.today().isoformat()}.csv"
    summary.to_csv(out, index=False)
    print(f"wrote -> {out}")


if __name__ == "__main__":
    main()
