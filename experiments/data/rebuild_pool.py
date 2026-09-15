# GenAI reference: I did as much of this file as I could for now and then used GenAI to write main() and run() from 
# my spec; the stage order, A7 gate before writing the manifest, excluding Tier 2 role species, and the candidate/
# domain table are mine. I pasted the relevant existing files so it used the relevant parts too. I reviewed the code, 
# added the docstrings/comments, and checked the rebuilt gate counts against my earlier eligibility results.
"""
Rebuild Tier 3 / Tier 4 candidates after a fixes re-extract.

For each candidate this runs the existing segment, split and regime profile scripts in order, checking
our A7 entry gate before writing a new split manifest. No pipeline logic is reimplemented here.

Examples being uv run python experiments/data/rebuild_pool.py --candidates

These candidate manifests have not been used for evaluation, so they may be rebuilt with `--force`.
Tier 2 role species are deliberately excluded.
"""

from __future__ import annotations

import os

os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

from animal_movement_fm.data.resample import CONTEXT_LENGTH, min_segment_bins
from animal_movement_fm.data.split import window_counts_per_individual

sys.path.insert(0, str(Path(__file__).resolve().parent))

from species_eligibility import (  # keep the A7 gate definition in one place # type: ignore
    EVALUATION_FLOOR,
    clears_gate1,
)

REPO = Path(__file__).resolve().parents[2]
PROCESSED = REPO / "data" / "processed"
SPLITS = REPO / "splits"
DATA_DIR = REPO / "experiments" / "data"

# candidate slug -> registered domain; recall we use "terrestrial", "marine", "flying"; and it labels LOCOMOTION, not 
# taxonomy and we apply same principle to qc.py applies to the speed ceilings
CANDIDATES: dict[str, str] = {
    "lynx_rufus": "terrestrial",
    "ovis_dalli": "terrestrial",
    "canis_latrans": "terrestrial",
    "connochaetes_taurinus": "terrestrial",
    "bison_bison": "terrestrial",
    "leptonychotes_weddellii": "marine",
    "odobenus_rosmarus": "marine",
    "arctocephalus_gazella": "marine",
    "lobodon_carcinophaga": "marine",
    "pygoscelis_adeliae": "marine",
    "eudyptes_chrysolophus": "marine",
    "aptenodytes_patagonicus": "marine",
    "diomedea_exulans": "flying",
    "thalassarche_chrysostoma": "flying",
    "thalassoica_antarctica": "flying",
    # movebank birds added for the first Tier 3 build
    "cathartes_aura": "flying",
    "gyps_himalayensis": "flying",
    "milvus_milvus": "flying",
    "pernis_apivorus": "flying",
    "falco_naumanni": "flying",
    "corvus_corone": "flying",
    "mycteria_americana": "flying",
    "haematopus_ostralegus": "flying",
    "branta_canadensis": "flying",
    "anser_albifrons": "flying",
    "anser_caerulescens": "flying",
}


# suppress the large profile/distance tables when rebuilding many species; failures are never suppressed
_NOISE_FROM = ("accumulated regime profiles:", "pairwise distance in standardised descriptor space",
               "Pick candidates FAR from")
# resume output for the A4/A10 decision block and final file summaries
_RESUME_AT = ("applying the frozen decision", "REGISTERED DECISION", "decision ->",
              "screening descriptor standardisation", "  this file moves with", "profiles ->")


def run(script: str, *args: str, quiet: bool = False) -> None:
    """Run one single-species pipeline script and stop on failure."""
    command = [sys.executable, str(DATA_DIR / script), *args]
    print(f"    $ {' '.join(command[1:])}", flush=True)
    if not quiet:
        result = subprocess.run(command, cwd=REPO)
        if result.returncode != 0:
            raise SystemExit(f"FAIL: {script} exited {result.returncode} for {args}")
        return

    result = subprocess.run(command, cwd=REPO, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr, file=sys.stderr)
        raise SystemExit(f"FAIL: {script} exited {result.returncode} for {args}")
    suppressing = False
    for line in result.stdout.splitlines():
        if line.strip().startswith(_NOISE_FROM):
            suppressing = True
            continue
        if line.strip().startswith(_RESUME_AT):
            suppressing = False
        if suppressing:
            continue
        print(f"    {line}" if line.strip() else "")


def gate(segments: Path) -> tuple[bool, int, int, int]:
    """Apply A7 gate 1 to a freshly built segment file."""
    counts = window_counts_per_individual(segments, context_length=CONTEXT_LENGTH)["n_windows"].to_numpy()
    above = int((counts >= EVALUATION_FLOOR).sum())
    any_window = int((counts >= 1).sum())
    return clears_gate1(above, any_window), above, any_window, int(counts.sum())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", action="store_true", help="rebuild the whole candidate pool")
    parser.add_argument("--species", nargs="*", default=[], help="slugs to rebuild")
    parser.add_argument("--stage", choices=("segments", "split", "profile", "all"), default="all")
    parser.add_argument("--decide", action="store_true",
                        help="after profiling, re-run the A4 threshold decision and refreeze the A10 "
                             "descriptor standardisation over the updated table")
    args = parser.parse_args()

    slugs = list(args.species)
    if args.candidates:
        slugs += [s for s in CANDIDATES if s not in slugs]
    if not slugs:
        raise SystemExit("give --candidates or --species")

    unknown = [s for s in slugs if s not in CANDIDATES]
    if unknown:
        raise SystemExit(f"FAIL: no registered domain for {', '.join(unknown)}")

    # frame the profile table once so the QC rebuild can be diffed afterwards
    profiles = REPO / "results" / "regime_profiles.csv"
    snapshot = REPO / "results" / "regime_profiles_pre_qc.csv"
    if args.stage in ("profile", "all") and profiles.exists() and not snapshot.exists():
        snapshot.write_bytes(profiles.read_bytes())
        print(f"snapshot -> {snapshot}")

    blocked, rebuilt = [], []
    for slug in slugs:
        fixes = PROCESSED / f"{slug}_fixes.parquet"
        segments = PROCESSED / f"{slug}_segments.parquet"
        manifest = SPLITS / f"{slug}_split_seed42.json"
        print(f"\n=== {slug} ({CANDIDATES[slug]}) ===", flush=True)

        if args.stage in ("segments", "all"):
            if not fixes.exists():
                raise SystemExit(f"FAIL: missing {fixes}; run extract_species_fixes.py first")
            run("build_species_segments.py", "--input", str(fixes), "--output", str(segments),
                "--min-segment-bins", str(min_segment_bins(CONTEXT_LENGTH)))

        if args.stage in ("split", "profile", "all"):
            passes, above, any_window, total = gate(segments)
            print(f"    gate: n(>={EVALUATION_FLOOR})={above}  n(>=1)={any_window}  "
                  f"windows={total:,}  -> {'PASS' if passes else 'FAIL'}", flush=True)
            if not passes:
                blocked.append((slug, above, any_window))
                print("    skipping split and profile; this species no longer clears A7 gate 1", flush=True)
                continue

        if args.stage in ("split", "all"):
            run("make_split_manifest.py", "--segments", str(segments), "--name", slug, "--force")

        if args.stage in ("profile", "all"):
            # regime_profile.py updates this species' row in the shared profile table
            run("regime_profile.py", "--segments", str(segments), "--name", slug,
                "--domain", CANDIDATES[slug], "--split", str(manifest), "--split-part", "train",
                quiet=True)
        rebuilt.append(slug)

    if args.decide and rebuilt:
        # re run the A4 decision and refreeze the A10 descriptor standardisation after rebuilding
        last = rebuilt[-1]
        print(f"\n=== A4 decision and A10 refreeze (run through {last}) ===", flush=True)
        run("regime_profile.py",
            "--segments", str(PROCESSED / f"{last}_segments.parquet"), "--name", last,
            "--domain", CANDIDATES[last], "--split", str(SPLITS / f"{last}_split_seed42.json"),
            "--split-part", "train", "--decide-threshold", quiet=True)

    print(f"\nrebuilt {len(rebuilt)} of {len(slugs)} species")
    if blocked:
        print("\nno longer clear A7 gate 1 after QC:")
        for slug, above, any_window in blocked:
            print(f"   {slug:<26} n(>=25)={above} (need 20)   n(>=1)={any_window} (need 40)")
        print("These drop out of the Tier 3/4 pool. That is a QC consequence and belongs in the amendment.")


if __name__ == "__main__":
    main()
