"""
Measure how far each model's prediction moves away from persistence.

Our sealed evaluation reports error distributions, but similar errors can could be from different predictions; so this script 
instead measures the geodesic distance between each model's prediction and persistence's prediction, which is just the last 
observed fix. That distance is therefore how far the model chose to move in prediction space.

It only reads prediction files already written by the sealed evaluation. No test data are re-opened, no registered quantity is 
changed etc. 
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
SEALED = REPO / "results" / "tier2_sealed"
PREDICTIONS = SEALED / "predictions"
VERDICT = SEALED / "tier2_verdict.json"

ARMS = ("V", "Z", "B", "S", "Za")
SEEDS = (0, 1, 2, 3, 4)
UNTRAINED = {"V"}  # one shared file, so there is no seed suffix
EARTH_RADIUS_M = 6_371_008.8


def geodesic_m(lon1, lat1, lon2, lat2) -> np.ndarray:
    """Return great circle distance in metres using the same spherical Earth assumption as the harness."""
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi, dlambda = phi2 - phi1, np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def load(species: str, arm: str, seed: int | None) -> pd.DataFrame | None:
    name = f"{species}__{arm}.parquet" if seed is None else f"{species}__{arm}_s{seed}.parquet"
    path = PREDICTIONS / name
    if not path.exists():
        return None
    frame = pd.read_parquet(path)
    return frame.sort_values(["individual_id", "timestamp"]).reset_index(drop=True)


# GenAI reference: I asked GenAI to write this main reporting workflow from my analysis specification, including the checks
# aggregation, printing and JSON output
def main() -> None:
    if not VERDICT.exists():
        raise SystemExit(f"FAIL: {VERDICT} not found; run tier2_sealed_eval.py --report first")
    verdict = json.loads(VERDICT.read_text())

    rows_out = {}
    for species, record in verdict["species"].items():
        boundary = float(record["regime_boundary_m"])
        persistence = load(species, "P", None)
        if persistence is None:
            print(f"{species}: no persistence predictions, skipped")
            continue

        # Persistence predicts the last fix, so its error is the animal's actual step.
        step = persistence["error_m"].to_numpy(float)
        low = step <= boundary

        print(
            f"\n=== {species}   {len(persistence):,} windows"
            f"   travel {int((~low).sum()):,}   low-displacement {int(low.sum()):,}"
            f"   (boundary {boundary:,.1f} m)"
        )
        header = (
            f"{'arm':4} {'median |pred-P|':>16} {'% of step':>10} {'<1% of step':>12}"
            f" {'low-disp m':>12} {'low-disp %':>11}"
        )
        print(header)
        print("-" * len(header))

        for arm in ARMS:
            seeds = (None,) if arm in UNTRAINED else SEEDS
            per_seed = []
            for seed in seeds:
                frame = load(species, arm, seed)
                if frame is None:
                    continue
                if not (
                    frame["individual_id"].to_numpy() == persistence["individual_id"].to_numpy()
                ).all():
                    raise SystemExit(f"FAIL: {species} {arm} seed {seed} is not row-aligned with P")
                moved = geodesic_m(
                    persistence["pred_lon"].to_numpy(float),
                    persistence["pred_lat"].to_numpy(float),
                    frame["pred_lon"].to_numpy(float),
                    frame["pred_lat"].to_numpy(float),
                )
                share = moved / np.maximum(step, 1e-9)
                per_seed.append(
                    (
                        float(np.median(moved)),
                        float(np.median(share)),
                        float((share < 0.01).mean()),
                        float(np.median(moved[low])) if low.any() else float("nan"),
                        float(np.median(share[low])) if low.any() else float("nan"),
                    )
                )
            if not per_seed:
                continue

            # Median across seeds, matching A3's aggregation of the headline metrics.
            median = np.median(np.asarray(per_seed), axis=0)
            print(
                f"{arm:4} {median[0]:16,.1f} {median[1] * 100:9.1f}% {median[2] * 100:11.1f}%"
                f" {median[3]:12,.1f} {median[4] * 100:10.1f}%"
            )
            rows_out.setdefault(species, {})[arm] = {
                "median_offset_from_persistence_m": median[0],
                "median_offset_as_share_of_step": median[1],
                "share_of_windows_within_1pc_of_persistence": median[2],
                "low_displacement_median_offset_m": median[3],
                "low_displacement_offset_as_share_of_step": median[4],
            }

    out = SEALED / "tier2_persistence_distance.json"
    out.write_text(json.dumps({"species": rows_out}, indent=2) + "\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
