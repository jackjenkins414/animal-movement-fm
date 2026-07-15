# GenAI reference: I did the file logic, helper functions, and diagnostic framing; GenAI wrote main to orchestrate the 
# cached cadence runs, recompute same cadence persistence guards, and produce concise print/export summaries. Verified 
# by checking the 6h persistence re-match, confirming all averaging models used identical cached segment windows per 
# cadence, and comparing the saved CSV against the printed skill table.
"""
Cadence sweep: displacement-averaging skill relative to same cadence persistence.

Training-free, so it runs directly on the segment parquets cached by pass 1 (cadence_sweep.py); no resampling, no train 
birds. Introduces the RELATIVE metric the whole sweep turns on which is given by:

    skill = (persistence_error - model_error) / persistence_error   (per cadence, per percentile)

Positive skill = the model beats persistence at that cadence. Because skill is a ratio against the SAME-cadence persistence 
baseline, the step-length scaling that makes absolute km incomparable across cadences cancels out; so skill IS comparable 
across cadences, and absolute km is not.

"""

from __future__ import annotations

import os
os.environ["OMP_NUM_THREADS"] = "1" 

from pathlib import Path

import numpy as np
import pandas as pd

from animal_movement_fm.data.split import make_split, window_counts_per_individual
from animal_movement_fm.data.benchmark import evaluate_on_split
from animal_movement_fm.models.persistence import Persistence
from animal_movement_fm.models.averaging import DisplacementAveraging


REPO = Path(__file__).resolve().parents[2]
SEGMENTS_6H = REPO / "data" / "processed" / "stork_segments.parquet"  
RESULTS = REPO / "results"

CADENCE_GRID = ["24h", "12h", "6h", "3h", "1h", "30min"]

FULL_COVERAGE = {"6h", "3h", "1h", "30min"}

K_GRID = [1, 3, 10]
CONTEXT_LENGTH = 24


def err_km(results: pd.DataFrame) -> np.ndarray:
    return results["error_m"].to_numpy() / 1000.0


def ladder(err: np.ndarray) -> dict:
    return {
        "median_km": float(np.median(err)),
        "p90_km": float(np.percentile(err, 90)),
        "p99_km": float(np.percentile(err, 99)),
    }


def skill(base: dict, model: dict) -> dict:
    """Fractional error reduction vs same-cadence persistence, per percentile."""
    return {
        f"skill_{p}": (base[f"{p}_km"] - model[f"{p}_km"]) / base[f"{p}_km"]
        for p in ("median", "p90", "p99")
    }


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)

    counts = window_counts_per_individual(SEGMENTS_6H)
    split = make_split(counts, strata=None, min_windows=100, seed=42)
    test_ids = split.test
    print(f"canonical split: {len(test_ids)} test birds (reused across cadences)\n")

    models = [Persistence()] + [DisplacementAveraging(k=k) for k in K_GRID]

    rows = []
    for label in CADENCE_GRID:
        seg_path = RESULTS / f"segments_sweep_{label}.parquet"
        if not seg_path.exists():
            print(f"[{label:>5}]  cached segments missing ({seg_path.name}) -- "
                  f"rerun cadence_sweep.py; skipping")
            continue

        # Evaluate every model on identical windows at this cadence.
        errs = {}
        for m in models:
            res = evaluate_on_split(m, seg_path, test_ids,
                                    context_length=CONTEXT_LENGTH, show_progress=False)
            errs[m.name] = ladder(err_km(res))

        base = errs["persistence"]
        full = label in FULL_COVERAGE
        flag = "" if full else "  [subset: migration-biased, not headline-comparable]"

        # Coherence guard: 6h persistence must re-match the canonical headline.
        note = ""
        if label == "6h":
            note = ("  <- 6h persistence median re-match check "
                    f"(want ~0.599): {base['median_km']:.3f}")
        print(f"[{label:>5}] persistence median={base['median_km']:.3f} km  "
              f"p99={base['p99_km']:.1f} km{note}{flag}")

        for k in K_GRID:
            mname = f"averaging_k{k}"
            s = skill(base, errs[mname])
            print(f"          {mname:<14} "
                  f"skill_median={s['skill_median']:+.1%}  "
                  f"skill_p90={s['skill_p90']:+.1%}  "
                  f"skill_p99={s['skill_p99']:+.1%}")
            rows.append({
                "cadence": label, "full_coverage": full, "model": mname,
                "persist_median_km": base["median_km"],
                "persist_p99_km": base["p99_km"],
                "model_median_km": errs[mname]["median_km"],
                "model_p99_km": errs[mname]["p99_km"],
                **s,
            })
        print()

    table = pd.DataFrame(rows)
    out_path = RESULTS / "cadence_sweep_averaging.csv"
    table.to_csv(out_path, index=False)

    print("Constant-velocity (k=1) skill over persistence, full-coverage cadences:")
    cv = table[(table["model"] == "averaging_k1") & table["full_coverage"]]
    order = {"30min": 0, "1h": 1, "3h": 2, "6h": 3}
    cv = cv.assign(_o=cv["cadence"].map(order)).sort_values("_o")
    for _, r in cv.iterrows():
        print(f"  {r['cadence']:>5}  median {r['skill_median']:+.1%}   "
              f"p90 {r['skill_p90']:+.1%}   p99 {r['skill_p99']:+.1%}")
    print(f"\nsummary -> {out_path}")


if __name__ == "__main__":
    main()