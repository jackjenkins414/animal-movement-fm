# GenAI reference: I laid out the module docstring, function names, intended logic, and then used GenAI to write 
# the implementation to save time. I verified the generated script by checking that C=24 targets were fully present 
# in the C=12 paired run, confirming regime labels were inherited correctly for the paired comparison. It was fine. 
"""
C=24 vs C=12 context length ablation on the time_season cell.

A shorter context yields more windows per run, which matters most for our fragmented tracks (1.04x for stork, 
up to 2.62x for Weddell seal). This checks what that extra data costs in skill.

Every C=24 target is also a C=12 target, so the comparison is paired: c12_paired scores C=12 on the C=24 targets, 
carrying the C=24 regime labels, and is the decision number. c24_native and c12_native are each arm on its own 
eligible set. Regimes are tagged per arm on its native frame because tag_regime is hysteretic.

    uv run python experiments/reporting/context_ablation.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments" / "reporting"))

from aggregate_multiseed import add_regime  # noqa: E402  same tagging as the headline results

RESULTS = REPO / "results" / "multiseed"
OUT = REPO / "results" / "context_ablation"

CELL = "time_season"
ARCHS = ["lstm", "transformer"]
SEEDS = [0, 1, 2, 3, 4]
REGIMES = ["foraging", "moving", "launch", "settle"]
ARMS = ["c24_native", "c12_paired", "c12_native"]
KEY = ["segment_id", "timestamp"]

ACTIONS = {
    "preserved": "register context 12 as an amendment; rebuild Tier-2 eligibility at C=12",
    "degraded": "retain context 24; report the ablation as a null result",
}


def load(arch: str, seed: int, context: int) -> pd.DataFrame:
    run_id = f"{arch}_{CELL}_s{seed}" + ("" if context == 24 else f"_c{context}")
    path = RESULTS / f"preds_{run_id}.parquet"
    if not path.exists():
        raise SystemExit(f"FAIL: missing {path} - has the subjob for {run_id} finished?")
    d = pd.read_parquet(path)
    return d[d["split"] == "test"].copy()


def stats(errors: np.ndarray) -> dict:
    if errors.size == 0:
        return {"n": 0, "median_km": np.nan, "p90_km": np.nan, "p99_km": np.nan}
    return {
        "n": int(errors.size),
        "median_km": float(np.median(errors)),
        "p90_km": float(np.percentile(errors, 90)),
        "p99_km": float(np.percentile(errors, 99)),
    }


def summarise(tagged: pd.DataFrame) -> list[dict]:
    rows = [{"regime": r, **stats(tagged.loc[tagged["regime"] == r, "error_km"].to_numpy())}
            for r in REGIMES]
    rows.append({"regime": "all",
                 **stats(tagged.loc[tagged["regime"] != "unknown", "error_km"].to_numpy())})
    return rows


def pair(c24: pd.DataFrame, c12: pd.DataFrame, arch: str, seed: int) -> pd.DataFrame:
    """C=12 errors on the C=24 targets, relabelled with the C=24 regimes."""
    k24 = pd.MultiIndex.from_frame(c24[KEY])
    k12 = pd.MultiIndex.from_frame(c12[KEY])
    missing = int((~k24.isin(k12)).sum())
    if missing:
        raise SystemExit(
            f"FAIL: {missing} C=24 targets absent from the C=12 run ({arch}, seed {seed}). "
            f"The arms should share a manifest - check both used splits/stork_split_seed42.json."
        )
    return c12.drop(columns=["regime"]).merge(c24[KEY + ["regime"]], on=KEY, how="inner")


def collect() -> pd.DataFrame:
    """One row per arch, seed, arm and regime."""
    rows = []
    for arch in ARCHS:
        for seed in SEEDS:
            c24 = add_regime(load(arch, seed, 24))
            c12 = add_regime(load(arch, seed, 12))
            paired = pair(c24, c12, arch, seed)

            for arm, frame in zip(ARMS, (c24, paired, c12)):
                for r in summarise(frame):
                    rows.append({"arch": arch, "seed": seed, "arm": arm, **r})

            extra = len(c12) - len(paired)
            share = 100 * extra / max(len(c12), 1)
            print(f"{arch:<12} s{seed}: c24={len(c24):,}  c12={len(c12):,}  "
                  f"paired={len(paired):,}  extra={extra:,} ({share:.1f}%)")
    return pd.DataFrame(rows)


def aggregate(per_run: pd.DataFrame) -> pd.DataFrame:
    """Mean and spread across seeds."""
    return (
        per_run
        .groupby(["arch", "arm", "regime"])
        .agg(
            median_km=("median_km", "mean"),
            median_sd=("median_km", "std"),
            p90_km=("p90_km", "mean"),
            p90_sd=("p90_km", "std"),
            p99_km=("p99_km", "mean"),
            p99_sd=("p99_km", "std"),
            n=("n", "mean"),
        )
        .reset_index()
        .round(3)
    )


def report(agg: pd.DataFrame) -> dict:
    """Print the moving-regime p99 comparison, return whether each arch held up."""
    moving = agg[agg.regime == "moving"].set_index(["arch", "arm"])
    verdicts = {}

    print("\n=== moving-regime p99, mean +/- sd over 5 seeds ===")
    for arch in ARCHS:
        try:
            base, paired, native = (moving.loc[(arch, arm)] for arm in ARMS)
        except KeyError:
            continue

        delta = 100 * (paired.p99_km - base.p99_km) / base.p99_km
        pooled_sd = float(np.hypot(base.p99_sd, paired.p99_sd))
        preserved = abs(paired.p99_km - base.p99_km) <= 2 * pooled_sd
        verdicts[arch] = bool(preserved)

        print(f"  {arch:<12} C=24 paired  {base.p99_km:7.1f} +/-{base.p99_sd:4.1f} km")
        print(f"  {'':<12} C=12 paired  {paired.p99_km:7.1f} +/-{paired.p99_sd:4.1f} km   "
              f"delta {delta:+5.1f}%   2sd {2 * pooled_sd:.1f} km   "
              f"-> {'PRESERVED' if preserved else 'DEGRADED'}")
        print(f"  {'':<12} C=12 native  {native.p99_km:7.1f} km on {int(native.n):,} windows "
              f"(vs {int(base.n):,} at C=24)")
    return verdicts


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    per_run = collect()
    per_run.to_csv(OUT / "context_ablation_by_run.csv", index=False)

    agg = aggregate(per_run)
    agg.to_csv(OUT / "context_ablation_summary.csv", index=False)

    verdicts = report(agg)
    verdict = "preserved" if verdicts and all(verdicts.values()) else "degraded"
    print(f"\nVERDICT: {verdict.upper()}  ->  {ACTIONS[verdict]}")

    decision = {
        "metric": "moving-regime p99, paired targets, test split, 5 training seeds",
        "rule_set_before_reading":
            "preserved iff |C12_paired - C24| <= 2 * pooled seed sd, BOTH architectures",
        "per_arch_preserved": verdicts,
        "verdict": verdict,
    }
    (OUT / "decision.json").write_text(json.dumps(decision, indent=2))
    print(f"wrote -> {OUT}")


if __name__ == "__main__":
    main()