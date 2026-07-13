# GenAI reference: All these steps took forever, so when I got to the final step I asked GenAI to do it for me. 
# Specifically, I gave it a skeleton of the function names and what they should do (intended behaviour), and GenAI
# implented the regime tagging, bucket summaries, main method and plotting successfully. I added docstrings and 
# verified it by checking regime counts against the underlying true_step_km distribution, manually inspecting 
# segment level launch/settle labels to confirm switches only occurred within segment_id boundaries, comparing 
# bucket medians against direct filtered pandas calculations. 
"""
Step 6: regime decomposition of error (final obs step...).

Splits every prediction into foraging / moving / transition using the true step the bird made (true_step_km, already
in each dump) and the empirical 60 m foraging/moving cut. Reports error per bucket per model, testing the thesis's central 
claim and a note about "transitions are hopeless":

We defined: 
  foraging (< 60 m step)     : persistence near optimal; learned models can't win
  moving   (>= 60 m step)    : learned models earn their keep
  transition (regime flips)  : the onset event; least predictable for everyone

A transition is a step whose regime differs from the PREVIOUS step's regime (forage->move launch, or move->forage settle). 
It overrides the plain foraging/moving label. These are the switching moments that John flagged with me in our prev meeting.
"""

from __future__ import annotations

import os
os.environ["OMP_NUM_THREADS"] = "1"

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "experiments" / "results"

MODELS = ["persistence", "avg_k1", "avg_k3", "lstm", "transformer", "lstm_time", "transformer_time"]
REGIME_CUT_KM = 0.06        # Nominal foraging/moving trough (§5)
HYST_LOW_KM = 0.04          # Clearly foraging below this
HYST_HIGH_KM = 0.10         # Clearly moving above this --- 0.04-0.10 is the grey band


def load(name: str) -> pd.DataFrame:
    return (pd.read_parquet(RESULTS / f"preds_{name}.parquet")
            .sort_values(["segment_id", "timestamp"])
            .reset_index(drop=True))


def _hysteresis_regime(step_km: np.ndarray) -> np.ndarray:
    """
    Foraging/moving label per step with hysteresis, computed within one segment.

    A step is 'moving' above HYST_HIGH_KM, 'foraging' below HYST_LOW_KM; a step in the grey band inherits the previous 
    step's clear regime. This stops threshold jitter around the 60 m cut from manufacturing spurious switches. NaN steps 
    (first-in-segment) stay 'unknown'.
    """
    out = np.empty(len(step_km), dtype=object)
    prev = None
    for i, s in enumerate(step_km):
        if not np.isfinite(s):
            out[i] = "unknown"
            continue
        if s >= HYST_HIGH_KM:
            cur = "moving"
        elif s < HYST_LOW_KM:
            cur = "foraging"
        else:
            cur = prev if prev is not None else ("moving" if s >= REGIME_CUT_KM else "foraging")
        out[i] = cur
        prev = cur
    return out


def tag_regime(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a 4way regime label: foraging / moving / launch / settle.

    Base foraging/moving is assigned with hysteresis. A regime CHANGE from the previous step is then split by direction:
      launch  foraging -> moving  (onset; expected hard")
      settle  moving  -> foraging (settling down; expected easy?)
    """
    df = df.copy()
    base_parts = []
    for _, seg in df.groupby("segment_id", sort=False):
        base_parts.append(pd.Series(
            _hysteresis_regime(seg["true_step_km"].to_numpy()), index=seg.index))
    df["base_regime"] = pd.concat(base_parts).reindex(df.index)

    prev = df.groupby("segment_id")["base_regime"].shift(1)
    launch = (prev == "foraging") & (df["base_regime"] == "moving")
    settle = (prev == "moving") & (df["base_regime"] == "foraging")

    regime = df["base_regime"].to_numpy().copy()
    regime[launch.to_numpy()] = "launch"
    regime[settle.to_numpy()] = "settle"
    df["regime"] = regime
    return df


def bucket_stats(df: pd.DataFrame, model: str) -> list[dict]:
    rows = []
    for reg in ["foraging", "moving", "launch", "settle"]:
        e = df.loc[df["regime"] == reg, "error_km"].to_numpy()
        if e.size == 0:
            continue
        rows.append({
            "model": model, "regime": reg, "n": e.size,
            "median_km": np.median(e), "mean_km": np.mean(e),
            "p90_km": np.percentile(e, 90), "p99_km": np.percentile(e, 99),
        })
    return rows


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)

    # only decompose models that actually have a preds parquet, so step6 runs before every cell of the
    # time 2x2 is trained (missing models are skipped with a note, not an error)
    present = [m for m in MODELS if (RESULTS / f"preds_{m}.parquet").exists()]
    missing = [m for m in MODELS if m not in present]
    if missing:
        print(f"skipping models with no preds parquet: {missing}\n")

    all_rows = []
    regime_counts = None
    for m in present:
        df = tag_regime(load(m))
        if regime_counts is None:
            regime_counts = df["regime"].value_counts()
        all_rows.extend(bucket_stats(df, m))

    table = pd.DataFrame(all_rows)
    table.to_csv(RESULTS / "step6_regime_error.csv", index=False)

    print("regime composition (predictions):")
    total = regime_counts.sum()
    for reg in ["foraging", "moving", "launch", "settle", "unknown"]:
        c = int(regime_counts.get(reg, 0))
        print(f"  {reg:<11} {c:>8,} ({c/total:5.1%})")
    print()

    # Per-regime table, models side by side on median
    print("median error by regime and model (km):")
    piv = table.pivot(index="model", columns="regime", values="median_km")
    piv = piv.reindex(index=present, columns=["foraging", "moving", "launch", "settle"])
    print(piv.to_string(float_format=lambda v: f"{v:.3f}"))
    print()
    print("p99 error by regime and model (km):")
    piv99 = table.pivot(index="model", columns="regime", values="p99_km")
    piv99 = piv99.reindex(index=present, columns=["foraging", "moving", "launch", "settle"])
    print(piv99.to_string(float_format=lambda v: f"{v:.2f}"))

    # Grouped bar: median error per regime, per model
    fig, (a_med, a_p99) = plt.subplots(1, 2, figsize=(13, 5))
    regimes = ["foraging", "moving", "launch", "settle"]
    x = np.arange(len(regimes))
    w = 0.15
    colours = {"persistence": "#4C72B0", "avg_k1": "#8172B3", "avg_k3": "#937860",
               "lstm": "#DA8BC3", "transformer": "#C44E52",
               "lstm_time": "#55A868", "transformer_time": "#CCB974"}
    for i, m in enumerate(present):
        med = [table[(table.model == m) & (table.regime == r)]["median_km"].values
               for r in regimes]
        med = [v[0] if len(v) else np.nan for v in med]
        a_med.bar(x + i * w, med, w, label=m, color=colours.get(m, "#888888"))
        p99 = [table[(table.model == m) & (table.regime == r)]["p99_km"].values
               for r in regimes]
        p99 = [v[0] if len(v) else np.nan for v in p99]
        a_p99.bar(x + i * w, p99, w, label=m, color=colours.get(m, "#888888"))
    centre = (len(present) - 1) / 2.0 * w   # centre each regime's tick under its group, whatever the model count
    for ax, ttl in [(a_med, "median"), (a_p99, "p99")]:
        ax.set_xticks(x + centre); ax.set_xticklabels(regimes)
        ax.set_ylabel(f"{ttl} error (km)"); ax.set_title(f"{ttl} error by regime")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / "step6_regime_error.png", dpi=140); plt.close(fig)

    print(f"\ntable -> {RESULTS / 'step6_regime_error.csv'}")
    print(f"plot  -> {RESULTS / 'step6_regime_error.png'}")


if __name__ == "__main__":
    main()