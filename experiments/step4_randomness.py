# GenAI reference: Here I wrote the diagnostic framing, imports/docstrings/comments, per_window_predictability(), 
# and initial versions of segment_displacements() and autocorr(); GenAI corrected those two functions to avoid 
# biased lag normalisation, then helped me do the main method, plot and minor print cleanup. I validated the output
# by checking displacement magnitudes against haversine step lengths, confirming autocorrelation behaved on a
# simple synthetic series, and verified matched window counts/timestamps against preds_persistence.parquet. 
"""
Step 4; randomness / predictability: is the movement even learnable? Again for obsidian notes hence naming

If the next displacement is uncorrelated with recent displacements, no model beats the trivial baselines; we're basically
predicting a coin flip. This aims to quantify how much learnable structure exists, split by regime, and ties it to actual error.

Works on displacements (consecutive step VECTORS), not positions;  positions are trivially autocorrelated (the bird is near 
where it was) and say nothing about predictability of MOVEMENT. Two displacement quantities per step:
  magnitude   step length (km)
  direction   cosine of the turn angle vs the previous step (+1 straight on, -1 reversal, 0 orthogonal); directional persistence

Notes for obs:
  4a  displacement autocorrelation vs lag, aggregated, split foraging/migration
  4b  per window predictability score (lag-1 autocorr of the step series within the window); distribution, split by regime
  4c  predictability vs persistence error: does low predictability actually cause high error? The demonstration that error 
      is driven by intrinsic randomness, not model weakness.

Note again that model-independent for 4a/4b (segments only); 4c joins preds_persistence.parquet.
"""

from __future__ import annotations

import os
os.environ["OMP_NUM_THREADS"] = "1"

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from animal_movement_fm.eval.metrics import haversine_distance

REPO = Path(__file__).resolve().parents[1]
SEGMENTS = REPO / "data" / "processed" / "stork_segments.parquet"
RESULTS = REPO / "experiments" / "results"

REGIME_CUT_KM = 0.06  # foraging/moving trough
MAX_LAG = 10
WINDOW = 24 # context_length used by the models
MIN_WINDOW_STEPS = 6  # need enough steps for a stable per-window autocorrel


def load_segments() -> pd.DataFrame:
    return duckdb.sql(f"""
        SELECT segment_id, individual_id, bin_index, timestamp, lon, lat
        FROM '{SEGMENTS}'
        ORDER BY segment_id, bin_index
    """).df()


def segment_displacements(seg: pd.DataFrame) -> dict:
    """
    Step vectors, magnitudes (km) and direction-persistence for one segment.

    Displacement is done in a local equirectangular projection (km east/north) so vector operations 
    (dot products, turn angles) are Euclidean valid over the short per-step distances; magnitudes are 
    cross checked against haversine.
    """
    lon = seg["lon"].to_numpy(); lat = seg["lat"].to_numpy()
    lat0 = np.radians(np.mean(lat))
    # local tangent plane km
    east = np.radians(lon) * np.cos(lat0) * 6371.0
    north = np.radians(lat) * 6371.0
    d_east = np.diff(east); d_north = np.diff(north)  # step vectors (km)
    mag = np.hypot(d_east, d_north) # step length km

    # direction persistence; cosine between consecutive step vectors
    v = np.stack([d_east, d_north], axis=1) # (n_steps, 2)
    cos = np.full(len(mag), np.nan)
    for i in range(1, len(mag)):
        a, b = v[i - 1], v[i]
        na, nb = mag[i - 1], mag[i]
        if na > 1e-6 and nb > 1e-6:
            cos[i] = np.dot(a, b) / (na * nb)
    return {"mag": mag, "cos": cos}


def autocorr(x: np.ndarray, max_lag: int) -> np.ndarray:
    """Sample autocorrelation of a 1-D series for lags 1..max_lag (NaN-safe)."""
    x = x[np.isfinite(x)]
    n = len(x)
    if n < max_lag + 2:
        return np.full(max_lag, np.nan)
    x = x - x.mean()
    denom = np.dot(x, x)
    if denom < 1e-12:
        return np.full(max_lag, np.nan)
    return np.array([np.dot(x[:-k], x[k:]) / denom for k in range(1, max_lag + 1)])


def per_window_predictability(mag: np.ndarray) -> list[dict]:
    """
    Slide a WINDOW-length window over a segments step magnitude series; for each, lag1 autocorrelation = how well each 
    step predicts the next within the window. Regime tagged by the window's median step (foraging vs moving).
    """
    out = []
    n = len(mag)
    for t in range(WINDOW, n):
        w = mag[t - WINDOW:t]
        w = w[np.isfinite(w)]
        if len(w) < MIN_WINDOW_STEPS:
            continue
        ac1 = autocorr(w, 1)[0]
        if not np.isfinite(ac1):
            continue
        med = np.median(w)
        out.append({
            "end_idx": t,
            "lag1_autocorr": ac1,
            "median_step_km": med,
            "regime": "moving" if med >= REGIME_CUT_KM else "foraging",
        })
    return out


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    segs = load_segments()
    groups = list(segs.groupby("segment_id", sort=False))
    print(f"{len(groups)} segments\n")

    # ---- 4a: Aggregate autocorrelation vs lag, split by regime ----
    ac_mag_forag, ac_mag_migr = [], []
    ac_cos_all = []
    windows = []
    for sid, seg in groups:
        d = segment_displacements(seg)
        mag, cos = d["mag"], d["cos"]
        if len(mag) < MAX_LAG + 2:
            continue
        # Regime of the whole segment by median step
        if np.median(mag) >= REGIME_CUT_KM:
            ac_mag_migr.append(autocorr(mag, MAX_LAG))
        else:
            ac_mag_forag.append(autocorr(mag, MAX_LAG))
        ac_cos_all.append(autocorr(cos, MAX_LAG))
        windows.extend({**w, "segment_id": sid} for w in per_window_predictability(mag))

    ac_mag_forag = np.nanmean(np.vstack(ac_mag_forag), axis=0)
    ac_mag_migr = np.nanmean(np.vstack(ac_mag_migr), axis=0)
    ac_cos_all = np.nanmean(np.vstack(ac_cos_all), axis=0)

    lags = np.arange(1, MAX_LAG + 1)
    print("4a  displacement-magnitude autocorrelation vs lag:")
    print("  lag :", "  ".join(f"{l:>5}" for l in lags))
    print("  forg:", "  ".join(f"{v:>5.2f}" for v in ac_mag_forag))
    print("  migr:", "  ".join(f"{v:>5.2f}" for v in ac_mag_migr))
    print("  dir :", "  ".join(f"{v:>5.2f}" for v in ac_cos_all), "(direction persistence)\n")

    # ---- 4b: Per-window predictability distribution, by regime ----
    win = pd.DataFrame(windows)
    print(f"4b  per-window lag-1 autocorrelation  (n={len(win):,} windows)")
    for reg in ["foraging", "moving"]:
        s = win.loc[win["regime"] == reg, "lag1_autocorr"]
        print(f"  {reg:<9} median={s.median():+.3f}  mean={s.mean():+.3f}  "
              f"frac<0={np.mean(s < 0):.1%}  (n={len(s):,})")
    print()

    # ---- 4c: Predictability vs persistence error ----
    preds = pd.read_parquet(RESULTS / "preds_persistence.parquet")
    # Map each window to the prediction it precedes: window ends at bin t, the
    # model then predicts bin t. Join on (segment_id, end bin_index) via the
    # segment frame to recover the timestamp preds carries.
    seg_idx = segs[["segment_id", "bin_index", "timestamp"]].copy()
    seg_idx["end_idx"] = seg_idx.groupby("segment_id").cumcount()
    win = win.merge(seg_idx[["segment_id", "end_idx", "timestamp"]],
                    on=["segment_id", "end_idx"], how="left")
    tie = win.merge(preds[["segment_id", "timestamp", "error_km"]],
                    on=["segment_id", "timestamp"], how="inner")
    print(f"4c  predictability vs error  (n={len(tie):,} matched)")
    if len(tie):
        tie["ac_bin"] = pd.cut(tie["lag1_autocorr"], bins=[-1, -0.3, 0, 0.3, 0.6, 1.0])
        summ = tie.groupby("ac_bin", observed=True)["error_km"].agg(["median", "mean", "size"])
        with pd.option_context("display.float_format", lambda v: f"{v:,.3f}"):
            print(summ.to_string())
        corr = tie["lag1_autocorr"].corr(np.log1p(tie["error_km"]), method="spearman")
        print(f"\n  Spearman(lag1_autocorr, log error) = {corr:+.3f}")
        print("  (negative => more predictable windows have lower error, as expected)")

    # ---- Plots ----
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5))
    a1.axhline(0, color="grey", lw=0.8)
    a1.plot(lags, ac_mag_forag, "o-", label="foraging (magnitude)", color="#4C72B0")
    a1.plot(lags, ac_mag_migr, "o-", label="migration (magnitude)", color="#C44E52")
    a1.plot(lags, ac_cos_all, "s--", label="direction persistence", color="#55A868")
    a1.set_xlabel("lag (steps)"); a1.set_ylabel("autocorrelation")
    a1.set_title("4a  displacement autocorrelation by regime"); a1.legend()

    for reg, c in [("foraging", "#4C72B0"), ("moving", "#C44E52")]:
        s = win.loc[win["regime"] == reg, "lag1_autocorr"]
        a2.hist(s, bins=60, histtype="step", linewidth=1.4, color=c, label=reg)
    a2.axvline(0, color="grey", lw=0.8)
    a2.set_xlabel("per-window lag-1 autocorrelation"); a2.set_ylabel("count")
    a2.set_title("4b  per-window predictability"); a2.legend()
    fig.tight_layout()
    out = RESULTS / "step4_randomness.png"
    fig.savefig(out, dpi=150); plt.close(fig)

    win.to_parquet(RESULTS / "step4_window_predictability.parquet", index=False)
    print(f"\nplot    -> {out}")
    print(f"windows -> {RESULTS / 'step4_window_predictability.parquet'}")


if __name__ == "__main__":
    main()