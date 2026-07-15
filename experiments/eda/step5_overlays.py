# GenAI reference: I wrote the first 4 functions in this file, and the doctrings, imports etc as usual. GenAI assisted 
# in the plotting heavy sections overlay_panel, localisation_panel and main method and minor print formatting. I verified
# the output by checking that segment and bird rankings matched the persistence prediction dump and visually inspecting the
# overlays/localisation plots to ensure prediction stubs and coloured error segments aligned with the true tracks.
"""
Step 5: prediction overlays, per-segment ranking, and error localisation (obsidian note again).

Uses the dumped per-prediction parquets (persistence + transformer). We plot the predictions, rank them, show where the error is:

Notes:
  5a  rank all test segments best -> worst by persistence median error (full table dumped; also a per-bird aggregate for the 
      "105 tracks" framing)
  5b  predicted vs actual overlays for 9 representative segments (3 best / 3 median / 3 worst by persistence error). True track 
      as a line; each prediction drawn as a stub from the true previous point to the predicted point, so the visible gap IS the 
      error. Persistence vs transformer overlaid.
  5c  error localisation: the same 9 tracks coloured by per-step error, so the high-error stretches (migration) light up against 
      the low-error body.
"""

from __future__ import annotations

import os
os.environ["OMP_NUM_THREADS"] = "1"

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results"

REF_MODEL = "persistence" # ranking reference
OVERLAY_MODELS = ["persistence", "transformer"]


def load(name: str) -> pd.DataFrame:
    df = pd.read_parquet(RESULTS / f"preds_{name}.parquet")
    return df.sort_values(["segment_id", "timestamp"]).reset_index(drop=True)


def rank_segments(ref: pd.DataFrame) -> pd.DataFrame:
    """Per-segment error summary, ranked best -> worst by median error."""
    g = ref.groupby("segment_id")
    tbl = pd.DataFrame({
        "individual_id": g["individual_id"].first(),
        "n_preds": g.size(),
        "median_km": g["error_km"].median(),
        "mean_km": g["error_km"].mean(),
        "p99_km": g["error_km"].quantile(0.99),
        "max_km": g["error_km"].max(),
    }).reset_index()
    return tbl.sort_values("median_km").reset_index(drop=True)


def rank_birds(ref: pd.DataFrame) -> pd.DataFrame:
    """Per bird aggregate, for the 105 tracks."""
    g = ref.groupby("individual_id")
    tbl = pd.DataFrame({
        "n_segments": g["segment_id"].nunique(),
        "n_preds": g.size(),
        "median_km": g["error_km"].median(),
        "mean_km": g["error_km"].mean(),
        "p99_km": g["error_km"].quantile(0.99),
    }).reset_index()
    return tbl.sort_values("median_km").reset_index(drop=True)


def pick_representatives(ranked: pd.DataFrame, k: int = 3) -> list[tuple[str, str]]:
    """3 best, 3 median, 3 worst segment_ids, tagged by band."""
    n = len(ranked)
    mid = n // 2
    picks = []
    for sid in ranked["segment_id"].iloc[:k]:
        picks.append((sid, "best"))
    for sid in ranked["segment_id"].iloc[mid - k // 2: mid - k // 2 + k]:
        picks.append((sid, "median"))
    for sid in ranked["segment_id"].iloc[-k:]:
        picks.append((sid, "worst"))
    return picks


def overlay_panel(ax, seg_preds: dict[str, pd.DataFrame], sid: str, band: str) -> None:
    """True track + per-prediction error stubs for each overlay model."""
    ref = seg_preds[OVERLAY_MODELS[0]]
    # True track --- prev_true -> target chain; use target positions in order.
    ax.plot(ref["target_lon"], ref["target_lat"], "-", color="0.35",
            linewidth=1.0, label="true", zorder=1)

    colours = {"persistence": "#4C72B0", "transformer": "#C44E52"}
    for m in OVERLAY_MODELS:
        d = seg_preds[m]
        # Stub: from true previous point to predicted point.
        segs = np.stack([
            np.c_[d["prev_true_lon"], d["prev_true_lat"]],
            np.c_[d["pred_lon"], d["pred_lat"]],
        ], axis=1)
        segs = segs[np.isfinite(segs).all(axis=(1, 2))]
        lc = LineCollection(segs, colors=colours[m], linewidths=0.6, alpha=0.5, zorder=2)
        ax.add_collection(lc)
        ax.plot([], [], "-", color=colours[m], label=m)         # legend proxy

    med = ref["error_km"].median()
    ax.set_title(f"{band}: {sid[:16]}…  med={med:.2f} km", fontsize=9)
    ax.set_xlabel("lon"); ax.set_ylabel("lat")
    ax.tick_params(labelsize=7)


def localisation_panel(ax, ref_seg: pd.DataFrame, sid: str, band: str) -> None:
    """True track coloured by per-step persistence error (log scale)."""
    lon = ref_seg["target_lon"].to_numpy()
    lat = ref_seg["target_lat"].to_numpy()
    err = ref_seg["error_km"].to_numpy()
    pts = np.c_[lon, lat].reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    c = np.log10(np.clip(err[1:], 1e-2, None))
    lc = LineCollection(segs, cmap="viridis", array=c, linewidths=1.6)
    ax.add_collection(lc)
    ax.autoscale()
    ax.set_title(f"{band}: {sid[:16]}…", fontsize=9)
    ax.set_xlabel("lon"); ax.set_ylabel("lat")
    ax.tick_params(labelsize=7)
    return lc


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    preds = {m: load(m) for m in OVERLAY_MODELS}
    ref = preds[REF_MODEL]

    # 5a ranking
    seg_rank = rank_segments(ref)
    bird_rank = rank_birds(ref)
    seg_rank.to_parquet(RESULTS / "step5_segment_ranking.parquet", index=False)
    bird_rank.to_parquet(RESULTS / "step5_bird_ranking.parquet", index=False)
    print(f"{len(seg_rank)} segments, {len(bird_rank)} birds ranked "
          f"by {REF_MODEL} median error\n")
    print("best 3 segments:");   print(seg_rank.head(3).to_string(index=False))
    print("\nworst 3 segments:"); print(seg_rank.tail(3).to_string(index=False))
    print(f"\nbest bird median  = {bird_rank['median_km'].iloc[0]:.3f} km")
    print(f"worst bird median = {bird_rank['median_km'].iloc[-1]:.3f} km")

    picks = pick_representatives(seg_rank)

    # 5b overlays grid
    fig, axes = plt.subplots(3, 3, figsize=(13, 12))
    for ax, (sid, band) in zip(axes.ravel(), picks):
        seg_preds = {m: preds[m][preds[m]["segment_id"] == sid] for m in OVERLAY_MODELS}
        overlay_panel(ax, seg_preds, sid, band)
    axes.ravel()[0].legend(fontsize=7, loc="best")
    fig.suptitle("Predicted (stubs) vs actual (grey) — persistence vs transformer",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(RESULTS / "step5_overlays.png", dpi=140); plt.close(fig)

    # 5c localisation grid
    fig, axes = plt.subplots(3, 3, figsize=(13, 12))
    last_lc = None
    for ax, (sid, band) in zip(axes.ravel(), picks):
        ref_seg = ref[ref["segment_id"] == sid]
        last_lc = localisation_panel(ax, ref_seg, sid, band)
    fig.colorbar(last_lc, ax=axes, shrink=0.6, label="log10 per-step error (km)")
    fig.suptitle("Error localisation — true track coloured by persistence error",
                 fontsize=12)
    fig.savefig(RESULTS / "step5_localisation.png", dpi=140); plt.close(fig)

    print(f"\noverlays      -> {RESULTS / 'step5_overlays.png'}")
    print(f"localisation  -> {RESULTS / 'step5_localisation.png'}")
    print(f"segment rank  -> {RESULTS / 'step5_segment_ranking.parquet'}")
    print(f"bird rank     -> {RESULTS / 'step5_bird_ranking.parquet'}")

if __name__ == "__main__":
    main()