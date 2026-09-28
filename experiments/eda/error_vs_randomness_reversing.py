# GenAI reference: I wrote the beginning implementation of this file, but again, due to time, got GenAI to finish it form my 
# specification. Specifically I wrote`kappa_tag`, `build_cells`, `von_mises_a`, `generator_parameters`, `one_step_m`, `travel_p90_skill`, 
# and `track_indices`; the remaining functions were written by GenAI and I have reviewed them but left them as is. Docs are mine bar those 
# functions and minor comments. 
"""
Here we explored if the LSTM's advantage over persistence follow Rbar_corr (turning concentration) rather than rho (signed persistence)?

A reversing CRW at concentration kappa has the same Rbar as a forward CRW at that kappa and the opposite rho. If the LSTM's skill is the same 
for both, it tracks Rbar_corr; if it collapses on the reversing tracks, it tracks rho. The oracle, which knows the generator, is the ceiling 
for a mean- isplacement forecast from the last heading.

Synthetic only; nothing here touches our real data.
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import inspect
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from scipy.special import i0, i1  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import error_vs_randomness as EVR  # noqa: E402

from animal_movement_fm.analysis.randomness import mean_resultant_length_corrected  # noqa: E402
from animal_movement_fm.data.synthetic import (  # noqa: E402
    generate_brownian_motion,
    generate_crw,
    generate_switching_crw,
)
from animal_movement_fm.eval.harness import evaluate  # noqa: E402
from animal_movement_fm.eval.metrics import haversine_distance  # noqa: E402
from animal_movement_fm.models.lstm import LSTM  # noqa: E402
from animal_movement_fm.models.persistence import Persistence  # noqa: E402

OUT = REPO / "results" / "randomness" / "synthetic"

KAPPAS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)
LOW_DISPLACEMENT_FRACTION = 0.10  # travel boundary = 0.1 x median one-step step on training tracks, as A4
MIN_TURNS = 30  # per track, as the real analysis
NOISE_SCALE = 0.05  # LSTM/oracle is reported only where the oracle's skill clears this

# reference palette, categorical slots 1-3, plus a recessive neutral
FAMILY_STYLE = {
    "crw_forward": ("#2a78d6", "CRW, forward (mu = 0)"),
    "crw_reversing": ("#eb6834", "CRW, reversing (mu = pi)"),
    "switching_crw": ("#1baf7a", "switching CRW"),
    "brownian_motion": ("#8c8a84", "Brownian motion"),
}
# this below is GenAI for reference
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#ecebe7"
LSTM_COLOUR, ORACLE_COLOUR = "#2a78d6", "#eb6834"  # as LSTM-S / XGB-S in the randomness_species figures

# real held-out species Rbar_corr marked on skill_vs_rbar.png; label and text row (goose raised to clear wolf)
REAL_SPECIES = REPO / "results" / "randomness" / "species_randomness.csv"
REAL_TICKS = {
    "canis_lupus": ("wolf", 0),
    "anser_albifrons": ("goose", 1),
    "larus_fuscus": ("gull", 0),
    "streptopelia_turtur": ("dove", 0),
    "mirounga_angustirostris": ("NES", 0),
}


def kappa_tag(kappa: float) -> str:
    return f"k{kappa:g}".replace(".", "p")


def build_cells() -> dict[str, dict]:
    """Build the synthetic experiment grid. kappa = 0 is run once because turning is uniform."""
    cells = {
        "brownian_motion": {
            "family": "brownian_motion",
            "make": generate_brownian_motion,
            "kwargs": {},
        },
        "switching_crw": {
            "family": "switching_crw",
            "make": generate_switching_crw,
            "kwargs": {},
        },
    }

    for kappa in KAPPAS:
        cells[f"crw_{kappa_tag(kappa)}_fwd"] = {
            "family": "crw_forward",
            "make": generate_crw,
            "kwargs": {"turning_kappa": kappa, "turning_mu": 0.0},
            "kappa": kappa,
        }

        if kappa > 0:
            cells[f"crw_{kappa_tag(kappa)}_rev"] = {
                "family": "crw_reversing",
                "make": generate_crw,
                "kwargs": {"turning_kappa": kappa, "turning_mu": np.pi},
                "kappa": kappa,
            }

    return cells

# Oracle

def von_mises_a(kappa: float) -> float:
    """Return the von Mises mean resultant length A(kappa) = I1(kappa) / I0(kappa)."""
    return 0.0 if kappa == 0 else float(i1(kappa) / i0(kappa))


def generator_parameters(spec: dict) -> dict:
    """Return generator defaults overridden by the cell-specific arguments."""
    defaults = {
        name: p.default
        for name, p in inspect.signature(spec["make"]).parameters.items()
        if p.default is not inspect.Parameter.empty
    }
    return {**defaults, **spec["kwargs"]}


def oracle_coefficients(track: pd.DataFrame, spec: dict) -> tuple[np.ndarray, float]:
    """
    Expected next displacement along the last heading, per fix, and the rotation mu applied to that heading.

    CRW: mean step x A(kappa), rotated by mu. Switching CRW: the expectation over the next state given the currentcone (the state of the step 
    into the last fix) and p_stay, never the target step's own state. BM: 0.
    """
    params = generator_parameters(spec)
    n = len(track)
    if spec["family"] == "brownian_motion":
        return np.zeros(n), 0.0
    if spec["family"] in ("crw_forward", "crw_reversing"):
        coef = params["step_shape"] * params["step_scale"] * von_mises_a(params["turning_kappa"])
        return np.full(n, coef), float(params["turning_mu"])
    if spec["family"] == "switching_crw":
        by_state = np.array([
            params["step_shape"] * params["foraging_step_scale"] * von_mises_a(params["foraging_kappa"]),
            params["step_shape"] * params["transit_step_scale"] * von_mises_a(params["transit_kappa"]),
        ])
        state = track["state"].to_numpy().astype(int)
        return params["p_stay"] * by_state[state] + (1 - params["p_stay"]) * by_state[1 - state], 0.0
    raise ValueError(spec["family"])


def oracle_errors(df_test: pd.DataFrame, pers: pd.DataFrame, spec: dict) -> np.ndarray:
    """Oracle error on exactly persistence's windows; target i, last fix i - 1, heading from i - 2 to i - 1."""
    parts = []
    for ind, track in df_test.sort_values(["individual_id", "timestamp"]).groupby("individual_id", sort=False):
        lon, lat = track["lon"].to_numpy(), track["lat"].to_numpy()
        coef, mu = oracle_coefficients(track, spec)
        heading = np.arctan2(lat[1:-1] - lat[:-2], lon[1:-1] - lon[:-2]) + mu  # Into fix i - 1, for targets i >= 2.
        c = coef[1:-1]  # Indexed by the last fix, i - 1.
        pred_lon = lon[1:-1] + c * np.cos(heading)
        pred_lat = lat[1:-1] + c * np.sin(heading)
        parts.append(pd.DataFrame({
            "individual_id": ind,
            "timestamp": track["timestamp"].to_numpy()[2:],
            "oracle_error_m": haversine_distance(pred_lon, pred_lat, lon[2:], lat[2:]),
        }))
    merged = pers[["individual_id", "timestamp"]].merge(pd.concat(parts), on=["individual_id", "timestamp"],
                                                       how="left", validate="one_to_one")
    if merged["oracle_error_m"].isna().any():
        raise SystemExit("FAIL: oracle predictions do not cover every persistence window")
    return merged["oracle_error_m"].to_numpy()


# Scoring

def one_step_m(df: pd.DataFrame) -> pd.Series:
    """Return the one step Haversine distance within each track."""
    g = df.groupby("individual_id", sort=False)
    return pd.Series(
        haversine_distance(
            g["lon"].shift().to_numpy(),
            g["lat"].shift().to_numpy(),
            df["lon"].to_numpy(),
            df["lat"].to_numpy(),
        ),
        index=df.index,
    )

def track_indices(track: pd.DataFrame, boundary_m: float) -> dict:
    """Compute Rbar_corr and rho from turns whose adjacent steps both clear the travel boundary."""
    lon, lat = track["lon"].to_numpy(), track["lat"].to_numpy()
    heading = np.arctan2(np.diff(lat), np.diff(lon))
    steps = haversine_distance(lon[:-1], lat[:-1], lon[1:], lat[1:])
    moving = steps >= boundary_m

    pair = moving[:-1] & moving[1:]
    delta = np.diff(heading)[pair]
    turns = np.arctan2(np.sin(delta), np.cos(delta))
    n = int(turns.size)

    return {
        "n_turns": n,
        "rbar_corr": mean_resultant_length_corrected(turns) if n >= MIN_TURNS else np.nan,
        "rho": float(np.cos(turns).mean()) if n >= MIN_TURNS else np.nan,
    }


def travel_p90_skill(pers_err: np.ndarray, model_err: np.ndarray, travel: np.ndarray) -> float:
    if travel.sum() == 0:
        return np.nan

    p90_p = np.percentile(pers_err[travel], 90)
    return float(1.0 - np.percentile(model_err[travel], 90) / p90_p) if p90_p > 0 else np.nan


def run_cell(name: str, spec: dict, seed: int) -> pd.DataFrame:
    torch.manual_seed(seed)  # Per-cell and seed, so a cell run alone matches the same cell inside the full grid. 
    df = spec["make"](n_individuals=EVR.N_INDIVIDUALS, n_steps=EVR.N_STEPS, seed=seed, **spec["kwargs"])
    df_train, df_test = EVR.split_by_individual(df, EVR.TEST_FRAC, seed)
    boundary_m = LOW_DISPLACEMENT_FRACTION * float(np.nanmedian(one_step_m(df_train.reset_index(drop=True))))

    pers = evaluate(Persistence(), df_test, context_length=EVR.CONTEXT_LENGTH, show_progress=False)
    lstm_model = LSTM(hidden_dim=64, num_layers=1, context_length=EVR.CONTEXT_LENGTH, epochs=30, batch_size=64,
                      lr=1e-3, device="cpu")
    lstm_model.fit(df_train)
    lstm = evaluate(lstm_model, df_test, context_length=EVR.CONTEXT_LENGTH, show_progress=False)
    same = (pers[["individual_id", "timestamp"]].to_numpy() == lstm[["individual_id", "timestamp"]].to_numpy()).all()
    if len(pers) != len(lstm) or not same:
        raise SystemExit(f"FAIL: {name}: persistence and LSTM windows do not align")

    oracle = oracle_errors(df_test, pers, spec)
    if spec["family"] == "brownian_motion" or spec.get("kappa") == 0:
        if not np.allclose(oracle, pers["error_m"].to_numpy(), rtol=1e-9, atol=1e-6):
            raise SystemExit(f"FAIL: {name}: with A = 0 the oracle must equal persistence")

    # Persistence's error is the true step, so it defines the travel stratum window by window.
    rows = []
    for ind, track in df_test.groupby("individual_id", sort=False):
        mask = (pers["individual_id"] == ind).to_numpy()
        pe, le, oe = pers.loc[mask, "error_m"].to_numpy(), lstm.loc[mask, "error_m"].to_numpy(), oracle[mask]
        travel = pe >= boundary_m
        rows.append({
            "cell": name,
            "seed": seed,
            "family": spec["family"],
            "kappa": spec.get("kappa", np.nan),
            "individual_id": ind,
            "n_windows": int(mask.sum()),
            "travel_fraction": float(travel.mean()),
            "travel_p90_skill": travel_p90_skill(pe, le, travel),
            "advantage_mean": float((pe.mean() - le.mean()) / pe.mean()),
            "oracle_travel_p90_skill": travel_p90_skill(pe, oe, travel),
            "oracle_advantage_mean": float((pe.mean() - oe.mean()) / pe.mean()),
            "boundary_m": boundary_m,
            **track_indices(track, boundary_m),
        })
    return pd.DataFrame(rows)


def style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#b9b7b0")
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def figure(per_track: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), sharey=True, facecolor=SURFACE)
    panels = (("rbar_corr", "Rbar_corr (per test track)"), ("rho", "rho (per test track)"))
    for ax, (index, label) in zip(axes, panels, strict=True):
        style(ax)
        ax.axhline(0, color="#b9b7b0", linewidth=1)
        for family, (colour, name) in FAMILY_STYLE.items():
            sub = per_track[per_track["family"] == family]
            ax.scatter(sub[index], sub["travel_p90_skill"], s=16, color=colour, alpha=0.75, edgecolor="none",
                       label=name)
        ax.set_xlabel(label, color=INK, fontsize=10)
    axes[0].set_ylabel("LSTM travel p90 skill against persistence", color=INK, fontsize=10)
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("Synthetic tracks: does LSTM skill follow turning concentration or signed persistence?",
                 color=INK, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def figure_rbar(cells: pd.DataFrame, path: Path) -> None:
    """LSTM and oracle skill against Rbar_corr on the CRW grid, with the real held-out species' Rbar_corr as ticks."""
    real = pd.read_csv(REAL_SPECIES).set_index("species")["rbar_corr"]
    fig, ax = plt.subplots(figsize=(8, 5.2), facecolor=SURFACE)
    style(ax)
    ax.axhline(0, color="#b9b7b0", linewidth=1)
    series = [(cells[cells["family"] == fam].sort_values("kappa"), marker, name)
              for fam, marker, name in (("crw_forward", "o", "forward"), ("crw_reversing", "s", "reversing"))]
    for d, marker, name in series:
        yerr = [d["lstm_skill"] - d["lstm_skill_min"], d["lstm_skill_max"] - d["lstm_skill"]]
        ax.errorbar(d["rbar_corr"], d["lstm_skill"], yerr=yerr, fmt=marker + "-", ms=7, color=LSTM_COLOUR, lw=1.2,
                    elinewidth=1.2, mec=SURFACE, mew=1.2, label=f"LSTM, {name}")
    for d, marker, name in series:
        ax.plot(d["rbar_corr"], d["oracle_skill"], marker + "--", ms=7, color=ORACLE_COLOUR, lw=1.2, mfc=SURFACE,
                mew=1.5, label=f"oracle, {name}")
    bottom = -0.075
    crw = cells[cells["family"].isin(["crw_forward", "crw_reversing"])]
    ax.set_ylim(bottom, max(0.34, float(crw[["oracle_skill", "lstm_skill_max"]].max().max()) + 0.02))
    for species, (label, row) in REAL_TICKS.items():
        x = float(real.loc[species])
        ax.plot([x, x], [bottom, bottom + 0.017], color=INK, lw=1.5, solid_capstyle="butt")
        ax.text(x, bottom + 0.021 + 0.013 * row, label, ha="center", va="bottom", fontsize=8.5, color=INK)
    ax.set_xlabel("Rbar_corr (cell median across seeds)", color=INK, fontsize=10)
    ax.set_ylabel("travel p90 skill against persistence", color=INK, fontsize=10)
    ax.set_title("Synthetic CRW: LSTM and oracle skill against turning concentration", color=INK, fontsize=11)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left", ncol=2)
    fig.text(0.01, 0.01, "Bars: LSTM min-max over seeds. Oracle forward and reversing coincide by construction. "
             "Ticks: real held-out species' Rbar_corr (results.md sec 21.1).", color=INK_2, fontsize=7.5)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def summarise(per_track: pd.DataFrame) -> pd.DataFrame:
    """Median over tracks per cell and seed, then the median and range across seeds per cell."""
    by_seed = per_track.groupby(["cell", "seed"], sort=False).agg(
        family=("family", "first"), kappa=("kappa", "first"), tracks=("individual_id", "size"),
        rbar_corr=("rbar_corr", "median"), rho=("rho", "median"),
        lstm_skill=("travel_p90_skill", "median"), oracle_skill=("oracle_travel_p90_skill", "median"),
        lstm_advantage=("advantage_mean", "median"), oracle_advantage=("oracle_advantage_mean", "median"),
    ).reset_index()
    by_seed["lstm_over_oracle"] = np.where(by_seed["oracle_skill"] > NOISE_SCALE,
                                           by_seed["lstm_skill"] / by_seed["oracle_skill"], np.nan)
    by_seed.to_csv(OUT / "cells_by_seed.csv", index=False)

    values = ["rbar_corr", "rho", "lstm_skill", "oracle_skill", "lstm_over_oracle", "lstm_advantage",
              "oracle_advantage"]
    grouped = by_seed.groupby("cell", sort=False)
    cells = grouped[["family", "kappa"]].first()
    cells["seeds"] = grouped.size()
    for v in values:
        cells[v] = grouped[v].median()
        cells[f"{v}_min"] = grouped[v].min()
        cells[f"{v}_max"] = grouped[v].max()
    cells.to_csv(OUT / "cells.csv")

    rows = []
    mirrored = per_track["family"].isin(["crw_forward", "crw_reversing"])
    for tracks, subset in (("all_cells", per_track), ("mirrored_crw", per_track[mirrored])):
        for outcome in ("travel_p90_skill", "advantage_mean"):
            for index in ("rbar_corr", "rho"):
                ok = subset[[outcome, index]].dropna()
                rows.append({"tracks": tracks, "outcome": outcome, "index": index,
                             "rho_s": float(spearmanr(ok[index], ok[outcome]).statistic), "n_tracks": len(ok)})
    spearman = pd.DataFrame(rows)
    spearman.to_csv(OUT / "spearman.csv", index=False)

    show = ["seeds", "rbar_corr", "rho", "lstm_skill", "lstm_skill_min", "lstm_skill_max", "oracle_skill",
            "lstm_over_oracle"]
    with pd.option_context("display.width", 200, "display.float_format", lambda v: f"{v:+.3f}"):
        print(cells[show].to_string())
        print(spearman.to_string(index=False))
    return cells


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cell", help="run this one cell only and report its time")
    parser.add_argument("--seeds", type=int, default=1, help="run seeds 0 .. N - 1 (default 1)")
    parser.add_argument("--list", action="store_true", help="list the cells and stop")
    parser.add_argument("--figure-only", action="store_true",
                        help="rebuild the summaries and both figures from the saved per_track.csv; nothing is retrained")
    args = parser.parse_args()

    cells = build_cells()
    if args.list:
        print("\n".join(cells))
        return
    if args.figure_only:
        per_track = pd.read_csv(OUT / "per_track.csv")
        summary = summarise(per_track)
        figure(per_track, OUT / "skill_vs_index.png")
        figure_rbar(summary, OUT / "skill_vs_rbar.png")
        print(f"\nrebuilt summaries and figures in {OUT.relative_to(REPO)}/ from per_track.csv")
        return
    if args.cell and args.cell not in cells:
        raise SystemExit(f"unknown cell {args.cell!r}; see --list")
    if args.seeds < 1:
        raise SystemExit("--seeds must be at least 1")
    OUT.mkdir(parents=True, exist_ok=True)

    names = [args.cell] if args.cell else list(cells)
    parts, started = [], time.time()
    for seed in range(args.seeds):
        for name in names:
            t0 = time.time()
            part = run_cell(name, cells[name], seed)
            part.to_csv(OUT / f"cell_{name}_s{seed}.csv", index=False)
            parts.append(part)
            print(f"  s{seed} {name:<18} {len(part):>3} tracks  Rbar_corr {part['rbar_corr'].median():.3f}  "
                  f"rho {part['rho'].median():+.3f}  LSTM {part['travel_p90_skill'].median():+.3f}  "
                  f"oracle {part['oracle_travel_p90_skill'].median():+.3f}  ({time.time() - t0:.0f} s)", flush=True)

    if args.cell:
        print(f"\n{args.cell}, {args.seeds} seed(s): {time.time() - started:.0f} s; {len(cells)} cells in the full grid")
        return

    per_track = pd.concat(parts, ignore_index=True)
    per_track.to_csv(OUT / "per_track.csv", index=False)
    summary = summarise(per_track)
    figure(per_track, OUT / "skill_vs_index.png")
    figure_rbar(summary, OUT / "skill_vs_rbar.png")
    print(f"\nwrote {OUT.relative_to(REPO)}/  ({time.time() - started:.0f} s)")


if __name__ == "__main__":
    main()
