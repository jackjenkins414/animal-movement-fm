# GenAI reference: due to time, I used GenAI to draft this script from my outline; to answer the question "does already reported travel p90 skill
# track how random or directed a species moves"; from my outline ofthe training individuals only from the frozen splits, one bin 6 h steps via 
# cadencesteps, the metrics and their filters (Rbar_corr primary, rho, etc. I did have to change one thing when checking it, corrected the 
# implementation of`travel_boundary` which originally assumed every profile table had `selected_boundary_m`; I added the fallback to 
# `median_anchor_boundary_m` for the bird profiles. 
"""
Cross-species movement randomness, against predictability already reported by the sealed evaluators.

Descriptive EDA on TRAINING individuals only; no val or test track is read and no evaluator is run. Skill numbers are read from the Tier 4 and bird 
study report JSONs. Nothing here is a registered result.
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments" / "training"))

import bird_train as BT  # noqa: E402

from animal_movement_fm.analysis.randomness import (  # noqa: E402
    mean_resultant_length,
    mean_resultant_length_corrected,
)
from animal_movement_fm.analysis.regime_profile import (  # noqa: E402
    MIN_STEPS_PER_AMPLITUDE_BIN,
    cadence_steps,
    coverage_distances,
    directional_persistence_from_turns,
    local_en_metres,
    normalise_individual_ids,
    restrict_to_individuals,
    transform_descriptors,
)
from animal_movement_fm.data.split import load_split  # noqa: E402

PROFILE_TABLES = (REPO / "results" / "regime_profiles.csv", REPO / "results" / "birds" / "regime_profiles.csv")
PROCESSED = REPO / "data" / "processed"
OUT = REPO / "results" / "randomness"
TIER2_DECISION = REPO / "registered" / "tier2_regime_decision.json"
BIRD_POOLS = REPO / "registered" / "birds_source_pool.json"
TIER4_REPORT = REPO / "results" / "tier4_sealed" / "tier4_results.json"
BIRD_REPORT = REPO / "results" / "birds_sealed" / "birds_results.json"

# Profile-table name -> the segment file training reads (the A9-QC'd stork build, not the Tier 1 stork_segments). 
SEGMENT_STEM = {"white_stork": "ciconia_ciconia"}
# registered pool name -> profile-table name
PROFILE_NAME = {"ciconia_ciconia": "white_stork"}

HELD_OUT = {
    "larus_fuscus": "gull",
    "canis_lupus": "wolf",
    "mirounga_angustirostris": "NES",
    "anser_albifrons": "goose",
    "streptopelia_turtur": "dove",
}
HELD_OUT_BIRDS = ("larus_fuscus", "anser_albifrons", "streptopelia_turtur")

MIN_TURNS = 30  # Per individual, primary
MIN_TURNS_SENSITIVITY = 20  # Species medians only, no intervals
MIN_PAIRS = 30  # Consecutive one-bin step pairs for the lag-1 autocorrelation
MIN_STEPS = 100  # One-bin steps for tail ratio, intermittency and both diel amplitudes
N_PHASES = 4  # 6 h bins per day; bin_index counts bins from the epoch, so bin_index mod 4 is the UTC quarter of the day
N_BOOT = 2_000
SEED = 42
GUARD_TOL = 1e-9
N_NEIGHBOURS = 5

# Per-individual metric -> the individual-level gate it is summarised under.
METRICS = {
    "rbar_corr": "turns",
    "rho": "turns",
    "rbar_raw": "turns",
    "rbar_floor": "turns",
    "rbar_corr_medfilt": "turns_medfilt",
    "rho_medfilt": "turns_medfilt",
    "ac1": "pairs",
    "ac1_nozero": "pairs_nozero",
    "diel_phase": "steps",
    "diel_utc": "steps",
    "tail_ratio": "steps",
    "intermittency": "steps",
}
SENSITIVITY_T20 = ("rbar_corr", "rho")

# Metric space for the neighbour reading; transforms follow DESCRIPTOR_TRANSFORMS where the descriptor exists there. 
SPACE = ["rbar_corr", "rho", "ac1", "diel_phase", "tail_ratio", "intermittency"]
SPACE_TRANSFORMS = {
    "rbar_corr": "identity",
    "rho": "identity",
    "ac1": "identity",
    "diel_phase": "log10",
    "tail_ratio": "log10",
    "intermittency": "logit",
}

SPEARMAN_METRICS = ("rbar_corr", "rho", "ac1", "ac1_nozero", "diel_phase", "diel_utc", "tail_ratio", "intermittency")
SKILL_COLUMNS = ("S", "LSTMS", "XGBS")
MD_SPEARMAN = (("rbar_corr", "Rbar_corr"), ("rho", "rho"), ("ac1", "lag-1 autocorr"))
NAMED_ARMS_T4 = ("LSTMB26", "XGBB26", "B26", "Z26")  # gull, wolf, NES
NAMED_ARMS_BIRDS = ("LSTMBb", "XGBBb", "Bb", "Zb", "Zbc")  # gull, goose, dove

# Reference palette, categorical slots 1 and 2, plus recessive neutrals. 
BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#b9b7b0"
SURFACE, INK, INK_2 = "#fcfcfb", "#0b0b0b", "#52514e"


# ---------------------------------------------------------------------------------------------------------------------
# Loading and guards


def load_profiles() -> pd.DataFrame:
    frames = [pd.read_csv(path).assign(profile_table=path.relative_to(REPO).as_posix()) for path in PROFILE_TABLES]
    table = pd.concat(frames, ignore_index=True)
    if table["species"].duplicated().any():
        raise SystemExit(f"FAIL: species in both profile tables: {table.loc[table['species'].duplicated(), 'species'].tolist()}")
    return table


def travel_boundary(row: pd.Series) -> float:
    """Selected boundary; the birds table carries only the 0.1 x p50 anchor, which is the frozen rule for every species."""
    if "selected_boundary_m" in row.index and pd.notna(row["selected_boundary_m"]):
        return float(row["selected_boundary_m"])
    return float(row["median_anchor_boundary_m"])


def registered_boundaries() -> dict[str, float]:
    """The six Tier 2 role species from the registered decision, the three held-out birds as pinned in bird_train."""
    pinned = dict(json.loads(TIER2_DECISION.read_text())["tier2_boundaries_m"])
    pinned.update(BT.HELD_OUT_BOUNDARY_M)
    return pinned


def check_boundaries(profiles: pd.DataFrame) -> None:
    for species, pinned in registered_boundaries().items():
        value = travel_boundary(profiles.set_index("species").loc[species])
        if abs(value - pinned) > GUARD_TOL:
            raise SystemExit(f"FAIL: {species} boundary {value!r} in the profile table differs from registered {pinned!r}")


def main_pool() -> list[str]:
    pools = json.loads(BIRD_POOLS.read_text())["pools"]["main"]
    names = pools if isinstance(pools, list) else list(pools)
    return [PROFILE_NAME.get(name, name) for name in names]


def training_frame(row: pd.Series) -> pd.DataFrame:
    """Segments of the species' training individuals, and nothing else."""
    stem = SEGMENT_STEM.get(row["species"], row["species"])
    frame = pd.read_parquet(PROCESSED / f"{stem}_segments.parquet")
    train = load_split(REPO / row["split_manifest"]).train
    scoped = restrict_to_individuals(frame, train)
    ids = normalise_individual_ids(scoped["individual_id"].to_numpy())
    if not set(ids) <= set(normalise_individual_ids(train)):
        raise SystemExit(f"FAIL: {row['species']} frame holds individuals outside the training split")
    return scoped.assign(_id=ids)


# ---------------------------------------------------------------------------------------------------------------------
# Per-individual metrics


def one_bin_steps(g: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    cadence_steps' retained steps, re-derived so the pairing and bin phase are available.

    Returns step lengths, their destination bin phase, and the two members of every consecutive pair of retained steps
    that share their middle fix within one segment.
    """
    frame = g.sort_values(["segment_id", "bin_index"]).reset_index(drop=True)
    lon = frame["lon"].to_numpy(dtype=float)
    lat = frame["lat"].to_numpy(dtype=float)
    bin_index = frame["bin_index"].to_numpy()
    group = frame["segment_id"].to_numpy()
    keep = (
        (group[1:] == group[:-1])
        & (np.diff(bin_index) == 1)
        & np.isfinite(lon[:-1]) & np.isfinite(lon[1:]) & np.isfinite(lat[:-1]) & np.isfinite(lat[1:])
    )
    east, north = local_en_metres(lon, lat)
    mag = np.hypot(east, north)
    pair = keep[:-1] & keep[1:]
    return mag[keep], (bin_index[1:] % N_PHASES)[keep], mag[:-1][pair], mag[1:][pair]


def amplitude(mag: np.ndarray, keys: np.ndarray) -> float:
    """profile_species' amplitude: range of bin means over bins with enough steps, relative to the overall mean."""
    mean = float(mag.mean())
    grouped = pd.Series(mag).groupby(keys)
    means = grouped.mean()[grouped.size() >= MIN_STEPS_PER_AMPLITUDE_BIN]
    if mean <= 0 or len(means) < 2:
        return np.nan
    return float((means.max() - means.min()) / mean)


def lag1_spearman(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < MIN_PAIRS or np.all(a == a[0]) or np.all(b == b[0]):
        return np.nan
    return float(spearmanr(a, b).statistic)


def individual_metrics(g: pd.DataFrame, boundary_m: float, species_median_m: float) -> dict:
    travel = cadence_steps(g, moving_threshold_m=boundary_m)
    medfilt = cadence_steps(g, moving_threshold_m=species_median_m)
    mag, phase, first, second = one_bin_steps(g)
    if not np.array_equal(mag, travel.mag_m):
        raise SystemExit(f"FAIL: re-derived step lengths differ from cadence_steps for individual {g['_id'].iloc[0]}")

    turns, turns_med = travel.turn, medfilt.turn
    n_turns = int(turns.size)
    nonzero = (first > 0) & (second > 0)
    row = {
        "individual_id": g["_id"].iloc[0],
        "n_steps": int(mag.size),
        "n_turns": n_turns,
        "n_turns_medfilt": int(turns_med.size),
        "n_pairs": int(first.size),
        "n_pairs_nozero": int(nonzero.sum()),
        "zero_step_fraction": float((mag == 0).mean()) if mag.size else np.nan,
        "rho": directional_persistence_from_turns(turns),
        "rbar_raw": mean_resultant_length(turns),
        "rbar_corr": mean_resultant_length_corrected(turns),
        "rbar_floor": float(np.sqrt(np.pi / (4 * n_turns))) if n_turns else np.nan,
        "rho_medfilt": directional_persistence_from_turns(turns_med),
        "rbar_corr_medfilt": mean_resultant_length_corrected(turns_med),
        "ac1": lag1_spearman(first, second),
        "ac1_nozero": lag1_spearman(first[nonzero], second[nonzero]),
        "diel_phase": np.nan,
        "diel_utc": np.nan,
        "tail_ratio": np.nan,
        "intermittency": np.nan,
    }
    if mag.size >= MIN_STEPS:
        p50 = float(np.percentile(mag, 50))
        row["tail_ratio"] = float(np.percentile(mag, 99) / p50) if p50 > 0 else np.nan
        row["intermittency"] = float((mag < boundary_m).mean())
        row["diel_phase"] = amplitude(mag, phase)
        row["diel_utc"] = amplitude(mag, travel.hour)
    return row


def gate(ind: pd.DataFrame, kind: str, min_turns: int = MIN_TURNS) -> pd.Series:
    if kind == "turns":
        return ind["n_turns"] >= min_turns
    if kind == "turns_medfilt":
        return ind["n_turns_medfilt"] >= min_turns
    if kind == "pairs":
        return ind["n_pairs"] >= MIN_PAIRS
    if kind == "pairs_nozero":
        return ind["n_pairs_nozero"] >= MIN_PAIRS
    if kind == "steps":
        return ind["n_steps"] >= MIN_STEPS
    raise ValueError(kind)


def bootstrap_median(values: np.ndarray) -> tuple[float, float, float, int]:
    """Median over individuals, with a percentile interval from resampling individuals. Fresh seed per call."""
    values = values[np.isfinite(values)]
    n = values.size
    if n == 0:
        return np.nan, np.nan, np.nan, 0
    idx = np.random.default_rng(SEED).integers(0, n, size=(N_BOOT, n))
    medians = np.median(values[idx], axis=1)
    lo, hi = np.percentile(medians, [2.5, 97.5])
    return float(np.median(values)), float(lo), float(hi), n


# ---------------------------------------------------------------------------------------------------------------------
# Skill, read from the sealed reports


def ladder_lookup(path: Path) -> dict[str, dict[str, float]]:
    blocks = json.loads(path.read_text())["ladder"]["blocks"]
    return {rung: values for block in blocks.values() for rung, values in block.items()}


def skill_table() -> pd.DataFrame:
    t4, birds = ladder_lookup(TIER4_REPORT), ladder_lookup(BIRD_REPORT)
    for rung in SKILL_COLUMNS:
        if abs(t4[rung]["larus_fuscus"] - birds[rung]["larus_fuscus"]) > GUARD_TOL:
            raise SystemExit(f"FAIL: the gull's {rung} differs between the Tier 4 and bird reports")
    rows = []
    for species in HELD_OUT:
        source = birds if species in ("anser_albifrons", "streptopelia_turtur") else t4
        row = {"species": species, "held_out": HELD_OUT[species]}
        row.update({rung: source[rung].get(species, np.nan) for rung in SKILL_COLUMNS})
        row.update({rung: t4[rung].get(species, np.nan) for rung in NAMED_ARMS_T4})
        row.update({rung: birds[rung].get(species, np.nan) for rung in NAMED_ARMS_BIRDS})
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------------------------------
# Reporting helpers


def fmt(value: float, digits: int = 3) -> str:
    return "—" if not np.isfinite(value) else f"{value:+.{digits}f}"


def fmt_ci(row: pd.Series, metric: str, digits: int = 3) -> str:
    if not np.isfinite(row[metric]):
        return "—"
    plain = lambda v: f"{v:.{digits}f}"  # noqa: E731
    return f"{plain(row[metric])} [{plain(row[f'{metric}_lo'])}, {plain(row[f'{metric}_hi'])}]"


def md_table(header: list[str], rows: list[list[str]], align: list[str] | None = None) -> str:
    align = align or ["---"] * len(header)
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(align) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GREY)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(color="#ecebe7", linewidth=0.8)
    ax.set_axisbelow(True)


def figure_skill(species: pd.DataFrame, skills: pd.DataFrame, path: Path) -> None:
    held = species.set_index("species").loc[list(HELD_OUT)]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), sharey=True, facecolor=SURFACE)
    for ax, (metric, label) in zip(axes, (("rbar_corr", "Rbar_corr (species median)"), ("rho", "rho (species median)")), strict=True):
        style(ax)
        ax.axhline(0, color=GREY, linewidth=1)
        x = held[metric].to_numpy()
        xerr = np.vstack([x - held[f"{metric}_lo"].to_numpy(), held[f"{metric}_hi"].to_numpy() - x])
        for rung, colour, marker, name in (("LSTMS", BLUE, "o", "LSTM-S"), ("XGBS", ORANGE, "s", "XGB-S")):
            y = skills.set_index("species").loc[list(HELD_OUT), rung].to_numpy()
            ax.errorbar(x, y, xerr=xerr, fmt=marker, ms=8, color=colour, ecolor=colour, elinewidth=1.5,
                        mec=SURFACE, mew=1.5, label=name)
        y_label = skills.set_index("species").loc[list(HELD_OUT)][["LSTMS", "XGBS"]].max(axis=1).to_numpy()
        for xi, yi, name in zip(x, y_label, HELD_OUT.values(), strict=True):
            ax.annotate(name, (xi, yi), xytext=(5, 6), textcoords="offset points", fontsize=9, color=INK)
        ax.set_xlabel(label, color=INK, fontsize=10)
    axes[0].set_ylabel("travel p90 skill against persistence", color=INK, fontsize=10)
    axes[0].legend(frameon=False, fontsize=9, loc="upper left")
    fig.suptitle("Held-out species: specialist skill against turning-angle concentration and persistence",
                 color=INK, fontsize=11)
    fig.text(0.01, 0.01, "Bars: 95% bootstrap interval over training individuals. n = 5 species.", color=INK_2, fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def figure_space(species: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5.6), facecolor=SURFACE)
    style(ax)
    role = species["role"]
    groups = (
        ("other", GREY, 28, "other species"),
        ("main_pool", BLUE, 40, "22 main-pool birds"),
        ("held_out", ORANGE, 70, "held-out five"),
    )
    for key, colour, size, name in groups:
        sub = species[role == key]
        ax.scatter(sub["rbar_corr"], sub["ac1"], s=size, color=colour, edgecolor=SURFACE, linewidth=1.2, label=name,
                   zorder=3 if key != "other" else 2)
    for _, row in species[role == "held_out"].iterrows():
        ax.annotate(HELD_OUT[row["species"]], (row["rbar_corr"], row["ac1"]), xytext=(6, 5),
                    textcoords="offset points", fontsize=9, color=INK)
    ax.set_xlabel("Rbar_corr, turning-angle concentration (species median)", color=INK, fontsize=10)
    ax.set_ylabel("lag-1 Spearman autocorrelation of step length (species median)", color=INK, fontsize=10)
    ax.set_title("46 species, training individuals", color=INK, fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


# ---------------------------------------------------------------------------------------------------------------------


def main() -> None:
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    profiles = load_profiles()
    check_boundaries(profiles)
    pool = main_pool()
    missing = sorted(set(pool) - set(profiles["species"]))
    if missing:
        raise SystemExit(f"FAIL: main-pool birds missing from the profile tables: {missing}")

    individual_rows, species_rows, guard_rows, dove_check = [], [], [], {}
    for _, prof in profiles.iterrows():
        name = prof["species"]
        scoped = training_frame(prof)
        pooled = cadence_steps(scoped)
        rho_pooled = directional_persistence_from_turns(pooled.turn)
        diff = abs(rho_pooled - float(prof["rho"]))
        if diff > GUARD_TOL or pooled.turn.size != int(prof["n_turns_used"]):
            raise SystemExit(
                f"FAIL guard: {name} pooled median-filtered rho {rho_pooled!r} vs table {prof['rho']!r} "
                f"(|diff| {diff:.3e}), turns {pooled.turn.size} vs {prof['n_turns_used']}. Nothing written."
            )
        guard_rows.append({"species": name, "rho_table": float(prof["rho"]), "rho_code": rho_pooled, "abs_diff": diff})

        boundary = travel_boundary(prof)
        species_median = float(np.median(pooled.mag_m))
        ind = pd.DataFrame(
            [individual_metrics(g, boundary, species_median) for _, g in scoped.groupby("_id", sort=False)]
        )
        if int(ind["n_turns_medfilt"].sum()) != pooled.turn.size:
            raise SystemExit(f"FAIL: {name} per-individual median-filter turns do not sum to the pooled count")
        ind.insert(0, "species", name)
        individual_rows.append(ind)

        role = "held_out" if name in HELD_OUT else ("main_pool" if name in pool else "other")
        out = {
            "species": name,
            "domain": prof["domain"],
            "role": role,
            "travel_boundary_m": boundary,
            "species_median_step_m": species_median,
            "n_individuals_train": len(ind),
            "n_individuals_turns30": int(gate(ind, "turns").sum()),
            "n_individuals_turns20": int(gate(ind, "turns", MIN_TURNS_SENSITIVITY).sum()),
            "n_turns_travel_filter": int(ind["n_turns"].sum()),
            "n_turns_travel_filter_kept": int(ind.loc[gate(ind, "turns"), "n_turns"].sum()),
            "zero_step_fraction": float((pooled.mag_m == 0).mean()),
            "rho_pooled_medfilt": rho_pooled,
            "diel_amplitude_profile": float(prof["diel_amplitude"]),
        }
        for metric, kind in METRICS.items():
            med, lo, hi, n = bootstrap_median(ind.loc[gate(ind, kind), metric].to_numpy(dtype=float))
            out.update({metric: med, f"{metric}_lo": lo, f"{metric}_hi": hi, f"{metric}_n": n})
        for metric in SENSITIVITY_T20:
            values = ind.loc[gate(ind, "turns", MIN_TURNS_SENSITIVITY), metric].to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            out[f"{metric}_t20"] = float(np.median(values)) if values.size else np.nan
        species_rows.append(out)

        if name == "streptopelia_turtur":
            pooled_mag, phase, _, _ = one_bin_steps(scoped)
            if not np.array_equal(pooled_mag, pooled.mag_m):
                raise SystemExit("FAIL: the dove's pooled re-derived steps do not align with cadence_steps")
            hours = pd.Series(pooled.mag_m).groupby(pooled.hour)
            dove_check = {
                "hours": pd.DataFrame({"n": hours.size(), "mean_m": hours.mean(), "median_m": hours.median()}),
                "phases": pd.Series(pooled.mag_m).groupby(phase).agg(["size", "mean", "median"]),
                "overall_mean_m": float(pooled.mag_m.mean()),
                "utc_pooled": amplitude(pooled.mag_m, pooled.hour),
                "phase_pooled": amplitude(pooled.mag_m, phase),
                "profile": float(prof["diel_amplitude"]),
            }
        print(f"  {name:<26} {len(ind):>4} train ind  {out['n_individuals_turns30']:>4} with >= {MIN_TURNS} turns  "
              f"Rbar_corr {out['rbar_corr']:.3f}  guard {diff:.1e}", flush=True)

    species = pd.DataFrame(species_rows)
    individuals = pd.concat(individual_rows, ignore_index=True)
    species.to_csv(OUT / "species_randomness.csv", index=False)
    individuals.to_parquet(OUT / "per_individual.parquet", index=False)

    # (a) held-out table and Spearman across the five
    skills = skill_table()
    held = species.set_index("species").loc[list(HELD_OUT)]
    spearman_rows = []
    for metric in SPEARMAN_METRICS:
        for rung in SKILL_COLUMNS:
            x = held[metric].to_numpy(dtype=float)
            y = skills.set_index("species").loc[list(HELD_OUT), rung].to_numpy(dtype=float)
            ok = np.isfinite(x) & np.isfinite(y)
            rho_s = float(spearmanr(x[ok], y[ok]).statistic) if ok.sum() >= 3 else np.nan
            spearman_rows.append({"metric": metric, "skill": rung, "rho_s": rho_s, "n": int(ok.sum())})
    spearman = pd.DataFrame(spearman_rows)
    spearman.to_csv(OUT / "heldout_spearman.csv", index=False)

    # (b) neighbours among the 22 main-pool birds, in a fresh descriptive standardisation over every complete species
    complete = species.dropna(subset=SPACE).reset_index(drop=True)
    excluded = sorted(set(species["species"]) - set(complete["species"]))
    transformed = transform_descriptors(complete[SPACE].astype(float), SPACE_TRANSFORMS)
    standardisation = {
        "descriptors": SPACE,
        "transforms": SPACE_TRANSFORMS,
        "centre": transformed.mean().to_dict(),
        "scale": transformed.std(ddof=0).replace(0, 1.0).to_dict(),
    }
    distances = coverage_distances(complete, descriptors=SPACE, standardisation=standardisation)
    pool_complete = [s for s in pool if s in distances.columns]

    figure_skill(species, skills, OUT / "skill_vs_rbar_rho.png")
    figure_space(species, OUT / "metric_space.png")

    # markdown
    lines = [
        "# Cross-species movement randomness (descriptive, training individuals only)",
        "",
        "Written by `experiments/eda/randomness_species.py`. Not a registered result. Every metric is computed on the",
        "training individuals of the frozen seed-42 splits; no validation or test track is read, and no evaluator is run.",
        "Skill numbers are travel p90 skill against persistence, read from `results/tier4_sealed/tier4_results.json`",
        "(gull, wolf, NES) and `results/birds_sealed/birds_results.json` (goose, dove, and the gull's bird-study arms).",
        "",
        "Steps are one-bin 6 h steps within segments (`cadence_steps`). Turns: both steps at or above the species' travel",
        f"boundary. An individual enters the turn metrics with >= {MIN_TURNS} turns, the lag-1 autocorrelation with",
        f">= {MIN_PAIRS} consecutive step pairs, and the step metrics with >= {MIN_STEPS} steps. Species value = median over",
        f"individuals; interval = 2.5–97.5 percentiles of {N_BOOT:,} bootstrap resamples of individuals, seed {SEED}.",
        f"Guard: pooled median-filtered rho reproduces both profile tables on all {len(species)} species, largest",
        f"difference {max(r['abs_diff'] for r in guard_rows):.1e}.",
        "",
        "**LSTM-S and XGB-S** use one configuration and budget across studies: the bird study's arms take Tier 3's",
        "recipe by construction (`bird_train.py` maps each to its Tier 3 counterpart; `build_config` from",
        "`tier3_train.py`; 20,000 applied updates; XGB-S is `WindowXGBoost` defaults on all training windows, seed 0),",
        "and the gull's are carried from Tier 3. Hardware differs: Tier 3 trained on gpuhopper, the bird study on",
        "gpuvolta in fp32. **S** selects its checkpoint on validation individuals of the same species (as does LSTM-S);",
        "XGB-S carries no selection privilege.",
        "",
        "## (a) Held-out species",
        "",
    ]
    rows = []
    for sp, short in HELD_OUT.items():
        r = held.loc[sp]
        rows.append([
            short, f"{int(r['rbar_corr_n'])} / {int(r['n_individuals_train'])}", fmt_ci(r, "rbar_corr"), fmt_ci(r, "rho"),
            fmt_ci(r, "ac1"), fmt_ci(r, "diel_phase", 2), fmt_ci(r, "diel_utc", 2), fmt_ci(r, "tail_ratio", 1),
            fmt_ci(r, "intermittency"),
        ])
    lines += [md_table(
        ["species", "ind. (>= 30 turns / train)", "Rbar_corr", "rho", "lag-1 autocorr", "diel (bin phase)",
         "diel (UTC hour)", "tail ratio", "intermittency"], rows)]
    lines += [
        "",
        "Diel amplitude: range of mean step across bins (>= 30 steps each), over the individual's mean step. Bin phase",
        "(`bin_index mod 4`, the UTC quarter of the day) is primary here; the UTC-hour version matches the profile",
        "tables. Neither corrects UTC to local solar time.",
        "",
        "Skill (travel p90 skill against persistence):",
        "",
    ]
    sk = skills.set_index("species")
    arm_cols = [("S", "S †"), ("LSTMS", "LSTM-S"), ("XGBS", "XGB-S")] + [(a, a) for a in NAMED_ARMS_T4 + NAMED_ARMS_BIRDS]
    lines += [md_table(
        ["species"] + [label for _, label in arm_cols],
        [[HELD_OUT[sp]] + [fmt(sk.loc[sp, col], 4) for col, _ in arm_cols] for sp in HELD_OUT],
    )]
    lines += [
        "",
        "† validation-selected. LSTM-S, XGB-S and S are the predictability columns; the pooled and zero-shot arms are",
        "shown by name only, for the neighbour reading in (b).",
        "",
        "Spearman rank correlation across the five held-out species (`heldout_spearman.csv` has every metric against",
        "S, LSTM-S and XGB-S):",
        "",
    ]
    sp_idx = spearman.set_index(["metric", "skill"])
    lines += [md_table(
        ["metric", "vs LSTM-S: rho_s (n)", "vs XGB-S: rho_s (n)"],
        [[label] + [f"{fmt(sp_idx.loc[(m, s), 'rho_s'], 2)} ({int(sp_idx.loc[(m, s), 'n'])})" for s in ("LSTMS", "XGBS")]
         for m, label in MD_SPEARMAN],
    )]
    lines += [
        "",
        "At n = 5 a perfect ordering has one-tailed p = 1/120.",
        "",
        "## (b) Nearest main-pool birds to each held-out bird",
        "",
        "Euclidean distance in a fresh, descriptive standardisation over every species with all six metrics",
        f"({', '.join(SPACE)}; log10 on diel and tail ratio, logit on intermittency).",
        f"Excluded for a missing metric: {', '.join(excluded) if excluded else 'none'}.",
        "",
    ]
    for bird in HELD_OUT_BIRDS:
        if bird not in distances.index:
            lines += [f"**{HELD_OUT[bird]}**: not placed (missing a metric).", ""]
            continue
        nearest = distances.loc[bird, pool_complete].sort_values().head(N_NEIGHBOURS)
        cs = complete.set_index("species")
        lines += [f"**{HELD_OUT[bird]}**", ""]
        lines += [md_table(
            ["main-pool bird", "distance", "Rbar_corr", "rho", "lag-1 autocorr", "diel (phase)", "tail ratio",
             "intermittency"],
            [[nb, f"{d:.2f}"] + [f"{cs.loc[nb, m]:.3f}" for m in SPACE] for nb, d in nearest.items()],
        ), ""]
    lines += [
        "## (c) The dove's diel amplitude",
        "",
        f"Profile table: {dove_check['profile']:.4f}. Recomputed UTC-hour amplitude on the same pooled steps:",
        f"{dove_check['utc_pooled']:.4f}. Bin-phase amplitude on the same steps: {dove_check['phase_pooled']:.4f}. Overall",
        f"mean step {dove_check['overall_mean_m']:,.1f} m.",
        "",
    ]
    hours = dove_check["hours"]
    lines += [md_table(
        ["UTC hour", "steps", "mean (m)", "median (m)", f"kept (>= {MIN_STEPS_PER_AMPLITUDE_BIN})"],
        [[str(h), f"{int(r['n']):,}", f"{r['mean_m']:,.1f}", f"{r['median_m']:,.1f}",
          "yes" if r["n"] >= MIN_STEPS_PER_AMPLITUDE_BIN else "no"] for h, r in hours.iterrows()],
    ), ""]
    phases = dove_check["phases"]
    lines += [md_table(
        ["bin phase (UTC)", "steps", "mean (m)", "median (m)"],
        [[f"{6 * p:02d}–{6 * p + 6:02d}", f"{int(r['size']):,}", f"{r['mean']:,.1f}", f"{r['median']:,.1f}"]
         for p, r in phases.iterrows()],
    ), ""]
    lines += [
        "## Not done",
        "",
        "An entropy-based predictability bound (Π_max) is future work: too few turns per marine individual.",
        "",
    ]
    (OUT / "heldout_table.md").write_text("\n".join(lines))
    print(f"\nwrote {OUT.relative_to(REPO)}/: species_randomness.csv, per_individual.parquet, heldout_spearman.csv, "
          f"heldout_table.md, skill_vs_rbar_rho.png, metric_space.png  ({time.time() - started:.0f} s)")


if __name__ == "__main__":
    main()
