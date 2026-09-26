# GenAI reference: I specified this file, outlined our planned 95 run plan by index (pooled arms, then the goose and dove specialists arm by
# arm, then Fb last in gull, goose, dove order), also gave it my previous Tier 2/3/4 counterparts to use by import and write in the same light,
# Fb fine tuning Zb seed 0, the six goose exclusions read from the registered file and removed before the k% subsets are drawn, the pinned
# held-out boundaries, the refusals, and records tagged "birds"
"""
Train the bird only study's arms, one model per PBS subjob (`preregistration_birds.md` sec 4).
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO))

import tier2_final_train as T  # noqa: E402
import tier2_inner_loop as IL  # noqa: E402
import tier3_train as T3  # noqa: E402
import tier4_train as T4  # noqa: E402

from animal_movement_fm.data.enu import SCALE_FLOOR_M  # noqa: E402
from animal_movement_fm.data.multispecies import SourcePool  # noqa: E402
from animal_movement_fm.data.split import load_split  # noqa: E402
from animal_movement_fm.train.finetune import nested_subsets  # noqa: E402
from animal_movement_fm.train.loop import TrainConfig, train  # noqa: E402
from animal_movement_fm.train.pool import PreparedPool  # noqa: E402
from experiments.data import source_pool_weights as SPW  # noqa: E402

RESULTS = REPO / "results" / "birds"
CHECKPOINTS = REPO / "checkpoints" / "birds"
REGISTERED_POOL = SPW.BIRDS_OUT

REGISTERED_STEPS = T3.REGISTERED_STEPS  # 20,000 applied updates at batch 256, as Tier 3/4
SEEDS = T.SEEDS
SPLIT_STEM = T4.SPLIT_STEM
XGB_POOLED_DRAWS = T4.XGB_POOLED_DRAWS

# preregistration_birds.md sec 2, in its order: gull, goose, dove
HELD_OUT = ("larus_fuscus", "anser_albifrons", "streptopelia_turtur")
# the gull's specialists are carried from Tiers 2-3, so only these two are trained here
SPECIALIST_TARGETS = ("anser_albifrons", "streptopelia_turtur")
CARRIED_GULL_ARMS = frozenset({"P", "V", "S", "ARIMA", "AVG3", "AVG10", "SB", "Sk", "LSTMS", "XGBS"})
if set(HELD_OUT) != SPW.BIRDS_HELD_OUT:
    raise RuntimeError("bird_train.HELD_OUT disagrees with source_pool_weights.BIRDS_HELD_OUT")

# A4 travel boundaries (0.1 x p50 on training individuals), pinned at full precision and checked on every load against
# the profile tables and the frozen rule. The evaluator imports these. The gull's is registered in
# registered/tier2_regime_decision.json; the goose's is its profile row as built on all 299 training individuals; the
# dove's is the median anchor in results/birds/regime_profiles.csv, where --out-csv refuses --decide-threshold.
HELD_OUT_BOUNDARY_M: dict[str, float] = {
    "larus_fuscus": 411.451080530957,
    "anser_albifrons": 82.13945953878455,
    "streptopelia_turtur": 25.806078308740823,
}
PROFILE_TABLES: dict[str, Path] = {
    "larus_fuscus": IL.PROFILES_CSV,
    "anser_albifrons": IL.PROFILES_CSV,
    "streptopelia_turtur": SPW.BIRDS_RESULTS / "regime_profiles.csv",
}

# arm -> its training data and the counterpart whose recipe it takes. `pool` is `main` or `congener` (a node under
# `probabilities` in the registered file) or a held-out bird's own training individuals. The configuration, the
# checkpoint-selection privilege and the XGBoost sampling all come from the counterpart, so they match by construction.
ARMS: dict[str, dict] = {
    "Zb": {"pool": "main", "recipe": (T4, "Z26")},
    "Bb": {"pool": "main", "recipe": (T4, "B26")},
    "Zbc": {"pool": "congener", "recipe": (T4, "Z26")},
    "Bbc": {"pool": "congener", "recipe": (T4, "B26")},
    "LSTMBb": {"pool": "main", "recipe": (T4, "LSTMB26")},
    "XGBBb": {"pool": "main", "recipe": (T4, "XGBB26"), "xgb": True},
    "S": {"pool": "held_out_train", "recipe": (T, "S")},
    "SB": {"pool": "held_out_train", "recipe": (T, "SB")},
    "Sk": {"pool": "held_out_subset", "recipe": (T3, "Sk")},
    "LSTMS": {"pool": "held_out_train", "recipe": (T3, "LSTMS")},
    "XGBS": {"pool": "held_out_train", "recipe": (T3, "XGBS"), "xgb": True},
    "Fb": {"pool": "held_out_subset", "recipe": (T3, "F9"), "init": "Zb_s0"},
}
POOLED_ARMS = tuple(arm for arm, spec in ARMS.items() if spec["pool"] in ("main", "congener"))


@dataclass(frozen=True)
class Run:
    arm: str
    species: str
    seed: int
    k_percent: int | None = None
    draw: int | None = None

    @property
    def run_id(self) -> str:
        if self.k_percent is not None:
            return f"{self.arm}_{self.species}_k{self.k_percent:03d}_d{self.draw}_s{self.seed}"
        return f"{self.arm}_{self.species}_s{self.seed}" if self.species else f"{self.arm}_s{self.seed}"

    @property
    def uses_validation_selection(self) -> bool:
        return counterpart(self).uses_validation_selection

    @property
    def deferred(self) -> bool:  # read by nothing here; kept for the Tier 2 helpers' interface
        return False


def counterpart(run: Run):
    """The Tier 2/3/4 run whose recipe this bird run takes."""
    module, arm = ARMS[run.arm]["recipe"]
    return module.Run(arm=arm, species=run.species, seed=run.seed, k_percent=run.k_percent, draw=run.draw)


def build_plan() -> list[Run]:
    """The bird plan. Position is the PBS array index: append, never insert or reorder."""
    plan = []
    for arm in ("Zb", "Bb", "Zbc", "Bbc", "LSTMBb", "XGBBb"):
        plan += [Run(arm, "", s) for s in SEEDS]
    for arm in ("S", "SB"):
        plan += [Run(arm, sp, s) for sp in SPECIALIST_TARGETS for s in SEEDS]
    plan += [
        Run("Sk", sp, 0, k, d)
        for sp in SPECIALIST_TARGETS
        for k in T3.SK_FRACTIONS
        for d in range(T.FINETUNE_DRAWS_PER_FRACTION[k])
    ]
    plan += [Run("LSTMS", sp, s) for sp in SPECIALIST_TARGETS for s in SEEDS]
    plan += [Run("XGBS", sp, 0) for sp in SPECIALIST_TARGETS]
    plan += [
        Run("Fb", sp, T.FINETUNE_INIT_SEED, k, d)
        for sp in HELD_OUT
        for k, draws in T.FINETUNE_DRAWS_PER_FRACTION.items()
        for d in range(draws)
    ]
    return plan


PLAN = build_plan()
if len(PLAN) != 95:
    raise RuntimeError(f"the bird plan has {len(PLAN)} runs; preregistration_birds.md sec 4 records 95")


# Registered pool.
def load_registered_pool() -> dict:
    """Read the registered bird pools and check them against the constants source_pool_weights.py wrote them from."""
    payload = json.loads(REGISTERED_POOL.read_text())
    for label, registered in (("main", SPW.BIRDS_MAIN_PERCENT), ("congener", SPW.BIRDS_CONGENER_PERCENT)):
        got = {name: round(100.0 * p, 3) for name, p in payload["probabilities"][label].items()}
        if got != registered:
            raise SystemExit(f"FAIL: {REGISTERED_POOL.name} {label} is {got}; the pinned constants say {registered}")
    exclusions = {species: list(ids) for species, ids in SPW.BIRDS_TRAINING_EXCLUSIONS.items()}
    if payload.get("training_exclusions") != exclusions:
        raise SystemExit(
            f"FAIL: {REGISTERED_POOL.name} training_exclusions are {payload.get('training_exclusions')}; "
            f"source_pool_weights.py registers {exclusions}"
        )
    if set(payload["held_out"]) != set(HELD_OUT):
        raise SystemExit(f"FAIL: {REGISTERED_POOL.name} holds out {payload['held_out']}, not {list(HELD_OUT)}")
    return payload


def assert_bird_sources(stems) -> None:
    """Pooled bird arms draw only from the registered birds; never a held-out bird or a still-reserved species."""
    stems = set(stems)
    for label, forbidden in (("held-out", set(HELD_OUT)), ("reserved", set(SPW.BIRDS_RESERVED))):
        if stems & forbidden:
            raise SystemExit(f"FAIL: {sorted(stems & forbidden)} is a {label} species and cannot enter a source pool")
    unknown = stems - set(SPW.BIRDS_CONGENER_PERCENT)
    if unknown:
        raise SystemExit(f"FAIL: {sorted(unknown)} is not a registered bird source")


# Held-out boundaries, and the two Tier 2 helpers pointed at the bird held-out set.
def check_boundaries() -> dict[str, float]:
    """Fail unless each pinned boundary matches its profile row and the frozen A4 rule on training individuals."""
    frozen = json.loads(IL.REGISTERED_DECISION.read_text())
    if int(frozen["anchor_quantile"]) != 50:
        raise SystemExit("FAIL: the frozen A4 rule is no longer anchored on p50")
    multiplier = float(frozen["multiplier"])
    for species, pinned in HELD_OUT_BOUNDARY_M.items():
        row = pd.read_csv(PROFILE_TABLES[species]).set_index("species").loc[species]
        if str(row["split_part"]) != "train":
            raise SystemExit(f"FAIL: {species}'s profile row is not computed on training individuals")
        values = {
            "median_anchor_boundary_m": float(row["median_anchor_boundary_m"]),
            f"{multiplier:g} x step_p50_m": multiplier * float(row["step_p50_m"]),
        }
        if "selected_boundary_m" in row.index and not pd.isna(row["selected_boundary_m"]):
            values["selected_boundary_m"] = float(row["selected_boundary_m"])
        if species in frozen["tier2_boundaries_m"]:
            values["registered"] = float(frozen["tier2_boundaries_m"][species])
        for label, value in values.items():
            if not math.isclose(value, pinned, rel_tol=1e-9, abs_tol=1e-9):
                raise SystemExit(f"FAIL: {species} boundary is pinned at {pinned!r} m but {label} gives {value!r} m")
    return dict(HELD_OUT_BOUNDARY_M)


def bird_attached_entry(species: str) -> tuple[str, str, str]:
    """(segments stem, split stem, profile key) for a held-out bird; the three are the same stem for all of them."""
    if species not in HELD_OUT:
        raise SystemExit(f"{species} is not a held-out bird")
    return species, species, species


def bird_boundary_m(profile_key: str) -> float:
    if profile_key not in HELD_OUT_BOUNDARY_M:
        raise SystemExit(f"{profile_key} has no pinned bird boundary")
    return HELD_OUT_BOUNDARY_M[profile_key]


def bind_tier2_helpers() -> None:
    """
    Point the Tier 2 validation monitor at the bird held-out set.

    `T._make_monitor` finds a species' split through `T.attached_entry` and its travel boundary through
    `T.boundary_m`, and both only know Tier 2's held-out species. They are rebound here, as `T.RESULTS` and
    `T.CHECKPOINTS` already are, so checkpoint selection is Tier 2's own code rather than a copy of it.
    """
    check_boundaries()
    T.attached_entry = bird_attached_entry
    T.boundary_m = bird_boundary_m
    T.RESULTS, T.CHECKPOINTS = RESULTS, CHECKPOINTS


# Training data.
def held_out_train_ids(species: str, payload: dict) -> list:
    """A held-out bird's training individuals minus its registered exclusions, in manifest order."""
    train_ids = list(load_split(T.SPLITS / f"{species}_split_seed42.json").train)
    excluded = {str(i) for i in payload["training_exclusions"].get(species, [])}
    missing = sorted(excluded - {str(i) for i in train_ids})
    if missing:
        raise SystemExit(f"FAIL: exclusions {missing} are not in {species}'s training partition")
    return [i for i in train_ids if str(i) not in excluded]


def finetune_subset(run: Run, payload: dict):
    """The (k, draw) subset for Fb and Sk alike, drawn after the exclusions are removed."""
    return nested_subsets(held_out_train_ids(run.species, payload), draw=run.draw)[run.k_percent]


def pooled_source(run: Run, payload: dict) -> tuple[SourcePool, str]:
    if run.species:
        raise SystemExit(f"FAIL: {run.arm} is a pooled arm and cannot be attached to {run.species}")
    label = ARMS[run.arm]["pool"]
    node = f"probabilities.{label}"
    weights = payload["probabilities"][label]
    assert_bird_sources(weights)
    species = {stem: T._load_species_windows(stem, SPLIT_STEM.get(stem, stem), "train") for stem in sorted(weights)}
    return SourcePool.from_registered_weights(species, REGISTERED_POOL, pool=node), node


def held_out_source(run: Run, payload: dict) -> tuple[SourcePool, str, object]:
    """A held-out bird's own training pool: all of it bar its exclusions, or the registered (k, draw) subset of that."""
    allowed = HELD_OUT if run.arm == "Fb" else SPECIALIST_TARGETS
    if run.species not in allowed:
        raise SystemExit(f"FAIL: {run.arm} cannot train on {run.species!r}; its targets are {list(allowed)}")
    subset = None
    if ARMS[run.arm]["pool"] == "held_out_subset":
        subset = finetune_subset(run, payload)
        ids = list(subset.individuals)
        node = f"subset:{run.species}:k{run.k_percent}:d{run.draw}"
    else:
        ids = held_out_train_ids(run.species, payload)
        node = f"specialist:{run.species}"
    windows = T._load_species_windows(run.species, run.species, "train", ids)
    return SourcePool({run.species: windows}, {run.species: 1.0}), node, subset


def training_source(run: Run, payload: dict):
    if ARMS[run.arm]["pool"].startswith("held_out"):
        return held_out_source(run, payload)
    pool, node = pooled_source(run, payload)
    return pool, node, None


# Configuration.
def build_config(run: Run, grid_index: int, steps: int, pool_names, payload: dict) -> TrainConfig:
    """The counterpart's own configuration: Tier 2's for S and SB, Tier 3's for Sk, LSTMS and Fb, Tier 4's otherwise."""
    module, _ = ARMS[run.arm]["recipe"]
    twin = counterpart(run)
    if module is T:
        if T.tier_e_scale_floor(twin) != SCALE_FLOOR_M:
            raise RuntimeError(f"{run.arm}'s counterpart trains under a different scale floor")
        return T.build_config(twin, grid_index, steps, n_species=len(pool_names))
    return module.build_config(twin, grid_index, steps, pool_names, payload)


def initial_state(run: Run, grid_index: int) -> tuple[dict, str | None]:
    """Fb initialises from Zb seed 0; it refuses to start until that run has reported."""
    donor_id = ARMS[run.arm].get("init")
    if not donor_id:
        return {}, None
    path = RESULTS / f"{donor_id}_reported.pt"
    if not path.exists():
        raise SystemExit(f"FAIL: {run.arm} fine-tunes {donor_id}, but {path} does not exist yet; train it first")
    donor = torch.load(path, map_location="cpu", weights_only=False)
    d_model, layers, _ = IL.GRID[grid_index]
    if (donor["config"]["d_model"], donor["config"]["num_layers"]) != (d_model, layers):
        raise SystemExit(f"FAIL: {donor_id} is not at grid {grid_index}'s geometry")
    return {"init_state_dict": donor["model"]}, f"birds:{donor_id}"


# Dry run.
def dry_run(payload: dict, index: int | None) -> None:
    """Build one plan index's training data, or every pool and held-out training set, and stop before training."""
    bind_tier2_helpers()
    if index is None:
        missing = [
            str(path)
            for stem in sorted(set(payload["probabilities"]["congener"]) | set(HELD_OUT))
            for path in (IL.PROCESSED / f"{stem}_segments.parquet",
                         IL.SPLITS / f"{SPLIT_STEM.get(stem, stem)}_split_seed42.json")
            if not path.exists()
        ]
        if missing:
            raise SystemExit("FAIL: missing input files:\n  " + "\n  ".join(missing))
        for label in ("main", "congener"):
            pool, node = pooled_source(Run("Zb" if label == "main" else "Zbc", "", 0), payload)
            windows = sum(pool.species[name].n_windows for name in pool.names)
            print(f"{node:<24} {len(pool.names):>2} species, {windows:,} windows")
        for species in HELD_OUT:
            ids = held_out_train_ids(species, payload)
            windows = T._load_species_windows(species, species, "train", ids).n_windows
            print(f"{'held_out:' + species:<24} {len(ids)} individuals, {windows:,} windows")
        print(f"Fb donor         : {'present' if (RESULTS / 'Zb_s0_reported.pt').exists() else 'not yet trained'}")
        print("\ndry run only; nothing was trained and no test partition was opened.")
        return

    run = PLAN[index]
    grid_index, _ = T.selected_grid_index(None)
    print(f"index {index:>2}  {run.run_id}")
    source_pool, node, subset = training_source(run, payload)
    windows = sum(source_pool.species[name].n_windows for name in source_pool.names)
    print(f"  pool           : {node}, {len(source_pool.names)} species, {windows:,} windows")
    excluded = payload["training_exclusions"].get(run.species, [])
    if excluded:
        print(f"  excluded       : {len(excluded)} {run.species} training individuals (registered)")
    if subset is not None:
        print(f"  subset         : k={run.k_percent}% draw {run.draw}, "
              f"{subset.n_individuals} of {subset.n_available} individuals")
    if ARMS[run.arm].get("xgb"):
        sampling = "all training windows" if run.species else f"{XGB_POOLED_DRAWS:,} A8 draws"
        print(f"  model          : WindowXGBoost, {sampling}, seed {run.seed}")
        print("  dry run only; nothing trained, no test partition opened.")
        return
    config = build_config(run, grid_index, REGISTERED_STEPS, source_pool.names, payload)
    print(f"  config         : {counterpart(run).arm}'s recipe, trunk {config.trunk}, d_model {config.d_model}, "
          f"layers {config.num_layers}, lr {config.lr:g}, package {config.use_fm_package}, "
          f"recon {config.use_reconstruction_head}, selection {run.uses_validation_selection}")
    _, donor_id = initial_state(run, grid_index)  # Fb refuses here until Zb_s0 has reported
    if donor_id:
        print(f"  initialised from: {donor_id}")
    print("  dry run only; nothing trained, no test partition opened.")


# Runners.
def run_torch(run: Run, index: int, steps: int, grid_index: int, source: str, payload: dict) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    started = time.time()
    source_pool, node, subset = training_source(run, payload)
    pool = PreparedPool.from_source_pool(source_pool, device=device, sigma=SCALE_FLOOR_M)
    print(f"pool             : {pool.n_windows:,} windows over {len(pool.names)} species [{node}], "
          f"built in {time.time() - started:.0f}s", flush=True)

    config = build_config(run, grid_index, steps, pool.names, payload)
    initialisation, donor_id = initial_state(run, grid_index)
    if donor_id:
        print(f"initialised from : {donor_id}", flush=True)
    monitor, selection = T._make_monitor(run, device, grid_index, config.steps)
    result = train(
        config, pool, device=device,
        checkpoint_path=CHECKPOINTS / f"{run.run_id}_g{grid_index:02d}_{config.steps}.pt",
        progress=lambda step, loss: print(f"  step {step:>7,}  loss {loss:.5f}", flush=True),
        **monitor, **initialisation,
    )
    print(f"applied updates  : {result.applied_updates:,} of {config.steps:,}; "
          f"withheld {result.skipped_steps:,} ({result.withheld_fraction:.2%})", flush=True)
    model, selection_payload = T._resolve_reported_model(run, result, selection)

    reported = RESULTS / f"{run.run_id}_reported.pt"
    torch.save(
        {"run_id": run.run_id, "config": asdict(config), "grid_index": grid_index, "pool_species": list(pool.names),
         "selection": {k: v for k, v in selection_payload.items() if k != "history"},
         "model": model.state_dict()},
        reported,
    )
    record = base_record(run, index, grid_index, source, node, payload) | {
        "config": asdict(config),
        "n_parameters": model.n_parameters(),
        "pool_species": list(pool.names),
        "pool_windows": pool.n_windows,
        "subset": T3.subset_record(subset),
        "initialised_from": donor_id,
        "final_loss": result.final_loss,
        "loss_history": result.loss_history,
        "loss_shares": result.loss_shares,
        "skipped_steps": result.skipped_steps,
        "attempted_batches": result.attempted_batches,
        "applied_updates": result.applied_updates,
        "withheld_fraction": result.withheld_fraction,
        "nonfinite_steps": result.nonfinite_steps,
        "checkpoint_selection": selection_payload,
        "reported_checkpoint": reported.name,
    }
    write_record(run, record)


def run_xgb(run: Run, index: int, grid_index: int, source: str, payload: dict) -> None:
    from animal_movement_fm.models.xgboost import WindowXGBoost

    source_pool, node, _ = training_source(run, payload)
    if ARMS[run.arm]["pool"] == "held_out_train":
        windows = source_pool.species[run.species]
        features, targets, _ = windows.gather(np.arange(windows.n_windows, dtype=np.int64), sigma=SCALE_FLOOR_M)
        sampled = "all training windows"
    else:
        rng = np.random.default_rng(run.seed)
        features, targets, _ = source_pool.sample(rng, XGB_POOLED_DRAWS)
        sampled = f"{XGB_POOLED_DRAWS:,} A8 draws, seed {run.seed}"
    print(f"xgboost data     : {len(features):,} windows ({sampled}) [{node}]", flush=True)
    started = time.time()
    model = WindowXGBoost(random_state=run.seed, n_jobs=int(os.environ.get("PBS_NCPUS", "0")) or None)
    model.fit(features.astype(np.float32), targets.astype(np.float32))
    stem_path = RESULTS / f"{run.run_id}_xgb"
    model.save(stem_path)
    record = base_record(run, index, grid_index, source, node, payload) | {
        "model": "WindowXGBoost",
        "params": model.params,
        "training_windows": int(len(features)),
        "sampling": sampled,
        "fit_seconds": time.time() - started,
        "reported_checkpoint": f"{stem_path.name}_*.ubj",
    }
    write_record(run, record)


def base_record(run: Run, index: int, grid_index: int, source: str, node: str, payload: dict) -> dict:
    return {
        "run_id": run.run_id,
        "plan_index": index,
        "arm": run.arm,
        "recipe": counterpart(run).arm,
        "species": run.species or None,
        "seed": run.seed,
        "k_percent": run.k_percent,
        "draw": run.draw,
        "grid_index": grid_index,
        "configuration_source": source,
        "training_pool": node,
        "excluded_individuals": list(payload["training_exclusions"].get(run.species, [])),
        "tier": "birds",
        "registered_by": "preregistration_birds.md",
        "exploratory": False,
        "opened_test_partition": False,
    }


def write_record(run: Run, record: dict) -> None:
    path = RESULTS / f"{run.run_id}.json"
    path.write_text(json.dumps(record, indent=2, default=str) + "\n")
    print(f"wrote {path}", flush=True)


def print_plan(plan: list[Run]) -> None:
    print(f"{'idx':>4}  {'arm':<7} {'species':<22} {'seed':>4} {'k':>5} {'draw':>5}")
    for i, run in enumerate(plan):
        k = "-" if run.k_percent is None else f"{run.k_percent}%"
        d = "-" if run.draw is None else run.draw
        print(f"{i:>4}  {run.arm:<7} {run.species or '-':<22} {run.seed:>4} {k:>5} {d:>5}")
    print(f"\n{len(plan)} bird-study runs (0-{len(plan) - 1}).")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=int, default=None, help="plan index; defaults to $PBS_ARRAY_INDEX")
    parser.add_argument("--steps", type=int, default=None, help="optimiser updates; must be the registered value")
    parser.add_argument("--list", action="store_true", help="print the plan and exit")
    parser.add_argument("--dry-run", action="store_true", help="build the data for --index, or every pool, and exit")
    args = parser.parse_args()

    if args.list:
        return print_plan(PLAN)

    raw = args.index if args.index is not None else os.environ.get("PBS_ARRAY_INDEX")
    index = int(raw) if raw is not None else None
    if index is not None and not 0 <= index < len(PLAN):
        raise SystemExit(f"FAIL: index {index} is outside 0..{len(PLAN) - 1}; see --list")

    if args.dry_run:
        return dry_run(load_registered_pool(), index)

    if index is None:
        raise SystemExit("FAIL: no --index and no $PBS_ARRAY_INDEX")
    run = PLAN[index]
    if not ARMS[run.arm].get("xgb") and args.steps != REGISTERED_STEPS:
        raise SystemExit(f"FAIL: --steps must be {REGISTERED_STEPS:,} for this plan; got {args.steps}")

    payload = load_registered_pool()
    bind_tier2_helpers()
    grid_index, source = T.selected_grid_index(None)
    RESULTS.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    print(f"run              : {run.run_id}  (plan index {index} of {len(PLAN) - 1})", flush=True)
    print(f"configuration    : grid {grid_index} from {source}", flush=True)
    if ARMS[run.arm].get("xgb"):
        run_xgb(run, index, grid_index, source, payload)
    else:
        run_torch(run, index, REGISTERED_STEPS, grid_index, source, payload)


if __name__ == "__main__":
    main()
