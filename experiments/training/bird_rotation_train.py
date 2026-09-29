# GenAI reference: Again, I specified the driver for this rotating bird study, which essentially was to mport bird_train and
# take its arm recipes (Bb, LSTM-Bb, XGB-Bb and Zb pooled per fold; LSTM-S and XGB-S per held-out bird); the append only 198 
# run plan with Zb last; fold pools/A8 weights come from our registered rotation file and are checked against the pinned values,
# exclusions apply to both pooled and specialist training, and all 27 held-out boundaries are pinned/checked against the profile 
# tables. Seeds are 0-2 except XGBoost at seed 0, and we have to tag runs as exploratory. GenAI coded that; I checked it. 
"""
Train the rotating held-out bird study's arms, one model per PBS subjob (exploratory).

Fold 0 is the bird study's gull, goose and dove, carried as reported; nothing is trained for it here. Will finish rest of docs later. 
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

import bird_train as BT  # noqa: E402

from animal_movement_fm.data.enu import SCALE_FLOOR_M  # noqa: E402
from animal_movement_fm.data.multispecies import SourcePool  # noqa: E402
from animal_movement_fm.data.split import id_key, load_split  # noqa: E402
from animal_movement_fm.train.loop import train  # noqa: E402
from animal_movement_fm.train.pool import PreparedPool  # noqa: E402

T, IL, SPW = BT.T, BT.IL, BT.SPW

RESULTS = REPO / "results" / "birds_rotation"
CHECKPOINTS = REPO / "checkpoints" / "birds_rotation"
REGISTERED_FOLDS = SPW.BIRDS_ROTATION_OUT
FREEZE = "birds_rotation_exploratory_freeze.md"

REGISTERED_STEPS = BT.REGISTERED_STEPS  # 20,000 applied updates at batch 256, as the bird study
SEEDS = (0, 1, 2)
XGB_SEED = 0
XGB_POOLED_DRAWS = BT.XGB_POOLED_DRAWS

# folds 1-9; fold 0 is carried from the bird study
FOLDS: dict[int, tuple[str, ...]] = {k: tuple(sorted(v)) for k, v in SPW.BIRDS_ROTATION_FOLDS.items() if k != 0}
HELD_OUT: tuple[str, ...] = tuple(name for k in sorted(FOLDS) for name in FOLDS[k])  # fold order, then name
FOLD_OF: dict[str, int] = {name: k for k, members in FOLDS.items() for name in members}
if len(HELD_OUT) != 27 or set(HELD_OUT) & set(BT.HELD_OUT):
    raise RuntimeError("the rotation holds out 27 birds, none of them the bird study's three")

# A4 travel boundaries (0.1 x p50 on all training individuals, before exclusions), pinned at full precision and checked
# on every load. Stork is keyed `white_stork` in the profile tables and in registered/tier2_regime_decision.json.
HELD_OUT_BOUNDARY_M: dict[str, float] = {
    "anas_platyrhynchos": 14.6855758969494,
    "anser_anser": 40.826409151579725,
    "anser_brachyrhynchus": 137.67519816784366,
    "anser_caerulescens": 118.008819071936,
    "anser_fabalis": 62.80675583469281,
    "ardea_alba": 61.722061715992574,
    "branta_canadensis": 36.46574202019987,
    "branta_leucopsis": 101.2416336618018,
    "cathartes_aura": 74.88568358042241,
    "ciconia_ciconia": 61.10385331506313,
    "corvus_corone": 17.330279685696198,
    "diomedea_exulans": 6654.988554189454,
    "falco_naumanni": 204.4322645743916,
    "falco_tinnunculus": 42.85179513629322,
    "gyps_himalayensis": 304.60498561861414,
    "haematopus_ostralegus": 138.80780851282964,
    "hydroprogne_caspia": 384.26945981605263,
    "larus_argentatus": 153.0372216734714,
    "milvus_milvus": 65.67025743079682,
    "morus_bassanus": 1361.041002298103,
    "mycteria_americana": 59.15078671522125,
    "pelecanus_occidentalis": 385.5659078570528,
    "pernis_apivorus": 43.04533963381915,
    "phoenicopterus_roseus": 33.698446521914825,
    "platalea_leucorodia": 60.95459614377669,
    "thalassarche_chrysostoma": 4399.306296407899,
    "thalassoica_antarctica": 4588.27414508472,
}
PROFILE_KEY: dict[str, str] = {name: SPW.TIER4_PROFILE_KEY.get(name, name) for name in HELD_OUT}
MAIN_SNAPSHOT = SPW.BIRDS_RESULTS / "regime_profiles_main_snapshot.csv"  # never Gadi's results/regime_profiles.csv
BIRD_PROFILES = SPW.BIRDS_RESULTS / "regime_profiles.csv"
if set(HELD_OUT_BOUNDARY_M) != set(HELD_OUT):
    raise RuntimeError("the pinned boundaries do not cover exactly the 27 held-out birds")

# arm -> the bird study's recipe for it; `pool` is `fold` (the run's fold pool) or the held-out bird's own training birds
ARMS: dict[str, dict] = {
    arm: {**BT.ARMS[arm], "pool": "fold" if BT.ARMS[arm]["pool"] == "main" else "held_out_train"}
    for arm in ("Bb", "LSTMBb", "XGBBb", "LSTMS", "XGBS", "Zb")
}


@dataclass(frozen=True)
class Run:
    arm: str
    fold: int
    species: str
    seed: int

    # bird_train.counterpart reads these; nothing in the rotation is a subset run
    k_percent = None
    draw = None

    @property
    def run_id(self) -> str:
        return f"{self.arm}_{self.species}_s{self.seed}" if self.species else f"{self.arm}_f{self.fold}_s{self.seed}"

    @property
    def uses_validation_selection(self) -> bool:
        return BT.counterpart(self).uses_validation_selection

    @property
    def deferred(self) -> bool:  # read by nothing here; kept for the Tier 2 helpers' interface
        return False


def build_plan() -> list[Run]:
    """The rotation plan. Position is the PBS array index: append, never insert or reorder."""
    plan = []
    for arm in ("Bb", "LSTMBb"):
        plan += [Run(arm, k, "", s) for k in sorted(FOLDS) for s in SEEDS]
    plan += [Run("XGBBb", k, "", XGB_SEED) for k in sorted(FOLDS)]
    plan += [Run("LSTMS", FOLD_OF[sp], sp, s) for sp in HELD_OUT for s in SEEDS]
    plan += [Run("XGBS", FOLD_OF[sp], sp, XGB_SEED) for sp in HELD_OUT]
    plan += [Run("Zb", k, "", s) for k in sorted(FOLDS) for s in SEEDS]
    return plan


PLAN = build_plan()
if len(PLAN) != 198:
    raise RuntimeError(f"the rotation plan has {len(PLAN)} runs; the freeze records 198")


# Registered folds.
def load_registered_folds() -> dict:
    """Read the registered folds and check them against the constants source_pool_weights.py wrote them from."""
    payload = json.loads(REGISTERED_FOLDS.read_text())
    name = REGISTERED_FOLDS.name
    folds = {int(k): list(v) for k, v in payload["folds"].items()}
    if folds != {k: sorted(v) for k, v in SPW.BIRDS_ROTATION_FOLDS.items()}:
        raise SystemExit(f"FAIL: {name} folds are {folds}; the pinned folds differ")
    if payload["train_window_counts"] != SPW.BIRDS_ROTATION_TRAIN_WINDOWS:
        raise SystemExit(f"FAIL: {name} training-window counts differ from the pinned constants")
    exclusions = {species: list(ids) for species, ids in SPW.BIRDS_ROTATION_TRAINING_EXCLUSIONS.items()}
    if payload["training_exclusions"] != exclusions:
        raise SystemExit(f"FAIL: {name} training_exclusions differ from source_pool_weights.py's")
    birds = sorted(payload["birds"])
    pools = {int(k): v for k, v in payload["pools"].items()}
    if pools != {k: SPW.rotation_pool(members, birds) for k, members in folds.items()}:
        raise SystemExit(f"FAIL: {name} pools do not follow the genus-mates-out rule")
    SPW.refuse_rotation_leaks(folds, pools, birds)
    recomputed = SPW.rotation_probabilities(payload["pools"], payload["train_window_counts"])
    drift = sorted(
        f"{k}.{n}" for k, probs in recomputed.items() for n, p in probs.items()
        if abs(payload["probabilities"].get(k, {}).get(n, float("nan")) - p) > 1e-12
    )
    if drift or set(payload["probabilities"]) != set(recomputed):
        raise SystemExit(f"FAIL: {name} probabilities no longer follow from its counts: {drift or 'folds differ'}")
    return payload


def assert_fold_sources(fold: int, stems) -> None:
    """A fold's pooled arms draw from its registered pool only: never its own bird, a genus-mate or a reserved one."""
    stems = set(stems)
    members = set(FOLDS[fold])
    genera = {SPW.genus(n) for n in members}
    leaks = sorted(n for n in stems if n in members or SPW.genus(n) in genera or n in SPW.BIRDS_RESERVED)
    if leaks:
        raise SystemExit(f"FAIL: fold {fold}'s pool would draw {leaks}, a held-out bird, genus-mate or reserved species")
    unknown = stems - set(SPW.BIRDS_ROTATION_TRAIN_WINDOWS)
    if unknown:
        raise SystemExit(f"FAIL: {sorted(unknown)} is not one of the 30 rotation birds")


# Boundaries, and the Tier 2 helpers pointed at the rotation's held-out set.
def check_boundaries() -> dict[str, float]:
    """Fail unless each pinned boundary matches its profile row, the frozen A4 rule and any registered value."""
    frozen = json.loads(IL.REGISTERED_DECISION.read_text())
    if int(frozen["anchor_quantile"]) != 50:
        raise SystemExit("FAIL: the frozen A4 rule is no longer anchored on p50")
    multiplier = float(frozen["multiplier"])
    snapshot = pd.read_csv(MAIN_SNAPSHOT).set_index("species")
    birds = pd.read_csv(BIRD_PROFILES).set_index("species")
    for species, pinned in HELD_OUT_BOUNDARY_M.items():
        key = PROFILE_KEY[species]
        tables = [table for table in (snapshot, birds) if key in table.index]
        if len(tables) != 1:
            raise SystemExit(f"FAIL: {key} has {len(tables)} profile rows across the two tables, not one")
        row = tables[0].loc[key]
        if str(row["split_part"]) != "train":
            raise SystemExit(f"FAIL: {key}'s profile row is not computed on training individuals")
        values = {
            "median_anchor_boundary_m": float(row["median_anchor_boundary_m"]),
            f"{multiplier:g} x step_p50_m": multiplier * float(row["step_p50_m"]),
        }
        if "selected_boundary_m" in row.index and not pd.isna(row["selected_boundary_m"]):
            values["selected_boundary_m"] = float(row["selected_boundary_m"])
        if key in frozen["tier2_boundaries_m"]:
            values["registered"] = float(frozen["tier2_boundaries_m"][key])
        for label, value in values.items():
            if not math.isclose(value, pinned, rel_tol=1e-9, abs_tol=1e-9):
                raise SystemExit(f"FAIL: {species} boundary is pinned at {pinned!r} m but {label} gives {value!r} m")
    return dict(HELD_OUT_BOUNDARY_M)


def rotation_attached_entry(species: str) -> tuple[str, str, str]:
    """(segments stem, split stem, profile key) for a held-out bird; stork's split and profile key differ."""
    if species not in HELD_OUT:
        raise SystemExit(f"{species} is not a rotation held-out bird")
    return species, SPW.SPLIT_STEM.get(species, species), PROFILE_KEY[species]


def rotation_boundary_m(profile_key: str) -> float:
    by_key = {PROFILE_KEY[name]: value for name, value in HELD_OUT_BOUNDARY_M.items()}
    if profile_key not in by_key:
        raise SystemExit(f"{profile_key} has no pinned rotation boundary")
    return by_key[profile_key]


def bind_tier2_helpers() -> None:
    """Point the Tier 2 validation monitor at the rotation's held-out birds, as bird_train does for its three."""
    check_boundaries()
    T.attached_entry = rotation_attached_entry
    T.boundary_m = rotation_boundary_m
    T.RESULTS, T.CHECKPOINTS = RESULTS, CHECKPOINTS


# Training data.
def training_ids(stem: str, payload: dict) -> list:
    """One bird's training individuals minus its registered exclusions, in manifest order."""
    train = list(load_split(T.SPLITS / f"{SPW.SPLIT_STEM.get(stem, stem)}_split_seed42.json").train)
    excluded = {id_key(i) for i in payload["training_exclusions"].get(stem, [])}
    missing = sorted(excluded - {id_key(i) for i in train})
    if missing:
        raise SystemExit(f"FAIL: exclusions {missing} are not in {stem}'s training partition")
    return [i for i in train if id_key(i) not in excluded]


def species_windows(stem: str, payload: dict):
    return T._load_species_windows(stem, SPW.SPLIT_STEM.get(stem, stem), "train", training_ids(stem, payload))


def check_window_counts(pool: SourcePool, payload: dict) -> None:
    """Every species' built windows equal its registered count, which is counted without its exclusions."""
    drift = [
        f"{name}: {pool.species[name].n_windows:,} built, {payload['train_window_counts'][name]:,} registered"
        for name in pool.names if pool.species[name].n_windows != payload["train_window_counts"][name]
    ]
    if drift:
        raise SystemExit("FAIL: training windows differ from the registered counts\n  " + "\n  ".join(drift))


def pooled_source(run: Run, payload: dict) -> tuple[SourcePool, str]:
    if run.species:
        raise SystemExit(f"FAIL: {run.arm} is a pooled arm and cannot be attached to {run.species}")
    if run.fold not in FOLDS:
        raise SystemExit(f"FAIL: fold {run.fold} is not one of the rotation's trained folds {sorted(FOLDS)}")
    node = f"probabilities.{run.fold}"
    weights = payload["probabilities"][str(run.fold)]
    assert_fold_sources(run.fold, weights)
    species = {stem: species_windows(stem, payload) for stem in sorted(weights)}
    return SourcePool.from_registered_weights(species, REGISTERED_FOLDS, pool=node), node


def held_out_source(run: Run, payload: dict) -> tuple[SourcePool, str]:
    """A held-out bird's own training individuals, bar its exclusions."""
    if run.species not in HELD_OUT:
        raise SystemExit(f"FAIL: {run.arm} cannot train on {run.species!r}; it is not a rotation held-out bird")
    if FOLD_OF[run.species] != run.fold:
        raise SystemExit(f"FAIL: {run.species} is held out in fold {FOLD_OF[run.species]}, not {run.fold}")
    return SourcePool({run.species: species_windows(run.species, payload)}, {run.species: 1.0}), \
        f"specialist:{run.species}"


def training_source(run: Run, payload: dict) -> tuple[SourcePool, str]:
    return pooled_source(run, payload) if ARMS[run.arm]["pool"] == "fold" else held_out_source(run, payload)


def excluded_in(pool_names, payload: dict) -> dict[str, list]:
    return {name: payload["training_exclusions"][name] for name in pool_names if name in payload["training_exclusions"]}


# Dry run.
def dry_run(payload: dict, index: int | None) -> None:
    """Build one plan index's training data, or every bird's, and stop before training."""
    bind_tier2_helpers()
    if index is None:
        missing = [
            str(path)
            for stem in payload["birds"]
            for path in (IL.PROCESSED / f"{stem}_segments.parquet",
                         IL.SPLITS / f"{SPW.SPLIT_STEM.get(stem, stem)}_split_seed42.json")
            if not path.exists()
        ]
        if missing:
            raise SystemExit("FAIL: missing input files:\n  " + "\n  ".join(missing))
        built = {stem: species_windows(stem, payload) for stem in payload["birds"]}
        check_window_counts(SourcePool(built, {stem: 1.0 for stem in built}), payload)
        for k in sorted(FOLDS):
            weights = payload["probabilities"][str(k)]
            assert_fold_sources(k, weights)
            print(f"fold {k}: pool {len(weights)} birds, {sum(built[n].n_windows for n in weights):,} windows; "
                  f"held out {', '.join(FOLDS[k])}")
        print(f"\n{len(built)} birds built, every count matching the registered file after exclusions.")
        print("dry run only; nothing was trained and no test partition was opened.")
        return

    run = PLAN[index]
    grid_index, _ = T.selected_grid_index(None)
    print(f"index {index:>3}  {run.run_id}  (fold {run.fold})")
    source_pool, node = training_source(run, payload)
    check_window_counts(source_pool, payload)
    windows = sum(source_pool.species[name].n_windows for name in source_pool.names)
    print(f"  pool           : {node}, {len(source_pool.names)} species, {windows:,} windows")
    for name, ids in excluded_in(source_pool.names, payload).items():
        print(f"  excluded       : {len(ids)} {name} training individuals (registered)")
    if ARMS[run.arm].get("xgb"):
        sampling = "all training windows" if run.species else f"{XGB_POOLED_DRAWS:,} A8 draws"
        print(f"  model          : WindowXGBoost, {sampling}, seed {run.seed}")
        print("  dry run only; nothing trained, no test partition opened.")
        return
    config = BT.build_config(run, grid_index, REGISTERED_STEPS, source_pool.names, payload)
    print(f"  config         : {BT.counterpart(run).arm}'s recipe, trunk {config.trunk}, d_model {config.d_model}, "
          f"layers {config.num_layers}, lr {config.lr:g}, package {config.use_fm_package}, "
          f"recon {config.use_reconstruction_head}, selection {run.uses_validation_selection}")
    print("  dry run only; nothing trained, no test partition opened.")


# Runners.
def run_torch(run: Run, index: int, steps: int, grid_index: int, source: str, payload: dict) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    started = time.time()
    source_pool, node = training_source(run, payload)
    check_window_counts(source_pool, payload)
    pool = PreparedPool.from_source_pool(source_pool, device=device, sigma=SCALE_FLOOR_M)
    print(f"pool             : {pool.n_windows:,} windows over {len(pool.names)} species [{node}], "
          f"built in {time.time() - started:.0f}s", flush=True)

    config = BT.build_config(run, grid_index, steps, pool.names, payload)
    monitor, selection = T._make_monitor(run, device, grid_index, config.steps)
    result = train(
        config, pool, device=device,
        checkpoint_path=CHECKPOINTS / f"{run.run_id}_g{grid_index:02d}_{config.steps}.pt",
        progress=lambda step, loss: print(f"  step {step:>7,}  loss {loss:.5f}", flush=True),
        **monitor,
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
    record = base_record(run, index, grid_index, source, node, source_pool.names, payload) | {
        "config": asdict(config),
        "n_parameters": model.n_parameters(),
        "pool_species": list(pool.names),
        "pool_windows": pool.n_windows,
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

    source_pool, node = training_source(run, payload)
    check_window_counts(source_pool, payload)
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
    record = base_record(run, index, grid_index, source, node, source_pool.names, payload) | {
        "model": "WindowXGBoost",
        "params": model.params,
        "training_windows": int(len(features)),
        "sampling": sampled,
        "fit_seconds": time.time() - started,
        "reported_checkpoint": f"{stem_path.name}_*.ubj",
    }
    write_record(run, record)


def base_record(run: Run, index: int, grid_index: int, source: str, node: str, pool_names, payload: dict) -> dict:
    return {
        "run_id": run.run_id,
        "plan_index": index,
        "arm": run.arm,
        "recipe": BT.counterpart(run).arm,
        "fold": run.fold,
        "species": run.species or None,
        "seed": run.seed,
        "grid_index": grid_index,
        "configuration_source": source,
        "training_pool": node,
        "excluded_individuals": excluded_in(pool_names, payload),
        "tier": "birds_rotation",
        "registered_by": FREEZE,
        "exploratory": True,
        "opened_test_partition": False,
    }


def write_record(run: Run, record: dict) -> None:
    path = RESULTS / f"{run.run_id}.json"
    path.write_text(json.dumps(record, indent=2, default=str) + "\n")
    print(f"wrote {path}", flush=True)


def print_plan(plan: list[Run]) -> None:
    print(f"{'idx':>4}  {'arm':<7} {'fold':>4}  {'species':<26} {'seed':>4}")
    for i, run in enumerate(plan):
        print(f"{i:>4}  {run.arm:<7} {run.fold:>4}  {run.species or '-':<26} {run.seed:>4}")
    print(f"\n{len(plan)} rotation runs (0-{len(plan) - 1}).")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=int, default=None, help="plan index; defaults to $PBS_ARRAY_INDEX")
    parser.add_argument("--steps", type=int, default=None, help="optimiser updates; must be the registered value")
    parser.add_argument("--list", action="store_true", help="print the plan and exit")
    parser.add_argument("--dry-run", action="store_true", help="build the data for --index, or every bird, and exit")
    args = parser.parse_args()

    if args.list:
        return print_plan(PLAN)

    raw = args.index if args.index is not None else os.environ.get("PBS_ARRAY_INDEX")
    index = int(raw) if raw is not None else None
    if index is not None and not 0 <= index < len(PLAN):
        raise SystemExit(f"FAIL: index {index} is outside 0..{len(PLAN) - 1}; see --list")

    if args.dry_run:
        return dry_run(load_registered_folds(), index)

    if index is None:
        raise SystemExit("FAIL: no --index and no $PBS_ARRAY_INDEX")
    run = PLAN[index]
    if not ARMS[run.arm].get("xgb") and args.steps != REGISTERED_STEPS:
        raise SystemExit(f"FAIL: --steps must be {REGISTERED_STEPS:,} for this plan; got {args.steps}")

    payload = load_registered_folds()
    bind_tier2_helpers()
    grid_index, source = T.selected_grid_index(None)
    RESULTS.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    print(f"run              : {run.run_id}  (plan index {index} of {len(PLAN) - 1}, fold {run.fold})", flush=True)
    print(f"configuration    : grid {grid_index} from {source}", flush=True)
    if ARMS[run.arm].get("xgb"):
        run_xgb(run, index, grid_index, source, payload)
    else:
        run_torch(run, index, REGISTERED_STEPS, grid_index, source, payload)


if __name__ == "__main__":
    main()
