# GenAI reference: I used GenAI to draft this file, based off my specification and the previous Tier training file. I gave it my 
# preregistration_tier3.md which records my decisions (arms, pools, budgets, selection rules, the baseline ladder, the reserved 
# species). Obviously it reuses the Tier 2 machinery in tier2_final_train.py and tier2_inner_loop.py rather than copying it. 
# I review it before any run.
"""
Train the registered Tier 3 arms, one model per PBS subjob (`preregistration_tier3.md` sec 3 and 3.1).

Note that this script never opens a test partition. Pooled arms can only draw from the nine registered sources, never from a held-out 
or reserved species. Specialists and fine-tuned arms use a held-out species' training individuals, and select checkpoints on its 
validation individuals, on S's registered terms.
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import tier2_final_train as T  # noqa: E402
import tier2_inner_loop as IL  # noqa: E402

from animal_movement_fm.data.enu import SCALE_FLOOR_M  # noqa: E402
from animal_movement_fm.data.multispecies import SourcePool  # noqa: E402
from animal_movement_fm.train.loop import TrainConfig, train  # noqa: E402
from animal_movement_fm.train.pool import PreparedPool  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results" / "tier3"
CHECKPOINTS = REPO / "checkpoints" / "tier3"
REGISTERED_POOL = REPO / "registered" / "tier3_source_pool.json"

REGISTERED_STEPS = 20_000  # preregistration_tier3.md sec 3; the Tier 2 budget, held fixed
TIER_E_LONG_STEPS = 40_000  # the budget check, Tier E only
SEEDS = T.SEEDS
DOMAINS = ("flying", "marine", "terrestrial")  # lookup-row order for B9dom

# preregistration_tier3.md sec 1.1 as corrected 20-09-2026: the nine sources and their A8 draw probabilities
# in percent, to three decimals; checked against the registered file on every load, so the guard compares against constants, 
# not another written file
REGISTERED_SOURCE_PERCENT = {
    "ciconia_ciconia": 32.601,
    "branta_canadensis": 11.463,
    "thalassoica_antarctica": 1.838,
    "cervus_elaphus": 27.081,
    "bison_bison": 7.051,
    "canis_latrans": 7.006,
    "mirounga_leonina": 8.363,
    "arctocephalus_gazella": 2.671,
    "eudyptes_chrysolophus": 1.927,
}
# preregistration_tier3.md sec 2: withheld from every Tier 3 and Tier 4 run
RESERVED = frozenset({"anser_albifrons", "ovis_dalli", "pygoscelis_adeliae"})
# the one source whose split manifest stem differs from its segments stem
SPLIT_STEM = {"ciconia_ciconia": "stork"}

XGB_POOLED_DRAWS = 1_000_000  # sec 3.1
LSTM_D_MODEL, LSTM_LAYERS = 64, 1  # Tier 1's settled LSTM (methods.md), not tuned
SK_FRACTIONS = (1, 10)

# arm -> how it is built. `pool` names the training data; `per_species` attaches the arm to each held-out species.
ARMS: dict[str, dict] = {
    "Z9": {"pool": "source_pool", "package": True, "recon": True},
    "B9": {"pool": "source_pool", "package": False, "recon": False},
    "Za9": {"pool": "without_domain", "package": True, "recon": True, "per_species": True},
    "Zd3": {"pool": "domain", "package": True, "recon": True, "per_species": True},
    "Zd1": {"pool": "tier2_domain_source", "package": True, "recon": True, "per_species": True},
    "B9dom": {"pool": "source_pool", "package": True, "recon": False, "context": False, "domain_embed": 16},
    "LSTMS": {"pool": "held_out_train", "trunk": "lstm", "per_species": True, "selection": True},
    "LSTMB3": {"pool": "tier2_source_pool", "trunk": "lstm"},
    "LSTMB9": {"pool": "source_pool", "trunk": "lstm"},
    "Sk": {"pool": "held_out_subset", "package": True, "recon": True, "per_species": True, "selection": True},
    "F3c": {"pool": "held_out_subset", "package": True, "recon": True, "per_species": True, "selection": True,
            "init": ("tier2", "Z_s0")},
    "F9": {"pool": "held_out_subset", "package": True, "recon": True, "per_species": True, "selection": True,
           "init": ("tier3", "Z9_s0")},
    "XGBS": {"pool": "held_out_train", "xgb": True, "per_species": True},
    "XGBB3": {"pool": "tier2_source_pool", "xgb": True},
    "XGBB9": {"pool": "source_pool", "xgb": True},
    # Tier E only
    "Z9long": {"pool": "source_pool", "package": True, "recon": True, "tier_e": True},
}


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
        return bool(ARMS[self.arm].get("selection"))

    @property
    def deferred(self) -> bool:  # read by nothing here; kept for the Tier 2 helpers' interface
        return False


HELD_OUT = tuple(entry[0] for entry in T.HELD_OUT)


def build_plan() -> list[Run]:
    """The registered Tier 3 plan. Position is the PBS array index: append, never insert or reorder."""
    plan = [Run("Z9", "", s) for s in SEEDS] + [Run("B9", "", s) for s in SEEDS]
    for arm in ("Za9", "Zd3", "Zd1"):
        plan += [Run(arm, sp, s) for sp in HELD_OUT for s in SEEDS]
    plan += [Run("B9dom", "", s) for s in SEEDS]
    plan += [Run("LSTMS", sp, s) for sp in HELD_OUT for s in SEEDS]
    plan += [Run("LSTMB3", "", s) for s in SEEDS] + [Run("LSTMB9", "", s) for s in SEEDS]
    plan += [
        Run("Sk", sp, 0, k, d)
        for sp in HELD_OUT
        for k in SK_FRACTIONS
        for d in range(T.FINETUNE_DRAWS_PER_FRACTION[k])
    ]
    for arm in ("F3c", "F9"):
        plan += [
            Run(arm, sp, T.FINETUNE_INIT_SEED, k, d)
            for sp in HELD_OUT
            for k, draws in T.FINETUNE_DRAWS_PER_FRACTION.items()
            for d in range(draws)
        ]
    plan += [Run("XGBS", sp, 0) for sp in HELD_OUT]
    plan += [Run("XGBB3", "", s) for s in SEEDS] + [Run("XGBB9", "", s) for s in SEEDS]
    return plan


PLAN = build_plan()
TIER_E_PLAN = [Run("Z9long", "", s) for s in SEEDS]
if len(PLAN) != 158:
    raise RuntimeError(f"the registered Tier 3 plan has {len(PLAN)} runs; preregistration_tier3.md registers 158")


# Registered pool. 
def load_registered_pool() -> dict:
    """Read the registered Tier 3 pool and check it against the preregistered constants."""
    payload = json.loads(REGISTERED_POOL.read_text())
    got = {name: round(100.0 * p, 3) for name, p in payload["source_pool"].items()}
    if got != REGISTERED_SOURCE_PERCENT:
        raise SystemExit(f"FAIL: {REGISTERED_POOL.name} source_pool is {got}; sec 1.1 registers {REGISTERED_SOURCE_PERCENT}")
    if "sub_pools" not in payload:
        raise SystemExit(
            f"FAIL: {REGISTERED_POOL.name} has no sub_pools. Run `source_pool_weights.py --tier3` once to register them."
        )
    return payload


def domain_of(payload: dict) -> dict[str, str]:
    domains = dict(payload["fixed"])
    for domain, added in payload["added"].items():
        domains.update({name: domain for name in added})
    return domains


def assert_tier3_sources(stems) -> None:
    """Pooled Tier 3 arms draw only from the registered nine; never a held-out or reserved species."""
    stems = set(stems)
    for label, forbidden in (("held-out", set(HELD_OUT)), ("reserved", RESERVED)):
        if stems & forbidden:
            raise SystemExit(f"FAIL: {sorted(stems & forbidden)} is a {label} species and cannot enter a source pool")
    unknown = stems - set(REGISTERED_SOURCE_PERCENT)
    if unknown:
        raise SystemExit(f"FAIL: {sorted(unknown)} is not a registered Tier 3 source")


def pooled_source(run: Run, payload: dict) -> tuple[SourcePool, str]:
    """The registered SourcePool for a pooled arm, with its node name."""
    spec = ARMS[run.arm]
    held = T.held_out_entry(run.species) if run.species else None
    if spec["pool"] == "tier2_source_pool":
        stems = [stem for stem, *_ in T.SOURCES]
        T._assert_source_only(stems)
        species = {stem: T._load_species_windows(stem, split, "train") for stem, split, *_ in T.SOURCES}
        return IL.load_source_pool(species, pool="source_pool"), "tier2:source_pool"
    if spec["pool"] == "tier2_domain_source":
        stem = held[4]
        T._assert_source_only([stem])
        windows = T._load_species_windows(stem, SPLIT_STEM.get(stem, stem), "train")
        return SourcePool({stem: windows}, {stem: 1.0}), f"tier2_domain_source:{stem}"

    if spec["pool"] == "source_pool":
        node = "source_pool"
    elif spec["pool"] == "without_domain":
        node = f"sub_pools.without_{held[3]}"
    elif spec["pool"] == "domain":
        node = f"sub_pools.domain_{held[3]}"
    else:
        raise SystemExit(f"FAIL: {run.arm} is not a pooled arm")
    weights = payload
    for part in node.split("."):
        weights = weights[part]
    assert_tier3_sources(weights)
    species = {
        stem: T._load_species_windows(stem, SPLIT_STEM.get(stem, stem), "train") for stem in sorted(weights)
    }
    return SourcePool.from_registered_weights(species, REGISTERED_POOL, pool=node), node


def held_out_source(run: Run) -> tuple[SourcePool, str, object]:
    """A held-out species' own training pool: all of it, or F's registered subset."""
    stem, split_stem, *_ = T.held_out_entry(run.species)
    subset = None
    if ARMS[run.arm]["pool"] == "held_out_subset":
        subset = T.finetune_subset(run)
        windows = T._load_species_windows(stem, split_stem, "train", subset.individuals)
        node = f"subset:{stem}:k{run.k_percent}:d{run.draw}"
    else:
        windows = T._load_species_windows(stem, split_stem, "train")
        node = f"specialist:{stem}"
    return SourcePool({stem: windows}, {stem: 1.0}), node, subset


def training_source(run: Run, payload: dict):
    if ARMS[run.arm]["pool"].startswith("held_out"):
        return held_out_source(run)
    pool, node = pooled_source(run, payload)
    return pool, node, None


# Configuration. 
def build_config(run: Run, grid_index: int, steps: int, pool_names, payload: dict) -> TrainConfig:
    spec = ARMS[run.arm]
    d_model, layers, lr = IL.GRID[grid_index]
    if spec.get("trunk") == "lstm":
        return TrainConfig(
            d_model=LSTM_D_MODEL, num_layers=LSTM_LAYERS, lr=lr, steps=steps, seed=run.seed,
            use_fm_package=False, use_reconstruction_head=False, trunk="lstm",
        )
    extra = {}
    if spec.get("domain_embed"):
        domains = domain_of(payload)
        extra = {
            "use_context_conditioning": False,
            "species_embed_dim": spec["domain_embed"],
            "n_species": len(DOMAINS),
            "species_label_map": tuple(DOMAINS.index(domains[name]) for name in pool_names),
        }
    config = TrainConfig(
        d_model=d_model, num_layers=layers, lr=lr, steps=steps, seed=run.seed,
        use_fm_package=spec["package"], use_reconstruction_head=spec["recon"], **extra,
    )
    # The registered arms keep Tier 2's package definitions exactly. 
    if run.arm in ("Z9", "Za9", "Zd3", "Zd1", "Sk", "F3c", "F9", "Z9long"):
        if not (config.use_fm_package and config.use_reconstruction_head and config.use_context_conditioning):
            raise RuntimeError(f"{run.arm} must carry the complete registered FM package")
    if run.arm == "B9" and (config.use_fm_package or config.use_reconstruction_head):
        raise RuntimeError("B9 must be the trunk with the FM package stripped")
    return config


def initial_state(run: Run, grid_index: int) -> tuple[dict, str | None]:
    init = ARMS[run.arm].get("init")
    if not init:
        return {}, None
    where, donor_id = init
    directory = T.TIER2_FINAL_RESULTS if where == "tier2" else RESULTS
    path = directory / f"{donor_id}_reported.pt"
    if not path.exists():
        raise SystemExit(f"FAIL: {run.arm} fine-tunes {donor_id}, but {path} does not exist yet; train it first")
    donor = torch.load(path, map_location="cpu", weights_only=False)
    d_model, layers, _ = IL.GRID[grid_index]
    if (donor["config"]["d_model"], donor["config"]["num_layers"]) != (d_model, layers):
        raise SystemExit(f"FAIL: {donor_id} is not at grid {grid_index}'s geometry")
    return {"init_state_dict": donor["model"]}, f"{where}:{donor_id}"


# Dry run.
def dry_run(payload: dict) -> None:
    """
    Build every registered Tier 3 pool and stop before training.

    A missing or misnamed segments file or split manifest is otherwise found inside the first queued job, after the
    queue wait. This checks the files, builds the nine-source pool and each sub-pool from the registered weights and
    reports the window counts, so the data side of a 158-run submission is verified in one short job.
    """
    wanted = [(stem, SPLIT_STEM.get(stem, stem)) for stem in sorted(REGISTERED_SOURCE_PERCENT)]
    held_out = [(stem, split_stem) for stem, split_stem, *_ in T.HELD_OUT]
    missing = [
        str(path)
        for stem, split_stem in wanted + held_out
        for path in (IL.PROCESSED / f"{stem}_segments.parquet", IL.SPLITS / f"{split_stem}_split_seed42.json")
        if not path.exists()
    ]
    if missing:
        raise SystemExit("FAIL: missing input files:\n  " + "\n  ".join(missing))
    print(f"files            : {len(wanted + held_out)} species, segments and splits all present")

    species = {stem: T._load_species_windows(stem, split_stem, "train") for stem, split_stem in wanted}
    for node in ["source_pool"] + [f"sub_pools.{name}" for name in sorted(payload["sub_pools"])]:
        weights = payload
        for part in node.split("."):
            weights = weights[part]
        assert_tier3_sources(weights)
        pool = SourcePool.from_registered_weights(species, REGISTERED_POOL, pool=node)
        windows = sum(species[name].n_windows for name in pool.names)
        print(f"{node:<28} {len(pool.names)} species, {windows:,} windows")

    for stem, split_stem in held_out:
        print(f"{'held_out:' + stem:<28} {T._load_species_windows(stem, split_stem, 'train').n_windows:,} windows")

    donor = T.TIER2_FINAL_RESULTS / "Z_s0_reported.pt"
    print(f"F3c donor        : {'present' if donor.exists() else 'MISSING'} ({donor.name})")
    print("F9 donor         : Z9_s0_reported.pt, written by plan index 0")
    print("\ndry run only; nothing was trained and no test partition was opened.")


# Runners. 
def run_torch(run: Run, index: int, steps: int, grid_index: int, source: str, tier_e: bool, payload: dict) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    started = time.time()
    source_pool, node, subset = training_source(run, payload)
    pool = PreparedPool.from_source_pool(source_pool, device=device, sigma=SCALE_FLOOR_M)
    print(f"pool             : {pool.n_windows:,} windows over {list(pool.names)} [{node}], "
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
    extra = {}
    if config.species_label_map:
        extra = {"domain_rows": list(DOMAINS), "label_map": list(config.species_label_map)}
    torch.save(
        {"run_id": run.run_id, "config": asdict(config), "grid_index": grid_index, "pool_species": list(pool.names),
         "selection": {k: v for k, v in selection_payload.items() if k != "history"},
         "model": model.state_dict(), **extra},
        reported,
    )
    record = base_record(run, index, grid_index, source, node, tier_e) | {
        "config": asdict(config),
        "n_parameters": model.n_parameters(),
        "pool_species": list(pool.names),
        "pool_windows": pool.n_windows,
        "subset": subset_record(subset),
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
        **extra,
    }
    write_record(run, record)


def run_xgb(run: Run, index: int, grid_index: int, source: str, payload: dict) -> None:
    from animal_movement_fm.models.xgboost import WindowXGBoost

    source_pool, node, _ = training_source(run, payload)
    if ARMS[run.arm]["pool"] == "held_out_train":
        stem = next(iter(source_pool.names))
        windows = source_pool.species[stem]
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
    record = base_record(run, index, grid_index, source, node, False) | {
        "model": "WindowXGBoost",
        "params": model.params,
        "training_windows": int(len(features)),
        "sampling": sampled,
        "fit_seconds": time.time() - started,
        "reported_checkpoint": f"{stem_path.name}_*.ubj",
    }
    write_record(run, record)


def base_record(run: Run, index: int, grid_index: int, source: str, node: str, tier_e: bool) -> dict:
    return {
        "run_id": run.run_id,
        "plan_index": index,
        "arm": run.arm,
        "species": run.species or None,
        "seed": run.seed,
        "k_percent": run.k_percent,
        "draw": run.draw,
        "grid_index": grid_index,
        "configuration_source": source,
        "training_pool": node,
        "tier": "E" if tier_e else "3",
        "registered_by": None if tier_e else "preregistration_tier3.md",
        "exploratory": tier_e,
        "opened_test_partition": False,
    }


def subset_record(subset) -> dict | None:
    if subset is None:
        return None
    return {
        "draw_seed": subset.seed,
        "n_individuals": subset.n_individuals,
        "n_individuals_available": subset.n_available,
        "realised_fraction": subset.realised_fraction,
        "raised_to_floor": subset.floored,
    }


def write_record(run: Run, record: dict) -> None:
    path = RESULTS / f"{run.run_id}.json"
    path.write_text(json.dumps(record, indent=2, default=str) + "\n")
    print(f"wrote {path}", flush=True)


def print_plan(plan: list[Run], tier_e: bool) -> None:
    print(f"{'idx':>4}  {'arm':<7} {'species':<24} {'seed':>4} {'k':>5} {'draw':>5}")
    for i, run in enumerate(plan):
        k = "-" if run.k_percent is None else f"{run.k_percent}%"
        d = "-" if run.draw is None else run.draw
        print(f"{i:>4}  {run.arm:<7} {run.species or '-':<24} {run.seed:>4} {k:>5} {d:>5}")
    kind = "exploratory Tier E" if tier_e else "registered Tier 3"
    print(f"\n{len(plan)} {kind} runs (0-{len(plan) - 1}).")


def main() -> None:
    global RESULTS, CHECKPOINTS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=int, default=None, help="plan index; defaults to $PBS_ARRAY_INDEX")
    parser.add_argument("--steps", type=int, default=None, help="optimiser updates; must be the registered value")
    parser.add_argument("--list", action="store_true", help="print the plan and exit")
    parser.add_argument("--tier-e", action="store_true", help="run the exploratory Tier E plan (the 40k budget check)")
    parser.add_argument("--dry-run", action="store_true", help="build the registered pools and exit without training")
    args = parser.parse_args()

    plan = TIER_E_PLAN if args.tier_e else PLAN
    if args.list:
        return print_plan(plan, args.tier_e)
    if args.dry_run:
        return dry_run(load_registered_pool())
    if args.tier_e:
        RESULTS, CHECKPOINTS = T.TIER_E_RESULTS, T.TIER_E_CHECKPOINTS
    # The Tier 2 helpers resolve their checkpoint-selection files through their own module globals. 
    T.RESULTS, T.CHECKPOINTS = RESULTS, CHECKPOINTS

    raw = args.index if args.index is not None else os.environ.get("PBS_ARRAY_INDEX")
    if raw is None:
        raise SystemExit("FAIL: no --index and no $PBS_ARRAY_INDEX")
    index = int(raw)
    if not 0 <= index < len(plan):
        raise SystemExit(f"FAIL: index {index} is outside 0..{len(plan) - 1}; see --list")
    run = plan[index]
    expected = TIER_E_LONG_STEPS if args.tier_e else REGISTERED_STEPS
    if not ARMS[run.arm].get("xgb") and args.steps != expected:
        raise SystemExit(f"FAIL: --steps must be {expected:,} for this plan; got {args.steps}")

    payload = load_registered_pool()
    grid_index, source = T.selected_grid_index(None)
    RESULTS.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    print(f"run              : {run.run_id}  (plan index {index} of {len(plan) - 1})", flush=True)
    print(f"configuration    : grid {grid_index} from {source}", flush=True)
    if ARMS[run.arm].get("xgb"):
        run_xgb(run, index, grid_index, source, payload)
    else:
        run_torch(run, index, expected, grid_index, source, args.tier_e, payload)


if __name__ == "__main__":
    main()
