# GenAI reference: I specified this file, the whole 91 run plan by index with Z26 first and F26 last, the arms and their
# pools from tier4_exploratory_freeze.md sec 2, F26 fine-tuning Z26 seed 0 on F9's own subsets and recipe, B26dom's 26 
# species domain label map, the refusal of held-out and reserved species, and records tagged tier "4" and exploratory. 
# GenAI wrote it against that specification using files that I gave it. It imports tier3_train.py and the same Tier 2 
# machinery behind it rather than just restating it again. I review it before any run; added docs and some comments
# are my own; others I kept from GenAI 
"""
Train the Tier 4 arms, one model per PBS subjob (`tier4_exploratory_freeze.md` sec 2).
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

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO))

import tier2_final_train as T  # noqa: E402
import tier2_inner_loop as IL  # noqa: E402
import tier3_train as T3  # noqa: E402

from animal_movement_fm.data.enu import SCALE_FLOOR_M  # noqa: E402
from animal_movement_fm.data.multispecies import SourcePool  # noqa: E402
from animal_movement_fm.train.loop import TrainConfig, train  # noqa: E402
from animal_movement_fm.train.pool import PreparedPool  # noqa: E402
from experiments.data import source_pool_weights as SPW  # noqa: E402

RESULTS = REPO / "results" / "tier4"
CHECKPOINTS = REPO / "checkpoints" / "tier4"
REGISTERED_POOL = REPO / "registered" / "tier4_source_pool.json"

REGISTERED_STEPS = T3.REGISTERED_STEPS  # 20,000 applied updates at batch 256, Tier 3's budget unchanged
SEEDS = T.SEEDS
HELD_OUT = T3.HELD_OUT
# the same lookup row order as B9dom, so a Tier 4 domain row means what a Tier 3 domain row means
DOMAINS = T3.DOMAINS
SPLIT_STEM = T3.SPLIT_STEM
XGB_POOLED_DRAWS = T3.XGB_POOLED_DRAWS
LSTM_D_MODEL, LSTM_LAYERS = T3.LSTM_D_MODEL, T3.LSTM_LAYERS

# the registered pool and its percentages live in source_pool_weights.py, which wrote the file; they are imported so
# the training side and the artefact side cannot drift apart
REGISTERED_SOURCE_PERCENT = SPW.TIER4_SOURCE_PERCENT
REGISTERED_SUB_POOL_PERCENT = SPW.TIER4_SUB_POOL_PERCENT
RESERVED = SPW.TIER4_RESERVED
NO_MIROUNGA = SPW.TIER4_NO_MIROUNGA

# TrainConfig is frozen for the duration of Tier 4. A field added mid-campaign changes asdict(config) and the resume
# guard in train/loop.py then refuses to continue any checkpoint written before it, stranding every requeued subjob
FROZEN_TRAIN_CONFIG_FIELDS = (
    "batch_size", "d_model", "dropout", "freeze_except_species_embedding", "grad_clip", "lr", "mask_ratio",
    "n_species", "nhead", "num_layers", "random_species_labels", "reconstruction_weight", "seed",
    "species_embed_dim", "species_label_map", "steps", "trunk", "use_context_conditioning", "use_fm_package",
    "use_reconstruction_head", "warmup_fraction",
)

# arm -> how it is built, in the shape tier3_train.ARMS uses. `pool` names the training data; `per_species` attaches
# the arm to each held-out species; `species` pins a domain arm to the one species it is scored on.
# Freeze names map to keys here: Zd-F -> ZdF, Zd-T -> ZdT, Zd-M -> ZdM, ZdM-noMir -> ZdMnoMir,
# BdM-noMir -> BdMnoMir, LSTM-B26 -> LSTMB26, XGB-B26 -> XGBB26.
ARMS: dict[str, dict] = {
    "Z26": {"pool": "source_pool", "package": True, "recon": True},
    "B26": {"pool": "source_pool", "package": False, "recon": False},
    "Za26": {"pool": "without_domain", "package": True, "recon": True, "per_species": True},
    "ZdF": {"pool": "domain", "package": True, "recon": True, "species": "larus_fuscus"},
    "ZdT": {"pool": "domain", "package": True, "recon": True, "species": "canis_lupus"},
    "ZdM": {"pool": "domain", "package": True, "recon": True, "species": "mirounga_angustirostris"},
    "BdM": {"pool": "domain", "package": False, "recon": False, "species": "mirounga_angustirostris"},
    "ZdMnoMir": {"pool": "no_mirounga", "package": True, "recon": True, "species": "mirounga_angustirostris"},
    "BdMnoMir": {"pool": "no_mirounga", "package": False, "recon": False, "species": "mirounga_angustirostris"},
    "B26dom": {"pool": "source_pool", "package": True, "recon": False, "context": False, "domain_embed": 16},
    "LSTMB26": {"pool": "source_pool", "trunk": "lstm"},
    "XGBB26": {"pool": "source_pool", "xgb": True},
    "F26": {"pool": "held_out_subset", "package": True, "recon": True, "per_species": True, "selection": True,
            "init": ("tier4", "Z26_s0")},
}

# tier4_exploratory_freeze.md sec 2: read from Tier 3's sealed prediction files, never retrained here.
CARRIED_ARMS = frozenset({
    "P", "V", "ARIMA", "AVG3", "AVG10", "S", "SB", "Sk", "Zd1", "LSTMS", "XGBS", "F9", "F3c",
    "Z9", "B9", "Za9", "Zd3", "B9dom", "LSTMB3", "LSTMB9", "XGBB3", "XGBB9", "E1embed_dom",
})


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


def build_plan() -> list[Run]:
    """The Tier 4 plan. Position is the PBS array index: append, never insert or reorder."""
    plan = [Run("Z26", "", s) for s in SEEDS] + [Run("B26", "", s) for s in SEEDS]
    plan += [Run("Za26", sp, s) for sp in HELD_OUT for s in SEEDS]
    for arm in ("ZdF", "ZdT", "ZdM", "BdM", "ZdMnoMir", "BdMnoMir"):
        plan += [Run(arm, ARMS[arm]["species"], s) for s in SEEDS]
    plan += [Run("B26dom", "", s) for s in SEEDS]
    plan += [Run("LSTMB26", "", s) for s in SEEDS]
    plan += [Run("XGBB26", "", s) for s in SEEDS]
    plan += [
        Run("F26", sp, T.FINETUNE_INIT_SEED, k, d)
        for sp in HELD_OUT
        for k, draws in T.FINETUNE_DRAWS_PER_FRACTION.items()
        for d in range(draws)
    ]
    return plan


PLAN = build_plan()
if len(PLAN) != 91:
    raise RuntimeError(f"the Tier 4 plan has {len(PLAN)} runs; tier4_exploratory_freeze.md sec 2 records 91")


# Registered pool.
def load_registered_pool() -> dict:
    """Read the registered Tier 4 pool and check it against the constants source_pool_weights.py wrote it from."""
    payload = json.loads(REGISTERED_POOL.read_text())
    got = {name: round(100.0 * p, 3) for name, p in payload["source_pool"].items()}
    if got != REGISTERED_SOURCE_PERCENT:
        raise SystemExit(
            f"FAIL: {REGISTERED_POOL.name} source_pool is {got}; the freeze registers {REGISTERED_SOURCE_PERCENT}"
        )
    if "sub_pools" not in payload or "domains" not in payload:
        raise SystemExit(
            f"FAIL: {REGISTERED_POOL.name} has no sub_pools or domains. Run `source_pool_weights.py --tier4` once."
        )
    for label, registered in REGISTERED_SUB_POOL_PERCENT.items():
        if label not in payload["sub_pools"]:
            raise SystemExit(f"FAIL: {REGISTERED_POOL.name} has no sub-pool {label}")
        got = {name: round(100.0 * p, 3) for name, p in payload["sub_pools"][label].items()}
        if got != registered:
            raise SystemExit(f"FAIL: {REGISTERED_POOL.name} {label} is {got}; the freeze registers {registered}")
    return payload


def domain_of(payload: dict) -> dict[str, str]:
    """Tier 4's payload carries a flat domain map; there is no selection rule and so no fixed/added split."""
    return dict(payload["domains"])


def assert_tier4_sources(stems) -> None:
    """Pooled Tier 4 arms draw only from the registered 26; never a held-out or reserved species."""
    stems = set(stems)
    for label, forbidden in (("held-out", set(HELD_OUT)), ("reserved", set(RESERVED))):
        if stems & forbidden:
            raise SystemExit(f"FAIL: {sorted(stems & forbidden)} is a {label} species and cannot enter a source pool")
    unknown = stems - set(REGISTERED_SOURCE_PERCENT)
    if unknown:
        raise SystemExit(f"FAIL: {sorted(unknown)} is not a registered Tier 4 source")


def pool_node(run: Run) -> str:
    """The node in the registered file that this arm draws from."""
    spec = ARMS[run.arm]
    if spec["pool"] == "source_pool":
        return "source_pool"
    if spec["pool"] == "no_mirounga":
        return f"sub_pools.{NO_MIROUNGA}"
    held = T.held_out_entry(run.species)
    if spec["pool"] == "without_domain":
        return f"sub_pools.without_{held[3]}"
    if spec["pool"] == "domain":
        return f"sub_pools.domain_{held[3]}"
    raise SystemExit(f"FAIL: {run.arm} is not a pooled arm")


def pooled_source(run: Run, payload: dict) -> tuple[SourcePool, str]:
    """The registered SourcePool for a pooled arm, with its node name."""
    node = pool_node(run)
    weights = payload
    for part in node.split("."):
        weights = weights[part]
    assert_tier4_sources(weights)
    species = {
        stem: T._load_species_windows(stem, SPLIT_STEM.get(stem, stem), "train") for stem in sorted(weights)
    }
    return SourcePool.from_registered_weights(species, REGISTERED_POOL, pool=node), node


def held_out_source(run: Run) -> tuple[SourcePool, str, object]:
    """F26's training pool: the registered fine-tuning subset of one held-out species' training individuals."""
    stem, split_stem, *_ = T.held_out_entry(run.species)
    # the subset is a property of (species, k, draw) alone, so this is F9's and Sk's own draw, not a new one
    subset = T.finetune_subset(run)
    windows = T._load_species_windows(stem, split_stem, "train", subset.individuals)
    return SourcePool({stem: windows}, {stem: 1.0}), f"subset:{stem}:k{run.k_percent}:d{run.draw}", subset


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
    # the Z arms keep Tier 2's package definition exactly, and the B arms are the trunk with it stripped
    if run.arm in ("Z26", "Za26", "ZdF", "ZdT", "ZdM", "ZdMnoMir", "F26"):
        if not (config.use_fm_package and config.use_reconstruction_head and config.use_context_conditioning):
            raise RuntimeError(f"{run.arm} must carry the complete registered FM package")
    if run.arm in ("B26", "BdM", "BdMnoMir") and (config.use_fm_package or config.use_reconstruction_head):
        raise RuntimeError(f"{run.arm} must be the trunk with the FM package stripped")
    return config


def initial_state(run: Run, grid_index: int) -> tuple[dict, str | None]:
    """F26 initialises from Z26 seed 0 (A13); it refuses to start until that run has reported."""
    init = ARMS[run.arm].get("init")
    if not init:
        return {}, None
    where, donor_id = init
    directory = RESULTS if where == "tier4" else T.TIER2_FINAL_RESULTS
    path = directory / f"{donor_id}_reported.pt"
    if not path.exists():
        raise SystemExit(f"FAIL: {run.arm} fine-tunes {donor_id}, but {path} does not exist yet; train it first")
    donor = torch.load(path, map_location="cpu", weights_only=False)
    d_model, layers, _ = IL.GRID[grid_index]
    if (donor["config"]["d_model"], donor["config"]["num_layers"]) != (d_model, layers):
        raise SystemExit(f"FAIL: {donor_id} is not at grid {grid_index}'s geometry")
    return {"init_state_dict": donor["model"]}, f"{where}:{donor_id}"


# Dry run.
def dry_run(payload: dict, index: int | None) -> None:
    """
    Build one plan index's pool, or every registered node, and stop before training.

    Seventeen of the 26 sources have never been through a training run, so this is also their first pass through the
    loaders. A missing or misnamed segments file or split manifest is otherwise found inside the first queued job,
    after the queue wait.
    """
    if index is None:
        nodes = ["source_pool"] + [f"sub_pools.{name}" for name in sorted(payload["sub_pools"])]
        species = {
            stem: T._load_species_windows(stem, SPLIT_STEM.get(stem, stem), "train")
            for stem in sorted(REGISTERED_SOURCE_PERCENT)
        }
        for node in nodes:
            weights = payload
            for part in node.split("."):
                weights = weights[part]
            assert_tier4_sources(weights)
            pool = SourcePool.from_registered_weights(species, REGISTERED_POOL, pool=node)
            windows = sum(species[name].n_windows for name in pool.names)
            print(f"{node:<30} {len(pool.names):>2} species, {windows:,} windows")
        print("\ndry run only; nothing was trained and no test partition was opened.")
        return

    run = PLAN[index]
    grid_index, _ = T.selected_grid_index(None)
    print(f"index {index:>2}  {run.run_id}")
    source_pool, node, subset = training_source(run, payload)
    windows = sum(source_pool.species[name].n_windows for name in source_pool.names)
    print(f"  pool           : {node}, {len(source_pool.names)} species, {windows:,} windows")
    if subset is not None:
        print(f"  subset         : k={run.k_percent}% draw {run.draw}, "
              f"{subset.n_individuals} of {subset.n_available} individuals")
    # the XGBoost rungs fit once and build no TrainConfig, so the dry run branches where the runner does
    if ARMS[run.arm].get("xgb"):
        print(f"  model          : WindowXGBoost, {XGB_POOLED_DRAWS:,} A8 draws, seed {run.seed}")
        print("  dry run only; nothing trained, no test partition opened.")
        return
    config = build_config(run, grid_index, REGISTERED_STEPS, source_pool.names, payload)
    label = f", label map {len(config.species_label_map)} rows" if config.species_label_map else ""
    # the LSTM rungs keep Tier 1's settled geometry, so print what was built rather than the grid cell
    print(f"  config         : grid {grid_index}, trunk {config.trunk}, "
          f"d_model {config.d_model}, layers {config.num_layers}, lr {config.lr:g}, "
          f"package {config.use_fm_package}, recon {config.use_reconstruction_head}{label}")
    _, donor_id = initial_state(run, grid_index)  # F26 refuses here until Z26_s0 has reported
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
    extra = {}
    if config.species_label_map:
        extra = {"domain_rows": list(DOMAINS), "label_map": list(config.species_label_map)}
    torch.save(
        {"run_id": run.run_id, "config": asdict(config), "grid_index": grid_index, "pool_species": list(pool.names),
         "selection": {k: v for k, v in selection_payload.items() if k != "history"},
         "model": model.state_dict(), **extra},
        reported,
    )
    record = base_record(run, index, grid_index, source, node) | {
        "config": asdict(config),
        "n_parameters": model.n_parameters(),
        "pool_species": list(pool.names),
        "pool_windows": pool.n_windows,
        "subset": T3.subset_record(subset),
        "initialised_from": donor_id,
        "final_loss": result.final_loss,
        "loss_history": result.loss_history,
        # per-species shares of the batch loss, logged by train() every 500 applied updates; this is the
        # tier4_exploratory_freeze.md sec 4 dilution diagnostic, read the same way as Tier 3 freeze sec 7.1
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
    rng = np.random.default_rng(run.seed)
    features, targets, _ = source_pool.sample(rng, XGB_POOLED_DRAWS)
    sampled = f"{XGB_POOLED_DRAWS:,} A8 draws, seed {run.seed}"
    print(f"xgboost data     : {len(features):,} windows ({sampled}) [{node}]", flush=True)
    started = time.time()
    model = WindowXGBoost(random_state=run.seed, n_jobs=int(os.environ.get("PBS_NCPUS", "0")) or None)
    model.fit(features.astype(np.float32), targets.astype(np.float32))
    stem_path = RESULTS / f"{run.run_id}_xgb"
    model.save(stem_path)
    record = base_record(run, index, grid_index, source, node) | {
        "model": "WindowXGBoost",
        "params": model.params,
        "training_windows": int(len(features)),
        "sampling": sampled,
        "fit_seconds": time.time() - started,
        "reported_checkpoint": f"{stem_path.name}_*.ubj",
    }
    write_record(run, record)


def base_record(run: Run, index: int, grid_index: int, source: str, node: str) -> dict:
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
        "tier": "4",
        "registered_by": "tier4_exploratory_freeze.md",
        # Tier 4 is exploratory throughout: no number from it may be reported as a tiered result
        "exploratory": True,
        "opened_test_partition": False,
    }


def write_record(run: Run, record: dict) -> None:
    path = RESULTS / f"{run.run_id}.json"
    path.write_text(json.dumps(record, indent=2, default=str) + "\n")
    print(f"wrote {path}", flush=True)


def print_plan(plan: list[Run]) -> None:
    print(f"{'idx':>4}  {'arm':<9} {'species':<24} {'seed':>4} {'k':>5} {'draw':>5}")
    for i, run in enumerate(plan):
        k = "-" if run.k_percent is None else f"{run.k_percent}%"
        d = "-" if run.draw is None else run.draw
        print(f"{i:>4}  {run.arm:<9} {run.species or '-':<24} {run.seed:>4} {k:>5} {d:>5}")
    print(f"\n{len(plan)} exploratory Tier 4 runs (0-{len(plan) - 1}).")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=int, default=None, help="plan index; defaults to $PBS_ARRAY_INDEX")
    parser.add_argument("--steps", type=int, default=None, help="optimiser updates; must be the registered value")
    parser.add_argument("--list", action="store_true", help="print the plan and exit")
    parser.add_argument("--dry-run", action="store_true", help="build the pool for --index, or every node, and exit")
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

    # the Tier 2 helpers resolve their checkpoint-selection files through their own module globals
    T.RESULTS, T.CHECKPOINTS = RESULTS, CHECKPOINTS

    payload = load_registered_pool()
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
