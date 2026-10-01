# GenAI reference: As with our training scripts, I used GenAI to write this from my details, which were a 2 x 2 on the B trunk over 
# the 26 source pool, sampling {A8, domain balanced} x FFN {dense, MoE}, with B26 carried from Tier 4 as the A8 dense cell; three 
# new arms at seeds 0-4, appended in the order B26bal, B26moe, B26balmoe; balanced weights as the registered Tier 4 domain sub-pools
# divided by 3 etc. nly; I reviewed it before any run
"""
Train the pooling study's arms, one model per PBS subjob (exploratory; `pooling_exploratory_freeze.md`).
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

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO))

import tier4_train as T4  # noqa: E402

from animal_movement_fm.data.enu import SCALE_FLOOR_M  # noqa: E402
from animal_movement_fm.data.multispecies import SourcePool  # noqa: E402
from animal_movement_fm.models.moe import N_EXPERTS, TOP_K  # noqa: E402
from animal_movement_fm.train.loop import MOE_AUX_WEIGHT, TrainConfig, train  # noqa: E402
from animal_movement_fm.train.pool import PreparedPool  # noqa: E402

T, IL = T4.T, T4.IL

RESULTS = REPO / "results" / "pooling"
CHECKPOINTS = REPO / "checkpoints" / "pooling"
FREEZE = "pooling_exploratory_freeze.md"

REGISTERED_STEPS = T4.REGISTERED_STEPS  # 20,000 applied updates at batch 256, Tier 4's budget unchanged
SEEDS = T4.SEEDS  # B26's seeds
ROUTING_WINDOWS = 2_048  # per source, evenly spaced through its training windows

# Domain-balanced draw probabilities, percent: each domain gets a third, split within the domain by A8 as before. They are
# the registered Tier 4 domain sub-pools divided by 3, which equals the full pool renormalised within each domain.
BALANCED_SOURCE_PERCENT: dict[str, float] = {
    # flying
    "ciconia_ciconia": 7.619,
    "cathartes_aura": 3.896,
    "anser_caerulescens": 3.792,
    "mycteria_americana": 3.151,
    "haematopus_ostralegus": 3.028,
    "corvus_corone": 2.838,
    "branta_canadensis": 2.679,
    "falco_naumanni": 1.518,
    "milvus_milvus": 1.348,
    "pernis_apivorus": 1.273,
    "gyps_himalayensis": 0.832,
    "diomedea_exulans": 0.494,
    "thalassarche_chrysostoma": 0.436,
    "thalassoica_antarctica": 0.43,
    # marine
    "mirounga_leonina": 13.32,
    "leptonychotes_weddellii": 5.079,
    "arctocephalus_gazella": 4.254,
    "eudyptes_chrysolophus": 3.069,
    "odobenus_rosmarus": 2.868,
    "lobodon_carcinophaga": 2.764,
    "aptenodytes_patagonicus": 1.978,
    # terrestrial
    "cervus_elaphus": 16.555,
    "connochaetes_taurinus": 5.032,
    "bison_bison": 4.31,
    "canis_latrans": 4.283,
    "lynx_rufus": 3.154,
}

# The 2 x 2. B26 (Tier 4 plan indices 5-9) is the A8, dense cell and is carried from Tier 4's sealed prediction files.
ARMS: dict[str, dict] = {
    "B26bal": {"balanced": True, "trunk": "transformer"},
    "B26moe": {"balanced": False, "trunk": "moe"},
    "B26balmoe": {"balanced": True, "trunk": "moe"},
}


@dataclass(frozen=True)
class Run:
    arm: str
    seed: int

    @property
    def run_id(self) -> str:
        return f"{self.arm}_s{self.seed}"

    @property
    def uses_validation_selection(self) -> bool:  
        return False


def build_plan() -> list[Run]:
    """The pooling plan. Position is the PBS array index: append, never insert or reorder."""
    return [Run(arm, s) for arm in ARMS for s in SEEDS]


PLAN = build_plan()
if len(PLAN) != 15:
    raise RuntimeError(f"the pooling plan has {len(PLAN)} runs; the freeze records 15")


# Registered pool.
def balanced_weights(payload: dict) -> dict[str, float]:
    """Each registered domain sub-pool, divided by 3."""
    return {
        name: p / 3.0
        for domain in T4.DOMAINS
        for name, p in payload["sub_pools"][f"domain_{domain}"].items()
    }


def load_registered_pool() -> dict:
    """Tier 4's registered pool, checked by tier4_train, and the balanced weights derived from it, checked here."""
    payload = T4.load_registered_pool()
    got = {name: round(100.0 * p, 3) for name, p in balanced_weights(payload).items()}
    if got != BALANCED_SOURCE_PERCENT:
        raise SystemExit(f"FAIL: the balanced weights are {got}; the freeze registers {BALANCED_SOURCE_PERCENT}")
    return payload


def training_source(run: Run, payload: dict) -> tuple[SourcePool, str]:
    """All 26 sources' training windows, drawn by A8 exactly as B26 drew them, or domain-balanced."""
    weights = balanced_weights(payload) if ARMS[run.arm]["balanced"] else payload["source_pool"]
    T4.assert_tier4_sources(weights)
    species = {
        stem: T._load_species_windows(stem, T4.SPLIT_STEM.get(stem, stem), "train") for stem in sorted(weights)
    }
    if ARMS[run.arm]["balanced"]:
        return SourcePool(species, weights), "source_pool:domain_balanced"
    return SourcePool.from_registered_weights(species, T4.REGISTERED_POOL), "source_pool"


def build_config(run: Run, grid_index: int, steps: int) -> TrainConfig:
    """B26's configuration with the arm's trunk; the plain trunk, so no package and no reconstruction head."""
    d_model, layers, lr = IL.GRID[grid_index]
    return TrainConfig(
        d_model=d_model, num_layers=layers, lr=lr, steps=steps, seed=run.seed,
        use_fm_package=False, use_reconstruction_head=False, trunk=ARMS[run.arm]["trunk"],
    )


# Routing.
def routing_summary(model, pool: PreparedPool, domains: dict[str, str]) -> dict:
    """
    Expert use per source on its own training windows, per layer: mean router probability, share of the top-k
    assignments, and how often a window's tokens share their most common top-1 expert. Domain totals are left to the
    report. No validation or test window is read.
    """
    model.eval()
    per_source = {}
    with torch.no_grad():
        for i, name in enumerate(pool.names):
            start, end = (int(b) for b in pool.species_bounds[i])
            n = min(ROUTING_WINDOWS, end - start)
            rows = torch.linspace(start, end - 1, n, device=pool.features.device).long()
            model(pool.features[rows])
            layers = []
            for probs in model.encoder.router_probs:
                top1 = probs.argmax(dim=-1)
                assigned = torch.zeros_like(probs).scatter(-1, probs.topk(TOP_K, dim=-1).indices, 1.0)
                layers.append({
                    "mean_prob": probs.mean(dim=(0, 1)).tolist(),
                    "top_k_share": (assigned.mean(dim=(0, 1)) / TOP_K).tolist(),
                    "window_agreement": float((top1 == top1.mode(dim=1).values[:, None]).float().mean()),
                })
            per_source[name] = {"domain": domains[name], "windows": n, "layers": layers}
    return per_source


# Dry run.
def dry_run(payload: dict, index: int | None) -> None:
    """Build every source and both weightings, or one plan index's pool and model, and stop before training."""
    if index is None:
        for label, weights in (("A8", payload["source_pool"]), ("domain-balanced", balanced_weights(payload))):
            T4.assert_tier4_sources(weights)
            shares = {d: sum(p for n, p in weights.items() if payload["domains"][n] == d) for d in T4.DOMAINS}
            print(f"{label:<16} " + "  ".join(f"{d} {100 * s:5.1f}%" for d, s in shares.items()))
        species = {
            stem: T._load_species_windows(stem, T4.SPLIT_STEM.get(stem, stem), "train")
            for stem in sorted(payload["source_pool"])
        }
        print(f"{len(species)} sources built, {sum(w.n_windows for w in species.values()):,} training windows")
        print("\ndry run only; nothing was trained and no test partition was opened.")
        return

    run = PLAN[index]
    grid_index, _ = T.selected_grid_index(None)
    print(f"index {index:>2}  {run.run_id}")
    source_pool, node = training_source(run, payload)
    windows = sum(source_pool.species[name].n_windows for name in source_pool.names)
    print(f"  pool           : {node}, {len(source_pool.names)} species, {windows:,} windows")
    config = build_config(run, grid_index, REGISTERED_STEPS)
    print(f"  config         : grid {grid_index}, trunk {config.trunk}, d_model {config.d_model}, "
          f"layers {config.num_layers}, lr {config.lr:g}, {config.build().n_parameters():,} parameters")
    print("  dry run only; nothing trained, no test partition opened.")


# Runner.
def run_torch(run: Run, index: int, steps: int, grid_index: int, source: str, payload: dict) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    started = time.time()
    source_pool, node = training_source(run, payload)
    pool = PreparedPool.from_source_pool(source_pool, device=device, sigma=SCALE_FLOOR_M)
    print(f"pool             : {pool.n_windows:,} windows over {len(pool.names)} species [{node}], "
          f"built in {time.time() - started:.0f}s", flush=True)

    config = build_config(run, grid_index, steps)
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
         "selection": selection_payload, "model": model.state_dict()},
        reported,
    )
    moe = {}
    if config.trunk == "moe":
        moe = {
            "moe": {"n_experts": N_EXPERTS, "top_k": TOP_K, "aux_weight": MOE_AUX_WEIGHT},
            "routing": routing_summary(model, pool, T4.domain_of(payload)),
        }
    record = base_record(run, index, grid_index, source, node) | {
        "config": asdict(config),
        "n_parameters": model.n_parameters(),
        "pool_species": list(pool.names),
        "pool_windows": pool.n_windows,
        "draw_probabilities": dict(zip(pool.names, pool.probabilities.tolist())),
        "final_loss": result.final_loss,
        # MoE arms: the total loss, which includes MOE_AUX_WEIGHT x the load-balancing term.
        "loss_history": result.loss_history,
        # Per-source shares of the forecast loss, the dilution diagnostic as Tier 4 freeze sec 4 reads it.
        "loss_shares": result.loss_shares,
        "skipped_steps": result.skipped_steps,
        "attempted_batches": result.attempted_batches,
        "applied_updates": result.applied_updates,
        "withheld_fraction": result.withheld_fraction,
        "nonfinite_steps": result.nonfinite_steps,
        "checkpoint_selection": selection_payload,
        "reported_checkpoint": reported.name,
        **moe,
    }
    write_record(run, record)


def base_record(run: Run, index: int, grid_index: int, source: str, node: str) -> dict:
    return {
        "run_id": run.run_id,
        "plan_index": index,
        "arm": run.arm,
        "seed": run.seed,
        "sampling": "domain_balanced" if ARMS[run.arm]["balanced"] else "A8",
        "trunk": ARMS[run.arm]["trunk"],
        "grid_index": grid_index,
        "configuration_source": source,
        "training_pool": node,
        "tier": "pooling",
        "registered_by": FREEZE,
        # Exploratory throughout: no number from it may be reported as a tiered result. 
        "exploratory": True,
        "opened_test_partition": False,
    }


def write_record(run: Run, record: dict) -> None:
    path = RESULTS / f"{run.run_id}.json"
    path.write_text(json.dumps(record, indent=2, default=str) + "\n")
    print(f"wrote {path}", flush=True)


def print_plan(plan: list[Run]) -> None:
    print(f"{'idx':>4}  {'arm':<10} {'sampling':<16} {'trunk':<12} {'seed':>4}")
    for i, run in enumerate(plan):
        sampling = "domain_balanced" if ARMS[run.arm]["balanced"] else "A8"
        print(f"{i:>4}  {run.arm:<10} {sampling:<16} {ARMS[run.arm]['trunk']:<12} {run.seed:>4}")
    print(f"\n{len(plan)} exploratory pooling runs (0-{len(plan) - 1}); B26 is carried from Tier 4.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=int, default=None, help="plan index; defaults to $PBS_ARRAY_INDEX")
    parser.add_argument("--steps", type=int, default=None, help="optimiser updates; must be the registered value")
    parser.add_argument("--list", action="store_true", help="print the plan and exit")
    parser.add_argument("--dry-run", action="store_true", help="build the pool for --index, or every source, and exit")
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
    if args.steps != REGISTERED_STEPS:
        raise SystemExit(f"FAIL: --steps must be {REGISTERED_STEPS:,} for this plan; got {args.steps}")

    # The Tier 2 helpers resolve their files through their own module globals.
    T.RESULTS, T.CHECKPOINTS = RESULTS, CHECKPOINTS

    run = PLAN[index]
    payload = load_registered_pool()
    grid_index, source = T.selected_grid_index(None)
    RESULTS.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    print(f"run              : {run.run_id}  (plan index {index} of {len(PLAN) - 1})", flush=True)
    print(f"configuration    : grid {grid_index} from {source}", flush=True)
    run_torch(run, index, REGISTERED_STEPS, grid_index, source, payload)


if __name__ == "__main__":
    main()
