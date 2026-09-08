# GenAI reference: assisted with some of the surrounding PBS/CLI orchestration, defensive checks progress output where 
# I outlined the arm definitions, training pools and S/F selection rules etc. 
"""
Train the registered Tier 2 comparator arms, one model per PBS subjob.

Z and B train on the full three species source pool; S trains on the held out species' own training individuals; Za trains 
on the source sub pool with the matching domain removed. P and V require no training and are handled by sealed evaluation.

This script never opens a test partition. Z, B and Za are also prevented from opening held-out species data. S may use 
held-out validation data only for its registered checkpoint-selection rule; the other trained arms report their final 
checkpoint.
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import copy
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tier2_inner_loop import GRID, SOURCES  # single source of truth for the registered grid order

from animal_movement_fm.data.benchmark import load_split_frame
from animal_movement_fm.data.multispecies import SourcePool, SpeciesWindows
from animal_movement_fm.data.resample import CONTEXT_LENGTH
from animal_movement_fm.data.enu import SCALE_FLOOR_M
from animal_movement_fm.data.split import load_split
from animal_movement_fm.eval.fm_predictor import MovementFMPredictor, evaluate_batched
from animal_movement_fm.eval.harness import evaluate
from animal_movement_fm.eval.selection import score_fold
from animal_movement_fm.models.persistence import Persistence
from animal_movement_fm.train.finetune import Subset, nested_subsets
from animal_movement_fm.train.loop import SDPA_BACKEND, TrainConfig, train
from animal_movement_fm.train.pool import PreparedPool

REPO = Path(__file__).resolve().parents[2]
PROCESSED = REPO / "data" / "processed"
SPLITS = REPO / "splits"
WEIGHTS = REPO / "results" / "source_pool_weights.json"
INNER_LOOP_RESULTS = REPO / "results" / "tier2_inner_loop"
RESULTS = REPO / "results" / "tier2_final"
CHECKPOINTS = REPO / "checkpoints" / "tier2_final"

# Tier 1 multi-seed sweep  reused unchanged across all trained arms
SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)

# A5 minimum valid windows per individual for evaluation.
EVALUATION_FLOOR = 25

# S (and later F) selects the checkpoint with minimum validation travel regime p90; Z, B and Za have no validation privilege
# and report the final registered checkpoint
SELECTION_RULE = "argmin_validation_travel_p90"
SELECTION_EVERY = 1_000

# fine tuning (F), is 2,000 optimiser steps at a peak rate of 10%
FINETUNE_STEPS = 2_000
FINETUNE_LR_FRACTION = 0.1

# F_k is "Z further trained". Z has five seeds and the registered count of 21 runs (3 species x 7)
FINETUNE_INIT_ARM = "Z"
FINETUNE_INIT_SEED = 0

# three independent draws at k = 1% and at k = 10%, one at k = 100% where there is nothing to vary
FINETUNE_DRAWS_PER_FRACTION = {1: 3, 10: 3, 100: 1}

# (segments stem, split manifest stem, regime_profiles.csv row key, domain, matching source)
HELD_OUT: tuple[tuple[str, str, str, str, str], ...] = (
    (
        "larus_fuscus",
        "larus_fuscus",
        "larus_fuscus",
        "flying",
        "ciconia_ciconia",
    ),
    (
        "canis_lupus",
        "canis_lupus",
        "canis_lupus",
        "terrestrial",
        "cervus_elaphus",
    ),
    (
        "mirounga_angustirostris",
        "mirounga_angustirostris",
        "mirounga_angustirostris",
        "marine",
        "mirounga_leonina",
    ),
)

SOURCE_STEMS = frozenset(source[0] for source in SOURCES)
HELD_OUT_STEMS = frozenset(species[0] for species in HELD_OUT)


@dataclass(frozen=True)
class Run:
    """
    Describe one trained arm/species/seed combination.

    `species` is empty for Z and B. For S and Za it identifies the held-out species the run is attached to; Za still trains 
    only on source species.
    """

    arm: str
    species: str
    seed: int
    deferred: bool = False
    # F only; registered fraction of the held-out species' training individuals
    k_percent: int | None = None
    draw: int | None = None

    @property
    def run_id(self) -> str:
        base = (
            f"{self.arm}_{self.species}_s{self.seed}"
            if self.species
            else f"{self.arm}_s{self.seed}"
        )
        if self.k_percent is None:
            return base
        return f"{self.arm}_{self.species}_k{self.k_percent:03d}_d{self.draw}_s{self.seed}"

    @property
    def uses_validation_selection(self) -> bool:
        """
        Return whether this arm may use validation data for checkpoint selection.
        """
        if self.arm in {"S", "F"}:
            return True
        # a Tier E arm may claim the same privilege when it exists to be compared against S; it is
        # then scored on the individuals it selected on, and the evaluator flags it as optimistic
        return bool(TIER_E_ARMS.get(self.arm, {}).get("validation_selection"))


def build_plan() -> list[Run]:
    """
    Return the fixed PBS run plan.

    List position is the array index, so the order is load-bearing. Z, B and S form the critical path; deferred Za runs 
    follow them.
    """
    plan = [Run("Z", "", seed) for seed in SEEDS]
    plan += [Run("B", "", seed) for seed in SEEDS]
    for species, *_ in HELD_OUT:
        plan += [Run("S", species, seed) for seed in SEEDS]
    for species, *_ in HELD_OUT:
        plan += [Run("Za", species, seed, deferred=True) for seed in SEEDS]
    # F appended after the arms that have already run; 7 runs per species, in fraction order
    for species, *_ in HELD_OUT:
        for k_percent, draws in FINETUNE_DRAWS_PER_FRACTION.items():
            plan += [
                Run("F", species, FINETUNE_INIT_SEED, deferred=True, k_percent=k_percent, draw=draw)
                for draw in range(draws)
            ]
    return plan


PLAN = build_plan()
N_CRITICAL_PATH = sum(1 for run in PLAN if not run.deferred)

# Indices 0-39 are trained and reported so we append F at 40 onwards
PLAN_FROZEN_PREFIX: tuple[str, ...] = tuple(
    [f"Z_s{seed}" for seed in SEEDS]
    + [f"B_s{seed}" for seed in SEEDS]
    + [f"S_{species}_s{seed}" for species, *_ in HELD_OUT for seed in SEEDS]
    + [f"Za_{species}_s{seed}" for species, *_ in HELD_OUT for seed in SEEDS]
)
_plan_prefix = tuple(run.run_id for run in PLAN[: len(PLAN_FROZEN_PREFIX)])
if _plan_prefix != PLAN_FROZEN_PREFIX:
    raise RuntimeError(
        "the registered PLAN indices have moved. Expected the first "
        f"{len(PLAN_FROZEN_PREFIX)} runs to be\n  " + "\n  ".join(PLAN_FROZEN_PREFIX)
        + "\nbut the plan now starts\n  " + "\n  ".join(_plan_prefix)
        + "\nAppend to build_plan; never insert or reorder."
    )
del _plan_prefix


# Tier E additon here!
# exploratory variants we outlined in `tierE_exploratory_ablations.md`, not the prereg; so they use a separate run plan so 
# the registered PLAN indices never move. `--tier-e` switches to that plan and writes results/checkpoints to separate 
# directories, keeping Tier E artefacts out of anything that reads our registered Tier 2 outputs
TIER_E_RESULTS = REPO / "results" / "tier_e"
TIER_E_CHECKPOINTS = REPO / "checkpoints" / "tier_e"

# `RESULTS` is rebound to TIER_E_RESULTS by --tier-e, so it cannot be used to reach a registered
# arm from inside a Tier E run. This name is never rebound and is what a Tier E arm initialising
# from Z reads.
TIER2_FINAL_RESULTS = REPO / "results" / "tier2_final"

TIER_E_ARMS: dict[str, dict] = {
    # remove the masked reconstruction objective and keep everything else, so Z vs Zrecon differs in exactly one thing
    "Zrecon": {
        "description": "Z minus the masked reconstruction objective; conditioning path intact",
        "use_fm_package": True,
        "use_reconstruction_head": False,
    },
    # reconstruction head is trained to reproduce the normalised displacement of masked tokens, and sigma = 30m makes a 
    # near stationary window's normalised displacement enormous, so the objective explicitly teaches the model to emit motion
    # where the animal did not move; raising the floor should let the objective be kept
    "Zfloor300": {
        "description": "Z with the scale floor raised from 30 m to 300 m; full package retained",
        "use_fm_package": True,
        "use_reconstruction_head": True,
        "scale_floor_m": 300.0,
    },
    # closes the architecture ablation; Zrecon is conditioning-on/reconstruction-off; this is its complement, conditioning off/
    # reconstruction-on, which is B's trunk plus the objective. Together with Z (both on) and B (both off) the 2x2 is complete, 
    # so the conditioning paths contribution can be read directly instead of inferred from the Zrecon to B gap
    "Brecon": {
        "description": "B's trunk plus the masked reconstruction objective; no conditioning path",
        "use_fm_package": False,
        "use_reconstruction_head": True,
    },
    # Zfloor300 reduced the low displacement failure in proportion to how sigma compared with each species' regime boundary: wolf's 
    # is 131 m and it fell 9.61x -> 1.78x, while gull's is 411 m and NES's 1,609 m and both improved far less. If the floor has to 
    # exceed the low displacement scale rather than take any particular value, 1,000 m should help gull and NES where 300 m did not.
    "Zfloor1000": {
        "description": "Z with the scale floor raised from 30 m to 1,000 m; full package retained",
        "use_fm_package": True,
        "use_reconstruction_head": True,
        "scale_floor_m": 1000.0,
    },
    # E1 and E2
    # John flagged a gap in the original zero-shot setut; the model may need an explicit species signal rather than being asked to 
    # infer one from three pooled source species. E1 asks whether species identity is useful at all; E2 then asks whether that 
    # information can be added for an unseen species

    # E1 (Q1): B's trunk plus a learned per species embedding driving FiLM, with context conditioning and reconstruction both off; 
    # compared with B, species identity is the only new input. Scored on source-species validation data because only source species 
    # have embedding rows.
    "E1embed": {
        "description": (
            "B's trunk plus a learned 16-dim per-species embedding driving FiLM; no context conditioning, no reconstruction head. "
            "Source species validation only; not zero-shot"
        ),
        "use_fm_package": True,
        "use_context_conditioning": False,
        "use_reconstruction_head": False,
        "species_embed_dim": 16,
        "scored_partition": "source_validation",
    },

    # capacity control for E1embed. Labels are redrawn independently per window; a fixed permutation would just relabel the lookup 
    # rows and reproduce the same model.
    "E1shuffle": {
        "description": (
            "E1embed with the species label drawn uniformly at random per window; the capacity control that separates species "
            "information from FiLM capacity"
        ),
        "use_fm_package": True,
        "use_context_conditioning": False,
        "use_reconstruction_head": False,
        "species_embed_dim": 16,
        "random_species_labels": True,
        "scored_partition": "source_validation",
    },

    # E2 (Q2): start from E1embed, freeze the trained model, and fit one new 16-dim embedding for the held-out species using its 
    # training individuals. Evaluation is on that species' validation individuals. Unlike F, which tunes the full model, E2 adapts 
    # species identity only
    #
    # the higher LR is deliberate because only 16 parameters are being fitted. The fitted embedding norm and loss curve are reported 
    # so failed optimisation is visible
    "E2embed": {
        "description": (
            "E1embed with the whole trunk frozen and one 16-dim embedding fitted per held-out species on that species' training "
            "individuals; evaluated on its validation individuals"
        ),
        "use_fm_package": True,
        "use_context_conditioning": False,
        "use_reconstruction_head": False,
        "species_embed_dim": 16,
        "freeze_except_species_embedding": True,
        "init_from": "E1embed",

        # start from the model's normal random embedding initialisation. `donor_mean` remains a useful follow-up if E2 is flat, to 
        # check that a null result is not just an optimisation artefact
        "embedding_init": "random",

        "per_species": True,
        "training_pool": "attached_species_train",
        "lr": 1e-2,
        "scored_partition": "held_out_validation",
    },
    # ---------------------------------------------------------------- the F budget confound
    # A13(3): F100 and S are registered as differing only in whether the weights were pretrained,
    # but as budgeted they also differ 10x in optimiser steps (2,000 against 20,000) and in
    # checkpoint-selection resolution (two evaluations against twenty). These two arms take the
    # confound apart. Both train on the held-out species' *full* training set, both initialise from
    # Z at the matching seed, and both are scored on held-out VALIDATION because Tier E never opens
    # a test partition -- so neither is comparable to the section 11 curve, which is on test.
    #
    # Flong holds F's learning rate and gives it S's budget: did F simply need longer?
    "Flong": {
        "description": (
            "F at k = 100% with S's 20,000-step budget and F's 1e-4 peak rate; asks whether the "
            "registered 2,000-step budget was the binding constraint"
        ),
        "use_fm_package": True,
        "use_reconstruction_head": True,
        "lr": 1e-4,
        "init_from": "Z",
        "init_from_dir": "tier2_final",
        "per_species": True,
        "training_pool": "attached_species_train",
        "validation_selection": True,
        "scored_partition": "held_out_validation",
    },
    # Spre is S's recipe exactly -- same pool, same budget, same rate, same selection rule -- and
    # differs from S in one thing only: it starts from Z instead of from a random initialisation.
    # That is the comparison sec 3 wanted and the budget denied it.
    "Spre": {
        "description": (
            "S's full recipe -- same pool, 20,000 steps, 1e-3, same selection rule -- initialised "
            "from Z instead of randomly; isolates pretraining with everything else held constant"
        ),
        "use_fm_package": True,
        "use_reconstruction_head": True,
        "init_from": "Z",
        "init_from_dir": "tier2_final",
        "per_species": True,
        "training_pool": "attached_species_train",
        "validation_selection": True,
        "scored_partition": "held_out_validation",
    },
}

# flags a Tier E arm may set on the model or the training loop. Anything else in a spec is bookkeeping for this script and must not 
# silently reach TrainConfig
TIER_E_CONFIG_FLAGS = (
    "use_fm_package",
    "use_context_conditioning",
    "use_reconstruction_head",
    "species_embed_dim",
    "random_species_labels",
    "freeze_except_species_embedding",
)

# Tier E indices 0-19 are trained and reported (results.md section 7). The plan may only ever be appended to, so the first twenty run 
# ids are pinned here and checked at import; a reordering that would silently repoint a finished result at a different arm fails loudly 
# instead
TIER_E_FROZEN_PREFIX: tuple[str, ...] = tuple(
    [
        f"{arm}_s{seed}"
        for arm in ("Zrecon", "Zfloor300", "Brecon", "Zfloor1000", "E1embed", "E1shuffle")
        for seed in SEEDS
    ]
    + [
        f"E2embed_{species}_s{seed}"
        for species, *_ in HELD_OUT
        for seed in SEEDS
    ]
)


def tier_e_scale_floor(run: Run) -> float:
    """
    Return the scale floor a run trains under: the registered 30 m unless a Tier E arm overrides it.
    """
    return float(TIER_E_ARMS.get(run.arm, {}).get("scale_floor_m", SCALE_FLOOR_M))


def build_tier_e_plan() -> list[Run]:
    """
    Return every Tier E variant at each registered seed.

    Most variants are species independent and contribute one block of five seeds. A `per_species` variant is attached to each held-out 
    species in `HELD_OUT` order and contributes fifteen. New variants are appended to `TIER_E_ARMS`, never inserted, so an index that 
    has already been trained keeps pointing at the run that produced it.
    """
    plan: list[Run] = []
    for arm, spec in TIER_E_ARMS.items():
        if spec.get("per_species"):
            for species, *_ in HELD_OUT:
                plan += [Run(arm, species, seed) for seed in SEEDS]
        else:
            plan += [Run(arm, "", seed) for seed in SEEDS]
    return plan


TIER_E_PLAN = build_tier_e_plan()

_prefix = tuple(run.run_id for run in TIER_E_PLAN[: len(TIER_E_FROZEN_PREFIX)])
if _prefix != TIER_E_FROZEN_PREFIX:
    raise RuntimeError(
        "the trained Tier E indices have moved. Expected\n  "
        + "\n  ".join(TIER_E_FROZEN_PREFIX)
        + "\nbut the plan now starts\n  "
        + "\n  ".join(_prefix)
        + "\nAppend to TIER_E_ARMS; never insert or reorder."
    )
del _prefix


def donor_state_dict(run: Run, grid_index: int) -> tuple[dict, str]:
    """
    Load the weights a Tier E arm initialises from, minus the species lookup.

    E2 starts from E1embed at the same seed and fits nothing but a species vector, so the donor's own lookup; three rows, one per 
    source species; is deliberately dropped rather than reused: a held-out species is not one of those three, and seeding its 
    vector from one of them would put a source species' identity into the answer.

    The donor's trunk geometry is checked against this run's, because a silent mismatch would be caught only as a load error several 
    lines later and would read as a code fault rather than as "you have not trained E1embed at this configuration yet".
    """
    variant = TIER_E_ARMS.get(run.arm, {})
    donor_arm = variant["init_from"]
    donor_id = f"{donor_arm}_s{run.seed}"
    # most Tier E arms start from another Tier E arm; the budget arms start from the registered Z,
    # which lives in the Tier 2 directory
    donor_dir = (
        TIER2_FINAL_RESULTS
        if variant.get("init_from_dir") == "tier2_final"
        else TIER_E_RESULTS
    )
    path = donor_dir / f"{donor_id}_reported.pt"
    if not path.exists():
        raise SystemExit(
            f"FAIL: {run.arm} initialises from {donor_id}, but {path} does not exist. Train the "
            f"{donor_arm} block first; E2 is gated on E1 by design."
        )
    state = torch.load(path, map_location="cpu", weights_only=False)
    donor_config = state["config"]
    d_model, layers, _ = GRID[grid_index]
    if (donor_config["d_model"], donor_config["num_layers"]) != (d_model, layers):
        raise SystemExit(
            f"FAIL: {donor_id} was trained at d_model={donor_config['d_model']}, "
            f"{donor_config['num_layers']} layers; this run is at {d_model}, {layers}"
        )
    weights = {k: v for k, v in state["model"].items() if k != "species_embed.weight"}

    how = variant.get("embedding_init", "random")
    if how == "donor_mean":
        # one row, the centroid of the donor's source rows
        weights["species_embed.weight"] = state["model"]["species_embed.weight"].mean(
            dim=0, keepdim=True
        )
    elif how != "random":
        raise SystemExit(f"FAIL: unknown embedding_init {how!r}; use 'random' or 'donor_mean'")
    # only say how the lookup was initialised when the arm actually has one; Flong and Spre carry no
    # species table, and labelling their provenance "random embedding init" would describe a table
    # that does not exist
    if not variant.get("species_embed_dim"):
        return weights, donor_id
    return weights, f"{donor_id} ({how} embedding init)"


def held_out_entry(species: str) -> tuple[str, str, str, str, str]:
    for entry in HELD_OUT:
        if entry[0] == species:
            return entry
    raise SystemExit(f"{species} is not a registered held-out species")


def _assert_source_only(stems) -> None:
    """
    Fail if a Z, B or Za training pool contains anything outside the registered sources.

    This guard enforces the zero-shot data boundary directly at pool construction rather than relying on later review.
    """
    intruders = sorted(set(stems) & HELD_OUT_STEMS)
    if intruders:
        raise SystemExit(
            f"FAIL: {', '.join(intruders)} is a held-out species and cannot enter a Z, B or Za "
            "training pool. Z is zero-shot by construction; this is not a recoverable condition."
        )
    unknown = sorted(set(stems) - SOURCE_STEMS)
    if unknown:
        raise SystemExit(f"FAIL: {', '.join(unknown)} is not a registered source species")


def _load_species_windows(
    stem: str,
    split_stem: str,
    partition: str,
    individuals=None,
) -> SpeciesWindows:
    """
    Build one species' train or validation window index.

    Test partitions are refused unconditionally so sealed evaluation cannot be bypassed through this training script.

    `individuals` restricts the build to a subset of that partition and is used only by F, whose registered rule is a 
    fraction of the held-out species' training individuals. The subset is checked to be a subset: an id that is not in 
    the partition would silently widen the pool F is supposed to be limited of.
    """
    if partition == "test":
        raise SystemExit(
            "FAIL: this script never opens a test partition. Sealed evaluation happens in a "
            "separate script, after the freeze is committed."
        )
    if partition not in {"train", "val"}:
        raise SystemExit(f"FAIL: unknown partition {partition!r}")

    split = load_split(SPLITS / f"{split_stem}_split_seed42.json")
    wanted = getattr(split, partition)
    if individuals is not None:
        intruders = sorted(set(map(str, individuals)) - set(map(str, wanted)))
        if intruders:
            raise SystemExit(
                f"FAIL: {len(intruders)} requested individual(s) are not in {stem}'s {partition} "
                f"partition, e.g. {intruders[:3]}. A fine tuning subset must be drawn from it"
            )
        wanted = individuals

    return SpeciesWindows.build(
        PROCESSED / f"{stem}_segments.parquet",
        wanted,
        context_length=CONTEXT_LENGTH,
        name=stem,
    )


def finetune_subset(run: Run) -> Subset:
    """
    Return the registered fine tuning subset for one F run.

    The draw is a property of `(k, draw index)` alone, so the same draw index gives the same nested
    family every time this is called, and a rerun of one plan index reproduces its own subset.
    """
    stem, split_stem, _, _, _ = held_out_entry(run.species)
    split = load_split(SPLITS / f"{split_stem}_split_seed42.json")
    return nested_subsets(split.train, draw=run.draw)[run.k_percent]


def training_pool(run: Run, device: torch.device) -> tuple[PreparedPool, str]:
    """
    Build runs training pool and return its registered weight node name.

    Z/B use the full source pool, Za uses the registered "leave-one-domain-source-out" sub-pool, and S uses only its held-out species' 
    training individuals with weight 1.0.
    """
    if run.arm == "F":
        stem, split_stem, _, _, _ = held_out_entry(run.species)
        subset = finetune_subset(run)
        windows = _load_species_windows(stem, split_stem, "train", subset.individuals)
        pool = SourcePool({stem: windows}, {stem: 1.0})
        print(
            f"fine tune subset : k = {run.k_percent}% draw {run.draw} -> "
            f"{subset.n_individuals} of {subset.n_available} individuals "
            f"({subset.realised_fraction:.1%} realised)"
            + ("  [raised to the registered floor of 5]" if subset.floored else ""),
            flush=True,
        )
        return (
            PreparedPool.from_source_pool(pool, device=device, sigma=SCALE_FLOOR_M),
            f"finetune:{stem}:k{run.k_percent}:d{run.draw}",
        )

    variant = TIER_E_ARMS.get(run.arm, {})
    if run.arm == "S" or variant.get("training_pool") == "attached_species_train":
        stem, split_stem, _, _, _ = held_out_entry(run.species)
        windows = _load_species_windows(stem, split_stem, "train")
        pool = SourcePool({stem: windows}, {stem: 1.0})
        node = f"specialist:{stem}" if run.arm == "S" else f"attached_species_train:{stem}"
        return (
            PreparedPool.from_source_pool(pool, device=device, sigma=tier_e_scale_floor(run)),
            node,
        )

    if run.arm == "Za":
        _, _, _, _, matching_source = held_out_entry(run.species)
        node = f"sub_pools.without_{matching_source}"
        wanted = [s for s in SOURCES if s[0] != matching_source]
    else:
        node = "source_pool"
        wanted = list(SOURCES)

    _assert_source_only([s[0] for s in wanted])
    species = {
        stem: _load_species_windows(stem, split_stem, "train")
        for stem, split_stem, *_ in wanted
    }
    pool = SourcePool.from_registered_weights(species, WEIGHTS, pool=node)
    return (
        PreparedPool.from_source_pool(pool, device=device, sigma=tier_e_scale_floor(run)),
        node,
    )


def build_config(run: Run, grid_index: int, steps: int, n_species: int = 0) -> TrainConfig:
    """
    Build one run's configuration at the selected grid cell.

    Z, S and Za use the full FM package. B uses the same trunk with conditioning and reconstruction disabled; all other 
    training settings remain shared. A Tier E arm takes its flags from `TIER_E_ARMS` and is exempt
    from the registered package guards, which is the whole point of it.

    `n_species` is the size of the training pool's species list and sizes any learned species lookup. It is carried in the 
    configuration rather than passed at build time so a saved checkpoint rebuilds its own architecture, exactly as every other 
    architectural field already does.
    """
    d_model, layers, lr = GRID[grid_index]
    is_baseline = run.arm == "B"

    if run.arm == "F":
        # F is Z further trained, so the architecture is Z's exactly 
        config = TrainConfig(
            d_model=d_model,
            num_layers=layers,
            lr=FINETUNE_LR_FRACTION * lr,
            steps=FINETUNE_STEPS,
            seed=run.seed,
            use_fm_package=True,
            use_reconstruction_head=True,
        )
        if not (config.use_fm_package and config.use_reconstruction_head):
            raise RuntimeError("F must be same FM package as Z")
        return config

    variant = TIER_E_ARMS.get(run.arm)
    if variant:
        flags = {key: variant[key] for key in TIER_E_CONFIG_FLAGS if key in variant}
        if flags.get("species_embed_dim"):
            flags["n_species"] = n_species
        # a Tier E arm may override the selected learning rate; the registered cell was chosen for the full trunk and does 
        # not have to suit an arm that fits sixteen numbers
        config = TrainConfig(
            d_model=d_model,
            num_layers=layers,
            lr=float(variant.get("lr", lr)),
            steps=steps,
            seed=run.seed,
            **{"use_fm_package": True, "use_reconstruction_head": False, **flags},
        )
        # exploratory by construction; the registered package guards below do not apply to it
        return config

    config = TrainConfig(
        d_model=d_model,
        num_layers=layers,
        lr=lr,
        steps=steps,
        seed=run.seed,
        use_fm_package=not is_baseline,
        use_reconstruction_head=not is_baseline,
    )
    if is_baseline:
        if config.use_fm_package or config.use_reconstruction_head:
            raise RuntimeError("B must be the trunk with the FM package stripped")
    elif not (config.use_fm_package and config.use_reconstruction_head):
        raise RuntimeError(f"{run.arm} must train the complete registered FM package")
    return config


# GenAI reference: assisted with the decision file/CLI override plumbing and defensive checks in this function below
def selected_grid_index(explicit: int | None) -> tuple[int, str]:
    """
    Return the selected grid index and record where that choice came from.

    The inner loop decision file is authoritative. An explicit CLI override is allowed for deliberate reruns and is labelled 
    as such in the result record.
    """
    if explicit is not None:
        if not 0 <= explicit < len(GRID):
            raise SystemExit(f"grid index {explicit} is outside 0..{len(GRID) - 1}")
        print(
            f"WARNING: grid index {explicit} was passed on the command line rather than read from "
            "the selection decision file. It is recorded as `explicit_cli` in every result written "
            "by this run.",
            flush=True,
        )
        return explicit, "explicit_cli"

    path = INNER_LOOP_RESULTS / "tier2_selection.json"
    if not path.exists():
        raise SystemExit(
            f"FAIL: {path} does not exist, so no configuration has been selected by the registered "
            "rule. Run `tier2_inner_loop.py --select` over the 36 grid results first, or pass "
            "--grid-index explicitly if you are deliberately reproducing one cell."
        )
    decision = json.loads(path.read_text())
    index = int(decision["selected_grid_index"])
    if not 0 <= index < len(GRID):
        raise SystemExit(f"{path} names grid index {index}, which is outside the registered grid")
    return index, path.name


# validation scoring used only by arms with registered validation access;
def validation_frame(species: str) -> pd.DataFrame:
    """
    Load one held-out species' validation individuals without touching test data.
    """
    stem, split_stem, _, _, _ = held_out_entry(species)
    split = load_split(SPLITS / f"{split_stem}_split_seed42.json")
    frame = load_split_frame(
        PROCESSED / f"{stem}_segments.parquet",
        split.val,
    )
    return frame[
        ["individual_id", "bin_index", "timestamp", "lon", "lat"]
    ].reset_index(drop=True)


def boundary_m(profile_key: str) -> float:
    table = pd.read_csv(REPO / "results" / "regime_profiles.csv").set_index("species")
    if profile_key not in table.index:
        raise SystemExit(f"{profile_key} has no row in results/regime_profiles.csv")
    return float(table.loc[profile_key, "selected_boundary_m"])


def apply_evaluation_floor(
    model_results: pd.DataFrame,
    persistence_results: pd.DataFrame,
    floor: int = EVALUATION_FLOOR,
) -> tuple[pd.DataFrame, pd.DataFrame, int, int]:
    """
    Apply the A5 evaluation floor to model and persistence results identically.

    The same individual mask is applied to both aligned frames so the row correspondence required by `score_fold` is 
    preserved.
    """
    counts = persistence_results.groupby("individual_id").size()
    keep = set(counts[counts >= floor].index)
    mask = persistence_results["individual_id"].isin(keep).to_numpy()
    return (
        model_results.loc[mask].reset_index(drop=True),
        persistence_results.loc[mask].reset_index(drop=True),
        int(counts.size),
        len(keep),
    )


def score_against_persistence(
    model,
    frame: pd.DataFrame,
    persistence_results: pd.DataFrame,
    profile_key: str,
    label: str,
    device: torch.device,
) -> dict:
    """
    Score one model by individual weighted travel regime p90 skill against persistence.
    """
    predictor = MovementFMPredictor(model, device=device)
    model_results = evaluate_batched(predictor, frame, context_length=CONTEXT_LENGTH)
    model_results, persistence_floored, n_all, n_kept = apply_evaluation_floor(
        model_results,
        persistence_results,
    )
    fold_score = score_fold(
        label,
        model_results,
        persistence_floored,
        boundary_m=boundary_m(profile_key),
    )
    payload = asdict(fold_score)
    payload["n_individuals_scored"] = n_kept
    payload["n_individuals_before_floor"] = n_all
    payload["evaluation_floor"] = EVALUATION_FLOOR
    return payload

# GenAI reference: helped with the CLI/PBS orchestration, status output, checkpoint/result plumbing and defensive guards below; 
# the arm definitions and experiment rules were my own specification; whole file I have read over and also had GenAI check
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--index",
        type=int,
        default=None,
        help="run-plan index; defaults to $PBS_ARRAY_INDEX",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="registered optimiser steps",
    )
    parser.add_argument(
        "--grid-index",
        type=int,
        default=None,
        help="override the selected configuration; recorded as unregistered in the result file",
    )
    parser.add_argument("--list", action="store_true", help="print the run plan and exit")
    parser.add_argument(
        "--tier-e",
        action="store_true",
        help=(
            "run the exploratory Tier E plan instead of the registered one, writing to "
            "results/tier_e and checkpoints/tier_e. Not a preregistered arm and never a Tier 2 result."
        ),
    )
    args = parser.parse_args()

    global RESULTS, CHECKPOINTS
    plan = PLAN
    if args.tier_e:
        # redirect before anything resolves a path, so no Tier E artefact can land in the Tier 2 directories that 
        # report_selection and the sealed evaluation glob
        plan = TIER_E_PLAN
        RESULTS, CHECKPOINTS = TIER_E_RESULTS, TIER_E_CHECKPOINTS

    if args.list:
        return print_plan(plan, tier_e=args.tier_e)

    raw = args.index if args.index is not None else os.environ.get("PBS_ARRAY_INDEX")
    if raw is None:
        raise SystemExit("FAIL: no --index and no $PBS_ARRAY_INDEX in the environment")
    index = int(raw)
    if not 0 <= index < len(plan):
        raise SystemExit(f"index {index} is outside 0..{len(plan) - 1}; see --list")
    if args.steps is None:
        raise SystemExit(
            "FAIL: --steps is required. The optimiser step budget is a registered constant "
            "(preregistration_tier2_tuning.md §5) and must be passed explicitly, not defaulted here."
        )

    run = plan[index]
    grid_index, configuration_source = selected_grid_index(args.grid_index)
    d_model, layers, lr = GRID[grid_index]

    steps = args.steps
    if run.arm == "F":
        steps = FINETUNE_STEPS
        if args.steps != FINETUNE_STEPS:
            print(
                f"NOTE: --steps {args.steps:,} is ignored for F. The registered fine tuning budget "
                f"is {FINETUNE_STEPS:,} optimiser steps at a peak rate of "
                f"{FINETUNE_LR_FRACTION:.0%} of the selected cell's.",
                flush=True,
            )

    RESULTS.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"run              : {run.run_id}  (plan index {index} of {len(PLAN) - 1})")
    print(
        f"arm              : {run.arm}"
        + (f", attached to {run.species}" if run.species else "")
    )
    if run.deferred:
        print("                   registered but deferred; this arm is not on the pre-meeting path")
    print(
        f"configuration    : grid index {grid_index} "
        f"(d_model={d_model}, layers={layers}, lr={lr:g})"
    )
    print(f"                   from {configuration_source}")
    print(f"device           : {device}", flush=True)
    if device.type == "cuda":
        print(f"gpu              : {torch.cuda.get_device_name(0)}", flush=True)

    started = time.time()
    pool, pool_node = training_pool(run, device)
    print(
        f"pool             : {pool.n_windows:,} windows over {list(pool.names)} [{pool_node}], "
        f"{pool.bytes_on_device() / 1e6:.0f} MB on device, built in {time.time() - started:.0f}s",
        flush=True,
    )

    config = build_config(run, grid_index, steps, n_species=len(pool.names))
    checkpoint = CHECKPOINTS / f"{run.run_id}_g{grid_index:02d}_{config.steps}.pt"

    variant = TIER_E_ARMS.get(run.arm, {})
    initialisation: dict = {}
    donor_id = None
    if run.arm == "F":
        donor_id = f"{FINETUNE_INIT_ARM}_s{FINETUNE_INIT_SEED}"
        donor_path = RESULTS / f"{donor_id}_reported.pt"
        if not donor_path.exists():
            raise SystemExit(
                f"FAIL: F is Z further trained, but {donor_path} does not exist. F cannot be run "
                "before the arm it fine-tunes."
            )
        donor = torch.load(donor_path, map_location="cpu", weights_only=False)
        if donor["config"]["d_model"] != d_model or donor["config"]["num_layers"] != layers:
            raise SystemExit(
                f"FAIL: {donor_id} was trained at d_model={donor['config']['d_model']}, "
                f"{donor['config']['num_layers']} layers; this run is at {d_model}, {layers}"
            )
        initialisation["init_state_dict"] = donor["model"]
        print(
            f"initialised from : {donor_id} (registered: F_k is Z further trained; the seed is "
            "declared, see FINETUNE_INIT_SEED)",
            flush=True,
        )
    elif variant.get("init_from"):
        weights, donor_id = donor_state_dict(run, grid_index)
        initialisation["init_state_dict"] = weights
        print(f"initialised from : {donor_id} (species lookup dropped, refitted here)", flush=True)
    if config.lr != lr:
        print(
            f"learning rate    : {config.lr:g}, overriding the selected cell's {lr:g}. Exploratory "
            "Tier E override, recorded in the result file.",
            flush=True,
        )
    if config.species_embed_dim:
        print(
            f"species lookup   : {config.n_species} x {config.species_embed_dim}"
            + ("  [labels drawn at random -- control arm]" if config.random_species_labels else "")
            + ("  [trunk frozen]" if config.freeze_except_species_embedding else ""),
            flush=True,
        )

    monitor, selection = _make_monitor(run, device, grid_index, config.steps)
    result = train(
        config,
        pool,
        device=device,
        checkpoint_path=checkpoint,
        progress=lambda step, loss: print(
            f"  step {step:>7,}  loss {loss:.5f}",
            flush=True,
        ),
        **monitor,
        **initialisation,
    )

    print(f"\nfinal loss       : {result.final_loss:.5f}")
    # T3/T4: the four clocks. `applied` is the registered budget and must equal it; the gap between
    # attempts and applied is the withholding, and is the number the freeze reports.
    print(
        f"applied updates  : {result.applied_updates:,} of {config.steps:,} registered\n"
        f"attempted batches: {result.attempted_batches:,}\n"
        f"withheld         : {result.skipped_steps:,} ({result.withheld_fraction:.2%} of attempts) -- T3"
    )
    if result.skipped_steps:
        print(
            "WARNING: this run withheld optimiser steps. T4 means the budget was still reached, but "
            "the numerical path is not clean and the count belongs in the freeze.",
            flush=True,
        )

    reported_model, selection_payload = _resolve_reported_model(run, result, selection)

    # resolved once; the draw is deterministic, so this is the same subset the pool was built from
    subset = finetune_subset(run) if run.arm == "F" else None

    # Sealed evaluation reads this explicit reported checkpoint rather than reconstructing a filename.
    reported_checkpoint = RESULTS / f"{run.run_id}_reported.pt"
    torch.save(
        {
            "run_id": run.run_id,
            "config": asdict(config),
            "grid_index": grid_index,
            "selection": {k: v for k, v in selection_payload.items() if k != "history"},
            # the pool's species order is what a learned species lookup is indexed by, and it is
            # `sorted(weights)`, not the order SOURCES happens to be written in. Scoring a species
            # under the wrong row would be silent, so the mapping travels with the weights.
            "pool_species": list(pool.names),
            "model": reported_model.state_dict(),
        },
        reported_checkpoint,
    )

    payload = {
        "run_id": run.run_id,
        "plan_index": index,
        "arm": run.arm,
        "species": run.species or None,
        "seed": run.seed,
        "deferred_arm": run.deferred,
        "grid_index": grid_index,
        "configuration_source": configuration_source,
        "config": asdict(config),
        "n_parameters": reported_model.n_parameters(),
        "training_pool": pool_node,
        "pool_species": list(pool.names),
        "pool_windows": pool.n_windows,
        # the preregistration requires the realised individual and window counts alongside every k,
        # because the same fraction means a different absolute amount of data on each species
        "finetune": (
            {
                "k_percent": run.k_percent,
                "draw": run.draw,
                "draw_seed": subset.seed,
                "n_individuals": subset.n_individuals,
                "n_individuals_available": subset.n_available,
                "realised_fraction": subset.realised_fraction,
                "raised_to_floor": subset.floored,
                "n_windows": pool.n_windows,
                "initialised_from_arm": FINETUNE_INIT_ARM,
                "initialised_from_seed": FINETUNE_INIT_SEED,
                "registered_steps": FINETUNE_STEPS,
                "peak_lr": config.lr,
            }
            if run.arm == "F"
            else None
        ),
        "final_loss": result.final_loss,
        "loss_history": result.loss_history,
        "loss_shares": result.loss_shares,
        # T3/T4 diagnostics carried into the freeze.
        "skipped_steps": result.skipped_steps,
        "attempted_batches": result.attempted_batches,
        "applied_updates": result.applied_updates,
        "withheld_fraction": result.withheld_fraction,
        "nonfinite_steps": result.nonfinite_steps,
        "checkpoint_selection": selection_payload,
        "reported_checkpoint": reported_checkpoint.name,
        "opened_test_partition": False,
        # tier E artefacts are marked in the file as well as by their directory, so a stray copy
        # cannot be mistaken for a registered result
        "tier": "E" if args.tier_e else "2",
        "exploratory": bool(args.tier_e),
        "tier_e_variant": TIER_E_ARMS.get(run.arm, {}).get("description") if args.tier_e else None,
        "tier_e_scored_partition": variant.get("scored_partition") if args.tier_e else None,
        "initialised_from": donor_id,
        # a frozen-trunk arm fits one vector, so "did it move" is the first thing to check before
        # reading any score off it
        "trainable_parameters": sum(
            p.numel() for p in reported_model.parameters() if p.requires_grad
        ),
        "species_embedding": (
            {
                "dim": config.species_embed_dim,
                "n_rows": config.n_species,
                "row_norms": [
                    float(row.norm()) for row in reported_model.species_embed.weight.detach()
                ],
                "random_labels": config.random_species_labels,
            }
            if config.species_embed_dim
            else None
        ),
        # the representation this model was trained under; evaluation must use the same value
        "scale_floor_m": tier_e_scale_floor(run),
        # environment provenance: the attention backend is pinned in train() (PyTorch #125674)
        "sdpa_backend": SDPA_BACKEND,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "elapsed_s": time.time() - started,
    }
    out = RESULTS / f"{run.run_id}.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nwrote {out}  ({payload['elapsed_s'] / 60:.1f} min)")

# GenAI reference: assisted with the resumable monitor/history/checkpoint plumbing implementing that rule and I did the S 
# selects by validation travel regime p90 at the fixed; and also added the 2 functions below 
def _make_monitor(
    run: Run,
    device: torch.device,
    grid_index: int,
    steps: int,
) -> tuple[dict, dict]:
    """
    Build the periodic validation hook for arms allowed to select checkpoints.

    Z, B and Za receive no hook and never open validation data. For S, validation history is persisted after each evaluation 
    so a requeued PBS job can recover the same checkpoint selection state.
    """
    selection: dict = {"rule": None, "history": [], "best": None}
    if not run.uses_validation_selection:
        selection["rule"] = "final_checkpoint"
        return {}, selection

    stem, _, profile_key, _, _ = held_out_entry(run.species)
    identity = {
        "run_id": run.run_id,
        "grid_index": grid_index,
        "steps": steps,
        "rule": SELECTION_RULE,
        "every": SELECTION_EVERY,
        "floor": EVALUATION_FLOOR,
    }
    history_path = CHECKPOINTS / f"{run.run_id}_selection.json"
    if history_path.exists():
        stored = json.loads(history_path.read_text())
        if stored.get("identity") == identity:
            selection["history"] = stored["history"]
            if selection["history"]:
                selection["best"] = min(
                    selection["history"],
                    key=lambda row: row["model_p90_m"],
                )
                print(
                    f"resumed          : {len(selection['history'])} validation evaluations "
                    f"recovered, best so far step {selection['best']['step']:,}",
                    flush=True,
                )
        else:
            print(
                f"discarding {history_path.name}: it belongs to a different configuration",
                flush=True,
            )

    frame = validation_frame(run.species)
    print(
        f"validation frame : {len(frame):,} fixes, {frame['individual_id'].nunique()} individuals "
        f"({stem}); used for checkpoint selection only",
        flush=True,
    )
    # Persistence is deterministic; score it once and reuse it for both comparison and travel labels.
    persistence_results = evaluate(
        Persistence(), frame, context_length=CONTEXT_LENGTH, show_progress=False
    )
    selection["rule"] = SELECTION_RULE
    selection["every"] = SELECTION_EVERY
    best = CHECKPOINTS / f"{run.run_id}_best.pt"

    def monitor(step: int, model) -> None:
        entry = score_against_persistence(
            model,
            frame,
            persistence_results,
            profile_key,
            stem,
            device,
        )
        entry["step"] = int(step)
        selection["history"].append(entry)
        incumbent = selection["best"]
        # Persistence is fixed across the run, so ranking by model p90 and skill is equivalent.
        if incumbent is None or entry["model_p90_m"] < incumbent["model_p90_m"]:
            selection["best"] = entry
            best.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "step": int(step),
                    "model": copy.deepcopy(model.state_dict()),
                },
                best,
            )
        history_path.parent.mkdir(parents=True, exist_ok=True)
        history_path.write_text(
            json.dumps({"identity": identity, "history": selection["history"]}, indent=2) + "\n"
        )
        print(
            f"  step {step:>7,}  val travel p90 {entry['model_p90_m']:>10,.0f} m  "
            f"skill {entry['skill']:+.4f}"
            + ("  <- best" if selection["best"] is entry else ""),
            flush=True,
        )

    return {"monitor": monitor, "monitor_every": SELECTION_EVERY}, selection


def _resolve_reported_model(run: Run, result, selection: dict):
    """
    Return the model reported by this run and the checkpoint-selection record.

    Z, B and Za report the final checkpoint. S reports the checkpoint with minimum validation travel regime p90.
    """
    if not run.uses_validation_selection:
        return result.model, {"rule": "final_checkpoint", "step": result.config.steps}

    best = selection.get("best")
    if best is None:
        raise SystemExit(
            "FAIL: S completed without a single validation evaluation, so no checkpoint could be "
            "selected. Check that the step budget is at least SELECTION_EVERY."
        )
    state = torch.load(
        CHECKPOINTS / f"{run.run_id}_best.pt",
        map_location="cpu",
        weights_only=False,
    )
    result.model.load_state_dict(state["model"])
    print(
        f"\nselected         : step {best['step']:,} of {result.config.steps:,}, "
        f"validation travel p90 {best['model_p90_m']:,.0f} m"
    )
    return result.model, {
        "rule": SELECTION_RULE,
        "every": SELECTION_EVERY,
        "evaluation_floor": EVALUATION_FLOOR,
        "selected_step": best["step"],
        "selected_validation_travel_p90_m": best["model_p90_m"],
        "history": selection["history"],
    }


def print_plan(plan: list[Run] | None = None, *, tier_e: bool = False) -> None:
    # GenAI reference: did all of this; I checked it 
    plan = PLAN if plan is None else plan
    print(f"{'idx':>4}  {'arm':<9} {'species':<24} {'seed':>4} {'k':>5} {'draw':>5}  path")
    for index, run in enumerate(plan):
        print(
            f"{index:>4}  {run.arm:<9} {run.species or '-':<24} {run.seed:>4} "
            f"{('-' if run.k_percent is None else f'{run.k_percent}%'):>5} "
            f"{('-' if run.draw is None else run.draw):>5}  "
            + ("deferred" if run.deferred else "critical")
        )
    if tier_e:
        print(f"\n{len(plan)} exploratory Tier E runs (0-{len(plan) - 1}).")
        for arm, spec in TIER_E_ARMS.items():
            print(f"  {arm:<7} {spec['description']}")
        print(
            "Governed by tierE_exploratory_ablations.md. Validation individuals only; no test "
            "partition, no success band, never a Tier 2 result."
        )
        return
    print(
        f"\n{N_CRITICAL_PATH} critical-path runs (0-{N_CRITICAL_PATH - 1}), "
        f"{len(PLAN) - N_CRITICAL_PATH} deferred "
        f"({N_CRITICAL_PATH}-{len(PLAN) - 1})."
    )
    print("P and V are untrained and are produced by the sealed-evaluation script.")


if __name__ == "__main__":
    main()
