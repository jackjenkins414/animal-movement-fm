# GenAI reference: This script was generated with the help of GenAI in sections, as with our other train scripts in this folder. Will
# add the details later. 
"""
Train the 1 h rebuild's arms for the 30 rotation birds, one model per PBS subjob (exploratory).

Direct horizons 1, 6 and 24 h from 12 contiguous 1 h bins; the context ablation extends the history to C = 24-96 bins, padded where 
bins are empty. Plan: --list. Data checks: --dry-run. Walltime and throughput: --calibrate.
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import json
import math
import resource
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO / "experiments" / "reporting"))
sys.path.insert(0, str(REPO))

import bird_rotation_train as RT  # noqa: E402
import tier2_sealed_eval as S2  # noqa: E402

from animal_movement_fm.data.enu import FEATURES, SCALE_FLOOR_M, enu_to_lonlat_exact  # noqa: E402
from animal_movement_fm.data.multispecies import HorizonWindows, SourcePool  # noqa: E402
from animal_movement_fm.data.split import load_split  # noqa: E402
from animal_movement_fm.eval.metrics import haversine_distance  # noqa: E402
from animal_movement_fm.train.loop import TrainConfig, train  # noqa: E402
from animal_movement_fm.train.loss import horizon_huber, radial_pseudo_huber  # noqa: E402
from animal_movement_fm.train.pool import PreparedPool, StreamingPool, core_matches, horizon_k  # noqa: E402

BT, T, SPW = RT.BT, RT.T, RT.SPW

SEGMENTS = REPO / "data" / "processed_1h"
BUILD = REPO / "results" / "birds_1h_build"
HORIZON_COUNTS = REPO / "results" / "cadence" / "horizon_counts.csv"
RESULTS = REPO / "results" / "birds_1h"
CHECKPOINTS = REPO / "checkpoints" / "birds_1h"
CALIBRATION = REPO / "results" / "birds_1h_calibration"
FREEZE = "birds_1h_exploratory_freeze.md"

BIN_SECONDS = 3_600
CORE = 12  # contiguous bins ending at the anchor; the main study's whole context
HORIZONS = (1, 6, 24)  # bins, direct
ABLATION_CONTEXTS = (24, 48, 72, 96)
CLIP = 1_000.0  # on d_east_scaled and d_north_scaled, every token, both studies (09-10)
FLOOR = 150  # windows per individual per horizon (150 h, = F 25 at 6 h)
MIN_SCORED_INDIVIDUALS = (
    5  # a bird is scored at h with at least this many test_scored individuals at the floor
)

REGISTERED_STEPS = RT.REGISTERED_STEPS  # 20,000 applied updates at batch 256
CALIBRATION_STEPS = 1_000
SEEDS = RT.SEEDS
XGB_SEED = RT.XGB_SEED
XGB_POOLED_DRAWS = RT.XGB_POOLED_DRAWS  # A8 draws per horizon, from windows with a target at h
SELECTION_EVERY = T.SELECTION_EVERY

CLIP_SAMPLE = 20_000  # windows per species for the clip shares
CORE_CHECK_SAMPLE = 2_000  # windows per species for core_matches
LOSS_PROBE = 4_096  # fixed training windows for calibration's per-head losses
PROBE_SEED = 123
GATHER_CHUNK = 50_000

_D_COLUMNS = [FEATURES.index("d_east_scaled"), FEATURES.index("d_north_scaled")]

# all ten folds; fold 0 is the bird study's gull, goose and dove, trained fresh here
FOLDS: dict[int, tuple[str, ...]] = {k: tuple(sorted(v)) for k, v in SPW.BIRDS_ROTATION_FOLDS.items()}
BIRDS: tuple[str, ...] = tuple(name for k in sorted(FOLDS) for name in FOLDS[k])  # fold order, then name
FOLD_OF: dict[str, int] = {name: k for k, members in FOLDS.items() for name in members}
if len(BIRDS) != 30 or set(FOLDS) != set(range(10)):
    raise RuntimeError("the 1 h study holds out 30 birds over folds 0-9")

BOUNDARY_M: dict[tuple[str, int], float] = {
    ("anas_platyrhynchos", 1): 30.0,
    ("anas_platyrhynchos", 6): 30.0,
    ("anas_platyrhynchos", 24): 30.0,
    ("anser_albifrons", 1): 30.0,
    ("anser_albifrons", 6): 74.4079373123508,
    ("anser_albifrons", 24): 108.20330758497148,
    ("anser_anser", 1): 30.0,
    ("anser_anser", 6): 50.172093984701405,
    ("anser_anser", 24): 36.34606558056062,
    ("anser_brachyrhynchus", 1): 30.0,
    ("anser_brachyrhynchus", 6): 102.79909112770557,
    ("anser_brachyrhynchus", 24): 95.74420697934391,
    ("anser_caerulescens", 1): 30.0,
    ("anser_caerulescens", 6): 105.1250883974752,
    ("anser_caerulescens", 24): 272.3405001465957,
    ("anser_fabalis", 1): 30.0,
    ("anser_fabalis", 6): 57.06584468880615,
    ("anser_fabalis", 24): 83.87311075360614,
    ("ardea_alba", 1): 30.0,
    ("ardea_alba", 6): 60.87236651792869,
    ("ardea_alba", 24): 30.0,
    ("branta_canadensis", 1): 30.0,
    ("branta_canadensis", 6): 37.39208574322446,
    ("branta_canadensis", 24): 46.474049705699166,
    ("branta_leucopsis", 1): 30.0,
    ("branta_leucopsis", 6): 101.17841846139322,
    ("branta_leucopsis", 24): 104.20568999289152,
    ("cathartes_aura", 1): 30.0,
    ("cathartes_aura", 6): 178.75914537182933,
    ("cathartes_aura", 24): 338.0731055022115,
    ("ciconia_ciconia", 1): 30.0,
    ("ciconia_ciconia", 6): 105.63938783625902,
    ("ciconia_ciconia", 24): 81.97816127701604,
    ("corvus_corone", 1): 30.0,
    ("corvus_corone", 6): 30.0,
    ("corvus_corone", 24): 30.0,
    ("diomedea_exulans", 1): 1193.3748663999745,
    ("diomedea_exulans", 6): 7223.27840494134,
    ("diomedea_exulans", 24): 27190.580017120377,
    ("falco_naumanni", 1): 30.0,
    ("falco_naumanni", 6): 215.87172970901153,
    ("falco_naumanni", 24): 123.2128819723826,
    ("falco_tinnunculus", 1): 30.0,
    ("falco_tinnunculus", 6): 48.353994369232794,
    ("falco_tinnunculus", 24): 30.0,
    ("gyps_himalayensis", 1): 30.0,
    ("gyps_himalayensis", 6): 330.03710929645484,
    ("gyps_himalayensis", 24): 1678.6807317453815,
    ("haematopus_ostralegus", 1): 30.0,
    ("haematopus_ostralegus", 6): 140.2211349638714,
    ("haematopus_ostralegus", 24): 37.60105435882638,
    ("hydroprogne_caspia", 1): 30.0,
    ("hydroprogne_caspia", 6): 444.8551660750345,
    ("hydroprogne_caspia", 24): 106.48904022033368,
    ("larus_argentatus", 1): 30.0,
    ("larus_argentatus", 6): 168.08109560701357,
    ("larus_argentatus", 24): 61.04463731321379,
    ("larus_fuscus", 1): 30.0,
    ("larus_fuscus", 6): 449.53996296650257,
    ("larus_fuscus", 24): 299.64475084429324,
    ("milvus_milvus", 1): 39.68841359306967,
    ("milvus_milvus", 6): 97.478672857635,
    ("milvus_milvus", 24): 83.23225439411519,
    ("morus_bassanus", 1): 103.0077814674288,
    ("morus_bassanus", 6): 1197.9747963816524,
    ("morus_bassanus", 24): 6219.51194183663,
    ("mycteria_americana", 1): 30.0,
    ("mycteria_americana", 6): 90.6347082870234,
    ("mycteria_americana", 24): 133.33696892483655,
    ("pelecanus_occidentalis", 1): 30.0,
    ("pelecanus_occidentalis", 6): 649.227595001907,
    ("pelecanus_occidentalis", 24): 938.1873960053484,
    ("pernis_apivorus", 1): 30.0,
    ("pernis_apivorus", 6): 65.82486716765281,
    ("pernis_apivorus", 24): 131.49175959527835,
    ("phoenicopterus_roseus", 1): 30.0,
    ("phoenicopterus_roseus", 6): 34.24563838789084,
    ("phoenicopterus_roseus", 24): 39.25071737776705,
    ("platalea_leucorodia", 1): 30.0,
    ("platalea_leucorodia", 6): 65.74716351246067,
    ("platalea_leucorodia", 24): 43.793643911932726,
    ("streptopelia_turtur", 1): 30.0,
    ("streptopelia_turtur", 6): 30.0,
    ("streptopelia_turtur", 24): 30.0,
    ("thalassarche_chrysostoma", 1): 952.6879705444834,
    ("thalassarche_chrysostoma", 6): 4248.961963375458,
    ("thalassarche_chrysostoma", 24): 12801.226758773066,
    ("thalassoica_antarctica", 1): 856.9011613586374,
    ("thalassoica_antarctica", 6): 4447.027684072229,
    ("thalassoica_antarctica", 24): 12693.7988297849,
}
FOLD0_PROBABILITIES: dict[str, float] = {
    "anas_platyrhynchos": 0.04901706921254385,
    "ardea_alba": 0.028132352965355174,
    "branta_canadensis": 0.0609893044056603,
    "branta_leucopsis": 0.0452695385350934,
    "cathartes_aura": 0.08379940524341223,
    "ciconia_ciconia": 0.17053911609243766,
    "corvus_corone": 0.06460396534076186,
    "diomedea_exulans": 0.011249824955363289,
    "falco_naumanni": 0.034545771207079835,
    "falco_tinnunculus": 0.024041615049197922,
    "gyps_himalayensis": 0.015904965370966865,
    "haematopus_ostralegus": 0.06892235845946633,
    "hydroprogne_caspia": 0.05915353701805389,
    "milvus_milvus": 0.030679401603443388,
    "morus_bassanus": 0.008382899979053797,
    "mycteria_americana": 0.07121511662781281,
    "pelecanus_occidentalis": 0.016201215942445696,
    "pernis_apivorus": 0.02897556651010868,
    "phoenicopterus_roseus": 0.052326963193724756,
    "platalea_leucorodia": 0.056337912735769095,
    "thalassarche_chrysostoma": 0.009932685243384608,
    "thalassoica_antarctica": 0.009779414308864516,
}
# training windows after exclusions: anchors with any target, then those with a target at 1, 6 and 24 h
TRAIN_WINDOWS: dict[str, tuple[int, int, int, int]] = {
    "anas_platyrhynchos": (304268, 301876, 299758, 296842),
    "anser_albifrons": (340157, 336311, 332870, 327941),
    "anser_anser": (445757, 443345, 441217, 439556),
    "anser_brachyrhynchus": (14876, 14697, 14368, 13760),
    "anser_caerulescens": (1067849, 1045375, 979267, 1056526),
    "anser_fabalis": (108981, 108527, 107667, 106260),
    "ardea_alba": (122318, 122166, 121711, 121540),
    "branta_canadensis": (456520, 452408, 448330, 446995),
    "branta_leucopsis": (293004, 291854, 290419, 287650),
    "cathartes_aura": (437910, 393920, 227746, 390054),
    "ciconia_ciconia": (1868190, 1694005, 1026627, 1848388),
    "corvus_corone": (511846, 507815, 503099, 499264),
    "diomedea_exulans": (17434, 17369, 17000, 15656),
    "falco_naumanni": (183503, 181818, 178752, 175889),
    "falco_tinnunculus": (41535, 37414, 30185, 38270),
    "gyps_himalayensis": (47832, 45817, 42732, 44201),
    "haematopus_ostralegus": (696047, 689658, 687109, 681038),
    "hydroprogne_caspia": (523020, 521108, 519242, 516598),
    "larus_argentatus": (1845484, 1839099, 1827225, 1815622),
    "larus_fuscus": (3594803, 3584880, 3566613, 3547539),
    "milvus_milvus": (47133, 39189, 16753, 45645),
    "morus_bassanus": (11746, 11523, 10769, 10393),
    "mycteria_americana": (148350, 119864, 5226, 134167),
    "pelecanus_occidentalis": (14051, 12369, 5379, 13602),
    "pernis_apivorus": (20569, 17735, 13494, 19456),
    "phoenicopterus_roseus": (400828, 398953, 396265, 396144),
    "platalea_leucorodia": (493398, 492334, 490103, 491626),
    "streptopelia_turtur": (182333, 181253, 178867, 178844),
    "thalassarche_chrysostoma": (17926, 17826, 17480, 16208),
    "thalassoica_antarctica": (20210, 20150, 19656, 17925),
}
if set(TRAIN_WINDOWS) != set(BIRDS) or set(BOUNDARY_M) != {(b, h) for b in BIRDS for h in HORIZONS}:
    raise RuntimeError(
        "the pinned counts and boundaries do not cover exactly the 30 birds and three horizons"
    )

# arm -> the bird study's recipe key (bird_train.ARMS), its training data, and its family
ARMS: dict[str, dict] = {
    "Bb": {"pool": "fold", "xgb": False},
    "LSTMBb": {"pool": "fold", "xgb": False},
    "LSTMS": {"pool": "held_out_train", "xgb": False},
    "XGBBb": {"pool": "fold", "xgb": True},
    "XGBS": {"pool": "held_out_train", "xgb": True},
}


@dataclass(frozen=True)
class Run:
    arm: str
    fold: int
    species: str
    seed: int
    horizon: int = 0  # XGB only: the one horizon its regressor predicts; neural runs predict all three
    context: int = CORE

    # bird_train.counterpart reads these; nothing here is a subset run
    k_percent = None
    draw = None

    @property
    def run_id(self) -> str:
        subject = f"{self.arm}_{self.species}" if self.species else f"{self.arm}_f{self.fold}"
        suffix = f"_h{self.horizon}" if self.horizon else f"_s{self.seed}"
        return subject + suffix + (f"_C{self.context}" if self.context != CORE else "")

    @property
    def xgb(self) -> bool:
        return ARMS[self.arm]["xgb"]

    @property
    def uses_validation_selection(self) -> bool:
        return self.arm == "LSTMS"


def build_plan() -> list[Run]:
    """The 1 h plan. Position is the PBS array index: append, never insert or reorder."""
    plan = []
    for arm in ("Bb", "LSTMBb"):
        plan += [Run(arm, k, "", s) for k in sorted(FOLDS) for s in SEEDS]
    plan += [Run("LSTMS", FOLD_OF[b], b, s) for b in BIRDS for s in SEEDS]
    plan += [Run("XGBBb", k, "", XGB_SEED, h) for k in sorted(FOLDS) for h in HORIZONS]
    plan += [Run("XGBS", FOLD_OF[b], b, XGB_SEED, h) for b in BIRDS for h in HORIZONS]
    # the context ablation: Bb then LSTM-Bb, each C x fold x seed, then XGB-Bb C x fold x horizon
    for arm in ("Bb", "LSTMBb"):
        plan += [Run(arm, k, "", s, 0, c) for c in ABLATION_CONTEXTS for k in sorted(FOLDS) for s in SEEDS]
    plan += [
        Run("XGBBb", k, "", XGB_SEED, h, c)
        for c in ABLATION_CONTEXTS
        for k in sorted(FOLDS)
        for h in HORIZONS
    ]
    return plan


PLAN = build_plan()
# (first, last, queue, what): every PBS array targets one queue
BLOCKS: tuple[tuple[int, int, str, str], ...] = (
    (0, 29, "gpuvolta", "Bb, 10 folds x 3 seeds"),
    (30, 59, "gpuvolta", "LSTM-Bb, 10 folds x 3 seeds"),
    (60, 149, "gpuvolta", "LSTM-S, 30 birds x 3 seeds"),
    (150, 179, "normal", "XGB-Bb, 10 folds x 3 horizons"),
    (180, 269, "normal", "XGB-S, 30 birds x 3 horizons"),
    (270, 389, "gpuvolta", "ablation Bb, 4 C x 10 folds x 3 seeds"),
    (390, 509, "gpuvolta", "ablation LSTM-Bb, 4 C x 10 folds x 3 seeds"),
    (510, 629, "normal", "ablation XGB-Bb, 4 C x 10 folds x 3 horizons"),
)
if (
    len(PLAN) != 630
    or [b[0] for b in BLOCKS] != [0] + [b[1] + 1 for b in BLOCKS[:-1]]
    or BLOCKS[-1][1] != 629
):
    raise RuntimeError(
        f"the 1 h plan has {len(PLAN)} runs; the build plan records 630 in contiguous queue blocks"
    )


def queue_of(index: int) -> str:
    return next(queue for first, last, queue, _ in BLOCKS if first <= index <= last)


if any(queue_of(i) != ("normal" if run.xgb else "gpuvolta") for i, run in enumerate(PLAN)):
    raise RuntimeError("a block's queue does not match its runs: neural on gpuvolta, XGBoost on normal")


# Pins, checked on every load.
def check_boundaries() -> dict[tuple[str, int], float]:
    """The 90 pinned travel boundaries against boundaries.csv: max(0.1 x p50, the 30 m scale floor) per bird and h."""
    table = pd.read_csv(BUILD / "boundaries.csv").set_index(["species", "horizon"])
    if set(table.index) != set(BOUNDARY_M):
        raise SystemExit("FAIL: boundaries.csv does not hold exactly the 90 pinned bird x horizon rows")
    for (bird, h), pinned in BOUNDARY_M.items():
        row = table.loc[(bird, h)]
        if not math.isclose(float(row["boundary_m"]), pinned, rel_tol=1e-12, abs_tol=0.0):
            raise SystemExit(
                f"FAIL: {bird} {h} h boundary is pinned at {pinned!r} m, boundaries.csv has "
                f"{float(row['boundary_m'])!r} m"
            )
        if not math.isclose(max(float(row["a4_m"]), SCALE_FLOOR_M), pinned, rel_tol=1e-12, abs_tol=0.0):
            raise SystemExit(f"FAIL: {bird} {h} h boundary is not max(A4, {SCALE_FLOOR_M:g} m)")
    return dict(BOUNDARY_M)


def check_fold0(payload: dict) -> dict[str, float]:
    """Fold 0's A8 probabilities: pinned, the build's file and the rotation rule recomputed must all agree."""
    import experiments.data.build_1h_segments as b1

    sources = {
        "fold0_probabilities.json": json.loads((BUILD / "fold0_probabilities.json").read_text()),
        "build_1h_segments.fold0_probabilities": b1.fold0_probabilities(payload),
    }
    for label, values in sources.items():
        if set(values) != set(FOLD0_PROBABILITIES) or any(
            not math.isclose(values[n], p, rel_tol=1e-12, abs_tol=0.0) for n, p in FOLD0_PROBABILITIES.items()
        ):
            raise SystemExit(f"FAIL: fold 0's pinned A8 probabilities differ from {label}")
    if set(FOLD0_PROBABILITIES) != set(payload["pools"]["0"]):
        raise SystemExit("FAIL: fold 0's pinned probabilities do not cover its registered pool")
    return dict(FOLD0_PROBABILITIES)


def check_horizon_counts() -> None:
    """The pinned 1 h training counts against horizon_counts.csv (training individuals after exclusions)."""
    counts = pd.read_csv(HORIZON_COUNTS, dtype={"horizon": str})
    counts = counts[counts["partition"] == "train"].set_index(["species", "horizon"])["windows"]
    for bird, pinned in TRAIN_WINDOWS.items():
        actual = tuple(int(counts[bird, key]) for key in ("any", "1", "6", "24"))
        if actual != pinned:
            raise SystemExit(
                f"FAIL: {bird}'s pinned 1 h training counts {pinned} differ from horizon_counts.csv {actual}"
            )


def load_registered() -> dict:
    """The registered rotation folds (checked by bird_rotation_train), plus this study's pins."""
    payload = RT.load_registered_folds()
    check_fold0(payload)
    check_boundaries()
    check_horizon_counts()
    return payload


def fold_probabilities(fold: int, payload: dict) -> dict[str, float]:
    return dict(FOLD0_PROBABILITIES) if fold == 0 else dict(payload["probabilities"][str(fold)])


def assert_fold_sources(fold: int, stems) -> None:
    """bird_rotation_train's rule for folds 1-9, applied to fold 0 as well."""
    if fold != 0:
        return RT.assert_fold_sources(fold, stems)
    stems, members = set(stems), set(FOLDS[0])
    genera = {SPW.genus(n) for n in members}
    leaks = sorted(n for n in stems if n in members or SPW.genus(n) in genera or n in SPW.BIRDS_RESERVED)
    if leaks:
        raise SystemExit(
            f"FAIL: fold 0's pool would draw {leaks}, a held-out bird, genus-mate or reserved species"
        )


# Training data.
def split_ids(stem: str, partition: str, payload: dict) -> list:
    """Training individuals (after exclusions) or validation individuals; the test partition is never read here."""
    if partition == "train":
        return RT.training_ids(stem, payload)
    if partition == "val":
        return list(load_split(T.SPLITS / f"{SPW.SPLIT_STEM.get(stem, stem)}_split_seed42.json").val)
    raise SystemExit(
        f"FAIL: the training driver reads train and val only, never {partition!r}; "
        "test is read by the sealed evaluator after the freeze"
    )


def horizon_windows(stem: str, ids, context: int) -> HorizonWindows:
    return HorizonWindows.build(
        SEGMENTS / f"{stem}_segments.parquet",
        ids,
        context=context,
        core=CORE,
        horizons=HORIZONS,
        bin_seconds=BIN_SECONDS,
        clip=CLIP,
        name=stem,
    )


def check_window_counts(pool: SourcePool) -> None:
    """Every species' built anchors and per-horizon targets equal its pinned 1 h training counts."""
    drift = []
    for name in pool.names:
        windows = pool.species[name]
        actual = (windows.n_windows, *(int((windows.targets[:, j] >= 0).sum()) for j in range(len(HORIZONS))))
        if actual != TRAIN_WINDOWS[name]:
            drift.append(f"{name}: built {actual}, pinned {TRAIN_WINDOWS[name]}")
    if drift:
        raise SystemExit("FAIL: 1 h training windows differ from the pinned counts\n  " + "\n  ".join(drift))


def training_source(run: Run, payload: dict, context: int | None = None) -> tuple[SourcePool, str]:
    """The run's training pool at `context` (default the run's): its fold pool, or its bird's training individuals."""
    context = run.context if context is None else context
    if ARMS[run.arm]["pool"] == "fold":
        if run.species:
            raise SystemExit(f"FAIL: {run.arm} is a pooled arm and cannot be attached to {run.species}")
        weights = fold_probabilities(run.fold, payload)
        assert_fold_sources(run.fold, weights)
        species = {
            stem: horizon_windows(stem, split_ids(stem, "train", payload), context)
            for stem in sorted(weights)
        }
        return SourcePool(species, weights), f"fold {run.fold}"
    if run.species not in BIRDS or FOLD_OF[run.species] != run.fold:
        raise SystemExit(f"FAIL: {run.species!r} is not held out in fold {run.fold}")
    windows = horizon_windows(run.species, split_ids(run.species, "train", payload), context)
    return SourcePool({run.species: windows}, {run.species: 1.0}), f"specialist:{run.species}"


def clip_shares(source_pool: SourcePool, seed: int = 0) -> dict:
    """
    Share of core and observed extended tokens whose d_east or d_north reached the clip, on up to CLIP_SAMPLE windows
    per species (seeded), with the largest per species.
    """
    rng = np.random.default_rng(seed)
    hits = {"core": [0, 0], "extended": [0, 0]}
    per_species = {}
    for name in source_pool.names:
        windows = source_pool.species[name]
        ids = np.sort(rng.choice(windows.n_windows, min(CLIP_SAMPLE, windows.n_windows), replace=False))
        features, token_mask, *_ = windows.gather(ids)
        clipped = (np.abs(features[..., _D_COLUMNS]) >= CLIP).any(axis=-1)
        core = np.zeros_like(token_mask)
        core[:, features.shape[1] - (CORE - 1) :] = True
        shares = {}
        for part, mask in (("core", core), ("extended", token_mask & ~core)):
            hits[part][0] += int(clipped[mask].sum())
            hits[part][1] += int(mask.sum())
            shares[part] = float(clipped[mask].mean()) if mask.any() else None
        per_species[name] = shares
    summary = {"sampled_per_species": CLIP_SAMPLE, "clip": CLIP}
    for part, (clipped_tokens, total_tokens) in hits.items():
        summary[part] = clipped_tokens / total_tokens if total_tokens else None
        species_shares = {n: s[part] for n, s in per_species.items() if s[part] is not None}
        summary[f"max_{part}"] = max(species_shares.items(), key=lambda kv: kv[1]) if species_shares else None
    return summary


def build_config(run: Run, grid_index: int, steps: int, pool_names, payload: dict) -> TrainConfig:
    """The bird study's recipe for the arm, with the horizon heads and, for C > 12, the padded history."""
    base = BT.build_config(run, grid_index, steps, pool_names, payload)
    padded = run.context > CORE
    return replace(
        base,
        horizons=HORIZONS,
        history=run.context if padded else 0,
        observed_flag=padded and base.trunk == "lstm",
    )


def prepared_pool(source_pool: SourcePool, k: tuple, context: int, device):
    if context == CORE:
        return PreparedPool.from_source_pool(source_pool, device=device, horizon_k=k)
    return StreamingPool.from_source_pool(source_pool, horizon_k=k, device=device)


def bytes_on_device(source_pool: SourcePool, context: int) -> int:
    """What PreparedPool.bytes_on_device would report (float32 features), or 0 for the streaming pool."""
    if context != CORE:
        return 0
    return sum(source_pool.species[n].n_windows for n in source_pool.names) * (CORE - 1) * len(FEATURES) * 4


def queue_guard(run: Run) -> None:
    """Neural runs need a GPU; XGBoost is refused on a GPU queue, where it would hold a GPU it never uses."""
    on_gpu = torch.cuda.is_available() or "gpu" in os.environ.get("PBS_QUEUE", "")
    if not run.xgb and not torch.cuda.is_available():
        raise SystemExit(
            f"FAIL: {run.arm} trains on a GPU; submit it with bird_1h_train.pbs ({queue_of(PLAN.index(run))})"
        )
    if run.xgb and on_gpu:
        raise SystemExit(
            f"FAIL: {run.arm} is a CPU run; submit it with bird_1h_train_cpu.pbs (normal), not a GPU queue"
        )


def freeze_guard() -> None:
    if not (REPO / "preregistration" / FREEZE).exists():
        raise SystemExit(
            f"FAIL: preregistration/{FREEZE} does not exist; the freeze is committed before any run"
        )


# Predictions and errors, shared with the sealed evaluator.
def predictions_m(model, windows: HorizonWindows, k: tuple, device, *, observed: bool, chunk: int = 8_192):
    """(n, H, 2) predicted east / north metres for every window: model output x k_h x the window scale."""
    model.eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, windows.n_windows, chunk):
            ids = np.arange(start, min(start + chunk, windows.n_windows))
            features, token_mask, _, _, scale = windows.gather(ids)
            prediction = model(
                torch.as_tensor(features, dtype=torch.float32, device=device),
                observed=torch.as_tensor(token_mask, device=device) if observed else None,
            )
            predictions.append(
                prediction.double().cpu().numpy() * np.asarray(k)[None, :, None] * scale[:, None, None]
            )
    return np.concatenate(predictions) if predictions else np.empty((0, len(HORIZONS), 2))


def horizon_frames(
    windows: HorizonWindows, pred_m: np.ndarray | None
) -> dict[int, tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Per horizon, (model frame, persistence frame) over the windows with a target there, in window order: individual_id,
    timestamp (the target fix's, UTC), target and predicted lon / lat, and error_m (great-circle metres to the target).
    Persistence predicts the anchor; `pred_m` None gives persistence for both. The sealed evaluator scores through here.
    Predictions invert at the predicted latitude (enu_to_lonlat_exact), the exact inverse of the target encoding
    (11-10, option (a)).
    """
    frames = {}
    anchor_lon, anchor_lat = windows.lon[windows.anchors], windows.lat[windows.anchors]
    for head, horizon in enumerate(windows.horizons):
        rows = np.flatnonzero(windows.targets[:, head] >= 0)
        target = windows.targets[rows, head]
        shared = {
            "individual_id": windows.individuals[windows.individual_index[rows]],
            "timestamp": pd.to_datetime(windows.epoch_s[target], unit="s", utc=True),
            "target_lon": windows.lon[target],
            "target_lat": windows.lat[target],
        }
        persistence_error = haversine_distance(
            anchor_lon[rows], anchor_lat[rows], windows.lon[target], windows.lat[target]
        )
        persistence = pd.DataFrame(
            shared | {"pred_lon": anchor_lon[rows], "pred_lat": anchor_lat[rows], "error_m": persistence_error}
        )
        if pred_m is None:
            model = persistence.copy()
        else:
            lon, lat = enu_to_lonlat_exact(
                pred_m[rows, head, 0], pred_m[rows, head, 1], anchor_lon[rows], anchor_lat[rows]
            )
            model_error = haversine_distance(lon, lat, windows.lon[target], windows.lat[target])
            model = pd.DataFrame(shared | {"pred_lon": lon, "pred_lat": lat, "error_m": model_error})
        frames[horizon] = (model, persistence)
    return frames


# LSTM-S checkpoint selection.
def selection_score(bird: str, frames: dict, qualifying: tuple[int, ...]) -> dict:
    """Mean over the qualifying horizons of model / P validation travel p90 (individuals at the floor)."""
    per_horizon = {}
    for horizon in qualifying:
        model_frame, p_frame = frames[horizon]
        model_summary = S2.summarise(model_frame, p_frame, BOUNDARY_M[(bird, horizon)], floor=FLOOR)
        persistence_summary = S2.summarise(p_frame, p_frame, BOUNDARY_M[(bird, horizon)], floor=FLOOR)
        per_horizon[str(horizon)] = {
            "model_p90_m": model_summary["travel"]["p90_m"],
            "p_p90_m": persistence_summary["travel"]["p90_m"],
            "n_individuals": model_summary["n_individuals"],
            "n_travel_windows": model_summary["travel"]["n_windows"],
        }
    return {
        "ratio": float(np.mean([v["model_p90_m"] / v["p_p90_m"] for v in per_horizon.values()])),
        "horizons": per_horizon,
    }


def qualifying_horizons(bird: str, frames: dict) -> tuple[int, ...]:
    """Horizons with at least one validation individual at the floor and a travel window among them."""
    qualifying = []
    for horizon, (_, p_frame) in frames.items():
        summary = S2.summarise(p_frame, p_frame, BOUNDARY_M[(bird, horizon)], floor=FLOOR)
        if summary["n_individuals"] >= 1 and summary["travel"]["n_windows"] > 0:
            qualifying.append(horizon)
    return tuple(qualifying)


def horizon_monitor(
    run: Run, payload: dict, device, grid_index: int, steps: int, k: tuple
) -> tuple[dict, dict]:
    """
    T._make_monitor's pattern for horizons: validation every SELECTION_EVERY updates, history persisted for a requeued
    job, the best checkpoint kept. The final checkpoint is reported when no horizon qualifies (recorded).
    """
    selection: dict = {"rule": None, "history": [], "best": None}
    if not run.uses_validation_selection:
        selection["rule"] = "final_checkpoint"
        return {}, selection
    validation = horizon_windows(run.species, split_ids(run.species, "val", payload), CORE)
    qualifying = qualifying_horizons(run.species, horizon_frames(validation, None))
    print(
        f"validation       : {validation.n_windows:,} windows, {len(validation.individuals)} individuals ({run.species}); "
        f"horizons at the {FLOOR}-window floor with travel: {list(qualifying) or 'none'}; selection only",
        flush=True,
    )
    if not qualifying:
        selection["rule"] = "final_checkpoint_no_validation_horizon"
        return {}, selection

    identity = {
        "run_id": run.run_id,
        "grid_index": grid_index,
        "steps": steps,
        "every": SELECTION_EVERY,
        "floor": FLOOR,
        "horizons": list(qualifying),
        "horizon_k": list(k),
    }
    history_path = CHECKPOINTS / f"{run.run_id}_selection.json"
    if history_path.exists():
        stored = json.loads(history_path.read_text())
        if stored.get("identity") == identity and stored["history"]:
            selection["history"] = stored["history"]
            selection["best"] = min(selection["history"], key=lambda row: row["ratio"])
            print(
                f"resumed          : {len(selection['history'])} validation evaluations, best so far step "
                f"{selection['best']['step']:,}",
                flush=True,
            )
        elif stored.get("identity") != identity:
            print(f"discarding {history_path.name}: it belongs to a different configuration", flush=True)
    selection.update(rule="argmin_mean_validation_travel_p90_ratio", every=SELECTION_EVERY)
    best_path = CHECKPOINTS / f"{run.run_id}_best.pt"

    def monitor(step: int, model) -> None:
        entry = selection_score(
            run.species,
            horizon_frames(validation, predictions_m(model, validation, k, device, observed=False)),
            qualifying,
        )
        entry["step"] = int(step)
        selection["history"].append(entry)
        if selection["best"] is None or entry["ratio"] < selection["best"]["ratio"]:
            selection["best"] = entry
            best_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {"step": int(step), "model": {n: t.detach().clone() for n, t in model.state_dict().items()}},
                best_path,
            )
        history_path.parent.mkdir(parents=True, exist_ok=True)
        history_path.write_text(
            json.dumps({"identity": identity, "history": selection["history"]}, indent=2) + "\n"
        )
        print(
            f"  step {step:>7,}  val travel p90 / P {entry['ratio']:.4f}"
            + ("  <- best" if selection["best"] is entry else ""),
            flush=True,
        )

    return {"monitor": monitor, "monitor_every": SELECTION_EVERY}, selection


def resolve_reported(run: Run, result, selection: dict):
    if selection["rule"] in ("final_checkpoint", "final_checkpoint_no_validation_horizon"):
        return result.model, {"rule": selection["rule"], "step": result.config.steps}
    best = selection.get("best")
    if best is None:
        raise SystemExit(
            "FAIL: LSTM-S completed without a validation evaluation; check steps >= SELECTION_EVERY"
        )
    state = torch.load(CHECKPOINTS / f"{run.run_id}_best.pt", map_location="cpu", weights_only=False)
    result.model.load_state_dict(state["model"])
    print(f"\nselected         : step {best['step']:,} of {result.config.steps:,}, ratio {best['ratio']:.4f}")
    return result.model, {
        "rule": selection["rule"],
        "every": SELECTION_EVERY,
        "floor": FLOOR,
        "selected_step": best["step"],
        "selected_ratio": best["ratio"],
        "history": selection["history"],
    }


# Calibration helpers.
class TimedPool:
    """A pool whose sample_horizon is timed between device synchronisations; everything else is the pool's."""

    def __init__(self, pool, device):
        self._pool, self._device, self.sample_seconds = pool, device, []

    def __getattr__(self, name):
        return getattr(self._pool, name)

    def _sync(self):
        if self._device.type == "cuda":
            torch.cuda.synchronize(self._device)

    def sample_horizon(self, generator, batch_size):
        self._sync()
        started = time.perf_counter()
        out = self._pool.sample_horizon(generator, batch_size)
        self._sync()
        self.sample_seconds.append(time.perf_counter() - started)
        return out


class GpuPoller:
    """nvidia-smi utilisation and memory every `every` seconds, in a background thread."""

    def __init__(self, every: float = 2.0):
        self.samples, self._stop, self._every = [], threading.Event(), every
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            try:
                line = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=utilization.gpu,memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=10,
                ).stdout.splitlines()[0]
                self.samples.append([float(v) for v in line.split(",")])
            except (OSError, IndexError, ValueError, subprocess.SubprocessError):
                pass
            self._stop.wait(self._every)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=15)

    def summary(self) -> dict:
        if not self.samples:
            return {"samples": 0}
        utilisation, memory = np.array(self.samples).T
        return {
            "samples": len(utilisation),
            "utilisation_mean_pct": float(utilisation.mean()),
            "utilisation_median_pct": float(np.median(utilisation)),
            "memory_used_max_mib": float(memory.max()),
        }


def probe_losses(model, probe, observed: bool) -> dict:
    """Per-head mean and median radial pseudo-Huber on a fixed set of training windows."""
    features, token_mask, targets, target_mask, _ = probe
    model.eval()
    with torch.no_grad():
        prediction = model(features, observed=token_mask if observed else None)
        _, mean = horizon_huber(prediction, targets, target_mask)
        per_window = radial_pseudo_huber(prediction - targets, reduction="none")
    model.train()
    return {
        str(h): {"mean": float(mean[j]), "median": float(per_window[target_mask[:, j], j].median())}
        for j, h in enumerate(HORIZONS)
    }


# Runners.
def run_torch(
    run: Run, index: int, steps: int, grid_index: int, source: str, payload: dict, calibrate: bool = False
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    started = time.time()
    source_pool, node = training_source(run, payload)
    check_window_counts(source_pool)
    k = horizon_k(source_pool)
    shares = clip_shares(source_pool)
    pool = prepared_pool(source_pool, k, run.context, device)
    built = time.time() - started
    print(
        f"pool             : {pool.n_windows:,} windows over {len(pool.names)} species [{node}], C = {run.context}, "
        f"{type(pool).__name__}, {pool.bytes_on_device() / 1e9:.2f} GB on device, built in {built:.0f}s",
        flush=True,
    )
    print(f"k_h              : {', '.join(f'{h} h {v:.6g}' for h, v in zip(HORIZONS, k))}", flush=True)
    config = build_config(run, grid_index, CALIBRATION_STEPS if calibrate else steps, pool.names, payload)
    padded = run.context > CORE

    if calibrate:
        return calibrate_torch(run, config, pool, device, k, shares, built, padded)

    monitor, selection = horizon_monitor(run, payload, device, grid_index, config.steps, k)
    result = train(
        config,
        pool,
        device=device,
        checkpoint_path=CHECKPOINTS / f"{run.run_id}_g{grid_index:02d}_{config.steps}.pt",
        progress=lambda step, loss: print(f"  step {step:>7,}  loss {loss:.5f}", flush=True),
        **monitor,
    )
    print(
        f"applied updates  : {result.applied_updates:,} of {config.steps:,}; "
        f"withheld {result.skipped_steps:,} ({result.withheld_fraction:.2%})",
        flush=True,
    )
    model, selection_payload = resolve_reported(run, result, selection)
    reported = RESULTS / f"{run.run_id}_reported.pt"
    torch.save(
        {
            "run_id": run.run_id,
            "config": asdict(config),
            "grid_index": grid_index,
            "pool_species": list(pool.names),
            "horizon_k": list(k),
            "context": run.context,
            "selection": {key: v for key, v in selection_payload.items() if key != "history"},
            "model": model.state_dict(),
        },
        reported,
    )
    record = base_record(run, index, grid_index, source, node, source_pool.names, payload, k, shares) | {
        "config": asdict(config),
        "n_parameters": model.n_parameters(),
        "pool_windows": pool.n_windows,
        "pool_kind": type(pool).__name__,
        "final_loss": result.final_loss,
        "loss_history": result.loss_history,
        "head_loss_history": result.head_loss_history,
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


def calibrate_torch(run, config, pool, device, k, shares, built, padded) -> None:
    """1,000 updates, no checkpoint, no record, no validation or test; throughput, GPU use and per-head losses."""
    probe = pool.sample_horizon(torch.Generator(device=device).manual_seed(PROBE_SEED), LOSS_PROBE)
    torch.manual_seed(config.seed)
    start_losses = probe_losses(config.build(input_dim=pool.n_features).to(device), probe, padded)
    timed, ticks = TimedPool(pool, device), []
    every = 50
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with GpuPoller() as poller:
        began = time.time()
        result = train(
            config,
            timed,
            device=device,
            checkpoint_path=None,
            checkpoint_every=0,
            log_every=every,
            progress=lambda step, loss: ticks.append((step, time.perf_counter())),
        )
        seconds = time.time() - began
    end_losses = probe_losses(result.model, probe, padded)
    step_times = np.diff([t for _, t in ticks]) / every
    sample_times = np.array(timed.sample_seconds)
    per_step = float(np.median(step_times)) if len(step_times) else seconds / config.steps
    payload = {
        "run_id": run.run_id,
        "context": run.context,
        "pool_kind": type(pool).__name__,
        "pool_windows": pool.n_windows,
        "bytes_on_device": pool.bytes_on_device(),
        "build_seconds": built,
        "updates": result.applied_updates,
        "train_seconds": seconds,
        "seconds_per_update_median": per_step,
        "encode_seconds_per_batch_median": float(np.median(sample_times)),
        "gpu_seconds_per_update_median": per_step - float(np.median(sample_times)),
        "encode_share_of_update": float(np.median(sample_times)) / per_step,
        "extrapolated_train_hours_20000": per_step * REGISTERED_STEPS / 3600,
        "extrapolated_total_hours_20000": (built + per_step * REGISTERED_STEPS) / 3600,
        "gpu": poller.summary(),
        "peak_memory_allocated_gb": (
            torch.cuda.max_memory_allocated(device) / 1e9 if device.type == "cuda" else None
        ),
        "horizon_k": list(k),
        "clip_shares": shares,
        "probe": {"windows": LOSS_PROBE, "seed": PROBE_SEED, "start": start_losses, "end": end_losses},
        "head_loss_history": result.head_loss_history,
        "skipped_steps": result.skipped_steps,
        "note": "timing synchronises the device around each draw, so encode and GPU time do not overlap here",
    }
    write_calibration(run, payload)


def xgb_horizon_set(source_pool: SourcePool, run: Run, k: tuple) -> tuple[np.ndarray, np.ndarray, str]:
    """
    The run's horizon only, in the k_h target space: every window with a target at h for XGB-S; for XGB-Bb,
    XGB_POOLED_DRAWS A8 draws from the windows with a target at h, drawn as SourcePool.sample draws them. Features are
    float32; unobserved history bins are NaN.
    """
    head = HORIZONS.index(run.horizon)
    views = {name: source_pool.species[name].for_horizon(run.horizon) for name in source_pool.names}
    if ARMS[run.arm]["pool"] == "held_out_train":
        draws = {run.species: np.arange(views[run.species].n_windows)}
        sampled = f"all {views[run.species].n_windows:,} training windows with a target at {run.horizon} h"
    else:
        rng = np.random.default_rng(run.seed)  # SourcePool.sample's calls, in its order
        picks = rng.choice(len(source_pool.names), size=XGB_POOLED_DRAWS, p=source_pool.probabilities)
        draws = {}
        for i, name in enumerate(source_pool.names):
            count = int((picks == i).sum())
            if count:
                draws[name] = rng.integers(0, views[name].n_windows, size=count)
        sampled = (
            f"{XGB_POOLED_DRAWS:,} A8 draws from windows with a target at {run.horizon} h, seed {run.seed}"
        )
    features, targets = [], []
    for name, ids in draws.items():
        for start in range(0, len(ids), GATHER_CHUNK):
            batch_features, mask, batch_targets, _, _ = views[name].gather(ids[start : start + GATHER_CHUNK])
            features.append(np.where(mask[..., None], batch_features, np.nan).astype(np.float32))
            targets.append((batch_targets[:, head] / k[head]).astype(np.float32))
    return np.concatenate(features), np.concatenate(targets), sampled


def run_xgb(
    run: Run, index: int, grid_index: int, source: str, payload: dict, calibrate: bool = False
) -> None:
    from animal_movement_fm.models.xgboost import WindowXGBoost

    started = time.time()
    source_pool, node = training_source(run, payload)
    check_window_counts(source_pool)
    k = horizon_k(source_pool)  # the pool's k_h, as its neural arms use (10-10)
    shares = clip_shares(source_pool)
    features, targets, sampled = xgb_horizon_set(source_pool, run, k)
    built = time.time() - started
    print(
        f"xgboost data     : {len(features):,} windows ({sampled}) [{node}], C = {run.context}, "
        f"{features.nbytes / 1e9:.2f} GB, built in {built:.0f}s",
        flush=True,
    )
    model = WindowXGBoost(
        random_state=run.seed,
        n_jobs=int(os.environ.get("PBS_NCPUS", "0")) or None,
        allow_missing=run.context > CORE,
    )
    began = time.time()
    model.fit(features, targets)
    fit_seconds = time.time() - began
    if calibrate:
        return write_calibration(
            run,
            {
                "run_id": run.run_id,
                "context": run.context,
                "horizon": run.horizon,
                "training_windows": len(features),
                "feature_gb": features.nbytes / 1e9,
                "build_seconds": built,
                "fit_seconds": fit_seconds,
                "peak_rss_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                / (1e9 if sys.platform == "darwin" else 1e6),
                "horizon_k": list(k),
                "clip_shares": shares,
                "sampling": sampled,
            },
        )
    stem_path = RESULTS / f"{run.run_id}_xgb"
    model.save(stem_path)
    record = base_record(run, index, grid_index, source, node, source_pool.names, payload, k, shares) | {
        "model": "WindowXGBoost",
        "params": model.params,
        "allow_missing": model.allow_missing,
        "training_windows": int(len(features)),
        "sampling": sampled,
        "target_space": f"scaled displacement / k_h at {run.horizon} h",
        "fit_seconds": fit_seconds,
        "reported_checkpoint": f"{stem_path.name}_*.ubj",
    }
    write_record(run, record)


def base_record(
    run: Run,
    index: int,
    grid_index: int,
    source: str,
    node: str,
    pool_names,
    payload: dict,
    k: tuple,
    shares: dict,
) -> dict:
    return {
        "run_id": run.run_id,
        "plan_index": index,
        "arm": run.arm,
        "recipe": BT.counterpart(run).arm,
        "fold": run.fold,
        "species": run.species or None,
        "seed": run.seed,
        "horizon": run.horizon or None,
        "horizons": list(HORIZONS),
        "context": run.context,
        "clip": CLIP,
        "clip_shares": shares,
        "horizon_k": list(k),
        "grid_index": grid_index,
        "configuration_source": source,
        "training_pool": node,
        "pool_species": list(pool_names),
        "excluded_individuals": RT.excluded_in(pool_names, payload),
        "queue": queue_of(index),
        "tier": "birds_1h",
        "registered_by": FREEZE,
        "exploratory": True,
        "opened_test_partition": False,
    }


def write_record(run: Run, record: dict) -> None:
    path = RESULTS / f"{run.run_id}.json"
    path.write_text(json.dumps(record, indent=2, default=str) + "\n")
    print(f"wrote {path}", flush=True)


def write_calibration(run: Run, payload: dict) -> None:
    CALIBRATION.mkdir(parents=True, exist_ok=True)
    path = CALIBRATION / f"{run.run_id}_calibration.json"
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    print(
        json.dumps(
            {k: v for k, v in payload.items() if k not in ("head_loss_history",)}, indent=2, default=str
        )
    )
    print(f"wrote {path}; calibration only, no run record", flush=True)


# Dry run.
def check_all_birds(payload: dict) -> None:
    """Check every bird's inputs and pinned counts, then print the fold summary."""
    missing = [
        str(p)
        for b in BIRDS
        for p in (
            SEGMENTS / f"{b}_segments.parquet",
            T.SPLITS / f"{SPW.SPLIT_STEM.get(b, b)}_split_seed42.json",
        )
        if not p.exists()
    ]
    if missing:
        raise SystemExit("FAIL: missing input files:\n  " + "\n  ".join(missing))
    built = {b: horizon_windows(b, split_ids(b, "train", payload), CORE) for b in BIRDS}
    check_window_counts(SourcePool(built, {b: 1.0 for b in built}))
    for fold in sorted(FOLDS):
        weights = fold_probabilities(fold, payload)
        assert_fold_sources(fold, weights)
        print(
            f"fold {fold}: pool {len(weights)} birds, {sum(built[n].n_windows for n in weights):,} windows; "
            f"held out {', '.join(FOLDS[fold])}"
        )
    print(
        f"\n{len(built)} birds built; every count matches the pins; 90 boundaries and fold 0's probabilities "
        "match their files."
    )
    print("dry run only; nothing was trained and no test partition was opened.")


def dry_run(payload: dict, index: int | None) -> None:
    """Every bird's training windows against the pins, or one index's pool, k_h, clip shares and core check."""
    # load_registered raised on any failed pin before this point
    print(
        "pins: PASS (registered folds; 90 boundaries vs boundaries.csv; fold 0 probabilities vs "
        "fold0_probabilities.json and the rotation rule; 1 h training counts vs horizon_counts.csv)"
    )
    if index is None:
        return check_all_birds(payload)

    run = PLAN[index]
    started = time.time()
    print(f"index {index:>3}  {run.run_id}  (fold {run.fold}, C = {run.context}, queue {queue_of(index)})")
    source_pool, node = training_source(run, payload)
    check_window_counts(source_pool)
    windows = sum(source_pool.species[training_windows].n_windows for training_windows in source_pool.names)
    print(
        f"  pool           : {node}, {len(source_pool.names)} species, {windows:,} windows; counts match the pins "
        f"at any / {' / '.join(f'{h} h' for h in HORIZONS)}"
    )
    for name, ids in RT.excluded_in(source_pool.names, payload).items():
        print(f"  excluded       : {len(ids)} {name} training individuals (registered)")
    k = horizon_k(source_pool)
    print(f"  k_h            : {', '.join(f'{h} h {v:.6g}' for h, v in zip(HORIZONS, k))}")
    shares = clip_shares(source_pool)
    print(
        f"  clip shares    : core {shares['core']:.2e} (max {shares['max_core'][0]} {shares['max_core'][1]:.2e})"
        + (
            f", extended {shares['extended']:.2e} (max {shares['max_extended'][0]} {shares['max_extended'][1]:.2e})"
            if shares["extended"] is not None
            else ", no extended tokens"
        )
        + f"; up to {CLIP_SAMPLE:,} windows per species"
    )
    if run.context > CORE:
        main_pool, _ = training_source(run, payload, CORE)
        rng = np.random.default_rng(0)
        ids = {
            training_windows: np.sort(
                rng.choice(w.n_windows, min(CORE_CHECK_SAMPLE, w.n_windows), replace=False)
            )
            for training_windows, w in main_pool.species.items()
        }
        failures = core_matches(main_pool, source_pool, ids)
        print(
            f"  core_matches   : "
            + ("PASS" if not failures else "FAIL " + "; ".join(failures[:5]))
            + f" (C = {run.context} against C = {CORE}, up to {CORE_CHECK_SAMPLE:,} windows per species, then k_h)"
        )
        if failures:
            raise SystemExit("FAIL: the ablation's core differs from the main design's")
    else:
        print(f"  core_matches   : not applicable at C = {CORE}")
    if run.xgb:
        training_windows = (
            XGB_POOLED_DRAWS
            if ARMS[run.arm]["pool"] == "fold"
            else TRAIN_WINDOWS[run.species][1 + HORIZONS.index(run.horizon)]
        )
        print(
            f"  model          : WindowXGBoost at {run.horizon} h, {training_windows:,} windows, allow_missing {run.context > CORE}, "
            f"features {training_windows * (run.context - 1) * len(FEATURES) * 4 / 1e9:.2f} GB float32"
        )
    else:
        grid_index, _ = T.selected_grid_index(None)
        config = build_config(run, grid_index, REGISTERED_STEPS, source_pool.names, payload)
        print(
            f"  bytes_on_device: {bytes_on_device(source_pool, run.context) / 1e9:.2f} GB "
            f"({'PreparedPool' if run.context == CORE else 'StreamingPool'})"
        )
        print(
            f"  config         : {BT.counterpart(run).arm}'s recipe, trunk {config.trunk}, d_model {config.d_model}, "
            f"layers {config.num_layers}, lr {config.lr:g}, horizons {config.horizons}, history {config.history}, "
            f"observed_flag {config.observed_flag}, selection {run.uses_validation_selection}"
        )
    print(f"  dry run only; nothing trained, no test partition opened ({time.time() - started:.0f}s).")


def print_plan(plan: list[Run], *, runs: bool = False) -> None:
    print(f"{'indices':<10} {'n':>4}  {'queue':<9} what")
    for first, last, queue, what in BLOCKS:
        print(f"{first:>3}-{last:<6} {last - first + 1:>4}  {queue:<9} {what}")
    print(f"\n{len(plan)} runs (0-{len(plan) - 1}); main study 0-269, context ablation 270-629.")
    if runs:
        print(f"\n{'idx':>4}  {'arm':<7} {'fold':>4}  {'species':<26} {'seed':>4} {'h':>3} {'C':>3}  queue")
        for i, run in enumerate(plan):
            print(
                f"{i:>4}  {run.arm:<7} {run.fold:>4}  {run.species or '-':<26} {run.seed:>4} "
                f"{run.horizon or '-':>3} {run.context:>3}  {queue_of(i)}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=int, default=None, help="plan index; defaults to $PBS_ARRAY_INDEX")
    parser.add_argument(
        "--steps", type=int, default=None, help="optimiser updates; must be the registered value"
    )
    parser.add_argument("--list", action="store_true", help="print the plan's blocks and exit")
    parser.add_argument("--list-runs", action="store_true", help="print every index and exit")
    parser.add_argument(
        "--dry-run", action="store_true", help="check the data for --index, or every bird, and exit"
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help=f"{CALIBRATION_STEPS:,} updates (XGB: one fit) to {CALIBRATION.name}/; no record",
    )
    args = parser.parse_args()

    if args.list or args.list_runs:
        return print_plan(PLAN, runs=args.list_runs)
    raw = args.index if args.index is not None else os.environ.get("PBS_ARRAY_INDEX")
    index = int(raw) if raw is not None else None
    if index is not None and not 0 <= index < len(PLAN):
        raise SystemExit(f"FAIL: index {index} is outside 0..{len(PLAN) - 1}; see --list")
    if args.dry_run:
        return dry_run(load_registered(), index)
    if index is None:
        raise SystemExit("FAIL: no --index and no $PBS_ARRAY_INDEX")

    run = PLAN[index]
    queue_guard(run)
    if not args.calibrate:
        freeze_guard()
        if not run.xgb and args.steps != REGISTERED_STEPS:
            raise SystemExit(f"FAIL: --steps must be {REGISTERED_STEPS:,} for this plan; got {args.steps}")
    payload = load_registered()
    grid_index, source = T.selected_grid_index(None)
    for directory in ((CALIBRATION,) if args.calibrate else (RESULTS, CHECKPOINTS)):
        directory.mkdir(parents=True, exist_ok=True)
    print(
        f"run              : {run.run_id}  (plan index {index} of {len(PLAN) - 1}, fold {run.fold}, "
        f"C = {run.context}{', CALIBRATION' if args.calibrate else ''})",
        flush=True,
    )
    print(f"configuration    : grid {grid_index} from {source}", flush=True)
    if run.xgb:
        run_xgb(run, index, grid_index, source, payload, calibrate=args.calibrate)
    else:
        run_torch(run, index, REGISTERED_STEPS, grid_index, source, payload, calibrate=args.calibrate)


if __name__ == "__main__":
    main()
