# GenAI reference: after a meeting with John exposed a gap in my original zero-shot reasoning, I asked GenAI to implement 
# this file since it wasn't a major project deliverable and I wanted quick experiment results; it did the added the source
# validation control, p1-p99 box, Mahalanobis depth, FiLM output space check, and handling for the degenerate `dt_bins` 
# axes. I reviewed the full implementation and it was correct and as I intended. Comments and doctrings also GenAi this time
#, except the file docstring which I added 
"""
Tier E / E0: check whether the FiLM conditioning path extrapolates on held-out species.

This is exploratory and governed by `tierE_exploratory_ablations.md`, not the Tier 2 preregistration. Source species read 
from training data and held-out species from validation only; so not test partition opened.

The conditioning path uses three per-window summaries from `context_conditioning` to produce FiLM `gamma` and `beta`. Here
we checks whether held-out species fall outside the region seen during source training, first in conditioning-input space 
and then, when trained Z checkpoints are available, in the realised FiLM outputs.
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments" / "training"))

import tier2_final_train as T  # noqa: E402
from tier2_inner_loop import SOURCES  # noqa: E402

from animal_movement_fm.data.enu import SCALE_FLOOR_M  # noqa: E402
from animal_movement_fm.models.fm import CONTEXT_COND_DIM, context_conditioning  # noqa: E402
from animal_movement_fm.train.loop import TrainConfig  # noqa: E402

OUT = REPO / "results" / "tier_e"

# Names in the order `context_conditioning` stacks them.
COND_NAMES = ("mean_log_scale", "mean_dt_bins", "max_dt_bins")

# Gather in blocks so large species never materialise all windows at once; only the (N, 3)
# conditioning summary is kept.
GATHER_CHUNK = 50_000

# The central box is the pooled source-training [p1, p99] range on each axis. Unlike the hull,
# it is less distorted by isolated extreme windows and better reflects where the MLP saw density.
CENTRAL_LOW, CENTRAL_HIGH = 1.0, 99.0


def conditioning_statistics(
    windows,
    *,
    sigma: float = SCALE_FLOOR_M,
    limit: int | None = None,
    seed: int = 42,
) -> np.ndarray:
    """
    Return the conditioning summary for one species as `(n_windows, CONTEXT_COND_DIM)`. The values 
    come directly from `models.fm.context_conditioning`, so this stays aligned with the model input. 
    `limit` applies deterministic subsampling to keep the source-region calculation tractable; 
    held-out species are normally taken in full.
    """
    total = windows.n_windows
    ids = np.arange(total, dtype=np.int64)
    if limit and total > limit:
        ids = np.sort(np.random.default_rng(seed).choice(total, size=limit, replace=False))

    blocks = []
    for start in range(0, len(ids), GATHER_CHUNK):
        features, _, _ = windows.gather(ids[start : start + GATHER_CHUNK], sigma=sigma)
        cond = context_conditioning(torch.as_tensor(features, dtype=torch.float32))
        blocks.append(cond.numpy())
    return np.concatenate(blocks) if blocks else np.empty((0, CONTEXT_COND_DIM), dtype=np.float32)


DEGENERATE_SPREAD = 1e-9


def hull_predicate(points: np.ndarray, tolerance: float = 1e-9):
    """
    Return a containment predicate and note for the region spanned by `points`.
    Constant conditioning axes are removed before building the region. With these 6-hourly windows,
    the cadence terms can be constant, so the effective space may collapse to a single
    `mean_log_scale` axis rather than a true 3D hull. The remaining region is treated as an interval
    in one dimension or a convex hull in higher dimensions, while removed axes must still match.
    """
    low, high = points.min(axis=0), points.max(axis=0)
    spread = high - low
    active = np.flatnonzero(spread > DEGENERATE_SPREAD)
    constant = np.flatnonzero(spread <= DEGENERATE_SPREAD)
    names_active = [COND_NAMES[i] for i in active]
    names_constant = [f"{COND_NAMES[i]}={low[i]:g}" for i in constant]

    def constants_ok(query: np.ndarray) -> np.ndarray:
        if constant.size == 0:
            return np.ones(len(query), dtype=bool)
        return (
            (query[:, constant] >= low[constant] - tolerance)
            & (query[:, constant] <= high[constant] + tolerance)
        ).all(axis=1)

    if active.size == 0:
        note = "degenerate: every conditioning axis is constant across the source pool"
        return (lambda query: constants_ok(query)), note

    if active.size == 1:
        axis = int(active[0])
        note = (
            f"effective dimension 1 ({names_active[0]}); the region is the interval "
            f"[{low[axis]:.4f}, {high[axis]:.4f}]"
            + (f", with {', '.join(names_constant)} constant" if names_constant else "")
        )

        def inside(query: np.ndarray) -> np.ndarray:
            within = (query[:, axis] >= low[axis] - tolerance) & (
                query[:, axis] <= high[axis] + tolerance
            )
            return within & constants_ok(query)

        return inside, note

    from scipy.spatial import ConvexHull

    hull = ConvexHull(points[:, active])
    equations = hull.equations  # (n_facets, n_active + 1)
    note = (
        f"effective dimension {active.size} ({', '.join(names_active)}); "
        f"{len(hull.vertices)} vertices, {equations.shape[0]} facets"
        + (f"; {', '.join(names_constant)} constant" if names_constant else "")
    )

    def inside(query: np.ndarray) -> np.ndarray:
        offsets = query[:, active] @ equations[:, :-1].T + equations[:, -1]
        return (offsets <= tolerance).all(axis=1) & constants_ok(query)

    return inside, note


def mahalanobis(query: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """
    Return each query point's Mahalanobis distance from the reference distribution.
    """
    centre = reference.mean(axis=0)
    covariance = np.cov(reference, rowvar=False)
    precision = np.linalg.pinv(np.atleast_2d(covariance))
    delta = query - centre
    return np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", delta, precision, delta), 0.0))


def marginal_percentiles(query: np.ndarray, reference: np.ndarray) -> list[float]:
    """
    Return the percentile of each query-axis median within the reference marginal.
    """
    return [
        float(100.0 * (reference[:, axis] < np.median(query[:, axis])).mean())
        for axis in range(query.shape[1])
    ]


def describe(values: np.ndarray) -> dict:
    """Per-axis summary of one species' conditioning statistics."""
    return {
        "n_windows": int(len(values)),
        "axes": {
            name: {
                "mean": float(values[:, axis].mean()),
                "sd": float(values[:, axis].std()),
                "min": float(values[:, axis].min()),
                "p01": float(np.percentile(values[:, axis], 1)),
                "p50": float(np.percentile(values[:, axis], 50)),
                "p99": float(np.percentile(values[:, axis], 99)),
                "max": float(values[:, axis].max()),
            }
            for axis, name in enumerate(COND_NAMES)
        },
    }


def containment(query: np.ndarray, source: np.ndarray, inside) -> dict:
    """
    Score one species against the pooled source-training conditioning region. The checks cover the 
    full per-axis range, the effective source region from `hull_predicate`, and the tighter per-axis 
    central box. Together they distinguish broad containment from regions where the source data 
    actually had much density.
    """
    low, high = source.min(axis=0), source.max(axis=0)
    core_low = np.percentile(source, CENTRAL_LOW, axis=0)
    core_high = np.percentile(source, CENTRAL_HIGH, axis=0)

    in_range = ((query >= low) & (query <= high)).all(axis=1)
    in_core = ((query >= core_low) & (query <= core_high)).all(axis=1)
    in_hull = inside(query) if inside is not None else np.zeros(len(query), dtype=bool)

    distance = mahalanobis(query, source)
    return {
        "n_windows": int(len(query)),
        "fraction_in_axis_range": float(in_range.mean()),
        "fraction_in_source_region": float(in_hull.mean()) if inside is not None else None,
        "fraction_in_central_box": float(in_core.mean()),
        "per_axis_fraction_in_range": {
            name: float(((query[:, axis] >= low[axis]) & (query[:, axis] <= high[axis])).mean())
            for axis, name in enumerate(COND_NAMES)
        },
        "median_percentile_within_source_marginal": dict(
            zip(COND_NAMES, marginal_percentiles(query, source), strict=True)
        ),
        "mahalanobis_to_source": {
            "p50": float(np.percentile(distance, 50)),
            "p90": float(np.percentile(distance, 90)),
            "max": float(distance.max()),
        },
    }


def film_response(cond: dict[str, np.ndarray], seeds, device) -> dict | None:
    """
    Compare realised FiLM parameters after passing each species through a trained `cond_proj`. This is 
    the direct output-space check: `gamma` and `beta` are what the trunk actually receives. Results are 
    calculated per Z seed and summarised by the median across seeds, using only the data already loaded 
    for the input-space analysis.
    """
    source_names = [stem for stem, *_ in SOURCES]
    per_seed: dict[str, list[dict]] = {name: [] for name in cond}

    used = []
    for seed in seeds:
        path = T.RESULTS / f"Z_s{seed}_reported.pt"
        if not path.exists():
            continue
        state = torch.load(path, map_location="cpu", weights_only=False)
        model = TrainConfig(**state["config"]).build()
        model.load_state_dict(state["model"])
        model.eval().to(device)
        if model.cond_proj is None:
            raise SystemExit(f"FAIL: {path.name} has no conditioning path; that is not Z")
        used.append(seed)

        realised = {}
        with torch.no_grad():
            for name, values in cond.items():
                out = model.cond_proj(torch.as_tensor(values, dtype=torch.float32, device=device))
                gamma, beta = out.chunk(2, dim=-1)
                realised[name] = (gamma.cpu().numpy(), beta.cpu().numpy())

        # Reference envelope: every FiLM setting produced by source-training windows.
        source_gamma = np.concatenate([realised[n][0] for n in source_names])
        source_beta = np.concatenate([realised[n][1] for n in source_names])
        gamma_low, gamma_high = source_gamma.min(axis=0), source_gamma.max(axis=0)
        beta_low, beta_high = source_beta.min(axis=0), source_beta.max(axis=0)

        for name, (gamma, beta) in realised.items():
            inside_gamma = ((gamma >= gamma_low) & (gamma <= gamma_high)).all(axis=1)
            inside_beta = ((beta >= beta_low) & (beta <= beta_high)).all(axis=1)
            per_seed[name].append(
                {
                    # Share of windows with every FiLM channel inside the source envelope.
                    "fraction_gamma_in_source_envelope": float(inside_gamma.mean()),
                    "fraction_beta_in_source_envelope": float(inside_beta.mean()),
                    # Worst-channel overshoot, scaled by the source envelope width.
                    "median_gamma_overshoot": float(
                        np.median(
                            (
                                np.maximum(
                                    np.maximum(gamma_low - gamma, gamma - gamma_high), 0.0
                                )
                                / np.maximum(gamma_high - gamma_low, 1e-12)
                            ).max(axis=1)
                        )
                    ),
                    "mean_abs_gamma": float(np.abs(gamma).mean()),
                    "mean_abs_beta": float(np.abs(beta).mean()),
                }
            )

    if not used:
        return None

    keys = (
        "fraction_gamma_in_source_envelope",
        "fraction_beta_in_source_envelope",
        "median_gamma_overshoot",
        "mean_abs_gamma",
        "mean_abs_beta",
    )
    return {
        "seeds": used,
        "species": {
            name: {key: float(np.median([row[key] for row in rows])) for key in keys}
            for name, rows in per_seed.items()
            if rows
        },
    }

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-sample",
        type=int,
        default=250_000,
        help="windows per source species used to build the hull; 0 uses every window",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=SCALE_FLOOR_M,
        help="scale floor; must match the arm being reasoned about (registered value is 30 m)",
    )
    parser.add_argument(
        "--no-film",
        action="store_true",
        help="skip stage 2 even if trained Z checkpoints are present",
    )
    args = parser.parse_args()

    device = torch.device("cpu")
    started = time.time()
    print("Tier E / E0 - conditioning-path extrapolation check")
    print("partition        : source TRAIN + held-out VALIDATION; no test partition is opened")
    print(f"scale floor      : {args.sigma:g} m")
    print(f"conditioning     : {', '.join(COND_NAMES)}\n", flush=True)

    limit = args.source_sample or None
    cond: dict[str, np.ndarray] = {}

    for stem, split_stem, _ in SOURCES:
        windows = T._load_species_windows(stem, split_stem, "train")
        cond[stem] = conditioning_statistics(windows, sigma=args.sigma, limit=limit)
        print(
            f"  source   {stem:<26} train {windows.n_windows:>9,} windows"
            f" -> {len(cond[stem]):>7,} sampled",
            flush=True,
        )

    source_pooled = np.concatenate([cond[stem] for stem, *_ in SOURCES])

    # In-distribution control: source species on validation windows the MLP did not train on.
    control: dict[str, np.ndarray] = {}
    for stem, split_stem, _ in SOURCES:
        windows = T._load_species_windows(stem, split_stem, "val")
        control[stem] = conditioning_statistics(windows, sigma=args.sigma)
        print(
            f"  control  {stem:<26} val   {windows.n_windows:>9,} windows", flush=True
        )

    held_out: dict[str, np.ndarray] = {}
    for stem, split_stem, *_ in T.HELD_OUT:
        windows = T._load_species_windows(stem, split_stem, "val")
        held_out[stem] = conditioning_statistics(windows, sigma=args.sigma)
        print(
            f"  held out {stem:<26} val   {windows.n_windows:>9,} windows", flush=True
        )

    try:
        inside, hull_note = hull_predicate(source_pooled)
    except Exception as error:  # qhull raises several distinct types on awkward input
        inside = None
        # Qhull's full error dump is huge; the first line is enough for this diagnostic.
        hull_note = f"not computed: {type(error).__name__}: {str(error).splitlines()[0]}"
        print(f"\nWARNING: source region {hull_note}", flush=True)
    print(f"source region    : {hull_note}\n", flush=True)

    payload = {
        "tier": "E",
        "experiment": "E0_conditioning_extrapolation",
        "exploratory": True,
        "opened_test_partition": False,
        "source_partition": "train",
        "held_out_partition": "validation",
        "scale_floor_m": args.sigma,
        "source_sample_per_species": limit,
        "conditioning_axes": list(COND_NAMES),
        "source_region": hull_note,
        "central_box_percentiles": [CENTRAL_LOW, CENTRAL_HIGH],
        "distributions": {
            "source_train": {name: describe(values) for name, values in cond.items()},
            "source_val_control": {name: describe(values) for name, values in control.items()},
            "held_out_val": {name: describe(values) for name, values in held_out.items()},
        },
        "containment": {
            "source_val_control": {
                name: containment(values, source_pooled, inside)
                for name, values in control.items()
            },
            "held_out_val": {
                name: containment(values, source_pooled, inside)
                for name, values in held_out.items()
            },
        },
    }

    # Check how many conditioning dimensions actually vary across the source pool.
    centred = source_pooled - source_pooled.mean(axis=0)
    singular = np.linalg.svd(centred, compute_uv=False)
    variance = singular**2
    payload["source_effective_dimensionality"] = {
        "axis_sd": dict(zip(COND_NAMES, source_pooled.std(axis=0).tolist(), strict=True)),
        "pca_explained_variance_ratio": (variance / variance.sum()).tolist(),
    }

    if not args.no_film:
        payload["film_response"] = film_response(
            {**cond, **{f"heldout::{k}": v for k, v in held_out.items()}}, T.SEEDS, device
        )
        if payload["film_response"] is None:
            print("\nno trained Z checkpoints found; stage 2 (FiLM response) skipped", flush=True)

    print("\n=== conditioning statistics, median [p1, p99] ===")
    header = f"{'species':<28} {'partition':<13}" + "".join(f"{n:>26}" for n in COND_NAMES)
    print(header)
    print("-" * len(header))
    for group, label in (
        ("source_train", "source/train"),
        ("source_val_control", "source/val"),
        ("held_out_val", "heldout/val"),
    ):
        for name, entry in payload["distributions"][group].items():
            cells = "".join(
                (
                    f"{entry['axes'][axis]['p50']:.3f} "
                    f"[{entry['axes'][axis]['p01']:.3f}, {entry['axes'][axis]['p99']:.3f}]"
                ).rjust(26)
                for axis in COND_NAMES
            )
            print(f"{name:<28} {label:<13}{cells}")

    print("\n=== containment in the pooled source TRAINING region ===")
    header = (
        f"{'species':<28} {'group':<12} {'in range':>10} {'in region':>11}"
        f" {'in p1-p99 box':>15} {'mahalanobis p50':>17}"
    )
    print(header)
    print("-" * len(header))
    for group, label in (("source_val_control", "control"), ("held_out_val", "held out")):
        for name, entry in payload["containment"][group].items():
            hull_cell = (
                "—"
                if entry["fraction_in_source_region"] is None
                else f"{entry['fraction_in_source_region']:.1%}"
            )
            print(
                f"{name:<28} {label:<12} {entry['fraction_in_axis_range']:>9.1%}"
                f" {hull_cell:>11} {entry['fraction_in_central_box']:>14.1%}"
                f" {entry['mahalanobis_to_source']['p50']:>17.2f}"
            )

    film = payload.get("film_response")
    if film:
        print(f"\n=== realised FiLM parameters, median over Z seeds {film['seeds']} ===")
        header = (
            f"{'species':<34} {'gamma in envelope':>19} {'beta in envelope':>18}"
            f" {'gamma overshoot':>17} {'mean |gamma|':>14}"
        )
        print(header)
        print("-" * len(header))
        for name, row in film["species"].items():
            print(
                f"{name:<34} {row['fraction_gamma_in_source_envelope']:>19.1%}"
                f" {row['fraction_beta_in_source_envelope']:>18.1%}"
                f" {row['median_gamma_overshoot']:>17.3f} {row['mean_abs_gamma']:>14.3f}"
            )

    payload["elapsed_s"] = time.time() - started
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "tier_e_conditioning_hull.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nwrote {path}  ({payload['elapsed_s'] / 60:.1f} min)")
    print(
        "\nTier E is exploratory. No success band is applied, no test partition was opened, and "
        "nothing here may be reported as a Tier 2 result."
    )


if __name__ == "__main__":
    main()
