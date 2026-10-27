# GenAI reference: The beauty of GenAI; I suspected the Transformer was collapsing/underperforming, so I outlined the checks I wanted 
# around loss, prediction behaviour, masking, context and validation skill, and got GenAI to write this diagnostic script. I later used 
# GenAI to help with a repair mode that scores the transformer repair probe's runs with the same samples and diagnostics (loss / zero, 
# input sensitivity, |pred| / |actual|, cos(pred, actual), validation skill, paired logged loss against V0) and GenAI generalised 
# run_fold to a list of runs and wrote the repair report. 26-10-2026: for the BbF extension's diagnostic, GenAI let
# run_fold take a caller's own entries and mode (bird_1h_bbf_extension_diagnostic.py); the default paths are unchanged.
"""
Transformer collapse diagnosis for the 1 h pooled birds (exploratory; no training; training and validation only).

Why does Bb under-predict at C = 12 and predict almost nothing at C >= 48, while LSTM-Bb does well on the same inputs?
Reads the 1 h run records and reported checkpoints of Bb and LSTM-Bb (folds 0, 3, 6; seed 0; C = 12, 24, 48, 96) and scores them on a 
fixed A8 sample of each fold's training windows and on its birds' validation windows:

  1. loss against the zero and last-step predictors, and the record's loss history
  2. prediction behaviour: |pred| / |actual|, cosines to the last step and to the target, slope and R^2 on the last step
  3. bug checks: masked tokens, the last token, eval against train mode without dropout, core only, input magnitudes
  4. validation travel p90 skill against P for the last-step predictor and the two arms
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import copy
import json
import sys
import time
import zlib
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments" / "training"))
sys.path.insert(0, str(REPO / "experiments" / "reporting"))
sys.path.insert(0, str(REPO))

import bird_1h_train as b1h  # noqa: E402
import tier3_sealed_eval as E3  # noqa: E402
import transformer_repair_train as TR  # noqa: E402

from animal_movement_fm.data.multispecies import HorizonWindows, SourcePool, history_positions  # noqa: E402
from animal_movement_fm.data.split import id_key, load_split  # noqa: E402
from animal_movement_fm.train.loss import horizon_huber  # noqa: E402

S2, T, SPW = b1h.S2, b1h.T, b1h.SPW
CORE, HORIZONS, BOUNDARY_M, FOLDS = b1h.CORE, b1h.HORIZONS, b1h.BOUNDARY_M, b1h.FOLDS

OUT = REPO / "results" / "transformer_diagnosis"
REPORT_JSON = OUT / "transformer_diagnosis.json"
REPORT_MD = OUT / "transformer_diagnosis.md"
REPAIR_OUT = TR.RESULTS / "validation"  # beside the repair probe's run records
REPAIR_JSON = REPAIR_OUT / "transformer_repair.json"
REPAIR_MD = REPAIR_OUT / "transformer_repair.md"

DIAGNOSIS_FOLDS = (0, 3, 6)
SEED = 0  # the runs' training seed
CONTEXTS = (12, 24, 48, 96)
ARMS = ("Bb", "LSTMBb")
PARTITIONS = ("train", "val")  # never test

N_TRAIN = 20_000  # A8 draws from the fold's training windows
MAX_VAL = 3_000  # validation windows per bird, uniform
N_CHECK = 2_000  # validation windows for the bug checks
SAMPLE_SEED = 2026
CHUNK = 4_096

LOG_EVERY = 500  # loop.train logs one batch's loss every 500 applied updates
SMOOTH = 5  # rolling median over logged points
CURVE_STEPS = (2_000, 5_000, 10_000, 15_000, 20_000)  # 2,000 is the end of warmup
PLATEAU = 0.05  # "within 5% of its final level"

FULL_SHARE = 0.9  # the 1 h freeze's ablation strata: full at or above, gappy below
NOISE_SD = 10.0  # masked-token check: noise written into unobserved tokens
MIN_MOVE_M = 1.0  # cosines skip vectors shorter than this
MAGNITUDE_LEVELS = (10.0, 100.0)
_D = [b1h.FEATURES.index("d_east_scaled"), b1h.FEATURES.index("d_north_scaled")]


# Guards.
def split_ids(stem: str, partition: str, payload: dict) -> list:
    """Training (after exclusions) or validation individuals; anything else is refused before a file is read."""
    if partition not in PARTITIONS:
        raise SystemExit(
            f"FAIL: the transformer diagnosis reads train and val only, never {partition!r}; "
            "test is read by the sealed evaluators alone"
        )
    ids = b1h.split_ids(stem, partition, payload)
    assert_not_test(stem, ids)
    return ids


def check_ids(stem: str, ids, test_ids) -> None:
    """Refuse if any individual about to be built is in the bird's test partition."""
    leaked = sorted({id_key(i) for i in ids} & {id_key(i) for i in test_ids})
    if leaked:
        raise SystemExit(f"FAIL: {stem}: {len(leaked)} test individuals would be read ({leaked[:3]} ...)")


def assert_not_test(stem: str, ids) -> None:
    manifest = T.SPLITS / f"{SPW.SPLIT_STEM.get(stem, stem)}_split_seed42.json"
    check_ids(stem, ids, load_split(manifest).test)


def diagnosis_run(arm: str, fold: int, context: int):
    run = b1h.Run(arm, fold, "", SEED, 0, context)
    if run not in b1h.PLAN:
        raise SystemExit(f"FAIL: {run.run_id} is not in the 1 h plan")
    return run


def load_run(run, results_dir: Path | None = None, tier: str | None = None) -> tuple[nn.Module, np.ndarray, dict]:
    """
    (model in eval mode, k_h, record), checked against the record: clean, 20,000 updates, k_h and C agree, and the
    record's tier when one is given. The 1 h study's runs by default; the repair probe's with its results and tag.
    """
    results_dir = b1h.RESULTS if results_dir is None else results_dir
    path = results_dir / f"{run.run_id}.json"
    if not path.exists():
        raise SystemExit(f"FAIL: {path} does not exist")
    record = json.loads(path.read_text())
    problems = []
    if record.get("opened_test_partition") is not False:
        problems.append("opened_test_partition is not false")
    if tier is not None and record.get("tier") != tier:
        problems.append(f"record tier {record.get('tier')!r}, not {tier!r}")
    if record.get("context") != run.context:
        problems.append(f"record context {record.get('context')}")
    if record.get("applied_updates") != b1h.REGISTERED_STEPS:
        problems.append(f"applied_updates {record.get('applied_updates')}")
    loaded = E3.load_reported(results_dir, run.run_id)
    if loaded is None:
        problems.append("no reported checkpoint")
    else:
        model, state = loaded
        if list(state["horizon_k"]) != list(record["horizon_k"]) or state["context"] != run.context:
            problems.append("checkpoint disagrees with its record on k_h or C")
    if problems:
        raise SystemExit(f"FAIL: {run.run_id}: " + "; ".join(problems))
    return model.eval(), np.asarray(record["horizon_k"], dtype=float), record


def diagnosis_entries(fold: int) -> list[dict]:
    """The diagnosis' runs: Bb and LSTM-Bb at every C. `pair` links a C > 12 run to its C = 12 run for core-only."""
    entries = []
    for arm in ARMS:
        for context in CONTEXTS:
            run = diagnosis_run(arm, fold, context)
            entries.append({"label": f"{arm}_C{context}", "run_id": run.run_id, "context": context, "pair": arm,
                            "transformer": arm == "Bb", "load": lambda run=run: load_run(run)})
    return entries


def repair_entries(fold: int) -> list[dict]:
    """The repair probe's runs on this fold, read from its own results with its tag checked."""
    runs = [run for run in TR.PLAN if run.fold == fold]
    if not runs:
        raise SystemExit(f"FAIL: the repair plan has no run on fold {fold}; it covers {TR.PROBE_FOLDS}")
    return [{"label": f"{run.base}{run.variant}_C{run.context}", "run_id": run.run_id, "context": run.context,
             "pair": f"{run.base}{run.variant}", "transformer": run.base == "Bb",
             "load": lambda run=run: load_run(run, TR.RESULTS, TR.TAG)} for run in runs]


# Windows and samples.
def build(stem: str, partition: str, payload: dict, context: int = CORE) -> HorizonWindows:
    return b1h.horizon_windows(stem, split_ids(stem, partition, payload), context)


def subset(windows: HorizonWindows, ids: np.ndarray) -> HorizonWindows:
    ids = np.asarray(ids, dtype=np.int64)
    return replace(
        windows, anchors=windows.anchors[ids], targets=windows.targets[ids], individual_index=windows.individual_index[ids]
    )


def at_context(windows: HorizonWindows, context: int) -> HorizonWindows:
    """
    The same windows gathered at `context`. HorizonWindows.build uses context only to store it (anchors and targets
    come from the core); check_context_reuse confirms it on real data every run.
    """
    return replace(windows, context=int(context))


def training_sample(windows: dict[str, HorizonWindows], probabilities: dict[str, float], n: int, seed: int):
    """Training's draw rule: species by the fold's A8 probabilities, then a window uniformly within it (with replacement)."""
    rng = np.random.default_rng(seed)
    names = sorted(probabilities)
    p = np.array([probabilities[s] for s in names], dtype=float)
    picks = rng.choice(len(names), size=n, p=p / p.sum())
    return {
        name: np.sort(rng.integers(0, windows[name].n_windows, size=int((picks == i).sum())))
        for i, name in enumerate(names)
        if (picks == i).any()
    }


def validation_sample(n_windows: int, cap: int, stem: str) -> np.ndarray:
    """Up to `cap` windows uniformly without replacement, seeded by the bird so it is the same in every fold."""
    if n_windows <= cap:
        return np.arange(n_windows)
    rng = np.random.default_rng([SAMPLE_SEED, zlib.crc32(stem.encode())])
    return np.sort(rng.choice(n_windows, cap, replace=False))


def history_share(windows: HorizonWindows) -> np.ndarray:
    """Share of extended-history bins observed (the 1 h freeze's strata); NaN at C = 12."""
    if windows.context == CORE:
        return np.full(windows.n_windows, np.nan)
    grid = history_positions(windows.keys, windows.anchors, windows.context)
    return (grid[:, : windows.context - CORE] >= 0).mean(axis=1)


def gather(windows: HorizonWindows, context: int) -> dict[str, np.ndarray]:
    w = at_context(windows, context)
    features, token_mask, scaled, target_mask, scale = w.gather(np.arange(w.n_windows))
    return {
        "features": features.astype(np.float32),
        "token_mask": token_mask,
        "scaled": scaled,
        "target_mask": target_mask,
        "scale": scale,
        "share": history_share(w),
    }


def stack(parts: list[dict]) -> dict[str, np.ndarray]:
    return {key: np.concatenate([p[key] for p in parts]) for key in parts[0]}


def check_context_reuse(stem: str, payload: dict, ids: np.ndarray, sample: HorizonWindows) -> dict:
    """The sample gathered via at_context equals the same windows built directly at the longest C."""
    context = max(CONTEXTS)
    direct = subset(build(stem, "val", payload, context), ids)
    rows = np.arange(min(200, len(ids)))
    for a, b in zip(at_context(sample, context).gather(rows), direct.gather(rows)):
        if not np.array_equal(a, b):
            raise SystemExit(f"FAIL: {stem}: windows gathered at C = {context} differ from a direct build")
    return {"bird": stem, "context": context, "windows": int(len(rows)), "equal": True}


# Predictors.
def predict(model: nn.Module, features: np.ndarray, token_mask: np.ndarray, context: int, chunk: int = CHUNK):
    """
    (n, H, 2) output in the model's target units (metres / (scale x k_h)); the padding mask is passed for C > 12 only,
    as training and the sealed evaluator do. The model's mode is left as it is (see fast_path_check).
    """
    out = []
    with torch.no_grad():
        for start in range(0, len(features), chunk):
            x = torch.as_tensor(features[start : start + chunk], dtype=torch.float32)
            mask = torch.as_tensor(token_mask[start : start + chunk]) if context > CORE else None
            out.append(model(x, observed=mask).double().numpy())
    return np.concatenate(out) if out else np.empty((0, len(HORIZONS), 2))


def last_step(features: np.ndarray, k) -> np.ndarray:
    """
    The last-step predictor in model units: the last core token's scaled step x h / k_h, i.e. the last 1 h step x h in
    metres divided by (scale x k_h). It carries the token's +-1,000 clip.
    """
    step = np.asarray(features[:, -1, _D], dtype=float)
    return step[:, None, :] * np.asarray(HORIZONS, float)[None, :, None] / np.asarray(k, float)[None, :, None]


def to_metres(pred: np.ndarray, k, scale: np.ndarray) -> np.ndarray:
    return pred * np.asarray(k, float)[None, :, None] * np.asarray(scale, float)[:, None, None]


def loss_table(pred: np.ndarray, scaled: np.ndarray, target_mask: np.ndarray, k) -> dict:
    """horizon_huber against targets / k_h, the training loss; overall (mean over heads) and per head."""
    targets = scaled / np.asarray(k, float)[None, :, None]
    loss, heads = horizon_huber(torch.as_tensor(pred), torch.as_tensor(targets), torch.as_tensor(target_mask))
    return {"mean": float(loss)} | {f"h{h}": float(v) for h, v in zip(HORIZONS, heads)}


def reference_losses(pred: np.ndarray, features: np.ndarray, scaled, target_mask, k) -> dict:
    return {
        "model": loss_table(pred, scaled, target_mask, k),
        "zero": loss_table(np.zeros_like(pred), scaled, target_mask, k),
        "last_step": loss_table(last_step(features, k), scaled, target_mask, k),
    }


# 1. Loss history.
def loss_curve(record: dict) -> dict:
    """The record's logged losses (one batch every 500 updates), median-smoothed, at CURVE_STEPS, and the plateau step."""
    history = np.asarray(record.get("loss_history") or [], dtype=float)
    expected = b1h.REGISTERED_STEPS // LOG_EVERY
    if len(history) != expected:
        return {"points": int(len(history)), "note": f"expected {expected} logged points; not summarised"}
    steps = LOG_EVERY * np.arange(1, expected + 1)
    smooth = pd.Series(history).rolling(SMOOTH, center=True, min_periods=1).median().to_numpy()
    final = smooth[-1]
    outside = np.flatnonzero(np.abs(smooth - final) > PLATEAU * abs(final))
    within_from = int(steps[outside[-1] + 1]) if len(outside) and outside[-1] + 1 < len(steps) else int(steps[0])
    out = {"points": int(len(history)), "smoothed": {str(s): float(smooth[steps == s][0]) for s in CURVE_STEPS}}
    out["within_5pct_of_final_from"] = within_from
    heads = record.get("head_loss_history") or []
    if len(heads) == expected:
        frame = pd.DataFrame(np.asarray(heads, dtype=float), columns=[f"h{h}" for h in HORIZONS])
        smooth_heads = frame.rolling(SMOOTH, center=True, min_periods=1).median()
        out["smoothed_heads"] = {
            str(s): {c: _f(smooth_heads[c].to_numpy()[steps == s][0]) for c in frame.columns} for s in CURVE_STEPS
        }
    return out


# 2. Prediction behaviour.
def _cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine per row; NaN on a zero vector (callers keep only vectors of at least MIN_MOVE_M)."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return (a * b).sum(axis=-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))


def slope_r2(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """OLS slope (with intercept) and R^2 of y on x."""
    x, y = np.asarray(x, float).ravel(), np.asarray(y, float).ravel()
    if len(x) < 3 or np.var(x) == 0:
        return float("nan"), float("nan")
    slope = np.cov(x, y, bias=True)[0, 1] / np.var(x)
    r = np.corrcoef(x, y)[0, 1] if np.var(y) > 0 else 0.0
    return float(slope), float(r * r)


def behaviour(pred_m, actual_m, last_m, scale, target_mask, boundaries: dict[int, float]) -> dict:
    """
    Per horizon, on travel windows (|actual| >= the bird's boundary at h, as the scoring): median |pred| / |actual|,
    median |pred| and |actual| in metres, median cosine of pred to the last step and to the actual displacement (vectors
    under 1 m skipped), and the slope and R^2 of pred on last step x h, both divided by the window scale. Also the median
    |pred| over every window with a target.
    """
    out = {}
    for j, h in enumerate(HORIZONS):
        has = target_mask[:, j]
        p, a, l, s = pred_m[has, j], actual_m[has, j], last_m[has, j], scale[has]
        norm_p, norm_a, norm_l = (np.linalg.norm(v, axis=-1) for v in (p, a, l))
        travel = norm_a >= boundaries[h]
        entry = {"n_windows": int(has.sum()), "n_travel": int(travel.sum()),
                 "median_pred_m_all": _median(norm_p)}
        if travel.any():
            p, a, l, s = p[travel], a[travel], l[travel], s[travel]
            norm_p, norm_a, norm_l = norm_p[travel], norm_a[travel], norm_l[travel]
            moving = norm_p >= MIN_MOVE_M
            slope, r2 = slope_r2(l / s[:, None], p / s[:, None])
            entry |= {
                "ratio": _median(norm_p / norm_a),
                "median_pred_m": _median(norm_p),
                "median_actual_m": _median(norm_a),
                "cos_last": _median(_cos(p, l)[moving & (norm_l >= MIN_MOVE_M)]),
                "cos_actual": _median(_cos(p, a)[moving]),
                "share_pred_under_1m": float((~moving).mean()),
                "slope": slope,
                "r2": r2,
            }
        out[str(h)] = entry
    return out


def strata_rows(share: np.ndarray) -> dict[str, np.ndarray]:
    if np.isnan(share).all():
        return {"all": np.ones(len(share), dtype=bool)}
    return {"all": np.ones(len(share), dtype=bool), "full": share >= FULL_SHARE, "gappy": share < FULL_SHARE}


# 3. Bug checks.
def masked_noise_check(model, batch: dict, context: int, rng) -> dict:
    """Noise written into unobserved tokens; a masked trunk must not move. LSTM-Bb reads a flag, not a mask (information)."""
    features, mask = batch["features"], batch["token_mask"]
    base = predict(model, features, mask, context)
    noise = rng.normal(0.0, NOISE_SD, features.shape).astype(np.float32)
    other = predict(model, np.where(mask[..., None], features, noise), mask, context)
    diff = np.abs(other - base).max(axis=(1, 2))
    gappy = ~mask.all(axis=1)
    return {
        "n_windows": int(len(diff)),
        "n_with_unobserved": int(gappy.sum()),
        "max_abs_diff": _f(diff[gappy].max()) if gappy.any() else None,
        "median_abs_diff": _f(np.median(diff[gappy])) if gappy.any() else None,
        "median_abs_output": _f(np.median(np.abs(base))),
    }


def token_sensitivity(model, batch: dict, context: int) -> dict:
    """
    Per head, median |change| in output (model units) when the last token's d_east_scaled gets +1, when the last token is
    swapped for another window's, and (contrast) when the earliest observed token's d_east_scaled gets +1; with the
    median |output| beside them.
    """
    features, mask = batch["features"], batch["token_mask"]
    base = predict(model, features, mask, context)
    n = len(features)
    bumped = features.copy()
    bumped[:, -1, _D[0]] += 1.0
    swapped = features.copy()
    swapped[:, -1] = np.roll(features[:, -1], 1, axis=0)
    earliest = features.copy()
    earliest[np.arange(n), mask.argmax(axis=1), _D[0]] += 1.0
    out = {"median_abs_output": _per_head(np.linalg.norm(base, axis=-1))}
    for name, perturbed in (("last_plus_1", bumped), ("last_swapped", swapped), ("earliest_plus_1", earliest)):
        out[name] = _per_head(np.linalg.norm(predict(model, perturbed, mask, context) - base, axis=-1))
    return out


def fast_path_check(model, batch: dict, context: int) -> dict:
    """
    eval() (the nested-tensor fast path, where torch takes it) against train() with every dropout at 0 (no fast path,
    as in training): the outputs must agree to float precision.
    """
    twin = copy.deepcopy(model)
    for module in twin.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0.0
        if isinstance(module, nn.MultiheadAttention):
            module.dropout = 0.0
    twin.train()
    features, mask = batch["features"], batch["token_mask"]
    diff = np.abs(predict(twin, features, mask, context) - predict(model.eval(), features, mask, context)).max(axis=(1, 2))
    out = {"max_abs_diff": _f(diff.max())}
    for name, rows in strata_rows(batch["share"]).items():
        if name != "all":
            out[f"max_abs_diff_{name}"] = _f(diff[rows].max()) if rows.any() else None
    return out


def core_only(model, batch: dict, context: int) -> np.ndarray:
    """The C > 12 model with every extended token masked and zeroed, so it sees the core alone (model units)."""
    mask = batch["token_mask"].copy()
    mask[:, : context - CORE] = False
    return predict(model, np.where(mask[..., None], batch["features"], 0.0).astype(np.float32), mask, context)


def magnitudes(batch: dict, context: int, species: np.ndarray) -> dict:
    """Share of core and observed extended tokens with max(|d_east|, |d_north|) above each level, at the clip, and p99."""
    size = np.abs(batch["features"][..., _D]).max(axis=-1)
    core = np.zeros_like(batch["token_mask"])
    core[:, context - CORE :] = True
    out = {}
    for part, rows in (("core", core), ("extended", batch["token_mask"] & ~core)):
        values = size[rows]
        entry = {"tokens": int(rows.sum())}
        if len(values):
            entry |= {f"share_over_{int(v)}": float((values > v).mean()) for v in MAGNITUDE_LEVELS}
            entry |= {"share_at_clip": float((values >= b1h.CLIP).mean()), "p99": float(np.percentile(values, 99))}
            top = MAGNITUDE_LEVELS[-1]
            by_species = {s: float((size[rows & (species[:, None] == s)] > top).mean())
                          for s in np.unique(species) if (rows & (species[:, None] == s)).any()}
            worst = max(by_species.items(), key=lambda kv: kv[1])
            entry[f"top_species_over_{int(top)}"] = [worst[0], worst[1]]
        out[part] = entry
    return out


# 4. Reference skill.
def travel_p90(windows: HorizonWindows, pred_m: np.ndarray | None, bird: str) -> dict[str, dict]:
    """Per horizon, travel p90 (individual weighted, no floor: a capped validation sample) and its n's; None = P."""
    out = {}
    for h, (frame, persistence) in b1h.horizon_frames(windows, pred_m).items():
        summary = S2.summarise(frame, persistence, BOUNDARY_M[(bird, h)], floor=1)
        out[str(h)] = {"p90_m": summary["travel"].get("p90_m"), "n_individuals": summary["n_individuals"],
                       "n_travel": summary["travel"]["n_windows"]}
    return out


# Helpers.
def _f(value):
    value = float(value)
    return None if not np.isfinite(value) else value


def _median(values) -> float | None:
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    return float(np.median(values)) if len(values) else None


def _per_head(values: np.ndarray) -> dict:
    return {f"h{h}": _f(np.median(values[:, j])) for j, h in enumerate(HORIZONS)}


def _slice(batch: dict, rows) -> dict:
    return {key: value[rows] for key, value in batch.items()}


# One fold.
def run_fold(fold: int, *, n_train: int = N_TRAIN, max_val: int = MAX_VAL, n_check: int = N_CHECK,
             dry_run: bool = False, repair: bool = False, entries: list[dict] | None = None,
             mode: str | None = None) -> dict:
    """
    One fold's diagnostics for the diagnosis' runs, or with `repair` for the repair probe's (same samples). Another
    study passes its own `entries` (as diagnosis_entries builds them, any fold) and names its `mode`.
    """
    if entries is None:
        if repair and fold not in TR.PROBE_FOLDS:
            raise SystemExit(f"FAIL: the repair probe covers folds {TR.PROBE_FOLDS}, not {fold}")
        if not repair and fold not in DIAGNOSIS_FOLDS:
            raise SystemExit(f"FAIL: the diagnosis covers folds {DIAGNOSIS_FOLDS}, not {fold}")
        entries = repair_entries(fold) if repair else diagnosis_entries(fold)
    elif not mode or repair:
        raise SystemExit("FAIL: a caller's own entries need a mode, and are not the repair probe's")
    started = time.time()
    payload = b1h.load_registered()
    probabilities = b1h.fold_probabilities(fold, payload)
    b1h.assert_fold_sources(fold, probabilities)
    pool_names, held_out = sorted(probabilities), list(FOLDS[fold])
    group_of = {s: "pool" for s in pool_names} | {s: "held_out" for s in held_out}
    contexts = sorted({e["context"] for e in entries})
    loaded = {} if dry_run else {e["label"]: e["load"]() for e in entries}  # fail before building anything

    train_windows = {s: build(s, "train", payload) for s in pool_names}
    b1h.check_window_counts(SourcePool(train_windows, probabilities))
    picks = training_sample(train_windows, probabilities, n_train, SAMPLE_SEED + fold)
    train_parts = {s: subset(train_windows[s], ids) for s, ids in picks.items()}
    del train_windows

    val_parts, val_ids = {}, {}
    for stem in pool_names + held_out:
        windows = build(stem, "val", payload)
        if windows.n_windows:
            val_ids[stem] = validation_sample(windows.n_windows, max_val, stem)
            val_parts[stem] = subset(windows, val_ids[stem])
    birds = list(val_parts)
    reuse = check_context_reuse(birds[0], payload, val_ids[birds[0]], val_parts[birds[0]])
    offsets = np.cumsum([0] + [val_parts[b].n_windows for b in birds])
    bird_rows = {b: np.arange(offsets[i], offsets[i + 1]) for i, b in enumerate(birds)}
    check_rows = np.sort(np.random.default_rng(SAMPLE_SEED + 100 + fold).choice(
        offsets[-1], min(n_check, int(offsets[-1])), replace=False))
    train_species = np.concatenate([[s] * train_parts[s].n_windows for s in train_parts])
    print(f"fold {fold}: {len(train_species):,} training draws over {len(train_parts)} species; "
          f"{offsets[-1]:,} validation windows over {len(birds)} birds ({len(held_out)} held out); "
          f"built in {time.time() - started:.0f}s", flush=True)

    out = {
        "mode": mode or ("repair" if repair else "diagnosis"), "labels": [e["label"] for e in entries],
        "fold": fold, "seed": None if mode else SEED, "contexts": contexts, "arms": list(ARMS), "held_out": held_out,
        "pool": pool_names, "exploratory": True, "opened_test_partition": False, "dry_run": dry_run,
        "samples": {"n_train": int(len(train_species)), "max_val_per_bird": max_val, "n_val": int(offsets[-1]),
                    "n_check": int(len(check_rows)), "sample_seed": SAMPLE_SEED,
                    "val_windows": {b: int(val_parts[b].n_windows) for b in birds}},
        "context_reuse": reuse, "group_of": group_of,
        "loss_curves": {}, "losses": {"train": {}, "val": {}}, "behaviour": {}, "checks": {}, "magnitudes": {},
        "core_only": {}, "skill": {}, "loss_histories": {},
    }

    core_val = stack([gather(val_parts[b], CORE) for b in birds])
    last_m = to_metres(last_step(core_val["features"], np.ones(len(HORIZONS))), np.ones(len(HORIZONS)), core_val["scale"])
    actual_m = core_val["scaled"] * core_val["scale"][:, None, None]
    pred_m_by_label = {"last_step": last_m}
    c12_check = {}

    for context in contexts:
        train = stack([gather(train_parts[s], context) for s in train_parts])
        val = stack([gather(val_parts[b], context) for b in birds]) if context != CORE else core_val
        if context > CORE:
            out["magnitudes"][str(context)] = magnitudes(train, context, train_species)
        if dry_run:
            print(f"  C = {context}: gathered ({time.time() - started:.0f}s)", flush=True)
            continue
        for e in (e for e in entries if e["context"] == context):
            model, k, record = loaded[e["label"]]
            key, run_id = e["label"], e["run_id"]
            t0 = time.time()
            out["loss_curves"][key] = loss_curve(record)
            if repair:
                out["loss_histories"][key] = record.get("loss_history")
            pred_train = predict(model, train["features"], train["token_mask"], context)
            out["losses"]["train"][key] = reference_losses(pred_train, train["features"], train["scaled"],
                                                           train["target_mask"], k)
            pred_val = predict(model, val["features"], val["token_mask"], context)
            out["losses"]["val"][key] = {}
            for group in ("pool", "held_out"):
                rows = np.concatenate([bird_rows[b] for b in birds if group_of[b] == group])
                out["losses"]["val"][key][group] = reference_losses(
                    pred_val[rows], val["features"][rows], val["scaled"][rows], val["target_mask"][rows], k)
            pred_m = to_metres(pred_val, k, val["scale"])
            pred_m_by_label[key] = pred_m
            out["behaviour"][key] = {
                b: {name: behaviour(pred_m[r][m], actual_m[r][m], last_m[r][m], val["scale"][r][m],
                                    val["target_mask"][r][m], {h: BOUNDARY_M[(b, h)] for h in HORIZONS})
                    for name, m in strata_rows(val["share"][r]).items()}
                for b, r in bird_rows.items()
            }
            check = _slice(val, check_rows)
            rng = np.random.default_rng(SAMPLE_SEED + 200 + fold)
            entry = {"run_id": run_id, "token_sensitivity": token_sensitivity(model, check, context)}
            if context > CORE:
                entry["masked_noise"] = masked_noise_check(model, check, context, rng)
            if e["transformer"]:
                entry["fast_path"] = fast_path_check(model, check, context)
            out["checks"][key] = entry
            if context == CORE:
                c12_check[e["pair"]] = to_metres(pred_val[check_rows], k, check["scale"])
            else:
                full = to_metres(pred_val[check_rows], k, check["scale"])
                alone = to_metres(core_only(model, check, context), k, check["scale"])
                compared = [("full_history", full), ("core_only", alone)]
                if e["pair"] in c12_check:
                    compared.append(("c12_model", c12_check[e["pair"]]))
                out["core_only"][key] = {name: _per_head(np.linalg.norm(v, axis=-1)) for name, v in compared}
            print(f"  {run_id}: done in {time.time() - t0:.0f}s", flush=True)

    for b, r in bird_rows.items():
        persistence = travel_p90(val_parts[b], None, b)
        out["skill"][b] = {}
        for label, pred_m in pred_m_by_label.items():
            model_p90 = travel_p90(val_parts[b], pred_m[r], b)
            for h in model_p90:
                cell = out["skill"][b].setdefault(h, {"P_p90_m": persistence[h]["p90_m"],
                                                      "n_individuals": persistence[h]["n_individuals"],
                                                      "n_travel": persistence[h]["n_travel"]})
                p, m = persistence[h]["p90_m"], model_p90[h]["p90_m"]
                cell[label] = None if not p or m is None else 1.0 - m / p
    out["seconds"] = round(time.time() - started, 1)
    return out


# Report.
def _cells(folds: list[dict], path) -> list:
    values = []
    for fold in folds:
        value = path(fold)
        if isinstance(value, list):
            values += value
        elif value is not None:
            values.append(value)
    return values


def _m(values) -> str:
    value = _median([v for v in values if v is not None])
    return "–" if value is None else f"{value:.3g}"


def _get(tree, *keys):
    for key in keys:
        if not isinstance(tree, dict) or key not in tree:
            return None
        tree = tree[key]
    return tree


def summarise(folds: list[dict]) -> dict:
    """Medians over folds (losses, curves, checks) and over bird x fold cells (behaviour, skill)."""
    keys = [f"{arm}_C{c}" for arm in ARMS for c in CONTEXTS]
    summary = {"losses_train": {}, "losses_val": {}, "curves": {}, "behaviour": {}, "strata": {}, "core_only": {},
               "skill": {}}
    for key in keys:
        summary["losses_train"][key] = {
            ref: {head: _median(_cells(folds, lambda f: _get(f, "losses", "train", key, ref, head)))
                  for head in ("mean", *[f"h{h}" for h in HORIZONS])}
            for ref in ("model", "zero", "last_step")}
        summary["losses_val"][key] = {
            group: {ref: _median(_cells(folds, lambda f: _get(f, "losses", "val", key, group, ref, "mean")))
                    for ref in ("model", "zero", "last_step")}
            for group in ("pool", "held_out")}
        summary["curves"][key] = {
            **{str(s): _median(_cells(folds, lambda f: _get(f, "loss_curves", key, "smoothed", str(s))))
               for s in CURVE_STEPS},
            "within_5pct_of_final_from": _median(_cells(folds, lambda f: _get(f, "loss_curves", key,
                                                                             "within_5pct_of_final_from")))}
        summary["core_only"][key] = {
            name: _median(_cells(folds, lambda f: _get(f, "core_only", key, name, "h6")))
            for name in ("full_history", "core_only", "c12_model")}
        for h in map(str, HORIZONS):
            for group in ("pool", "held_out"):
                def cells(metric, stratum="all", group=group, h=h):
                    return _cells(folds, lambda f: [
                        _get(f, "behaviour", key, b, stratum, h, metric)
                        for b in f["behaviour"].get(key, {}) if f["group_of"][b] == group])
                summary["behaviour"].setdefault(key, {}).setdefault(h, {})[group] = {
                    metric: _median(cells(metric))
                    for metric in ("ratio", "cos_last", "cos_actual", "slope", "r2", "median_pred_m",
                                   "median_actual_m", "share_pred_under_1m")}
            for stratum in ("full", "gappy"):
                summary["strata"].setdefault(key, {}).setdefault(h, {})[stratum] = {
                    metric: _median(_cells(folds, lambda f: [
                        _get(f, "behaviour", key, b, stratum, h, metric) for b in f["behaviour"].get(key, {})]))
                    for metric in ("ratio", "slope", "cos_actual")}
    for group in ("pool", "held_out"):
        for h in map(str, HORIZONS):
            summary["skill"].setdefault(group, {})[h] = {
                label: _median(_cells(folds, lambda f: [
                    _get(f, "skill", b, h, label) for b in f["skill"] if f["group_of"][b] == group]))
                for label in ("last_step", *keys)} | {
                "cells": sum(1 for f in folds for b in f["skill"] if f["group_of"][b] == group)}
    return summary


READING = """\
Reading (fixed before the run, in the proposal of 17-10). Not exclusive; the report says which signs were seen.
- (a) learned zero: Bb's training loss at or near the zero predictor's from early on, |pred| / |actual| near 0,
  slope and R^2 near 0, and the bug checks pass.
- (b) undertraining: the loss still clearly falling from 10k to 15k, between zero and LSTM-Bb, with training and
  validation behaviour alike. LSTM-Bb saw the same 5.12 M draws, so (b) would mean the transformer optimises slowly
  on this budget, not that the data were too few.
- (c) a bug: the masked-token or fast-path check fails, or the C > 12 collapse appears only in eval mode or only on
  gappy windows.
No threshold or p value decides any of them."""


def write_markdown(folds: list[dict], summary: dict, path: Path) -> None:
    keys = [f"{arm}_C{c}" for arm in ARMS for c in CONTEXTS]
    lines = [
        "# Transformer collapse diagnosis (exploratory, validation only)", "",
        f"Folds {', '.join(str(f['fold']) for f in folds)}, seed {SEED}; Bb and LSTM-Bb at C = "
        f"{', '.join(map(str, CONTEXTS))}. Existing 1 h checkpoints; no training; no test partition read. "
        f"Samples: {N_TRAIN:,} A8 training draws per fold, up to {MAX_VAL:,} validation windows per bird, "
        f"{N_CHECK:,} for the checks. Pool birds recur across folds, so bird medians are over bird x fold cells. "
        "Validation skill is on a capped sample without the floor; not comparable to any sealed number.", "",
        READING, "",
        "## 1. Loss on the training sample (median over folds; mean over heads)", "",
        "| run | model | zero | last step | model / zero | 6 h model / zero | val pool model / zero | val held-out model / zero |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    ratio = lambda a, b: "–" if a is None or not b else f"{a / b:.3g}"  # noqa: E731
    for key in keys:
        t, v = summary["losses_train"][key], summary["losses_val"][key]
        lines.append(
            f"| {key} | {_m([t['model']['mean']])} | {_m([t['zero']['mean']])} | {_m([t['last_step']['mean']])} | "
            f"{ratio(t['model']['mean'], t['zero']['mean'])} | {ratio(t['model']['h6'], t['zero']['h6'])} | "
            f"{ratio(v['pool']['model'], v['pool']['zero'])} | {ratio(v['held_out']['model'], v['held_out']['zero'])} |")
    lines += ["", "## 1b. Logged training loss (5-point rolling median, median over folds)", "",
              "| run | " + " | ".join(f"{s // 1000}k" for s in CURVE_STEPS) + " | within 5% of final from |",
              "|---|" + "---:|" * (len(CURVE_STEPS) + 1)]
    for key in keys:
        c = summary["curves"][key]
        lines.append(f"| {key} | " + " | ".join(_m([c[str(s)]]) for s in CURVE_STEPS)
                     + f" | {_m([c['within_5pct_of_final_from']])} |")
    for group in ("pool", "held_out"):
        lines += ["", f"## 2. Validation behaviour, {group.replace('_', '-')} birds (travel windows; median over bird x fold)",
                  "", "| run | h | \\|pred\\| / \\|actual\\| | cos(pred, last) | cos(pred, actual) | slope | R² | "
                  "median \\|pred\\| m | median \\|actual\\| m | share \\|pred\\| < 1 m |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for key in keys:
            for h in map(str, HORIZONS):
                b = summary["behaviour"][key][h][group]
                lines.append(f"| {key} | {h} | {_m([b['ratio']])} | {_m([b['cos_last']])} | {_m([b['cos_actual']])} | "
                             f"{_m([b['slope']])} | {_m([b['r2']])} | {_m([b['median_pred_m']])} | "
                             f"{_m([b['median_actual_m']])} | {_m([b['share_pred_under_1m']])} |")
    lines += ["", "## 2b. Full against gappy history (C > 12, all birds; median over bird x fold)", "",
              "| run | h | full ratio | gappy ratio | full slope | gappy slope | full cos(pred, actual) | gappy cos(pred, actual) |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for key in keys:
        if key.endswith(f"_C{CORE}"):
            continue
        for h in map(str, HORIZONS):
            s = summary["strata"][key][h]
            lines.append(f"| {key} | {h} | {_m([s['full']['ratio']])} | {_m([s['gappy']['ratio']])} | "
                         f"{_m([s['full']['slope']])} | {_m([s['gappy']['slope']])} | "
                         f"{_m([s['full']['cos_actual']])} | {_m([s['gappy']['cos_actual']])} |")
    lines += ["", "## 3. Bug checks (per fold; output in model units; 6 h head for the token checks)", "",
              "| fold | run | masked noise max \\|Δ\\| | fast path max \\|Δ\\| | median \\|out\\| | last +1 | last swapped | "
              "earliest +1 |", "|---:|---|---:|---:|---:|---:|---:|---:|"]
    for fold in folds:
        for key in keys:
            c = fold["checks"].get(key)
            if not c:
                continue
            t = c["token_sensitivity"]
            lines.append(
                f"| {fold['fold']} | {key} | {_m([_get(c, 'masked_noise', 'max_abs_diff')])} | "
                f"{_m([_get(c, 'fast_path', 'max_abs_diff')])} | {_m([t['median_abs_output']['h6']])} | "
                f"{_m([t['last_plus_1']['h6']])} | {_m([t['last_swapped']['h6']])} | {_m([t['earliest_plus_1']['h6']])} |")
    lines += ["", "LSTM-Bb reads unobserved tokens as zeros with a flag, so its masked-noise column is information, "
              "not a check. Fast path: Bb only.", "",
              "## 3b. Core only: median 6 h |pred| in metres on the check windows (median over folds)", "",
              "| run | full history | extended tokens masked | the C = 12 model |", "|---|---:|---:|---:|"]
    for key in keys:
        if not key.endswith(f"_C{CORE}"):
            c = summary["core_only"][key]
            lines.append(f"| {key} | {_m([c['full_history']])} | {_m([c['core_only']])} | {_m([c['c12_model']])} |")
    lines += ["", "## 3c. Token magnitudes on the training sample, max(\\|d_east\\|, \\|d_north\\|) scaled", "",
              "| fold | C | part | tokens | > 10 | > 100 | at clip | p99 | most > 100 |", "|---:|---:|---|---:|---:|---:|---:|---:|---|"]
    for fold in folds:
        for context, parts in fold["magnitudes"].items():
            for part, e in parts.items():
                top = e.get("top_species_over_100")
                lines.append(f"| {fold['fold']} | {context} | {part} | {e['tokens']:,} | {_m([e.get('share_over_10')])} | "
                             f"{_m([e.get('share_over_100')])} | {_m([e.get('share_at_clip')])} | {_m([e.get('p99')])} | "
                             f"{'–' if not top else f'{top[0]} {top[1]:.3g}'} |")
    lines += ["", "## 4. Validation travel p90 skill against P (median over bird x fold)", "",
              "| group | h | cells | last step | " + " | ".join(keys) + " |",
              "|---|---:|---:|---:|" + "---:|" * len(keys)]
    for group in ("pool", "held_out"):
        for h in map(str, HORIZONS):
            s = summary["skill"][group][h]
            lines.append(f"| {group.replace('_', '-')} | {h} | {s['cells']} | {_m([s['last_step']])} | "
                         + " | ".join(_m([s[k]]) for k in keys) + " |")
    path.write_text("\n".join(lines) + "\n")


def report() -> None:
    files = [OUT / f"fold{k}.json" for k in DIAGNOSIS_FOLDS]
    missing = [p.name for p in files if not p.exists()]
    if missing:
        raise SystemExit(f"FAIL: missing {', '.join(missing)}; run every fold first")
    folds = [json.loads(p.read_text()) for p in files]
    summary = summarise(folds)
    REPORT_JSON.write_text(json.dumps({"exploratory": True, "opened_test_partition": False, "summary": summary,
                                       "folds": folds}, indent=1, default=str) + "\n")
    write_markdown(folds, summary, REPORT_MD)
    print(f"wrote {REPORT_JSON}\nwrote {REPORT_MD}")


# Repair probe report.
REPAIR_READING = """\
Reading (notes.md 17-10; validation-only model development, no freeze). The readings are input sensitivity (the
last-token swap), |pred| / |actual|, cos(pred, actual) and validation travel p90 skill; every variant is reported,
including any that stay collapsed. Loss / zero barely separates even LSTM-Bb (0.98 in the diagnosis), and skill gaps
inside +-0.05 on one seed are unresolved. Runs of one seed draw identical batches, so the logged losses are paired with
V0's batch for batch. Not tested here: the loss itself as a pull toward zero. Any repaired transformer scored on test
needs its own freeze first."""


def paired_loss(variant: list | None, reference: list | None, last: int = 10) -> dict | None:
    """Variant minus V0 at each logged point (identical batches): median over all points and the last `last`, and the
    share of points where the variant is below V0."""
    if not variant or not reference or len(variant) != len(reference):
        return None
    diff = np.asarray(variant, float) - np.asarray(reference, float)
    return {"points": int(len(diff)), "median": float(np.median(diff)), f"median_last_{last}": float(np.median(diff[-last:])),
            "share_below": float((diff < 0).mean())}


def repair_summary(folds: list[dict]) -> dict:
    """Per run label: losses, sensitivity and checks (median over folds); behaviour and skill (median over bird x fold
    cells, by group); the paired logged loss against V0."""
    labels = folds[0]["labels"]
    out = {}
    for label in labels:
        base, context = label.split("_C")
        v0 = f"{base[:-2]}V0_C{context}"
        entry = {
            "loss_over_zero": {}, "sensitivity_h6": {}, "checks": {}, "behaviour": {}, "skill": {},
            "paired_loss": None if base.endswith("V0") else [paired_loss(f["loss_histories"].get(label),
                                                                         f["loss_histories"].get(v0)) for f in folds],
        }
        for head in ("mean", "h6"):
            entry["loss_over_zero"][f"train_{head}"] = _median(_cells(folds, lambda f: _ratio(
                _get(f, "losses", "train", label, "model", head), _get(f, "losses", "train", label, "zero", head))))
        for group in ("pool", "held_out"):
            entry["loss_over_zero"][f"val_{group}"] = _median(_cells(folds, lambda f: _ratio(
                _get(f, "losses", "val", label, group, "model", "mean"),
                _get(f, "losses", "val", label, group, "zero", "mean"))))
        for name in ("median_abs_output", "last_plus_1", "last_swapped"):
            entry["sensitivity_h6"][name] = _median(_cells(folds, lambda f: _get(
                f, "checks", label, "token_sensitivity", name, "h6")))
        entry["checks"] = {
            "masked_noise_max": _median(_cells(folds, lambda f: _get(f, "checks", label, "masked_noise", "max_abs_diff"))),
            "fast_path_max": _median(_cells(folds, lambda f: _get(f, "checks", label, "fast_path", "max_abs_diff"))),
        }
        for h in map(str, HORIZONS):
            for group in ("pool", "held_out"):
                def birds_of(f, group=group):
                    return [b for b in f["behaviour"].get(label, {}) if f["group_of"][b] == group]
                entry["behaviour"].setdefault(h, {})[group] = {
                    metric: _median(_cells(folds, lambda f: [_get(f, "behaviour", label, b, "all", h, metric)
                                                             for b in birds_of(f)]))
                    for metric in ("ratio", "cos_actual", "median_pred_m")}
                entry["skill"].setdefault(h, {})[group] = _median(_cells(folds, lambda f: [
                    _get(f, "skill", b, h, label) for b in f["skill"] if f["group_of"][b] == group]))
        out[label] = entry
    return out


def _ratio(a, b):
    return None if a is None or not b else a / b


def write_repair_markdown(folds: list[dict], summary: dict, path: Path) -> None:
    labels = folds[0]["labels"]
    lines = [
        "# Transformer repair probe: validation diagnostics (exploratory, validation only)", "",
        f"Fold(s) {', '.join(str(f['fold']) for f in folds)}, seed {SEED}; the repair plan's runs "
        f"(`transformer_repair_train.py --list`). Same samples as the collapse diagnosis: {N_TRAIN:,} A8 training "
        f"draws, up to {MAX_VAL:,} validation windows per bird, {N_CHECK:,} for the checks. No test partition read. "
        "Validation skill is on a capped sample without the floor; not comparable to any sealed number.", "",
        REPAIR_READING, "",
        "## 1. Loss, input sensitivity and checks (6 h head for sensitivity; model units)", "",
        "| run | train loss / zero | 6 h train loss / zero | val pool loss / zero | val held-out loss / zero | "
        "median \\|out\\| | last +1 | last swapped | masked noise max \\|Δ\\| | fast path max \\|Δ\\| |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label in labels:
        s = summary[label]
        lz, sens, checks = s["loss_over_zero"], s["sensitivity_h6"], s["checks"]
        lines.append(f"| {label} | {_m([lz['train_mean']])} | {_m([lz['train_h6']])} | {_m([lz['val_pool']])} | "
                     f"{_m([lz['val_held_out']])} | {_m([sens['median_abs_output']])} | {_m([sens['last_plus_1']])} | "
                     f"{_m([sens['last_swapped']])} | {_m([checks['masked_noise_max']])} | {_m([checks['fast_path_max']])} |")
    lines += ["", "## 2. Validation behaviour and travel p90 skill against P (travel windows; median over birds)", "",
              "| run | h | \\|pred\\| / \\|actual\\| pool | held-out | cos(pred, actual) pool | held-out | "
              "median \\|pred\\| m pool | skill pool | skill held-out |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for label in labels:
        for h in map(str, HORIZONS):
            b, k = summary[label]["behaviour"][h], summary[label]["skill"][h]
            lines.append(f"| {label} | {h} | {_m([b['pool']['ratio']])} | {_m([b['held_out']['ratio']])} | "
                         f"{_m([b['pool']['cos_actual']])} | {_m([b['held_out']['cos_actual']])} | "
                         f"{_m([b['pool']['median_pred_m']])} | {_m([k['pool']])} | {_m([k['held_out']])} |")
    lines += ["", "## 3. Logged training loss against V0 on identical batches (variant − V0; per fold)", "",
              "| run | fold | points | median Δ | median Δ, last 10 | share below V0 |", "|---|---:|---:|---:|---:|---:|"]
    for label in labels:
        for fold, p in zip(folds, summary[label]["paired_loss"] or []):
            if p:
                lines.append(f"| {label} | {fold['fold']} | {p['points']} | {p['median']:.3g} | "
                             f"{p['median_last_10']:.3g} | {p['share_below']:.2f} |")
    path.write_text("\n".join(lines) + "\n")


def repair_report() -> None:
    files = [REPAIR_OUT / f"fold{k}.json" for k in TR.PROBE_FOLDS]
    missing = [p.name for p in files if not p.exists()]
    if missing:
        raise SystemExit(f"FAIL: missing {', '.join(missing)}; run --repair --fold first")
    folds = [json.loads(p.read_text()) for p in files]
    if any(f.get("mode") != "repair" for f in folds):
        raise SystemExit("FAIL: a repair fold file was not written by --repair")
    summary = repair_summary(folds)
    REPAIR_JSON.write_text(json.dumps({"exploratory": True, "opened_test_partition": False, "summary": summary,
                                       "folds": folds}, indent=1, default=str) + "\n")
    write_repair_markdown(folds, summary, REPAIR_MD)
    print(f"wrote {REPAIR_JSON}\nwrote {REPAIR_MD}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fold", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--repair", action="store_true", help="the repair probe's runs (results/transformer_repair/)")
    parser.add_argument("--n-train", type=int, default=N_TRAIN)
    parser.add_argument("--max-val", type=int, default=MAX_VAL)
    parser.add_argument("--n-check", type=int, default=N_CHECK)
    args = parser.parse_args()
    if args.report:
        return repair_report() if args.repair else report()
    if args.fold is None:
        parser.error("--fold or --report")
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
    result = run_fold(args.fold, n_train=args.n_train, max_val=args.max_val, n_check=args.n_check, dry_run=args.dry_run,
                      repair=args.repair)
    if args.dry_run:
        print(json.dumps({k: result[k] for k in ("context_reuse", "magnitudes")}, indent=1))
        for group in ("pool", "held_out"):
            for h in map(str, HORIZONS):
                cells = [result["skill"][b][h]["last_step"] for b in result["skill"] if result["group_of"][b] == group]
                print(f"last-step validation skill, {group}, {h} h: median {_m(cells)} over {len(cells)} birds")
        print(f"dry run done in {result['seconds']:.0f}s")
        return
    out_dir = REPAIR_OUT if args.repair else OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"fold{args.fold}.json"
    path.write_text(json.dumps(result, indent=1, default=str) + "\n")
    print(f"wrote {path} ({result['seconds']:.0f}s)")


if __name__ == "__main__":
    main()
