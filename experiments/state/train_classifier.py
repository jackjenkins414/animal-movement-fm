# GenAI reference: I specified the whole design (window->next-state target, XGBoost on the flattened 6-channel window
# plus momentum summaries derived from the same window, balanced weights, and the exact metric set: per-class
# precision/recall + one-vs-rest PR-AUC with migratory_launch as headline, and the can_launch foraging-ending hard
# slice). GenAI helped write the featurisation vectorisation, the metric printing and the JSON dump. Verified by a
# smoke run whose window counts / class prevalences match state_composition.py and the dataset builder's own numbers.
"""
State classifier: forecast the next movement state from a 24-fix window (Part B).

One lean model (not a ladder): an XGBoost multiclass classifier that, from the standard 24-fix / 23-step window (the
same input the LSTM/Transformer see), predicts the refined state of the NEXT fix -- the fix whose position the movement
head predicts. This is a genuine one-step forecast, so it can later become a second head on the neural trunk (John's
two-factor / mixture-of-experts idea); here we just measure how predictable state, and migratory-launch onset in
particular, actually is.

Target and slices
-----------------
Five classes (foraging / moving / local_launch / migratory_launch / settle). Accuracy is meaningless at ~2%
migratory-launch prevalence, so we report PER-CLASS precision/recall and one-vs-rest PR-AUC (average precision), with
`migratory_launch` as the headline class and its prevalence as the no-skill floor. We also report the CAN-LAUNCH slice
(windows whose last fix is foraging -- the only place a launch is mechanically possible): the honest "hard subset"
where the base rate is higher and the decision is real.

Features
--------
The (23, C) window is flattened, plus a few momentum summaries computed from the SAME window displacements (recent-
movement momentum was the dominant onset signal in the earlier probe). These summaries are a fixed function of the
window, so they add no information the neural head could not itself derive -- they just hand the trees the aggregate
directly. Magnitudes use hypot(dlon, dlat) in degrees, a monotone proxy (longitude compresses with latitude); the raw
per-step channels are also present, so the trees are not reliant on the proxy alone.

Run
---
    uv run python experiments/state/train_classifier.py            # full run (all train birds, 6 channels)
    uv run python experiments/state/train_classifier.py --quick    # subsample + fewer rounds, for a fast smoke test
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from animal_movement_fm.analysis.regime import REFINED_STATES
from animal_movement_fm.data.benchmark import load_split_frame
from animal_movement_fm.data.split import stork_split
from animal_movement_fm.state.dataset import IDX_TO_STATE, build_state_windows

REPO = Path(__file__).resolve().parents[2]
SEGMENTS = REPO / "data" / "processed" / "stork_segments.parquet"
RESULTS = REPO / "results"

HEADLINE = "migratory_launch"
SEED = 42
CONTEXT_LENGTH = 24


# --------------------------------------------------------------------------------------
# featurisation
# --------------------------------------------------------------------------------------
def featurise(sw) -> tuple[np.ndarray, list[str]]:
    """
    Flatten the (N, L, C) window and append momentum summaries derived from the same window.

    Returns the (N, L*C + n_summary) design matrix and the matching feature names (for importance readouts).
    """
    X = sw.X
    N, L, C = X.shape
    flat = X.reshape(N, L * C)
    flat_names = [f"{sw.channels[c]}_t{j}" for j in range(L) for c in range(C)]

    dlon, dlat = X[:, :, 0], X[:, :, 1]
    mag = np.hypot(dlon, dlat)  # per-step degree magnitude (proxy; longitude compresses with latitude)
    net_lon, net_lat = dlon.sum(1), dlat.sum(1)
    net = np.hypot(net_lon, net_lat)  # straight-line window displacement
    path = mag.sum(1)  # summed step length
    half = L // 2
    with np.errstate(divide="ignore", invalid="ignore"):
        straightness = np.where(path > 0, net / path, 0.0)  # directed transit ~1; out-and-back ~0

    summaries = np.column_stack(
        [
            mag.mean(1),
            mag.max(1),
            mag[:, -1],  # last step (current momentum)
            mag[:, -3:].mean(1),  # recent 3-step momentum
            mag.std(1),
            net,
            path,
            straightness,
            mag[:, half:].mean(1) - mag[:, :half].mean(1),  # late-minus-early (acceleration)
            net_lat,  # net N/S displacement (migration direction has a sign)
        ]
    )
    summary_names = [
        "mag_mean", "mag_max", "mag_last", "mag_last3", "mag_std",
        "net", "path", "straightness", "accel", "net_lat",
    ]
    # float32: XGBoost's hist method uses float32 internally, and it halves the design-matrix footprint.
    return np.hstack([flat, summaries]).astype(np.float32), flat_names + summary_names


# --------------------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------------------
def per_class_table(y_true: np.ndarray, y_pred: np.ndarray, proba: np.ndarray) -> pd.DataFrame:
    """Per-class precision / recall / F1 / support / prevalence / one-vs-rest PR-AUC (average precision)."""
    labels = list(range(len(REFINED_STATES)))
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    rows = []
    n = len(y_true)
    for k in labels:
        pos = y_true == k
        ap = average_precision_score(pos, proba[:, k]) if pos.any() and not pos.all() else np.nan
        rows.append(
            {
                "state": REFINED_STATES[k],
                "precision": p[k],
                "recall": r[k],
                "f1": f[k],
                "support": int(s[k]),
                "prevalence": s[k] / n,
                "pr_auc": ap,
                "pr_auc_lift": ap / (s[k] / n) if s[k] else np.nan,  # vs no-skill = prevalence
            }
        )
    return pd.DataFrame(rows).set_index("state")


def headline_slice(y_true, proba, mask, name: str) -> dict:
    """Headline migratory_launch metrics on a subset (mask) of the test windows."""
    k = REFINED_STATES.index(HEADLINE)
    yt, pr = y_true[mask], proba[mask, k]
    pos = yt == k
    ap = average_precision_score(pos, pr) if pos.any() and not pos.all() else float("nan")
    prev = float(pos.mean())
    return {
        "slice": name,
        "n": int(mask.sum()),
        "prevalence": prev,
        "pr_auc": float(ap),
        "pr_auc_lift": float(ap / prev) if prev else float("nan"),
    }


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--no-time", action="store_true", help="drop time-of-day channels")
    p.add_argument("--no-season", action="store_true", help="drop day-of-year channels")
    p.add_argument("--n-estimators", type=int, default=400)
    p.add_argument("--max-depth", type=int, default=6)
    p.add_argument("--learning-rate", type=float, default=0.1)
    p.add_argument(
        "--max-train-birds", type=int, default=0,
        help="0 = all; else subsample this many train individuals (quick smoke runs)",
    )
    p.add_argument("--quick", action="store_true", help="shorthand: 40 train birds, 60 rounds")
    return p.parse_args()


def build(split_ids, use_time, use_season):
    frame = load_split_frame(SEGMENTS, split_ids)
    return build_state_windows(
        frame, context_length=CONTEXT_LENGTH,
        use_time_features=use_time, use_season_features=use_season,
    )


def main() -> None:
    args = parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    use_time = not args.no_time
    use_season = not args.no_season
    n_estimators = 60 if args.quick else args.n_estimators
    max_train_birds = 40 if args.quick else args.max_train_birds

    # frozen manifest, not a recomputed split, so these numbers sit on the same individuals as the Gadi runs
    split = stork_split()
    train_ids = split.train
    if max_train_birds:
        rng = np.random.default_rng(SEED)
        train_ids = rng.permutation(train_ids)[:max_train_birds]

    print(
        f"channels: dlon,dlat"
        + (",tod_sin,tod_cos" if use_time else "")
        + (",doy_sin,doy_cos" if use_season else "")
        + f" | train birds: {len(train_ids)} | rounds: {n_estimators}"
    )

    t0 = time.time()
    tr = build(train_ids, use_time, use_season)
    va = build(split.val, use_time, use_season)
    te = build(split.test, use_time, use_season)
    Xtr, names = featurise(tr)
    Xva, _ = featurise(va)
    Xte, _ = featurise(te)
    print(
        f"windows -> train {len(tr.y):,} | val {len(va.y):,} | test {len(te.y):,} "
        f"| {Xtr.shape[1]} features | built in {time.time() - t0:.1f}s"
    )

    clf = XGBClassifier(
        objective="multi:softprob",
        num_class=len(REFINED_STATES),
        tree_method="hist",
        n_estimators=n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        early_stopping_rounds=30,
        random_state=SEED,
        n_jobs=-1,
    )
    # balanced weights so the ~2% migratory_launch class is not ignored (accuracy-chasing predicts 'moving').
    weights = compute_sample_weight("balanced", tr.y)
    t0 = time.time()
    clf.fit(Xtr, tr.y, sample_weight=weights, eval_set=[(Xva, va.y)], verbose=False)
    print(f"trained in {time.time() - t0:.1f}s (best_iteration={clf.best_iteration})")

    proba = clf.predict_proba(Xte)
    y_pred = proba.argmax(1)
    y_true = te.y

    # --- per-class table (headline metric set) ---
    table = per_class_table(y_true, y_pred, proba)
    print("\nper-class metrics (test):")
    print(table.to_string(float_format=lambda v: f"{v:6.3f}"))

    # --- accuracy floor, explicitly flagged as uninformative here ---
    acc = float((y_pred == y_true).mean())
    majority = float((y_true == np.bincount(y_true).argmax()).mean())
    print(f"\naccuracy {acc:.3f} (majority-class floor {majority:.3f}) -- uninformative at this imbalance")

    # --- headline slices: full test vs the can_launch (foraging-ending) hard subset ---
    can = te.meta["can_launch"].to_numpy()
    slices = [
        headline_slice(y_true, proba, np.ones(len(y_true), bool), "full_test"),
        headline_slice(y_true, proba, can, "can_launch"),
    ]
    print(f"\nheadline class = {HEADLINE}:")
    for sl in slices:
        print(
            f"  {sl['slice']:<11} n={sl['n']:>7,} prevalence={sl['prevalence']:.4f} "
            f"PR-AUC={sl['pr_auc']:.4f} (lift x{sl['pr_auc_lift']:.1f} vs no-skill)"
        )

    # --- confusion matrix ---
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(REFINED_STATES))))
    cm_df = pd.DataFrame(cm, index=REFINED_STATES, columns=REFINED_STATES)
    print("\nconfusion matrix (rows=true, cols=pred):")
    print(cm_df.to_string())

    # --- top feature importances (gain) ---
    imp = pd.Series(clf.feature_importances_, index=names).sort_values(ascending=False)
    print("\ntop 15 features (gain):")
    print(imp.head(15).to_string(float_format=lambda v: f"{v:.4f}"))

    # --- persist ---
    out = {
        "config": {
            "context_length": CONTEXT_LENGTH,
            "use_time_features": use_time,
            "use_season_features": use_season,
            "n_estimators": n_estimators,
            "max_depth": args.max_depth,
            "learning_rate": args.learning_rate,
            "n_train_birds": int(len(train_ids)),
            "best_iteration": int(clf.best_iteration),
            "seed": SEED,
        },
        "n_windows": {"train": int(len(tr.y)), "val": int(len(va.y)), "test": int(len(te.y))},
        "accuracy": acc,
        "majority_floor": majority,
        "per_class": table.reset_index().to_dict(orient="records"),
        "headline_slices": slices,
        "top_features": imp.head(20).to_dict(),
    }
    path = RESULTS / "state_classifier_metrics.json"
    path.write_text(json.dumps(out, indent=2, default=float))
    print(f"\nmetrics -> {path}")


if __name__ == "__main__":
    main()
