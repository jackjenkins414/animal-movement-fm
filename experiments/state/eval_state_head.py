# GenAI reference: I wrote everything above parse_args() and then used GenAI to finish the CLI, model-building,
# main training/evaluation loop, formatted reporting, and JSON/CSV exports to save time. I verified the 
# generated section by checking that baseline and treatment used identical seeds/configs except for use_state_head, 
# and that the saved JSON/CSV matched the printed table; got GenAI to refine my PBS script as well. 
"""
Does the two-factor state head improve movement forecasting? (Part B evaluation)

Trains two LSTMs on the seed-42 split, identical in every way except the head:

    baseline    time + season covariates, no state head       (lstm_h64_l1_time_season)
    treatment   same, plus the Rung-2 state head              (lstm_h64_l1_time_season_state)

Both are evaluated through the standard harness on the test split. The comparison is the movement metric that
matters, reported as the FULL error distribution (n, mean, median, p90, p99, std) BOTH overall AND stratified by the
true next state, so we can see whether the regime-conditioning helps the tail (moving / launch p99) without hurting
the median. Persistence is evaluated too as the skill floor. The state head's own next-state accuracy is reported as
a sanity check (it should roughly track the standalone classifier).

Expectation/guess: the moving/launch p99 improves under the treatment; the overall median barely moves.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import torch

from animal_movement_fm.analysis.regime import REFINED_STATES, label_migratory_launches, label_regimes
from animal_movement_fm.data.benchmark import load_split_frame
from animal_movement_fm.data.split import stork_split
from animal_movement_fm.eval.harness import evaluate
from animal_movement_fm.eval.metrics import summarise_errors
from animal_movement_fm.models.lstm import LSTM
from animal_movement_fm.models.persistence import Persistence
from animal_movement_fm.models.transformer import Transformer
from animal_movement_fm.state.dataset import build_state_windows

logging.basicConfig(level=logging.INFO, format="%(message)s")

REPO = Path(__file__).resolve().parents[2]
SEGMENTS = REPO / "data" / "processed" / "stork_segments.parquet"
RESULTS = REPO / "results" / "state_head"

SPLIT_SEED = 42
CONTEXT_LENGTH = 24
# established per architecture recipes (unchanged from the multi-seed sweep, so numbers stay comparable). Both are
# run with time + season covariates, the best-covariate cell.
BASE = {
    "lstm": dict(hidden_dim=64, num_layers=1, context_length=CONTEXT_LENGTH, epochs=80, batch_size=256, lr=1e-3),
    "transformer": dict(d_model=64, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.1,
                        context_length=CONTEXT_LENGTH, epochs=80, batch_size=256, lr=3e-4,
                        warmup_frac=0.1, grad_clip=1.0),
}
MODEL_CLS = {"lstm": LSTM, "transformer": Transformer}
STATE_ORDER = list(REFINED_STATES) + ["overall"]


# helper for stratif
def label_state_by_fix(frame: pd.DataFrame) -> pd.DataFrame:
    """
    (segment_id, timestamp, state) table for the target fix of every window, from the canonical labeller.
    """
    lab = label_migratory_launches(label_regimes(frame))
    return lab[["segment_id", "timestamp", "state"]]


def attach_state(results: pd.DataFrame, state_tbl: pd.DataFrame) -> pd.DataFrame:
    """
    Tag each prediction row with the true next-state of its target fix (join on segment_id + target timestamp).
    """
    return results.merge(state_tbl, on=["segment_id", "timestamp"], how="left")


def dist_by_state(results: pd.DataFrame) -> dict[str, dict]:
    """
    summarise_errors overall and per true next-state, in metres.
    """
    out = {"overall": summarise_errors(results["error_m"].to_numpy())}
    for state, g in results.groupby("state"):
        out[state] = summarise_errors(g["error_m"].to_numpy())
    return out


def state_head_accuracy(model: LSTM, frame: pd.DataFrame) -> dict:
    """
    Next-state accuracy of the treatment model's state head on `frame`.

    Rebuilds the net's exact input (build_state_windows gives the identical 6 channel window; the two displacement
    channels are standardised with the modesls stored train stats), runs the state head, and compares argmax to the
    canonical labels. Reports overall accuracy plus migratory_launch recall as the headline sanity check.
    """
    sw = build_state_windows(
        frame, context_length=model.context_length,
        use_time_features=model.use_time_features, use_season_features=model.use_season_features,
    )
    X = sw.X.astype(np.float32).copy()
    mean = model._mean.detach().cpu().numpy().reshape(1, 1, 2)
    std = model._std.detach().cpu().numpy().reshape(1, 1, 2)
    X[:, :, :2] = (X[:, :, :2] - mean) / std

    model.net.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(X), 50_000):  # chunk to keep memory modest
            xb = torch.tensor(X[start : start + 50_000], dtype=torch.float32).to(model.device)
            _, logits = model.net(xb)
            preds.append(logits.argmax(1).cpu().numpy())
    pred = np.concatenate(preds)

    mig = REFINED_STATES.index("migratory_launch")
    pos = sw.y == mig
    return {
        "accuracy": float((pred == sw.y).mean()),
        "migratory_launch_recall": float((pred[pos] == mig).mean()) if pos.any() else float("nan"),
        "n": int(len(sw.y)),
    }


def km(v: float) -> float:
    return v / 1000.0


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arch", choices=("lstm", "transformer"), default="lstm", help="architecture to compare")
    p.add_argument("--seed", type=int, default=0, help="TRAINING seed (weight init + shuffling); split stays 42")
    p.add_argument("--state-loss-weight", type=float, default=0.05,
                   help="lambda on the state cross-entropy (position MSE ~0.02 vs CE ~0.7, so ~0.03 balances them)")
    p.add_argument("--epochs", type=int, default=None, help="override epochs")
    p.add_argument("--max-train-birds", type=int, default=0, help="0 = all; subsample train individuals for a smoke run")
    p.add_argument("--quick", action="store_true", help="shorthand: 60 train birds, 8 epochs")
    return p.parse_args()


def build_model(arch: str, use_state_head: bool, seed: int, epochs: int, state_loss_weight: float, run_id: str):
    torch.manual_seed(seed)  # Re-seed before each model so baseline and treatment start identically. 
    cfg = dict(BASE[arch])
    cfg["epochs"] = epochs
    return MODEL_CLS[arch](
        **cfg,
        use_time_features=True,
        use_season_features=True,
        use_state_head=use_state_head,
        state_loss_weight=state_loss_weight,
        name=run_id,
    )


def main() -> None:
    args = parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    epochs = 8 if args.quick else (args.epochs or BASE[args.arch]["epochs"])
    max_train_birds = 60 if args.quick else args.max_train_birds

    if not SEGMENTS.exists():
        raise SystemExit(f"FAIL: data not found at {SEGMENTS}")

    # frozen manifest, not a recomputed split (see data/split.stork_split)
    split = stork_split()
    train_ids = split.train
    if max_train_birds:
        train_ids = np.random.default_rng(SPLIT_SEED).permutation(train_ids)[:max_train_birds]

    train_frame = load_split_frame(SEGMENTS, train_ids)
    test_frame = load_split_frame(SEGMENTS, split.test)
    print(
        f"arch {args.arch} | device cuda={torch.cuda.is_available()} | train {len(train_frame):,} rows / "
        f"{train_frame['segment_id'].nunique()} segs | test {len(test_frame):,} rows / "
        f"{test_frame['segment_id'].nunique()} segs | epochs {epochs} | lambda {args.state_loss_weight}"
    )

    models = {
        "baseline": build_model(args.arch, False, args.seed, epochs, args.state_loss_weight, f"{args.arch}_time_season"),
        "treatment": build_model(args.arch, True, args.seed, epochs, args.state_loss_weight, f"{args.arch}_time_season_state"),
    }

    state_tbl = label_state_by_fix(test_frame)
    dists: dict[str, dict] = {}

    # Persistence skill floor. 
    pers = attach_state(
        evaluate(Persistence(), test_frame, context_length=CONTEXT_LENGTH,
                 individual_col="segment_id", show_progress=False, extra_cols=("individual_id",)),
        state_tbl,
    )
    dists["persistence"] = dist_by_state(pers)

    head_acc = None
    for tag, model in models.items():
        print(f"\ntraining {tag} ({model.name})...")
        t0 = time.time()
        model.fit(train_frame, group_col="segment_id")
        print(f"  fit in {(time.time() - t0) / 60:.1f} min")

        results = attach_state(
            evaluate(model, test_frame, context_length=CONTEXT_LENGTH,
                     individual_col="segment_id", show_progress=False, extra_cols=("individual_id",)),
            state_tbl,
        )
        dists[tag] = dist_by_state(results)
        if tag == "treatment":
            head_acc = state_head_accuracy(model, test_frame)

    # ---- report: per-state median & p99 (km), baseline vs treatment vs persistence ----
    print("\n" + "=" * 92)
    print("movement error by true next-state (km): median | p99      [n]")
    print(f"{'state':<18}{'persist med':>12}{'base med':>10}{'treat med':>11}"
          f"{'base p99':>11}{'treat p99':>11}{'d p99 %':>9}")
    print("-" * 92)
    rows_out = []
    for state in STATE_ORDER:
        if state not in dists["treatment"]:
            continue
        p = dists["persistence"].get(state, {})
        b = dists["baseline"][state]
        t = dists["treatment"][state]
        dp99 = (t["p99_m"] - b["p99_m"]) / b["p99_m"] * 100.0 if b["p99_m"] else float("nan")
        print(f"{state:<18}{km(p.get('median_m', float('nan'))):>12.3f}{km(b['median_m']):>10.3f}"
              f"{km(t['median_m']):>11.3f}{km(b['p99_m']):>11.3f}{km(t['p99_m']):>11.3f}{dp99:>8.1f}%")
        rows_out.append({
            "state": state, "n": t["n"],
            "persist_median_km": km(p.get("median_m", float("nan"))),
            "base_median_km": km(b["median_m"]), "treat_median_km": km(t["median_m"]),
            "base_mean_km": km(b["mean_m"]), "treat_mean_km": km(t["mean_m"]),
            "base_p90_km": km(b["p90_m"]), "treat_p90_km": km(t["p90_m"]),
            "base_p99_km": km(b["p99_m"]), "treat_p99_km": km(t["p99_m"]),
            "delta_p99_pct": dp99,
        })

    ov_b, ov_t = dists["baseline"]["overall"], dists["treatment"]["overall"]
    ov_p = dists["persistence"]["overall"]
    print("-" * 92)
    print(f"overall skill vs persistence (median): baseline {1 - ov_b['median_m'] / ov_p['median_m']:+.3f} | "
          f"treatment {1 - ov_t['median_m'] / ov_p['median_m']:+.3f}")
    print(f"overall mean km : baseline {km(ov_b['mean_m']):.3f} -> treatment {km(ov_t['mean_m']):.3f}")
    if head_acc:
        print(f"state head (test): accuracy {head_acc['accuracy']:.3f} | "
              f"migratory_launch recall {head_acc['migratory_launch_recall']:.3f} (n={head_acc['n']:,})")

    # ---- persist ----
    out = {
        "config": {
            "arch": args.arch, "seed": args.seed, "split_seed": SPLIT_SEED, "epochs": epochs,
            "state_loss_weight": args.state_loss_weight, "n_train_birds": int(len(train_ids)),
            "context_length": CONTEXT_LENGTH,
        },
        "by_state": rows_out,
        "overall": {
            "persist_median_km": km(ov_p["median_m"]),
            "baseline": {k: km(v) if k.endswith("_m") else v for k, v in ov_b.items()},
            "treatment": {k: km(v) if k.endswith("_m") else v for k, v in ov_t.items()},
            "skill_median_baseline": 1 - ov_b["median_m"] / ov_p["median_m"],
            "skill_median_treatment": 1 - ov_t["median_m"] / ov_p["median_m"],
        },
        "state_head": head_acc,
    }
    tag = f"{args.arch}_s{args.seed}_lam{args.state_loss_weight}"
    path = RESULTS / f"eval_state_head_{tag}.json"
    path.write_text(json.dumps(out, indent=2, default=float))
    pd.DataFrame(rows_out).to_csv(RESULTS / f"eval_state_head_{tag}.csv", index=False)
    print(f"\nmetrics -> {path}")


if __name__ == "__main__":
    main()
