# GenAI reference: I wrote everything above parse_args() and then used GenAI to finish the CLI, model-building,
# main training/evaluation loop, formatted reporting, and JSON/CSV exports to save time. I verified the 
# generated section by checking that baseline and treatment used identical seeds/configs except for use_state_head, 
# and that the saved JSON/CSV matched the printed table; got GenAI to refine my PBS script as well. 
"""
Does regime conditioning improve movement forecasting? (Part B evaluation)

Trains two models on the frozen split, identical in every way except the head:

    baseline    time + season covariates, one plain position head
    treatment   same trunk, plus either the Rung-2 state head or the Rung-3 mixture of experts

Both are evaluated through the standard harness on the test split. The comparison is the movement metric that
matters, reported as the FULL error distribution (n, mean, median, p90, p99, std) BOTH overall AND stratified by the
true next state, so we can see whether the regime-conditioning helps the tail (moving / launch p99) without hurting
the median. Persistence is evaluated too as the skill floor. The treatment head's own next-state accuracy is
reported as a sanity check (it should roughly track the standalone classifier).

Treatments
----------
`--treatment state_head` is Rung 2: the soft state distribution is concatenated onto the trunk summary before a
single position head. That returned a null, for a structural reason since the head is linear, so the state enters
ADDITIVELY and can only apply a per regime offset; it cannot change how the head reads the trunk.

`--treatment moe` is Rung 3: a gate over K experts whose softmax WEIGHTS K independent position heads, so the
effective map `sum_k g_k W_k` varies window by window. That is MULTIPLICATIVE, and is the mechanism the Rung 2 null
did not test. For the MoE the report adds routing diagnostics (per expert usage, and mean gate mass per expert
within each true state), because a gate that collapses onto one expert silently degenerates to the baseline and
would otherwise look like a null for the wrong reason.

Versions for running
--------------------
    python experiments/state/eval_state_head.py                     # LSTM, Rung-2 state head (default)
    python experiments/state/eval_state_head.py --treatment moe     # LSTM, Rung-3 mixture of experts
    python experiments/state/eval_state_head.py --arch transformer --treatment moe
    python experiments/state/eval_state_head.py --quick             # Mac smoke: subsample birds, few epochs
    python experiments/state/eval_state_head.py --seed 1            # different TRAINING seed (split stays 42)
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


def head_diagnostics(model, frame: pd.DataFrame) -> dict:
    """
    Diagnostics for the treatment model's 5-way head (the Rung-2 state head, or the Rung-3 MoE gate).

    Rebuilds the net's exact input (build_state_windows gives the identical 6 channel window; the two displacement
    channels are standardised with the model's stored train stats), runs the head, and compares argmax to the
    canonical labels. Reports overall accuracy plus migratory_launch recall as the headline sanity check.

    For the MoE this doubles as the routing report. Two extra fields are returned, because a gate that has collapsed
    onto one expert reduces the mixture to a single shared head and would silently reproduce the baseline:

      expert_usage    mean softmax mass on each expert over all test windows (should track state prevalence)
      expert_by_state mean gate mass per expert WITHIN each true state; the diagonal is the routing we want, so an
                      off diagonal maximum means expert k is not specialising on state k
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
    preds, probs = [], []
    with torch.no_grad():
        for start in range(0, len(X), 50_000):  # chunk to keep memory modest
            xb = torch.tensor(X[start : start + 50_000], dtype=torch.float32).to(model.device)
            _, logits = model.net(xb)
            preds.append(logits.argmax(1).cpu().numpy())
            probs.append(torch.softmax(logits, dim=-1).cpu().numpy())
    pred = np.concatenate(preds)
    prob = np.concatenate(probs)

    mig = REFINED_STATES.index("migratory_launch")
    pos = sw.y == mig
    out = {
        "accuracy": float((pred == sw.y).mean()),
        "migratory_launch_recall": float((pred[pos] == mig).mean()) if pos.any() else float("nan"),
        "n": int(len(sw.y)),
    }

    if getattr(model, "use_moe", False):
        out["expert_usage"] = {REFINED_STATES[k]: float(prob[:, k].mean()) for k in range(prob.shape[1])}
        out["expert_by_state"] = {
            REFINED_STATES[s]: {
                REFINED_STATES[k]: float(prob[sw.y == s, k].mean()) for k in range(prob.shape[1])
            }
            for s in range(len(REFINED_STATES))
            if (sw.y == s).any()
        }
    return out


def km(v: float) -> float:
    return v / 1000.0


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arch", choices=("lstm", "transformer"), default="lstm", help="architecture to compare")
    p.add_argument("--treatment", choices=("state_head", "moe"), default="state_head",
                   help="what to compare against the baseline: the Rung-2 state head or the Rung-3 mixture of experts")
    p.add_argument("--seed", type=int, default=0, help="TRAINING seed (weight init + shuffling); split stays 42")
    p.add_argument("--state-loss-weight", type=float, default=0.05,
                   help="lambda on the state cross-entropy (position MSE ~0.02 vs CE ~0.7, so ~0.03 balances them)")
    p.add_argument("--epochs", type=int, default=None, help="override epochs")
    p.add_argument("--max-train-birds", type=int, default=0, help="0 = all; subsample train individuals for a smoke run")
    p.add_argument("--quick", action="store_true", help="shorthand: 60 train birds, 8 epochs")
    return p.parse_args()


def build_model(arch: str, treatment: str | None, seed: int, epochs: int, state_loss_weight: float, run_id: str):
    """
    Build one model. `treatment` is None for the baseline, else "state_head" (Rung 2) or "moe" (Rung 3).

    The two treatments are mutually exclusive by construction in the model classes, so they are selected here rather
    than passed as independent flags.
    """
    torch.manual_seed(seed)  # Reseed before each model so baseline and treatment start identically.
    cfg = dict(BASE[arch])
    cfg["epochs"] = epochs
    return MODEL_CLS[arch](
        **cfg,
        use_time_features=True,
        use_season_features=True,
        use_state_head=(treatment == "state_head"),
        use_moe=(treatment == "moe"),
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
        f"arch {args.arch} | treatment {args.treatment} | device cuda={torch.cuda.is_available()} | train {len(train_frame):,} rows / "
        f"{train_frame['segment_id'].nunique()} segs | test {len(test_frame):,} rows / "
        f"{test_frame['segment_id'].nunique()} segs | epochs {epochs} | lambda {args.state_loss_weight}"
    )

    treat_suffix = "state" if args.treatment == "state_head" else "moe"
    models = {
        "baseline": build_model(args.arch, None, args.seed, epochs, args.state_loss_weight,
                                f"{args.arch}_time_season"),
        "treatment": build_model(args.arch, args.treatment, args.seed, epochs, args.state_loss_weight,
                                 f"{args.arch}_time_season_{treat_suffix}"),
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
            head_acc = head_diagnostics(model, test_frame)

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
        label = "state head" if args.treatment == "state_head" else "moe gate"
        print(f"{label} (test): accuracy {head_acc['accuracy']:.3f} | "
              f"migratory_launch recall {head_acc['migratory_launch_recall']:.3f} (n={head_acc['n']:,})")

    # ---- MoE routing: has the gate collapsed, and does expert k specialise on state k? ----
    if head_acc and "expert_usage" in head_acc:
        print("\nexpert usage (mean gate mass over all test windows):")
        for state, mass in head_acc["expert_usage"].items():
            print(f"  expert[{state:<17}] {mass:6.3f}")
        top = max(head_acc["expert_usage"].values())
        print(f"  -> max {top:.3f}; a value near 1.0 means the gate collapsed and the mixture is one shared head")

        print("\nmean gate mass per expert WITHIN each true state (diagonal = intended routing):")
        hdr = "".join(f"{s[:9]:>10}" for s in REFINED_STATES)
        print(f"{'true state':<18}{hdr}")
        for state, row in head_acc["expert_by_state"].items():
            cells = "".join(f"{row[k]:>10.3f}" for k in REFINED_STATES)
            argmax_k = max(row, key=row.get)
            flag = "" if argmax_k == state else f"  <- routed to {argmax_k}"
            print(f"{state:<18}{cells}{flag}")

    # ---- persist ----
    out = {
        "config": {
            "arch": args.arch, "treatment": args.treatment, "seed": args.seed, "split_seed": SPLIT_SEED,
            "epochs": epochs, "state_loss_weight": args.state_loss_weight,
            "n_train_birds": int(len(train_ids)), "context_length": CONTEXT_LENGTH,
        },
        "by_state": rows_out,
        "overall": {
            "persist_median_km": km(ov_p["median_m"]),
            "baseline": {k: km(v) if k.endswith("_m") else v for k, v in ov_b.items()},
            "treatment": {k: km(v) if k.endswith("_m") else v for k, v in ov_t.items()},
            "skill_median_baseline": 1 - ov_b["median_m"] / ov_p["median_m"],
            "skill_median_treatment": 1 - ov_t["median_m"] / ov_p["median_m"],
        },
        "head": head_acc,
    }
    # treatment is in the filename so Rung-2 and Rung-3 cells never overwrite each other
    tag = f"{args.arch}_{args.treatment}_s{args.seed}_lam{args.state_loss_weight}"
    path = RESULTS / f"eval_state_head_{tag}.json"
    path.write_text(json.dumps(out, indent=2, default=float))
    pd.DataFrame(rows_out).to_csv(RESULTS / f"eval_state_head_{tag}.csv", index=False)
    print(f"\nmetrics -> {path}")


if __name__ == "__main__":
    main()
