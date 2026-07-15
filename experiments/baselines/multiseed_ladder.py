# GenAI reference: Since this a straightforward script, everything from line 38 onwards was 
# written and then verfied by GenAI (bar this comment below which it added too.)
# Experiment: 10-rep multi-seed benchmark of the baseline ladder.
# Question: are the rungs distinguishable once run-to-run variance is accounted for?
# Total variance: each master seed drives BOTH data generation AND torch init.
# Locked recipe: 60 epochs both neural rungs, warmup_frac=0.1 on the transformer.
# Untracked until a deliberate commit, per our experiment-script convention.
import os
os.environ["OMP_NUM_THREADS"] = "1"

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from animal_movement_fm.eval.harness import evaluate

from animal_movement_fm.models.persistence import Persistence
from animal_movement_fm.models.xgboost import XGBoost
from animal_movement_fm.models.lstm import LSTM
from animal_movement_fm.models.transformer import Transformer

from animal_movement_fm.data.synthetic import (
    generate_brownian_motion,
    generate_crw,
    generate_switching_crw,
)

SEEDS = list(range(10)) # 10 master seeds: 0..9
N_INDIVIDUALS = 20
N_STEPS = 500
CONTEXT_LENGTH = 24
TRAIN_FRAC = 0.8   # 16 train /4 test individuals

RESULTS_DIR = Path("results")
RAW_CSV = RESULTS_DIR / "multiseed_ladder_raw.csv"
SUMMARY_CSV = RESULTS_DIR / "multiseed_ladder_summary.csv"


def make_datasets(seed: int) -> dict[str, pd.DataFrame]:
    """Three synthetic datasets generated under one master seed."""
    return {
        "brownian_motion": generate_brownian_motion(
            n_individuals=N_INDIVIDUALS, n_steps=N_STEPS, seed=seed),
        "crw": generate_crw(
            n_individuals=N_INDIVIDUALS, n_steps=N_STEPS, seed=seed),
        "switching_crw": generate_switching_crw(
            n_individuals=N_INDIVIDUALS, n_steps=N_STEPS, seed=seed),
    }


def split_by_individual(df, train_frac=TRAIN_FRAC):
    ids = df["individual_id"].unique()
    n_train = int(round(len(ids) * train_frac))
    train_ids, test_ids = ids[:n_train], ids[n_train:]
    return (df[df["individual_id"].isin(train_ids)].copy(),
            df[df["individual_id"].isin(test_ids)].copy())


def fresh_models():
    """Locked recipe: 60 epochs both neural rungs, warmup_frac=0.1 on the transformer."""
    return [
        Persistence(),
        XGBoost(),
        LSTM(hidden_dim=64, num_layers=1, context_length=CONTEXT_LENGTH,
             epochs=60, batch_size=64, lr=1e-3, device="cpu"),
        Transformer(d_model=64, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.1,
                    context_length=CONTEXT_LENGTH, epochs=60, batch_size=64, lr=1e-3,
                    warmup_frac=0.1, device="cpu"),
    ]


def summarise(errors_m):
    return {
        "median_m": float(np.median(errors_m)),
        "mean_m": float(np.mean(errors_m)),
        "p99_m": float(np.percentile(errors_m, 99)),
    }


def run_one_seed(seed: int) -> list[dict]:
    """Full ladder on all three datasets under one master seed."""
    rows = []
    for ds_name, df in make_datasets(seed).items():
        train_df, test_df = split_by_individual(df)
        for model in fresh_models():
            torch.manual_seed(seed)  # drives neural init/shuffle; harmless for param-free models
            if hasattr(model, "fit"):
                model.fit(train_df)
            results = evaluate(model, test_df, context_length=CONTEXT_LENGTH, show_progress=False)
            stats = summarise(results["error_m"].to_numpy())
            stats.update(seed=seed, dataset=ds_name,
                         model=getattr(model, "name", type(model).__name__))
            rows.append(stats)
    return rows


def aggregate(raw: pd.DataFrame) -> pd.DataFrame:
    """Median + min–max across seeds, per (dataset, model)."""
    return raw.groupby(["dataset", "model"]).agg(
        median_m_med=("median_m", "median"),
        median_m_min=("median_m", "min"),
        median_m_max=("median_m", "max"),
        mean_m_med=("mean_m", "median"),
        p99_m_med=("p99_m", "median"),
        p99_m_min=("p99_m", "min"),
        p99_m_max=("p99_m", "max"),
        n_reps=("seed", "count"),
    ).reset_index()


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if RAW_CSV.exists():
        RAW_CSV.unlink()  # fresh start; we append each seed's block below

    print(f"Running {len(SEEDS)} seeds x 3 datasets x 4 models. "
          f"This is a long run (~1-2 h on CPU); partial results are saved after each seed.",
          flush=True)

    header_written = False
    for i, seed in enumerate(SEEDS, 1):
        print(f"\n===== seed {seed} ({i}/{len(SEEDS)}) =====", flush=True)
        block = pd.DataFrame(run_one_seed(seed))[
            ["seed", "dataset", "model", "median_m", "mean_m", "p99_m"]]
        block.to_csv(RAW_CSV, mode="a", header=not header_written, index=False)
        header_written = True

        sc = block[block.dataset == "switching_crw"].set_index("model")
        if {"lstm_h64_l1", "transformer_d64_l2"}.issubset(sc.index):
            print(f"  switching-CRW p99 — LSTM {sc.loc['lstm_h64_l1','p99_m']:,.0f}  "
                  f"Transformer {sc.loc['transformer_d64_l2','p99_m']:,.0f}", flush=True)

    # aggregate from the raw CSV so even a partial/resumed file summarises fine
    summary = aggregate(pd.read_csv(RAW_CSV))
    summary.to_csv(SUMMARY_CSV, index=False)

    for ds in ["brownian_motion", "crw", "switching_crw"]:
        sub = summary[summary.dataset == ds]
        print(f"\n=== {ds} (median across {int(sub['n_reps'].iloc[0])} seeds; [min-max]) ===")
        for _, r in sub.sort_values("p99_m_med").iterrows():
            print(f"  {r['model']:<22} "
                  f"median {r['median_m_med']:>7,.0f} [{r['median_m_min']:,.0f}-{r['median_m_max']:,.0f}]   "
                  f"mean {r['mean_m_med']:>7,.0f}   "
                  f"p99 {r['p99_m_med']:>7,.0f} [{r['p99_m_min']:,.0f}-{r['p99_m_max']:,.0f}]")

    sc = summary[summary.dataset == "switching_crw"].set_index("model")
    if {"lstm_h64_l1", "transformer_d64_l2"}.issubset(sc.index):
        l, t = sc.loc["lstm_h64_l1"], sc.loc["transformer_d64_l2"]
        overlap = (l["p99_m_min"] <= t["p99_m_max"]) and (t["p99_m_min"] <= l["p99_m_max"])
        print(f"\nHEADLINE — switching-CRW p99 across {int(l['n_reps'])} seeds:")
        print(f"  LSTM        median {l['p99_m_med']:,.0f}  [{l['p99_m_min']:,.0f}, {l['p99_m_max']:,.0f}]")
        print(f"  Transformer median {t['p99_m_med']:,.0f}  [{t['p99_m_min']:,.0f}, {t['p99_m_max']:,.0f}]")
        print(f"  Ranges {'OVERLAP -> indistinguishable' if overlap else 'do NOT overlap -> a real gap'}")

    print(f"\nRaw:     {RAW_CSV}\nSummary: {SUMMARY_CSV}")


if __name__ == "__main__":
    main()