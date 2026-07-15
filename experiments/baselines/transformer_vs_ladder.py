# GenAI reference: Since this a straightforward script, everything from line 40 onwards was 
# written and then verfied by GenAI
# Experiment: does attention beat recurrence on the synthetic ladder?
# Headline question: the switching-CRW p99 tail, where LSTM and XGBoost currently tie.
# Untracked until a deliberate commit, per our experiment-script convention.

# OpenMP cap MUST be set before numpy/torch/xgboost import (Apple Silicon segfault fix);
# torch + xgboost ship separate OpenMP runtimes and collide otherwise.
import os
os.environ["OMP_NUM_THREADS"] = "1"

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

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

SEED = 0
N_INDIVIDUALS = 20
N_STEPS = 500
CONTEXT_LENGTH = 24
TRAIN_FRAC = 0.8   # 16 train / 4 test individuals to match  rest of the ladder
SAVE_PLOTS = True
PLOT_DIR = Path("results/plots")


def make_datasets() -> dict[str, pd.DataFrame]:
    """Build the three synthetic datasets. Edit the calls here to match synthetic.py."""
    return {
        "brownian_motion": generate_brownian_motion(
            n_individuals=N_INDIVIDUALS, n_steps=N_STEPS, seed=SEED),
        "crw": generate_crw(
            n_individuals=N_INDIVIDUALS, n_steps=N_STEPS, seed=SEED),
        "switching_crw": generate_switching_crw(
            n_individuals=N_INDIVIDUALS, n_steps=N_STEPS, seed=SEED),
    }


def split_by_individual(df: pd.DataFrame, train_frac: float = TRAIN_FRAC):
    """Held-out-individuals split: first 80% of ids train, rest test (no within-track leakage)."""
    ids = df["individual_id"].unique()
    n_train = int(round(len(ids) * train_frac))
    train_ids, test_ids = ids[:n_train], ids[n_train:]
    train_df = df[df["individual_id"].isin(train_ids)].copy()
    test_df = df[df["individual_id"].isin(test_ids)].copy()
    return train_df, test_df


def fresh_models() -> list:
    """One fresh model set per dataset (trainable models must not carry weights across datasets)."""
    return [
        Persistence(),
        XGBoost(),
        LSTM(hidden_dim=64, num_layers=1, context_length=CONTEXT_LENGTH,
             epochs=60, batch_size=64, lr=1e-3, device="cpu"),
        Transformer(d_model=64, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.1,
                    context_length=CONTEXT_LENGTH, epochs=60, batch_size=64, lr=1e-3,
                    warmup_frac=0.1, device="cpu"),
    ]


def summarise(errors_m: np.ndarray) -> dict:
    """Median / mean / p99 in metres — computed directly from error_m so we don't depend
    on summarise_errors' internal column names."""
    return {
        "median_m": float(np.median(errors_m)),
        "mean_m": float(np.mean(errors_m)),
        "p99_m": float(np.percentile(errors_m, 99)),
    }


def run() -> pd.DataFrame:
    datasets = make_datasets()
    rows = []
    per_pred_errors: dict[str, dict[str, np.ndarray]] = {}  # dataset -> model -> error_m (for CDF)

    for ds_name, df in datasets.items():
        train_df, test_df = split_by_individual(df)
        per_pred_errors[ds_name] = {}

        for model in fresh_models():
            torch.manual_seed(SEED)  # deterministic init/shuffle for the neural rungs

            # Trainable rungs fit on train; parameter-free rungs (persistence) skip.
            if hasattr(model, "fit"):
                model.fit(train_df)

            results = evaluate(model, test_df, context_length=CONTEXT_LENGTH, show_progress=False)
            errs = results["error_m"].to_numpy()

            stats = summarise(errs)
            stats.update(dataset=ds_name, model=getattr(model, "name", type(model).__name__))
            rows.append(stats)
            per_pred_errors[ds_name][stats["model"]] = errs

    table = pd.DataFrame(rows)[["dataset", "model", "median_m", "mean_m", "p99_m"]]

    # ---- printed report ----
    for ds_name in datasets:
        sub = table[table.dataset == ds_name].sort_values("median_m")
        print(f"\n=== {ds_name} ===")
        print(sub.to_string(index=False, float_format=lambda x: f"{x:,.1f}"))

    # ---- the headline: transformer vs LSTM on the switching-CRW tail ----
    sc = table[table.dataset == "switching_crw"].set_index("model")
    if {"lstm_h64_l1", "transformer_d64_l2"}.issubset(sc.index):
        lstm_p99 = sc.loc["lstm_h64_l1", "p99_m"]
        tfm_p99 = sc.loc["transformer_d64_l2", "p99_m"]
        delta = (lstm_p99 - tfm_p99) / lstm_p99 * 100
        print(f"\nHEADLINE — switching-CRW p99: "
              f"LSTM {lstm_p99:,.0f} m  vs  Transformer {tfm_p99:,.0f} m  "
              f"({delta:+.1f}% vs LSTM)")

    # ---- CDF on switching-CRW (the diagnostic plot) ----
    if SAVE_PLOTS:
        PLOT_DIR.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(7, 5))
        for model_name, errs in per_pred_errors["switching_crw"].items():
            x = np.sort(errs)
            y = np.arange(1, len(x) + 1) / len(x)
            plt.plot(x, y, label=model_name, linewidth=1.6)
        plt.xlabel("haversine error (m)")
        plt.ylabel("cumulative probability")
        plt.title("Switching-CRW error CDF — does attention compress the tail?")
        plt.xlim(0, np.percentile(
            np.concatenate(list(per_pred_errors["switching_crw"].values())), 99.5))
        plt.legend()
        plt.grid(alpha=0.3)
        out = PLOT_DIR / "cdf_switching_crw_transformer.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"\nSaved CDF to {out}")

    return table


if __name__ == "__main__":
    table = run()
    table.to_csv("results/transformer_ladder.csv", index=False)