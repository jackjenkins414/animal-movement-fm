# GenAI reference: The majority of this script was generated with GenAI assistance. I verified the output
# by inspecting the code logic and checking the generated csv/plot values against printed summary statistics
# and also confirming the results match the expected behaviour of the synthetic data regimes. 
"""Per-track LSTM advantage over persistence vs the randomness index.

Headline figure for the randomness analysis: does the LSTM's advantage over
persistence grow as a track gets more directed (higher rho)? Pools all three
synthetic regimes so rho sweeps its full range (BM ~0 -> switching ~0.6 ->
CRW ~0.86), and plots the *mean* fractional advantage per track:

    advantage = (persistence_mean_err - lstm_mean_err) / persistence_mean_err

> 0 means the LSTM beats persistence on that track. Mean (not median) is used
because the LSTM's win comes from the directed steps the mean captures; the
median story is the separate per-state step-length plot.

Only persistence + LSTM are fit, so xgboost is never imported and the
torch/xgboost OpenMP segfault cannot occur -- no OMP guard needed.

Run: uv run python experiments/error_vs_randomness.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from animal_movement_fm.data.synthetic import (
    generate_brownian_motion,
    generate_crw,
    generate_switching_crw,
)
from animal_movement_fm.eval.harness import evaluate
from animal_movement_fm.models.persistence import Persistence
from animal_movement_fm.models.lstm import LSTM
from animal_movement_fm.analysis.randomness import track_randomness

N_INDIVIDUALS = 100   # ~20 test tracks per dataset under a 20% split
N_STEPS = 500
TEST_FRAC = 0.2
CONTEXT_LENGTH = 24
SEED = 0

OUT = Path("results")
PLOTS = OUT / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)

DATASET_COLOURS = {  # match the validated rho histogram
    "brownian_motion": "C0",
    "crw": "C1",
    "switching_crw": "C2",
}


def split_by_individual(df, test_frac, seed):
    rng = np.random.default_rng(seed)
    individuals = df["individual_id"].unique()
    rng.shuffle(individuals)
    n_test = max(1, int(round(len(individuals) * test_frac)))
    test_ids = set(individuals[:n_test])
    return (
        df[~df["individual_id"].isin(test_ids)].copy(),
        df[df["individual_id"].isin(test_ids)].copy(),
    )


def per_track_mean_error(model, df_test, *, needs_fit=False, df_train=None):
    """One mean haversine error per individual track."""
    if needs_fit:
        model.fit(df_train)
    results = evaluate(
        model, df_test, context_length=CONTEXT_LENGTH, show_progress=False
    )
    return results.groupby("individual_id")["error_m"].mean()


def main():
    torch.manual_seed(SEED)
    datasets = {
        "brownian_motion": generate_brownian_motion(
            n_individuals=N_INDIVIDUALS, n_steps=N_STEPS, seed=SEED
        ),
        "crw": generate_crw(
            n_individuals=N_INDIVIDUALS, n_steps=N_STEPS, seed=SEED
        ),
        "switching_crw": generate_switching_crw(
            n_individuals=N_INDIVIDUALS, n_steps=N_STEPS, seed=SEED
        ),
    }

    parts = []
    for name, df in datasets.items():
        df_train, df_test = split_by_individual(df, TEST_FRAC, SEED)

        pers = per_track_mean_error(Persistence(), df_test)
        lstm = per_track_mean_error(
            LSTM(
                hidden_dim=64, num_layers=1, context_length=CONTEXT_LENGTH,
                epochs=30, batch_size=64, lr=1e-3, device="cpu",
            ),
            df_test, needs_fit=True, df_train=df_train,
        )
        rho = track_randomness(df_test).set_index("individual_id")["persistence"]

        joined = pd.DataFrame({"pers_mean": pers, "lstm_mean": lstm, "rho": rho})
        joined["advantage"] = (
            (joined["pers_mean"] - joined["lstm_mean"]) / joined["pers_mean"]
        )
        joined["dataset"] = name
        parts.append(joined.reset_index())
        print(
            f"{name:16s}  tracks={len(joined):3d}  "
            f"rho={joined['rho'].mean():.3f}  "
            f"advantage={joined['advantage'].mean():+.3f}"
        )

    combined = pd.concat(parts, ignore_index=True)
    combined.to_csv(OUT / "error_vs_randomness.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    for name, g in combined.groupby("dataset"):
        ax.scatter(g["rho"], g["advantage"], s=40, alpha=0.7,
                   color=DATASET_COLOURS[name], label=name)
    ax.axhline(0, color="k", lw=1, ls="--", alpha=0.6)  # LSTM == persistence

    # Illustrative across-regime trend. DELETE if you can't defend a global
    # linear fit over three clustered rho values to John.
    m, b = np.polyfit(combined["rho"], combined["advantage"], 1)
    xs = np.linspace(combined["rho"].min(), combined["rho"].max(), 100)
    ax.plot(xs, m * xs + b, color="grey", lw=1.2, alpha=0.5)

    ax.set_xlabel("directional persistence  (rho = mean cos turning angle)")
    ax.set_ylabel("LSTM advantage over persistence\n(mean fractional error reduction)")
    ax.set_title("LSTM beats persistence on directed tracks, ties on random ones")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS / "error_vs_randomness.png", dpi=150)
    print(f"\nsaved {PLOTS / 'error_vs_randomness.png'}")


if __name__ == "__main__":
    main()