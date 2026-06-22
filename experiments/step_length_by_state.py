# GenAI reference: The majority of this script was generated with GenAI assistance. I verified the output
# by inspecting the code logic and and checked that persistence error equals step length, and confirming that 
# the printed mean/median values match the vertical lines in the plot. 
"""Why persistence wins the median but loses the mean on switching-CRW.

Persistence predicts zero displacement, so its per-step error IS the true step
length. Switching-CRW mixes short foraging steps with long directed transit
steps, so that error distribution is bimodal: the MEDIAN sits in the foraging
mode (small -> persistence looks unbeatable), while the MEAN is dragged right by
the transit tail (large -> persistence looks bad, and the LSTM claws it back).
This is the mechanism behind the median/mean split in the bar charts.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from animal_movement_fm.data.synthetic import generate_switching_crw

N_INDIVIDUALS = 100
N_STEPS = 500
SEED = 0
M_PER_DEG = 111_320.0  # equator approx; synthetic data starts at lat 0, tiny steps

OUT = Path("results/plots")
OUT.mkdir(parents=True, exist_ok=True)

STATE_NAMES = {0: "foraging", 1: "transit"}
STATE_COLOURS = {0: "C0", 1: "C3"}


def main():
    df = generate_switching_crw(
        n_individuals=N_INDIVIDUALS, n_steps=N_STEPS, seed=SEED
    ).sort_values(["individual_id", "timestamp"])

    # The step landing on row t is governed by that row's state. diff() within
    # each track leaves the first row NaN (no preceding position) -> dropped.
    g = df.groupby("individual_id", sort=False)
    step_m = np.hypot(g["lon"].diff(), g["lat"].diff()) * M_PER_DEG
    step = pd.DataFrame({"step_m": step_m, "state": df["state"]}).dropna()

    overall_median = step["step_m"].median()
    overall_mean = step["step_m"].mean()
    print(f"step length (m):  median={overall_median:.1f}  mean={overall_mean:.1f}")

    for s in (0, 1):
        sub = step.loc[step["state"] == s, "step_m"]
        print(
            f"  {STATE_NAMES[s]:9s} n={len(sub):6d}  "
            f"median={sub.median():.1f}  mean={sub.mean():.1f}"
        )

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.logspace(
        np.log10(max(step["step_m"].min(), 1.0)),
        np.log10(step["step_m"].max()),
        60,
    )

    for s in (0, 1):
        sub = step.loc[step["state"] == s, "step_m"]
        ax.hist(
            sub,
            bins=bins,
            alpha=0.6,
            color=STATE_COLOURS[s],
            label=STATE_NAMES[s],
        )

    ax.axvline(
        overall_median,
        color="k",
        ls="-",
        lw=1.5,
        label=f"overall median ({overall_median:.0f} m)",
    )
    ax.axvline(
        overall_mean,
        color="k",
        ls="--",
        lw=1.5,
        label=f"overall mean ({overall_mean:.0f} m)",
    )

    ax.set_xscale("log")
    ax.set_xlabel("step length = persistence per-step error (m, log scale)")
    ax.set_ylabel("number of steps")
    ax.set_title(
        "Persistence error is bimodal: median sits in foraging, mean pulled by transit"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "step_length_by_state.png", dpi=150)

    print(f"\nsaved {OUT / 'step_length_by_state.png'}")


if __name__ == "__main__":
    main()