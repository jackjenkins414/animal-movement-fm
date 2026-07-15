"""Validate the randomness index on synthetic data and plot its distribution.

We build a simple sanity check; the index should recover the regime each generator was built
with and produce per-track distribution plot
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from animal_movement_fm.data.synthetic import (
    generate_brownian_motion,
    generate_crw,
    generate_switching_crw,
)
from animal_movement_fm.analysis.randomness import track_randomness

SEED = 0
N_INDIVIDUALS = 150   
N_STEPS = 500        

OUT = Path("results/plots")
OUT.mkdir(parents=True, exist_ok=True)


def main() -> None:
    # signatures from synthetic.py
    datasets = {
        "brownian_motion": generate_brownian_motion(
            n_individuals=N_INDIVIDUALS, n_steps=N_STEPS, seed=SEED
        ),
        "crw": generate_crw(n_individuals=N_INDIVIDUALS, n_steps=N_STEPS, seed=SEED),
        "switching_crw": generate_switching_crw(
            n_individuals=N_INDIVIDUALS, n_steps=N_STEPS, seed=SEED
        ),
    }

    summaries = {}
    for name, df in datasets.items():
        r = track_randomness(df)
        summaries[name] = r
        print(
            f"{name:16s}  rho: mean={r['persistence'].mean():.3f}  "
            f"median={r['persistence'].median():.3f}  "
            f"sd={r['persistence'].std():.3f}  n={len(r)}"
        )

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(-1, 1, 41)
    for name, r in summaries.items():
        ax.hist(r["persistence"], bins=bins, alpha=0.55, label=name)
    ax.set_xlabel("directional persistence  (rho = mean cos turning angle)")
    ax.set_ylabel("number of tracks")
    ax.set_title("Randomness index by synthetic regime")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "randomness_distribution.png", dpi=150)
    print(f"\nsaved {OUT / 'randomness_distribution.png'}")


if __name__ == "__main__":
    main()