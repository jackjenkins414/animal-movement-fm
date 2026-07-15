"""
Transformer convergence figure: iter-1 (lr=3e-4, kept) vs iter-2 (lr=1e-3, worse).

Per-epoch train_mse for the two convergence runs (from the Gadi logs
train_transformer.o173521958 and .o173559409), embedded here so the figure is reproducible on the Mac
without the logs. Saves to experiments/results/transformer_convergence_curves.png.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results" / "transformer_convergence_curves.png"

# iter-1: lr=3e-4, warmup 0.1, grad_clip 1.0 (the converged Transformer we carry forward)
ITER1 = [
    0.0482305, 0.0320241, 0.0308668, 0.0303539, 0.0301228, 0.0298506, 0.0296012, 0.0295002,
    0.0294065, 0.0291861, 0.0290717, 0.0289042, 0.0288014, 0.0285639, 0.02809, 0.027104,
    0.0259669, 0.0252444, 0.024994, 0.024634, 0.0244537, 0.0243343, 0.0241357, 0.0239409,
    0.0238907, 0.0237187, 0.0237456, 0.0236956, 0.0234404, 0.0235214, 0.023352, 0.0232625,
    0.0232922, 0.023181, 0.0231311, 0.0230804, 0.0229936, 0.0229702, 0.0229404, 0.022844,
    0.0227798, 0.0227811, 0.0226422, 0.0227016, 0.0225398, 0.0224802, 0.0224356, 0.0223681,
    0.0223788, 0.0222872, 0.0222515, 0.0221629, 0.0221314, 0.0221388, 0.0220908, 0.0220597,
    0.0219395, 0.0219397, 0.0218434, 0.0217981, 0.0217938, 0.0217797, 0.0217451, 0.0216927,
    0.0216767, 0.0216549, 0.0216413, 0.0216227, 0.0215532, 0.0215567, 0.0215205, 0.021482,
    0.0214862, 0.0214136, 0.0214411, 0.0214109, 0.0214287, 0.0213912, 0.021428, 0.0213903,
]

# iter-2: lr=1e-3, everything else identical (converged worse -> not carried)
ITER2 = [
    0.0509928, 0.0310407, 0.0303859, 0.0300562, 0.029936, 0.0299944, 0.0299186, 0.0299419,
    0.0300552, 0.0299626, 0.0298188, 0.0297267, 0.0295426, 0.0294336, 0.0294656, 0.0293201,
    0.0292726, 0.0291541, 0.0290377, 0.0289955, 0.0289069, 0.0288479, 0.0288594, 0.0286855,
    0.0287032, 0.0285619, 0.0285486, 0.028525, 0.0285093, 0.0284567, 0.0284274, 0.0283739,
    0.0283215, 0.0282704, 0.0281674, 0.0282292, 0.0281315, 0.0280325, 0.0280042, 0.0279658,
    0.0278241, 0.0277665, 0.027719, 0.0276374, 0.0274811, 0.0274271, 0.0273838, 0.0272734,
    0.0271376, 0.02691, 0.0267569, 0.0265565, 0.0262299, 0.0260269, 0.0259523, 0.0255707,
    0.0250648, 0.025052, 0.0249696, 0.0246843, 0.0244788, 0.0238606, 0.0239405, 0.0237873,
    0.0235183, 0.0234672, 0.0232767, 0.0232601, 0.023145, 0.0230703, 0.0229963, 0.0229371,
    0.0230173, 0.022863, 0.0228147, 0.0227797, 0.0227946, 0.0227336, 0.0227191, 0.0226901,
]

LSTM_REF = 0.0179  # converged matched-LSTM train_mse, for reference


def main() -> None:
    epochs = range(1, 81)
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(epochs, ITER1, color="#2f6f3e", lw=2, label=f"iter-1  lr=3e-4  (kept)  → {ITER1[-1]:.4f}")
    ax.plot(epochs, ITER2, color="#c0472b", lw=2, label=f"iter-2  lr=1e-3  (worse) → {ITER2[-1]:.4f}")
    ax.axhline(LSTM_REF, color="#666666", ls="--", lw=1.2, label=f"matched-LSTM ref = {LSTM_REF:.4f}")

    ax.set_xlabel("epoch")
    ax.set_ylabel("train MSE (displacement, deg$^2$)")
    ax.set_title("Transformer convergence at 6h: peak learning-rate experiment")
    ax.set_xlim(1, 80)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=140)
    plt.close(fig)
    print(f"saved -> {OUT}")
    print(f"iter-1 final {ITER1[-1]:.4f} | iter-2 final {ITER2[-1]:.4f} | epochs {len(ITER1)}/{len(ITER2)}")


if __name__ == "__main__":
    main()
