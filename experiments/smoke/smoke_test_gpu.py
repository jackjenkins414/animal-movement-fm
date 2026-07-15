# GenAI reference: Note I used GenAI to help me write this script. 
"""
GPU smoke test for animal-movement-fm on NCI Gadi.

Proves the whole pipeline works end-to-end on a GPU before committing real SU to a full training run. It checks, in order:

  1. torch can see CUDA and a GPU is selected.
  2. A model trains for a couple of epochs on a small data subset (actually uses the GPU).
  3. The trained model checkpoints to disk (`save`).
  4. The checkpoint reloads onto the GPU (`from_checkpoint(device="cuda")`); the path that exercised the scaler stats device bug we fixed just before
  5. The reloaded model reproduces the original model's prediction (round-trip fidelity).

Any failure raises SystemExit with a FAIL message, so a non-zero exit = something's wrong.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from animal_movement_fm.eval.harness import PredictionContext, iter_windows
from animal_movement_fm.models.transformer import Transformer

logging.basicConfig(level=logging.INFO, format="%(message)s")

DATA = Path("data/processed/stork_segments.parquet")
N_INDIVIDUALS = 25        # small subset -> fast smoke test, not a real run
CONTEXT_LENGTH = 24
EPOCHS = 2


def main() -> None:
    # 1. Is the GPU actually there and visible to torch?
    print("torch version :", torch.__version__)
    print("cuda build    :", torch.version.cuda)
    print("cuda available:", torch.cuda.is_available())
    if not torch.cuda.is_available():
        raise SystemExit(
            "FAIL: torch.cuda.is_available() is False -- not on a GPU node, "
            "or the wrong torch build is installed."
        )
    print("gpu           :", torch.cuda.get_device_name(0))

    # 2. Load a small subset of the staged data (a handful of individuals).
    if not DATA.exists():
        raise SystemExit(f"FAIL: data not found at {DATA.resolve()}")
    df = pd.read_parquet(DATA)
    keep = df["individual_id"].drop_duplicates().head(N_INDIVIDUALS)
    sub = (
        df[df["individual_id"].isin(keep)]
        .sort_values(["individual_id", "timestamp"])
        .reset_index(drop=True)
    )
    print(f"subset        : {len(sub):,} rows / {sub['individual_id'].nunique()} individuals")

    # 3. Train briefly. This is what puts work on the GPU.
    model = Transformer(epochs=EPOCHS, batch_size=256, context_length=CONTEXT_LENGTH)
    model.fit(sub)
    print("trained device:", model.device)
    if model.device != "cuda":
        raise SystemExit(f"FAIL: model trained on {model.device!r}, expected 'cuda'.")

    # Build one prediction context from the first trajectory long enough to form a window.
    history = ts = None
    for ind in keep:
        traj = sub[sub["individual_id"] == ind]
        windows = iter_windows(traj, CONTEXT_LENGTH)
        if windows:
            history, ts = windows[0][0], windows[0][2]
            break
    if history is None:
        raise SystemExit("FAIL: no trajectory long enough to form a window.")
    ctx = PredictionContext(history=history, timestamp=pd.Timestamp(ts))
    pred_before = model.predict(ctx)

    # 4. Checkpoint round-trip, reloading onto the GPU.
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = Path(tmp) / "smoke.pt"
        model.save(ckpt)
        reloaded = Transformer.from_checkpoint(ckpt, device="cuda")
        pred_after = reloaded.predict(ctx)

    # 5. The reloaded model must reproduce the original prediction exactly.
    print("pred before save:", pred_before)
    print("pred after  load:", pred_after)
    if not np.allclose(pred_before, pred_after, atol=1e-6):
        raise SystemExit("FAIL: prediction changed across the checkpoint round-trip.")

    print("\nSMOKE TEST PASSED: GPU training + checkpoint round-trip OK.")


if __name__ == "__main__":
    main()
