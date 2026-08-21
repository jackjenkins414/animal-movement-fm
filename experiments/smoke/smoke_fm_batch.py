# GenAI reference: This smoke script is GenAI generated from my specification; the registered window counts, draw weights and grid 
# are mine, and the script re derives them. I validated the output.

from __future__ import annotations

import os

os.environ["OMP_NUM_THREADS"] = "1"

import json
from pathlib import Path

import numpy as np
import torch

from animal_movement_fm.data.multispecies import SourcePool, SpeciesWindows
from animal_movement_fm.data.resample import CONTEXT_LENGTH
from animal_movement_fm.models.fm import MovementFM
from animal_movement_fm.train.loss import normalised_huber

REPO = Path(__file__).resolve().parents[2]
PROCESSED = REPO / "data" / "processed"
SPLITS = REPO / "splits"
WEIGHTS = REPO / "results" / "source_pool_weights.json"

# Segments stem, split manifest stem, registered training partition windows (A9)
SOURCES = (
    ("ciconia_ciconia", "stork", 806595),
    ("cervus_elaphus", "cervus_elaphus", 556579),
    ("mirounga_leonina", "mirounga_leonina", 53076),
)

# The registered configuration grid, for the parameter counts the tie break reads.
GRID = [
    {"d_model": d, "num_layers": n, "lr": lr}
    for d in (128, 256)
    for n in (2, 4)
    for lr in (1e-4, 3e-4, 1e-3)
]


def main() -> None:
    print(f"context length C = {CONTEXT_LENGTH} (A6)\n")

    species, failures = {}, []
    for stem, split_stem, expected in SOURCES:
        manifest = json.loads((SPLITS / f"{split_stem}_split_seed42.json").read_text())
        windows = SpeciesWindows.build(
            PROCESSED / f"{stem}_segments.parquet",
            manifest["train"],
            context_length=CONTEXT_LENGTH,
            name=stem,
        )
        species[stem] = windows
        status = "OK " if windows.n_windows == expected else "MISMATCH"
        if windows.n_windows != expected:
            failures.append(f"{stem}: {windows.n_windows} windows, registered {expected}")
        print(f"  {stem:24s} {windows.n_windows:>8,} windows  (registered {expected:>8,})  {status}")

    if failures:
        raise SystemExit("FAIL: ingest disagrees with the registered counts\n  " + "\n  ".join(failures))

    pool = SourcePool.from_registered_weights(species, WEIGHTS)
    print("\nregistered draw probabilities:")
    for name, p in zip(pool.names, pool.probabilities):
        print(f"  {name:24s} {p * 100:7.3f}%")

    rng = np.random.default_rng(42)
    realised = pool.realised_species_shares(rng, draws=200_000)
    print("\nrealised over 200,000 draws:")
    for name, share in realised.items():
        print(f"  {name:24s} {share * 100:7.3f}%")

    features, target, scale = pool.sample(rng, batch_size=256)
    print(f"\nbatch: features {features.shape}, target {target.shape}, scale {scale.shape}")
    print(f"  window scale metres: p1 {np.percentile(scale, 1):,.0f}  "
          f"p50 {np.percentile(scale, 50):,.0f}  p99 {np.percentile(scale, 99):,.0f}")
    print(f"  normalised target magnitude: median {np.median(np.linalg.norm(target, axis=1)):.3f}  "
          f"p99 {np.percentile(np.linalg.norm(target, axis=1), 99):.1f}")
    if not np.isfinite(features).all():
        raise SystemExit("FAIL: non-finite features in the batch")

    print("\nregistered configuration grid (parameter counts feed tie break 3):")
    for i, config in enumerate(GRID):
        model = MovementFM(d_model=config["d_model"], num_layers=config["num_layers"])
        print(f"  [{i:2d}] d_model={config['d_model']:>3} layers={config['num_layers']} "
              f"lr={config['lr']:<6} params={model.n_parameters():>9,}")

    model = MovementFM(d_model=128, num_layers=2)
    x = torch.tensor(features, dtype=torch.float32)
    y = torch.tensor(target, dtype=torch.float32)
    loss = normalised_huber(model(x), y)
    loss.backward()
    no_grad = [n for n, p in model.named_parameters() if p.grad is None or torch.all(p.grad == 0)]
    print(f"\nforward/backward on d_model=128, layers=2: loss {loss.item():.4f}")
    if no_grad:
        raise SystemExit(f"FAIL: parameters received no gradient: {no_grad}")
    print(f"  every one of {len(list(model.parameters()))} parameter tensors received gradient")

    baseline = MovementFM(d_model=128, num_layers=2, use_fm_package=False)
    print(f"\nZ params {model.n_parameters():,}  |  B params {baseline.n_parameters():,}  "
          f"(difference is the conditioning path)")
    print("\nsmoke OK; nothing trained, nothing sent to Gadi")


if __name__ == "__main__":
    main()
