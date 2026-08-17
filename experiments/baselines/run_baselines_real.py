# GenAI reference: I implemented the baseline structure and got GenAI to add concise progress/output print statements 
# and minor cleanup around result display.
"""Run the training-free baselines on real held out stork test data."""

from pathlib import Path

from animal_movement_fm.data.benchmark import run_ladder
from animal_movement_fm.data.split import make_split, window_counts_per_individual
from animal_movement_fm.models.persistence import Persistence
from animal_movement_fm.models.averaging import DisplacementAveraging

SEGMENTS = Path.home() / "animal-movement-fm" / "data" / "processed" / "stork_segments.parquet"

counts = window_counts_per_individual(SEGMENTS, context_length=24)  # Tier 1, pre A6
split = make_split(counts, strata=None, min_windows=100, seed=42)

predictors = [
    Persistence(),
    DisplacementAveraging(k=1),
    DisplacementAveraging(k=3),
    DisplacementAveraging(k=5),
    DisplacementAveraging(k=10),
]

print(f"running {len(predictors)} baselines on {len(split.test)} held-out test storks\n")
table = run_ladder(predictors, SEGMENTS, split.test, context_length=24)  # Tier 1, pre A6
print("\n" + "=" * 70)
print(table.to_string(index=False))