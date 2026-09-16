# GenAI reference: used GenAi to implemtat this file; windows as offsets into flat arrays over maximal one bin continuous 
# runs under our A2 pol, ids cannocalised, context only encoding to ensure taget ccant leak into scale, and species draws
# read from registered weights. I check over the code; and added docstrings. 
"""
Tier 2 ingest utilities for building per species window indices and sampling our registered multi-species source pool. 
Windows follow our A2 continuity rule, and the source species draws use the fixed probabilities stored in 
registered/tier2_source_pool_weights.json (moved from results/ on 16-09-2026, A16 follow-up).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from animal_movement_fm.data.enu import SCALE_FLOOR_M, encode_window, enu_steps, scale_target
from animal_movement_fm.data.resample import CONTEXT_LENGTH
from animal_movement_fm.data.split import id_key


def _contiguous_run_bounds(segment_id: np.ndarray, bin_index: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Return start/stop offsets for maximal contiguous one bin runs.
    """
    n = len(bin_index)
    if n == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    
    new_run = np.ones(n, dtype=bool)
    same_segment = segment_id[1:] == segment_id[:-1]
    new_run[1:] = ~(same_segment & (np.diff(bin_index) == 1))
    starts = np.flatnonzero(new_run)
    stops = np.append(starts[1:], n)
    return starts, stops


@dataclass
class SpeciesWindows:
    """
    One species' valid windows; stored as offsets into flat position arrays rather than materialised arrays.
    """

    name: str
    lon: np.ndarray
    lat: np.ndarray
    epoch_s: np.ndarray
    individual_index: np.ndarray
    starts: np.ndarray
    context_length: int

    @property
    def n_windows(self) -> int:
        return int(len(self.starts))

    @classmethod
    def build(
        cls,
        segments_parquet: str | Path,
        individual_ids,
        *,
        context_length: int = CONTEXT_LENGTH,
        name: str | None = None,
    ) -> "SpeciesWindows":
        """Build the valid window index for selected individuals of one species."""
        import duckdb

        con = duckdb.connect()
        present = con.execute(
            f"SELECT DISTINCT individual_id FROM '{segments_parquet}'"
        ).df()["individual_id"]
        wanted = {id_key(i) for i in individual_ids}
        selected = present[[id_key(value) in wanted for value in present]]
        if wanted and selected.empty:
            raise ValueError(
                f"none of the {len(wanted)} requested individuals appear in {segments_parquet}; "
                "this is a manifest/segments mismatch, not an empty split"
            )

        # register ids in the parquet columns own dtype then push the partition filter into DuckDB; coordinates belonging to
        # validation or test individuals therefore never enter a source-training frame.
        con.register("split_ids", pd.DataFrame({"individual_id": selected.to_numpy()}))
        # Get epoch seconds in DuckDB to avoid datetime unit mismatches in pandas.
        frame = con.execute(
            f"""
            SELECT s.segment_id, s.individual_id, s.bin_index, epoch(s.timestamp) AS epoch_s, s.lon, s.lat
            FROM '{segments_parquet}' s
            SEMI JOIN split_ids ids ON s.individual_id = ids.individual_id
            ORDER BY s.segment_id, s.bin_index
            """
        ).df()

        segment_id = frame["segment_id"].to_numpy()
        bin_index = frame["bin_index"].to_numpy()
        starts, stops = _contiguous_run_bounds(segment_id, bin_index)

        span = context_length + 1
        offsets = [
            np.arange(start, stop - span + 1, dtype=np.int64)
            for start, stop in zip(starts, stops)
            if stop - start >= span
        ]
        window_starts = (
            np.concatenate(offsets) if offsets else np.empty(0, dtype=np.int64)
        )

        individuals, individual_index = np.unique(
            np.array([id_key(v) for v in frame["individual_id"]]), return_inverse=True
        )
        return cls(
            name=name or Path(segments_parquet).stem,
            lon=frame["lon"].to_numpy(float),
            lat=frame["lat"].to_numpy(float),
            epoch_s=frame["epoch_s"].to_numpy(float),
            individual_index=individual_index[window_starts + context_length]
            if len(window_starts)
            else np.empty(0, dtype=np.int64),
            starts=window_starts,
            context_length=context_length,
        )

    def gather(
        self, window_ids: np.ndarray, *, sigma: float = SCALE_FLOOR_M
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Materialise a batch as (features, target_scaled, scale).

        `sigma` is the registered scale floor and defaults to it, so registered arms are unaffected.
        A Tier E variant may pass a different value; whatever is passed here must also be passed at
        inference, or the model is scored on a representation it was not trained on.
        """
        window_ids = np.asarray(window_ids, dtype=np.int64)
        base = self.starts[window_ids][:, None]
        idx = base + np.arange(self.context_length + 1, dtype=np.int64)[None, :]
        lon, lat = self.lon[idx], self.lat[idx]

        # Use real representative-fix times rather than assuming exact 6h spacing.
        epoch_s = self.epoch_s[idx]

        # Encode only the context so the target step cannot affect the window scale.
        c = self.context_length
        features, scale = encode_window(
            lon[:, :c], lat[:, :c], timestamp_s=epoch_s[:, :c], sigma=sigma
        )
        target_step = enu_steps(lon[:, self.context_length - 1 :], lat[:, self.context_length - 1 :])[:, 0, :]
        return features, scale_target(target_step, scale), scale


class SourcePool:
    """
    Registered source-species pool and draw probabilities.
    """

    def __init__(self, species: dict[str, SpeciesWindows], weights: dict[str, float]) -> None:
        missing = set(weights) - set(species)
        if missing:
            raise ValueError(f"weights reference species with no windows: {sorted(missing)}")
        self.names = sorted(weights)
        self.species = species
        total = sum(weights[n] for n in self.names)
        self.probabilities = np.array([weights[n] / total for n in self.names], dtype=float)

    @classmethod
    def from_registered_weights(
        cls,
        species: dict[str, SpeciesWindows],
        weights_json: str | Path,
        pool: str = "source_pool",
    ) -> "SourcePool":
        """Load the full source pool or a registered leave-one-out sub-pool from JSON."""
        payload = json.loads(Path(weights_json).read_text())
        node = payload
        for part in pool.split("."):
            node = node[part]
        return cls({k: species[k] for k in node}, dict(node))

    def sample(self, rng: np.random.Generator, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Draw one mixed-species batch under the registered sampling rule.
        """
        picks = rng.choice(len(self.names), size=batch_size, p=self.probabilities)
        features, targets, scales = [], [], []
        for index, name in enumerate(self.names):
            count = int((picks == index).sum())
            if not count:
                continue
            windows = self.species[name]
            ids = rng.integers(0, windows.n_windows, size=count)
            f, t, s = windows.gather(ids)
            features.append(f)
            targets.append(t)
            scales.append(s)
        return np.concatenate(features), np.concatenate(targets), np.concatenate(scales)

    def realised_species_shares(self, rng: np.random.Generator, draws: int = 100_000) -> dict[str, float]:
        """
        Return realised source-species shares from repeated sampler draws.
        """
        picks = rng.choice(len(self.names), size=draws, p=self.probabilities)
        return {name: float((picks == i).mean()) for i, name in enumerate(self.names)}
