# GenAI reference: used GenAi to implemtat this file; windows as offsets into flat arrays over maximal one bin continuous 
# runs under our A2 pol, ids cannocalised, context only encoding to ensure taget ccant leak into scale, and species draws
# read from registered weights. I check over the code; and added docstrings. 
"""
Tier 2 ingest utilities for building per species window indices and sampling our registered multi-species source pool. 
Windows follow our A2 continuity rule, and the source species draws use the fixed probabilities stored in 
results/source_pool_weights.json.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from animal_movement_fm.data.enu import encode_window, enu_steps, scale_target
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
        # Get epoch seconds in DuckDB to avoid datetime unit mismatches in pandas.
        frame = con.execute(
            f"""
            SELECT segment_id, individual_id, bin_index, epoch(timestamp) AS epoch_s, lon, lat
            FROM '{segments_parquet}'
            ORDER BY segment_id, bin_index
            """
        ).df()

        # Canonicalise IDs so numeric Movebank and string RAATD manifests compare safely.
        wanted = {id_key(i) for i in individual_ids}
        keep = np.fromiter((id_key(v) in wanted for v in frame["individual_id"]), bool, len(frame))
        if wanted and not keep.any():
            raise ValueError(
                f"none of the {len(wanted)} requested individuals appear in {segments_parquet}; "
                "this is a manifest/segments mismatch, not an empty split"
            )
        frame = frame.loc[keep].reset_index(drop=True)

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

    def gather(self, window_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Materialise a batch as (features, target_scaled, scale).
        """
        window_ids = np.asarray(window_ids, dtype=np.int64)
        base = self.starts[window_ids][:, None]
        idx = base + np.arange(self.context_length + 1, dtype=np.int64)[None, :]
        lon, lat = self.lon[idx], self.lat[idx]

        # Use real representative-fix times rather than assuming exact 6h spacing.
        epoch_s = self.epoch_s[idx]

        # Encode only the context so the target step cannot affect the window scale.
        c = self.context_length
        features, scale = encode_window(lon[:, :c], lat[:, :c], timestamp_s=epoch_s[:, :c])
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