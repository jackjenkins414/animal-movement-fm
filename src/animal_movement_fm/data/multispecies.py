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
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from animal_movement_fm.data import era5
from animal_movement_fm.data.enu import (
    FEATURES,
    SCALE_FLOOR_M,
    encode_history,
    encode_window,
    enu_steps,
    scale_target,
)
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


def _read_segments(segments_parquet: str | Path, individual_ids, env_parquet: str | Path | None = None) -> pd.DataFrame:
    """
    One species' segment rows for the selected individuals, ordered by segment then bin.

    With `env_parquet`, the ERA5 columns (era5.RAW) are left joined on (segment_id, bin_index); a row with no env row keeps NaN there.
    """
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
    # get epoch seconds in DuckDB to avoid datetime unit mismatches in pandas
    if env_parquet is not None:
        return _read_segments_env(con, segments_parquet, env_parquet)
    frame = con.execute(
        f"""
        SELECT s.segment_id, s.individual_id, s.bin_index, epoch(s.timestamp) AS epoch_s, s.lon, s.lat
        FROM '{segments_parquet}' s
        SEMI JOIN split_ids ids ON s.individual_id = ids.individual_id
        ORDER BY s.segment_id, s.bin_index
        """
    ).df()
    return frame


# GenAI reference: I specified ERA5 covariates for the env study; a left join of the env file on (segment_id, bin_index) only when one is given, so 
# the env off query is the one above, unchanged. GenAI wrote this; I checked it
def _read_segments_env(con, segments_parquet: str | Path, env_parquet: str | Path) -> pd.DataFrame:
    """
    `_read_segments`' rows with the ERA5 columns left-joined; `split_ids` must already be registered on `con`.
    """
    columns = ", ".join(f"e.{name}" for name in era5.RAW)
    frame = con.execute(
        f"""
        SELECT s.segment_id, s.individual_id, s.bin_index, epoch(s.timestamp) AS epoch_s, s.lon, s.lat, {columns}
        FROM '{segments_parquet}' s
        SEMI JOIN split_ids ids ON s.individual_id = ids.individual_id
        LEFT JOIN '{env_parquet}' e ON s.segment_id = e.segment_id AND s.bin_index = e.bin_index
        ORDER BY s.segment_id, s.bin_index
        """
    ).df()
    if frame.duplicated(["segment_id", "bin_index"]).any():
        raise ValueError(f"{env_parquet} repeats a (segment_id, bin_index) key; the join would duplicate fixes")
    return frame


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
        frame = _read_segments(segments_parquet, individual_ids)

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


SEGMENT_KEY_STRIDE = np.int64(1) << 32  # Separate key range for each segment.


def _segment_keys(segment_id: np.ndarray, bin_index: np.ndarray) -> np.ndarray:
    """
    Offset bin indices by segment rank. Rows must be ordered by segment, then bin.
    """
    new_segment = np.ones(len(bin_index), dtype=bool)
    new_segment[1:] = segment_id[1:] != segment_id[:-1]
    segment_rank = np.cumsum(new_segment) - 1

    return segment_rank.astype(np.int64) * SEGMENT_KEY_STRIDE + bin_index.astype(np.int64)


def direct_anchors(
    keys: np.ndarray,
    core: int,
    horizons: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Find anchors with a contiguous core and at least one requested target.

    Return anchor positions and target positions, using -1 for missing targets.
    """
    anchors = np.arange(core - 1, len(keys))
    anchors = anchors[keys[anchors] - keys[anchors - core + 1] == core - 1]

    targets = np.full((len(anchors), len(horizons)), -1, dtype=np.int64)
    for column, horizon in enumerate(horizons):
        target_keys = keys[anchors] + horizon
        target_positions = np.minimum(
            np.searchsorted(keys, target_keys),
            max(len(keys) - 1, 0),
        )
        targets[:, column] = np.where(
            keys[target_positions] == target_keys,
            target_positions,
            -1,
        )

    keep = (targets >= 0).any(axis=1)
    return anchors[keep], targets[keep]


def history_positions(
    keys: np.ndarray,
    anchors: np.ndarray,
    context: int,
) -> np.ndarray:
    """
    Find history positions within each anchor's segment, oldest first.

    Return an (n_anchors, context) array with -1 for missing bins.
    """
    positions = anchors[:, None] + np.arange(-context + 1, 1)[None, :]
    safe_positions = np.maximum(positions, 0)
    columns = keys[safe_positions] - (keys[anchors] - context + 1)[:, None]

    same_segment = (
        keys[safe_positions] // SEGMENT_KEY_STRIDE
        == (keys[anchors] // SEGMENT_KEY_STRIDE)[:, None]
    )
    valid = (positions >= 0) & same_segment & (columns >= 0)

    grid = np.full(positions.shape, -1, dtype=np.int64)
    rows = np.broadcast_to(np.arange(len(anchors))[:, None], positions.shape)
    grid[rows[valid], columns[valid]] = positions[valid]

    return grid


@dataclass
class HorizonWindows:
    """One species' windows, stored as anchor and target offsets into flat arrays."""

    name: str
    lon: np.ndarray
    lat: np.ndarray
    epoch_s: np.ndarray
    keys: np.ndarray
    individuals: np.ndarray  # canonical IDs, indexed by individual_index
    individual_index: np.ndarray  # One index per anchor
    anchors: np.ndarray
    targets: np.ndarray  # (n_anchors, n_horizons); -1 for missing targets
    context: int
    core: int
    horizons: tuple[int, ...]
    bin_seconds: int
    clip: float | None = None
    env: np.ndarray | None = None  # (rows, len(era5.RAW)) ERA5 at each row, NaN where missing; context = core only
    env_stats: tuple[np.ndarray, np.ndarray] | None = None  # (mean, sd) of the standardised channels, training pool's

    @property
    def n_windows(self) -> int:
        return int(len(self.anchors))

    @classmethod
    def build(
        cls,
        segments_parquet: str | Path,
        individual_ids,
        *,
        context: int,
        core: int,
        horizons: tuple[int, ...],
        bin_seconds: int,
        clip: float | None = None,
        name: str | None = None,
        env_parquet: str | Path | None = None,
    ) -> "HorizonWindows":
        """
        Build windows for the selected individuals of one species.

        With `env_parquet`, each row keeps its ERA5 covariates; `gather` then needs `env_stats` (see pool.env_stats).
        """
        if context < core:
            raise ValueError(f"context {context} is shorter than the {core}-bin core")
        if env_parquet is not None and context != core:
            raise ValueError(f"covariates are for the {core}-bin core only, not context {context}")
        # GenAI ref: completed a small part of this function
        frame = _read_segments(segments_parquet, individual_ids, env_parquet=env_parquet)
        bins = frame["bin_index"].to_numpy(np.int64)
        if len(bins) and (bins.min() < 0 or bins.max() >= SEGMENT_KEY_STRIDE):
            raise ValueError("bin indices must lie in [0, 2**32) for the segment keys")

        keys = _segment_keys(frame["segment_id"].to_numpy(), bins)
        horizons = tuple(int(h) for h in horizons)
        anchors, targets = direct_anchors(keys, core, horizons)
        individuals, individual_index = np.unique(
            np.array([id_key(v) for v in frame["individual_id"]]),
            return_inverse=True,
        )

        return cls(
            name=name or Path(segments_parquet).stem,
            lon=frame["lon"].to_numpy(float),
            lat=frame["lat"].to_numpy(float),
            epoch_s=frame["epoch_s"].to_numpy(float),
            keys=keys,
            individuals=individuals,
            individual_index=individual_index[anchors],
            anchors=anchors,
            targets=targets,
            context=int(context),
            core=int(core),
            horizons=horizons,
            bin_seconds=int(bin_seconds),
            clip=clip,
            env=None if env_parquet is None else frame[list(era5.RAW)].to_numpy(float),
        )

    def for_horizon(self, h: int) -> "HorizonWindows":
        """
        Keep windows with a target at h, retaining all their horizon targets.
        """
        keep = self.targets[:, self.horizons.index(h)] >= 0

        return replace(
            self,
            anchors=self.anchors[keep],
            targets=self.targets[keep],
            individual_index=self.individual_index[keep],
        )

    def gather(
        self,
        window_ids: np.ndarray,
        *,
        sigma: float = SCALE_FLOOR_M,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Return features, token mask, scaled targets, target mask and window scale.

        Targets use the core's scale. Missing targets are zero and masked out. With `env`, the 15 standardised ERA5 channels follow the 12 
        movement channels on every token (era5.ENV_CHANNELS).
        """
        window_ids = np.asarray(window_ids, dtype=np.int64)
        anchors = self.anchors[window_ids]
        history_grid = history_positions(self.keys, anchors, self.context)
        history_indices = np.maximum(history_grid, 0)

        features, token_mask, scale = encode_history(
            self.lon[history_indices],
            self.lat[history_indices],
            timestamp_s=self.epoch_s[history_indices],
            observed=history_grid >= 0,
            core=self.core,
            sigma=sigma,
            bin_seconds=self.bin_seconds,
            clip=self.clip,
        )

        target_positions = self.targets[window_ids]
        target_mask = target_positions >= 0
        pairs = np.stack(
            [
                np.broadcast_to(anchors[:, None], target_positions.shape),
                np.maximum(target_positions, 0),
            ],
            axis=-1,
        )

        steps = enu_steps(self.lon[pairs], self.lat[pairs])[..., 0, :]
        scaled_targets = np.where(
            target_mask[..., None],
            scale_target(steps, scale[:, None]),
            0.0,
        )

        if self.env is not None:
            if self.env_stats is None:
                raise ValueError(f"{self.name}: env is set but env_stats is not; take them from pool.env_stats")
            channels = self._env_channels(history_indices[:, 1:], features)
            features = np.concatenate([features, era5.standardise(channels, *self.env_stats)], axis=-1)

        return features, token_mask, scaled_targets, target_mask, scale

    def env_tokens(self, window_ids: np.ndarray, *, sigma: float = SCALE_FLOOR_M) -> np.ndarray:
        """
        Unstandardised (n, tokens, 15) ERA5 channels of the windows, for the pool's statistics.
        """
        if self.env is None:
            raise ValueError(f"{self.name} has no covariates")
        window_ids = np.asarray(window_ids, dtype=np.int64)
        features = replace(self, env=None, env_stats=None).gather(window_ids, sigma=sigma)[0]
        rows = history_positions(self.keys, self.anchors[window_ids], self.context)[:, 1:]
        return self._env_channels(rows, features)

    def _env_channels(self, rows: np.ndarray, features: np.ndarray) -> np.ndarray:
        """Token channels from the ERA5 rows each token lands on (n, tokens) and its movement features."""
        if self.context != self.core:
            raise ValueError(f"covariates are for the {self.core}-bin core only, not context {self.context}")
        return era5.token_channels(
            self.env[rows],
            features[..., FEATURES.index("heading_cos")],
            features[..., FEATURES.index("heading_sin")],
        )


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

    def sample(self, rng: np.random.Generator, batch_size: int) -> tuple[np.ndarray, ...]:
        """
        Draw one mixed-species batch under the registered sampling rule.

        Returns whatever the species' `gather` returns, concatenated: (features, targets, scale) for SpeciesWindows,
        five arrays for HorizonWindows. The draws do not depend on which.
        """
        picks = rng.choice(len(self.names), size=batch_size, p=self.probabilities)
        parts = []
        for index, name in enumerate(self.names):
            count = int((picks == index).sum())
            if not count:
                continue
            windows = self.species[name]
            ids = rng.integers(0, windows.n_windows, size=count)
            parts.append(windows.gather(ids))
        return tuple(np.concatenate(column) for column in zip(*parts))

    def realised_species_shares(self, rng: np.random.Generator, draws: int = 100_000) -> dict[str, float]:
        """
        Return realised source-species shares from repeated sampler draws.
        """
        picks = rng.choice(len(self.names), size=draws, p=self.probabilities)
        return {name: float((picks == i).mean()) for i, name in enumerate(self.names)}
