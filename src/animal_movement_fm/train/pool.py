"""
Pre encode Tier 2 source pool and keep it resident on the training device.

This avoids us repeatedly encoding NumPy batches on the CPU while the GPU waits. Windows are encoded in chunks, concatenated 
once on the target device, and sampled with the same registered species then window rule used by SourcePool.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace

import numpy as np
import torch

from animal_movement_fm.data.enu import N_FEATURES, SCALE_FLOOR_M
from animal_movement_fm.data.multispecies import HorizonWindows, SourcePool

ENCODE_CHUNK = 50_000


@dataclass
class PreparedPool:
    """
    Encoded windows for one training pool, store contiguously on a single device.

    Each species occupies one half-open row range in `species_bounds`. Sampling therefore keeps the registered hierarchical 
    rule without needing a separate row index lookup.
    """

    names: tuple[str, ...]
    features: torch.Tensor # (N, seq, n_features)
    targets: torch.Tensor # (N, 2)
    species_bounds: torch.Tensor  # (S, 2) int64
    probabilities: torch.Tensor  # (S,)
    token_mask: torch.Tensor | None = None  # (N, seq) bool, True where observed; horizon pools only
    target_mask: torch.Tensor | None = None  # (N, H) bool, True where the target exists; horizon pools only
    horizon_k: tuple[float, ...] | None = None  # k_h the (N, H, 2) targets were divided by

    @property
    def n_windows(self) -> int:
        return int(self.features.shape[0])

    @property
    def device(self) -> torch.device:
        return self.features.device

    @property
    def n_features(self) -> int:
        return int(self.features.shape[-1])

    def bytes_on_device(self) -> int:
        """
        Return memory used by the encoded feature tensor
        """
        return self.features.element_size() * self.features.nelement()

    @classmethod
    def from_source_pool(
        cls,
        pool: SourcePool,
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        chunk: int = ENCODE_CHUNK,
        sigma: float = SCALE_FLOOR_M,
        horizon_k: tuple[float, ...] | None = None,
    ) -> "PreparedPool":
        """
        Encode every window in pool and move the resulting tensors to device`.

        Encoding is chunked to limit temporary host memory use. Species remain contiguous in the final tensors so sampling 
        can use simple row ranges.

        A pool of HorizonWindows also keeps its token and target masks, and needs `horizon_k`: its targets are stored
        divided by k_h (see `horizon_k`).
        """
        device = torch.device(device)
        names = tuple(pool.names)
        horizon = _horizon_pool(pool)
        if horizon != (horizon_k is not None):
            raise ValueError("horizon_k is required for a pool of HorizonWindows, and only for one")
        if horizon:
            _check_k(pool, horizon_k)

        feature_blocks = []
        target_blocks = []
        token_blocks = []
        target_mask_blocks = []
        bounds = []
        cursor = 0

        for name in names:
            windows = pool.species[name]
            total = windows.n_windows

            if total == 0:
                raise ValueError(
                    f"species {name!r} contributes no windows to this pool. An empty species cannot be "
                    "drawn from and its registered weight would silently redistribute."
                )

            for start in range(0, total, chunk):
                ids = np.arange(start, min(start + chunk, total), dtype=np.int64)
                if horizon:
                    features, token_mask, targets, target_mask, _ = windows.gather(ids, sigma=sigma)
                    targets = divide_by_k(targets, horizon_k)
                    token_blocks.append(torch.as_tensor(token_mask, dtype=torch.bool).to(device))
                    target_mask_blocks.append(torch.as_tensor(target_mask, dtype=torch.bool).to(device))
                else:
                    features, targets, _ = windows.gather(ids, sigma=sigma)

                feature_blocks.append(torch.as_tensor(features, dtype=dtype).to(device))
                target_blocks.append(torch.as_tensor(targets, dtype=dtype).to(device))

            bounds.append((cursor, cursor + total))
            cursor += total

        return cls(
            names=names,
            features=torch.cat(feature_blocks),
            targets=torch.cat(target_blocks),
            species_bounds=torch.tensor(bounds, dtype=torch.int64, device=device),
            probabilities=torch.as_tensor(
                pool.probabilities,
                dtype=torch.float64,
            ).to(device),
            token_mask=torch.cat(token_blocks) if horizon else None,
            target_mask=torch.cat(target_mask_blocks) if horizon else None,
            horizon_k=tuple(float(k) for k in horizon_k) if horizon else None,
        )

    def sample(
        self,
        generator: torch.Generator,
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Draw one batch under the registered species-then-window sampling rule.

        Returns `(features, targets, species_index)`. The species index is retained so realised loss/sample shares can be audited later.
        """
        if self.target_mask is not None:
            raise ValueError("a horizon pool is drawn with sample_horizon; sample would drop its masks")
        rows, picks = _draw_rows(self.probabilities, self.species_bounds, generator, batch_size)
        return self.features[rows], self.targets[rows], picks

    def sample_horizon(
        self,
        generator: torch.Generator,
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        `sample`'s draws on a horizon pool: (features, token_mask, targets (B, H, 2) / k_h, target_mask, species_index).
        """
        if self.target_mask is None:
            raise ValueError("sample_horizon needs a pool built from HorizonWindows")
        rows, picks = _draw_rows(self.probabilities, self.species_bounds, generator, batch_size)
        return self.features[rows], self.token_mask[rows], self.targets[rows], self.target_mask[rows], picks

    def realised_species_shares(
        self,
        generator: torch.Generator,
        draws: int = 200_000,
    ) -> dict[str, float]:
        """
        Estimate the realised species shares produced by the sampler.

        Reporting diagnostic only; doesnt feed back into the registered probabilities or training configuration.
        """
        picks = torch.multinomial(
            self.probabilities,
            draws,
            replacement=True,
            generator=generator,
        )

        return {
            name: float((picks == i).to(torch.float64).mean())
            for i, name in enumerate(self.names)
        }


def _draw_rows(
    probabilities: torch.Tensor,
    species_bounds: torch.Tensor,
    generator: torch.Generator,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    (rows, species_index) under the registered species-then-window rule; every pool draws through this.
    """
    picks = torch.multinomial(
        probabilities,
        batch_size,
        replacement=True,
        generator=generator,
    )

    starts = species_bounds[picks, 0]
    widths = species_bounds[picks, 1] - starts

    offsets = (
        torch.rand(
            batch_size,
            generator=generator,
            device=widths.device,
            dtype=torch.float64,
        )
        * widths.to(torch.float64)
    ).to(torch.int64)

    rows = starts + torch.clamp(offsets, max=widths - 1)
    return rows, picks


# GenAI reference: I specified the 1 h study's pools; horizon targets divided by one k_h per horizon and pool (our A8 weighted median 
# of r_h over the training windows), sample_horizon on samples draws, a pool that encodes on the CPU per batch for C > 12, and a check 
# that the C = 12 core is bitwise the main design's at every C. GenAI wrote this section; I checked it
def _horizon_pool(pool: SourcePool) -> bool:
    """True for a pool of HorizonWindows, False for SpeciesWindows; a mixture is refused."""
    kinds = {isinstance(pool.species[name], HorizonWindows) for name in pool.names}
    if len(kinds) != 1:
        raise ValueError("a pool mixes HorizonWindows with SpeciesWindows")
    return kinds.pop()


def _check_k(pool: SourcePool, horizon_k) -> None:
    horizons = {pool.species[name].horizons for name in pool.names}
    if len(horizons) != 1:
        raise ValueError(f"species disagree on horizons: {sorted(horizons)}")
    if len(horizon_k) != len(horizons.pop()) or not all(np.isfinite(k) and k > 0 for k in horizon_k):
        raise ValueError(f"horizon_k {horizon_k} needs one finite positive k per horizon")


def divide_by_k(targets: np.ndarray, horizon_k) -> np.ndarray:
    """(n, H, 2) scaled targets divided by k_h per horizon; every pool divides through this."""
    return targets / np.asarray(horizon_k, dtype=float)[None, :, None]


def weighted_median(values: np.ndarray, weights: np.ndarray, rtol: float = 1e-9) -> float:
    """
    The value where the cumulative weight reaches half; where it lands on half (within rtol of the total) the
    midpoint of the two values either side, so equal weights give np.median.
    """
    order = np.argsort(values, kind="stable")
    values, cumulative = np.asarray(values, dtype=float)[order], np.cumsum(np.asarray(weights, dtype=float)[order])
    if not len(values):
        raise ValueError("no values to take a median of")
    half, tolerance = cumulative[-1] / 2, rtol * cumulative[-1]
    i = int(np.searchsorted(cumulative, half - tolerance))
    if abs(cumulative[i] - half) <= tolerance and i + 1 < len(values):
        return float((values[i] + values[i + 1]) / 2)
    return float(values[i])


def horizon_ratios(windows: HorizonWindows, *, sigma: float = SCALE_FLOOR_M, chunk: int = ENCODE_CHUNK) -> list:
    """
    r_h = |target| / window scale for every window with a target at h, per horizon, from the 12-bin core alone.
    """
    core = replace(windows, context=windows.core)  # Targets and scale come from the core, so C cannot move them. 
    ratios = [[] for _ in windows.horizons]
    for start in range(0, core.n_windows, chunk):
        ids = np.arange(start, min(start + chunk, core.n_windows), dtype=np.int64)
        _, _, targets, target_mask, _ = core.gather(ids, sigma=sigma)
        r = np.linalg.norm(targets, axis=-1)
        for j in range(len(windows.horizons)):
            ratios[j].append(r[target_mask[:, j], j])
    return [np.concatenate(parts) if parts else np.empty(0) for parts in ratios]


def horizon_k(pool: SourcePool, *, sigma: float = SCALE_FLOOR_M, chunk: int = ENCODE_CHUNK) -> tuple[float, ...]:
    """
    k_h per horizon: the median of r_h over the pool's windows with a target at h, each window weighted p_s / n_s
    (n_s the species' windows, so a species weighs at h what the sampler gives the h head). One species: np.median.
    """
    if not _horizon_pool(pool):
        raise ValueError("horizon_k needs a pool of HorizonWindows")
    horizons = pool.species[pool.names[0]].horizons
    _check_k(pool, (1.0,) * len(horizons))
    values, weights = [[] for _ in horizons], [[] for _ in horizons]
    for name, p in zip(pool.names, pool.probabilities):
        windows = pool.species[name]
        for j, r in enumerate(horizon_ratios(windows, sigma=sigma, chunk=chunk)):
            values[j].append(r)
            weights[j].append(np.full(len(r), p / windows.n_windows))
    return tuple(weighted_median(np.concatenate(v), np.concatenate(w)) for v, w in zip(values, weights))


@dataclass
class StreamingPool:
    """
    Horizon windows encoded on the CPU per batch, for histories too long to hold encoded (C > 12).

    Draws through `_draw_rows` as PreparedPool does, so the same generator gives the same rows; the batch is then
    gathered from each species' HorizonWindows and moved to the device.
    """

    names: tuple[str, ...]
    species: tuple[HorizonWindows, ...]
    species_bounds: torch.Tensor  # (S, 2) int64, on the device
    probabilities: torch.Tensor  # (S,), on the device
    horizon_k: tuple[float, ...]
    sigma: float = SCALE_FLOOR_M
    dtype: torch.dtype = torch.float32

    @property
    def n_windows(self) -> int:
        return int(self.species_bounds[-1, 1])

    @property
    def n_features(self) -> int:
        return N_FEATURES

    @property
    def device(self) -> torch.device:
        return self.species_bounds.device

    def bytes_on_device(self) -> int:
        """Nothing is held encoded; the device holds one batch at a time."""
        return 0

    @classmethod
    def from_source_pool(
        cls,
        pool: SourcePool,
        *,
        horizon_k: tuple[float, ...],
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        sigma: float = SCALE_FLOOR_M,
    ) -> "StreamingPool":
        if not _horizon_pool(pool):
            raise ValueError("StreamingPool needs a pool of HorizonWindows")
        _check_k(pool, horizon_k)
        device = torch.device(device)
        names = tuple(pool.names)
        bounds, cursor = [], 0
        for name in names:
            total = pool.species[name].n_windows
            if total == 0:
                raise ValueError(
                    f"species {name!r} contributes no windows to this pool. An empty species cannot be "
                    "drawn from and its registered weight would silently redistribute."
                )
            bounds.append((cursor, cursor + total))
            cursor += total
        return cls(
            names=names,
            species=tuple(pool.species[name] for name in names),
            species_bounds=torch.tensor(bounds, dtype=torch.int64, device=device),
            probabilities=torch.as_tensor(pool.probabilities, dtype=torch.float64).to(device),
            horizon_k=tuple(float(k) for k in horizon_k),
            sigma=float(sigma),
            dtype=dtype,
        )

    def gather_rows(self, rows: np.ndarray, picks: np.ndarray) -> tuple[np.ndarray, ...]:
        """Encode pool rows on the CPU, in the order given: (features, token_mask, targets / k_h, target_mask)."""
        out = None
        starts = self.species_bounds[:, 0].cpu().numpy()
        for s in np.unique(picks):
            where = np.flatnonzero(picks == s)
            features, token_mask, targets, target_mask, _ = self.species[s].gather(
                rows[where] - starts[s], sigma=self.sigma
            )
            parts = (features, token_mask, divide_by_k(targets, self.horizon_k), target_mask)
            if out is None:
                out = [np.zeros((len(rows),) + p.shape[1:], dtype=p.dtype) for p in parts]
            for column, part in zip(out, parts):
                column[where] = part
        return tuple(out)

    def sample_horizon(
        self,
        generator: torch.Generator,
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """PreparedPool.sample_horizon's contract, encoded per batch."""
        rows, picks = _draw_rows(self.probabilities, self.species_bounds, generator, batch_size)
        features, token_mask, targets, target_mask = self.gather_rows(rows.cpu().numpy(), picks.cpu().numpy())
        return (
            torch.as_tensor(features, dtype=self.dtype).to(self.device),
            torch.as_tensor(token_mask, dtype=torch.bool).to(self.device),
            torch.as_tensor(targets, dtype=self.dtype).to(self.device),
            torch.as_tensor(target_mask, dtype=torch.bool).to(self.device),
            picks,
        )

    def realised_species_shares(self, generator: torch.Generator, draws: int = 200_000) -> dict[str, float]:
        """As PreparedPool's; a reporting diagnostic only."""
        picks = torch.multinomial(self.probabilities, draws, replacement=True, generator=generator)
        return {name: float((picks == i).to(torch.float64).mean()) for i, name in enumerate(self.names)}


def core_matches(
    main: SourcePool,
    ablation: SourcePool,
    ids: Mapping[str, np.ndarray],
    *,
    sigma: float = SCALE_FLOOR_M,
) -> list[str]:
    """
    Failures (empty if none) of the ablation's C = 12 core against the main design's: equal anchor sets and targets
    per species, and on `ids` (window ids per species) core tokens, scale, targets and target masks bitwise equal;
    then k_h bitwise equal.
    """
    failures = []
    if tuple(main.names) != tuple(ablation.names) or not np.array_equal(main.probabilities, ablation.probabilities):
        return ["species or probabilities differ"]
    for name in main.names:
        m, a = main.species[name], ablation.species[name]
        if m.context != m.core:
            failures.append(f"{name}: main context {m.context} is not the {m.core}-bin core")
        if (m.core, m.horizons, m.bin_seconds, m.clip) != (a.core, a.horizons, a.bin_seconds, a.clip):
            failures.append(f"{name}: core, horizons, bin_seconds or clip differ")
            continue
        if not (np.array_equal(m.anchors, a.anchors) and np.array_equal(m.targets, a.targets)
                and np.array_equal(m.keys, a.keys)):
            failures.append(f"{name}: anchor sets or target positions differ")
            continue
        f_m, _, t_m, tm_m, s_m = m.gather(ids[name], sigma=sigma)
        f_a, _, t_a, tm_a, s_a = a.gather(ids[name], sigma=sigma)
        checks = {
            "core tokens": np.array_equal(f_a[:, f_a.shape[1] - f_m.shape[1]:], f_m),
            "scale": np.array_equal(s_a, s_m),
            "targets": np.array_equal(t_a, t_m) and np.array_equal(tm_a, tm_m),
        }
        failures += [f"{name}: {what} differ" for what, ok in checks.items() if not ok]
    if not failures and horizon_k(main, sigma=sigma) != horizon_k(ablation, sigma=sigma):
        failures.append("k_h differ")
    return failures
