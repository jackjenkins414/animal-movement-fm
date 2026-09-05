"""
Build the registered fine-tuning (F) subsets.

Under `preregistration_tier2.md` sec 3, F_k further trains Z on a fraction k of a held-out species' training individuals. 
Sampling is always by individual from the frozen split manifest, and subsets are nested within each draw so smaller 
fractions are contained in larger ones.

Each draw uses a fixed seed, and a minimum of five individuals is used when the requested fraction would be smaller. The 
realised subset size is returned so floored fractions are reported accurately.

This logic lives separately from the runner because it is registered behaviour that should be testable on its own.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# registered fine-tuning fractions from preregistration_tier2.md sec 3
REGISTERED_FRACTIONS_PERCENT: tuple[int, ...] = (1, 10, 100)
SUBSET_FLOOR = 5

# keep subset draw seeds separate from the split seed (42) and training seeds (0-4)
DRAW_SEED_BASE = 1_000


@dataclass(frozen=True)
class Subset:
    """One fine-tuning subset and the information needed to reproduce it."""

    k_percent: int
    draw: int
    seed: int
    individuals: np.ndarray
    n_available: int
    floored: bool

    @property
    def n_individuals(self) -> int:
        return int(len(self.individuals))

    @property
    def realised_fraction(self) -> float:
        """Return the actual share of the training pool used after applying the floor."""
        return self.n_individuals / self.n_available if self.n_available else 0.0


def draw_seed(draw: int) -> int:
    return DRAW_SEED_BASE + int(draw)


def subset_size(n_available: int, k_percent: int, floor: int = SUBSET_FLOOR) -> tuple[int, bool]:
    """
    Return `(n_selected, floored)` for one requested fraction.

    The raw fraction is rounded half up, then the minimum floor is applied and capped at the available pool size. `floored` 
    records when the floor, rather than k, determined the subset size.
    """
    if n_available < 1:
        raise ValueError("a fine-tuning pool needs at least one individual")
    if not 0 < k_percent <= 100:
        raise ValueError(f"k_percent must lie in (0, 100]; got {k_percent}")

    raw = int(np.floor(n_available * k_percent / 100.0 + 0.5))
    selected = min(max(raw, floor), n_available)
    return selected, selected > raw


def nested_subsets(
    individuals,
    *,
    draw: int,
    fractions: tuple[int, ...] = REGISTERED_FRACTIONS_PERCENT,
    floor: int = SUBSET_FLOOR,
) -> dict[int, Subset]:
    """
    Return nested fine-tuning subsets for one draw.

    A single seeded permutation is used and each fraction takes a prefix, so smaller subsets are always contained within larger 
    ones. Individuals are sorted before permutation so the draw depends on the set of IDs, not their order in the manifest.
    """
    pool = np.sort(np.asarray(individuals))
    if len(pool) != len(np.unique(pool)):
        raise ValueError("the training pool contains duplicate individual ids")

    seed = draw_seed(draw)
    order = np.random.default_rng(seed).permutation(len(pool))

    subsets = {}
    for k_percent in sorted(fractions):
        count, floored = subset_size(len(pool), k_percent, floor)
        subsets[k_percent] = Subset(
            k_percent=k_percent,
            draw=draw,
            seed=seed,
            # sorting only changes output order; the selected individuals are still the same prefix
            individuals=np.sort(pool[order[:count]]),
            n_available=len(pool),
            floored=floored,
        )
    return subsets