# GenAI reference: Used GenAI to write these tests from the behaviours I specified; like subsets are nested across k,
# draws are reproducible/independent and don’t depend on manifest order etc. I checked them all too. 

from __future__ import annotations

import numpy as np
import pytest

from animal_movement_fm.train.finetune import (
    REGISTERED_FRACTIONS_PERCENT,
    SUBSET_FLOOR,
    draw_seed,
    nested_subsets,
    subset_size,
)

POOL = np.array([f"ind{i:03d}" for i in range(630)])  # Stork's training pool, roughly. 


def test_the_registered_fractions_are_what_the_preregistration_fixed():
    assert REGISTERED_FRACTIONS_PERCENT == (1, 10, 100)
    assert SUBSET_FLOOR == 5


def test_a_draw_is_nested_across_k():
    subsets = nested_subsets(POOL, draw=0)
    small = set(subsets[1].individuals)
    medium = set(subsets[10].individuals)
    full = set(subsets[100].individuals)
    assert small < medium < full
    assert full == set(POOL)


def test_draws_are_reproducible_and_independent():
    first = nested_subsets(POOL, draw=0)[10].individuals
    again = nested_subsets(POOL, draw=0)[10].individuals
    other = nested_subsets(POOL, draw=1)[10].individuals
    assert np.array_equal(first, again)
    assert not np.array_equal(first, other)
    assert draw_seed(0) != draw_seed(1)


def test_the_draw_does_not_depend_on_manifest_order():
    """The permutation is over the sorted set, so a reshuffled manifest gives the same subset."""
    shuffled = np.random.default_rng(7).permutation(POOL)
    assert np.array_equal(
        nested_subsets(POOL, draw=2)[1].individuals,
        nested_subsets(shuffled, draw=2)[1].individuals,
    )


def test_one_percent_of_a_large_pool_is_the_fraction_not_the_floor():
    subset = nested_subsets(POOL, draw=0)[1]
    assert subset.n_individuals == 6  # 1% of 630, rounded
    assert subset.floored is False
    assert subset.n_available == 630


def test_a_small_pool_is_raised_to_the_floor_and_says_so():
    """`canis_lupus` trains on 99 individuals, where 1% is one animal."""
    pool = np.array([f"w{i:02d}" for i in range(99)])
    subset = nested_subsets(pool, draw=0)[1]
    assert subset.n_individuals == SUBSET_FLOOR
    assert subset.floored is True
    # The realised fraction is what gets reported, not the nominal 1%.
    assert subset.realised_fraction == pytest.approx(5 / 99)


def test_the_floor_never_exceeds_what_exists():
    count, floored = subset_size(3, 1)
    assert count == 3 and floored is True


def test_k_100_is_the_whole_pool_and_is_never_floored():
    subset = nested_subsets(POOL, draw=0)[100]
    assert subset.n_individuals == len(POOL)
    assert subset.floored is False
    assert subset.realised_fraction == 1.0


@pytest.mark.parametrize("k", [0, -1, 101])
def test_fractions_outside_the_registered_range_are_refused(k):
    with pytest.raises(ValueError):
        subset_size(100, k)


def test_a_duplicated_individual_is_refused():
    with pytest.raises(ValueError, match="duplicate"):
        nested_subsets(np.array(["a", "b", "b"]), draw=0)
