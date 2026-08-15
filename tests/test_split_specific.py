# GenAI reference: I outlined that val and test are drawn only from individuals above the evaluation floor while the 
# below floor anaimals still reach training, that the 3 splits are disjoint and cover everything with a min of 1 
# window, and that A7 gives us a larger training pool than the filter then split option on the same input, etc. Note, 
# the row order test is that mac vs gadi bug i hit; each of these has been checked. 
"""
Tests for the A7 split specific policy.

Note: kept separate from `test_split.py` so our frozen manifest regression tests for `make_split` stay clearly about the artefact 
Tier 1 depends on.
"""

import numpy as np
import pandas as pd
import pytest

from animal_movement_fm.data.split import make_split, make_split_specific


def _counts(n=100, seed=0):
    """Individuals with a spread of window counts, some below the evaluation floor."""
    rng = np.random.default_rng(seed)
    ids = rng.choice(np.arange(10_000, 99_999), size=n, replace=False)
    windows = np.concatenate([
        rng.integers(1, 25, size=n // 2),    # training-only: below the floor
        rng.integers(25, 500, size=n - n // 2),  # eligible for evaluation
    ])
    return pd.DataFrame({"individual_id": ids, "n_windows": windows})


def test_val_and_test_come_only_from_the_eligible_pool():
    counts = _counts()
    split = make_split_specific(counts)
    floor = counts.set_index("individual_id")["n_windows"]
    for group in (split.val, split.test):
        assert (floor.loc[group] >= 25).all()


def test_short_individuals_reach_training_rather_than_being_discarded():
    """The whole point of A7: below-floor animals still supply supervision."""
    counts = _counts()
    split = make_split_specific(counts)
    floor = counts.set_index("individual_id")["n_windows"]
    train_windows = floor.loc[split.train]
    assert (train_windows < 25).any(), "no below-floor individual reached training"
    # Everything with at least one window is somewhere.
    assigned = set(split.train) | set(split.val) | set(split.test)
    assert assigned == set(counts.loc[counts["n_windows"] >= 1, "individual_id"])


def test_splits_are_disjoint():
    split = make_split_specific(_counts())
    assert not (set(split.train) & set(split.val))
    assert not (set(split.train) & set(split.test))
    assert not (set(split.val) & set(split.test))


def test_val_and_test_are_exactly_the_requested_size():
    split = make_split_specific(_counts(), n_val=10, n_test=10)
    assert len(split.val) == 10
    assert len(split.test) == 10


def test_it_yields_more_training_individuals_than_filter_then_split():
    """A7's stated benefit, on the same input."""
    counts = _counts(n=200)
    a7 = make_split_specific(counts, eval_min_windows=25)
    old = make_split(counts, min_windows=25)
    assert len(a7.train) > len(old.train)


def test_row_order_of_counts_cannot_change_the_split():
    """The Mac-vs-Gadi bug: a seeded permutation still inherits the order it is handed."""
    counts = _counts()
    shuffled = counts.sample(frac=1, random_state=7).reset_index(drop=True)
    a = make_split_specific(counts)
    b = make_split_specific(shuffled)
    assert sorted(a.test) == sorted(b.test)
    assert sorted(a.val) == sorted(b.val)
    assert sorted(a.train) == sorted(b.train)


def test_the_seed_determines_the_draw():
    counts = _counts()
    assert sorted(make_split_specific(counts, seed=1).test) != sorted(
        make_split_specific(counts, seed=2).test
    )
    assert sorted(make_split_specific(counts, seed=1).test) == sorted(
        make_split_specific(counts, seed=1).test
    )


def test_too_small_an_eligible_pool_raises_rather_than_silently_shrinking():
    """A species that cannot fill validation and test must fail gate 1, not produce a short split."""
    counts = pd.DataFrame({"individual_id": np.arange(30), "n_windows": [100] * 15 + [1] * 15})
    with pytest.raises(ValueError, match="fewer than"):
        make_split_specific(counts, n_val=10, n_test=10)


def test_a_training_floor_above_the_evaluation_floor_is_rejected():
    """That would make the training pool a subset of the evaluation pool, inverting the policy."""
    with pytest.raises(ValueError, match="inverts the policy"):
        make_split_specific(_counts(), eval_min_windows=25, train_min_windows=50)


def test_individuals_with_no_windows_are_in_no_split():
    counts = _counts(n=40)
    counts.loc[counts.index[:5], "n_windows"] = 0
    split = make_split_specific(counts)
    zero = set(counts.loc[counts["n_windows"] == 0, "individual_id"])
    assert not (zero & (set(split.train) | set(split.val) | set(split.test)))
