# GenAI reference: GenAI created these tests; I said that context_length should be keyword only and carry no default on all four 
# entry points, and ommiting it fails at bind time before any and all data is even touched. Found leftover C=24 from Tier 1 check, 
# which is what motivated these tests. 

from __future__ import annotations

import inspect

import pytest

from animal_movement_fm.data.benchmark import evaluate_on_split, run_ladder
from animal_movement_fm.data.split import window_counts_per_individual
from animal_movement_fm.eval.harness import evaluate

GUARDED = (evaluate, window_counts_per_individual, evaluate_on_split, run_ladder)


@pytest.mark.parametrize("func", GUARDED, ids=lambda f: f.__name__)
def test_context_length_has_no_default(func):
    """A caller that forgets context_length must fail, not silently pick a tier's constant."""
    parameter = inspect.signature(func).parameters["context_length"]
    assert parameter.default is inspect.Parameter.empty, (
        f"{func.__name__} has a default context_length of {parameter.default!r}; "
        "a defaulted context length is a silent wrong answer, not a convenience"
    )


@pytest.mark.parametrize("func", GUARDED, ids=lambda f: f.__name__)
def test_context_length_is_keyword_only(func):
    """Keyword-only, so it cannot be supplied by accident through positional drift."""
    parameter = inspect.signature(func).parameters["context_length"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY, (
        f"{func.__name__} takes context_length as {parameter.kind}; it must be keyword-only"
    )


@pytest.mark.parametrize("func", GUARDED, ids=lambda f: f.__name__)
def test_omitting_context_length_raises(func):
    """The behavioural half of the guard: omission raises before any data is touched."""
    with pytest.raises(TypeError, match="context_length"):
        inspect.signature(func).bind(*[object()] * _n_positional(func))


def _n_positional(func) -> int:
    return sum(
        1
        for p in inspect.signature(func).parameters.values()
        if p.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD and p.default is inspect.Parameter.empty
    )
