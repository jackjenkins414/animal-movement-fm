# GenAI reference: Used GenAI to write these tests from the behaviours I specified; same as before; registered arms have 
# no species lookup, conditioning stays the same, lookup rows are applied per window, bad configs/donors are rejected,freezing
# only leaves the lookup trainable, and the predictor won’t guess a species row. I checked all the tests were correct.

from __future__ import annotations

import numpy as np
import pytest
import torch

from animal_movement_fm.data.enu import N_FEATURES
from animal_movement_fm.eval.fm_predictor import MovementFMPredictor
from animal_movement_fm.models.fm import CONTEXT_COND_DIM, MovementFM
from animal_movement_fm.train.loop import TrainConfig, train
from animal_movement_fm.train.pool import PreparedPool

BATCH, SEQ = 6, 11
NAMES = ("cervus_elaphus", "ciconia_ciconia", "mirounga_leonina")


def _features(seed: int = 0) -> torch.Tensor:
    return torch.randn(BATCH, SEQ, N_FEATURES, generator=torch.Generator().manual_seed(seed))


def _pool(names=NAMES, per_species: int = 40, seed: int = 3) -> PreparedPool:
    """A synthetic pool with the same shape contract as the real one."""
    generator = torch.Generator().manual_seed(seed)
    total = per_species * len(names)
    return PreparedPool(
        names=tuple(names),
        features=torch.randn(total, SEQ, N_FEATURES, generator=generator),
        targets=torch.randn(total, 2, generator=generator) * 0.5,
        species_bounds=torch.tensor(
            [[i * per_species, (i + 1) * per_species] for i in range(len(names))],
            dtype=torch.int64,
        ),
        probabilities=torch.full((len(names),), 1.0 / len(names), dtype=torch.float64),
    )


def _config(**overrides) -> TrainConfig:
    base = dict(d_model=16, num_layers=1, lr=1e-3, steps=8, seed=0, batch_size=8)
    return TrainConfig(**{**base, **overrides})


def _embedding_config(**overrides) -> TrainConfig:
    defaults = dict(
        use_fm_package=True,
        use_context_conditioning=False,
        use_reconstruction_head=False,
        n_species=len(NAMES),
        species_embed_dim=4,
    )
    return _config(**{**defaults, **overrides})


# --------------------------------------------------------------------- geometry


def test_registered_arms_carry_no_species_lookup():
    z = MovementFM(d_model=16, num_layers=1, use_fm_package=True)
    b = MovementFM(d_model=16, num_layers=1, use_fm_package=False)
    assert z.species_embed is None and b.species_embed is None
    # the registered conditioning vector is the three window statistics and nothing else
    assert z.cond_proj[0].in_features == CONTEXT_COND_DIM
    assert b.cond_proj is None


def test_the_embedding_arm_conditions_on_identity_alone():
    model = MovementFM(
        d_model=16,
        num_layers=1,
        use_fm_package=True,
        use_context_conditioning=False,
        n_species=3,
        species_embed_dim=4,
    )
    assert model.cond_proj[0].in_features == 4
    species = torch.tensor([0, 1, 2, 0, 1, 2])
    cond = model.conditioning_inputs(_features(), species=species)
    # with the window statistics switched off the conditioning vector *is* the looked-up row
    assert torch.equal(cond, model.species_embed(species))
    # ... and it does not move when the window does
    assert torch.equal(cond, model.conditioning_inputs(_features(seed=1), species=species))


def test_the_species_row_changes_the_forecast():
    torch.manual_seed(0)
    model = MovementFM(
        d_model=16,
        num_layers=1,
        use_fm_package=True,
        use_context_conditioning=False,
        n_species=3,
        species_embed_dim=4,
    ).eval()
    with torch.no_grad():
        model.species_embed.weight.copy_(torch.tensor([[1.0] * 4, [-1.0] * 4, [0.5] * 4]))
        features = _features()
        rows = [model(features, species=torch.full((BATCH,), i)) for i in range(3)]
        assert not torch.allclose(rows[0], rows[1], atol=1e-4)
        # the row is applied per window, not per batch
        mixed = model(features, species=torch.tensor([0, 1, 2, 0, 1, 2]))
        for position, row in enumerate([0, 1, 2, 0, 1, 2]):
            assert torch.allclose(mixed[position], rows[row][position], atol=1e-6)


def test_a_model_with_a_lookup_refuses_to_run_without_one():
    model = MovementFM(
        d_model=16,
        num_layers=1,
        use_fm_package=True,
        use_context_conditioning=False,
        n_species=3,
        species_embed_dim=4,
    )
    with pytest.raises(ValueError, match="species index"):
        model(_features())


def test_the_conditioning_path_cannot_be_left_with_no_input():
    with pytest.raises(ValueError, match="conditioning input"):
        MovementFM(d_model=16, num_layers=1, use_fm_package=True, use_context_conditioning=False)


# --------------------------------------------------------------------- configuration guards


@pytest.mark.parametrize(
    "overrides",
    [
        dict(species_embed_dim=4, n_species=0),
        dict(random_species_labels=True),
        dict(freeze_except_species_embedding=True),
    ],
)
def test_incoherent_configurations_are_refused(overrides):
    with pytest.raises(ValueError):
        _config(**overrides)


# --------------------------------------------------------------------- training


def test_the_embedding_trains_and_the_control_differs_from_it():
    pool = _pool()
    embed = train(_embedding_config(), pool, checkpoint_every=0)
    control = train(_embedding_config(random_species_labels=True), pool, checkpoint_every=0)

    rows = embed.model.species_embed.weight.detach()
    assert rows.shape == (len(NAMES), 4)
    assert torch.isfinite(rows).all()
    # the rows separate: a lookup that stayed at its initialisation would be a null result about
    # the optimiser rather than about species information
    assert float((rows[0] - rows[1]).norm()) > 0
    # the control is a different run, not a relabelling of the same one
    assert embed.loss_history != control.loss_history


def test_the_registered_rng_stream_is_untouched_by_the_new_switches():
    """B and the embedding arm draw the same batches; only the control consumes extra randomness."""
    pool = _pool()
    baseline = train(_config(use_fm_package=False), pool, log_every=1, checkpoint_every=0)
    repeat = train(_config(use_fm_package=False), pool, log_every=1, checkpoint_every=0)
    assert baseline.loss_history == repeat.loss_history


def test_freezing_leaves_only_the_lookup_trainable():
    pool = _pool()
    donor = train(_embedding_config(), pool, checkpoint_every=0)
    weights = {k: v for k, v in donor.model.state_dict().items() if k != "species_embed.weight"}

    target = _pool(names=("larus_fuscus",))
    fitted = train(
        _embedding_config(n_species=1, freeze_except_species_embedding=True, lr=1e-2, steps=12),
        target,
        checkpoint_every=0,
        init_state_dict=weights,
    )

    trainable = [n for n, p in fitted.model.named_parameters() if p.requires_grad]
    assert trainable == ["species_embed.weight"]
    # the donated trunk is returned exactly as it was handed over
    state = fitted.model.state_dict()
    assert all(torch.equal(state[k], v) for k, v in weights.items())
    # and the one free vector actually moved
    assert float(fitted.model.species_embed.weight.detach().norm()) > 0.05


def test_a_mismatched_donor_is_refused():
    pool = _pool()
    donor = train(_config(use_fm_package=False), pool, checkpoint_every=0)
    with pytest.raises(ValueError, match="init_state_dict"):
        train(
            _embedding_config(),
            pool,
            checkpoint_every=0,
            init_state_dict=donor.model.state_dict(),
        )


# --------------------------------------------------------------------- inference


def _history():
    lon = np.tile(np.linspace(10.0, 10.1, 12), (3, 1))
    lat = np.tile(np.linspace(-30.0, -29.9, 12), (3, 1))
    stamps = (np.tile(np.arange(12) * 21600.0 + 1.7e9, (3, 1))).astype("datetime64[s]")
    return np.stack([lon, lat], axis=-1), stamps


def test_the_predictor_will_not_guess_a_species_row():
    model = MovementFM(
        d_model=16,
        num_layers=1,
        use_fm_package=True,
        use_context_conditioning=False,
        n_species=3,
        species_embed_dim=4,
    )
    with pytest.raises(ValueError, match="species_index is required"):
        MovementFMPredictor(model, device="cpu")
    with pytest.raises(ValueError, match="outside the trained table"):
        MovementFMPredictor(model, device="cpu", species_index=3)
    with pytest.raises(ValueError, match="no species embedding"):
        MovementFMPredictor(
            MovementFM(d_model=16, num_layers=1, use_fm_package=True),
            device="cpu",
            species_index=0,
        )


def test_the_predictor_applies_the_row_it_was_given():
    torch.manual_seed(0)
    model = MovementFM(
        d_model=16,
        num_layers=1,
        use_fm_package=True,
        use_context_conditioning=False,
        n_species=3,
        species_embed_dim=4,
    )
    with torch.no_grad():
        model.species_embed.weight.copy_(torch.tensor([[2.0] * 4, [-2.0] * 4, [0.0] * 4]))
    history, stamps = _history()
    first = MovementFMPredictor(model, device="cpu", species_index=0).predict_batch(history, stamps)
    second = MovementFMPredictor(model, device="cpu", species_index=1).predict_batch(history, stamps)
    assert first.shape == (3, 2)
    assert not np.allclose(first, second)

def test_domain_rows_read_the_matching_source_row_and_nothing_else():
    from types import SimpleNamespace

    from experiments.reporting.tier_e_eval import species_row

    with_table = SimpleNamespace(species_embed=object())
    pool = ["cervus_elaphus", "ciconia_ciconia", "mirounga_leonina"]
    assert species_row(with_table, pool, "ciconia_ciconia", "E1embed") == 1
    assert species_row(with_table, pool, "larus_fuscus", "E1embed", "ciconia_ciconia") == 1
    assert species_row(with_table, pool, "canis_lupus", "E1embed", "cervus_elaphus") == 0
    with pytest.raises(LookupError):
        species_row(with_table, pool, "larus_fuscus", "E1embed")
    with pytest.raises(LookupError):
        species_row(with_table, pool, "larus_fuscus", "E1embed", "not_in_pool")
    assert species_row(SimpleNamespace(species_embed=None), pool, "larus_fuscus", "Z", "ciconia_ciconia") is None
    assert species_row(with_table, ["larus_fuscus"], "larus_fuscus", "E2embed", "ciconia_ciconia") == 0
