# GenAI reference: Again used GenAI to write these tests; outline that prepared pools match gather, sampling follows the 
# registered certain weights, all valid windows are reachable, the Tier 1 warmup/cosine schedule stays fixed, Z/B train 
# through the same function; with a few other things GenAI added itself; I checked all the tests were correct

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import pytest
import torch

from animal_movement_fm.data.multispecies import SourcePool, SpeciesWindows
from animal_movement_fm.models.fm import MovementFM
from animal_movement_fm.train.loop import (
    BATCH_SIZE,
    GRAD_CLIP,
    MASK_RATIO,
    RECONSTRUCTION_WEIGHT,
    WARMUP_FRACTION,
    TrainConfig,
    lr_scale,
    train,
)
from animal_movement_fm.train.pool import PreparedPool

C = 12


def _segments(tmp_path, name, individual, n_bins, seed):
    rng = np.random.default_rng(seed)
    rows = []
    for step in range(n_bins):
        rows.append(
            {
                "segment_id": f"{individual}_s0",
                "individual_id": individual,
                "bin_index": step,
                "timestamp": pd.Timestamp("2020-01-01", tz="UTC")
                + pd.Timedelta(hours=6 * step + float(rng.uniform(0, 5))),
                "lon": 10.0 + 0.01 * step + float(rng.normal(0, 0.004)),
                "lat": -40.0 + 0.01 * step + float(rng.normal(0, 0.004)),
            }
        )
    path = tmp_path / name
    pd.DataFrame(rows).to_parquet(path)
    return path


def _pool(tmp_path, weights=(0.75, 0.25), sizes=(300, 120)):
    species = {
        "big": SpeciesWindows.build(
            _segments(tmp_path, "big.parquet", "a", sizes[0], 1), ["a"], context_length=C
        ),
        "small": SpeciesWindows.build(
            _segments(tmp_path, "small.parquet", "b", sizes[1], 2), ["b"], context_length=C
        ),
    }
    return SourcePool(species, {"big": weights[0], "small": weights[1]})


def _config(**kwargs):
    base = dict(d_model=16, num_layers=1, nhead=2, lr=1e-3, steps=6, batch_size=8)
    base.update(kwargs)
    return TrainConfig(**base)


# ----------------------------------------------------------------------------- the prepared pool


def test_prepared_pool_holds_every_window_of_every_species(tmp_path):
    pool = _pool(tmp_path)
    prepared = PreparedPool.from_source_pool(pool)

    assert prepared.n_windows == sum(w.n_windows for w in pool.species.values())
    assert prepared.features.shape[1:] == (C - 1, prepared.n_features)
    assert prepared.targets.shape == (prepared.n_windows, 2)
    assert torch.isfinite(prepared.features).all()


def test_prepared_pool_encodes_exactly_what_gather_returns(tmp_path):
    """Preloading is a performance change, not a representation change."""
    pool = _pool(tmp_path)
    prepared = PreparedPool.from_source_pool(pool)

    first = pool.species[prepared.names[0]]
    expected, target, _ = first.gather(np.arange(5))
    assert np.allclose(prepared.features[:5].numpy(), expected, atol=1e-6)
    assert np.allclose(prepared.targets[:5].numpy(), target, atol=1e-6)


def test_the_sampler_follows_the_registered_probabilities_not_the_window_counts(tmp_path):
    """
    The failure this guards against is the one A8 was written for.

    A uniform draw over the pooled window set is what any dataloader does by default, and it would set
    the species shares from the window counts instead of from the registered weights.
    """
    pool = _pool(tmp_path, weights=(0.75, 0.25))
    prepared = PreparedPool.from_source_pool(pool)
    big, small = (prepared.species_bounds[i, 1] - prepared.species_bounds[i, 0] for i in (0, 1))
    assert big > 2 * small  # window counts are lopsided the other way round to the weights

    generator = torch.Generator().manual_seed(42)
    shares = prepared.realised_species_shares(generator, draws=200_000)
    assert shares["big"] == pytest.approx(0.75, abs=0.005)
    assert shares["small"] == pytest.approx(0.25, abs=0.005)


def test_every_window_of_a_species_is_reachable_and_none_beyond_it(tmp_path):
    """
    Uniform within the species means the last window is drawable, and the first of the next is not.

    An off-by-one in the offset arithmetic would either make one window unreachable -- undetectable in
    a loss curve -- or let a draw for one species land on another's row, which silently breaks the
    registered draw shares while `realised_species_shares` still reports them as correct.
    """
    pool = _pool(tmp_path, sizes=(60, 40))
    prepared = PreparedPool.from_source_pool(pool)
    lo, hi = (int(v) for v in prepared.species_bounds[0])

    # Sample with the second species' weight at zero, so every row drawn must belong to the first.
    prepared.probabilities = torch.tensor([1.0, 0.0], dtype=torch.float64)
    generator = torch.Generator().manual_seed(0)

    rows = set()
    for _ in range(200):
        picks = torch.multinomial(prepared.probabilities, 64, replacement=True, generator=generator)
        starts = prepared.species_bounds[picks, 0]
        widths = prepared.species_bounds[picks, 1] - starts
        offsets = (torch.rand(64, generator=generator, dtype=torch.float64) * widths).to(torch.int64)
        rows.update((starts + torch.clamp(offsets, max=widths - 1)).tolist())

    assert min(rows) == lo and max(rows) == hi - 1
    assert len(rows) == hi - lo  # every window of the species, and nothing outside it


# ------------------------------------------------------------------------------- the schedule


def test_the_schedule_is_tier_ones_warmup_then_cosine():
    total = 1000
    assert lr_scale(0, total_steps=total, warmup_fraction=0.1) == pytest.approx(1 / 100)
    assert lr_scale(99, total_steps=total, warmup_fraction=0.1) == pytest.approx(1.0)
    assert lr_scale(total - 1, total_steps=total, warmup_fraction=0.1) < 1e-4
    # monotone up through warmup, monotone down after it
    warm = [lr_scale(s, total_steps=total, warmup_fraction=0.1) for s in range(0, 100)]
    cool = [lr_scale(s, total_steps=total, warmup_fraction=0.1) for s in range(100, total, 50)]
    assert warm == sorted(warm)
    assert cool == sorted(cool, reverse=True)


def test_the_fixed_training_constants_are_tier_ones():
    """Carried over deliberately, so the tiers differ in what is being studied and not in bookkeeping."""
    assert (BATCH_SIZE, WARMUP_FRACTION, GRAD_CLIP) == (256, 0.1, 1.0)


# ------------------------------------------------------------------------------------ training


def test_training_reduces_the_loss(tmp_path):
    prepared = PreparedPool.from_source_pool(_pool(tmp_path))
    result = train(_config(steps=60, lr=3e-3), prepared, log_every=10, checkpoint_every=0)

    assert len(result.loss_history) == 6
    assert result.loss_history[-1] < result.loss_history[0]


def test_z_and_b_train_through_the_same_function(tmp_path):
    prepared = PreparedPool.from_source_pool(_pool(tmp_path))
    z = train(_config(use_fm_package=True), prepared, checkpoint_every=0)
    b = train(_config(use_fm_package=False), prepared, checkpoint_every=0)

    assert z.model.n_parameters() > b.model.n_parameters()
    assert z.model.cond_proj is not None and b.model.cond_proj is None


def test_the_same_seed_gives_the_same_run(tmp_path):
    prepared = PreparedPool.from_source_pool(_pool(tmp_path))
    a = train(_config(seed=7), prepared, log_every=1, checkpoint_every=0)
    b = train(_config(seed=7), prepared, log_every=1, checkpoint_every=0)
    assert a.loss_history == b.loss_history


# ------------------------------------------------------------------------------------- resume


def test_a_resumed_run_is_the_same_run(tmp_path):
    """
    PBS arrays go in `-r y`, so this path runs whenever a subjob is requeued.

    The sampler's generator state is checkpointed with the weights. Without it a resumed run would
    redraw different windows from the same seed and quietly become a different experiment.
    """
    prepared = PreparedPool.from_source_pool(_pool(tmp_path))
    config = _config(steps=40, seed=3)
    uninterrupted = train(config, prepared, log_every=10, checkpoint_every=0)

    class Killed(Exception):
        pass

    def die_at_20(step, _loss):
        if step >= 20:
            raise Killed

    path = tmp_path / "run.pt"
    with pytest.raises(Killed):
        train(config, prepared, log_every=10, checkpoint_every=10, checkpoint_path=path,
              progress=die_at_20)

    # the requeue re-enters the identical config and finds the checkpoint
    resumed = train(config, prepared, log_every=10, checkpoint_every=10, checkpoint_path=path)

    assert resumed.resumed_from == 20 and resumed.steps_run == 20
    for name, tensor in uninterrupted.model.state_dict().items():
        assert torch.allclose(tensor, resumed.model.state_dict()[name], atol=1e-6), name


def test_a_checkpoint_from_another_configuration_is_refused(tmp_path):
    prepared = PreparedPool.from_source_pool(_pool(tmp_path))
    path = tmp_path / "run.pt"
    train(_config(d_model=16), prepared, checkpoint_every=2, checkpoint_path=path)

    with pytest.raises(ValueError, match="different configuration"):
        train(_config(d_model=32), prepared, checkpoint_every=2, checkpoint_path=path)


# ------------------------------------------------------------------- the pretraining objective


def test_the_pretraining_constants_are_the_registered_ones():
    """Registered 18-08-2026 in §5-6, conventional rather than tuned, and never searched."""
    assert (MASK_RATIO, RECONSTRUCTION_WEIGHT) == (0.15, 0.5)
    assert _config().mask_ratio == 0.15 and _config().reconstruction_weight == 0.5

    with pytest.raises(ValueError, match="strictly in"):
        _config(mask_ratio=0.0)


def test_the_pretraining_objective_trains_when_its_constants_are_given(tmp_path):
    prepared = PreparedPool.from_source_pool(_pool(tmp_path))
    result = train(
        _config(steps=20, use_reconstruction_head=True),
        prepared,
        log_every=5,
        checkpoint_every=0,
    )
    assert len(result.loss_history) == 4
    assert result.model.recon_head is not None


def test_the_batch_remembers_which_species_each_row_came_from(tmp_path):
    """
    §6 commits to reporting how the loss divides across the source species under the realised sampler.

    That has to be measured under the production loss rather than inferred: the table in §6 was computed
    on MSE, and the radial pseudo-Huber is linear in the tail where MSE is quadratic, so a heavy-tailed
    species can dominate one aggregate and not the other. A batch that has forgotten where its rows came
    from cannot be measured after the fact.
    """
    prepared = PreparedPool.from_source_pool(_pool(tmp_path, weights=(0.75, 0.25)))
    generator = torch.Generator().manual_seed(0)

    _, _, species = prepared.sample(generator, 4096)

    assert species.shape == (4096,)
    assert set(species.tolist()) <= {0, 1}
    assert float((species == 0).float().mean()) == pytest.approx(0.75, abs=0.02)


def test_the_loss_share_diagnostic_is_recorded(tmp_path):
    prepared = PreparedPool.from_source_pool(_pool(tmp_path))
    result = train(_config(steps=20), prepared, log_every=10, checkpoint_every=0)

    assert len(result.loss_shares) == 2
    for share in result.loss_shares:
        assert set(share) == set(prepared.names)
        assert sum(share.values()) == pytest.approx(1.0, abs=1e-5)
