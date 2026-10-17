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


# GenAI reference: tests for the 1 h study's pools (step 3.3, 10-10), from the behaviours I specified: sample's draws
# unchanged, sample_horizon on the same draws, targets divided by k_h, k_h one bird's np.median and independent of C,
# the streaming pool equal to the prepared one on the core, and core_matches catching a perturbed core.
from dataclasses import replace  # noqa: E402

from animal_movement_fm.data.multispecies import HorizonWindows  # noqa: E402
from animal_movement_fm.train.pool import (  # noqa: E402
    StreamingPool,
    core_matches,
    horizon_k,
    horizon_ratios,
    weighted_median,
)

HOUR = 3600
HORIZONS = (1, 6, 24)


def _hourly(tmp_path, name, individual, bins, seed, step_deg=0.002):
    rng = np.random.default_rng(seed)
    rows = [
        {"segment_id": f"{individual}_s0", "individual_id": individual, "bin_index": int(b),
         "timestamp": pd.Timestamp("2020-01-01", tz="UTC") + pd.Timedelta(seconds=HOUR * int(b) + int(rng.integers(0, 3000))),
         "lon": 10.0 + step_deg * b + rng.normal(0, 2e-4), "lat": 50.0 + rng.normal(0, 2e-4)}
        for b in bins
    ]
    path = tmp_path / name
    pd.DataFrame(rows).to_parquet(path)
    return path


def _horizon_pool(tmp_path, context=12, weights=(0.6, 0.4)):
    # holes in both histories, and b moves five times as far per bin, so the two species' r_h differ
    paths = {
        "a": _hourly(tmp_path, "a.parquet", "a", np.concatenate(([0, 1, 5], np.arange(10, 140))), 1),
        "b": _hourly(tmp_path, "b.parquet", "b", np.setdiff1d(np.arange(110), np.arange(40, 44)), 2, 0.01),
    }
    species = {
        name: HorizonWindows.build(path, [name], context=context, core=12, horizons=HORIZONS, bin_seconds=HOUR)
        for name, path in paths.items()
    }
    return SourcePool(species, dict(zip(("a", "b"), weights)))


def test_sample_draws_are_the_registered_rule_written_out(tmp_path):
    """The original sample(), inline: factoring the draws out must not move a single row."""
    prepared = PreparedPool.from_source_pool(_pool(tmp_path))
    features, targets, picks = prepared.sample(torch.Generator().manual_seed(11), 64)

    generator = torch.Generator().manual_seed(11)
    expected_picks = torch.multinomial(prepared.probabilities, 64, replacement=True, generator=generator)
    starts = prepared.species_bounds[expected_picks, 0]
    widths = prepared.species_bounds[expected_picks, 1] - starts
    offsets = (torch.rand(64, generator=generator, dtype=torch.float64) * widths.to(torch.float64)).to(torch.int64)
    rows = starts + torch.clamp(offsets, max=widths - 1)
    assert torch.equal(picks, expected_picks)
    assert torch.equal(features, prepared.features[rows]) and torch.equal(targets, prepared.targets[rows])


def test_sample_horizon_draws_the_rows_sample_draws(tmp_path):
    pool = _horizon_pool(tmp_path)
    k = horizon_k(pool)
    prepared = PreparedPool.from_source_pool(pool, horizon_k=k)
    plain = PreparedPool(prepared.names, prepared.features, prepared.targets, prepared.species_bounds,
                         prepared.probabilities)
    features, token_mask, targets, target_mask, picks = prepared.sample_horizon(torch.Generator().manual_seed(3), 64)
    old_features, old_targets, old_picks = plain.sample(torch.Generator().manual_seed(3), 64)
    assert torch.equal(picks, old_picks) and torch.equal(features, old_features) and torch.equal(targets, old_targets)
    assert token_mask.shape == (64, 11) and targets.shape == (64, 3, 2) and target_mask.shape == (64, 3)


def test_horizon_pool_stores_gather_with_targets_divided_by_k(tmp_path):
    pool = _horizon_pool(tmp_path)
    k = horizon_k(pool)
    prepared = PreparedPool.from_source_pool(pool, horizon_k=k)
    assert prepared.horizon_k == k
    first = pool.species[prepared.names[0]]
    features, token_mask, targets, target_mask, _ = first.gather(np.arange(5))
    assert torch.equal(prepared.features[:5], torch.as_tensor(features, dtype=torch.float32))
    assert torch.equal(prepared.token_mask[:5], torch.as_tensor(token_mask))
    assert torch.equal(prepared.target_mask[:5], torch.as_tensor(target_mask))
    expected = targets / np.asarray(k)[None, :, None]
    assert torch.equal(prepared.targets[:5], torch.as_tensor(expected, dtype=torch.float32))
    assert (prepared.targets[~prepared.target_mask] == 0).all()


def test_horizon_and_plain_pools_refuse_each_others_calls(tmp_path):
    pool = _horizon_pool(tmp_path)
    with pytest.raises(ValueError, match="horizon_k is required"):
        PreparedPool.from_source_pool(pool)
    with pytest.raises(ValueError, match="one finite positive k per horizon"):
        PreparedPool.from_source_pool(pool, horizon_k=(1.0, 2.0))
    prepared = PreparedPool.from_source_pool(pool, horizon_k=(1.0, 2.0, 3.0))
    with pytest.raises(ValueError, match="sample_horizon"):
        prepared.sample(torch.Generator().manual_seed(0), 8)
    with pytest.raises(ValueError, match="horizon_k is required"):
        PreparedPool.from_source_pool(_pool(tmp_path), horizon_k=(1.0, 2.0, 3.0))
    with pytest.raises(ValueError, match="HorizonWindows"):
        PreparedPool.from_source_pool(_pool(tmp_path)).sample_horizon(torch.Generator().manual_seed(0), 8)


def test_weighted_median_reduces_to_np_median():
    rng = np.random.default_rng(0)
    for n in (1, 2, 7, 10, 1001, 1000):
        values = rng.lognormal(size=n)
        assert weighted_median(values, np.full(n, 0.37 / n)) == np.median(values)
    values, counts = np.array([3.0, 1.0, 2.0, 5.0]), np.array([1, 4, 2, 3])
    assert weighted_median(values, counts) == np.median(np.repeat(values, counts))
    assert weighted_median(values, np.array([1, 1, 1, 7])) == 5.0


def test_horizon_k_of_one_species_is_its_median_r(tmp_path):
    windows = _horizon_pool(tmp_path).species["a"]
    _, _, targets, target_mask, _ = windows.gather(np.arange(windows.n_windows))
    r = np.linalg.norm(targets, axis=-1)
    k = horizon_k(SourcePool({"a": windows}, {"a": 1.0}))
    assert k == tuple(float(np.median(r[target_mask[:, j], j])) for j in range(3))


def test_horizon_k_weights_each_window_p_over_n_and_ignores_context(tmp_path):
    pool = _horizon_pool(tmp_path, weights=(0.6, 0.4))
    ratios = {name: horizon_ratios(pool.species[name]) for name in pool.names}
    expected = tuple(
        weighted_median(
            np.concatenate([ratios[n][j] for n in pool.names]),
            np.concatenate([np.full(len(ratios[n][j]), p / pool.species[n].n_windows)
                            for n, p in zip(pool.names, pool.probabilities)]),
        )
        for j in range(3)
    )
    assert horizon_k(pool) == expected
    assert horizon_k(_horizon_pool(tmp_path, weights=(0.1, 0.9))) != expected  # the weights matter here
    assert horizon_k(_horizon_pool(tmp_path, context=48)) == expected


def test_streaming_pool_equals_the_prepared_pool_on_the_core(tmp_path):
    main = _horizon_pool(tmp_path)
    k = horizon_k(main)
    prepared = PreparedPool.from_source_pool(main, horizon_k=k)
    same = StreamingPool.from_source_pool(main, horizon_k=k).sample_horizon(torch.Generator().manual_seed(9), 128)
    expected = prepared.sample_horizon(torch.Generator().manual_seed(9), 128)
    assert all(torch.equal(x, y) for x, y in zip(same, expected))

    streamed = StreamingPool.from_source_pool(_horizon_pool(tmp_path, context=48), horizon_k=k)
    assert streamed.n_windows == prepared.n_windows and streamed.n_features == prepared.n_features
    features, token_mask, targets, target_mask, picks = streamed.sample_horizon(torch.Generator().manual_seed(9), 128)
    assert features.shape == (128, 47, 12) and token_mask.shape == (128, 47)
    assert torch.equal(features[:, -11:], expected[0]) and token_mask[:, -11:].all()
    assert torch.equal(targets, expected[2]) and torch.equal(target_mask, expected[3]) and torch.equal(picks, expected[4])
    assert not token_mask.all()  # the holes reach the extended history
    assert (features[~token_mask] == 0).all()


def test_core_matches_across_context_and_catches_a_perturbed_core(tmp_path):
    main = _horizon_pool(tmp_path)
    ids = {name: np.arange(main.species[name].n_windows) for name in main.names}
    for context in (24, 48, 96):
        assert core_matches(main, _horizon_pool(tmp_path, context=context), ids) == []

    ablation = _horizon_pool(tmp_path, context=48)
    a = ablation.species["a"]
    lon = a.lon.copy()
    lon[a.anchors[3] - 4] += 1e-5  # one core fix of one window, about a metre
    ablation.species["a"] = replace(a, lon=lon)
    failures = core_matches(main, ablation, ids)
    assert "a: core tokens differ" in failures and not any(f.startswith("b:") for f in failures)
    assert core_matches(main, main, {"a": ids["a"][:2], "b": ids["b"][:2]}) == []
    assert core_matches(_horizon_pool(tmp_path, context=24), ablation, ids)[0].startswith("a: main context 24")


# GenAI reference: tests for the horizon configuration and loop (step 3.6, 10-10), from the behaviours I specified: an
# old configuration rebuilds and trains exactly as before (losses pinned from the loop before the 3.6 edit, Mac, torch
# 2.5.1, CPU, one thread), a horizon configuration trains on prepared and streaming pools, a padded history reaches the
# model as `observed`, and every mismatch between configuration and pool is refused.
import json  # noqa: E402
from dataclasses import asdict  # noqa: E402

PRE_HORIZON_LOSSES = {
    "B": (dict(use_fm_package=False), [0.7962692975997925, 0.4073374569416046, 0.6680665612220764]),
    "Z_recon": (dict(use_reconstruction_head=True), [1.4976184368133545, 0.9505036473274231, 0.8956759572029114]),
    "LSTM": (dict(use_fm_package=False, trunk="lstm"), [0.5296225547790527, 0.3056190013885498, 0.34819459915161133]),
    "MoE": (dict(use_fm_package=False, trunk="moe"), [1.3469558954238892, 1.0312877893447876, 1.238032341003418]),
}


def _old_dict():
    """A configuration as records written before the horizon fields hold it."""
    saved = asdict(_config(trunk="lstm", use_fm_package=False))
    for key in ("horizons", "history", "observed_flag"):
        saved.pop(key)
    return saved


def test_an_old_config_dict_rebuilds_identically():
    config = TrainConfig(**_old_dict())
    assert config == _config(trunk="lstm", use_fm_package=False)
    assert (config.horizons, config.history, config.observed_flag) == ((), 0, False)
    torch.manual_seed(0)
    built = config.build()
    torch.manual_seed(0)
    direct = MovementFM(input_dim=12, d_model=16, nhead=4, num_layers=1, dim_feedforward=32, dropout=0.1,
                        use_fm_package=False, trunk="lstm")
    assert built.n_horizons == 0 and built.input_proj.in_features == 12
    assert all(torch.equal(a, b) for a, b in zip(built.state_dict().values(), direct.state_dict().values()))


@pytest.mark.parametrize("name", sorted(PRE_HORIZON_LOSSES))
def test_old_configs_reproduce_the_pre_horizon_losses_bitwise(tmp_path, name):
    kwargs, expected = PRE_HORIZON_LOSSES[name]
    result = train(_config(steps=3, **kwargs), PreparedPool.from_source_pool(_pool(tmp_path)), log_every=1,
                   checkpoint_every=0)
    assert result.loss_history == expected
    assert result.head_loss_history == []


def test_a_horizon_config_round_trips_through_json():
    config = _config(horizons=(1, 6, 24), history=48, trunk="lstm", use_fm_package=False, observed_flag=True)
    assert TrainConfig(**json.loads(json.dumps(asdict(config)))) == config
    model = config.build()
    assert model.n_horizons == 3 and model.observed_flag and model.input_proj.in_features == 13


def test_horizon_configs_refuse_what_they_cannot_train():
    plain = dict(use_fm_package=False, horizons=(1, 6, 24))
    with pytest.raises(ValueError, match="reconstruction"):
        _config(horizons=(1, 6, 24), use_reconstruction_head=True)
    with pytest.raises(ValueError, match="distinct positive"):
        _config(**plain | dict(horizons=(1, 1, 24)))
    with pytest.raises(ValueError, match="set horizons"):
        _config(use_fm_package=False, history=48)
    with pytest.raises(ValueError, match="must exceed"):
        _config(**plain, history=12)
    with pytest.raises(ValueError, match="cannot mask"):
        _config(horizons=(1, 6, 24), history=48)
    with pytest.raises(ValueError, match="cannot mask"):
        _config(**plain, history=48, trunk="moe")
    with pytest.raises(ValueError, match="observed_flag"):
        _config(**plain, history=48, trunk="lstm")
    with pytest.raises(ValueError, match="observed_flag"):
        _config(**plain, trunk="lstm", observed_flag=True)
    with pytest.raises(ValueError, match="observed_flag"):
        _config(**plain, history=48, observed_flag=True)


def test_horizon_runs_refuse_a_mismatched_pool(tmp_path):
    main = _horizon_pool(tmp_path)
    k = horizon_k(main)
    prepared = PreparedPool.from_source_pool(main, horizon_k=k)
    horizon = _config(use_fm_package=False, horizons=(1, 6, 24), steps=2)
    with pytest.raises(ValueError, match="needs a horizon pool"):
        train(horizon, PreparedPool.from_source_pool(_pool(tmp_path)), checkpoint_every=0)
    with pytest.raises(ValueError, match="needs a horizon pool"):
        train(_config(use_fm_package=False, steps=2), prepared, checkpoint_every=0)
    with pytest.raises(ValueError, match="3 horizons, the configuration 2"):
        train(_config(use_fm_package=False, horizons=(1, 6), steps=2), prepared, checkpoint_every=0)
    with pytest.raises(ValueError, match="history 0 needs 11"):
        train(horizon, StreamingPool.from_source_pool(_horizon_pool(tmp_path, context=24), horizon_k=k),
              checkpoint_every=0)
    with pytest.raises(ValueError, match="history 48 needs 47"):
        train(_config(use_fm_package=False, horizons=(1, 6, 24), history=48, steps=2), prepared, checkpoint_every=0)


@pytest.mark.parametrize("trunk", ["transformer", "lstm"])
def test_horizon_configs_train_on_prepared_and_streaming_pools(tmp_path, trunk):
    k = horizon_k(_horizon_pool(tmp_path))
    for context, make in ((12, PreparedPool), (24, StreamingPool)):
        pool = make.from_source_pool(_horizon_pool(tmp_path, context=context), horizon_k=k)
        padded = context > 12
        config = _config(use_fm_package=False, trunk=trunk, horizons=(1, 6, 24), history=context if padded else 0,
                         observed_flag=padded and trunk == "lstm", steps=6)
        result = train(config, pool, log_every=2, checkpoint_every=0)
        assert result.applied_updates == 6 and result.skipped_steps == 0
        assert len(result.head_loss_history) == 3 and all(len(h) == 3 for h in result.head_loss_history)
        assert np.isfinite(result.head_loss_history).all() and np.isfinite(result.loss_history).all()
        shares = result.loss_shares[-1]
        assert set(shares) == {"a", "b"} and sum(shares.values()) == pytest.approx(1.0, abs=1e-5)


def test_a_padded_history_reaches_the_transformer_as_observed(tmp_path):
    """Garbage in the padded tokens changes nothing: the mask is applied, not just carried."""
    pool = _horizon_pool(tmp_path, context=24)
    prepared = PreparedPool.from_source_pool(pool, horizon_k=horizon_k(pool))
    assert not prepared.token_mask.all()
    config = _config(use_fm_package=False, horizons=(1, 6, 24), history=24, steps=4, dropout=0.0)
    clean = train(config, prepared, log_every=1, checkpoint_every=0)
    garbage = replace(prepared, features=torch.where(prepared.token_mask[..., None], prepared.features,
                                                     torch.full_like(prepared.features, 500.0)))
    noisy = train(config, garbage, log_every=1, checkpoint_every=0)
    assert np.allclose(clean.loss_history, noisy.loss_history, atol=1e-6, rtol=0)
    assert np.allclose(clean.head_loss_history, noisy.head_loss_history, atol=1e-6, rtol=0)

from animal_movement_fm.data import era5  # noqa: E402
from animal_movement_fm.train.pool import env_stats  # noqa: E402


def _env_species(tmp_path, name, n_bins, d_lon, d_lat, seed, missing_bin=None):
    """A straight track (exactly east or north) on a 1 h grid and its synthetic ERA5 file; with and without env."""
    bins = np.arange(n_bins)
    path = tmp_path / f"{name}.parquet"
    pd.DataFrame({"segment_id": f"{name}_s0", "individual_id": name, "bin_index": bins,
                  "timestamp": [pd.Timestamp("2020-01-01", tz="UTC") + pd.Timedelta(hours=int(b)) for b in bins],
                  "lon": 10.0 + d_lon * bins, "lat": 50.0 + d_lat * bins}).to_parquet(path)
    values = np.random.default_rng(seed).normal(50.0, 10.0, size=(n_bins, len(era5.RAW))).astype(np.float32)
    if missing_bin is not None:
        values[missing_bin, era5.RAW.index("blh")] = np.nan
    env = pd.DataFrame({"segment_id": f"{name}_s0", "bin_index": bins,
                        **{v: values[:, i] for i, v in enumerate(era5.RAW)}})
    env.to_parquet(tmp_path / f"{name}_env.parquet", index=False)
    build = dict(context=12, core=12, horizons=HORIZONS, bin_seconds=HOUR)
    return (HorizonWindows.build(path, [name], **build, env_parquet=tmp_path / f"{name}_env.parquet"),
            HorizonWindows.build(path, [name], **build), values.astype(float))


def test_env_stats_are_the_p_over_n_weighted_moments_of_observed_tokens(tmp_path):
    # a moves east (tail = u, cross = v), b north (tail = v, cross = -u); b's bin 11 lacks blh
    a, a_off, a_env = _env_species(tmp_path, "a", 16, 0.01, 0.0, 1)
    b, b_off, b_env = _env_species(tmp_path, "b", 14, 0.0, 0.01, 2, missing_bin=11)
    pool = SourcePool({"a": a, "b": b}, {"a": 0.6, "b": 0.4})
    assert (a.n_windows, b.n_windows) == (4, 2)

    def wind(x, east):
        u10, v10, u100, v100 = x[:4]
        return [u10, v10, u100, v100] if east else [v10, -u10, v100, -u100]

    tokens, weights = [], []
    for windows, values, p, east in ((a, a_env, 0.6, True), (b, b_env, 0.4, False)):
        for anchor in windows.anchors:
            for row in range(anchor - 10, anchor + 1):
                if np.isfinite(values[row]).all():
                    tokens.append(np.concatenate([values[row], wind(values[row], east)]))
                    weights.append(p / windows.n_windows)
    tokens, weights = np.array(tokens), np.array(weights)
    assert len(tokens) == 6 * 11 - 2  # bin 11 is a token in both of b's windows
    mean = weights @ tokens / weights.sum()
    sd = np.sqrt(weights @ (tokens - mean) ** 2 / weights.sum())

    got_mean, got_sd = env_stats(pool)
    np.testing.assert_allclose(got_mean, mean, rtol=1e-12)
    np.testing.assert_allclose(got_sd, sd, rtol=1e-9)
    chunked = env_stats(pool, chunk=1)
    np.testing.assert_allclose(chunked[0], got_mean, rtol=1e-12)
    np.testing.assert_allclose(chunked[1], got_sd, rtol=1e-12)
    assert not np.allclose(env_stats(SourcePool({"a": a, "b": b}, {"a": 0.1, "b": 0.9}))[0], got_mean)

    c, c_off, _ = _env_species(tmp_path, "c", 60, 0.01, 0.002, 3)  # long enough for 24 h targets
    d, d_off, _ = _env_species(tmp_path, "d", 50, 0.003, 0.01, 4, missing_bin=30)
    with_env = SourcePool({"c": c, "d": d}, {"c": 0.6, "d": 0.4})
    assert horizon_k(with_env) == horizon_k(SourcePool({"c": c_off, "d": d_off}, {"c": 0.6, "d": 0.4}))
    with pytest.raises(ValueError, match="no covariates"):
        env_stats(SourcePool({"a": a_off, "b": b}, {"a": 0.6, "b": 0.4}))


from animal_movement_fm.data.enu import N_FEATURES  # noqa: E402
from animal_movement_fm.models.xgboost import WindowXGBoost  # noqa: E402

PRE_ENV_HORIZON_LOSSES = {
    "transformer": [1.0093885660171509, 0.9137147068977356, 0.9078122973442078],
    "lstm": [0.5896711945533752, 0.535841166973114, 0.538352906703949],
}
ENV = era5.N_ENV_CHANNELS


def _env_pool(tmp_path):
    """Two straight-ish tracks with ERA5 (one missing token), standardised on their own pool's statistics."""
    c, c_off, _ = _env_species(tmp_path, "c", 80, 0.01, 0.002, 3)
    d, d_off, _ = _env_species(tmp_path, "d", 70, 0.003, 0.01, 4, missing_bin=30)
    weights = {"c": 0.6, "d": 0.4}
    stats = env_stats(SourcePool({"c": c, "d": d}, weights))
    on = SourcePool({"c": replace(c, env_stats=stats), "d": replace(d, env_stats=stats)}, weights)
    return on, SourcePool({"c": c_off, "d": d_off}, weights)


def test_a_pre_env_config_dict_rebuilds_identically():
    saved = asdict(_config(trunk="lstm", use_fm_package=False, horizons=(1, 6, 24)))
    assert saved.pop("env_channels") == 0
    config = TrainConfig(**saved)
    assert config == _config(trunk="lstm", use_fm_package=False, horizons=(1, 6, 24))
    torch.manual_seed(0)
    built = config.build()
    torch.manual_seed(0)
    explicit = config.build(input_dim=N_FEATURES)
    assert built.input_proj.in_features == 12
    assert all(torch.equal(a, b) for a, b in zip(built.state_dict().values(), explicit.state_dict().values()))


@pytest.mark.parametrize("trunk", sorted(PRE_ENV_HORIZON_LOSSES))
def test_horizon_configs_reproduce_the_pre_env_losses_bitwise(tmp_path, trunk):
    pool = _horizon_pool(tmp_path)
    prepared = PreparedPool.from_source_pool(pool, horizon_k=horizon_k(pool))
    result = train(_config(use_fm_package=False, trunk=trunk, horizons=(1, 6, 24), steps=3), prepared, log_every=1,
                   checkpoint_every=0)
    assert result.loss_history == PRE_ENV_HORIZON_LOSSES[trunk]


@pytest.mark.parametrize("trunk", ["transformer", "lstm"])
def test_env_configs_train_on_a_27_channel_pool(tmp_path, trunk):
    on, off = _env_pool(tmp_path)
    k = horizon_k(off)
    assert horizon_k(on) == k
    prepared = PreparedPool.from_source_pool(on, horizon_k=k)
    plain = PreparedPool.from_source_pool(off, horizon_k=k)
    assert prepared.n_features == N_FEATURES + ENV == 27
    assert torch.equal(prepared.features[..., :N_FEATURES], plain.features)
    config = _config(use_fm_package=False, trunk=trunk, horizons=(1, 6, 24), env_channels=ENV, steps=4)
    assert TrainConfig(**json.loads(json.dumps(asdict(config)))) == config
    assert config.build().input_proj.in_features == 27
    result = train(config, prepared, log_every=1, checkpoint_every=0)
    assert result.applied_updates == 4 and result.skipped_steps == 0
    assert np.isfinite(result.loss_history).all() and np.isfinite(result.head_loss_history).all()
    assert result.model.input_proj.in_features == 27
    # the covariates reach the model: zeroing them changes its predictions
    x = prepared.features[:32]
    result.model.eval()
    with torch.no_grad():
        blank = x.clone()
        blank[..., N_FEATURES:] = 0
        assert not torch.equal(result.model(x), result.model(blank))


def test_xgb_fits_on_27_channel_features(tmp_path):
    on, _ = _env_pool(tmp_path)
    windows = on.species["c"].for_horizon(6)
    features, token_mask, targets, _, _ = windows.gather(np.arange(windows.n_windows))
    assert features.shape[1:] == (11, 27) and token_mask.all()
    x = np.where(token_mask[..., None], features, np.nan).astype(np.float32)  # the 1 h driver's rule; nothing is NaN
    model = WindowXGBoost(n_estimators=5).fit(x, targets[:, 1])
    assert model.n_features_ == 11 * 27 and model.predict(x).shape == (len(x), 2)
    with pytest.raises(ValueError, match="flattened features"):
        model.predict(x[..., :N_FEATURES])


def test_channels_that_disagree_with_the_config_are_refused(tmp_path):
    on, off = _env_pool(tmp_path)
    k = horizon_k(off)
    env_config = _config(use_fm_package=False, horizons=(1, 6, 24), env_channels=ENV, steps=2)
    with pytest.raises(ValueError, match="12 channels per token; env_channels 15 needs 27"):
        train(env_config, PreparedPool.from_source_pool(off, horizon_k=k), checkpoint_every=0)
    with pytest.raises(ValueError, match="27 channels per token; env_channels 0 needs 12"):
        train(replace(env_config, env_channels=0), PreparedPool.from_source_pool(on, horizon_k=k), checkpoint_every=0)
    with pytest.raises(ValueError, match="env_channels 14 needs 26"):
        train(replace(env_config, env_channels=14), PreparedPool.from_source_pool(on, horizon_k=k), checkpoint_every=0)


def test_env_configs_refuse_what_they_cannot_take():
    plain = dict(use_fm_package=False, horizons=(1, 6, 24), env_channels=ENV)
    with pytest.raises(ValueError, match="padded history"):
        _config(**plain, history=48)
    with pytest.raises(ValueError, match="padded history"):
        _config(**plain, history=48, trunk="lstm", observed_flag=True)
    with pytest.raises(ValueError, match="no FM package"):
        _config(**plain | dict(use_fm_package=True))
    with pytest.raises(ValueError, match="no FM package"):
        _config(**plain, trunk="moe")
    with pytest.raises(ValueError, match="no FM package"):
        _config(use_fm_package=False, use_reconstruction_head=True, env_channels=ENV)
    with pytest.raises(ValueError, match="negative"):
        _config(**plain | dict(env_channels=-1))


def test_a_pre_env_training_checkpoint_resumes_only_without_covariates(tmp_path):
    pool = _horizon_pool(tmp_path)
    prepared = PreparedPool.from_source_pool(pool, horizon_k=horizon_k(pool))
    config = _config(use_fm_package=False, horizons=(1, 6, 24), steps=8, seed=2)
    uninterrupted = train(config, prepared, log_every=4, checkpoint_every=0)

    class Killed(Exception):
        pass

    def die_at_4(step, _loss):
        if step >= 4:
            raise Killed

    path = tmp_path / "run.pt"
    with pytest.raises(Killed):
        train(config, prepared, log_every=4, checkpoint_every=4, checkpoint_path=path, progress=die_at_4)
    state = torch.load(path, weights_only=False)
    assert state["config"].pop("env_channels") == 0  # as a checkpoint written before E4 holds it
    torch.save(state, path)

    on, _ = _env_pool(tmp_path)  # 27 channels, so the run reaches the resume check
    with pytest.raises(ValueError, match="different configuration"):
        train(replace(config, env_channels=ENV), PreparedPool.from_source_pool(on, horizon_k=horizon_k(on)),
              checkpoint_every=4, checkpoint_path=path)
    with pytest.raises(ValueError, match="different configuration"):
        train(replace(config, lr=2e-3), prepared, checkpoint_every=4, checkpoint_path=path)
    resumed = train(config, prepared, log_every=4, checkpoint_every=4, checkpoint_path=path)
    assert resumed.resumed_from == 4 and resumed.steps_run == 4
    for name, tensor in uninterrupted.model.state_dict().items():
        assert torch.allclose(tensor, resumed.model.state_dict()[name], atol=1e-6), name
    assert torch.load(path, weights_only=False)["config"]["env_channels"] == 0  # rewritten whole from here on


# GenAI reference: I specified the repair probe's config tests etc
from animal_movement_fm.models.fm import signed_log1p  # noqa: E402
from animal_movement_fm.train.loop import RESUME_DEFAULTS  # noqa: E402
from animal_movement_fm.train.loss import horizon_huber  # noqa: E402

REPAIR_FIELDS = {"input_scale": "sqrt_d", "norm_first": False, "step_transform": "none"}
# recorded on 17-10 with the code before the repair fields existed: 3 steps, C = 24, StreamingPool
PRE_REPAIR_HISTORY_LOSSES = {
    "transformer": [1.0384384393692017, 0.9649322628974915, 0.9255337715148926],
    "lstm": [0.3777693212032318, 0.3303797245025635, 0.3404565155506134],
}


def _plain(**kwargs):
    return _config(use_fm_package=False, horizons=(1, 6, 24), **kwargs)


@pytest.mark.parametrize("trunk", sorted(PRE_REPAIR_HISTORY_LOSSES))
def test_defaults_reproduce_the_pre_repair_losses_on_a_padded_history_bitwise(tmp_path, trunk):
    pool = _horizon_pool(tmp_path, context=24)
    streamed = StreamingPool.from_source_pool(pool, horizon_k=horizon_k(pool))
    config = _plain(trunk=trunk, history=24, observed_flag=trunk == "lstm", steps=3)
    assert {name: getattr(config, name) for name in REPAIR_FIELDS} == REPAIR_FIELDS
    assert train(config, streamed, log_every=1, checkpoint_every=0).loss_history == PRE_REPAIR_HISTORY_LOSSES[trunk]


@pytest.mark.parametrize("trunk", ["transformer", "lstm"])
def test_a_pre_repair_config_dict_rebuilds_identically(trunk):
    saved = asdict(_plain(trunk=trunk))
    for name, default in REPAIR_FIELDS.items():
        assert saved.pop(name) == default
    config = TrainConfig(**saved)
    assert config == _plain(trunk=trunk)
    torch.manual_seed(0)
    built = config.build()
    torch.manual_seed(0)
    explicit = MovementFM(input_dim=12, d_model=16, nhead=2, num_layers=1, dim_feedforward=32, dropout=0.1,
                          use_fm_package=False, trunk=trunk, n_horizons=3)
    assert list(built.state_dict()) == list(explicit.state_dict())
    assert all(torch.equal(a, b) for a, b in zip(built.state_dict().values(), explicit.state_dict().values()))
    assert built.encoder.norm is None if trunk == "transformer" else True


def _inputs(n=6, tokens=11, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(n, tokens, 12, generator=generator) * 50


def test_input_scale_none_drops_exactly_the_factor_of_sqrt_d_model():
    torch.manual_seed(3)
    scaled = _plain(d_model=16).build().eval()
    torch.manual_seed(3)
    plain = _plain(d_model=16, input_scale="none").build().eval()
    assert list(scaled.state_dict()) == list(plain.state_dict())
    assert all(torch.equal(a, b) for a, b in zip(scaled.state_dict().values(), plain.state_dict().values()))
    seen = {}
    for name, model in (("scaled", scaled), ("plain", plain)):
        model.pos_encoder.register_forward_hook(lambda m, args, out, name=name: seen.__setitem__(name, args[0]))
        with torch.no_grad():
            model(_inputs())
    torch.testing.assert_close(seen["scaled"], seen["plain"] * 4.0)  # sqrt(16)


def test_norm_first_sets_every_layer_and_adds_only_the_final_norm():
    torch.manual_seed(0)
    post = _plain(num_layers=3).build()
    torch.manual_seed(0)
    pre = _plain(num_layers=3, norm_first=True).build()
    assert all(not layer.norm_first for layer in post.encoder.layers) and post.encoder.norm is None
    assert all(layer.norm_first for layer in pre.encoder.layers) and isinstance(pre.encoder.norm, torch.nn.LayerNorm)
    assert set(pre.state_dict()) - set(post.state_dict()) == {"encoder.norm.weight", "encoder.norm.bias"}
    assert all(torch.equal(post.state_dict()[k], pre.state_dict()[k]) for k in post.state_dict())
    with torch.no_grad():
        assert not torch.allclose(post.eval()(_inputs()), pre.eval()(_inputs()))


def test_signed_log1p_is_odd_monotone_and_touches_only_the_two_step_channels():
    x = _inputs(n=4)
    x[:, :3] = 0.0  # masked tokens are zero and must stay zero
    y = signed_log1p(x)
    assert torch.equal(y[..., 2:], x[..., 2:])
    assert torch.equal(y[:, :3], x[:, :3])
    torch.testing.assert_close(signed_log1p(-x)[..., :2], -y[..., :2])
    grid = torch.linspace(-1000, 1000, 2001)
    values = signed_log1p(torch.stack([grid, grid] + [torch.zeros_like(grid)] * 10, dim=-1))[:, 0]
    assert torch.all(torch.diff(values) > 0)
    assert values[-1] == pytest.approx(np.log1p(1000.0))


@pytest.mark.parametrize("trunk", ["transformer", "lstm"])
def test_step_transform_changes_the_model_input_and_nothing_else(trunk):
    torch.manual_seed(1)
    plain = _plain(trunk=trunk).build().eval()
    torch.manual_seed(1)
    logged = _plain(trunk=trunk, step_transform="signed_log1p").build().eval()
    assert all(torch.equal(a, b) for a, b in zip(plain.state_dict().values(), logged.state_dict().values()))
    x = _inputs()
    with torch.no_grad():
        torch.testing.assert_close(logged(x), plain(signed_log1p(x)))
        assert not torch.allclose(logged(x), plain(x))


def test_step_transform_leaves_the_targets_and_loss_definition_alone(tmp_path):
    pool = _horizon_pool(tmp_path)
    prepared = PreparedPool.from_source_pool(pool, horizon_k=horizon_k(pool))
    features, token_mask, targets, target_mask, _ = prepared.sample_horizon(torch.Generator().manual_seed(4), 32)
    torch.manual_seed(5)
    logged = _plain(step_transform="signed_log1p").build().eval()
    torch.manual_seed(5)
    plain = _plain().build().eval()
    with torch.no_grad():
        assert horizon_huber(logged(features), targets, target_mask)[0] == horizon_huber(
            plain(signed_log1p(features)), targets, target_mask)[0]
    assert torch.equal(prepared.sample_horizon(torch.Generator().manual_seed(4), 32)[2], targets)


def test_the_repair_switches_are_refused_where_they_do_not_belong():
    for bad in (dict(input_scale="half"), dict(step_transform="log")):
        with pytest.raises(ValueError, match="must be one of"):
            _plain(**bad)
    for bad in (dict(input_scale="none"), dict(norm_first=True)):
        with pytest.raises(ValueError, match="Transformer trunk only"):
            _plain(trunk="lstm", **bad)
        with pytest.raises(ValueError, match="Transformer trunk only"):
            _plain(trunk="moe", **bad)
    with pytest.raises(ValueError, match="forecast arms"):
        _config(use_reconstruction_head=True, step_transform="signed_log1p")
    assert _plain(trunk="lstm", step_transform="signed_log1p").build().step_transform == "signed_log1p"


def test_resume_defaults_cover_every_field_added_after_checkpoints_existed():
    assert RESUME_DEFAULTS == {"env_channels": 0} | REPAIR_FIELDS
    defaults = asdict(_config())
    assert all(defaults[name] == value for name, value in RESUME_DEFAULTS.items())


@pytest.mark.parametrize("variant", [dict(input_scale="none"), dict(norm_first=True), dict(step_transform="signed_log1p")])
def test_a_pre_repair_checkpoint_resumes_only_at_the_defaults(tmp_path, variant):
    pool = _horizon_pool(tmp_path)
    prepared = PreparedPool.from_source_pool(pool, horizon_k=horizon_k(pool))
    config = _plain(steps=8, seed=2)
    uninterrupted = train(config, prepared, log_every=4, checkpoint_every=0)

    class Killed(Exception):
        pass

    def die_at_4(step, _loss):
        if step >= 4:
            raise Killed

    path = tmp_path / "run.pt"
    with pytest.raises(Killed):
        train(config, prepared, log_every=4, checkpoint_every=4, checkpoint_path=path, progress=die_at_4)
    state = torch.load(path, weights_only=False)
    for name in REPAIR_FIELDS:
        state["config"].pop(name)  # as a checkpoint written before the repair probe holds it
    torch.save(state, path)

    with pytest.raises(ValueError, match="different configuration"):
        train(replace(config, **variant), prepared, checkpoint_every=4, checkpoint_path=path)
    resumed = train(config, prepared, log_every=4, checkpoint_every=4, checkpoint_path=path)
    assert resumed.resumed_from == 4 and resumed.steps_run == 4
    for name, tensor in uninterrupted.model.state_dict().items():
        assert torch.allclose(tensor, resumed.model.state_dict()[name], atol=1e-6), name
