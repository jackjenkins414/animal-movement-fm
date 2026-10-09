# GenAI reference: GenAI made this file from what I outlined; Z/B share one class, conditioning only uses the context and 
# scales with the window, masked tokens use a learned embedding, all params get gradient, and Huber is rotation invariant 
# with the right quadratic/linear limits. I validated all the tests.

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pytest
import torch

from animal_movement_fm.data.enu import BIN_SECONDS, N_FEATURES, encode_window
from animal_movement_fm.models.fm import (
    CONTEXT_COND_DIM,
    MOVEMENT_CHANNELS,
    MovementFM,
    context_conditioning,
)
from animal_movement_fm.train.loss import HUBER_DELTA, masked_reconstruction_loss, normalised_huber

BATCH, SEQ = 8, 11


def _clock(batch, positions=SEQ + 1, rng=None):
    """Epoch seconds per position. Irregular unless a caller wants an exact one-bin grid."""
    t0 = np.datetime64("2021-06-15T00:00:00").astype("datetime64[s]").astype("int64")
    if rng is None:
        step = np.full((batch, positions - 1), float(BIN_SECONDS))
    else:
        step = rng.uniform(0.4, 1.9, size=(batch, positions - 1)) * BIN_SECONDS
    return t0 + np.concatenate([np.zeros((batch, 1)), np.cumsum(step, axis=1)], axis=1)


def _batch(seed=0):
    rng = np.random.default_rng(seed)
    lon = np.cumsum(rng.normal(0, 0.01, size=(BATCH, 12)), axis=1)
    lat = np.cumsum(rng.normal(0, 0.01, size=(BATCH, 12)), axis=1) - 40.0
    features, scale = encode_window(lon, lat, timestamp_s=_clock(BATCH, rng=rng))
    return torch.tensor(features, dtype=torch.float32), torch.tensor(scale, dtype=torch.float32)


def test_forward_shape():
    model = MovementFM(d_model=32, num_layers=1)
    features, _ = _batch()
    assert model(features).shape == (BATCH, 2)


def test_z_and_b_are_one_class_and_b_has_no_conditioning_path():
    z = MovementFM(d_model=32, num_layers=1, use_fm_package=True)
    b = MovementFM(d_model=32, num_layers=1, use_fm_package=False)
    assert z.cond_proj is not None and b.cond_proj is None
    assert z.n_parameters() > b.n_parameters()
    features, _ = _batch()
    assert b(features).shape == z(features).shape


def test_conditioning_reads_only_the_context_window():
    """Defined for any species, because it is a function of the window in front of the model."""
    features, _ = _batch()
    cond = context_conditioning(features)
    assert cond.shape == (BATCH, CONTEXT_COND_DIM)
    assert torch.isfinite(cond).all()


def test_conditioning_moves_with_the_window_scale():
    """A larger-scale window must produce a different conditioning vector, or it conditions on nothing."""
    rng = np.random.default_rng(1)
    small = np.cumsum(rng.normal(0, 0.0001, size=(1, 12)), axis=1)
    large = np.cumsum(rng.normal(0, 0.5, size=(1, 12)), axis=1)
    lat = np.zeros((1, 12)) - 40.0
    one_bin = _clock(1)
    f_small = torch.tensor(encode_window(small, lat, timestamp_s=one_bin)[0], dtype=torch.float32)
    f_large = torch.tensor(encode_window(large, lat, timestamp_s=one_bin)[0], dtype=torch.float32)
    assert not torch.allclose(context_conditioning(f_small), context_conditioning(f_large))


def test_external_conditioning_hook_is_wired_for_the_taxon_ablation():
    """The MoveFormer-style external vector must be a flag, not a rewrite."""
    model = MovementFM(d_model=32, num_layers=1, external_cond_dim=5)
    features, _ = _batch()
    out = model(features, external_cond=torch.zeros(BATCH, 5))
    assert out.shape == (BATCH, 2)
    with pytest.raises(ValueError, match="external_cond"):
        model(features)


def test_a_masked_token_is_replaced_by_a_learned_embedding_not_zeroed():
    """
    Zeroing the movement channels would make a hidden token indistinguishable from a stationary one.

    Heading is already gated to zero below 10 m and roughly a third of steps in this corpus are low
    displacement, so "hidden" and "did not move" would collide precisely where movement is hardest to
    predict, and the reconstruction target would be ambiguous there. A learned embedding says "this one
    is hidden" in a way no real observation can imitate.
    """
    model = MovementFM(d_model=32, num_layers=1, use_reconstruction_head=True).eval()
    features, _ = _batch()
    mask = torch.zeros(BATCH, SEQ, dtype=torch.bool)
    mask[0, 3] = True

    with torch.no_grad():
        masked = model.encode(features, token_mask=mask)
        clean = model.encode(features)
        # a genuinely stationary token: movement channels zero, covariates intact
        stationary = features.clone()
        stationary[0, 3, list(MOVEMENT_CHANNELS)] = 0.0
        zeroed = model.encode(stationary)

    assert not torch.allclose(masked[0, 3], clean[0, 3])
    assert not torch.allclose(masked[0, 3], zeroed[0, 3])  # hidden != stationary
    assert model.mask_token is not None and model.mask_token.requires_grad


def test_the_mask_token_receives_gradient():
    model = MovementFM(d_model=32, num_layers=1, use_reconstruction_head=True)
    features, _ = _batch()
    mask = torch.zeros(BATCH, SEQ, dtype=torch.bool)
    mask[:, 2] = True
    prediction, _ = model(features, token_mask=mask)
    normalised_huber(prediction, torch.zeros(BATCH, 2)).backward()
    assert model.mask_token.grad is not None and torch.any(model.mask_token.grad != 0)


def test_reconstruction_head_returns_per_token_reconstruction():
    model = MovementFM(d_model=32, num_layers=1, use_reconstruction_head=True)
    features, _ = _batch()
    pred, recon = model(features)
    assert pred.shape == (BATCH, 2)
    assert recon.shape == (BATCH, SEQ, 2)


def test_backward_pass_reaches_every_parameter():
    """A conditioning path that receives no gradient is decoration."""
    model = MovementFM(d_model=32, num_layers=1)
    features, _ = _batch()
    target = torch.zeros(BATCH, 2)
    normalised_huber(model(features), target).backward()
    unused = [n for n, p in model.named_parameters() if p.grad is None or torch.all(p.grad == 0)]
    assert not unused, f"parameters received no gradient: {unused}"


def test_the_loss_depends_on_the_error_only_through_its_magnitude():
    """
    Rotation invariance. This is why the loss is radial rather than component-wise.

    Every number Section 3 reports is a geodesic distance, which does not care how an error decomposes
    into east and north. The component-wise Huber this replaced did: at magnitude 2 it charged 1.50 for
    an error along an axis and 1.83 for the same distance at 45 degrees, so it trained for a geometry
    the evaluation does not use.
    """
    zero = torch.zeros(1, 2)
    magnitude = 2.0
    angles = torch.linspace(0, 2 * np.pi, 17)[:-1]
    losses = [
        normalised_huber(
            torch.tensor([[magnitude * float(torch.cos(a)), magnitude * float(torch.sin(a))]]), zero
        ).item()
        for a in angles
    ]
    assert max(losses) - min(losses) < 1e-6

    # and the anisotropy the component-wise form had, recorded so the change is not undone by accident
    def component_wise(vector):
        return float(
            torch.nn.functional.huber_loss(
                torch.tensor([vector]), zero, delta=HUBER_DELTA, reduction="mean"
            )
        )

    assert component_wise([2.0, 0.0]) == pytest.approx(0.75, rel=1e-6)
    assert component_wise([2**0.5, 2**0.5]) > 1.15 * component_wise([2.0, 0.0])


def test_the_loss_is_quadratic_below_delta_and_linear_above():
    """delta = 1.0 sits at the measured typical window: median normalised error 0.797 / 0.982 / 0.996."""
    zero = torch.zeros(1, 2)
    tiny = normalised_huber(torch.tensor([[0.01, 0.0]]), zero).item()
    # rel=5e-3 rather than tighter: sqrt(1 + 1e-4) in float32 is where the slack comes from, not the form
    assert tiny == pytest.approx(0.5 * 0.01**2, rel=5e-3)  # -> r^2 / 2

    large = normalised_huber(torch.tensor([[100.0, 0.0]]), zero).item()
    assert large == pytest.approx(100.0 - HUBER_DELTA, rel=1e-3)  # -> r - delta

    # a 100x error costs ~100x, not ~10,000x: the tail is what the Huber form is for
    unit = normalised_huber(torch.tensor([[1.0, 0.0]]), zero).item()
    assert 90 < large / unit < 250


def test_masked_reconstruction_ignores_unmasked_tokens():
    recon = torch.zeros(2, 4, 2)
    target = torch.ones(2, 4, 2)
    mask = torch.zeros(2, 4, dtype=torch.bool)
    assert masked_reconstruction_loss(recon, target, mask).item() == 0.0
    mask[:, 0] = True
    assert masked_reconstruction_loss(recon, target, mask).item() > 0.0


def test_model_is_deterministic_in_eval_mode():
    model = MovementFM(d_model=32, num_layers=1).eval()
    features, _ = _batch()
    with torch.no_grad():
        assert torch.allclose(model(features), model(features))


# tests for the Tier 3 LSTM trunk and the window model predictor 
def test_lstm_trunk_forward_and_backward():
    model = MovementFM(d_model=16, num_layers=1, use_fm_package=False, trunk="lstm")
    features, _ = _batch()
    out = model(features)
    assert out.shape == (features.shape[0], 2)
    out.sum().backward()
    assert all(p.grad is not None for p in model.parameters())
    assert model.pos_encoder is None and isinstance(model.encoder, torch.nn.LSTM)


def test_lstm_trunk_refuses_the_fm_package():
    with pytest.raises(ValueError):
        MovementFM(d_model=16, num_layers=1, use_fm_package=True, trunk="lstm")
    with pytest.raises(ValueError):
        MovementFM(d_model=16, num_layers=1, use_fm_package=False, use_reconstruction_head=True, trunk="lstm")
    with pytest.raises(ValueError):
        MovementFM(d_model=16, num_layers=1, trunk="gru")


def test_train_config_carries_the_trunk_and_old_configs_rebuild_as_transformers():
    from animal_movement_fm.train.loop import TrainConfig

    lstm = TrainConfig(d_model=16, num_layers=1, lr=1e-3, steps=10, use_fm_package=False, trunk="lstm").build()
    assert lstm.trunk == "lstm"
    old = TrainConfig(d_model=16, num_layers=1, lr=1e-3, steps=10).build()
    assert old.trunk == "transformer"


def test_window_model_predictor_matches_the_fm_path_for_a_zero_model():
    from animal_movement_fm.eval.fm_predictor import WindowModelPredictor

    class Zero:
        name = "zero"

        def predict(self, features):
            return np.zeros((features.shape[0], 2))

    rng = np.random.default_rng(3)
    lon = 10 + np.cumsum(rng.normal(scale=0.01, size=(4, SEQ + 1)), axis=1)
    lat = 50 + np.cumsum(rng.normal(scale=0.01, size=(4, SEQ + 1)), axis=1)
    history = np.stack([lon, lat], axis=-1)
    stamps = (np.arange(SEQ + 1) * BIN_SECONDS)[None, :].repeat(4, axis=0).astype("datetime64[s]")
    pred = WindowModelPredictor(Zero()).predict_batch(history, stamps)
    # a zero displacement is persistence: the last observed fix
    np.testing.assert_allclose(pred, history[:, -1, :])


LONG = 47  # a C = 48 history: 36 extended tokens, then the 11-token core


def _padded(seed=0, batch=BATCH):
    """Features with holes in the extended history; the core (last 11 tokens) is always observed."""
    generator = torch.Generator().manual_seed(seed)
    features = torch.randn(batch, LONG, N_FEATURES, generator=generator)
    observed = torch.rand(batch, LONG, generator=generator) > 0.4
    observed[:, -11:] = True
    observed[0, :-11] = False  # one window with no extended token observed
    return torch.where(observed[..., None], features, torch.zeros(())), observed


def test_n_horizons_zero_is_the_old_model():
    for kwargs in (dict(), dict(use_fm_package=False), dict(use_fm_package=False, trunk="lstm"),
                   dict(use_reconstruction_head=True)):
        torch.manual_seed(5)
        old = MovementFM(d_model=32, num_layers=2, **kwargs).eval()
        torch.manual_seed(5)
        new = MovementFM(d_model=32, num_layers=2, n_horizons=0, observed_flag=False, **kwargs).eval()
        old_state, new_state = old.state_dict(), new.state_dict()
        assert old_state.keys() == new_state.keys()
        assert all(torch.equal(old_state[k], new_state[k]) for k in old_state)
        features, _ = _batch()
        with torch.no_grad():
            a, b = old(features), new(features)
        a, b = (a, b) if isinstance(a, tuple) else ((a,), (b,))
        assert all(torch.equal(x, y) for x, y in zip(a, b)) and a[0].shape == (BATCH, 2)


def test_one_east_north_head_per_horizon():
    model = MovementFM(d_model=32, num_layers=1, use_fm_package=False, n_horizons=3).eval()
    assert model.head.out_features == 6
    pred = model(_batch()[0])
    assert pred.shape == (BATCH, 3, 2)
    assert torch.equal(pred, model.head(model.encode(_batch()[0])[:, -1, :]).view(BATCH, 3, 2))
    with pytest.raises(ValueError, match="output_dim"):
        MovementFM(n_horizons=3, output_dim=4, use_fm_package=False)
    with pytest.raises(ValueError, match="output_dim"):
        MovementFM(n_horizons=-1, use_fm_package=False)


@pytest.mark.parametrize("mode", ["eval", "train"])
def test_padded_tokens_cannot_move_the_transformer_output(mode):
    model = MovementFM(d_model=32, num_layers=2, dropout=0.0, use_fm_package=False, n_horizons=3)
    model.train(mode == "train")
    features, observed = _padded()
    garbage = torch.where(observed[..., None], features, 1e3 * torch.randn(features.shape))
    with torch.no_grad():
        clean, noisy = model(features, observed=observed), model(garbage, observed=observed)
        unmasked = model(garbage)
    assert torch.allclose(clean, noisy, atol=1e-6, rtol=0)
    assert not torch.allclose(clean, unmasked, atol=1e-3)  # without the mask the garbage does reach the read-out
    assert torch.isfinite(clean).all()


def test_a_fully_observed_mask_is_no_mask():
    model = MovementFM(d_model=32, num_layers=2, use_fm_package=False, n_horizons=3).eval()
    features, _ = _batch()
    with torch.no_grad():
        assert torch.allclose(model(features), model(features, observed=torch.ones(BATCH, SEQ, dtype=torch.bool)),
                              atol=1e-6, rtol=0)


def test_the_lstm_reads_the_observed_flag_as_its_last_channel():
    torch.manual_seed(0)
    model = MovementFM(input_dim=N_FEATURES + 1, d_model=16, num_layers=1, use_fm_package=False, trunk="lstm",
                       n_horizons=3, observed_flag=True).eval()
    features, observed = _padded()
    assert model.input_proj.in_features == N_FEATURES + 1
    with torch.no_grad():
        pred = model(features, observed=observed)
        flipped = observed.clone()
        flipped[1, -12] = ~flipped[1, -12]  # the last extended token
        moved = model(features, observed=flipped)
        expected, _ = model.encoder(model.input_proj(torch.cat([features, observed[..., None].float()], dim=-1)))
    assert pred.shape == (BATCH, 3, 2)
    assert torch.equal(pred, model.head(expected[:, -1, :]).view(BATCH, 3, 2))
    assert not torch.equal(pred[1], moved[1]) and torch.equal(pred[2:], moved[2:])
    with pytest.raises(ValueError, match="needs `observed`"):
        model(features)
    plain = MovementFM(d_model=16, num_layers=1, use_fm_package=False, trunk="lstm")
    with pytest.raises(ValueError, match="cannot mask"):
        plain(features, observed=observed)


def test_padding_is_refused_where_it_cannot_be_honoured():
    features, observed = _padded()
    with pytest.raises(ValueError, match="FM package"):
        MovementFM(d_model=16, num_layers=1)(features, observed=observed)
    with pytest.raises(ValueError, match="FM package"):
        MovementFM(d_model=16, num_layers=1, use_fm_package=False, use_reconstruction_head=True)(
            features, observed=observed)
    with pytest.raises(ValueError, match="MoE"):
        MovementFM(d_model=16, num_layers=1, use_fm_package=False, trunk="moe")(features, observed=observed)
    with pytest.raises(ValueError, match="observed flag"):
        MovementFM(d_model=16, num_layers=1, use_fm_package=False, observed_flag=True)


def test_gradient_reaches_every_parameter_through_a_padded_history():
    features, observed = _padded()
    for model in (
        MovementFM(d_model=16, num_layers=2, use_fm_package=False, n_horizons=3),
        MovementFM(input_dim=N_FEATURES + 1, d_model=16, num_layers=2, use_fm_package=False, trunk="lstm",
                   n_horizons=3, observed_flag=True),
    ):
        model(features, observed=observed).sum().backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


# GenAI reference: tests for horizon_huber (step 3.5, 10-10), from the behaviours I specified: missing targets ignored
# in value and gradient, heads weighted equally whatever their counts, an empty head dropped, and the full case equal to
# the mean of three normalised_huber.
from animal_movement_fm.train.loss import horizon_huber  # noqa: E402


def _horizon_targets(seed=0, batch=64):
    generator = torch.Generator().manual_seed(seed)
    pred = torch.randn(batch, 3, 2, generator=generator, requires_grad=True)
    target = 3.0 * torch.randn(batch, 3, 2, generator=generator)
    return pred, target


def test_horizon_huber_is_the_mean_of_three_normalised_huber_when_full():
    pred, target = _horizon_targets()
    loss, per_head = horizon_huber(pred, target, torch.ones(64, 3, dtype=torch.bool))
    heads = torch.stack([normalised_huber(pred[:, j], target[:, j]) for j in range(3)])
    assert torch.allclose(per_head, heads, atol=1e-6, rtol=0)
    assert torch.allclose(loss, heads.mean(), atol=1e-6, rtol=0)


def test_horizon_huber_ignores_missing_targets_in_value_and_gradient():
    pred, target = _horizon_targets()
    mask = torch.rand(64, 3, generator=torch.Generator().manual_seed(1)) > 0.3
    loss, per_head = horizon_huber(pred, target, mask)
    poisoned = torch.where(mask[..., None], target, torch.full_like(target, float("nan")))
    loss_nan, per_head_nan = horizon_huber(pred, poisoned, mask)
    assert torch.equal(loss, loss_nan) and torch.equal(per_head, per_head_nan)
    loss_nan.backward()
    assert torch.isfinite(pred.grad).all() and (pred.grad[~mask] == 0).all()
    for j in range(3):
        assert torch.allclose(per_head[j], normalised_huber(pred[mask[:, j], j], target[mask[:, j], j]), atol=1e-6)


def test_horizon_huber_weights_heads_equally_not_by_their_counts():
    pred, target = _horizon_targets()
    mask = torch.ones(64, 3, dtype=torch.bool)
    mask[4:, 2] = False  # the third head keeps 4 windows of 64
    loss, per_head = horizon_huber(pred, target, mask)
    assert torch.allclose(loss, per_head.mean(), atol=1e-6, rtol=0)


def test_an_empty_head_is_dropped_from_the_batch_mean():
    pred, target = _horizon_targets()
    mask = torch.ones(64, 3, dtype=torch.bool)
    mask[:, 1] = False
    loss, per_head = horizon_huber(pred, target, mask)
    assert torch.isnan(per_head[1]) and torch.isfinite(per_head[[0, 2]]).all()
    assert torch.allclose(loss, per_head[[0, 2]].mean(), atol=1e-6, rtol=0)
    loss.backward()
    assert (pred.grad[:, 1] == 0).all()
    assert torch.isnan(horizon_huber(pred, target, torch.zeros(64, 3, dtype=torch.bool))[0])  # T3 withholds it
