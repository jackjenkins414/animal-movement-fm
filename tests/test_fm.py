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
