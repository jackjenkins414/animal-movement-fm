# GenAI reference: I asked for tests for the pooling study's MoE trunk, which is the gate, the load balancing loss, the parameter
# counts agreed for the active parameter match, that evaluation runs the experts rather than a fused dense path, that routing reads 
# only the window, and that a MoE configuration rebuilds and trains through the registered machinery. GenAI wrote them.

import inspect
from dataclasses import asdict

import torch

import animal_movement_fm.train.loop as loop
from animal_movement_fm.data.enu import N_FEATURES
from animal_movement_fm.models.moe import N_EXPERTS, TOP_K, MoEEncoder, MoEFeedForward
from animal_movement_fm.train.loop import TrainConfig, train
from tests.test_b26_golden import golden_pool


def _config(**kwargs) -> TrainConfig:
    base = dict(d_model=256, num_layers=4, lr=1e-3, steps=5, seed=0,
                use_fm_package=False, use_reconstruction_head=False, trunk="moe")
    return TrainConfig(**(base | kwargs))


def test_the_agreed_design_constants():
    assert (N_EXPERTS, TOP_K, loop.MOE_AUX_WEIGHT) == (8, 2, 1e-2)


def test_each_token_mixes_its_top_k_experts_with_renormalised_weights():
    torch.manual_seed(0)
    ffn = MoEFeedForward(d_model=8, hidden=6, n_experts=4, top_k=2, dropout=0.0)
    x = torch.randn(3, 5, 8)
    out, probs, _ = ffn(x)
    expected = torch.zeros_like(out)
    for b in range(3):
        for t in range(5):
            top, index = probs[b, t].topk(2)
            for weight, e in zip(top / top.sum(), index):
                expert = torch.relu(x[b, t] @ ffn.w1[e] + ffn.b1[e]) @ ffn.w2[e] + ffn.b2[e]
                expected[b, t] += weight * expert
    assert torch.allclose(out, expected, atol=1e-6)
    assert torch.allclose(probs.sum(-1), torch.ones(3, 5))


def test_the_balancing_loss_is_one_when_routing_is_uniform_and_grows_under_collapse():
    ffn = MoEFeedForward(d_model=8, hidden=4, n_experts=8, top_k=2, dropout=0.0)
    x = torch.randn(64, 12, 8).abs()
    with torch.no_grad():
        ffn.router.weight.zero_()
    assert torch.isclose(ffn(x)[2], torch.tensor(1.0))
    with torch.no_grad():
        ffn.router.weight[:2] = 50.0  # Positive inputs, so every token prefers experts 0 and 1
    assert ffn(x)[2] > 3.9  # E / k = 4 when every token goes to the same two experts


def test_active_parameters_match_the_dense_layer_and_totals_are_as_proposed():
    dense = TrainConfig(d_model=256, num_layers=4, lr=1e-3, steps=1, use_fm_package=False,
                        use_reconstruction_head=False).build()
    moe = _config().build()
    assert dense.n_parameters() == 2_112_258
    assert moe.n_parameters() == 5_279_490
    layer = moe.encoder.layers[0].moe
    assert sum(p.numel() for p in layer.parameters()) == 1_054_720
    per_expert = (layer.w1[0].numel() + layer.b1[0].numel() + layer.w2[0].numel() + layer.b2[0].numel())
    active = TOP_K * per_expert + layer.router.weight.numel()
    assert active == 265_216
    dense_ffn = dense.encoder.layers[0]
    assert sum(p.numel() for m in (dense_ffn.linear1, dense_ffn.linear2) for p in m.parameters()) == 262_912


def test_evaluation_runs_the_experts_not_a_fused_dense_path():
    torch.manual_seed(0)
    model = _config(d_model=32, num_layers=2, dropout=0.0).build()
    x = torch.randn(16, 12, N_FEATURES)
    model.train()
    trained_mode = model(x)
    model.eval()
    with torch.no_grad():
        evaluated = model(x)
    assert torch.allclose(trained_mode, evaluated, atol=1e-5)


def test_routing_reads_only_the_window():
    assert list(inspect.signature(MoEFeedForward.forward).parameters) == ["self", "x"]
    torch.manual_seed(0)
    model = _config(d_model=32, num_layers=2).build().eval()
    x = torch.randn(8, 12, N_FEATURES)
    with torch.no_grad():
        model(x)
        batched = [p[3] for p in model.encoder.router_probs]
        model(x[3:4])
        alone = [p[0] for p in model.encoder.router_probs]
    for a, b in zip(batched, alone):
        assert torch.allclose(a, b, atol=1e-6)


def test_layers_start_as_copies_and_experts_do_not():
    encoder = MoEEncoder(d_model=16, nhead=2, dim_feedforward=32, num_layers=3)
    first, last = encoder.layers[0], encoder.layers[2]
    for (name, a), b in zip(first.named_parameters(), last.parameters()):
        assert torch.equal(a, b), name
    assert not torch.equal(first.moe.w1[0], first.moe.w1[1])
    assert first.moe.w1.shape == (N_EXPERTS, 16, 32 // TOP_K)


def test_a_moe_checkpoint_rebuilds_from_its_saved_config():
    torch.manual_seed(0)
    config = _config(d_model=32, num_layers=2)
    model = config.build().eval()
    rebuilt = TrainConfig(**asdict(config)).build().eval()
    rebuilt.load_state_dict(model.state_dict())  # strict, as the sealed evaluators load
    x = torch.randn(4, 12, N_FEATURES)
    with torch.no_grad():
        assert torch.equal(model(x), rebuilt(x))


def test_moe_trains_through_the_registered_loop_with_the_balancing_term(monkeypatch):
    config = _config(d_model=32, num_layers=2, steps=5)
    torch.manual_seed(config.seed)
    initial_router = config.build().encoder.layers[0].moe.router.weight.detach().clone()
    result = train(config, golden_pool(), device="cpu", log_every=1)
    assert result.applied_updates == 5
    assert all(torch.isfinite(torch.tensor(result.loss_history)))
    assert not torch.equal(result.model.encoder.layers[0].moe.router.weight, initial_router)
    monkeypatch.setattr(loop, "MOE_AUX_WEIGHT", 0.0)
    without = train(config, golden_pool(), device="cpu", log_every=1)
    assert without.loss_history[0] != result.loss_history[0]
