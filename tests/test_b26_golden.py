# GenAI reference: I specified a golden test for B26's existing code path (grid 11, plain trunk) before adding MoE changes to fm.py and 
# loop.py, so we can check the pooling edits leave B26 bit-identical. GenAI wrote the fixture and fingerprints; the pinned values came 
# from the unchanged Mac code (torch 2.5.1, CPU, one thread) before edits.
"""
B26's model and training path, pinned bit for bit.
"""

from __future__ import annotations

import hashlib

import torch

from animal_movement_fm.data.enu import N_FEATURES
from animal_movement_fm.train.loop import TrainConfig, train
from animal_movement_fm.train.pool import PreparedPool

CONTEXT = 12
STEPS = 20

GOLDEN_INIT = "61fe7f4a174b0912b20186f3c0d73d297d7841fce623564deb39011f0ba17b3a"
GOLDEN_EVAL_OUTPUT = "84b8f6d7eee4b340dfeb3d73015916933e0b22bfbe51a4ab2e908dfee83abaae"
GOLDEN_TRAINED = "9b484dcaa7ea375c52c8c5dd5c8dd1d26affef0be4bd4bac80ff79e3c1f5442b"
GOLDEN_LOSSES = (
    0.36254066228866577, 1.814397931098938, 0.5035970211029053, 1.1541305780410767, 1.0604348182678223,
    0.39541226625442505, 0.32961222529411316, 0.48486873507499695, 0.32995331287384033, 0.2103710174560547,
    0.263412743806839, 0.2977794408798218, 0.3033550977706909, 0.26547718048095703, 0.20608988404273987,
    0.210432231426239, 0.20777803659439087, 0.20952124893665314, 0.22104589641094208, 0.2069181650876999,
)


def golden_pool() -> PreparedPool:
    """Three synthetic species at 0.5 / 0.3 / 0.2, built directly so no parquet is read."""
    generator = torch.Generator().manual_seed(1234)
    sizes = (400, 300, 200)
    features = torch.randn(sum(sizes), CONTEXT, N_FEATURES, generator=generator)
    targets = 0.5 * torch.randn(sum(sizes), 2, generator=generator)
    bounds, cursor = [], 0
    for size in sizes:
        bounds.append((cursor, cursor + size))
        cursor += size
    return PreparedPool(
        names=("a", "b", "c"),
        features=features,
        targets=targets,
        species_bounds=torch.tensor(bounds, dtype=torch.int64),
        probabilities=torch.tensor([0.5, 0.3, 0.2], dtype=torch.float64),
    )


def b26_config(steps: int = STEPS) -> TrainConfig:
    """B26's configuration at grid 11, as tier4_train.build_config returns it, on a short budget."""
    return TrainConfig(
        d_model=256, num_layers=4, lr=1e-3, steps=steps, seed=0,
        use_fm_package=False, use_reconstruction_head=False,
    )


def digest(tensors) -> str:
    sha = hashlib.sha256()
    for name, tensor in tensors:
        sha.update(name.encode())
        sha.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return sha.hexdigest()


def fingerprints() -> dict:
    """Initial weights, the eval-mode forward the sealed evaluators run, and a short training run."""
    # CPU init and kernels split work across threads, so the bits depend on the thread count: pin it.
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        return _fingerprints()
    finally:
        torch.set_num_threads(threads)


def _fingerprints() -> dict:
    pool = golden_pool()
    config = b26_config()
    # train() seeds the global RNG and then builds, so this is the state it starts from.
    torch.manual_seed(config.seed)
    model = config.build(input_dim=pool.n_features)
    init = digest(model.state_dict().items())
    model.eval()
    with torch.no_grad():
        output = model(pool.features[:64])
    result = train(config, pool, device="cpu", log_every=1)
    return {
        "init": init,
        "eval_output": digest([("output", output)]),
        "trained": digest(result.model.state_dict().items()),
        "losses": tuple(result.loss_history),
    }


def test_b26_is_bit_identical_to_the_tier4_code_path():
    got = fingerprints()
    assert got["init"] == GOLDEN_INIT
    assert got["eval_output"] == GOLDEN_EVAL_OUTPUT
    assert got["losses"] == GOLDEN_LOSSES
    assert got["trained"] == GOLDEN_TRAINED
