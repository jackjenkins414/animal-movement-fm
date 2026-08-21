"""
Tier 2 training loop for one configuration on one source pool fold.

The same loop trains both the FM and our pooled baseline; they differ only through model flags. Training uses the fixed 
Tier 1 optimiser recipe, runs for optimiser steps under with replacement sampling, and can resume from checkpoints while 
restoring model, optimiser, scheduler, sampler, and RNG state.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

from animal_movement_fm.data.enu import N_FEATURES
from animal_movement_fm.models.fm import MovementFM
from animal_movement_fm.train.loss import (
    HUBER_DELTA,
    masked_reconstruction_loss,
    normalised_huber,
    radial_pseudo_huber,
)
from animal_movement_fm.train.pool import PreparedPool

# fixed training constants we carry over from Tier 1
BATCH_SIZE = 256  # Tier 1; all architectures
WARMUP_FRACTION = 0.1  # Tier 1 Transformer; `lr` is the peak reached at the end of warmup
GRAD_CLIP = 1.0  # Tier 1 Transformer
DROPOUT = 0.1  # Tier 1 Transformer
NHEAD = 4  # Tier 1 Transformer
FEEDFORWARD_MULTIPLIER = 2  # dim_feedforward = 2 x d_model, as `MovementFM` already defaults

# fixed auxiliary reconstruction settings.
MASK_RATIO = 0.15
RECONSTRUCTION_WEIGHT = 0.5


@dataclass(frozen=True)
class TrainConfig:
    """
    One registered training configuration and its run identity.

    Only `d_model`, `num_layers`, and `lr` vary across the tuning grid; the remaining training settings are fixed 
    constants.
    """

    d_model: int
    num_layers: int
    lr: float
    steps: int
    seed: int = 0
    batch_size: int = BATCH_SIZE
    warmup_fraction: float = WARMUP_FRACTION
    grad_clip: float = GRAD_CLIP
    dropout: float = DROPOUT
    nhead: int = NHEAD
    use_fm_package: bool = True  # false is our pooled source baseline B
    use_reconstruction_head: bool = False
    mask_ratio: float = MASK_RATIO
    reconstruction_weight: float = RECONSTRUCTION_WEIGHT

    def __post_init__(self) -> None:
        if self.steps < 1:
            raise ValueError("steps must be at least 1")
        if not 0.0 < self.mask_ratio < 1.0:
            raise ValueError("mask_ratio is a proportion of tokens and must lie strictly in (0, 1)")

    def build(self, *, input_dim: int = N_FEATURES) -> MovementFM:
        return MovementFM(
            input_dim=input_dim,
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            dim_feedforward=FEEDFORWARD_MULTIPLIER * self.d_model,
            dropout=self.dropout,
            use_fm_package=self.use_fm_package,
            use_reconstruction_head=self.use_reconstruction_head,
        )


@dataclass
class TrainResult:
    model: MovementFM
    config: TrainConfig
    steps_run: int
    loss_history: list[float] = field(default_factory=list)
    loss_shares: list[dict] = field(default_factory=list)
    resumed_from: int = 0

    @property
    def final_loss(self) -> float:
        return self.loss_history[-1] if self.loss_history else float("nan")


def lr_scale(step: int, *, total_steps: int, warmup_fraction: float) -> float:
    """
    Return the learning rate multiplier for linear warmup followed by cosine decay.

    `step` is zero based. The multiplier reaches 1.0 at the end of warmup and then decays toward zero.
    """
    warmup_steps = int(warmup_fraction * total_steps)
    if warmup_steps > 0 and step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def _sample_token_mask(
    shape: tuple[int, int], ratio: float, generator: torch.Generator, device: torch.device
) -> torch.Tensor:
    """
    Sample a Bernoulli token mask with at least one masked token per sequence.

    For rows where the initial draw masks nothing, we select one token uniformly so every sequence contributes to the 
    reconstruction objective.
    """
    mask = torch.rand(shape, generator=generator, device=device) < ratio
    empty = ~mask.any(dim=1)
    if empty.any():
        forced = torch.randint(
            0, shape[1], (int(empty.sum()),), generator=generator, device=device
        )
        mask[empty, forced] = True
    return mask


def _species_loss_share(model, features, targets, species, names, config) -> dict:
    """
    Return each source species' share of the batch forecast loss.

    Reporting diagnostic only; shares are computed under the production radial pseudo-Huber loss and do not feed back 
    into training decisions.
    """
    with torch.no_grad():
        output = model(features)
        prediction = output[0] if isinstance(output, tuple) else output
        per_window = radial_pseudo_huber(prediction - targets, delta=HUBER_DELTA, reduction="none")
        total = per_window.sum()
        if total <= 0:
            return {name: float("nan") for name in names}
        return {
            name: float(per_window[species == index].sum() / total)
            for index, name in enumerate(names)
        }


def train(
    config: TrainConfig,
    pool: PreparedPool,
    *,
    device: torch.device | str | None = None,
    checkpoint_path: str | Path | None = None,
    checkpoint_every: int = 2_000,
    log_every: int = 500,
    progress=None,
) -> TrainResult:
    """
    Train one configuration on one prepared source pool.

    If `checkpoint_path` exists, the run resumes from that saved state. Model, optimiser, scheduler, sampler generator, 
    and global RNG state are restored so a resumed run follows the same sampling/dropout sequence as an uninterrupted run.
    """
    device = torch.device(device or pool.features.device)
    torch.manual_seed(config.seed)

    model = config.build(input_dim=pool.n_features).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimiser,
        lambda step: lr_scale(
            step, total_steps=config.steps, warmup_fraction=config.warmup_fraction
        ),
    )

    # separate sampler RNG keeps window draws independent of unrelated global RNG use
    generator = torch.Generator(device=device)
    generator.manual_seed(config.seed)

    start_step, history, loss_shares = 0, [], []
    checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
    if checkpoint_path is not None and checkpoint_path.exists():
        state = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if state["config"] != asdict(config):
            raise ValueError(
                f"{checkpoint_path} was written by a different configuration. A requeued subjob must "
                "resume its own run; delete the checkpoint or fix the run id rather than continuing "
                "one configuration's training into another's."
            )
        model.load_state_dict(state["model"])
        optimiser.load_state_dict(state["optimiser"])
        scheduler.load_state_dict(state["scheduler"])
        generator.set_state(state["generator"].to(generator.get_state().device))

        # dropout uses the global RNG; so restore it as well as the sampler generator
        torch.set_rng_state(state["torch_rng"])
        if state.get("cuda_rng") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        start_step, history = state["step"], list(state["loss_history"])
        loss_shares = list(state.get("loss_shares", []))

    def save(step: int) -> None:
        if checkpoint_path is None:
            return
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = checkpoint_path.with_suffix(checkpoint_path.suffix + ".partial")
        torch.save(
            {
                "config": asdict(config),
                "step": step,
                "model": model.state_dict(),
                "optimiser": optimiser.state_dict(),
                "scheduler": scheduler.state_dict(),
                "generator": generator.get_state(),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                "loss_history": history,
                "loss_shares": loss_shares,
                "species": list(pool.names),
            },
            tmp,
        )
        # atomicish checkpoint write; we rename only after the temporary file is complete
        tmp.replace(checkpoint_path)

    model.train()
    for step in range(start_step, config.steps):
        features, targets, species = pool.sample(generator, config.batch_size)

        optimiser.zero_grad(set_to_none=True)
        if config.use_reconstruction_head:
            token_mask = _sample_token_mask(
                features.shape[:2], config.mask_ratio, generator, device
            )
            prediction, reconstruction = model(features, token_mask=token_mask)
            movement = features[..., :2]  # scaled east/north target for reconstruction
            forecast_loss = normalised_huber(prediction, targets)
            reconstruction_loss = masked_reconstruction_loss(
                reconstruction,
                movement,
                token_mask,
            )
            loss = forecast_loss + config.reconstruction_weight * reconstruction_loss
        else:
            loss = normalised_huber(model(features), targets)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimiser.step()
        scheduler.step()

        if (step + 1) % log_every == 0 or step + 1 == config.steps:
            history.append(float(loss.detach()))
            # report realised loss composition under the production loss
            loss_shares.append(
                _species_loss_share(model, features, targets, species, pool.names, config)
            )
        # save only states the loop has actually reached
        if checkpoint_every and (step + 1) % checkpoint_every == 0:
            save(step + 1)
        if progress is not None and history and (step + 1) % log_every == 0:
            progress(step + 1, history[-1])

    save(config.steps)
    return TrainResult(
        model=model,
        config=config,
        steps_run=config.steps - start_step,
        loss_history=history,
        loss_shares=loss_shares,
        resumed_from=start_step,
    )
