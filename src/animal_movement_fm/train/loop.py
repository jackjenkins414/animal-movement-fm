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

# cap on stored non-finite gradient records; once a run is poisoned every later step is
# non-finite too, and the first few are the only informative ones.
NONFINITE_RECORD_LIMIT = 20


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
    grad_norms: list[dict] = field(default_factory=list)
    nonfinite_steps: list[dict] = field(default_factory=list)
    skipped_steps: int = 0
    resumed_from: int = 0

    @property
    def final_loss(self) -> float:
        return self.loss_history[-1] if self.loss_history else float("nan")

    @property
    def first_nonfinite_step(self) -> int | None:
        """
        Zero-based step of the earliest recorded non-finite gradient norm, if any.
        """
        return self.nonfinite_steps[0]["step"] if self.nonfinite_steps else None


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


def _nonfinite_record(step, grad_norm, loss, features, targets, species, names, model) -> dict:
    """
    Describe the batch and the parameters involved in a non-finite gradient norm.

    Called after `clip_grad_norm_` has already run, which is deliberate. The clip multiplies every gradient by
    `max_norm / (total_norm + 1e-6)`, so with a non-finite norm that coefficient is zero: finite elements become exactly
    zero and any infinite element becomes NaN. `origin_parameters` therefore separates the two cases. An empty list means
    the elements were individually finite and only their sum of squares overflowed, so the clip zeroed the step cleanly;
    a populated list names the tensors that were already non-finite before the clip could act.
    """
    return {
        "step": int(step),
        # `inf` and `nan` distinguish where the failure began. An infinite norm with no named origin tensor is an
        # aggregate overflow in the norm itself and the step is simply lost. A NaN norm means a NaN was already
        # circulating in the backward pass, which the clip can neither cause nor contain.
        "kind": "inf" if math.isinf(float(grad_norm)) else "nan",
        "grad_norm": float(grad_norm),
        "loss": float(loss),
        "loss_finite": bool(math.isfinite(float(loss))),
        "max_abs_feature": float(features.abs().max()),
        "max_abs_target": float(targets.abs().max()),
        "batch_species": {
            name: int((species == index).sum()) for index, name in enumerate(names)
        },
        "origin_parameters": [
            name
            for name, parameter in model.named_parameters()
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all()
        ],
    }


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
    diagnose: bool = False,
    skip_nonfinite: bool = False,
) -> TrainResult:
    """
    Train one configuration on one prepared source pool.

    If `checkpoint_path` exists, the run resumes from that saved state. Model, optimiser, scheduler, sampler generator,
    and global RNG state are restored so a resumed run follows the same sampling/dropout sequence as an uninterrupted run.

    `diagnose` only changes how often the pre-clip gradient norm is inspected, never what is computed or applied, so a
    diagnostic run and a scored run follow identical numerics from the same seed. It is off by default because reading the
    norm forces a host/device synchronisation on every step; with it off the norm is inspected at the logging cadence, which
    still detects a poisoned run but localises it only to within `log_every` steps.

    `skip_nonfinite` does change what is applied: when the pre-clip norm is non-finite the optimiser step is not taken, so
    the update never reaches the weights. The learning-rate schedule still advances, keeping the cosine curve on the
    registered timeline, and the count of skipped steps is returned so it can be reported. It is off by default and is not
    part of any registered configuration; it exists to be measured before it is proposed as an amendment.
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
    grad_norms, nonfinite_steps, skipped_steps = [], [], 0
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
        grad_norms = list(state.get("grad_norms", []))
        nonfinite_steps = list(state.get("nonfinite_steps", []))
        skipped_steps = int(state.get("skipped_steps", 0))

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
                "grad_norms": grad_norms,
                "nonfinite_steps": nonfinite_steps,
                "skipped_steps": skipped_steps,
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

        # GenAI suggested this addition; we coded this part
        # `clip_grad_norm_` returns the global gradient norm before clipping; keep it for diagnostics;
        # a non finite norm can indicate a gradient spike that will propagate NaNs into the optimiser state 
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        at_log_step = (step + 1) % log_every == 0 or step + 1 == config.steps
        skip_this_step = False
        if diagnose or skip_nonfinite or at_log_step:
            # reading the norm synchronises with the device, so with neither flag set it is read at the logging cadence
            norm = float(grad_norm)
            if diagnose:
                # the per-step input extreme is recorded alongside the norm so the two can be correlated afterwards
                grad_norms.append(
                    {
                        "step": int(step),
                        "grad_norm": norm,
                        "loss": float(loss.detach()),
                        "max_abs_feature": float(features.abs().max()),
                    }
                )
            if not math.isfinite(norm):
                if len(nonfinite_steps) < NONFINITE_RECORD_LIMIT:
                    nonfinite_steps.append(
                        _nonfinite_record(
                            step, norm, loss.detach(), features, targets, species, pool.names, model
                        )
                    )
                if skip_nonfinite:
                    skip_this_step = True
                    skipped_steps += 1

        # the schedule advances either way, so a skipped step costs an update but does not shift the cosine curve
        if not skip_this_step:
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
        grad_norms=grad_norms,
        nonfinite_steps=nonfinite_steps,
        skipped_steps=skipped_steps,
        resumed_from=start_step,
    )
