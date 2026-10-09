"""
Tier 2 training loop for one configuration on one source pool fold.

The same loop trains both the FM and our pooled baseline; they differ only through model flags. Training uses the fixed 
Tier 1 optimiser recipe, runs for optimiser steps under with replacement sampling, and can resume from checkpoints while 
restoring model, optimiser, scheduler, sampler, and RNG state.
"""

from __future__ import annotations

import contextlib
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

try:  # torch >= 2.3
    from torch.nn.attention import SDPBackend, sdpa_kernel
except ImportError:  # pragma: no cover - older torch on some machines
    SDPBackend = sdpa_kernel = None

from animal_movement_fm.data.enu import N_FEATURES
from animal_movement_fm.data.resample import CONTEXT_LENGTH
from animal_movement_fm.models.fm import MovementFM
from animal_movement_fm.train.loss import (
    HUBER_DELTA,
    horizon_huber,
    masked_reconstruction_loss,
    normalised_huber,
    radial_pseudo_huber,
)
from animal_movement_fm.train.pool import PreparedPool, StreamingPool

# fixed training constants we carry over from Tier 1
BATCH_SIZE = 256  # Tier 1; all architectures
WARMUP_FRACTION = 0.1  # Tier 1 Transformer; `lr` is the peak reached at the end of warmup
GRAD_CLIP = 1.0  # Tier 1 Transformer
DROPOUT = 0.1  # Tier 1 Transformer
NHEAD = 4  # Tier 1 Transformer
FEEDFORWARD_MULTIPLIER = 2  # dim_feedforward = 2 x d_model, as `MovementFM` already defaults

# fixed auxiliary reconstruction settings
MASK_RATIO = 0.15
RECONSTRUCTION_WEIGHT = 0.5

# pooling study, MoE trunk only; weight on the router's load balancing loss; will use the source we first found but reffed: Fedus et al (2022)
MOE_AUX_WEIGHT = 1e-2

# cap on stored non-finite gradient records; once a run is poisoned every later step is non-finite too, and the first few are the 
# only informative ones
NONFINITE_RECORD_LIMIT = 20

# T3: an optimiser step whose pre-clip gradient norm is not finite is not applied. Registered in preregistration_tier2_tuning.md on 
# 22-08-2026; the skipped count is reported for every run
SKIP_NONFINITE_STEPS = True

# T4: the registered budget counts *applied* optimiser updates, not attempted batches, and the learning-rate schedule advances only 
# when an update is applied. A run therefore keeps drawing batches until it has made `config.steps` updates. This cap stops a 
# pathological run from drawing forever; a run that hits it has failed and says so rather than silently reporting a short budget
MAX_ATTEMPT_MULTIPLIER = 4

# Every training run pins the scaled dot product attention backend to `math`.It is here rather than in a runner so that no arm can be 
# launched without it

# PyTorch issue #125674: `ScaledDotProductEfficientAttentionBackward0` returns NaN from finite float32 inputs of large magnitude while 
# the math backend stays finite. Measured on this problem, torch 2.5.1+cu121 / H200, 2,000 applied updates, grid 11, fold 1 (a fold whose 
# training pool contains white stork, so a fold where the failure actually occurs):
#
#   efficient, raw inputs      14.13% of attempts withheld, every one a genuine non-finite gradient
#   efficient, asinh inputs     0%
#   math,      raw inputs       0%
#   math,      asinh inputs     0%
#
# Two independent interventions each remove the failure, which places it in the kernel rather than in the input representation. Pinning 
# `math` costs ~15% wall clock (1.17 vs 1.02 min over an identical 2,000 attempts) and leaves the registered representation untouched, so 
# it is the cheaper of the two; inference is unaffected; the defect is in the backward pass; but the setting is harmless there
SDPA_BACKEND = "math"


def sdpa_context(backend: str):
    """
    Return a context manager pinning the attention backend, or a null context for `"auto"`.

    `"auto"` leaves the choice to torch and exists only so a diagnostic can reproduce the defective path deliberately. Scored runs take 
    the module default.
    """
    if backend == "auto":
        return contextlib.nullcontext()
    if sdpa_kernel is None:
        raise RuntimeError(
            "this torch build has no torch.nn.attention.sdpa_kernel, so the attention backend cannot be pinned and PyTorch #125674 cannot " \
            "be ruled out. Report the torch version."
        )
    try:
        selected = {"math": SDPBackend.MATH, "efficient": SDPBackend.EFFICIENT_ATTENTION}[backend]
    except KeyError:
        raise ValueError(f"unknown sdpa backend {backend!r}") from None
    return sdpa_kernel(selected)


def _clip_grad_norm_stable(parameters, max_norm: float) -> tuple[float, bool]:
    """
    Return `(total_norm, any_nonfinite_element)` and clip the gradients in place.

    Two different questions have to be answered separately, and `clip_grad_norm_` conflates them:

    1. Does any individual gradient tensor hold a NaN or an Inf? Nothing can repair that; the update has to be withheld.
    2. Are all elements finite while the float32 *aggregate* sum of squares overflows? That is an arithmetic artefact, and recomputing the 
    norm in float64 recovers a usable value.

    The fast path is float32 and costs what the stock helper costs. Only when that comes back non-finite do we pay for the per-tensor finite 
    check and the float64 recomputation, which is rare once the underlying defect is fixed. The clip is applied only when the norm is finite, 
    so a withheld step leaves the gradients untouched and `origin_parameters` in the diagnostic record names exactly the tensors that were 
    already bad, rather than tensors the clip itself poisoned ; `max_norm` is unchanged at the registered 1.0; only the arithmetic used to reach 
    it differs.
    """
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return 0.0, False

    total = float(torch.linalg.vector_norm(torch.stack([g.norm(2) for g in grads]), 2))
    nonfinite_element = False
    if not math.isfinite(total):
        nonfinite_element = any(not torch.isfinite(g).all() for g in grads)
        if not nonfinite_element:
            # every element is finite; only the float32 aggregate overflowed
            total = float(
                torch.linalg.vector_norm(
                    torch.stack(
                        [torch.linalg.vector_norm(g, 2, dtype=torch.float64) for g in grads]
                    ),
                    2,
                )
            )

    if math.isfinite(total) and total > max_norm:
        coefficient = max_norm / (total + 1e-6)
        for grad in grads:
            grad.mul_(coefficient)
    return total, nonfinite_element


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
    # Tier E only. Every field below defaults to the registered behaviour, so a configuration written before they existed rebuilds identically 
    # and every registered arm is untouched
    use_context_conditioning: bool = True
    n_species: int = 0
    species_embed_dim: int = 0
    # the shuffled-label control: the embedding is indexed by an i.i.d. uniform draw per window instead of the true species; a fixed permutation 
    # of the labels would not be a control at all, because a learned lookup table is invariant to relabelling; only destroying the correspondence
    # window by window leaves the capacity while removing the information.
    random_species_labels: bool = False
    # E2: everything except the species lookup is held fixed, so the only thing fitted is identity
    freeze_except_species_embedding: bool = False
    # Tier 3 baseline ladder (17-09-2026): the sequence encoder; configurations written before this field rebuild as transformers
    trunk: str = "transformer"
    # Tier 3 B9dom: pool species index -> lookup row, so nine species can share three domain rows; empty means the row is the species itself, which 
    # is every arm before this field existed
    species_label_map: tuple[int, ...] = ()
    horizons: tuple[int, ...] = ()
    history: int = 0
    observed_flag: bool = False

    def __post_init__(self) -> None:
        # this dataclass is frozen, so normalisation goes through object.__setattr__. The map is stored as a tuple
        # so a configuration rebuilt from a saved JSON record, where it is a list, equals the one that was saved.
        object.__setattr__(self, "species_label_map", tuple(int(row) for row in self.species_label_map))
        object.__setattr__(self, "horizons", tuple(int(h) for h in self.horizons))
        if self.horizons and (len(set(self.horizons)) != len(self.horizons) or min(self.horizons) < 1):
            raise ValueError(f"horizons {self.horizons} must be distinct positive bin counts")
        if self.horizons and self.use_reconstruction_head:
            raise ValueError("the reconstruction head is not part of the horizon design")
        if self.history and not self.horizons:
            raise ValueError("a padded history belongs to the horizon design; set horizons")
        if self.history and self.history <= CONTEXT_LENGTH:
            raise ValueError(f"history {self.history} must exceed the {CONTEXT_LENGTH}-bin core; 0 is the core alone")
        if self.history and (self.use_fm_package or self.trunk == "moe"):
            raise ValueError("a padded history needs a plain Transformer or an LSTM; the FM package and MoE cannot mask")
        if self.observed_flag != (self.history > 0 and self.trunk == "lstm"):
            raise ValueError("observed_flag is required for an LSTM on a padded history, and only there")
        if self.species_label_map and not self.species_embed_dim:
            raise ValueError("species_label_map routes species to lookup rows and needs a species embedding")
        if self.species_label_map and not all(0 <= row < self.n_species for row in self.species_label_map):
            raise ValueError(f"species_label_map rows must lie in 0..{self.n_species - 1}")
        if self.species_label_map and self.random_species_labels:
            raise ValueError("a label map and random labels are two different arms; pick one")
        if self.steps < 1:
            raise ValueError("steps must be at least 1")
        if not 0.0 < self.mask_ratio < 1.0:
            raise ValueError("mask_ratio is a proportion of tokens and must lie strictly in (0, 1)")
        if self.species_embed_dim and self.n_species < 1:
            raise ValueError("species_embed_dim needs n_species >= 1")
        if self.random_species_labels and not self.species_embed_dim:
            raise ValueError(
                "random_species_labels is the control for the species embedding and means nothing without one"
            )
        if self.freeze_except_species_embedding and not self.species_embed_dim:
            raise ValueError(
                "freeze_except_species_embedding would leave no trainable parameter at all unless the model carries a species embedding"
            )

    def build(self, *, input_dim: int = N_FEATURES) -> MovementFM:
        return MovementFM(
            input_dim=input_dim + int(self.observed_flag),
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            dim_feedforward=FEEDFORWARD_MULTIPLIER * self.d_model,
            dropout=self.dropout,
            use_fm_package=self.use_fm_package,
            use_reconstruction_head=self.use_reconstruction_head,
            use_context_conditioning=self.use_context_conditioning,
            n_species=self.n_species,
            species_embed_dim=self.species_embed_dim,
            trunk=self.trunk,
            n_horizons=len(self.horizons),
            observed_flag=self.observed_flag,
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
    head_loss_history: list[list[float]] = field(default_factory=list)  # horizon runs: per-head loss at each log
    skipped_steps: int = 0
    attempted_batches: int = 0
    applied_updates: int = 0
    resumed_from: int = 0

    @property
    def withheld_fraction(self) -> float:
        """Withheld attempts as a share of all attempts. Zero in a healthy run."""
        return self.skipped_steps / self.attempted_batches if self.attempted_batches else 0.0

    @property
    def final_loss(self) -> float:
        return self.loss_history[-1] if self.loss_history else float("nan")

    @property
    def first_nonfinite_step(self) -> int | None:
        """
        Zero-based step of the earliest recorded non finite gradient norm, if any.
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

    For rows where the initial draw masks nothing, we select one token uniformly so every sequence contributes to the reconstruction objective.
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

    Under T4 the clip is applied only when the norm is finite, so by the time this runs on a non-finite norm the
    gradients are exactly as autograd left them. `origin_parameters` therefore names the tensors that are genuinely
    non-finite, with no need to reason about what the clip did to them; an empty list means every element was finite and
    only the aggregate overflowed. The caller records `nonfinite_element` alongside, which answers the same question
    directly.
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


def _species_loss_share(
    model, features, targets, species, names, config, cond_species=None, observed=None, target_mask=None
) -> dict:
    """
    Return each source species' share of the batch forecast loss.

    Reporting diagnostic only; shares are computed under the production radial pseudo-Huber loss and do not feed back 
    into training decisions. `cond_species` is the index the conditioning path was given for this
    batch, which is the shuffled draw rather than the true label under the control arm; the shares
    are still attributed to the true species, since that is what the reader wants to know.
    """
    with torch.no_grad():
        output = model(features, species=cond_species, observed=observed)
        prediction = output[0] if isinstance(output, tuple) else output
        per_window = radial_pseudo_huber(prediction - targets, delta=HUBER_DELTA, reduction="none")
        if target_mask is not None:
            # horizon runs: a window's loss is its mean over the heads it has a target for
            per_window = torch.where(target_mask, per_window, torch.zeros_like(per_window)).sum(1) / target_mask.sum(1)
        total = per_window.sum()
        if total <= 0:
            return {name: float("nan") for name in names}
        return {
            name: float(per_window[species == index].sum() / total)
            for index, name in enumerate(names)
        }

# GenAI reference: GenAI helped draft this training function from my spec, including checkpoint/resume, horizon/reconstruction branches, 
# conditioning, gradient handling logging and the applied update budget
def train(
    config: TrainConfig,
    pool: PreparedPool | StreamingPool,
    *,
    device: torch.device | str | None = None,
    checkpoint_path: str | Path | None = None,
    checkpoint_every: int = 2_000,
    log_every: int = 500,
    progress=None,
    diagnose: bool = False,
    skip_nonfinite: bool = SKIP_NONFINITE_STEPS,
    monitor=None,
    monitor_every: int = 0,
    strict_budget: bool = True,
    sdpa_backend: str = SDPA_BACKEND,
    init_state_dict: dict | None = None,
) -> TrainResult:
    """
    Train one configuration on one prepared source pool.

    If `checkpoint_path` exists, the run resumes from that saved state. Model, optimiser, scheduler, sampler generator,
    and global RNG state are restored so a resumed run follows the same sampling/dropout sequence as an uninterrupted run.

    `diagnose` only changes how often the pre-clip gradient norm is inspected, never what is computed or applied, so a
    diagnostic run and a scored run follow identical numerics from the same seed. It is off by default because reading the
    norm forces a host/device synchronisation on every step; with it off the norm is inspected at the logging cadence, which
    still detects a poisoned run but localises it only to within `log_every` steps.

    `skip_nonfinite` withholds the optimiser step when the pre-clip norm is non-finite, so the update never reaches the
    weights, and the count is returned so it can be reported. This is registered behaviour (T3) and defaults to on;
    passing false reproduces the pre-amendment failure and is for diagnostics only.

    **T4.** `config.steps` counts *applied* optimiser updates, not attempted batches, and the schedule advances only with
    the optimiser. T3 originally advanced the schedule on a withheld attempt so the cosine curve stayed on the registered
    timeline; that was measured at a 0.5% withholding rate. At the rates actually observed (up to 90% on one arm) it
    consumed the whole curve while updating a fraction of the time, and left different arms with different effective
    budgets. The loop therefore keeps drawing batches until it has made `config.steps` updates, capped at
    `MAX_ATTEMPT_MULTIPLIER x steps`, and raises rather than returning a short budget if it cannot get there.
    `attempted_batches`, `applied_updates` and `skipped_steps` are all reported. `strict_budget` defaults to true so a 
    scored run that cannot reach its budget raises instead of quietly reporting a short one; diagnostics pass false, because 
    characterising the broken path is the whole point.

    `monitor`, if given, is called as `monitor(step, model)` every `monitor_every` optimiser steps with the model in eval mode, 
    and is what lets the arms holding the registered validation privilege (S, and later F) select a checkpoint. It observes only: 
    it draws no random numbers and touches neither the optimiser nor the sampler, so a monitored run and an unmonitored run from 
    the same seed follow identical numerics. Z, B and Za pass no monitor and never reach the branch.

    `sdpa_backend` pins the attention kernel for the whole loop and defaults to the module constant, so every arm gets it without 
    the runner having to remember. Only a diagnostic passes `"auto"` or `"efficient"`, to reproduce the defective path on purpose. 
    See `SDPA_BACKEND`.

    `init_state_dict` initialises the weights from an already-trained model before the optimiser is built, which is what Tier E's 
    frozen-trunk arm needs; the species lookup is allowed to be absent from it, because that arm's whole purpose is to fit a table 
    the donor model did not have. Any other missing or unexpected key is refused rather than silently ignored. Combined with
    `config.freeze_except_species_embedding` the optimiser then sees only the lookup table, so the run fits one vector per species 
    and nothing else. Registered arms pass neither and are unaffected

    A configuration with `horizons` draws with `sample_horizon` from a horizon pool (PreparedPool, or StreamingPool for a
    padded history) and trains on `horizon_huber`; with `history` > 0 the pool's token mask reaches the model as
    `observed`. Per-head losses are logged in `head_loss_history`. A configuration without horizons is untouched.
    """
    device = torch.device(device or pool.device)
    horizon = bool(config.horizons)
    pool_k = getattr(pool, "horizon_k", None)
    if horizon != (pool_k is not None):
        raise ValueError("a horizon configuration needs a horizon pool, and a horizon pool a horizon configuration")
    if horizon and len(pool_k) != len(config.horizons):
        raise ValueError(f"the pool has {len(pool_k)} horizons, the configuration {len(config.horizons)}")
    torch.manual_seed(config.seed)

    model = config.build(input_dim=pool.n_features).to(device)

    if init_state_dict is not None:
        missing, unexpected = model.load_state_dict(init_state_dict, strict=False)
        # the donor may legitimately lack the species lookup; nothing else may differ, or the arm is not the model it claims to be built 
        # from
        unexplained = set(missing) - {"species_embed.weight"}
        if unexplained or unexpected:
            raise ValueError(
                "init_state_dict does not match this configuration: "
                f"missing {sorted(unexplained)}, unexpected {sorted(unexpected)}"
            )

    if config.freeze_except_species_embedding:
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith("species_embed."))

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise ValueError("no trainable parameters; the freeze left nothing for the optimiser")
    optimiser = torch.optim.Adam(trainable, lr=config.lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimiser,
        lambda step: lr_scale(
            step, total_steps=config.steps, warmup_fraction=config.warmup_fraction
        ),
    )

    # separate sampler RNG keeps window draws independent of unrelated global RNG use
    generator = torch.Generator(device=device)
    generator.manual_seed(config.seed)

    start_step, attempted, history, loss_shares, head_history = 0, 0, [], [], []
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
        # a checkpoint written before T3 can hold NaN weights from a diverged run; resuming one would
        # silently continue a dead model and produce a result file that looks like every other
        if not all(torch.isfinite(parameter).all() for parameter in model.parameters()):
            raise ValueError(
                f"{checkpoint_path} holds non-finite weights, so the run it came from had already "
                "diverged. Delete it and start that budget again rather than resuming a dead model."
            )
        optimiser.load_state_dict(state["optimiser"])
        scheduler.load_state_dict(state["scheduler"])
        generator.set_state(state["generator"].to(generator.get_state().device))

        # dropout uses the global RNG; so restore it as well as the sampler generator
        torch.set_rng_state(state["torch_rng"])
        if state.get("cuda_rng") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        start_step, history = state["step"], list(state["loss_history"])
        # pre-T4 checkpoints have no attempt count; treating it as the applied count is the only
        # honest fallback and is flagged by attempted == applied on a run that clearly skipped
        attempted = int(state.get("attempted", state["step"]))
        loss_shares = list(state.get("loss_shares", []))
        grad_norms = list(state.get("grad_norms", []))
        nonfinite_steps = list(state.get("nonfinite_steps", []))
        skipped_steps = int(state.get("skipped_steps", 0))
        head_history = list(state.get("head_loss_history", []))

    def save(step: int) -> None:
        if checkpoint_path is None:
            return
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = checkpoint_path.with_suffix(checkpoint_path.suffix + ".partial")
        torch.save(
            {
                "config": asdict(config),
                "step": step,
                "attempted": attempted,
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
                "head_loss_history": head_history,
                "species": list(pool.names),
            },
            tmp,
        )
        # atomicish checkpoint write; we rename only after the temporary file is complete
        tmp.replace(checkpoint_path)

    model.train()
    # T4's four clocks. `applied` is the registered budget; `attempted` is how many batches it cost.
    # They are equal in a healthy run and the gap between them is the thing worth reporting.
    applied = start_step
    max_attempts = MAX_ATTEMPT_MULTIPLIER * config.steps
    # the pin covers every attempt, including the backward pass, which is where #125674 bites
    label_map = None
    if config.species_label_map:
        if len(config.species_label_map) != len(pool.names):
            raise ValueError(
                f"species_label_map has {len(config.species_label_map)} entries for a pool of {len(pool.names)} species"
            )
        label_map = torch.as_tensor(config.species_label_map, dtype=torch.long, device=device)

    with sdpa_context(sdpa_backend):
        while applied < config.steps and attempted < max_attempts:
            step = attempted  # attempt index, so a diagnostic record points at the batch that failed
            attempted += 1
            observed = target_mask = None
            if horizon:
                features, token_mask, targets, target_mask, species = pool.sample_horizon(generator, config.batch_size)
                if features.shape[1] != (config.history or CONTEXT_LENGTH) - 1:
                    raise ValueError(
                        f"the pool's windows have {features.shape[1]} tokens; history {config.history} needs "
                        f"{(config.history or CONTEXT_LENGTH) - 1}"
                    )
                if config.history:
                    observed = token_mask
            else:
                features, targets, species = pool.sample(generator, config.batch_size)

            # the index the conditioning path is given. It is the true species except under the shuffled-label control, and None for every 
            # arm without a lookup table; the draw uses the sampler generator, so a control run is as reproducible from its seed as any other
            cond_species = None
            if config.species_embed_dim:
                cond_species = (
                    torch.randint(
                        0,
                        config.n_species,
                        (config.batch_size,),
                        generator=generator,
                        device=device,
                    )
                    if config.random_species_labels
                    else (label_map[species] if label_map is not None else species)
                )

            optimiser.zero_grad(set_to_none=True)
            if config.use_reconstruction_head:
                token_mask = _sample_token_mask(
                    features.shape[:2], config.mask_ratio, generator, device
                )
                prediction, reconstruction = model(
                    features, token_mask=token_mask, species=cond_species
                )
                movement = features[..., :2]  # scaled east/north target for reconstruction
                forecast_loss = normalised_huber(prediction, targets)
                reconstruction_loss = masked_reconstruction_loss(
                    reconstruction,
                    movement,
                    token_mask,
                )
                loss = forecast_loss + config.reconstruction_weight * reconstruction_loss
            elif horizon:
                loss, head_losses = horizon_huber(
                    model(features, species=cond_species, observed=observed), targets, target_mask
                )
            else:
                loss = normalised_huber(model(features, species=cond_species), targets)
            if config.trunk == "moe":
                # the load balancing term from the forward just run; every other trunk never reaches this
                loss = loss + MOE_AUX_WEIGHT * model.encoder.aux_loss

            loss.backward()

            # GenAI suggested this addition; we coded this part
            # the global gradient norm before clipping; a non finite norm can indicate a gradient spike
            # that will propagate NaNs into the optimiser state. T4 computes it stably; see the helper.
            norm, nonfinite_element = _clip_grad_norm_stable(trainable, config.grad_clip)
            if diagnose:
                # the per-attempt input extreme is recorded alongside the norm so the two can be correlated afterwards
                grad_norms.append(
                    {
                        "step": int(step),
                        "applied": int(applied),
                        "grad_norm": norm,
                        "nonfinite_element": bool(nonfinite_element),
                        "loss": float(loss.detach()),
                        "max_abs_feature": float(features.abs().max()),
                    }
                )

            if not math.isfinite(norm):
                if len(nonfinite_steps) < NONFINITE_RECORD_LIMIT:
                    record = _nonfinite_record(
                        step, norm, loss.detach(), features, targets, species, pool.names, model
                    )
                    # T4 distinguishes the two causes directly rather than inferring them from whether
                    # the clip left any origin tensors behind
                    record["nonfinite_element"] = bool(nonfinite_element)
                    record["applied"] = int(applied)
                    nonfinite_steps.append(record)
                if skip_nonfinite:
                    skipped_steps += 1
                    # T4: neither the optimiser nor the schedule advances. The registered cosine curve is
                    # defined over applied updates, so a withheld attempt costs a batch and nothing else.
                    continue

            optimiser.step()
            scheduler.step()
            applied += 1

            if applied % log_every == 0 or applied == config.steps:
                history.append(float(loss.detach()))
                if horizon:
                    head_history.append([float(v) for v in head_losses.detach()])
                # report realised loss composition under the production loss
                loss_shares.append(
                    _species_loss_share(
                        model, features, targets, species, pool.names, config, cond_species, observed, target_mask
                    )
                )
            # save only states the loop has actually reached
            if checkpoint_every and applied % checkpoint_every == 0:
                save(applied)
            if progress is not None and history and applied % log_every == 0:
                progress(applied, history[-1])
            # checkpoint selection for the arms registered as allowed to see a validation set. Placed
            # after the update so the model handed to the callback is the state reached at this update.
            if monitor is not None and monitor_every and applied % monitor_every == 0:
                model.eval()
                with torch.no_grad():
                    monitor(applied, model)
                model.train()

    if applied < config.steps:
        message = (
            f"exhausted {max_attempts:,} attempted batches after only {applied:,} of "
            f"{config.steps:,} applied updates ({skipped_steps:,} withheld)"
        )
        if strict_budget:
            raise RuntimeError(
                message + ". The numerical path is broken; a run that cannot reach its registered "
                "budget must fail rather than report a short one."
            )
        # a diagnostic exists to characterise the broken path, so it returns what it managed and
        # says so. The short budget is visible in `applied_updates` and in the payload.
        print(f"WARNING: {message}; returning a short run because strict_budget is off", flush=True)

    save(applied)
    return TrainResult(
        model=model,
        config=config,
        steps_run=applied - start_step,
        attempted_batches=attempted,
        applied_updates=applied,
        loss_history=history,
        loss_shares=loss_shares,
        grad_norms=grad_norms,
        nonfinite_steps=nonfinite_steps,
        head_loss_history=head_history,
        skipped_steps=skipped_steps,
        resumed_from=start_step,
    )
