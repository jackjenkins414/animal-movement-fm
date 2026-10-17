"""
Tier 2 foundation model used for both the pooled baseline and the FM variant.

The model shares one Transformer trunk across both configurations. Our FM package adds context derived conditioning and a masked 
reconstruction head, while the pooled baseline uses the same trunk and forecast head without those additions. Conditioning is then
derived from the current window so it remains defined for unseen species.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from animal_movement_fm.data.enu import FEATURES, N_FEATURES
from animal_movement_fm.models.moe import MoEEncoder

# feature indices used by the context-conditioning summary
_LOG_SCALE = FEATURES.index("log_scale")
_DT_BINS = FEATURES.index("dt_bins")

# movement derived channels hidden by the reconstruction mask
MOVEMENT_CHANNELS = tuple(
    FEATURES.index(name)
    for name in ("d_east_scaled", "d_north_scaled", "heading_cos", "heading_sin", "log_speed")
)

# context summary; mean log scale, mean dt, max dt
CONTEXT_COND_DIM = 3

# transformer repair probe; each switch defaults to the behaviour every earlier arm was trained with
INPUT_SCALES = ("sqrt_d", "none")
STEP_TRANSFORMS = ("none", "signed_log1p")
STEP_CHANNELS = (FEATURES.index("d_east_scaled"), FEATURES.index("d_north_scaled"))
if STEP_CHANNELS != (0, 1):
    raise RuntimeError("signed_log1p assumes the two displacement channels lead the token")


def signed_log1p(features: torch.Tensor) -> torch.Tensor:
    """sign(x) log1p(|x|) on d_east_scaled and d_north_scaled of every token; every other channel is untouched"""
    steps = features[..., :2]
    return torch.cat([torch.sign(steps) * torch.log1p(steps.abs()), features[..., 2:]], dim=-1)


class _PositionalEncoding(nn.Module):
    """
    Fixed sinusoidal positional encoding; from Vaswani, 2017.

    Stored as a buffer so it moves with the module and is preserved in checkpoints without becoming a trainable parameter.
    """

    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


def context_conditioning(features: torch.Tensor) -> torch.Tensor:
    """
    Summarise each context window into scale/cadence statistics for conditioning.

    `features` has shape `(batch, seq, N_FEATURES)` and the result has shape `(batch, CONTEXT_COND_DIM)`; with all components 
    are derived from the current window, so the same conditioning is available for unseen species at inference.
    """
    log_scale = features[:, :, _LOG_SCALE].mean(dim=1)
    dt_mean = features[:, :, _DT_BINS].mean(dim=1)
    dt_max = features[:, :, _DT_BINS].max(dim=1).values
    return torch.stack([log_scale, dt_mean, dt_max], dim=-1)


class MovementFM(nn.Module):
    """
    Cross-species one step movement forecaster (for now; multi-step is what we want next in the future).

    Inputs are normalised feature windows from `data.enu.encode_window`; outputs are the next `(east, north)` displacement in 
    the same normalised space. Setting `use_fm_package=False` gives the pooled baseline with the same Transformer trunk but 
    no conditioning path or reconstruction head.
    """

    def __init__(
        self,
        input_dim: int = N_FEATURES,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int | None = None,
        dropout: float = 0.1,
        output_dim: int = 2,
        *,
        use_fm_package: bool = True,
        external_cond_dim: int = 0,
        use_reconstruction_head: bool = False,
        use_context_conditioning: bool = True,
        n_species: int = 0,
        species_embed_dim: int = 0,
        trunk: str = "transformer",
        n_horizons: int = 0,
        observed_flag: bool = False,
        input_scale: str = "sqrt_d",
        norm_first: bool = False,
        step_transform: str = "none",
    ) -> None:
        super().__init__()
        # `trunk="lstm"` is our Tier 3 baseline-ladder rung; we reuse the whole Tier 2 pipeline (representation, pools, loss, schedule, selection, 
        # predictor) and swap only the sequence encoder. It is a plain trunk, B style: no conditioning path and no reconstruction head.
        if trunk not in ("transformer", "lstm", "moe"):
            raise ValueError(f"unknown trunk {trunk!r}; use 'transformer', 'lstm' or 'moe'")
        if trunk == "lstm" and (use_fm_package or use_reconstruction_head):
            raise ValueError("the LSTM trunk is a plain baseline rung; switch the FM package and reconstruction head off")
        self.trunk = trunk
        self.d_model = int(d_model)
        self.use_fm_package = bool(use_fm_package)
        self.external_cond_dim = int(external_cond_dim)
        self.use_reconstruction_head = bool(use_reconstruction_head)
        # Tier E; the conditioning path's inputs are now selectable. The registered arms take the three window statistics and nothing else, 
        # which is the default and is unchanged
        self.use_context_conditioning = bool(use_context_conditioning)
        self.n_species = int(n_species)
        self.species_embed_dim = int(species_embed_dim)
        self.n_horizons = int(n_horizons)
        if self.n_horizons < 0 or (self.n_horizons and output_dim != 2):
            raise ValueError("n_horizons > 0 gives each horizon an (east, north) head; leave output_dim at 2")
        # the flag is the last of input_dim's channels; the Transformer masks instead of flagging
        self.observed_flag = bool(observed_flag)
        if self.observed_flag and trunk != "lstm":
            raise ValueError("the observed flag is the LSTM's input; a Transformer trunk takes the padding mask instead")
        # repair probe switches; input_scale and norm_first are the Transformer trunk's; step_transform reads any trunk's input
        if input_scale not in INPUT_SCALES:
            raise ValueError(f"input_scale {input_scale!r}; use one of {INPUT_SCALES}")
        if step_transform not in STEP_TRANSFORMS:
            raise ValueError(f"step_transform {step_transform!r}; use one of {STEP_TRANSFORMS}")
        if trunk != "transformer" and (input_scale != "sqrt_d" or norm_first):
            raise ValueError("input_scale and norm_first belong to the Transformer trunk only")
        if step_transform != "none" and use_reconstruction_head:
            raise ValueError("the reconstruction head's targets are the raw steps; step_transform is for the forecast arms")
        self.input_scale = input_scale
        self.norm_first = bool(norm_first)
        self.step_transform = step_transform
        dim_feedforward = int(dim_feedforward or 2 * d_model)

        # continuous token features use a linear projection rather than an embedding lookup
        self.input_proj = nn.Linear(input_dim, d_model)
        if trunk == "lstm":
            # recurrence carries order, so there is no positional encoding
            self.pos_encoder = None
            self.encoder = nn.LSTM(
                d_model,
                d_model,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
        elif trunk == "moe":
            # pooling study; every layer's dense feedforward block becomes routed experts; the rest of the trunk and the path through encode(), 
            # is the Transformers
            self.pos_encoder = _PositionalEncoding(d_model)
            self.encoder = MoEEncoder(d_model, nhead, dim_feedforward, num_layers, dropout)
        else:
            self.pos_encoder = _PositionalEncoding(d_model)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
                norm_first=self.norm_first,
            )
            if self.norm_first:
                # Pre-LN leaves the residual stream unnormalised, so the stack ends in its own LayerNorm before the head;
                # torch skips the nested tensor path for norm_first layers, so it is switched off rather than warned about
                self.encoder = nn.TransformerEncoder(
                    encoder_layer, num_layers=num_layers, norm=nn.LayerNorm(d_model), enable_nested_tensor=False
                )
            else:
                self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # A learned per-species lookup; it exists only for species that were trained on, so an arm carrying it is not zero-shot on a species 
        # with no row; that is the whole point of the Tier E experiment it was added for, and it is why nothing registered uses it
        self.species_embed = None
        self.cond_proj = None
        if self.use_fm_package:
            if self.species_embed_dim:
                if self.n_species < 1:
                    raise ValueError(
                        f"species_embed_dim={self.species_embed_dim} needs n_species >= 1; a lookup table with no rows cannot be indexed"
                    )
                self.species_embed = nn.Embedding(self.n_species, self.species_embed_dim)
                nn.init.normal_(self.species_embed.weight, std=0.02)

            cond_dim = (
                (CONTEXT_COND_DIM if self.use_context_conditioning else 0)
                + self.species_embed_dim
                + self.external_cond_dim
            )
            if cond_dim == 0:
                raise ValueError(
                    "the FM package is enabled but every conditioning input is switched off, so cond_proj would have no input. Disable " \
                    "use_fm_package instead"
                )
            # FiLM style conditioningl; multiplicative and additive modulation from window statistics
            self.cond_proj = nn.Sequential(
                nn.Linear(cond_dim, d_model),
                nn.GELU(),
                nn.Linear(d_model, 2 * d_model),
            )

        self.head = nn.Linear(d_model, 2 * self.n_horizons if self.n_horizons else output_dim)

        # optional per token reconstruction head and learned mask embedding
        self.recon_head = nn.Linear(d_model, 2) if self.use_reconstruction_head else None
        self.mask_token = (
            nn.Parameter(torch.zeros(1, 1, d_model)) if self.use_reconstruction_head else None
        )
        if self.mask_token is not None:
            nn.init.normal_(self.mask_token, std=0.02)

    def conditioning_inputs(
        self,
        features: torch.Tensor,
        external_cond: torch.Tensor | None = None,
        species: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Assemble the vector handed to `cond_proj`.

        Concatenated in a fixed order (window statistics, species embedding, external descriptors) so a checkpoint's conditioning weights 
        always mean the same thing. The registered arms supply only the first block, and the other two are zero-width for them.
        """
        parts = []
        if self.use_context_conditioning:
            parts.append(context_conditioning(features))
        if self.species_embed is not None:
            if species is None:
                raise ValueError(
                    "this model carries a learned species embedding, so it cannot be run without a species index. A species with no row in " \
                    "the table has no embedding to look up, which is exactly why this arm is not zero-shot"
                )
            parts.append(self.species_embed(species))
        if self.external_cond_dim:
            if external_cond is None:
                raise ValueError(
                    f"external_cond_dim={self.external_cond_dim} but no external_cond was passed"
                )
            parts.append(external_cond)
        return torch.cat(parts, dim=-1)

    def encode(
        self,
        features: torch.Tensor,
        external_cond: torch.Tensor | None = None,
        token_mask: torch.Tensor | None = None,
        species: torch.Tensor | None = None,
        *,
        observed: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Encode a feature window and return per token hidden states.

        `observed` marks valid history tokens. The Transformer ignores missing tokens, while the LSTM uses an observed flag. The final 
        token is always observed.

        `token_mask` marks tokens hidden for reconstruction, replacing their embeddings with a learned mask token. Conditioning still 
        uses the original features.
        """
        if observed is not None:
            observed = observed.bool()
            if self.use_fm_package or self.use_reconstruction_head:
                raise ValueError(
                    "the FM package takes no padded history: its conditioning averages over every token, padding included"
                )
        if self.step_transform == "signed_log1p":
            # repair probe V3; the two step channels only, after the data's +-1,000 clip; masked tokens are 0 and stay 0,
            # and the conditioning path's log_scale and dt channels are untouched
            features = signed_log1p(features)
        if self.trunk == "lstm":
            if self.observed_flag:
                if observed is None:
                    raise ValueError("this LSTM reads an observed flag, so it needs `observed`")
                features = torch.cat([features, observed.unsqueeze(-1).to(features.dtype)], dim=-1)
            elif observed is not None:
                raise ValueError("an LSTM cannot mask a token; build it with observed_flag to read a padded history")
            # no sqrt(d_model) scaling; that pairs with the additive positional encoding, and the LSTM's tanh gates would saturate on it
            out, _ = self.encoder(self.input_proj(features))
            return out
        if observed is not None and self.trunk == "moe":
            raise ValueError("the MoE trunk has no padding mask")
        x = self.input_proj(features)
        if self.input_scale == "sqrt_d":  # every arm before the repair probe; "none" is its V1
            x = x * math.sqrt(self.d_model)
        if token_mask is not None:
            if token_mask.dtype != torch.bool:
                token_mask = token_mask.bool()
            x = torch.where(token_mask.unsqueeze(-1), self.mask_token.to(x.dtype), x)
        x = self.pos_encoder(x)
        # no causal mask; every context token is in the past relative to the forecast target
        out = self.encoder(x) if observed is None else self.encoder(x, src_key_padding_mask=~observed)

        if self.use_fm_package:
            cond = self.conditioning_inputs(features, external_cond=external_cond, species=species)
            gamma, beta = self.cond_proj(cond).chunk(2, dim=-1)
            out = out * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)
        return out

    def forward(
        self,
        features: torch.Tensor,
        external_cond: torch.Tensor | None = None,
        token_mask: torch.Tensor | None = None,
        species: torch.Tensor | None = None,
        *,
        observed: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        out = self.encode(
            features, external_cond=external_cond, token_mask=token_mask, species=species, observed=observed
        )
        # use the final token state as the window summary
        pred = self.head(out[:, -1, :])
        if self.n_horizons:
            pred = pred.view(-1, self.n_horizons, 2)
        if self.recon_head is not None:
            return pred, self.recon_head(out)
        return pred

    def n_parameters(self) -> int:
        """
        Return the number of trainable and non trainable parameters in the module.
        """
        return sum(p.numel() for p in self.parameters())
