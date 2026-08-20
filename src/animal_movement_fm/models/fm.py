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

# feature indices used by the context-conditioning summary
_LOG_SCALE = FEATURES.index("log_scale")
_DT_BINS = FEATURES.index("dt_bins")

# movement derived channels hidden by the reconstruction mask
MOVEMENT_CHANNELS = tuple(
    FEATURES.index(name)
    for name in ("d_east_scaled", "d_north_scaled", "heading_cos", "heading_sin", "log_speed")
)

# context summary; mean log scale, mean dt, max dt.
CONTEXT_COND_DIM = 3


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
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.use_fm_package = bool(use_fm_package)
        self.external_cond_dim = int(external_cond_dim)
        self.use_reconstruction_head = bool(use_reconstruction_head)
        dim_feedforward = int(dim_feedforward or 2 * d_model)

        # continuous token features use a linear projection rather than an embedding lookup
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = _PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        if self.use_fm_package:
            cond_dim = CONTEXT_COND_DIM + self.external_cond_dim
            # FiLM style conditioningl; multiplicative and additive modulation from window statistics
            self.cond_proj = nn.Sequential(
                nn.Linear(cond_dim, d_model),
                nn.GELU(),
                nn.Linear(d_model, 2 * d_model),
            )

        self.head = nn.Linear(d_model, output_dim)

        # optional per token reconstruction head and learned mask embedding
        self.recon_head = nn.Linear(d_model, 2) if self.use_reconstruction_head else None
        self.mask_token = (
            nn.Parameter(torch.zeros(1, 1, d_model)) if self.use_reconstruction_head else None
        )
        if self.mask_token is not None:
            nn.init.normal_(self.mask_token, std=0.02)

    def encode(
        self,
        features: torch.Tensor,
        external_cond: torch.Tensor | None = None,
        token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Encode a feature window and return per token Transformer states.

        `token_mask marks tokens hidden for the auxiliary reconstruction objective. Their projected embeddings are replaced 
        by a learned mask token, while conditioning is still computed from the original unmasked window statistics.
        """
        x = self.input_proj(features) * math.sqrt(self.d_model)
        if token_mask is not None:
            if token_mask.dtype != torch.bool:
                token_mask = token_mask.bool()
            x = torch.where(token_mask.unsqueeze(-1), self.mask_token.to(x.dtype), x)
        x = self.pos_encoder(x)
        # no causal mask; every context token is in the past relative to the forecast target
        out = self.encoder(x)

        if self.use_fm_package:
            cond = context_conditioning(features)
            if self.external_cond_dim:
                if external_cond is None:
                    raise ValueError(
                        f"external_cond_dim={self.external_cond_dim} but no external_cond was passed"
                    )
                cond = torch.cat([cond, external_cond], dim=-1)
            gamma, beta = self.cond_proj(cond).chunk(2, dim=-1)
            out = out * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)
        return out

    def forward(
        self,
        features: torch.Tensor,
        external_cond: torch.Tensor | None = None,
        token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        out = self.encode(features, external_cond=external_cond, token_mask=token_mask)
        # use the final token state as the window summary
        pred = self.head(out[:, -1, :])
        if self.recon_head is not None:
            return pred, self.recon_head(out)
        return pred

    def n_parameters(self) -> int:
        """
        Return the number of trainable and non trainable parameters in the module.
        """
        return sum(p.numel() for p in self.parameters())
