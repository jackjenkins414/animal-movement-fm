"""
Transformer baseline for animal movement prediction. 

The 6th and final rung on the baseline before the foundation model, and (aside from the training recipes) the same
architecture that the foundation model will be built from. Where the LSTM compresses the history into a single hidden 
state passed from step to step, the transformer reads the whole window at once: self-attention lets every position 
look directly at every other position, so there is no recurrent bottleneck between a regime switch early in the window
and the prediction at the end.

Role
----
This rung asks whether attention really buys us anything over recurrence on this data. The specific target is the switching
crw p99 tail, where the LSTM and XGBoost currently tie: a regime switch is a long-range cue attention can read directly 
rather than carrying through a hidden state. If attention cannot beat the LSTM comfortably, then.....................

Note
---- 
Everything except the architecutre is held identocal to the LSTM rung: same differenced displacement inputs, same train-statistic
standardisation, same single displacement target read off the last position, same checkpoint persistence. So any difference in 
the numbers is attributable to attention vs recurrence and NOTHING else...
"""

from __future__ import annotations

import logging 
import math 
from pathlib import Path

import numpy as np 
import pandas as pd 
import torch
import torch.nn as nn

from animal_movement_fm.analysis.regime import label_contiguous_regimes
from animal_movement_fm.eval.harness import iter_windows
from animal_movement_fm.features.time import day_of_year_sincos, local_solar_sincos
from animal_movement_fm.state.dataset import STATE_TO_IDX

logger = logging.getLogger(__name__)

class _PositionalEncoding(nn.Module):
    """
    Fixed sinusoidal positional encoding (orignal method from Vaswani et al., 2017)

    Self-attention is permutation invariant: shuffle the timesteps and it returns the same answer. So we stame each timestep with 
    a position signal before attention layers see it. Each embedding dimension is a sine or cosine wave; the wave runs tfrom high 
    frequency (early dims, flip fast between adjacent steps) to low frequency (later dims, drift slowly across the window), so 
    every position gets a unique multi-scale fingerprint and nearby positions get similar ones. 

    Note
    ---- 
    Fixed, not learned, so no parameters here, so it extends to sequences longer than anything seen in training (matters for the 
    foundation model's longer windows later). Registered as a buffer so it moves with `.to(device)` and is saved in `state_dict` 
    automatically, riding the existing checkpoint round-trip for free. 

    Shapes
    -----
    input: (batch, seq_len, d_model)
    output: (batch, seq_len, d_model); same shape bc position signal added on 
    """
    def __init__(self, d_model: int, max_len: int=512) -> None:
        super().__init__()

        # pe[pos, i] is the positional signal for position `pos` in embedding dim `i`
        pe = torch.zeros(max_len, d_model)

        # (max_len, 1); the position index 0, 1, 2 ... running down the column 
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1) # Forgot unsqueeze, but need for broadcasting when multiplying by div_term so (max_len, 1)

        # d_model/2); one frequency per pair of dims, geometrically spaced from fast (1) to slow (1/1000); the log/exp form is the 
        # numerically stable way to write 10000 ** (2i / d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))

        # even dims sine, odd dims cosine of the same frequency. The sin/cos pairing lets a fixed offset k be written as a linear 
        # map of the encoding, which is what lets attention reason about relative position, not just absolute 
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # (1, max_len, d_model); the leading 1 broadcasts one encoding across the batch; before it was (max_len, d_model)
        pe = pe.unsqueeze(0)

        # buffer not parameter; fixed values, no gradient, but travels with the module 
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor: 

        # add the position signal for the first seq_len positions; broadcasts over batch
        return x + self.pe[:, : x.size(1)]


class _TransformerNet(nn.Module): 
    """
    Inner network; input projection -> positional encoding -> Transformer encoder -> head 

    Where the LSTM walked the window step by step and handed ints final hidden state to the head, this reads the whole window at once.
    Self-attention lets the last position attend directly to every earlier position, so a regime switch early in the window reaches the 
    prediction without being squeezed throgh a recurrent bottleneck. 

    Shapes
    ------
    Input: (batch, seq_len, input_dim)
    Output: (batch, output_dim)                       plain net (both flags off)
    Output: (batch, output_dim), (batch, n_states)    when use_state_head OR use_moe is on

    Two-factor state head (Rung 2)
    ------------------------------
    Identical mechanism to the LSTM rung: when `use_state_head` is on, a second linear head reads the same window
    summary (the last position after attention) and emits `n_states` state logits; the softmax of those logits (soft,
    differentiable) is concatenated onto the summary before the position head, so the displacement prediction is
    conditioned on the regime attention thinks is coming. Off -> byte-for-byte our original net.

    Mixture of experts (Rung 3)
    ---------------------------
    Also identical to the LSTM rung. So see or docstring there for it in `lstm.py`
    """

    def __init__(
            self,
            input_dim: int = 2,
            d_model: int = 64,
            nhead: int = 4,
            num_layers: int = 2,
            dim_feedforward: int = 128,
            dropout: float = 0.1,
            output_dim: int = 2,
            use_state_head: bool = False,
            n_states: int = 5,
            use_moe: bool = False,
            n_experts: int = 5,
    ) -> None:
        super().__init__()
        self.d_model = d_model

        # (input_dim -> d_model); lift each 2D displacement up to the model width so the positional waves (which live in d_model dims) can be 
        # added to it. This linear layer is the input embedding; unlike NLP token lookup it is a plain projection, because our tokens 
        # are already continuous numbers, not symbols to look uop
        self.input_proj = nn.Linear(input_dim, d_model)

        self.pos_encoder = _PositionalEncoding(d_model)

        # one encoder layer = multi-head self attention + a small feed forward network, each wrapped in a residual connection and layer norm. 
        # and we stack num_layers of them 
        # note; we feed (batch, seq, feature), matching the LSTM rung
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # (d_model -> output_dim); same role as the LSTMs `fc`; turn the window summary into a 2-element displacement
        self.use_state_head = bool(use_state_head)
        self.n_states = int(n_states)
        self.use_moe = bool(use_moe)
        self.n_experts = int(n_experts)
        if self.use_moe:
            # Mixture of experts! (Rung 3), same mechanism as our LSTM rung; see notes in that file 
            self.gate = nn.Linear(d_model, self.n_experts)
            # K position heads held as one (K*output_dim) layer so the mixture is a single batched matmul
            self.experts = nn.Linear(d_model, self.n_experts * output_dim)
            self.output_dim = int(output_dim)
        elif self.use_state_head:
            self.state_head = nn.Linear(d_model, self.n_states)
            # position head conditioned on the soft state distribution appended to the window summary
            self.head = nn.Linear(d_model + self.n_states, output_dim)
        else:
            self.head = nn.Linear(d_model, output_dim)

    def forward(self, x: torch.Tensor):
        # x; (batch, seq_len, input_dim)

        # project up then scale; the position waves are bounded in [-1, 1] so we bring the learned content to a comparable magnitude (sqrt(d_model), 
        # the standard recipe) so neither content nor position swamps the other at initialisation 
        x = self.input_proj(x) * math.sqrt(self.d_model) # (batch_ seq_len, d_model)

        # (batch_ seq_len, d_model)
        x = self.pos_encoder(x)

        # no mask; the whole window is past relative to the target and we read only the last position, so there is no future leak. Bidirectional 
        # attention is correct here; the causal mask is the foundation model's concern when it trains on every position 
        out = self.encoder(x)

        # (batch, d_model); take the last position as the window summary, mirroring the LSTMs out[:, -1, :] so the two rungs differ only in how
        # that summary was formed
        last = out[:, -1, :]

        if self.use_moe:
            gate_logits = self.gate(last)  # (batch, K)
            g = torch.softmax(gate_logits, dim=-1).unsqueeze(-1) # (batch, K, 1)
            e = self.experts(last).view(-1, self.n_experts, self.output_dim) # (batch, K, output_dim)
            pos = (g * e).sum(dim=1) # (batch, output_dim)
            # gate logits ride in the state logits slot so the training loop, predict and the harness treat MoE and the Rung 2 head identically
            return pos, gate_logits

        if self.use_state_head:
            state_logits = self.state_head(last) # (batch, n_states)
            p_state = torch.softmax(state_logits, dim=-1)  # soft, differentiable conditioning
            pos = self.head(torch.cat([last, p_state], dim=-1))  # (batch, output_dim)
            return pos, state_logits

        # (batch, output_dim)
        return self.head(last)
    
# note; this is basically the LSTM class with the inner net swapped, but _make_windows, the standardisation and training loop is the same, the only 
# new material is the config fields
class Transformer:
    """Transformer baseline (Concrete Strategy).

    Trains once globally on the displacement series via `fit`, then predicts a single next position per call via `predict`. Trained weights and 
    input normalisation can be checkpointed with `save` / `from_checkpoint`.

    Held identical to the LSTM rung except for the inner network; same differenced displacement windows, same train statistic standardisation, same 
    single displacement target read off the last position. SO difference in results is attributable to attention vs recurrence.
    """

    def __init__(
        self,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        context_length: int = 24, # match harness, change/tune later
        epochs: int = 30,
        batch_size: int = 64,
        lr: float = 1e-3,
        warmup_frac: float = 0.0,
        grad_clip: float | None = None,
        use_time_features: bool = False,
        use_season_features: bool = False,
        use_state_head: bool = False,
        state_loss_weight: float = 0.5,
        n_states: int = 5,
        use_moe: bool = False,
        n_experts: int = 5,
        device: str | None = None,
        name: str | None = None,
    ) -> None:

        # architecture & training config
        self.d_model = int(d_model)
        self.nhead = int(nhead)
        self.num_layers = int(num_layers)
        self.dim_feedforward = int(dim_feedforward)
        self.dropout = float(dropout)
        self.context_length = int(context_length)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.warmup_frac = float(warmup_frac)
        # Max global grad norm for gradient clipping. None disables it (default), so existing behaviour and tests are 
        # unchanged. A finite value (like 1.0 for ex) stabilises training by capping the update size when a batch 
        # produces an unusually large gradient
        self.grad_clip = None if grad_clip is None else float(grad_clip)
        # Per-position covariates, each appended as a (sin, cos) pair tagged on the fix the displacement lands on:
        #   1. use_time_features; local solar time of day (diurnal; varies strongly WITHIN a window, which is what lets
        #                        attention weight individual positions by it)
        #   2. use_season_features; day of year (near constant across a 6 day window, so it conditions the whole window 
        #                         rather than differentiating positions)
        # Independent flags, so raw / time / season / both are all reachable for ablation. Both default off, which keeps
        # the raw sequence model, every existing checkpoint, and the tests unchanged
        self.use_time_features = bool(use_time_features)
        self.use_season_features = bool(use_season_features)
        self.input_dim = (
            2
            + (2 if self.use_time_features else 0)
            + (2 if self.use_season_features else 0)
        )

        # two factor state head (Rung 2); a second head predicts the next fix's state and its soft distribution
        # conditions the position head. On the window summary, so it does NOT change input_dim. Off by default,
        # leaving our prev raw/time/season models, their checkpoints and tests untouched
        self.use_state_head = bool(use_state_head)
        self.state_loss_weight = float(state_loss_weight)
        self.n_states = int(n_states)

        # MoE, the gate IS a state head, so requesting both is ambiguous rather than additive; refuse it explicitly 
        # instead of silently picking one
        self.use_moe = bool(use_moe)
        self.n_experts = int(n_experts)
        if self.use_moe and self.use_state_head:
            raise ValueError(
                "use_moe and use_state_head are mutually exclusive: the MoE gate already predicts the state. "
                "Pick one."
            )
        # see lstm.py again for clarification
        if self.use_moe and self.n_experts != self.n_states:
            raise ValueError(
                f"a supervised MoE gate needs one expert per state: got n_experts={self.n_experts} but "
                f"n_states={self.n_states}."
            )
        # both paths supervise a 5 way head against the canonical state labels, so they share one weight and one
        # label builder; `state_loss_weight` is the gate loss weight when use_moe is on.
        self.uses_state_labels = self.use_moe or self.use_state_head

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # set by `fit` (or `from_checkpoint`); None on a fresh, unfitted model
        self.net: _TransformerNet | None = None
        self._mean: np.ndarray | None = None   # (2,) displacement mean
        self._std: np.ndarray | None = None     # (2,) displacement std

        # auto label in results table
        suffix = (
            ("_time" if self.use_time_features else "")
            + ("_season" if self.use_season_features else "")
            + ("_state" if self.use_state_head else "")
            + (f"_moe{self.n_experts}" if self.use_moe else "")
        )
        self.name = name or f"transformer_d{self.d_model}_l{self.num_layers}{suffix}"

    def _covariate_channels(self, dest_ts, dest_lon) -> np.ndarray:
        """
        Per-position covariate channels for the destination fixes of a window's displacements.

        Displacement i spans fix i -> i+1, so it is tagged with fix i+1's clock and date; the last displacement therefore
        carries the most recent fix's time, i.e. the departure clock of the step being predicted. Channels are appended in
        a fixed order (time of day, then season) so the input layout is stable across `fit`, `predict` and checkpoints.
        All are already bounded in [-1, 1] and are never standardised.

        Returns
        -------
        np.ndarray, shape (len(dest_ts), n_covariate_channels)
        """
        chans = []
        if self.use_time_features:
            sin, cos = local_solar_sincos(dest_ts, dest_lon)
            chans += [sin, cos]
        if self.use_season_features:
            sin, cos = day_of_year_sincos(dest_ts)
            chans += [sin, cos]
        return np.stack(chans, axis=-1)

    def _make_windows(
        self, df: pd.DataFrame, group_col: str = "individual_id"
    ) -> "tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]":
        """Turn a trajectory DataFrame into (inputs, targets, time-features) tensors.

        Identical to the LSTM rung; window definition is delegated to the harness's `iter_windows` so the net trains on exactly the windows it
        is evaluated on. Each window's (context_length, 2) position history is differenced into (context_length - 1, 2) input displacements;
        the target is the single displacement from the last history position to the target position.

        When `use_time_features` is on, each displacement is also tagged with the local-solar (sin, cos) of the fix it lands on: displacement i 
        spans fix i -> i+1, so it carries fix i+1's clock. The last displacement therefore carries the most recent fix's time, i.e, the departure 
        clock of the step being predicted. Time features are left raw (already in [-1, 1]); only the displacement channels are standardised, in `fit`.

        Returns
        -------
        X : torch.Tensor, shape (n_windows, context_length - 1, 2)
        Y : torch.Tensor, shape (n_windows, 2)
        T : torch.Tensor | None, shape (n_windows, context_length - 1, 2) of (sin, cos), or None when time features are disabled.
        """
        xs, ys, ts = [], [], []
        needs_ts = self.use_time_features or self.use_season_features
        for _, traj in df.groupby(group_col, sort=False):
            traj_ts = traj["timestamp"].to_numpy() if needs_ts else None
            for history, target, _ts, t_idx in iter_windows(traj, self.context_length):
                xs.append(np.diff(history, axis=0)) # (W-1, 2) input displacements
                ys.append(target - history[-1]) # (2,) target displacement
                if needs_ts:
                    win_ts = traj_ts[t_idx - self.context_length : t_idx] # (W,) row aligned with history
                    ts.append(self._covariate_channels(win_ts[1:], history[1:, 0])) # destination fixes 1..W-1

        if not xs:
            raise ValueError(f"No windows produced: every individual has fewer than " f"{self.context_length + 1} positions.")

        X = torch.tensor(np.stack(xs), dtype=torch.float32)
        Y = torch.tensor(np.stack(ys), dtype=torch.float32)
        T = torch.tensor(np.stack(ts), dtype=torch.float32) if needs_ts else None
        return X, Y, T

    def _state_labels(self, df: pd.DataFrame, group_col: str = "individual_id") -> torch.Tensor:
        """Per-window next state labels, aligned one-to-one with `_make_windows` output.

        Identical to the LSTM rung: re-runs the same grouping + `iter_windows` loop so the i-th label matches the i-th
        (X, Y) window, and labels the window's TARGET fix with the canonical refined state (`state[t_idx]`) — the same
        fix whose displacement the position head predicts, and the same label the standalone classifier uses. Keyed on
        (segment_id, timestamp) so alignment is order-independent; requires a `segment_id` column.

        Returns
        -------
        torch.Tensor, shape (n_windows,), dtype long : integer state index (STATE_TO_IDX) per window.
        """
        if "segment_id" not in df.columns:
            raise ValueError(
                "use_state_head=True requires a 'segment_id' column to derive next state labels; "
                "the segment produc provides it."
            )
        labelled = label_contiguous_regimes( df, refined=True, min_run_length=self.context_length + 1)
        state_by_fix = labelled.set_index(["segment_id", "timestamp"])["state"].map(STATE_TO_IDX)

        ss: list[int] = []
        for _, traj in df.groupby(group_col, sort=False):
            idx = pd.MultiIndex.from_arrays([traj["segment_id"], traj["timestamp"]])
            traj_state = state_by_fix.reindex(idx).to_numpy()
            for _history, _target, _ts, t_idx in iter_windows(traj, self.context_length):
                ss.append(int(traj_state[t_idx]))

        if not ss:
            raise ValueError(
                f"No windows produced for state labels: every individual has fewer than "
                f"{self.context_length + 1} positions"
            )
        return torch.tensor(ss, dtype=torch.long)

    def fit(self, train_df: pd.DataFrame, group_col: str = "individual_id") -> "Transformer":
        """
        Train the network once on all trajectories in `train_df`.

        Builds displacement windows, standardises the inputs (storing the per-channel mean/std so `predict` can apply the identical transform), 
        then runs a standard mini-batch training loop. Returns self so callers can write `model = Transformer().fit(train_df)`.
        """
        X, Y, T = self._make_windows(train_df, group_col=group_col)   # (N, W-1, 2), (N, 2), (N, W-1, 2)|None

        # standardise the DISPLACEMENT channels using TRAIN statistics only, and remember them; predict() must reuse these exact stats. Time
        # features (if any) are already bounded in [-1, 1] and are NOT standardised.
        self._mean = X.mean(dim=(0, 1), keepdim=True) # (1, 1, 2)
        self._std = X.std(dim=(0, 1), keepdim=True).clamp_min(1e-8) # avoid /0 duhh (forgot this lol)
        Xn = (X - self._mean) / self._std

        # append the raw covariate channels AFTER standardisation, so the net sees (N, W-1, input_dim)
        if T is not None:
            Xn = torch.cat([Xn, T], dim=-1)

        # per window next state labels for the state head, aligned to (Xn, Y) by construction (same window loop)
        S = self._state_labels(train_df, group_col=group_col) if self.uses_state_labels else None

        # GPU preload: move the whole training tensor set onto the compute device once and index batches on device, rather than streaming through a
        # DataLoader that copies each batch CPU -> GPU. Our stork set is ~170 MB and so that fits easily in the V100's 32 GB, so this removes the
        # per-batch transfer + single process loader that starved the GPU in our prev iter1 (util ~15%). Shuffling still draws from torch's global RNG via
        # randperm, so the seeded determinism test is still unaffected
        Xn = Xn.to(self.device) # live on GPU
        Y = Y.to(self.device) # live on GPU
        if S is not None:
            S = S.to(self.device)
        n_windows = Xn.size(0)
        n_batches = math.ceil(n_windows / self.batch_size)

        self.net = _TransformerNet(
            input_dim=self.input_dim,
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            output_dim=2,
            use_state_head=self.use_state_head,
            n_states=self.n_states,
            use_moe=self.use_moe,
            n_experts=self.n_experts,
        ).to(self.device)

        optimiser = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        ############ GENAI REF: Will add details later
        # Warmup + cosine decay. warmup_frac<=0 is a constant-lr no-op, so the committed
        # default behaviour (and the passing tests) are unchanged; warmup_frac>0 applies the
        # transformer-standard recipe where `lr` is the peak reached at the end of warmup.
        total_steps = max(1, self.epochs * n_batches)
        warmup_steps = int(self.warmup_frac * total_steps)

        def _lr_scale(step: int) -> float:
            if self.warmup_frac <= 0.0:
                return 1.0
            if step < warmup_steps:
                return (step + 1) / max(1, warmup_steps)            # linear 0 -> 1
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * progress))       # cosine 1 -> 0

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimiser, _lr_scale)
        #################

        loss_fn = nn.MSELoss()
        ce_fn = nn.CrossEntropyLoss() if self.uses_state_labels else None

        self.net.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            epoch_state = 0.0
            # reshuffle each epoch; randperm on device, drawn from torch's global RNG
            perm = torch.randperm(n_windows, device=self.device)
            for start in range(0, n_windows, self.batch_size):
                idx = perm[start : start + self.batch_size]
                xb = Xn[idx]
                yb = Y[idx]

                optimiser.zero_grad()
                if self.uses_state_labels:
                    pred, state_logits = self.net(xb)
                    loss_pos = loss_fn(pred, yb)
                    loss_state = ce_fn(state_logits, S[idx])
                    # joint objective; position MSE (primary) + weighted state cross-entropy (auxiliary conditioning)
                    loss = loss_pos + self.state_loss_weight * loss_state
                    epoch_state += loss_state.item() * xb.size(0)
                else:
                    pred = self.net(xb)
                    loss = loss_fn(pred, yb)
                loss.backward()
                if self.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.grad_clip)
                optimiser.step()
                scheduler.step()

                epoch_loss += loss.item() * xb.size(0)

            epoch_loss /= n_windows
            if self.uses_state_labels:
                epoch_state /= n_windows
                logger.info(
                    "epoch %d/%d train_loss=%.6g state_ce=%.6g",
                    epoch + 1, self.epochs, epoch_loss, epoch_state,
                )
            else:
                logger.info("epoch %d/%d train_mse=%.6g", epoch + 1, self.epochs, epoch_loss)
        return self

    def predict(self, context) -> np.ndarray:
        """
        Predict the next (lon, lat) position from a window of history.

        Mirrors the input pipeline in `_make_windows`; the (W, 2) position history is differenced into (W-1, 2) displacements, standardised
        with the stored train stats, and passed through the net to predict the next displacement, which is added back onto the last 
        observed position.

        Parameters
        ----------
        context : PredictionContext, carrying `history` of shape (W, 2), columns (lon, lat). The transformer uses only the history; 
        the timestamp is ignored as well for now...

        Returns
        -------
        np.ndarray, shape (2,), the predicted next (lon, lat).
        """
        if self.net is None:
            raise RuntimeError("Model is not fitted. Call fit or load a checkpoint before predict")

        history = np.asarray(context.history, dtype=np.float32)
        if history.ndim != 2 or history.shape[1] != 2:
            raise ValueError(f"history must have shape (W, 2), we got {history.shape} instead")

        # single observation; no displacement to compute, so fall back to persistence (predict no movement), matching the harness 
        # contract that predict returns a (2,)
        if history.shape[0] < 2:
            return history[-1].copy()

        # (W-1, 2)
        disp = np.diff(history, axis=0)

        # standardise with the stored TRAIN stats (never computed here)
        mean = self._mean.detach().cpu().numpy().reshape(1, 2)
        std = self._std.detach().cpu().numpy().reshape(1, 2)
        disp_n = (disp - mean) / std

        # append the same destination tagged local-solar (sin, cos) built in _make_windows, using the per fix timestamps the 
        # harness threads through the context
        if self.use_time_features or self.use_season_features:
            if context.history_timestamps is None:
                raise ValueError(
                    "model was fitted with time/season features but context.history_timestamps is None; "
                    "the harness populates it, so this is a direct predict() call missing the times."
                )
            win_ts = np.asarray(context.history_timestamps)
            cov = self._covariate_channels(win_ts[1:], history[1:, 0]).astype(np.float32)  # (W-1, n_cov)
            feat = np.concatenate([disp_n, cov], axis=-1)  # (W-1, input_dim)
        else:
            feat = disp_n

        # (W-1, C) -> (1, W-1, C): the net expects a batch dimension
        x = torch.tensor(feat, dtype=torch.float32).unsqueeze(0).to(self.device)

        self.net.eval()
        with torch.no_grad():
            out = self.net(x)
            # with the state head on, the net returns (pos, state_logits); the state head conditions the position
            # internally, so predict still returns only the (lon, lat) position and the harness contract is unchanged.
            pred_disp = out[0] if self.uses_state_labels else out  # (1, 2)

        pred_disp = pred_disp.squeeze(0).cpu().numpy() # (2,)
        return history[-1] + pred_disp
    
# Note; this has been copied from the lstm.py and kept the majority of the structure the same (same save/load structure, docstirngs, torch.save
# scalar stats, path, handling, state dict loading, return logic). The only difference is the class type, config keys, contructor arguments 
# and network class, since we changed these to match the Transformer-specific config 
    def save(self, path: str | Path) -> None:
        """
        Save trained weights, scaler stats, and architecture to `path`.
        A checkpoint captures everything `predict` needs: the network weights, the input-normalisation stats, and the architecture required to
        rebuild the net before loading the weights into it. 
        """
        if self.net is None:
            raise RuntimeError("Nothing to save: model is not fitted.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(
            {
                "state_dict": self.net.state_dict(),
                "mean": self._mean,
                "std": self._std,
                "config": {
                    "d_model": self.d_model,
                    "nhead": self.nhead,
                    "num_layers": self.num_layers,
                    "dim_feedforward": self.dim_feedforward,
                    "dropout": self.dropout,
                    "context_length": self.context_length,
                    "use_time_features": self.use_time_features,
                    "use_season_features": self.use_season_features,
                    "use_state_head": self.use_state_head,
                    "use_moe": self.use_moe,
                    "n_experts": self.n_experts,
                    "state_loss_weight": self.state_loss_weight,
                    "n_states": self.n_states,
                    "name": self.name,
                },
            },
            path,
        )

    @classmethod
    def from_checkpoint(cls, path: str | Path, device: str | None = None) -> "Transformer":
        """
        Rebuild a fitted model from a checkpoint written by `save`.
        Reconstructs the model from the stored architecture, loads the weights, and restores the scaler stats; yielding an instance ready 
        to `predict` without any retraining.
        """
        path = Path(path)
        # weights_only=False: this is our own trusted checkpoint and it contains a non-tensor `config` dict; explicit here to 
        # silence the FutureWarning and stay correct when torch flips the default to True
        ckpt = torch.load(path, map_location=device or "cpu", weights_only=False)
        cfg = ckpt["config"]
        # old checkpoints predate these flags -> raw sequence, no state head
        use_time = cfg.get("use_time_features", False)
        use_season = cfg.get("use_season_features", False)
        use_state_head = cfg.get("use_state_head", False)
        use_moe = cfg.get("use_moe", False)
        n_experts = cfg.get("n_experts", 5)
        n_states = cfg.get("n_states", 5)

        # rebuild the wrapper with the stored architecture.
        model = cls(
            d_model=cfg["d_model"],
            nhead=cfg["nhead"],
            num_layers=cfg["num_layers"],
            dim_feedforward=cfg["dim_feedforward"],
            dropout=cfg["dropout"],
            context_length=cfg["context_length"],
            use_time_features=use_time,
            use_season_features=use_season,
            use_state_head=use_state_head,
            use_moe=use_moe,
            n_experts=n_experts,
            state_loss_weight=cfg.get("state_loss_weight", 0.5),
            n_states=n_states,
            device=device,
            name=cfg["name"],
        )

        # rebuild the net to match, load weights, restore scaler stats.
        model.net = _TransformerNet(
            input_dim=model.input_dim,
            d_model=cfg["d_model"],
            nhead=cfg["nhead"],
            num_layers=cfg["num_layers"],
            dim_feedforward=cfg["dim_feedforward"],
            dropout=cfg["dropout"],
            output_dim=2,
            use_state_head=use_state_head,
            n_states=n_states,
            use_moe=use_moe,
            n_experts=n_experts,
        ).to(model.device)
        model.net.load_state_dict(ckpt["state_dict"])
        model._mean = ckpt["mean"]
        model._std = ckpt["std"]

        return model
