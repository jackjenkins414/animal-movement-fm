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
from torch.utils.data import DataLoader, TensorDataset

from animal_movement_fm.eval.harness import iter_windows

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
    Output: (batch, output_dim)
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
        self.head = nn.Linear(d_model, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor: 
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

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # set by `fit` (or `from_checkpoint`); None on a fresh, unfitted model
        self.net: _TransformerNet | None = None
        self._mean: np.ndarray | None = None   # (2,) displacement mean
        self._std: np.ndarray | None = None     # (2,) displacement std

        # auto label in results table
        self.name = name or f"transformer_d{self.d_model}_l{self.num_layers}"

    def _make_windows(self, df: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor]:
        """Turn a trajectory DataFrame into (inputs, targets) displacement tensors.

        Identical to the LSTM rung; window definition is delegated to the harness's `iter_windows` so the net trains on exactly the windows it 
        is evaluated on. Each window's (context_length, 2) position history is differenced into (context_length - 1, 2) input displacements; 
        the target is the single displacement from the last history position to the target position.

        Returns
        -------
        X : torch.Tensor, shape (n_windows, context_length - 1, 2)
        Y : torch.Tensor, shape (n_windows, 2)
        """
        xs, ys = [], []
        for _, traj in df.groupby("individual_id", sort=False):
            for history, target, _ts, _t in iter_windows(traj, self.context_length):
                xs.append(np.diff(history, axis=0)) # (W-1, 2) input displacements
                ys.append(target - history[-1]) # (2,) target displacement

        if not xs:
            raise ValueError(f"No windows produced: every individual has fewer than " f"{self.context_length + 1} positions.")

        X = torch.tensor(np.stack(xs), dtype=torch.float32)
        Y = torch.tensor(np.stack(ys), dtype=torch.float32)
        return X, Y

    def fit(self, train_df: pd.DataFrame) -> "Transformer":
        """
        Train the network once on all trajectories in `train_df`.

        Builds displacement windows, standardises the inputs (storing the per-channel mean/std so `predict` can apply the identical transform), 
        then runs a standard mini-batch training loop. Returns self so callers can write `model = Transformer().fit(train_df)`.
        """
        X, Y = self._make_windows(train_df) # (N, W-1, 2), (N, 2)

        # standardise inputs using TRAIN statistics only, and remember them; predict() must reuse these exact stats, so they are stored on self
        self._mean = X.mean(dim=(0, 1), keepdim=True) # (1, 1, 2)
        self._std = X.std(dim=(0, 1), keepdim=True).clamp_min(1e-8) # avoid /0 duhh (forgot this lol)
        Xn = (X - self._mean) / self._std

        # targets left in raw displacement units, exactly as the LSTM rung
        loader = DataLoader(TensorDataset(Xn, Y), batch_size=self.batch_size, shuffle=True)

        self.net = _TransformerNet(
            input_dim=2,
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            output_dim=2,
        ).to(self.device)

        optimiser = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        ############ GENAI REF: Will add details later
        # Warmup + cosine decay. warmup_frac<=0 is a constant-lr no-op, so the committed
        # default behaviour (and the passing tests) are unchanged; warmup_frac>0 applies the
        # transformer-standard recipe where `lr` is the peak reached at the end of warmup.
        total_steps = max(1, self.epochs * len(loader))
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

        self.net.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)

                optimiser.zero_grad()
                pred = self.net(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                optimiser.step()
                scheduler.step()

                epoch_loss += loss.item() * xb.size(0)

            epoch_loss /= len(loader.dataset)
            logger.info("epoch %d/%d train_mse=%.6g", epoch + 1, self.epochs, epoch_loss)
        return self

    def predict(self, history: np.ndarray) -> np.ndarray:
        """
        Predict the next (lon, lat) position from a window of history.

        Mirrors the input pipeline in `_make_windows`; the (W, 2) position history is differenced into (W-1, 2) displacements, standardised
        with the stored train stats, and passed through the net to predict the next displacement, which is added back onto the last 
        observed position.

        Parameters
        ----------
        history : np.ndarray, shape (W, 2), columns (lon, lat).

        Returns
        -------
        np.ndarray, shape (2,), the predicted next (lon, lat).
        """
        if self.net is None:
            raise RuntimeError("Model is not fitted. Call fit or load a checkpoint before predict")

        history = np.asarray(history, dtype=np.float32)
        if history.ndim != 2 or history.shape[1] != 2:
            raise ValueError(f"history must have shape (W, 2), we got {history.shape} instead")

        # single observation; no displacement to compute, so fall back to persistence (predict no movement), matching the harness 
        # contract that predict returns a (2,)
        if history.shape[0] < 2:
            return history[-1].copy()

        # (W-1, 2)
        disp = np.diff(history, axis=0) 

        # standardise with the stored TRAIN stats (never computed here)
        mean = self._mean.numpy().reshape(1, 2)
        std = self._std.numpy().reshape(1, 2)
        disp_n = (disp - mean) / std

        # (W-1, 2) -> (1, W-1, 2): the net expects a batch dimension
        x = torch.tensor(disp_n, dtype=torch.float32).unsqueeze(0).to(self.device)

        self.net.eval()
        with torch.no_grad():
            pred_disp = self.net(x) # (1, 2)

        pred_disp = pred_disp.squeeze(0).cpu().numpy() # (2,)
        return history[-1] + pred_disp
    
# Note; this has been copied from the lstm.py and kept the majority of the structure the same (same save/load structure, docstirngs, torch.save
# scalar stats, path, handling, state dict loading, return logic). The only difference is the class type, config keys, contructor arguments 
# and network class, since we changed these to match the Transformer-specific config 
    def save(self, path: str | Path) -> None:
        """Save trained weights, scaler stats, and architecture to `path`.
        A checkpoint captures everything `predict` needs: the network weights,
        the input-normalisation stats, and the architecture required to
        rebuild the net before loading the weights into it. This decouples
        expensive training (on NCI) from cheap, reproducible evaluation.
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
                    "name": self.name,
                },
            },
            path,
        )

    @classmethod
    def from_checkpoint(cls, path: str | Path, device: str | None = None) -> "Transformer":
        """Rebuild a fitted model from a checkpoint written by `save`.
        Reconstructs the model from the stored architecture, loads the weights,
        and restores the scaler stats — yielding an instance ready to `predict`
        without any retraining.
        """
        path = Path(path)
        ckpt = torch.load(path, map_location=device or "cpu")
        cfg = ckpt["config"]

        # rebuild the wrapper with the stored architecture.
        model = cls(
            d_model=cfg["d_model"],
            nhead=cfg["nhead"],
            num_layers=cfg["num_layers"],
            dim_feedforward=cfg["dim_feedforward"],
            dropout=cfg["dropout"],
            context_length=cfg["context_length"],
            device=device,
            name=cfg["name"],
        )

        # rebuild the net to match, load weights, restore scaler stats.
        model.net = _TransformerNet(
            input_dim=2,
            d_model=cfg["d_model"],
            nhead=cfg["nhead"],
            num_layers=cfg["num_layers"],
            dim_feedforward=cfg["dim_feedforward"],
            dropout=cfg["dropout"],
            output_dim=2,
        ).to(model.device)
        model.net.load_state_dict(ckpt["state_dict"])
        model._mean = ckpt["mean"]
        model._std = ckpt["std"]

        return model