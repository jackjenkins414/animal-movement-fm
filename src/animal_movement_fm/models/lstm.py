"""
LSTM baseline for animal movement prediction. 

The 5th rung on our baseline ladder, and this is the first model that learns its own representation 
of the sequence rather than reading off hand-engineered features, and the first that is trained 
once and reloaded from a checkpoint rather than fit inline. 

Role: 
-----
XGBoost set the 'non-linear with hand engineered features' ceiling. We hope the LSTM sets the 
'learned representation with recurrence ceiling'; how much signal is recoverable when the model 
is free to learn what to remember from the raw displacement sequence. 
If the transformer cannot beat this comfortably, then attention is not buying us anything over 
recurrence for this data; a meaningful result either way. 
"""

from __future__ import annotations

import logging
from pathlib import Path 

import numpy as np 
import pandas as pd 
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset 

logger = logging.getLogger(__name__)

class _LSTMNet(nn.Module):
    """
    Inner network: LSTM encoder + linear head

    Shapes
    ------
    input: (batch, seq_len, input_dim)
    output: (batch, output_dim)
    """

    def __innit__(
            self, 
            input_dim: int = 2, 
            hidden_dim: int = 64, # DEFAULT FOR NOW 
            num_layers: int = 1, # SAME HERE
            output_dim: int = 2,
    ) -> None: 
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input-dim)

        # out: (batch, seq_len, hidden_dim)
        out, _ = self.lstm(x)

        # (batch, hidden_dim); summary of the window 
        last = out[:, -1, :]

        # (batch, output_dim)
        return self.fc(last)
    
class LSTM: 
    """LSTM baseline (Concrete Strategy). 

    Trains once globally on the displacement series via `fit`, then predicts a single next 
    position per call via  `predit`. Trained weights and input normalisation can be 
    checkpointed with `save` / `from_checkpoint`. 
    """

    def __init__(
        self, 
        hidden_dim: int = 64, 
        num_layers: int = 1, 
        context_length: int = 24, # match harness, change/tune later 
        epochs: int =30, 
        batch_size: int = 64, 
        lr: float = 1e-3, 
        device: str | None = None, 
        name: str | None = None, 
    ) -> None: 
        
        # architecture & training cofig 
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.context_length = int(context_length)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.lr = float(lr)

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # set by `fit` (or `from_checkpoint`); None on a fresh, unfitted model 
        self.next: _LSTMNet | None = None 
        self._mean: np.ndarray | None = None # (2,) displacement mean
        self._std: np.ndarray | None = None # (2,) displacement std 


        # auto label in results table 
        self.name = name or f"lstm_h{self.hidden_dim}_l{self.num_layers}"

    def _make_windows(self, df: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor]:
        """Turn a trajectory DataFrame into (inputs, targets) displacement tensors.
        
        For each individual, slide a window of size `context_length + 1` positions. 
        Differencing that window gives `context_length` displacementsL: the first 
        `context_length - 1` are the input the net consumes, the last is the target step 
        it must predict. 

        This mirrors `predict`, where `history` is `context_length` positions differenced
        and fed into the net - so the net trains on exactly the shape it sees at inference. 
        Differecning is done per individual, so windows never straddle a boundary between 
        two animals tracks.

        Returns
        -------
        X: torch.Tensor, shape (n_windows, context_length - 1, 2)
        Y: torch.Tensor, shape (n_windows, 2)
        """
        T = self.context_length
        xs, ys = [], []

        for _, traj in df.groupby(self.id_col, sort=False):
            pos = traj[self.coord_cols].to_numpy(dtype=np.float32) # (L,2)
            for i in range(len(pos) - T): 
                disp = np.diff(pos[i : i + T + 1], axis=0) # (T,2)
                xs.append(disp[:-1])# (T-1, 2) input
                ys.append(disp[-1]) #(2,) target 

        if not xs: 
            raise ValueError(
                f"No windows produced: every individual has fewer than {T + 1} positions so no (input, target) pair can be formed"
            )
        X = torch.tensor(np.stack(xs), dtype=torch.float32)
        Y = torch.tensor(np.stack(ys), dtype=torch.float32)
        return X, Y


