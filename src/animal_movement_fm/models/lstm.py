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

from animal_movement_fm.eval.harness import iter_windows

logger = logging.getLogger(__name__)

class _LSTMNet(nn.Module):
    """
    Inner network: LSTM encoder + linear head

    Shapes
    ------
    input: (batch, seq_len, input_dim)
    output: (batch, output_dim)
    """

    def __init__(
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
        self.net: _LSTMNet | None = None 
        self._mean: np.ndarray | None = None # (2,) displacement mean
        self._std: np.ndarray | None = None # (2,) displacement std 


        # auto label in results table 
        self.name = name or f"lstm_h{self.hidden_dim}_l{self.num_layers}"

    def _make_windows(self, df: pd.DataFrame, group_col: str = "individual_id") -> tuple[torch.Tensor, torch.Tensor]:
        """Turn a trajectory DataFrame into (inputs, targets) displacement tensors.
        
        Window definition is delegated to the harness's `iter_windows`, so the net trains on exactly 
        the windows it is evaluated on. Each window's (context_length, 2) position history is 
        differenced into (context_length - 1, 2) input displacements; the target is the single
        displacement from the last history position to the target position.

        Returns
        -------
        X : torch.Tensor, shape (n_windows, context_length - 1, 2)
        Y : torch.Tensor, shape (n_windows, 2)
        """
        xs, ys = [], []
        for _, traj in df.groupby(group_col, sort=False):
            for history, target, _ts, _t in iter_windows(traj, self.context_length):
                xs.append(np.diff(history, axis=0))   # (W-1, 2) input displacements
                ys.append(target - history[-1])       # (2,) target displacement

        if not xs:
            raise ValueError(
                f"No windows produced: every individual has fewer than "
                f"{self.context_length + 1} positions."
            )

        X = torch.tensor(np.stack(xs), dtype=torch.float32)
        Y = torch.tensor(np.stack(ys), dtype=torch.float32)
        return X, Y
    
    def fit(self, train_df: pd.DataFrame, group_col: str = "individual_id") -> "LSTM":
        """
        Train the network once on all trajectories in `train_df`.

        Build displacement windows, standardise the inputs (storing the per-channel mean/stf so `predict` 
        can apply the identical transform), then runs a standard mini batch training loop. Trained weights and 
        stats are stored on the instance. Returns self so we can write (callers can write) 
        `model = LSTM().fit(train.df)`
        """

        X, Y = self._make_windows(train_df, group_col=group_col)   # (N, W-1, 2), (N, 2)

        # standardise inputs using TRAIN statistics only, and remember them. Predict() must reuse these exact 
        # stats, so they are stored on self
        self._mean = X.mean(dim=(0, 1), keepdim=True) # (1, 1, 2)
        self._std = X.std(dim=(0, 1), keepdim=True).clamp_min(1e-8) # avoid /0
        Xn = (X - self._mean) / self._std 

        # Preload the full training tensors onto the GPU once so batches are indexed on device, eliminating per batch CPU -> GPU 
        # transfers while still preserving deterministic shuffling
        Xn = Xn.to(self.device)
        Y = Y.to(self.device)
        n_windows = Xn.size(0)

        self.net = _LSTMNet(
            input_dim=2, 
            hidden_dim=self.hidden_dim, 
            num_layers = self.num_layers, 
            output_dim=2,
        ).to(self.device)

        optimiser = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        self.net.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            # reshuffle each epoch; randperm on device etc 
            perm = torch.randperm(n_windows, device=self.device)
            for start in range(0, n_windows, self.batch_size):
                idx = perm[start : start + self.batch_size]
                xb = Xn[idx]
                yb = Y[idx]

                optimiser.zero_grad()
                pred = self.net(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                optimiser.step()

                epoch_loss += loss.item() * xb.size(0)

            epoch_loss /= n_windows
            logger.info("epoch %d/%d train_mse=%.6g", epoch + 1, self.epochs, epoch_loss)
        return self

    
    def predict(self, context) -> np.ndarray:
        """
        Predicts the next (lon, lat) position from a window of history.

        Mirrors the input pipeline in `_make_windows`: the (W, 2) position history is differenced into 
        (W-1, 2) displacements, standardised with the stored train stats, and passed through the net to 
        predict the next displacement, which is added back onto the last observed position. 

        Parameters
        ----------
        context: PredictionContext, carrying `history` of shape (W, 2), columns (lon, lat). The LSTM uses only the history; 
        the timestamp is ignored for now...

        Returns
        -------
        np.ndarray, shape (2,), the predicted next (lon, lat). 
        """

        if self.net is None:
            raise RuntimeError("Model is not fitted. Call fit or load a checkpoint before predict")
        
        history = np.asarray(context.history, dtype=np.float32)
        if history.ndim != 2 or history.shape[1] != 2: 
            raise ValueError(f"history must have shape (W, 2), we got {history.shape} instead")
        
        # single observaton; no displacement to compute, so fall back to persistence (predict no movement);
        # matches the harness contract that predict always returns a (2,) position. 
        if history.shape[0] < 2: 
            return history[-1].copy()
        
        disp = np.diff(history, axis=0) # (W-1,2)

        # standardise with the stored TRAIN stats (never computed here)
        mean = self._mean.detach().cpu().numpy().reshape(1,2)
        std = self._std.detach().cpu().numpy().reshape(1,2)
        disp_n = (disp - mean) / std 
        
        # (W-1,2) -> (1, W-1, 2): the net expects a batch dimension
        x = torch.tensor(disp_n, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        self.net.eval()
        with torch.no_grad():
            pred_disp = self.net(x) # (1,2)

        pred_disp = pred_disp.squeeze(0).cpu().numpy() # (2,)
        return history[-1] + pred_disp


# GENAI Reference: I asked GenAI to implement show me how to save and from_checkpoint methods to the LSTM
# model so a trained model can be saved on a supercomputer (for us NCI) and reloaded elwhere for evaluatin 
# fine-tuning without retraining. I noted in the prompt saying that the checkpoint shuld store everything 
# needed for predict, so trained weights, scalar statistics and architecture config required to rebuild 
# the network. 
# We verified the output by saving a fitted LSTM, reloading it with `from_checkpoint`, and confirming
# that the stored model produced the same predictions without retraining (it did). 
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
                    "hidden_dim": self.hidden_dim,
                    "num_layers": self.num_layers,
                    "context_length": self.context_length,
                    "name": self.name,
                },
            },
            path,
        )

    @classmethod
    def from_checkpoint(cls, path: str | Path, device: str | None = None) -> "LSTM":
        """Rebuild a fitted model from a checkpoint written by `save`.
        Reconstructs the model from the stored architecture, loads the weights,
        and restores the scaler stats — yielding an instance ready to `predict`
        without any retraining.
        """
        path = Path(path)
        # weights_only=False: this is our own trusted checkpoint and it contains a non-tensor `config` dict; explicit here to 
        # silence the FutureWarning and stay correct when torch flips the default to True
        ckpt = torch.load(path, map_location=device or "cpu", weights_only=False)
        cfg = ckpt["config"]

        # Rebuild the wrapper with the stored architecture.
        model = cls(
            hidden_dim=cfg["hidden_dim"],
            num_layers=cfg["num_layers"],
            context_length=cfg["context_length"],
            device=device,
            name=cfg["name"],
        )

        # Rebuild the net to match, load weights, restore scaler stats.
        model.net = _LSTMNet(
            input_dim=2,
            hidden_dim=cfg["hidden_dim"],
            num_layers=cfg["num_layers"],
            output_dim=2,
        ).to(model.device)
        model.net.load_state_dict(ckpt["state_dict"])
        model._mean = ckpt["mean"]
        model._std = ckpt["std"]

        return model
