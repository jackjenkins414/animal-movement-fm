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

from animal_movement_fm.analysis.regime import label_migratory_launches, label_regimes
from animal_movement_fm.eval.harness import iter_windows
from animal_movement_fm.features.time import day_of_year_sincos, local_solar_sincos
from animal_movement_fm.state.dataset import STATE_TO_IDX

logger = logging.getLogger(__name__)

class _LSTMNet(nn.Module):
    """
    Inner network: LSTM encoder + linear head(s).

    Shapes
    ------
    input: (batch, seq_len, input_dim)
    output: (batch, output_dim)                       when use_state_head is False
    output: (batch, output_dim), (batch, n_states)    when use_state_head is True

    Two-factor state head (Rung 2)
    ------------------------------
    When `use_state_head` is on, a second linear head reads the same trunk summary `last` and emits `n_states` state
    logits. The softmax of those logits (the soft state distribution, differentiable so gradients flow end to end) is
    concatenated onto `last` before the position head, so the displacement prediction is explicitly conditioned on the
    regime the trunk thinks is coming. With the flag off, the module is byte-for-byte our original raw sequence net
    (position head reads `last` directly), so every existing checkpoint, prediction and test is still unchanged.
    """

    def __init__(
            self,
            input_dim: int = 2,
            hidden_dim: int = 64, # DEFAULT FOR NOW
            num_layers: int = 1, # SAME HERE
            output_dim: int = 2,
            use_state_head: bool = False,
            n_states: int = 5,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
        )
        self.use_state_head = bool(use_state_head)
        self.n_states = int(n_states)
        if self.use_state_head:
            self.state_head = nn.Linear(hidden_dim, self.n_states)
            # position head conditioned on the soft state distribution appended to the trunk summary
            self.fc = nn.Linear(hidden_dim + self.n_states, output_dim)
        else:
            self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor):
        # x: (batch, seq_len, input-dim)

        # out: (batch, seq_len, hidden_dim)
        out, _ = self.lstm(x)

        # (batch, hidden_dim); summary of the window
        last = out[:, -1, :]

        if self.use_state_head:
            state_logits = self.state_head(last)   # (batch, n_states)
            p_state = torch.softmax(state_logits, dim=-1)  # soft, differentiable conditioning
            pos = self.fc(torch.cat([last, p_state], dim=-1)) # (batch, output_dim)
            return pos, state_logits

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
        use_time_features: bool = False,
        use_season_features: bool = False,
        use_state_head: bool = False,
        state_loss_weight: float = 0.5,
        n_states: int = 5,
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
        # Per-position covariates, each appended as a (sin, cos) pair tagged on the fix the displacement lands on:
        #   1. use_time_features; local solar time of day (diurnal; varies strongly WITHIN a window)
        #   2. use_season_features; day of year (near constant across a 6 day window, so it conditions the whole window rather than 
        #      differentiating positions)
        # Independent flags, so raw / time / season / both are all reachable for ablation. Both default off, which keeps the raw sequence 
        # model, every existing checkpoint, and the tests unchanged
        self.use_time_features = bool(use_time_features)
        self.use_season_features = bool(use_season_features)
        self.input_dim = (
            2
            + (2 if self.use_time_features else 0)
            + (2 if self.use_season_features else 0)
        )

        # two factor state head (Rung 2); a second head predicts the next fix's state and its soft distribution
        # conditions the position head. The state head is on the trunk, so it does NOT change input_dim. Off by
        # default, which leaves the raw/time/season models, their checkpoints and tests untouched.
        self.use_state_head = bool(use_state_head)
        self.state_loss_weight = float(state_loss_weight)
        self.n_states = int(n_states)

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # set by `fit` (or `from_checkpoint`); none on a fresh, unfitted model
        self.net: _LSTMNet | None = None
        self._mean: np.ndarray | None = None # (2,) displacement mean
        self._std: np.ndarray | None = None # (2,) displacement std


        # auto label in results table
        suffix = (
            ("_time" if self.use_time_features else "")
            + ("_season" if self.use_season_features else "")
            + ("_state" if self.use_state_head else "")
        )
        self.name = name or f"lstm_h{self.hidden_dim}_l{self.num_layers}{suffix}"

    def _covariate_channels(self, dest_ts, dest_lon) -> np.ndarray:
        """
        Per-position covariate channels for the destination fixes of a window's displacements.

        Displacement i spans fix i -> i+1, so it is tagged with fix i+1's clock and date; the last displacement
        therefore carries the most recent fix's time, i.e. the departure clock of the step being predicted. Channels
        are appended in a fixed order (time of day, then season) so the input layout is stable across `fit`,
        `predict` and checkpoints. All are already bounded in [-1, 1] and are never standardised.

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

        Window definition is delegated to the harness's `iter_windows`, so the net trains on exactly the windows it is 
        evaluated on. Each window's (context_length, 2) position history is differenced into (context_length - 1, 2) 
        input displacements; the target is the single displacement from the last history position to the target position.

        When `use_time_features` is on, each displacement is also tagged with the local solar (sin, cos) of the fix it 
        lands on: displacement i spans fix i -> i+1, so it carries fix i+1's clock. The last displacement therefore 
        carries the most recent fix's time, i.e. the departure clock of the step being predicted. Time features are left 
        raw (already in [-1, 1]); only the displacement channels are standardised, in `fit`.

        Returns
        -------
        X : torch.Tensor, shape (n_windows, context_length - 1, 2)
        Y : torch.Tensor, shape (n_windows, 2)
        T : torch.Tensor | None, shape (n_windows, context_length - 1, 2) of (sin, cos), or None when time features 
            are disabled.
        """
        xs, ys, ts = [], [], []
        needs_ts = self.use_time_features or self.use_season_features
        for _, traj in df.groupby(group_col, sort=False):
            traj_ts = traj["timestamp"].to_numpy() if needs_ts else None
            for history, target, _ts, t_idx in iter_windows(traj, self.context_length):
                xs.append(np.diff(history, axis=0))  # (W-1, 2) input displacements
                ys.append(target - history[-1])  # (2,) target displacement
                if needs_ts:
                    win_ts = traj_ts[t_idx - self.context_length : t_idx]   # (W,) row aligned with history
                    ts.append(self._covariate_channels(win_ts[1:], history[1:, 0])) # destination fixes 1..W-1

        if not xs:
            raise ValueError(
                f"No windows produced: every individual has fewer than "
                f"{self.context_length + 1} positions."
            )

        X = torch.tensor(np.stack(xs), dtype=torch.float32)
        Y = torch.tensor(np.stack(ys), dtype=torch.float32)
        T = torch.tensor(np.stack(ts), dtype=torch.float32) if needs_ts else None
        return X, Y, T

    def _state_labels(self, df: pd.DataFrame, group_col: str = "individual_id") -> torch.Tensor:
        """Per-window next state labels, aligned one-to-one with `_make_windows` output.

        Re-runs the exact same grouping + `iter_windows` loop as `_make_windows`, so the i-th label corresponds to the
        i-th (X, Y) window. The label is the canonical refined state of the window's TARGET fix (`state[t_idx]`), i.e.
        the same fix whose displacement the position head predicts, which makes the state head a "one step" forecast and
        identical to the labels the standalone classifier uses (`state.dataset.build_state_windows`).

        Labels come from the canonical labeller (`label_regimes` -> `label_migratory_launches`), keyed on
        (segment_id, timestamp) so alignment is order independent. Requires a `segment_id` column (the labeller defines
        states within segments); raises if it is missing.

        Returns
        -------
        torch.Tensor, shape (n_windows,), dtype long : integer state index (STATE_TO_IDX) per window.
        """
        if "segment_id" not in df.columns:
            raise ValueError(
                "use_state_head=True requires a 'segment_id' column to derive next-state labels; "
                "the segment product provides it."
            )
        labelled = label_migratory_launches(label_regimes(df))
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
                f"{self.context_length + 1} positions."
            )
        return torch.tensor(ss, dtype=torch.long)

    def fit(self, train_df: pd.DataFrame, group_col: str = "individual_id") -> "LSTM":
        """
        Train the network once on all trajectories in `train_df`.

        Build displacement windows, standardise the inputs (storing the per-channel mean/stf so `predict` 
        can apply the identical transform), then runs a standard mini batch training loop. Trained weights and 
        stats are stored on the instance. Returns self so we can write (callers can write) 
        `model = LSTM().fit(train.df)`
        """

        X, Y, T = self._make_windows(train_df, group_col=group_col)   # (N, W-1, 2), (N, 2), (N, W-1, 2)|None

        # standardise the DISPLACEMENT channels using TRAIN statistics only, and remember them. Predict() must reuse these exact stats,
        # so they are stored on self. Time features (if any) are already bounded in [-1, 1] and are NOT standardised.
        self._mean = X.mean(dim=(0, 1), keepdim=True) # (1, 1, 2)
        self._std = X.std(dim=(0, 1), keepdim=True).clamp_min(1e-8) # avoid /0
        Xn = (X - self._mean) / self._std

        # append the raw covariate channels AFTER standardisation, so the net sees (N, W-1, input_dim)
        if T is not None:
            Xn = torch.cat([Xn, T], dim=-1)

        # per-window next state labels for the state head, aligned to (Xn, Y) by construction (same window loop)
        S = self._state_labels(train_df, group_col=group_col) if self.use_state_head else None

        # preload the full training tensors onto the GPU once so batches are indexed on device, eliminating per batch CPU -> GPU
        # transfers while still preserving deterministic shuffling
        Xn = Xn.to(self.device)
        Y = Y.to(self.device)
        if S is not None:
            S = S.to(self.device)
        n_windows = Xn.size(0)

        self.net = _LSTMNet(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_layers = self.num_layers,
            output_dim=2,
            use_state_head=self.use_state_head,
            n_states=self.n_states,
        ).to(self.device)

        optimiser = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()
        ce_fn = nn.CrossEntropyLoss() if self.use_state_head else None

        self.net.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            epoch_state = 0.0
            # reshuffle each epoch; randperm on device etc
            perm = torch.randperm(n_windows, device=self.device)
            for start in range(0, n_windows, self.batch_size):
                idx = perm[start : start + self.batch_size]
                xb = Xn[idx]
                yb = Y[idx]

                optimiser.zero_grad()
                if self.use_state_head:
                    pred, state_logits = self.net(xb)
                    loss_pos = loss_fn(pred, yb)
                    loss_state = ce_fn(state_logits, S[idx])
                    # joint objective: position MSE (primary) + weighted state cross-entropy (auxiliary conditioning)
                    loss = loss_pos + self.state_loss_weight * loss_state
                    epoch_state += loss_state.item() * xb.size(0)
                else:
                    pred = self.net(xb)
                    loss = loss_fn(pred, yb)
                loss.backward()
                optimiser.step()

                epoch_loss += loss.item() * xb.size(0)

            epoch_loss /= n_windows
            if self.use_state_head:
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

        # ADD TIME: append the same destination tagged local solar (sin, cos) built in _make_windows, using the per fix timestamps 
        # the harness threads through the context
        if self.use_time_features or self.use_season_features:
            if context.history_timestamps is None:
                raise ValueError(
                    "model was fitted with time/season features but context.history_timestamps is None; "
                    "the harness populates it, so this is a direct predict() call missing the times."
                )
            win_ts = np.asarray(context.history_timestamps)
            cov = self._covariate_channels(win_ts[1:], history[1:, 0]).astype(np.float32)  # (W-1, n_cov)
            feat = np.concatenate([disp_n, cov], axis=-1)   # (W-1, input_dim)
        else:
            feat = disp_n

        # (W-1, C) -> (1, W-1, C): the net expects a batch dimension
        x = torch.tensor(feat, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        self.net.eval()
        with torch.no_grad():
            out = self.net(x)
            # with the state head on, the net returns (pos, state_logits); the state head conditions the position
            # internally, so predict still returns only the (lon, lat) position and the harness contract is unchanged
            pred_disp = out[0] if self.use_state_head else out  # (1,2)

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
                    "use_time_features": self.use_time_features,
                    "use_season_features": self.use_season_features,
                    "use_state_head": self.use_state_head,
                    "state_loss_weight": self.state_loss_weight,
                    "n_states": self.n_states,
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
        # old checkpoints predate these flags -> raw sequence, no state head
        use_time = cfg.get("use_time_features", False)
        use_season = cfg.get("use_season_features", False)
        use_state_head = cfg.get("use_state_head", False)
        n_states = cfg.get("n_states", 5)

        # Rebuild the wrapper with the stored architecture.
        model = cls(
            hidden_dim=cfg["hidden_dim"],
            num_layers=cfg["num_layers"],
            context_length=cfg["context_length"],
            use_time_features=use_time,
            use_season_features=use_season,
            use_state_head=use_state_head,
            state_loss_weight=cfg.get("state_loss_weight", 0.5),
            n_states=n_states,
            device=device,
            name=cfg["name"],
        )

        # Rebuild the net to match, load weights, restore scaler stats.
        model.net = _LSTMNet(
            input_dim=model.input_dim,
            hidden_dim=cfg["hidden_dim"],
            num_layers=cfg["num_layers"],
            output_dim=2,
            use_state_head=use_state_head,
            n_states=n_states,
        ).to(model.device)
        model.net.load_state_dict(ckpt["state_dict"])
        model._mean = ckpt["mean"]
        model._std = ckpt["std"]

        return model
