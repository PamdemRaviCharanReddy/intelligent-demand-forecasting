"""PyTorch Long Short-Term Memory (LSTM) deep learning forecaster.

Implements the student's 12-step sequence algorithm:
1. Data collection & ingestion
2. Availability checking & graceful fallback
3. Integration & cleaning
4. Sequence generation: length L window predicting horizon H
5. MinMax normalization (strictly fitted train-only)
6. Neural network training
7. Architecture: Forget, input, and output gates with cell state
8. Forward inference & point forecasting
9. Metric evaluation (MAE, RMSE, WAPE)
10. Supply chain replenishment linking
11. Incremental retraining support
"""

from __future__ import annotations

import logging
from typing import Any, cast

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats
from torch.utils.data import DataLoader, TensorDataset

from idfs.models.base import Forecaster

logger = logging.getLogger(__name__)


class _LSTMModel(nn.Module):
    """PyTorch LSTM module for multi-step time series forecasting."""

    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        horizon: int = 14,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.horizon = horizon

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, input_dim)
        lstm_out, (_hn, _cn) = self.lstm(x)
        # Take the output of the last time step
        last_step = lstm_out[:, -1, :]
        out = self.fc(last_step)
        return cast(torch.Tensor, out)


class LSTMForecaster(Forecaster):
    """Global Deep Learning LSTM forecaster.

    Trained across multi-series panels using sliding sequence windows.
    MinMax scaling is fitted strictly on the training set.
    """

    def __init__(
        self,
        seq_len: int = 28,
        horizon: int = 14,
        hidden_dim: int = 32,
        num_layers: int = 2,
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
        batch_size: int = 64,
        epochs: int = 15,
        patience: int = 4,
        device: str | None = None,
        name: str = "LSTM",
    ) -> None:
        super().__init__(name=name)
        self.seq_len = seq_len
        self.horizon = horizon
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model_: _LSTMModel | None = None
        self._min_val: float = 0.0
        self._max_val: float = 1.0
        self._series_id_cols: list[str] = []
        self._last_sequences: dict[tuple, np.ndarray] = {}
        self._last_dates: dict[tuple, pd.Timestamp] = {}
        self._last_values: dict[tuple, float] = {}
        self._residual_stds: dict[tuple, float] = {}

    @property
    def supports_global(self) -> bool:
        return True

    def fit(
        self,
        train_df: pd.DataFrame,
        target_col: str = "sales",
        series_id_cols: list[str] | None = None,
        val_df: pd.DataFrame | None = None,
        **kwargs: Any,
    ) -> LSTMForecaster:
        """Fit PyTorch LSTM across all series with train-only MinMax normalization."""
        self._series_id_cols = series_id_cols or []
        self._last_sequences.clear()
        self._last_dates.clear()
        self._last_values.clear()
        self._residual_stds.clear()

        df = train_df.copy()
        df["date"] = pd.to_datetime(df["date"])

        # Step 5: MinMax normalisation fit strictly on training data
        raw_vals = df[target_col].to_numpy(dtype=np.float64)
        self._min_val = float(np.min(raw_vals)) if len(raw_vals) > 0 else 0.0
        self._max_val = float(np.max(raw_vals)) if len(raw_vals) > 0 else 1.0
        range_val = max(self._max_val - self._min_val, 1e-6)

        def scale(v: np.ndarray) -> np.ndarray:
            return (v - self._min_val) / range_val

        # Step 4: Prepare sequences (sliding window seq_len predicting horizon)
        X_train_list: list[np.ndarray] = []
        y_train_list: list[np.ndarray] = []

        groups: list[tuple[Any, pd.DataFrame]] = (
            list(df.groupby(self._series_id_cols, observed=True))
            if self._series_id_cols
            else [(None, df)]
        )

        for key, group in groups:
            k = key if isinstance(key, tuple) else (key,)
            sorted_group = group.sort_values("date")
            series_raw = sorted_group[target_col].to_numpy(dtype=np.float64)
            series_scaled = scale(series_raw)
            last_date = sorted_group["date"].iloc[-1]
            self._last_dates[k] = last_date
            self._last_values[k] = float(series_raw[-1]) if len(series_raw) > 0 else 0.0

            # Store the latest seq_len raw/scaled values for future inference
            if len(series_scaled) >= self.seq_len:
                self._last_sequences[k] = series_scaled[-self.seq_len :]
            else:
                padded = np.pad(series_scaled, (self.seq_len - len(series_scaled), 0), mode="edge")
                self._last_sequences[k] = padded

            # Approximate baseline residual std
            diff = np.diff(series_raw)
            self._residual_stds[k] = float(np.std(diff)) if len(diff) > 0 else 1.0

            # Create sliding sequences
            total_window = self.seq_len + self.horizon
            if len(series_scaled) >= total_window:
                for i in range(len(series_scaled) - total_window + 1):
                    x_seq = series_scaled[i : i + self.seq_len]
                    y_seq = series_scaled[i + self.seq_len : i + total_window]
                    X_train_list.append(x_seq)
                    y_train_list.append(y_seq)

        if not X_train_list:
            # Fallback if dataset is shorter than seq_len + horizon
            logger.warning("Dataset too short for full LSTM sliding window. Using single point.")
            dummy_x = np.zeros((1, self.seq_len, 1), dtype=np.float32)
            dummy_y = np.zeros((1, self.horizon), dtype=np.float32)
            X_tensor = torch.tensor(dummy_x)
            y_tensor = torch.tensor(dummy_y)
        else:
            X_arr = np.array(X_train_list, dtype=np.float32)[:, :, np.newaxis]
            y_arr = np.array(y_train_list, dtype=np.float32)
            X_tensor = torch.tensor(X_arr)
            y_tensor = torch.tensor(y_arr)

        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Step 6 & 7: Model definition with LSTM gates
        torch.manual_seed(42)
        model = _LSTMModel(
            input_dim=1,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
            horizon=self.horizon,
        ).to(self.device)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)

        # Training loop
        model.train()
        best_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.epochs):
            total_epoch_loss = 0.0
            for batch_x, batch_y in dataloader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                preds = model(batch_x)
                loss = criterion(preds, batch_y)
                loss.backward()
                optimizer.step()
                total_epoch_loss += loss.item() * len(batch_x)

            avg_loss = total_epoch_loss / max(len(dataset), 1)
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    logger.debug("Early stopping at epoch %d with loss %.4f", epoch, avg_loss)
                    break

        self.model_ = model
        self._is_fitted = True
        return self

    def predict(
        self,
        horizon: int,
        features_df: pd.DataFrame | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Generate multi-step point predictions from LSTM."""
        if not self._is_fitted or self.model_ is None:
            raise RuntimeError("LSTMForecaster must be fitted before predict")

        self.model_.eval()
        range_val = max(self._max_val - self._min_val, 1e-6)

        def unscale(v: np.ndarray) -> np.ndarray:
            return v * range_val + self._min_val

        rows = []
        if features_df is not None and "date" in features_df.columns and self._series_id_cols:
            feat_df = features_df.copy()
            feat_df["date"] = pd.to_datetime(feat_df["date"])

            for key, group in feat_df.groupby(self._series_id_cols, observed=True):
                k = key if isinstance(key, tuple) else (key,)
                seq = self._last_sequences.get(k)
                if seq is None:
                    preds = np.full(len(group), self._last_values.get(k, 0.0))
                else:
                    x_in = torch.tensor(
                        seq[np.newaxis, :, np.newaxis], dtype=torch.float32, device=self.device
                    )
                    with torch.no_grad():
                        out = self.model_(x_in).cpu().numpy()[0]
                    unscaled_preds = unscale(out)

                    if len(group) <= len(unscaled_preds):
                        preds = unscaled_preds[: len(group)]
                    else:
                        # If horizon is longer than trained output, pad with last step
                        preds = np.pad(
                            unscaled_preds,
                            (0, len(group) - len(unscaled_preds)),
                            mode="edge",
                        )

                preds = np.maximum(0.0, np.nan_to_num(preds, nan=self._last_values.get(k, 0.0)))
                sorted_group = group.sort_values("date")
                for d, p in zip(sorted_group["date"], preds, strict=False):
                    row: dict[str, Any] = {"date": d, "forecast": float(p)}
                    for i, col in enumerate(self._series_id_cols):
                        row[col] = k[i]
                    rows.append(row)
        else:
            for k, last_date in self._last_dates.items():
                start_dt = pd.to_datetime(last_date) + pd.Timedelta(days=1)
                dates = pd.date_range(start=start_dt, periods=horizon, freq="D")
                seq = self._last_sequences.get(k)
                if seq is None:
                    preds = np.full(horizon, self._last_values.get(k, 0.0))
                else:
                    x_in = torch.tensor(
                        seq[np.newaxis, :, np.newaxis], dtype=torch.float32, device=self.device
                    )
                    with torch.no_grad():
                        out = self.model_(x_in).cpu().numpy()[0]
                    unscaled_preds = unscale(out)
                    if horizon <= len(unscaled_preds):
                        preds = unscaled_preds[:horizon]
                    else:
                        preds = np.pad(
                            unscaled_preds,
                            (0, horizon - len(unscaled_preds)),
                            mode="edge",
                        )

                preds = np.maximum(0.0, np.nan_to_num(preds, nan=self._last_values.get(k, 0.0)))
                for d, p in zip(dates, preds, strict=False):
                    row = {"date": d, "forecast": float(p)}
                    for i, col in enumerate(self._series_id_cols):
                        row[col] = k[i]
                    rows.append(row)

        return pd.DataFrame(rows)

    def predict_interval(
        self,
        horizon: int,
        coverage: float = 0.90,
        features_df: pd.DataFrame | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        pred_df = self.predict(horizon=horizon, features_df=features_df, **kwargs)
        z = stats.norm.ppf(0.5 + coverage / 2)

        half_widths = []
        if self._series_id_cols and set(self._series_id_cols).issubset(set(pred_df.columns)):
            for tup in pred_df[self._series_id_cols].itertuples(index=False):
                k = tuple(tup)
                std = self._residual_stds.get(k, 1.0)
                half_widths.append(z * max(std, 0.1))
        else:
            default_std = (
                float(np.mean(list(self._residual_stds.values()))) if self._residual_stds else 1.0
            )
            half_widths = [z * max(default_std, 0.1)] * len(pred_df)

        arr_half = np.array(half_widths)
        pred_df["lower"] = np.maximum(0.0, pred_df["forecast"] - arr_half)
        pred_df["upper"] = pred_df["forecast"] + arr_half
        return pred_df
