from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing


class PyTorchLSTM(nn.Module):
    """Single-layer LSTM head used by the forecasting engine."""

    def __init__(self, input_size: int = 1, hidden_size: int = 50, output_size: int = 1) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(sequence)
        return self.linear(out[:, -1, :])


def evaluate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mape = float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-9))) * 100)
    return mae, rmse, mape


def confidence_z_value(ci_pct: int) -> float:
    return {80: 1.282, 90: 1.645, 95: 1.960, 99: 2.576}.get(int(ci_pct), 1.960)


def run_holt_winters(
    train: pd.Series,
    test: pd.Series,
    horizon: int,
    ci_pct: int,
) -> dict[str, Any]:
    model = ExponentialSmoothing(
        train,
        trend="add",
        seasonal="add",
        seasonal_periods=min(365, len(train) // 2),
    )
    fit = model.fit(optimized=True)

    test_pred = fit.forecast(len(test))
    mae, rmse, mape = evaluate_metrics(test.values, test_pred.values)

    future_all = fit.forecast(len(test) + horizon)
    future_pred = future_all[-horizon:]
    future_dates = pd.date_range(test.index[-1] + pd.Timedelta(days=1), periods=horizon, freq="D")

    ci_half = confidence_z_value(ci_pct) * float(fit.resid.std())
    return {
        "model_name": "Holt-Winters Smoothing",
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "test_dates": test.index,
        "test_pred": test_pred.values,
        "future_dates": future_dates,
        "future_pred": future_pred.values,
        "future_lower": future_pred.values - ci_half,
        "future_upper": future_pred.values + ci_half,
    }


def run_arima(
    train: pd.Series,
    test: pd.Series,
    horizon: int,
    ci_pct: int,
) -> dict[str, Any]:
    alpha = 1 - (ci_pct / 100)
    fit = ARIMA(train, order=(5, 1, 0)).fit()

    test_result = fit.get_forecast(steps=len(test))
    test_pred = test_result.predicted_mean
    mae, rmse, mape = evaluate_metrics(test.values, test_pred.values)

    future_result = fit.get_forecast(steps=len(test) + horizon)
    future_pred = future_result.predicted_mean.iloc[-horizon:]
    future_conf = future_result.conf_int(alpha=alpha).iloc[-horizon:]
    future_dates = pd.date_range(test.index[-1] + pd.Timedelta(days=1), periods=horizon, freq="D")

    return {
        "model_name": "ARIMA (5,1,0)",
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "test_dates": test.index,
        "test_pred": test_pred.values,
        "future_dates": future_dates,
        "future_pred": future_pred.values,
        "future_lower": future_conf.iloc[:, 0].values,
        "future_upper": future_conf.iloc[:, 1].values,
    }


def run_lstm(
    train: pd.Series,
    test: pd.Series,
    horizon: int,
    ci_pct: int,
) -> dict[str, Any]:
    scaler = MinMaxScaler(feature_range=(-1, 1))
    train_scaled = scaler.fit_transform(train.values.reshape(-1, 1))
    lookback = 30

    x_sequences = []
    y_targets = []
    for index in range(len(train_scaled) - lookback):
        x_sequences.append(train_scaled[index : index + lookback])
        y_targets.append(train_scaled[index + lookback])

    x_train = torch.tensor(np.array(x_sequences), dtype=torch.float32)
    y_train = torch.tensor(np.array(y_targets), dtype=torch.float32)

    model = PyTorchLSTM()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for _ in range(60):
        optimizer.zero_grad()
        loss = criterion(model(x_train), y_train)
        loss.backward()
        optimizer.step()

    test_inputs = train_scaled[-lookback:].tolist()
    test_inputs.extend(scaler.transform(test.values.reshape(-1, 1)).tolist())
    test_inputs = np.array(test_inputs)

    test_preds = []
    for index in range(len(test)):
        seq = torch.tensor(test_inputs[index : index + lookback], dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            test_preds.append(model(seq).item())

    test_preds_inv = scaler.inverse_transform(np.array(test_preds).reshape(-1, 1)).flatten()
    mae, rmse, mape = evaluate_metrics(test.values, test_preds_inv)

    future_inputs = scaler.transform(test.values[-lookback:].reshape(-1, 1)).tolist()
    future_preds = []
    for _ in range(horizon):
        seq = torch.tensor(future_inputs[-lookback:], dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            prediction = model(seq).item()
        future_preds.append(prediction)
        future_inputs.append([prediction])

    future_preds_inv = scaler.inverse_transform(np.array(future_preds).reshape(-1, 1)).flatten()
    future_dates = pd.date_range(test.index[-1] + pd.Timedelta(days=1), periods=horizon, freq="D")
    ci_half = confidence_z_value(ci_pct) * float((test.values - test_preds_inv).std())

    return {
        "model_name": "LSTM (Deep Learning)",
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "test_dates": test.index,
        "test_pred": test_preds_inv,
        "future_dates": future_dates,
        "future_pred": future_preds_inv,
        "future_lower": future_preds_inv - ci_half,
        "future_upper": future_preds_inv + ci_half,
    }


def run_model(
    model_choice: str,
    train: pd.Series,
    test: pd.Series,
    horizon: int,
    ci_pct: int,
) -> dict[str, Any]:
    if "ARIMA" in model_choice:
        return run_arima(train, test, horizon, ci_pct)
    if "LSTM" in model_choice:
        return run_lstm(train, test, horizon, ci_pct)
    return run_holt_winters(train, test, horizon, ci_pct)
