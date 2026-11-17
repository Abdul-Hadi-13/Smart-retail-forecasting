"""
evaluation.py - a fair exam for the models.

TIME-BASED split: train on the past, test on the most recent weeks.
(Never shuffle time-series data - that would let the model "see the future".)

Metrics (lower is better for all three):
  MAE   = average miss, in units
  RMSE  = like MAE but punishes big misses harder
  WAPE  = total miss / total demand  -> "we are 22% off". Easy to explain to clients.
"""
import numpy as np
import pandas as pd

from src import config
from src.forecasting import MODELS, forecast


def mae(actual, predicted):
    return float(np.mean(np.abs(np.asarray(actual) - np.asarray(predicted))))


def rmse(actual, predicted):
    return float(np.sqrt(np.mean((np.asarray(actual) - np.asarray(predicted)) ** 2)))


def wape(actual, predicted):
    actual, predicted = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    total = np.sum(np.abs(actual))
    return float("nan") if total == 0 else float(np.sum(np.abs(actual - predicted)) / total)


def time_split(wide: pd.DataFrame, test_weeks: int = config.TEST_WEEKS):
    """Oldest weeks -> train, newest `test_weeks` -> test."""
    if len(wide) <= test_weeks + 8:
        raise ValueError("Not enough weeks of data for a time-based split.")
    return wide.iloc[:-test_weeks], wide.iloc[-test_weeks:]


def run_backtest(wide: pd.DataFrame, test_weeks: int = config.TEST_WEEKS) -> pd.DataFrame:
    """Forecast the hidden test weeks with every model. Returns Week, StockCode, Model, Actual, Forecast."""
    train, test = time_split(wide, test_weeks)
    rows = []
    for code in wide.columns:
        for name in MODELS:
            preds = forecast(name, train[code].values, len(test))
            for week, actual, pred in zip(test.index, test[code].values, preds):
                rows.append([week, code, name, actual, round(float(pred), 2)])
    return pd.DataFrame(rows, columns=["Week", "StockCode", "Model", "Actual", "Forecast"])


def score_models(backtest: pd.DataFrame) -> pd.DataFrame:
    """One row per model, using all products together."""
    rows = []
    for name, g in backtest.groupby("Model"):
        rows.append({"Model": name, "MAE": mae(g.Actual, g.Forecast),
                     "RMSE": rmse(g.Actual, g.Forecast), "WAPE": wape(g.Actual, g.Forecast)})
    out = pd.DataFrame(rows).sort_values("WAPE").reset_index(drop=True)
    out.insert(0, "Rank", range(1, len(out) + 1))
    return out.round(3)


def score_products(backtest: pd.DataFrame) -> pd.DataFrame:
    """One row per product x model."""
    rows = []
    for (code, name), g in backtest.groupby(["StockCode", "Model"]):
        rows.append({"StockCode": code, "Model": name, "MAE": mae(g.Actual, g.Forecast),
                     "RMSE": rmse(g.Actual, g.Forecast), "WAPE": wape(g.Actual, g.Forecast)})
    return pd.DataFrame(rows).round(3)


def future_forecast(wide: pd.DataFrame, horizon: int = config.FUTURE_WEEKS) -> pd.DataFrame:
    """Re-train on ALL history and forecast the real future. Returns Week, StockCode, Model, Forecast."""
    future_weeks = pd.date_range(wide.index[-1] + pd.Timedelta(weeks=1), periods=horizon, freq="W-MON")
    rows = []
    for code in wide.columns:
        for name in MODELS:
            for week, pred in zip(future_weeks, forecast(name, wide[code].values, horizon)):
                rows.append([week, code, name, round(float(pred), 2)])
    return pd.DataFrame(rows, columns=["Week", "StockCode", "Model", "Forecast"])
