"""
forecasting.py - five simple, explainable forecasting models.

Every model has the same shape:   model(history, horizon) -> array of future values
 - history : past weekly demand (numpy array, oldest first)
 - horizon : how many weeks to forecast
"""
import numpy as np
from sklearn.ensemble import RandomForestRegressor

from src import config
from src.features import make_lag_table


def naive(history, horizon):
    """Tomorrow looks like today: repeat the last observed week."""
    return np.repeat(history[-1], horizon).astype(float)


def seasonal_naive(history, horizon, season=config.SEASON_LENGTH):
    """
    Same week last year (season=52). The classic baseline to beat.
    If there is less than a full season of history, it repeats whatever history exists.
    """
    last_cycle = np.asarray(history[-season:], dtype=float)
    return np.resize(last_cycle, horizon)


def moving_average(history, horizon, window=4):
    """Forecast = average of the last 4 weeks."""
    return np.repeat(np.mean(history[-window:]), horizon).astype(float)


def exp_smoothing(history, horizon, alpha=0.3):
    """Simple exponential smoothing: recent weeks count more than old weeks."""
    level = float(history[0])
    for value in history[1:]:
        level = alpha * value + (1 - alpha) * level
    return np.repeat(level, horizon)


def random_forest(history, horizon, n_lags=4):
    """
    Machine-learning model. Learns the pattern 'last 4 weeks -> next week',
    then predicts one week at a time, feeding each prediction back in.
    """
    history = np.asarray(history, dtype=float)
    if len(history) < n_lags + 8:                       # too little data for ML
        return moving_average(history, horizon)
    X, y = make_lag_table(history, n_lags)
    model = RandomForestRegressor(n_estimators=100, max_depth=4, min_samples_leaf=3,
                                  random_state=config.RANDOM_STATE)
    model.fit(X, y)
    window = list(history[-n_lags:])                    # oldest -> newest
    preds = []
    for _ in range(horizon):
        pred = float(model.predict([window[::-1]])[0])  # newest first, like the training rows
        preds.append(pred)
        window = window[1:] + [pred]
    return np.array(preds)


MODELS = {
    "Naive": naive,
    "Seasonal Naive": seasonal_naive,
    "Moving Average": moving_average,
    "Exponential Smoothing": exp_smoothing,
    "Random Forest": random_forest,
}


def forecast(model_name: str, history, horizon: int):
    """Run any model by name. Demand can never be negative, so we clip at 0."""
    if model_name not in MODELS:
        raise ValueError(f"Unknown model '{model_name}'. Choose from {list(MODELS)}")
    history = np.asarray(history, dtype=float)
    if len(history) == 0:
        raise ValueError("History is empty")
    return np.clip(MODELS[model_name](history, horizon), 0, None)
