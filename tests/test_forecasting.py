"""Tests that forecasts, metrics and inventory formulas BEHAVE sensibly."""
import numpy as np
import pandas as pd
import pytest

from src.evaluation import mae, rmse, time_split, wape
from src.features import make_lag_table, make_weekly_demand
from src.forecasting import MODELS, forecast, seasonal_naive
from src.inventory import reorder_point, safety_stock, stock_status, z_score

HISTORY = np.array([10, 12, 9, 14, 11, 13, 10, 12, 15, 11, 12, 14, 10, 13, 12, 11], dtype=float)


@pytest.mark.parametrize("name", list(MODELS))
def test_every_model_returns_right_length_and_no_negatives(name):
    preds = forecast(name, HISTORY, 6)
    assert len(preds) == 6
    assert (preds >= 0).all()


@pytest.mark.parametrize("name", list(MODELS))
def test_constant_demand_gives_constant_forecast(name):
    preds = forecast(name, np.full(30, 20.0), 4)
    assert np.allclose(preds, 20.0, atol=1.0)


def test_seasonal_naive_repeats_last_cycle():
    preds = seasonal_naive(np.array([1, 2, 3, 4, 5, 6, 7, 8.0]), 8, season=4)
    assert list(preds) == [5, 6, 7, 8, 5, 6, 7, 8]


def test_seasonal_naive_uses_same_week_last_year():
    two_years = np.arange(104, dtype=float)             # week 0..103
    preds = forecast("Seasonal Naive", two_years[:96], 8)   # forecast weeks 96..103
    assert list(preds) == list(two_years[44:52])            # exactly 52 weeks earlier


def test_moving_average_value():
    assert forecast("Moving Average", np.array([2, 4, 6, 8.0]), 2)[0] == 5.0


def test_unknown_model_raises():
    with pytest.raises(ValueError):
        forecast("Magic", HISTORY, 3)


def test_metrics_perfect_and_known_values():
    assert mae([10, 20], [10, 20]) == 0 and rmse([10, 20], [10, 20]) == 0
    assert wape([10, 10], [8, 12]) == pytest.approx(0.2)        # total error 4 / total demand 20
    assert np.isnan(wape([0, 0], [1, 1]))                       # no demand -> undefined, not a crash


def test_time_split_has_no_leakage():
    wide = pd.DataFrame({"A": range(30)}, index=pd.date_range("2011-01-03", periods=30, freq="W-MON"))
    train, test = time_split(wide, 8)
    assert len(test) == 8 and train.index.max() < test.index.min()


def test_lag_table_shape():
    X, y = make_lag_table(np.arange(10.0), 4)
    assert X.shape == (6, 4) and list(X[0]) == [3, 2, 1, 0] and y[0] == 4


def test_weekly_demand_fills_missing_weeks_with_zero():
    dates = pd.to_datetime(["2011-01-03", "2011-01-04", "2011-01-17", "2011-01-18", "2011-01-31", "2011-02-06"])
    clean = pd.DataFrame({"InvoiceDate": dates, "StockCode": "A", "Description": "MUG", "Quantity": 5})
    weekly = make_weekly_demand(clean, top_n=1)
    assert len(weekly) == 5 and (weekly["Units"] == 0).sum() == 2   # two silent weeks become zeros


def test_inventory_formulas():
    assert z_score(0.95) == pytest.approx(1.645, abs=0.001)
    assert safety_stock(10, 4, 0.95) == 33                         # ceil(1.645 * 10 * 2)
    assert reorder_point(100, 2, 33) == 233
    assert safety_stock(10, 4, 0.99) > safety_stock(10, 4, 0.90)  # higher service -> more cushion
    assert safety_stock(0, 4, 0.95) == 0                           # no variability -> no cushion needed


def test_stock_status_labels():
    assert "Order now" in stock_status(50, 100)
    assert "Watch" in stock_status(110, 100)
    assert "OK" in stock_status(300, 100)


def test_bad_service_level_rejected():
    with pytest.raises(ValueError):
        z_score(1.5)
