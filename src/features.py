"""features.py - reshape transactions into weekly demand per product."""
import numpy as np
import pandas as pd

from src import config


def make_weekly_demand(clean: pd.DataFrame, top_n: int = config.TOP_N_PRODUCTS) -> pd.DataFrame:
    """
    Weekly units sold for the top-N products (weeks run Monday to Sunday).
    - The first/last week are dropped if they're incomplete (they'd look like fake dips).
    - Weeks with no sales are filled with 0 (no sales = zero demand, not missing data).
    Returns columns: Week, StockCode, Description, Units
    """
    df = clean.copy()
    df["Week"] = df["InvoiceDate"].dt.to_period("W-SUN").dt.start_time

    first_day, last_day = df["InvoiceDate"].min().normalize(), df["InvoiceDate"].max().normalize()
    weeks = pd.date_range(df["Week"].min(), df["Week"].max(), freq="W-MON")
    weeks = [w for w in weeks if w >= first_day and w + pd.Timedelta(days=6) <= last_day]
    df = df[df["Week"].isin(weeks)]

    # Pick reliable best-sellers: sold in at least half of all weeks, then biggest volume
    stats = df.groupby("StockCode").agg(units=("Quantity", "sum"), active_weeks=("Week", "nunique"))
    reliable = stats[stats["active_weeks"] >= 0.5 * len(weeks)]
    pool = reliable if len(reliable) >= top_n else stats
    top_codes = pool.sort_values("units", ascending=False).head(top_n).index

    wide = (df[df["StockCode"].isin(top_codes)]
            .pivot_table(index="Week", columns="StockCode", values="Quantity", aggfunc="sum")
            .reindex(weeks).fillna(0))
    wide.index.name = "Week"

    names = df.drop_duplicates("StockCode").set_index("StockCode")["Description"]
    long = wide.reset_index().melt(id_vars="Week", var_name="StockCode", value_name="Units")
    long["Description"] = long["StockCode"].map(names)
    return long.sort_values(["StockCode", "Week"]).reset_index(drop=True)[
        ["Week", "StockCode", "Description", "Units"]]


def to_wide(weekly_long: pd.DataFrame) -> pd.DataFrame:
    """Long table -> one column per product (easier for modelling)."""
    return weekly_long.pivot(index="Week", columns="StockCode", values="Units").sort_index()


def make_lag_table(values, n_lags: int = 4):
    """
    Build a 'learn from the past' table for the ML model.
    Row = [demand 1 week ago, 2 weeks ago, 3 weeks ago, 4 weeks ago]  ->  target = this week's demand
    """
    values = np.asarray(values, dtype=float)
    X = [values[i - n_lags:i][::-1] for i in range(n_lags, len(values))]
    y = values[n_lags:]
    return np.array(X), np.array(y)
