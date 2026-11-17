"""
inventory.py - how much stock should we hold?

Two ideas, both beginner-friendly:

  Safety stock  = z x (weekly demand std) x sqrt(lead time)
        An extra cushion for surprises. z comes from the service level (95% -> 1.645).

  Reorder point = (forecast weekly demand x lead time) + safety stock
        When stock falls to this number, place a new order.
"""
import math
from statistics import NormalDist

import pandas as pd

from src import config


def z_score(service_level: float) -> float:
    if not 0.5 <= service_level < 1:
        raise ValueError("service_level must be between 0.5 and 0.999")
    return NormalDist().inv_cdf(service_level)


def safety_stock(demand_std: float, lead_time_weeks: float, service_level: float) -> int:
    return math.ceil(z_score(service_level) * demand_std * math.sqrt(lead_time_weeks))


def reorder_point(weekly_demand: float, lead_time_weeks: float, safety: float) -> int:
    return math.ceil(weekly_demand * lead_time_weeks + safety)


def stock_status(current_stock: float, rop: float) -> str:
    if current_stock <= rop:
        return "🔴 Order now"
    if current_stock <= rop * 1.25:
        return "🟡 Watch"
    return "🟢 OK"


def inventory_plan(weekly_long: pd.DataFrame, future: pd.DataFrame,
                   lead_time_weeks: float = config.LEAD_TIME_WEEKS,
                   service_level: float = config.SERVICE_LEVEL,
                   recent_weeks: int = config.RECENT_WEEKS) -> pd.DataFrame:
    """
    weekly_long : Week, StockCode, Description, Units  (history)
    future      : Week, StockCode, Forecast            (one model's future forecast)
    """
    recent_cut = weekly_long["Week"].max() - pd.Timedelta(weeks=recent_weeks - 1)
    recent = weekly_long[weekly_long["Week"] >= recent_cut]
    stats = recent.groupby(["StockCode", "Description"])["Units"].agg(
        avg_weekly_demand="mean", demand_std="std").reset_index().fillna({"demand_std": 0})
    fc = future.groupby("StockCode")["Forecast"].agg(
        forecast_weekly="mean", units_next_4_weeks=lambda s: s.head(4).sum()).reset_index()

    plan = stats.merge(fc, on="StockCode", how="left")
    plan["forecast_weekly"] = plan["forecast_weekly"].fillna(plan["avg_weekly_demand"])
    plan["lead_time_weeks"] = lead_time_weeks
    plan["safety_stock"] = [safety_stock(s, lead_time_weeks, service_level) for s in plan["demand_std"]]
    plan["reorder_point"] = [reorder_point(d, lead_time_weeks, s)
                             for d, s in zip(plan["forecast_weekly"], plan["safety_stock"])]
    cols = ["avg_weekly_demand", "demand_std", "forecast_weekly", "units_next_4_weeks"]
    plan[cols] = plan[cols].round(1)
    return plan.sort_values("reorder_point", ascending=False).reset_index(drop=True)
