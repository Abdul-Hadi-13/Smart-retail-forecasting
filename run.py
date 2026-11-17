"""
run.py - runs the WHOLE project with one command:

    python run.py                 # real UCI data (downloads on first run)
    python run.py --synthetic     # no internet? use realistic fake data

Steps: get data -> clean -> weekly demand -> forecast + evaluate -> inventory plan -> charts
Then start the dashboard with:   streamlit run app/streamlit_app.py
"""
import argparse
import json
import time

import pandas as pd

from src import config, plots
from src.cleaning import clean_transactions, validate_clean
from src.data_loader import load_raw
from src.download_data import download_real_data, save_synthetic_data
from src.evaluation import future_forecast, run_backtest, score_models, score_products
from src.features import make_weekly_demand, to_wide
from src.inventory import inventory_plan

SOURCE_MARKER = config.DATA_RAW / "SOURCE.txt"


def step(n, text):
    print(f"\n[{n}/6] {text}")


def save_json(obj, path):
    path.write_text(json.dumps(obj, indent=2, default=str))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true", help="use generated demo data")
    args = parser.parse_args()
    start = time.time()
    config.make_folders()
    out = config.DATA_PROCESSED

    # 1 ---------------------------------------------------------------- data
    step(1, "Getting raw data")
    if args.synthetic:
        save_synthetic_data()
        SOURCE_MARKER.write_text("synthetic")
    elif config.RAW_FILE.exists():
        print(f"Using existing file: {config.RAW_FILE}")
    elif download_real_data():
        SOURCE_MARKER.write_text("uci")
    else:
        print("WARNING: falling back to SYNTHETIC demo data.")
        save_synthetic_data()
        SOURCE_MARKER.write_text("synthetic")
    data_source = SOURCE_MARKER.read_text().strip() if SOURCE_MARKER.exists() else "uci"

    # 2 -------------------------------------------------------------- cleaning
    step(2, "Cleaning & validating transactions")
    clean, report = clean_transactions(load_raw())
    problems = validate_clean(clean)
    assert not problems, f"Data quality problems: {problems}"
    report["data_source"] = data_source
    for key, value in report.items():
        print(f"   {key:<38} {value}")
    clean.to_csv(out / "clean_transactions.csv", index=False)
    save_json(report, out / "cleaning_report.json")

    # 3 ----------------------------------------------------------- aggregates
    step(3, "Building summaries and weekly demand")
    clean["Month"] = clean["InvoiceDate"].dt.to_period("M").astype(str)
    monthly = clean.groupby("Month").agg(Revenue=("Revenue", "sum"), Units=("Quantity", "sum"),
                                         Orders=("InvoiceNo", "nunique")).reset_index()
    country = clean.groupby("Country").agg(Revenue=("Revenue", "sum"), Orders=("InvoiceNo", "nunique"),
                                           Units=("Quantity", "sum")).reset_index().sort_values("Revenue", ascending=False)
    product_summary = clean.groupby(["StockCode", "Description"]).agg(
        Units=("Quantity", "sum"), Revenue=("Revenue", "sum")).reset_index().sort_values("Units", ascending=False)
    weekday = clean.groupby(clean["InvoiceDate"].dt.day_name())["Revenue"].sum().reset_index()
    weekday.columns = ["Weekday", "Revenue"]
    kpis = {"total_revenue": float(clean["Revenue"].sum()), "total_units": int(clean["Quantity"].sum()),
            "n_orders": int(clean["InvoiceNo"].nunique()), "n_products": int(clean["StockCode"].nunique()),
            "n_countries": int(clean["Country"].nunique()),
            "start": str(clean["InvoiceDate"].min().date()), "end": str(clean["InvoiceDate"].max().date()),
            "data_source": data_source}
    monthly.round(2).to_csv(out / "monthly_summary.csv", index=False)
    country.round(2).to_csv(out / "country_summary.csv", index=False)
    product_summary.round(2).to_csv(out / "product_summary.csv", index=False)
    weekday.round(2).to_csv(out / "weekday_summary.csv", index=False)
    save_json(kpis, out / "kpis.json")

    weekly = make_weekly_demand(clean)
    weekly.to_csv(out / "weekly_demand.csv", index=False)
    wide = to_wide(weekly)
    print(f"   {wide.shape[1]} products x {wide.shape[0]} full weeks")

    # 4 --------------------------------------------------- forecast + evaluate
    step(4, f"Backtesting 5 models (last {config.TEST_WEEKS} weeks hidden)")
    backtest = run_backtest(wide)
    metrics = score_models(backtest)
    per_product = score_products(backtest)
    print(metrics.to_string(index=False))
    backtest.to_csv(out / "backtest_forecasts.csv", index=False)
    metrics.to_csv(out / "model_metrics.csv", index=False)
    per_product.to_csv(out / "product_metrics.csv", index=False)

    best = metrics.iloc[0]["Model"]
    baseline = float(metrics.loc[metrics.Model == "Seasonal Naive", "WAPE"].iloc[0])
    gain = 100 * (baseline - metrics.iloc[0]["WAPE"]) / baseline if baseline else 0
    if best == "Seasonal Naive":
        print(f"   Best model: {best} (WAPE {metrics.iloc[0]['WAPE']:.1%}). The simple baseline won - no model beat it!")
    else:
        print(f"   Best model: {best} (WAPE {metrics.iloc[0]['WAPE']:.1%}, "
              f"{gain:.1f}% better than Seasonal Naive)")
    save_json({"best_model": best, "best_wape": float(metrics.iloc[0]["WAPE"]),
               "seasonal_naive_wape": baseline, "improvement_vs_seasonal_naive_pct": round(gain, 1),
               "test_weeks": config.TEST_WEEKS, "future_weeks": config.FUTURE_WEEKS},
              config.MODELS_DIR / "model_summary.json")

    # 5 ------------------------------------------------------------- inventory
    step(5, "Forecasting the future & planning inventory")
    future = future_forecast(wide)
    future.to_csv(out / "future_forecast.csv", index=False)
    plan = inventory_plan(weekly, future[future.Model == best])
    plan.to_csv(out / "inventory_plan.csv", index=False)
    print(plan[["StockCode", "Description", "forecast_weekly", "safety_stock", "reorder_point"]].head(5).to_string(index=False))

    # 6 ------------------------------------------------------------------ charts
    step(6, "Saving charts to reports/figures/")
    plots.monthly_revenue(monthly)
    plots.top_products(product_summary)
    plots.country_revenue(country)
    plots.weekly_total(weekly)
    plots.model_comparison(metrics)
    top_code = plan.iloc[0]["StockCode"]
    plots.forecast_example(wide, backtest, future, top_code, best, plan.iloc[0]["Description"])

    print(f"\nDONE in {time.time() - start:.0f}s.  Now run:  streamlit run app/streamlit_app.py")


if __name__ == "__main__":
    main()
