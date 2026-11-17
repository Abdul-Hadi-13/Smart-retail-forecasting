"""plots.py - saves clean PNG charts into reports/figures/ (used in the README and notebooks)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src import config

COLORS = {"main": "#4F46E5", "accent": "#F97316", "grey": "#9CA3AF", "green": "#10B981"}


def _style(ax, title, xlabel="", ylabel=""):
    ax.set_title(title, fontsize=13, fontweight="bold", loc="left")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)


def _save(fig, name):
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / name, dpi=130)
    plt.close(fig)


def monthly_revenue(monthly):
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(monthly["Month"], monthly["Revenue"], color=COLORS["main"])
    _style(ax, "Monthly revenue", ylabel="Revenue")
    ax.tick_params(axis="x", rotation=45)
    _save(fig, "01_monthly_revenue.png")


def top_products(product_summary, n=10):
    top = product_summary.head(n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.barh(top["Description"].str.title().str[:35], top["Units"], color=COLORS["accent"])
    _style(ax, f"Top {n} products by units sold", xlabel="Units")
    _save(fig, "02_top_products.png")


def country_revenue(country_summary, n=8):
    top = country_summary[country_summary["Country"] != "United Kingdom"].head(n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.barh(top["Country"], top["Revenue"], color=COLORS["green"])
    _style(ax, f"Top {n} export countries (excluding the UK)", xlabel="Revenue")
    _save(fig, "03_country_revenue.png")


def weekly_total(weekly_long):
    total = weekly_long.groupby("Week")["Units"].sum()
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(total.index, total.values, color=COLORS["main"], lw=2)
    ax.fill_between(total.index, total.values, alpha=0.12, color=COLORS["main"])
    _style(ax, "Weekly units sold (top products) - note the autumn peak", ylabel="Units")
    _save(fig, "04_weekly_demand.png")


def model_comparison(metrics):
    fig, ax = plt.subplots(figsize=(9, 4))
    colors = [COLORS["green"] if i == 0 else COLORS["grey"] for i in range(len(metrics))]
    ax.bar(metrics["Model"], metrics["WAPE"] * 100, color=colors)
    _style(ax, "Model comparison - WAPE % (lower is better)", ylabel="WAPE %")
    ax.tick_params(axis="x", rotation=20)
    _save(fig, "05_model_comparison.png")


def forecast_example(wide, backtest, future, code, model, description=""):
    hist = wide[code]
    bt = backtest[(backtest.StockCode == code) & (backtest.Model == model)]
    fut = future[(future.StockCode == code) & (future.Model == model)]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(hist.index, hist.values, color=COLORS["main"], lw=2, label="Actual demand")
    ax.plot(bt["Week"], bt["Forecast"], color=COLORS["accent"], lw=2, ls="--", label="Backtest forecast")
    ax.plot(fut["Week"], fut["Forecast"], color=COLORS["green"], lw=2, ls="--", label="Future forecast")
    ax.axvspan(bt["Week"].min(), bt["Week"].max(), color="grey", alpha=0.1)
    _style(ax, f"{description.title()[:40]} - {model}", ylabel="Units per week")
    ax.legend(frameon=False)
    _save(fig, "06_forecast_example.png")
