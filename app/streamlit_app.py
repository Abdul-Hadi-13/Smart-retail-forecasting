"""
Smart Retail Demand Forecasting & Inventory Optimization - interactive dashboard.
Run with:  streamlit run app/streamlit_app.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import config
from src.inventory import inventory_plan, stock_status

PRIMARY, ACCENT, GREEN, GREY = "#4F46E5", "#F97316", "#10B981", "#9CA3AF"

st.set_page_config(page_title="Smart Retail Forecasting", page_icon="🛒", layout="wide")

st.markdown("""
<style>
.hero {background: linear-gradient(120deg,#4F46E5 0%,#7C3AED 60%,#EC4899 100%);
       padding: 28px 32px; border-radius: 16px; color: white; margin-bottom: 18px;}
.hero h1 {margin: 0; font-size: 2rem; color: white;}
.hero p  {margin: 6px 0 0 0; opacity: .92; font-size: 1.02rem;}
.card {background: var(--secondary-background-color); border-radius: 14px; padding: 16px 18px;
       border-left: 5px solid #4F46E5; margin-bottom: 8px;}
.card .label {font-size: .8rem; opacity: .7; text-transform: uppercase; letter-spacing: .05em;}
.card .value {font-size: 1.7rem; font-weight: 700;}
.tip {background: rgba(79,70,229,.10); border-radius: 12px; padding: 12px 16px; margin: 6px 0 14px 0;}
.stTabs [data-baseweb="tab"] {font-size: 1rem; font-weight: 600;}
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------- helpers
def show(fig):
    """Display a Plotly chart (works on old and new Streamlit versions)."""
    try:
        st.plotly_chart(fig, width="stretch")
    except TypeError:
        st.plotly_chart(fig, use_container_width=True)


def table(df, **kwargs):
    try:
        st.dataframe(df, width="stretch", hide_index=True, **kwargs)
    except TypeError:
        st.dataframe(df, use_container_width=True, hide_index=True, **kwargs)


def card(label, value):
    st.markdown(f'<div class="card"><div class="label">{label}</div><div class="value">{value}</div></div>',
                unsafe_allow_html=True)


def tip(text):
    st.markdown(f'<div class="tip">💡 {text}</div>', unsafe_allow_html=True)


def clean_chart(fig, height=380):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=50, b=10), legend=dict(orientation="h", y=1.12))
    return fig


@st.cache_data
def load_csv(name, dates=None):
    return pd.read_csv(config.DATA_PROCESSED / name, parse_dates=dates or [])


@st.cache_data
def load_json(path):
    return json.loads(Path(path).read_text())


needed = ["weekly_demand.csv", "backtest_forecasts.csv", "future_forecast.csv", "model_metrics.csv", "kpis.json"]
if not all((config.DATA_PROCESSED / f).exists() for f in needed):
    st.error("No results found yet. Run `python run.py` in your terminal first, then refresh this page.")
    st.stop()

kpis = load_json(config.DATA_PROCESSED / "kpis.json")
summary = load_json(config.MODELS_DIR / "model_summary.json")
weekly = load_csv("weekly_demand.csv", ["Week"])
backtest = load_csv("backtest_forecasts.csv", ["Week"])
future = load_csv("future_forecast.csv", ["Week"])
metrics = load_csv("model_metrics.csv")
prod_metrics = load_csv("product_metrics.csv")
monthly = load_csv("monthly_summary.csv")
countries = load_csv("country_summary.csv")
products = load_csv("product_summary.csv")
weekday = load_csv("weekday_summary.csv")

# ----------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("⚙️ Controls")
    names = weekly.drop_duplicates("StockCode").set_index("StockCode")["Description"].str.title()
    order = weekly.groupby("StockCode")["Units"].sum().sort_values(ascending=False).index
    code = st.selectbox("Product", list(order), format_func=lambda c: f"{names[c]} ({c})")
    model_names = list(metrics["Model"])
    model = st.selectbox("Forecast model", model_names, index=0,
                         help="The list is sorted best to worst on the test weeks.")
    st.divider()
    st.subheader("📦 Inventory settings")
    service = st.slider("Service level", 0.80, 0.99, config.SERVICE_LEVEL, 0.01,
                        help="95% means we accept running out of stock in roughly 5 of 100 order cycles.")
    lead = st.slider("Supplier lead time (weeks)", 1, 8, config.LEAD_TIME_WEEKS)
    st.divider()
    if kpis["data_source"] == "synthetic":
        st.warning("Showing **synthetic demo data**. Run `python run.py` with internet for the real UCI data.")
    else:
        st.caption("Data: UCI Online Retail II (UK gift retailer, Dec 2009 - Dec 2011)")

st.markdown(f"""
<div class="hero"><h1>🛒 Smart Retail Demand Forecasting</h1>
<p>Predict what customers will buy, and know exactly when to reorder.
Data from {kpis['start']} to {kpis['end']}.</p></div>""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📊 Business Overview", "🔮 Forecast Explorer", "🏆 Model Comparison", "📦 Inventory Planner", "ℹ️ About"])

# ================================================================= TAB 1
with tab1:
    c = st.columns(4)
    with c[0]: card("Total revenue", f"£{kpis['total_revenue']:,.0f}")
    with c[1]: card("Units sold", f"{kpis['total_units']:,}")
    with c[2]: card("Orders", f"{kpis['n_orders']:,}")
    with c[3]: card("Products / Countries", f"{kpis['n_products']:,} / {kpis['n_countries']}")

    peak = monthly.loc[monthly["Revenue"].idxmax()]
    tip(f"**Seasonality:** revenue peaks in **{peak['Month']}** (£{peak['Revenue']:,.0f}). "
        "Stock must be built up *before* the peak, which is exactly what forecasts help with.")

    left, right = st.columns(2)
    with left:
        fig = px.bar(monthly, x="Month", y="Revenue", title="Monthly revenue", color_discrete_sequence=[PRIMARY])
        show(clean_chart(fig))
    with right:
        total = weekly.groupby("Week", as_index=False)["Units"].sum()
        fig = px.area(total, x="Week", y="Units", title="Weekly units sold (top products)",
                      color_discrete_sequence=[ACCENT])
        show(clean_chart(fig))

    left, right = st.columns(2)
    with left:
        top = products.head(10).iloc[::-1].copy()
        top["Description"] = top["Description"].str.title().str[:32]
        fig = px.bar(top, x="Units", y="Description", orientation="h", title="Top 10 products by units",
                     color_discrete_sequence=[GREEN])
        show(clean_chart(fig, 420))
    with right:
        top_c = countries.head(8).iloc[::-1]
        fig = px.bar(top_c, x="Revenue", y="Country", orientation="h", title="Revenue by country (top 8)",
                     color_discrete_sequence=[PRIMARY])
        show(clean_chart(fig, 420))

# ================================================================= TAB 2
with tab2:
    st.subheader(f"{names[code]}")
    hist = weekly[weekly["StockCode"] == code]
    bt = backtest[(backtest["StockCode"] == code) & (backtest["Model"] == model)]
    fut = future[(future["StockCode"] == code) & (future["Model"] == model)]
    pm = prod_metrics[(prod_metrics["StockCode"] == code) & (prod_metrics["Model"] == model)].iloc[0]

    c = st.columns(4)
    with c[0]: card("Avg weekly demand", f"{hist['Units'].mean():,.0f} units")
    with c[1]: card("Forecast next 8 weeks", f"{fut['Forecast'].sum():,.0f} units")
    with c[2]: card("Test error (WAPE)", f"{pm['WAPE']:.0%}")
    with c[3]: card("Test error (MAE)", f"{pm['MAE']:,.0f} units/wk")

    compare = st.multiselect("Compare more models (optional)", [m for m in model_names if m != model])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist["Week"], y=hist["Units"], name="Actual demand",
                             line=dict(color=PRIMARY, width=3)))
    fig.add_trace(go.Scatter(x=bt["Week"], y=bt["Forecast"], name=f"Backtest: {model}",
                             line=dict(color=ACCENT, width=3, dash="dash")))
    fig.add_trace(go.Scatter(x=fut["Week"], y=fut["Forecast"], name=f"Future: {model}",
                             line=dict(color=GREEN, width=3, dash="dot")))
    for other in compare:
        o = backtest[(backtest["StockCode"] == code) & (backtest["Model"] == other)]
        fig.add_trace(go.Scatter(x=o["Week"], y=o["Forecast"], name=f"Backtest: {other}",
                                 line=dict(width=1.5, dash="dash"), opacity=.8))
    fig.add_vrect(x0=bt["Week"].min(), x1=bt["Week"].max(), fillcolor=GREY, opacity=.12, line_width=0,
                  annotation_text="Test weeks (hidden from models)", annotation_position="top left")
    fig.update_layout(title="Actual demand vs forecast", yaxis_title="Units per week", hovermode="x unified")
    show(clean_chart(fig, 460))
    tip("The **grey zone** is the exam: the models never saw those weeks. "
        "The closer the orange line is to the blue line there, the more we can trust the green future forecast.")

    out = fut[["Week", "Forecast"]].copy()
    out["Week"] = out["Week"].dt.date
    left, right = st.columns([2, 1])
    with left:
        table(out)
    with right:
        st.download_button("⬇️ Download forecast (CSV)", out.to_csv(index=False),
                           file_name=f"forecast_{code}.csv", mime="text/csv")

# ================================================================= TAB 3
with tab3:
    best = metrics.iloc[0]
    if best["Model"] == "Seasonal Naive":
        st.success(f"🏆 **Seasonal Naive** (\"same week last year\") is the most accurate overall, off by "
                   f"**{best['WAPE']:.1%}** of total demand. No fancier model beat this simple baseline.")
    else:
        st.success(f"🏆 **{best['Model']}** is the most accurate overall: it is off by only "
                   f"**{best['WAPE']:.1%}** of total demand "
                   f"({summary['improvement_vs_seasonal_naive_pct']:.1f}% better than the Seasonal Naive baseline).")
    left, right = st.columns([1, 1])
    with left:
        shown = metrics.copy()
        shown["WAPE"] = (shown["WAPE"] * 100).round(1).astype(str) + "%"
        table(shown)
        tip("**WAPE** = total error ÷ total demand (e.g. 25% = we are off by a quarter). "
            "**MAE** = average miss in units. **RMSE** = like MAE but punishes big misses more. Lower is better.")
    with right:
        fig = px.bar(metrics, x="Model", y=metrics["WAPE"] * 100, title="WAPE % by model (lower is better)",
                     color="Model", color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(showlegend=False, yaxis_title="WAPE %")
        show(clean_chart(fig))

    winners = prod_metrics.loc[prod_metrics.groupby("StockCode")["WAPE"].idxmin()]["Model"].value_counts()
    fig = px.bar(winners.reset_index(), x="Model", y="count", title="How many products does each model win?",
                 color_discrete_sequence=[PRIMARY])
    fig.update_layout(yaxis_title="Products won")
    show(clean_chart(fig, 320))
    tip("Different products behave differently. A model that wins overall may lose on individual products, "
        "so always check the product level before trusting one number.")

# ================================================================= TAB 4
with tab4:
    st.subheader("Reorder points & safety stock")
    st.markdown(
        "**Safety stock** = extra cushion for surprises. **Reorder point** = when stock drops to this level, "
        "place an order. Change the sliders in the sidebar and everything updates.")
    best_model = summary["best_model"]
    plan = inventory_plan(weekly, future[future["Model"] == best_model], lead, service)
    plan["Description"] = plan["Description"].str.title()
    plan["current_stock"] = (plan["avg_weekly_demand"] * 2).round().astype(int)   # editable demo value

    st.caption(f"Based on the best model ({best_model}), lead time {lead} weeks, service level {service:.0%}. "
               "Type your real stock numbers in the **Current stock** column to see order alerts.")
    show_cols = ["StockCode", "Description", "forecast_weekly", "safety_stock", "reorder_point", "current_stock"]
    edited = st.data_editor(
        plan[show_cols], hide_index=True, key=f"inv_{lead}_{service}",
        disabled=[c for c in show_cols if c != "current_stock"],
        column_config={"StockCode": "Code", "forecast_weekly": st.column_config.NumberColumn("Forecast / week", format="%.0f"),
                       "safety_stock": "Safety stock", "reorder_point": "Reorder point",
                       "current_stock": st.column_config.NumberColumn("Current stock ✏️", min_value=0, step=1)})
    edited["Status"] = [stock_status(s, r) for s, r in zip(edited["current_stock"], edited["reorder_point"])]
    counts = edited["Status"].value_counts()
    c = st.columns(3)
    with c[0]: card("🔴 Order now", int(counts.get("🔴 Order now", 0)))
    with c[1]: card("🟡 Watch", int(counts.get("🟡 Watch", 0)))
    with c[2]: card("🟢 OK", int(counts.get("🟢 OK", 0)))
    table(edited[["Description", "current_stock", "reorder_point", "Status"]].sort_values("Status", ascending=False))

    top = plan.head(10).iloc[::-1]
    fig = go.Figure()
    fig.add_bar(y=top["Description"].str[:30], x=top["forecast_weekly"] * lead, name="Demand during lead time",
                orientation="h", marker_color=PRIMARY)
    fig.add_bar(y=top["Description"].str[:30], x=top["safety_stock"], name="Safety stock",
                orientation="h", marker_color=ACCENT)
    fig.update_layout(barmode="stack", title="Reorder point = demand during lead time + safety stock (top 10)")
    show(clean_chart(fig, 430))
    st.download_button("⬇️ Download inventory plan (CSV)", plan.drop(columns="current_stock").to_csv(index=False),
                       file_name="inventory_plan.csv", mime="text/csv")

# ================================================================= TAB 5
with tab5:
    report = load_json(config.DATA_PROCESSED / "cleaning_report.json")

    # ---------------- Data
    st.subheader("📁 Data")
    if kpis["data_source"] == "synthetic":
        st.warning("This dashboard is currently showing **synthetic demo data**, not the real dataset.")
    data_info = pd.DataFrame({
        "Item": ["Dataset", "Source", "Period", "Raw rows", "Rows after cleaning",
                 "Products forecast", "License"],
        "Details": [
            "Online Retail II (UK-based online gift retailer)",
            "UCI Machine Learning Repository, dataset 502: archive.ics.uci.edu/dataset/502/online+retail+ii",
            f"{kpis['start']} to {kpis['end']}",
            f"{report['rows_raw']:,}",
            f"{report['rows_clean']:,} ({report['pct_kept']}% kept)",
            f"{weekly['StockCode'].nunique()} best-selling products, weekly demand",
            "CC BY 4.0. Citation: Chen, D. (2012), DOI 10.24432/C5CG6D",
        ],
    })
    table(data_info)

    # ---------------- Models
    st.subheader("🤖 Models compared")
    model_info = pd.DataFrame({
        "Model": ["Naive", "Seasonal Naive", "Moving Average", "Exponential Smoothing", "Random Forest"],
        "Type": ["Baseline", "Baseline", "Statistical", "Statistical", "Machine learning"],
        "How it forecasts (plain English)": [
            "Next week equals last week.",
            "Next week equals the same week last year. The baseline to beat.",
            "Average of the last 4 weeks.",
            "Like a moving average, but recent weeks count more.",
            "Learns the pattern 'last 4 weeks -> next week' and predicts one week at a time.",
        ],
    })
    table(model_info)

    # ---------------- Testing
    st.subheader("🧪 How the models were tested")
    st.markdown(f"""
- **Time-based split:** the last **{config.TEST_WEEKS} weeks** were hidden from every model, then forecast and compared with what really happened.
- **Metrics (lower is better):** **WAPE** (total error / total demand, the headline number), **MAE** (average miss in units) and **RMSE** (punishes big misses more).
- **Best model here:** **{summary['best_model']}** with a WAPE of **{summary['best_wape']:.1%}**. See the *Model Comparison* tab for the full ranking.
""")

    # ---------------- Inventory
    st.subheader("📦 Inventory formulas")
    st.markdown("""
- **Safety stock** = z x weekly demand variability x square root of lead time
- **Reorder point** = forecast weekly demand x lead time + safety stock
""")

    # ---------------- Limitations
    st.subheader("⚠️ Limitations")
    st.markdown(f"""
- Only **two years** of data: the models see just two yearly cycles, so seasonality is learned from very few examples.
- Only the **top {weekly['StockCode'].nunique()} products** are forecast, not the full catalogue.
- No prices, promotions or holidays are used as inputs.
- Safety stock assumes demand is roughly bell-shaped and lead time is fixed.
- The *Current stock* values in the Inventory tab are **demo numbers**. Enter your real values.
""")