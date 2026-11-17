"""
download_data.py - get the raw transactions.

Option 1 (default): download the real UCI Online Retail II dataset (2 years).
Option 2 (fallback): build a realistic SYNTHETIC dataset with the same columns,
                     so the project still runs without internet.

Usage:
    python -m src.download_data               # real data
    python -m src.download_data --synthetic   # fake-but-realistic data
"""
import io
import sys
import urllib.request
import zipfile

import numpy as np
import pandas as pd

from src import config
from src.cleaning import standardize_columns


def download_real_data() -> bool:
    """Download the UCI zip, convert the Excel file to CSV. Returns True on success."""
    config.make_folders()
    try:
        print("Downloading UCI Online Retail II dataset (about 43 MB)...")
        req = urllib.request.Request(config.DATA_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=120) as response:
            payload = response.read()
        with zipfile.ZipFile(io.BytesIO(payload)) as zf:
            excel_name = [n for n in zf.namelist() if n.lower().endswith(".xlsx")][0]
            print(f"Reading {excel_name} - it has 2 sheets, ~1 million rows. Please wait 3-6 minutes, this happens only once...")
            sheets = pd.read_excel(zf.open(excel_name), sheet_name=None,
                                   dtype={"Invoice": str, "StockCode": str})
            df = standardize_columns(pd.concat(sheets.values(), ignore_index=True))
        df.to_csv(config.RAW_FILE, index=False)
        print(f"Saved {len(df):,} rows to {config.RAW_FILE}")
        return True
    except Exception as error:  # network blocked, site down, etc.
        print(f"Could not download the data automatically: {error}")
        print("Manual option: download from https://archive.ics.uci.edu/dataset/502/online+retail+ii")
        print("unzip it, open online_retail_II.xlsx, and save BOTH sheets as one CSV named data/raw/online_retail_ii.csv")
        print("(or just run: python run.py --synthetic)")
        return False


def make_synthetic_data(seed: int = 42) -> pd.DataFrame:
    """Create a realistic fake retail dataset (2 years, same columns as the real one)."""
    rng = np.random.default_rng(seed)
    days = pd.date_range("2009-12-01", "2011-12-09")        # 2 years, like the real dataset
    n_products = 60
    words_a = ["VINTAGE", "RETRO", "HAND-PAINTED", "WOODEN", "CERAMIC", "GLASS", "METAL", "FELT"]
    words_b = ["MUG", "LANTERN", "CANDLE HOLDER", "BUNTING", "CUSHION", "TEA SET", "LUNCH BAG", "SIGN"]
    codes = [f"{85000 + i}" for i in range(n_products)]
    names = [f"{words_a[i % 8]} {words_b[(i * 3) % 8]} {i // 8 + 1}" for i in range(n_products)]
    prices = np.round(rng.uniform(0.85, 12.5, n_products), 2)
    popularity = rng.lognormal(0, 1.0, n_products)
    popularity /= popularity.sum()
    xmas_lover = rng.uniform(0.0, 1.5, n_products)          # how strongly a product peaks in autumn

    countries = ["United Kingdom", "Germany", "France", "EIRE", "Spain", "Netherlands",
                 "Belgium", "Switzerland", "Portugal", "Australia"]
    country_p = np.array([0.88, 0.03, 0.025, 0.02, 0.01, 0.01, 0.007, 0.006, 0.006, 0.006])
    country_p = country_p / country_p.sum()

    rows, invoice_no = [], 540000
    for day in days:
        if day.dayofweek == 5:                                # shop is closed on Saturdays
            continue
        month_boost = 1 + 0.9 * np.exp(-((day.dayofyear - 315) ** 2) / (2 * 35 ** 2))  # Nov peak
        weekday_boost = 1.15 if day.dayofweek in (1, 3) else 1.0
        growth = 1 + 0.15 * (day - days[0]).days / 735          # business grows ~15% over 2 years
        n_invoices = rng.poisson(26 * growth * month_boost * weekday_boost)
        for _ in range(n_invoices):
            invoice_no += 1
            customer = float(rng.integers(12346, 18287)) if rng.random() > 0.2 else np.nan
            country = rng.choice(countries, p=country_p)
            time = day + pd.Timedelta(hours=int(rng.integers(8, 18)), minutes=int(rng.integers(0, 60)))
            weights = popularity * (1 + xmas_lover * (month_boost - 1))
            weights /= weights.sum()
            for p in rng.choice(n_products, size=int(rng.integers(1, 9)), replace=False, p=weights):
                qty = int(rng.choice([1, 2, 3, 4, 6, 8, 12, 24], p=[.2, .15, .1, .1, .2, .1, .1, .05]))
                rows.append([str(invoice_no), codes[p], names[p], qty, time, prices[p], customer, country])

    df = pd.DataFrame(rows, columns=["InvoiceNo", "StockCode", "Description", "Quantity",
                                     "InvoiceDate", "UnitPrice", "CustomerID", "Country"])

    # ---- add realistic "dirty" rows so the cleaning step has real work to do ----
    n = len(df)
    cancels = df.sample(int(n * 0.02), random_state=1).copy()
    cancels["InvoiceNo"] = "C" + cancels["InvoiceNo"]
    cancels["Quantity"] = -cancels["Quantity"]
    junk = df.sample(int(n * 0.004), random_state=2).copy()
    junk["StockCode"], junk["Description"], junk["Quantity"] = "POST", "POSTAGE", 1
    zero_price = df.sample(int(n * 0.002), random_state=3).copy()
    zero_price["UnitPrice"] = 0.0
    dupes = df.sample(int(n * 0.005), random_state=4).copy()
    df = pd.concat([df, cancels, junk, zero_price, dupes], ignore_index=True)
    return df.sort_values("InvoiceDate").reset_index(drop=True)


def save_synthetic_data() -> None:
    config.make_folders()
    df = make_synthetic_data()
    df.to_csv(config.RAW_FILE, index=False)
    print(f"Saved SYNTHETIC data: {len(df):,} rows to {config.RAW_FILE}")


if __name__ == "__main__":
    if "--synthetic" in sys.argv:
        save_synthetic_data()
    elif not download_real_data():
        sys.exit(1)
