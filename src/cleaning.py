"""
cleaning.py - turn messy raw transactions into trustworthy data.

Every step records HOW MANY rows it removed, so you can show clients
exactly what was cleaned and why (great for a portfolio!).
"""
import pandas as pd

from src import config

# Online Retail II uses slightly different column names than the older dataset.
# We rename everything to ONE standard so the rest of the project never has to care.
COLUMN_ALIASES = {"invoice": "InvoiceNo", "invoiceno": "InvoiceNo", "stockcode": "StockCode",
                  "description": "Description", "quantity": "Quantity", "invoicedate": "InvoiceDate",
                  "price": "UnitPrice", "unitprice": "UnitPrice", "customerid": "CustomerID",
                  "country": "Country"}


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """'Customer ID' -> 'CustomerID', 'Price' -> 'UnitPrice', 'Invoice' -> 'InvoiceNo', ..."""
    renamed = {c: COLUMN_ALIASES.get(str(c).replace(" ", "").lower(), c) for c in df.columns}
    return df.rename(columns=renamed)


REQUIRED_COLUMNS = ["InvoiceNo", "StockCode", "Description", "Quantity",
                    "InvoiceDate", "UnitPrice", "CustomerID", "Country"]


def clean_transactions(raw: pd.DataFrame):
    """Return (clean_dataframe, report_dict)."""
    raw = standardize_columns(raw)
    missing = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = raw.copy()
    report = {"rows_raw": len(df)}

    df["InvoiceNo"] = df["InvoiceNo"].astype(str).str.strip()
    df["StockCode"] = df["StockCode"].astype(str).str.strip().str.upper()
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")

    def step(name, keep_mask):
        nonlocal df
        report[name] = int((~keep_mask).sum())
        df = df[keep_mask]

    step("removed_missing_date_or_code", df["InvoiceDate"].notna() & (df["StockCode"] != "NAN"))
    step("removed_cancellations", ~df["InvoiceNo"].str.upper().str.startswith("C"))
    step("removed_zero_or_negative_quantity", df["Quantity"] > 0)
    step("removed_zero_or_negative_price", df["UnitPrice"] > 0)
    step("removed_non_product_codes", ~(df["StockCode"].isin(config.NON_PRODUCT_CODES)
                                        | df["StockCode"].str.startswith("GIFT_")))

    # One clean name per product: use the most common description
    names = (df.dropna(subset=["Description"])
               .groupby("StockCode")["Description"]
               .agg(lambda s: s.str.strip().str.upper().mode().iloc[0]))
    df["Description"] = df["StockCode"].map(names).fillna("UNKNOWN PRODUCT")

    # Duplicates are removed LAST, after names are standardised (rows that differed only
    # in description spelling are now identical)
    step("removed_duplicates", ~df.duplicated(keep="first"))

    df["Revenue"] = (df["Quantity"] * df["UnitPrice"]).round(2)
    df = df.reset_index(drop=True)
    report["rows_clean"] = len(df)
    report["pct_kept"] = round(100 * len(df) / max(report["rows_raw"], 1), 1)
    return df, report


def validate_clean(df: pd.DataFrame) -> list:
    """Quality checks. Returns a list of problems (an empty list = all good)."""
    problems = []
    if df.empty:
        return ["Dataset is empty"]
    if (df["Quantity"] <= 0).any():
        problems.append("Found zero/negative quantities")
    if (df["UnitPrice"] <= 0).any():
        problems.append("Found zero/negative prices")
    if df["InvoiceDate"].isna().any():
        problems.append("Found missing dates")
    if df["InvoiceNo"].astype(str).str.upper().str.startswith("C").any():
        problems.append("Cancellations are still present")
    if (df["StockCode"].isin(config.NON_PRODUCT_CODES) | df["StockCode"].str.startswith("GIFT_")).any():
        problems.append("Non-product codes are still present")
    if df.duplicated().any():
        problems.append("Duplicate rows found")
    return problems
