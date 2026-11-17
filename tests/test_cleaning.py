"""Data-quality tests: does cleaning remove what it should and keep what it should?"""
import pandas as pd
import pytest

from src.cleaning import clean_transactions, validate_clean


def make_raw():
    return pd.DataFrame({
        "InvoiceNo":   ["1001", "1002", "C1003", "1004", "1005", "1006", "1006"],
        "StockCode":   ["A1",   "A1",   "A1",    "POST", "A2",   "A2",   "A2"],
        "Description": ["MUG",  "MUG",  "MUG",   "POSTAGE", "CUP", "CUP", "CUP"],
        "Quantity":    [5,      3,      -2,      1,       -4,     2,      2],
        "InvoiceDate": ["2011-01-03 10:00", "2011-01-04 11:00", "2011-01-05 09:00", "2011-01-05 10:00",
                        "2011-01-06 10:00", "2011-01-06 12:00", "2011-01-06 12:00"],
        "UnitPrice":   [2.5,    2.5,    2.5,     15.0,    1.0,    0.0,    0.0],
        "CustomerID":  [1, 2, 3, 4, 5, 6, 6],
        "Country":     ["United Kingdom"] * 7,
    })


def test_only_good_rows_remain():
    clean, report = clean_transactions(make_raw())
    assert len(clean) == 2                         # only invoices 1001 and 1002 are valid
    assert set(clean["InvoiceNo"]) == {"1001", "1002"}
    assert report["removed_cancellations"] == 1
    assert report["removed_non_product_codes"] == 1


def test_clean_data_passes_validation():
    clean, _ = clean_transactions(make_raw())
    assert validate_clean(clean) == []


def test_revenue_is_quantity_times_price():
    clean, _ = clean_transactions(make_raw())
    assert (clean["Revenue"] == clean["Quantity"] * clean["UnitPrice"]).all()


def test_report_numbers_add_up():
    clean, report = clean_transactions(make_raw())
    removed = sum(v for k, v in report.items() if k.startswith("removed_"))
    assert report["rows_raw"] - removed == report["rows_clean"] == len(clean)


def test_validation_catches_bad_data():
    bad = make_raw()
    bad["InvoiceDate"] = pd.to_datetime(bad["InvoiceDate"])
    problems = validate_clean(bad)
    assert any("negative quantities" in p for p in problems)
    assert any("Cancellations" in p for p in problems)


def test_online_retail_ii_column_names_are_accepted():
    """The real Online Retail II file uses Invoice / Price / 'Customer ID'."""
    raw = make_raw().rename(columns={"InvoiceNo": "Invoice", "UnitPrice": "Price", "CustomerID": "Customer ID"})
    clean, _ = clean_transactions(raw)
    assert len(clean) == 2 and "UnitPrice" in clean.columns


def test_gift_vouchers_and_overlapping_sheets_are_removed():
    raw = make_raw()
    extra = raw.iloc[[0]].copy()                       # exact copy of a valid row (sheet overlap)
    voucher = raw.iloc[[0]].copy()
    voucher["StockCode"] = "GIFT_0001_20"
    clean, report = clean_transactions(__import__("pandas").concat([raw, extra, voucher]))
    assert len(clean) == 2 and report["removed_duplicates"] >= 1


def test_missing_column_raises_clear_error():
    with pytest.raises(ValueError, match="Missing columns"):
        clean_transactions(make_raw().drop(columns=["Country"]))
