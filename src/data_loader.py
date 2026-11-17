"""data_loader.py - small helpers to read the project's CSV files."""
import pandas as pd

from src import config


def load_raw() -> pd.DataFrame:
    if not config.RAW_FILE.exists():
        raise FileNotFoundError(
            f"{config.RAW_FILE} not found. Run `python run.py` (or `python -m src.download_data`) first.")
    return pd.read_csv(config.RAW_FILE, dtype={"InvoiceNo": str, "Invoice": str, "StockCode": str})


def load_processed(name: str, **kwargs) -> pd.DataFrame:
    """Read a file from data/processed/, e.g. load_processed('weekly_demand.csv')."""
    path = config.DATA_PROCESSED / name
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python run.py` first.")
    return pd.read_csv(path, **kwargs)
