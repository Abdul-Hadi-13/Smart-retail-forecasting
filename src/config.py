"""
config.py - ONE place for every setting in the project.
Want to change the lead time or forecast horizon? Change it here, nowhere else.
"""
from pathlib import Path

# ---------- Folders ----------
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
FIGURES_DIR = ROOT / "reports" / "figures"

# ---------- Data source (UCI "Online Retail II" dataset: Dec 2009 - Dec 2011, ~1.07M rows) ----------
DATA_URL = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
RAW_FILE = DATA_RAW / "online_retail_ii.csv"

# Rows that are NOT real products (postage, bank charges, manual adjustments...)
NON_PRODUCT_CODES = {"POST", "D", "DOT", "M", "BANK CHARGES", "CRUK",
                     "AMAZONFEE", "S", "B", "C2", "PADS", "ADJUST", "ADJUST2",
                     "TEST001", "TEST002", "SAMPLES"}

# ---------- Forecasting settings ----------
TOP_N_PRODUCTS = 20      # how many best-selling products we forecast
TEST_WEEKS = 8           # last 8 weeks are hidden from the models and used as the exam
FUTURE_WEEKS = 8         # how many weeks ahead we forecast for real
SEASON_LENGTH = 52       # "seasonal naive" = same week LAST YEAR (possible thanks to 2 years of data)
RANDOM_STATE = 42

# ---------- Inventory settings ----------
LEAD_TIME_WEEKS = 2      # time between placing an order and receiving it
SERVICE_LEVEL = 0.95     # 95% = we accept a stock-out in about 5 of 100 cycles
RECENT_WEEKS = 26        # weeks of history used to measure demand variability


def make_folders():
    for folder in (DATA_RAW, DATA_PROCESSED, MODELS_DIR, FIGURES_DIR):
        folder.mkdir(parents=True, exist_ok=True)
