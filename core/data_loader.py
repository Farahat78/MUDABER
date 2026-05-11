"""
core/data_loader.py
────────────────────────────────────────────────────────────────────────────────
Central data-loading layer for the Streamlit frontend.

Rules:
  - ONLY this module reads CSV files.
  - All other pages import from here.
  - Falls back to GitHub raw URL if local file is missing.
  - Returns empty DataFrames with correct schema on error (never crashes the app).
"""

from __future__ import annotations

import os
import logging
import pandas as pd
import streamlit as st

logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.join(_HERE, "..", "data")

# GitHub raw URLs (update to your actual repo)
_GITHUB_RAW_BASE = os.getenv(
    "DATA_GITHUB_RAW_BASE",
    "https://raw.githubusercontent.com/YOUR_USERNAME/prediction-pipeline/main/data"
)

PRODUCTS_LOCAL = os.path.join(_DATA_DIR, "latest_products.csv")
PREDICTIONS_LOCAL = os.path.join(_DATA_DIR, "predictions.csv")
FINANCIAL_LOCAL = os.path.join(_DATA_DIR, "financial_output.csv")

PRODUCTS_REMOTE   = f"{_GITHUB_RAW_BASE}/latest_products.csv"
PREDICTIONS_REMOTE = f"{_GITHUB_RAW_BASE}/predictions.csv"


# ── Generic loader ────────────────────────────────────────────────────────────
def _load_csv(local_path: str, remote_url: str, cache_key: str) -> pd.DataFrame:
    """Load CSV from local path first, then fall back to remote URL."""
    # 1. Try local (fastest — already on disk after pipeline run)
    if os.path.exists(local_path):
        try:
            df = pd.read_csv(local_path, encoding="utf-8-sig")
            logger.info(f"Loaded {cache_key} from local: {len(df)} rows")
            return df
        except Exception as e:
            logger.warning(f"Failed to read local {local_path}: {e}")

    # 2. Fall back to GitHub raw URL
    if "YOUR_USERNAME" not in remote_url:
        try:
            df = pd.read_csv(remote_url, encoding="utf-8")
            logger.info(f"Loaded {cache_key} from GitHub: {len(df)} rows")
            return df
        except Exception as e:
            logger.warning(f"Failed to read remote {remote_url}: {e}")

    # 3. Return empty frame
    logger.error(f"Could not load {cache_key} from any source.")
    return pd.DataFrame()


# ── Public API ────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def load_products() -> pd.DataFrame:
    """
    Load the latest scraped Carrefour products.
    Expected columns: main_category, sub_category, product_name, price,
                      discount, image_url, week_of_month, source, date
    """
    df = _load_csv(PRODUCTS_LOCAL, PRODUCTS_REMOTE, "latest_products")
    if df.empty:
        return df

    # Normalise column names
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Ensure numeric price
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.dropna(subset=["price"])
    df = df[df["price"] > 0]

    # Ensure discount is numeric (%)
    def extract_discount(val):
        if pd.isna(val) or str(val).strip() == "":
            return 0.0
        import re
        val_str = str(val)
        # Try to find an explicit percentage first
        m = re.search(r"(\d+(?:\.\d+)?)\s*%", val_str)
        if m:
            return float(m.group(1))
        # Try to find "خصم X"
        m2 = re.search(r"خصم\s*(\d+(?:\.\d+)?)", val_str)
        if m2:
            return float(m2.group(1))
        # Fallback to the first number found (capped later if needed)
        m3 = re.search(r"(\d+(?:\.\d+)?)", val_str)
        if m3:
            return float(m3.group(1))
        return 0.0

    df["discount"] = df["discount"].apply(extract_discount)

    return df.reset_index(drop=True)


@st.cache_data(ttl=3600, show_spinner=False)
def load_predictions() -> pd.DataFrame:
    """
    Load the latest price predictions from the pipeline.
    Expected columns: product_name_clean, date, current_price,
                      predicted_price, price_change, change_percentage, trend_label
    """
    df = _load_csv(PREDICTIONS_LOCAL, PREDICTIONS_REMOTE, "predictions")
    if df.empty:
        return df

    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    for col in ["current_price", "predicted_price", "price_change", "change_percentage"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.reset_index(drop=True)


def get_data_freshness() -> dict:
    """Return metadata about when each data file was last modified."""
    result = {}
    
    # Try to read the exact date from the CSV first
    try:
        if os.path.exists(PRODUCTS_LOCAL):
            df = pd.read_csv(PRODUCTS_LOCAL, usecols=["date"])
            result["products"] = df["date"].max()
        else:
            result["products"] = "not found"
    except Exception:
        if os.path.exists(PRODUCTS_LOCAL):
            import datetime
            mtime = os.path.getmtime(PRODUCTS_LOCAL)
            result["products"] = datetime.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
        else:
            result["products"] = "not found"
            
    # For predictions, just use mtime
    if os.path.exists(PREDICTIONS_LOCAL):
        import datetime
        mtime = os.path.getmtime(PREDICTIONS_LOCAL)
        result["predictions"] = datetime.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
    else:
        result["predictions"] = "not found"

    return result
