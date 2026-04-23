from __future__ import annotations

import glob
import importlib
import io
import os
from urllib.parse import urlparse

import pandas as pd
import requests
import streamlit as st

@st.cache_data(show_spinner=False)
def load_csv_data(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """Standardize an uploaded CSV into a daily Date-indexed dataframe with memory optimizations."""
    # First pass: read headers to find the timestamp column
    header_only = pd.read_csv(io.BytesIO(file_bytes), nrows=0)
    timestamp_candidates = [
        col for col in header_only.columns
        if any(token in str(col).lower() for token in ["time", "date", "timestamp"])
    ]
    if not timestamp_candidates:
        raise ValueError("No timestamp column found. Expected 'Date' or 'Time'.")
    
    timestamp_col = timestamp_candidates[0]
    
    # Identify OHLCV columns to keep (case-insensitive)
    standard_cols = ["open", "high", "low", "close", "volume", "price"]
    to_keep = [timestamp_col]
    found_standard = False
    for col in header_only.columns:
        if any(std in col.lower() for std in standard_cols):
            if col not in to_keep:
                to_keep.append(col)
                found_standard = True

    # If no standard columns are found, keep all columns so we don't end up with an empty dataframe
    if not found_standard:
        to_keep = list(header_only.columns)

    # Second pass: read only required columns to save memory
    raw = pd.read_csv(io.BytesIO(file_bytes), usecols=to_keep)

    # Robust Unix/ISO timestamp conversion
    if pd.api.types.is_numeric_dtype(raw[timestamp_col]):
        unit = "s"
        sample = raw[timestamp_col].dropna().iloc[0] if not raw[timestamp_col].dropna().empty else 0
        if sample > 1e12: unit = "ms"
        if sample > 1e15: unit = "us"
        raw[timestamp_col] = pd.to_datetime(raw[timestamp_col], unit=unit, errors="coerce")
    else:
        raw[timestamp_col] = pd.to_datetime(raw[timestamp_col], errors="coerce")

    # Drop invalid dates and set index
    raw = raw.dropna(subset=[timestamp_col]).set_index(timestamp_col).sort_index()
    
    # Resample to daily (takes first price of the day)
    # This massively reduces memory usage (e.g., from 7.5M rows to ~3k rows)
    daily = raw.resample("D").first()
    
    # Standardization: Title-case columns
    daily.columns = [col.strip().title() for col in daily.columns]
    
    # Fill gaps in the daily range
    full_range = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    daily = daily.reindex(full_range).ffill().dropna()
    daily.index.name = "Date"

    if daily.empty or len(daily.columns) == 0:
        raise ValueError(
            "The dataset was parsed but resulted in zero usable columns. "
            "Please ensure the CSV contains numeric price data."
        )

    return daily


@st.cache_data(show_spinner=False)
def normalize_remote_csv_link(link: str) -> str:
    """Convert known non-raw file links to directly downloadable CSV links."""
    parsed = urlparse(link)
    host = parsed.netloc.lower()

    if host in {"github.com", "www.github.com"} and "/blob/" in parsed.path:
        parts = [part for part in parsed.path.split("/") if part]
        if len(parts) >= 5 and parts[2] == "blob":
            owner, repo = parts[0], parts[1]
            branch = parts[3]
            file_path = "/".join(parts[4:])
            return f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{file_path}"

    return link


def _extract_kaggle_slug(link: str) -> str:
    if "kaggle.com/datasets/" in link:
        suffix = link.split("kaggle.com/datasets/")[1]
        parts = suffix.split("/")
        if len(parts) < 2:
            raise ValueError("Invalid Kaggle dataset URL format.")
        return f"{parts[0]}/{parts[1].split('?')[0]}"

    return link


@st.cache_data(show_spinner=False)
def fetch_data_from_link(link: str) -> pd.DataFrame:
    """Download and normalize dataset with persistent disk caching."""
    cleaned = link.strip()
    if not cleaned:
        raise ValueError("Please enter a valid URL or Kaggle slug.")

    # Setup persistent cache directory
    cache_dir = os.path.join(os.getcwd(), "data_cache")
    os.makedirs(cache_dir, exist_ok=True)

    import hashlib
    link_hash = hashlib.md5(cleaned.encode()).hexdigest()
    cached_file = os.path.join(cache_dir, f"{link_hash}.csv")

    # 1. Check local disk cache first (skip if file is too small / corrupted)
    if os.path.exists(cached_file) and os.path.getsize(cached_file) > 100:
        try:
            with open(cached_file, "rb") as f:
                return load_csv_data(f.read(), f"cached_{link_hash}.csv")
        except Exception:
            # Cache file is corrupted, delete it and re-download
            os.remove(cached_file)

    # 2. If not in cache, download
    if cleaned.startswith("http") and "kaggle.com" not in cleaned:
        download_link = normalize_remote_csv_link(cleaned)
        response = requests.get(download_link, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
        response.raise_for_status()
        
        # Save to disk cache
        with open(cached_file, "wb") as f:
            f.write(response.content)
            
        return load_csv_data(response.content, os.path.basename(urlparse(download_link).path) or "data.csv")

    # 3. Handle Kaggle
    slug = _extract_kaggle_slug(cleaned)
    kagglehub = importlib.import_module("kagglehub")
    dataset_path = kagglehub.dataset_download(slug)

    csv_files = glob.glob(os.path.join(dataset_path, "*.csv"))
    if not csv_files:
        raise FileNotFoundError("No CSV file found in the downloaded Kaggle dataset.")

    with open(csv_files[0], "rb") as handle:
        content = handle.read()
        # Save to disk cache for future quick access without kagglehub logic
        with open(cached_file, "wb") as f:
            f.write(content)
        return load_csv_data(content, os.path.basename(csv_files[0]))


def get_default_price_column(dataframe: pd.DataFrame) -> str:
    """Prefer Close when available; otherwise use the first numeric column."""
    if dataframe.empty or len(dataframe.columns) == 0:
        raise ValueError("Dataset has no columns. Please load a valid CSV.")
    if "Close" in dataframe.columns:
        return "Close"
    # Prefer the first numeric column that looks like a price
    for col in dataframe.columns:
        if any(kw in col.lower() for kw in ["price", "close", "value"]):
            return str(col)
    return str(dataframe.columns[0])
