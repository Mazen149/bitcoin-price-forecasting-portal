from __future__ import annotations

import glob
import importlib
import io
import os
from urllib.parse import urlparse

import pandas as pd
import requests
import streamlit as st

MIN_DAILY_ROWS = 60


@st.cache_data(show_spinner=False)
def load_csv_data(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """Standardize an uploaded CSV into a daily Date-indexed dataframe."""
    raw = pd.read_csv(io.BytesIO(file_bytes))

    timestamp_candidates = [
        col
        for col in raw.columns
        if any(token in str(col).lower() for token in ["time", "date", "timestamp"])
    ]
    if not timestamp_candidates:
        raise ValueError("No timestamp column found. Expected 'Date' or 'Time'.")

    timestamp_column = timestamp_candidates[0]
    raw[timestamp_column] = pd.to_datetime(raw[timestamp_column], errors="coerce")
    raw = raw.dropna(subset=[timestamp_column]).sort_values(timestamp_column).reset_index(drop=True)

    raw["Date"] = raw[timestamp_column].dt.date
    daily = raw.groupby("Date").first().reset_index()
    daily["Date"] = pd.to_datetime(daily["Date"])

    daily.columns = [col.strip().title() if col != "Date" else col for col in daily.columns]
    daily = daily.set_index("Date").sort_index()

    full_range = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    daily = daily.reindex(full_range).ffill().dropna()

    if len(daily) < MIN_DAILY_ROWS:
        raise ValueError(f"Need at least {MIN_DAILY_ROWS} days of data after cleaning.")

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
    """Download and normalize dataset from HTTP CSV links or Kaggle dataset slugs/URLs."""
    cleaned = link.strip()
    if not cleaned:
        raise ValueError("Please enter a valid URL or Kaggle slug.")

    if cleaned.startswith("http") and "kaggle.com" not in cleaned:
        download_link = normalize_remote_csv_link(cleaned)
        response = requests.get(download_link, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
        response.raise_for_status()

        content_type = response.headers.get("Content-Type", "").lower()
        payload_head = response.content[:500].lstrip().lower()
        if (
            "text/html" in content_type
            or payload_head.startswith(b"<!doctype html")
            or payload_head.startswith(b"<html")
        ):
            raise ValueError(
                "The provided link returned HTML instead of CSV. "
                "For GitHub, paste the file URL (with /blob/) or a raw.githubusercontent.com CSV link."
            )

        filename = os.path.basename(urlparse(download_link).path) or "downloaded_data.csv"
        return load_csv_data(response.content, filename)

    slug = _extract_kaggle_slug(cleaned)
    kagglehub = importlib.import_module("kagglehub")
    dataset_path = kagglehub.dataset_download(slug)

    csv_files = glob.glob(os.path.join(dataset_path, "*.csv"))
    if not csv_files:
        raise FileNotFoundError("No CSV file found in the downloaded Kaggle dataset.")

    with open(csv_files[0], "rb") as handle:
        return load_csv_data(handle.read(), os.path.basename(csv_files[0]))


def get_default_price_column(dataframe: pd.DataFrame) -> str:
    """Prefer Close when available; otherwise use the first column."""
    if "Close" in dataframe.columns:
        return "Close"
    return str(dataframe.columns[0])
