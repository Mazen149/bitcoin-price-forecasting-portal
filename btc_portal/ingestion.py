from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from .data_pipeline import fetch_data_from_link, standardize_and_load_data

UPLOADER_KEY_STATE = "uploader_key"
UPLOAD_NOTICE_STATE = "upload_notice"
DATAFRAME_STATE = "df"
DATA_SOURCE_STATE = "data_source"


def initialize_uploader_state() -> None:
    """Initialize upload-widget keys used to avoid file-uploader flicker."""
    if UPLOADER_KEY_STATE not in st.session_state:
        st.session_state[UPLOADER_KEY_STATE] = 0
    if UPLOAD_NOTICE_STATE not in st.session_state:
        st.session_state[UPLOAD_NOTICE_STATE] = None


def get_uploader_widget_key() -> str:
    return f"csv_uploader_{st.session_state.get(UPLOADER_KEY_STATE, 0)}"


def pop_upload_notice() -> str | None:
    return st.session_state.pop(UPLOAD_NOTICE_STATE, None)


def get_active_dataframe() -> pd.DataFrame | None:
    if DATAFRAME_STATE not in st.session_state:
        return None
    return st.session_state[DATAFRAME_STATE]


def _set_active_dataframe(dataframe: pd.DataFrame, source: str) -> None:
    st.session_state[DATAFRAME_STATE] = dataframe
    st.session_state[DATA_SOURCE_STATE] = source
    clear_analysis_state()


def clear_analysis_state() -> None:
    """Clear all analysis results and AI summaries when a new dataset is loaded."""
    keys_to_clear = [
        "explore_ai_summary",
        "forecast_ai_summary",
        "last_result",
        "last_train",
        "last_test",
        "last_ci_pct",
        "last_model_choice",
    ]
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]


def handle_uploaded_file(uploaded_file: object) -> str | None:
    """Parse uploaded CSV, persist it in session, then rerun to clear uploader UI artifacts."""
    file_obj = uploaded_file

    try:
        with st.spinner("⏳  Parsing and standardizing data…"):
            payload = getattr(cast_to_any(file_obj), "read")()
            filename = str(getattr(cast_to_any(file_obj), "name", "uploaded.csv"))
            dataframe = standardize_and_load_data(payload, filename)
            _set_active_dataframe(dataframe, "local")

        st.session_state[UPLOAD_NOTICE_STATE] = f"✅  Loaded **{filename}** — {len(dataframe):,} rows"
        st.session_state[UPLOADER_KEY_STATE] = st.session_state.get(UPLOADER_KEY_STATE, 0) + 1
        st.rerun()
    except Exception as exc:
        return str(exc)

    return None


def handle_remote_link_load(link_input: str) -> str | None:
    """Download CSV from URL or Kaggle and persist in Streamlit session state."""
    if not link_input.strip():
        return "Please enter a valid URL or Kaggle slug."

    try:
        with st.spinner("⏳  Fetching dataset…"):
            dataframe = fetch_data_from_link(link_input)
            _set_active_dataframe(dataframe, "remote")
            
            # Extract a filename-like string from the link for the notice
            import os
            from urllib.parse import urlparse
            display_name = os.path.basename(urlparse(link_input).path) or link_input
            if "?" in display_name: display_name = display_name.split("?")[0]
            if not display_name.endswith(".csv") and "/" not in display_name:
                display_name = f"{display_name}.csv"
            
            st.session_state[UPLOAD_NOTICE_STATE] = f"✅  Loaded **{display_name}** — {len(dataframe):,} rows"
            st.rerun()
    except Exception as exc:
        return str(exc)

    return None


def cast_to_any(value: object) -> Any:
    return value
