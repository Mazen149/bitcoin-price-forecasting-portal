from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import streamlit as st

MODEL_OPTIONS = [
    "LSTM (Deep Learning)",
    "Holt-Winters Smoothing",
    "ARIMA",
]


@dataclass(slots=True)
class EngineConfig:
    """Runtime controls selected by the user in Forecasting Engine."""

    price_col: str
    model_choice: str
    horizon: int
    ci_pct: int
    run_requested: bool


def _target_columns(dataframe: pd.DataFrame) -> list[str]:
    preferred = [col for col in ["Close", "Open", "High", "Low"] if col in dataframe.columns]
    if preferred:
        return preferred

    numeric_cols = [col for col in dataframe.columns if pd.api.types.is_numeric_dtype(dataframe[col])]
    if numeric_cols:
        return numeric_cols

    return [str(dataframe.columns[0])]


def render_engine_configuration(dataframe: pd.DataFrame) -> EngineConfig:
    """Render forecasting controls equivalent to the monolithic app configuration row."""
    c1, c2, c3, c4, c5 = st.columns([1.5, 2, 1.5, 1.5, 1.5])

    with c1:
        price_col = st.selectbox("Target Variable", _target_columns(dataframe))

    with c2:
        model_choice = st.selectbox("Algorithm", MODEL_OPTIONS, index=0)

    with c3:
        horizon = st.slider("Horizon (Days)", 7, 180, 90)

    with c4:
        ci_pct = st.selectbox("Confidence Band", [80, 90, 95, 99], index=2)

    with c5:
        st.write("<br>", unsafe_allow_html=True)
        run_requested = st.button("🚀  Run Simulation", type="primary", use_container_width=True)

    return EngineConfig(
        price_col=price_col,
        model_choice=model_choice,
        horizon=int(horizon),
        ci_pct=int(ci_pct),
        run_requested=run_requested,
    )
