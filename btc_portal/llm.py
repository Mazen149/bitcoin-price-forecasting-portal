from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

DEFAULT_GROQ_MODEL = "llama-3.1-8b-instant"


def _read_secret_value(name: str) -> str:
    try:
        if name in st.secrets:
            return str(st.secrets[name]).strip()
    except Exception:
        return ""
    return ""


def get_groq_api_key() -> str:
    """Read Groq API key from Streamlit secrets or environment variables."""
    secret_key = _read_secret_value("GROQ_API_KEY")
    if secret_key:
        return secret_key
    return os.getenv("GROQ_API_KEY", "").strip()


def get_groq_model() -> str:
    secret_model = _read_secret_value("GROQ_MODEL")
    if secret_model:
        return secret_model
    return os.getenv("GROQ_MODEL", "").strip() or DEFAULT_GROQ_MODEL


def _call_groq(prompt: str) -> str:
    api_key = get_groq_api_key()
    if not api_key:
        raise ValueError(
            "Groq API key is not configured. Add GROQ_API_KEY to Streamlit secrets "
            "or your environment variables."
        )

    try:
        from groq import Groq  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Missing dependency 'groq'. Install it with `pip install groq`.") from exc

    client = Groq(api_key=api_key)
    model = get_groq_model()

    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            top_p=0.9,
            max_tokens=700,
        )
    except Exception as exc:
        raise RuntimeError(f"Groq request failed: {exc}") from exc

    message = getattr(completion.choices[0], "message", None)
    text_output = (getattr(message, "content", "") or "").strip()
    if not text_output:
        raise RuntimeError("Groq returned an empty response.")
    return text_output


def _call_llm(prompt: str) -> str:
    return _call_groq(prompt)


def _series_change_pct(series: pd.Series, days: int) -> float:
    if len(series) <= days:
        return float("nan")
    base = float(series.iloc[-(days + 1)])
    latest = float(series.iloc[-1])
    if abs(base) < 1e-9:
        return float("nan")
    return ((latest - base) / base) * 100


def _format_float(value: float) -> str:
    if np.isnan(value):
        return "N/A"
    return f"{value:,.2f}"


def explain_explore_data_with_llm(dataframe: pd.DataFrame, price_col: str) -> str:
    """Generate natural-language insights for Explore Data visual diagnostics."""
    prices = dataframe[price_col].dropna()
    returns = (np.log(prices / prices.shift(1)).dropna() * 100).astype(float)

    change_7d = _series_change_pct(prices, 7)
    change_30d = _series_change_pct(prices, 30)
    annualized_vol = float(returns.std() * np.sqrt(365)) if len(returns) > 1 else float("nan")

    monthly_mean = returns.groupby(returns.index.month).mean() if not returns.empty else pd.Series(dtype=float)
    best_month = int(monthly_mean.idxmax()) if not monthly_mean.empty else None
    worst_month = int(monthly_mean.idxmin()) if not monthly_mean.empty else None

    volume_stats = "Volume column unavailable"
    if "Volume" in dataframe.columns:
        vol = dataframe["Volume"].dropna()
        if not vol.empty:
            volume_stats = f"Avg volume: {vol.mean():,.0f}; latest volume: {vol.iloc[-1]:,.0f}"

    context_lines = [
        f"Rows: {len(dataframe):,}",
        f"Date range: {dataframe.index.min().date()} to {dataframe.index.max().date()}",
        f"Target column: {price_col}",
        f"Latest price: {prices.iloc[-1]:,.2f}",
        f"Min/Max price: {prices.min():,.2f} / {prices.max():,.2f}",
        f"7-day change (%): {_format_float(change_7d)}",
        f"30-day change (%): {_format_float(change_30d)}",
        f"Mean daily log return (%): {_format_float(float(returns.mean()) if not returns.empty else float('nan'))}",
        f"Daily return std (%): {_format_float(float(returns.std()) if len(returns) > 1 else float('nan'))}",
        f"Annualized volatility estimate (%): {_format_float(annualized_vol)}",
        f"Best month index (1-12): {best_month if best_month is not None else 'N/A'}",
        f"Worst month index (1-12): {worst_month if worst_month is not None else 'N/A'}",
        volume_stats,
    ]

    prompt = (
        "You are a Senior Institutional Investment Strategist at a top-tier global bank.\n"
        "Provide professional, high-impact business insights based on the provided technical data.\n"
        "Focus on market structure, volatility regimes, and risk assessment that a professional trader or fund manager would find valuable.\n"
        "Use sophisticated financial terminology but keep the delivery EXTREMELY concise and punchy.\n"
        "Format the output elegantly with bold text, emojis, and clear spacing.\n"
        "Do NOT include generic advice or sections like “What to do next”.\n"
        "Avoid hype; focus on the data-driven market narrative.\n"
        "Use these sections exactly (including the emojis in the headers):\n"
        "### 💡 Strategic Business Insights\n"
        "### 📊 Market Regime Analysis\n"
        "### ⚠️ Institutional Risk Assessment\n"
        "CRITICAL INSTRUCTION: Under each section, you MUST provide EXACTLY 3 short bullet points. No more, no less.\n"
        "Keep the entire response under 120 words total.\n\n"
        "Explore data summary:\n"
        + "\n".join(f"- {line}" for line in context_lines)
    )

    return _call_llm(prompt)


def explain_forecast_with_llm(
    train: pd.Series,
    test: pd.Series,
    result: dict[str, Any],
    ci_pct: int,
) -> str:
    """Generate natural-language explanation for forecast quality and projection."""
    test_pred = np.asarray(result.get("test_pred", []), dtype=float)
    test_actual = np.asarray(test.values, dtype=float)
    future_pred = np.asarray(result.get("future_pred", []), dtype=float)
    future_lower = np.asarray(result.get("future_lower", []), dtype=float)
    future_upper = np.asarray(result.get("future_upper", []), dtype=float)

    forecast_change_pct = float("nan")
    if future_pred.size >= 2 and abs(future_pred[0]) > 1e-9:
        forecast_change_pct = ((future_pred[-1] - future_pred[0]) / future_pred[0]) * 100

    mean_bias = float("nan")
    if test_pred.size == test_actual.size and test_pred.size > 0:
        mean_bias = float((test_actual - test_pred).mean())

    avg_ci_width = float("nan")
    if future_lower.size == future_upper.size and future_lower.size > 0:
        avg_ci_width = float(np.mean(future_upper - future_lower))

    direction_label = "N/A"
    if future_pred.size >= 2:
        start = float(future_pred[0])
        end = float(future_pred[-1])
        if abs(end - start) < 1e-9:
            direction_label = "Flat"
        elif end > start:
            direction_label = "Increase"
        else:
            direction_label = "Decrease"

    context_lines = [
        f"Model: {result.get('model_name', 'Unknown')}",
        f"Train points: {len(train):,}",
        f"Test points: {len(test):,}",
        f"MAE: {float(result.get('mae', np.nan)):,.3f}",
        f"RMSE: {float(result.get('rmse', np.nan)):,.3f}",
        f"MAPE (%): {float(result.get('mape', np.nan)):,.3f}",
        f"Mean test-data bias (test data - prediction): {_format_float(mean_bias)}",
        f"Forecast horizon points: {len(future_pred)}",
        f"Forecast start/end: {_format_float(float(future_pred[0]) if future_pred.size else float('nan'))} -> "
        f"{_format_float(float(future_pred[-1]) if future_pred.size else float('nan'))}",
        f"Direction (from forecast start/end): {direction_label}",
        f"Forecast total change (%): {_format_float(forecast_change_pct)}",
        f"Average {ci_pct}% interval width: {_format_float(avg_ci_width)}",
    ]

    prompt = (
        "You are a Chief Investment Officer (CIO) reviewing predictive models for capital allocation.\n"
        "Translate the statistical metrics into a professional business outlook and operational strategy.\n"
        "Focus on the 'Bottom Line': What does this forecast imply for the market trend and institutional confidence?\n"
        "Interpret the accuracy metrics (MAE/MAPE) as 'Predictive Reliability' rather than just raw numbers.\n"
        "Format the output elegantly with bold text, emojis, and clear spacing.\n"
        "Do NOT include advice or 'next steps'. Focus on the high-level interpretation.\n"
        "Use these sections exactly (including the emojis in the headers):\n"
        "### 📈 Forecast Market Outlook\n"
        "### 🎯 Predictive Integrity & Reliability\n"
        "### 📉 Risk Exposure & Uncertainty\n"
        "CRITICAL INSTRUCTION: Under each section, you MUST provide EXACTLY 3 short bullet points. No more, no less.\n"
        "Keep the entire response under 120 words total.\n\n"
        "Forecast summary data (use for your analysis, but summarize in words):\n"
        + "\n".join(f"- {line}" for line in context_lines)
    )

    return _call_llm(prompt)