from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.seasonal import seasonal_decompose

from .ui import (
    C_BRIGHT_BLUE,
    C_GREEN,
    C_PRIMARY,
    C_RED,
    C_VOL_DOWN,
    C_VOL_UP,
    apply_layout,
)


def build_loader_price_figure(dataframe: pd.DataFrame, price_col: str) -> go.Figure:
    fig = go.Figure(
        go.Scatter(
            x=dataframe.index,
            y=dataframe[price_col],
            line=dict(color=C_PRIMARY, width=1.8),
            name=price_col,
            hovertemplate="<b>%{x|%Y-%m-%d}</b><br>$%{y:,.2f}<extra></extra>",
        )
    )
    apply_layout(fig, "Historical BTC Price Action", height=460)
    fig.update_layout(xaxis_title="Date", yaxis_title="Price (USD)", hovermode="x unified")
    return fig


def build_candlestick_volume_figure(
    dataframe: pd.DataFrame,
    price_col: str,
    has_ohlc: bool,
    has_volume: bool,
) -> go.Figure:
    rows = 2 if has_volume else 1
    heights = [0.72, 0.28] if has_volume else [1.0]

    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        row_heights=heights,
        vertical_spacing=0.04,
    )

    if has_ohlc:
        fig.add_trace(
            go.Candlestick(
                x=dataframe.index,
                open=dataframe["Open"],
                high=dataframe["High"],
                low=dataframe["Low"],
                close=dataframe["Close"],
                increasing_line_color=C_GREEN,
                decreasing_line_color=C_RED,
                name="OHLC",
            ),
            row=1,
            col=1,
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=dataframe.index,
                y=dataframe[price_col],
                line=dict(color=C_PRIMARY),
                name=price_col,
            ),
            row=1,
            col=1,
        )

    working = dataframe.copy()
    working["SMA_50"] = working[price_col].rolling(50).mean()
    fig.add_trace(
        go.Scatter(
            x=working.index,
            y=working["SMA_50"],
            line=dict(color=C_BRIGHT_BLUE, width=1.5, dash="dot"),
            name="50-SMA",
        ),
        row=1,
        col=1,
    )

    if has_volume:
        if has_ohlc:
            vol_colors = [
                C_VOL_UP if working["Close"].iloc[i] >= working["Open"].iloc[i] else C_VOL_DOWN
                for i in range(len(working))
            ]
        else:
            vol_colors = [C_VOL_UP] * len(working)

        fig.add_trace(
            go.Bar(
                x=working.index,
                y=working["Volume"],
                marker_color=vol_colors,
                marker_line_width=0,
                name="Volume",
                opacity=0.9,
            ),
            row=2,
            col=1,
        )
        fig.update_yaxes(title_text="Volume", row=2, col=1)

    apply_layout(fig, "Candlestick Chart & Volume", height=620)
    fig.update_layout(
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", y=1.06, x=0.5, xanchor="center"),
        hovermode="x unified",
    )
    fig.update_xaxes(title_text="Date", row=rows, col=1)
    fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
    return fig


def build_decomposition_figure(dataframe: pd.DataFrame, price_col: str) -> go.Figure:
    decomp = seasonal_decompose(dataframe[price_col].dropna(), period=30, model="additive")

    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        subplot_titles=("Observed", "Trend", "Seasonal", "Residual"),
        vertical_spacing=0.07,
    )
    fig.add_trace(
        go.Scatter(
            x=dataframe.index,
            y=decomp.observed,
            line=dict(color=C_PRIMARY, width=1.5),
            name="Observed",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=dataframe.index,
            y=decomp.trend,
            line=dict(color=C_BRIGHT_BLUE, width=1.5),
            name="Trend",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=dataframe.index,
            y=decomp.seasonal,
            line=dict(color=C_GREEN, width=1.5),
            name="Seasonal",
        ),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=dataframe.index,
            y=decomp.resid,
            mode="markers",
            marker=dict(color=C_RED, size=3),
            name="Residual",
        ),
        row=4,
        col=1,
    )

    apply_layout(fig, "Additive Decomposition", height=720)
    fig.update_layout(showlegend=False)
    for annotation in fig["layout"]["annotations"]:
        annotation["font"] = dict(color="#FFFFFF", size=13)

    return fig


def build_monthly_seasonality_figure(dataframe: pd.DataFrame, price_col: str) -> go.Figure:
    month_order = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ]

    dr = dataframe.copy()
    dr["log_return"] = np.log(dr[price_col] / dr[price_col].shift(1))
    dr["Month"] = dr.index.month_name()

    fig = go.Figure()
    for month in month_order:
        values = dr[dr["Month"] == month]["log_return"].dropna() * 100
        if not values.empty:
            fig.add_trace(go.Box(y=values, name=month[:3], marker_color=C_PRIMARY))

    apply_layout(fig, "Monthly Return Seasonality (%)", height=360)
    fig.update_layout(showlegend=False, xaxis_title="Month", yaxis_title="Log Return (%)")
    return fig


def build_return_distribution_figure(dataframe: pd.DataFrame, price_col: str) -> go.Figure:
    dr = dataframe.copy()
    dr["log_return"] = np.log(dr[price_col] / dr[price_col].shift(1))
    rc = dr["log_return"].dropna() * 100

    fig = go.Figure(
        go.Histogram(
            x=rc,
            nbinsx=80,
            marker_color=C_PRIMARY,
            opacity=0.75,
            name="Returns",
        )
    )

    if not rc.empty:
        mu = float(rc.mean())
        sigma = float(rc.std())
        if sigma > 0:
            x_norm = np.linspace(float(rc.min()), float(rc.max()), 200)
            y_norm = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-((x_norm - mu) ** 2) / (2 * sigma**2))
            y_norm = y_norm * len(rc) * (float(rc.max()) - float(rc.min())) / 80
            fig.add_trace(
                go.Scatter(
                    x=x_norm,
                    y=y_norm,
                    mode="lines",
                    line=dict(color=C_GREEN, width=2),
                    name="Normal Fit",
                )
            )

    apply_layout(fig, "Log-Return Distribution (%)", height=360)
    fig.update_layout(xaxis_title="Log Return (%)", yaxis_title="Frequency")
    return fig


def build_forecast_projection_figure(
    train: pd.Series,
    test: pd.Series,
    result: dict[str, Any],
    ci_pct: int,
) -> go.Figure:
    fig = go.Figure()
    plot_train = train[-365:] if len(train) > 365 else train

    fig.add_trace(
        go.Scatter(
            x=plot_train.index,
            y=plot_train,
            name="Training History",
            line=dict(color=C_BRIGHT_BLUE, width=1.5),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=test.index,
            y=test,
            name="Actual (Holdout)",
            line=dict(color=C_PRIMARY, width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=result["test_dates"],
            y=result["test_pred"],
            name="Backtest",
            line=dict(color=C_GREEN, dash="dot", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=result["future_dates"],
            y=result["future_pred"],
            name="Forecast",
            line=dict(color="#00FFD1", width=2.5),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([result["future_dates"], result["future_dates"][::-1]]),
            y=np.concatenate([result["future_upper"], result["future_lower"][::-1]]),
            fill="toself",
            fillcolor="rgba(0,255,209,0.10)",
            line=dict(width=0),
            name=f"{ci_pct}% CI",
        )
    )

    apply_layout(fig, f"{result['model_name']} — Price Projection", height=620)
    fig.update_layout(
        legend=dict(orientation="h", y=1.06, x=0.5, xanchor="center"),
        hovermode="x unified",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
    )
    return fig
