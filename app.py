"""
Bitcoin Price Forecasting - Multi-Page Streamlit Application
=============================================================
Pages   : 1. Data Loader
          2. Explore Data
          3. Forecasting Engine

Color Palette : Institutional Navy and Gold
Font          : Space Grotesk + JetBrains Mono
"""

from __future__ import annotations

import warnings

import streamlit as st

from btc_portal.configuration import render_engine_configuration
from btc_portal.data_pipeline import get_default_price_column
from btc_portal.forecasting import run_model
from btc_portal.ingestion import (
    get_active_dataframe,
    get_uploader_widget_key,
    handle_remote_link_load,
    handle_uploaded_file,
    initialize_uploader_state,
    pop_upload_notice,
)
from btc_portal.ui import (
    configure_page,
    inject_custom_css,
    kpi_row,
    no_data_gate,
    page_header,
    render_sidebar_navigation,
    section_title,
)
from btc_portal.visualization import (
    build_candlestick_volume_figure,
    build_decomposition_figure,
    build_forecast_projection_figure,
    build_loader_price_figure,
    build_monthly_seasonality_figure,
    build_return_distribution_figure,
)

warnings.filterwarnings("ignore")

configure_page()
inject_custom_css()


def render_data_loader_page() -> None:
    page_header(
        "₿  Data Loader",
        "Import historical BTC/USD market data via CSV upload or remote link.",
    )

    initialize_uploader_state()

    notice = pop_upload_notice()
    if notice:
        st.success(notice)

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("#### 🗂  Upload Local CSV")
        uploaded = st.file_uploader(
            "Drop your BTC/USD CSV here",
            type=["csv"],
            key=get_uploader_widget_key(),
            label_visibility="collapsed",
        )

        if uploaded is not None:
            upload_error = handle_uploaded_file(uploaded)
            if upload_error:
                st.error(f"❌  {upload_error}")

    with col2:
        st.markdown("#### 🌐  Fetch from URL or Kaggle")
        link_input = st.text_input(
            "Raw CSV URL or Kaggle dataset slug",
            placeholder="https://.../data.csv   or   mczielinski/bitcoin-...",
            label_visibility="collapsed",
        )
        if st.button("⬇️  Download & Load", type="primary"):
            remote_error = handle_remote_link_load(link_input)
            if remote_error:
                st.error(f"❌  {remote_error}")
            else:
                dataframe = get_active_dataframe()
                if dataframe is not None:
                    st.success(f"✅  Dataset loaded - {len(dataframe):,} rows.")

    dataframe = get_active_dataframe()
    if dataframe is None:
        return

    price_col = get_default_price_column(dataframe)

    section_title("Dataset Overview")
    kpi_row(
        [
            {"label": "Total Rows", "value": f"{len(dataframe):,}", "sub": "daily observations"},
            {
                "label": "Date Range",
                "value": f"{dataframe.index.max().year - dataframe.index.min().year} yrs",
                "sub": f"{dataframe.index.min().date()} -> {dataframe.index.max().date()}",
            },
            {"label": "Columns", "value": str(len(dataframe.columns)), "sub": "features available"},
            {
                "label": "Latest Price",
                "value": f"${dataframe[price_col].iloc[-1]:,.2f}",
                "sub": str(dataframe.index[-1].date()),
            },
        ]
    )

    with st.expander("🔍  First 20 rows"):
        st.dataframe(dataframe.head(20), use_container_width=True)

    with st.spinner("📊  Rendering chart..."):
        figure = build_loader_price_figure(dataframe, price_col)
        st.plotly_chart(figure, use_container_width=True)


def render_explore_page() -> None:
    page_header(
        "📊  Explore Data",
        "Deep-dive into price action, volatility, decomposition, and return distributions.",
    )

    dataframe = get_active_dataframe()
    if dataframe is None:
        no_data_gate()

    price_col = get_default_price_column(dataframe)
    has_ohlc = all(col in dataframe.columns for col in ["Open", "High", "Low", "Close"])
    has_volume = "Volume" in dataframe.columns

    with st.spinner("📊  Generating visualizations..."):
        section_title("Price Action & Trading Volume")
        candle_figure = build_candlestick_volume_figure(dataframe, price_col, has_ohlc, has_volume)
        st.plotly_chart(candle_figure, use_container_width=True)

        section_title("Time Series Decomposition  (30-Day Period)")
        if len(dataframe) >= 60:
            decomp_figure = build_decomposition_figure(dataframe, price_col)
            st.plotly_chart(decomp_figure, use_container_width=True)
        else:
            st.info("Need at least 60 days of data for decomposition.")

        section_title("Return Seasonality & Distribution")
        left_col, right_col = st.columns(2, gap="large")

        with left_col:
            monthly_figure = build_monthly_seasonality_figure(dataframe, price_col)
            st.plotly_chart(monthly_figure, use_container_width=True)

        with right_col:
            return_figure = build_return_distribution_figure(dataframe, price_col)
            st.plotly_chart(return_figure, use_container_width=True)


def render_forecasting_page() -> None:
    page_header(
        "🔮  Forecasting Engine",
        "Train, evaluate and project BTC prices with statistical and deep-learning models.",
    )

    dataframe = get_active_dataframe()
    if dataframe is None:
        no_data_gate()

    st.markdown("### ⚙️  Engine Configuration")
    config = render_engine_configuration(dataframe)

    series = dataframe[config.price_col].dropna()
    split_idx = int(len(series) * 0.85)
    train = series.iloc[:split_idx]
    test = series.iloc[split_idx:]

    if config.run_requested:
        with st.spinner(f"⏳  Training {config.model_choice}..."):
            result = run_model(config.model_choice, train, test, config.horizon, config.ci_pct)
            st.session_state["last_result"] = result
            st.session_state["last_train"] = train
            st.session_state["last_test"] = test
            st.session_state["last_ci_pct"] = config.ci_pct

    st.write("<hr>", unsafe_allow_html=True)

    if "last_result" in st.session_state:
        result = st.session_state["last_result"]
        last_train = st.session_state["last_train"]
        last_test = st.session_state["last_test"]
        ci_pct = int(st.session_state.get("last_ci_pct", config.ci_pct))

        section_title("Holdout Error Metrics")
        kpi_row(
            [
                {
                    "label": "Algorithm",
                    "value": str(result["model_name"]).split()[0],
                    "sub": "active model",
                },
                {"label": "MAE", "value": f"${result['mae']:,.2f}", "sub": "mean absolute error"},
                {"label": "RMSE", "value": f"${result['rmse']:,.2f}", "sub": "root mean squared error"},
                {"label": "MAPE", "value": f"{result['mape']:.2f}%", "sub": "percentage error"},
            ]
        )

        with st.spinner("📈  Rendering forecast chart..."):
            forecast_figure = build_forecast_projection_figure(last_train, last_test, result, ci_pct)
            st.plotly_chart(forecast_figure, use_container_width=True)
    else:
        kpi_row(
            [
                {"label": "Training Rows", "value": f"{len(train):,}", "sub": "85% split"},
                {"label": "Test Rows", "value": f"{len(test):,}", "sub": "15% holdout"},
                {"label": "Target Col", "value": config.price_col, "sub": "selected variable"},
                {"label": "Horizon", "value": f"{config.horizon}d", "sub": "forecast window"},
            ]
        )
        st.info("Configure the model above and click **Run Simulation** to begin.")


current_page = render_sidebar_navigation()

if current_page == "Data Loader":
    render_data_loader_page()
elif current_page == "Explore Data":
    render_explore_page()
elif current_page == "Forecasting Engine":
    render_forecasting_page()
