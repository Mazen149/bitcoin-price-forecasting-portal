from __future__ import annotations

import streamlit as st

from btc_portal.ingestion import (
    IngestionError,
    detect_default_price_column,
    detect_timestamp_column,
    get_price_candidates,
    get_timestamp_candidates,
    prepare_btc_history,
    read_btc_csv,
    read_btc_csv_from_link,
)


def main() -> None:
    st.set_page_config(page_title="Bitcoin Price Forecasting Portal", layout="wide")

    st.title("Bitcoin Price Forecasting Portal")
    st.caption("Milestone A: Data ingestion and validation for Kaggle BTC CSV files")

    st.info(
        "This commit implements only Part A (Data Ingestion). "
        "Forecasting and model comparison modules will be added in future commits."
    )

    source_mode = st.radio(
        "Data source",
        options=["Upload CSV", "Insert dataset link"],
        horizontal=True,
    )

    raw_df = None
    source_label = ""

    if source_mode == "Upload CSV":
        uploaded_file = st.file_uploader("Upload Bitcoin historical CSV", type=["csv"])

        if uploaded_file is None:
            st.markdown(
                "Upload a Kaggle-style BTC dataset to begin. "
                "Typical timestamp columns include `Date` and `Timestamp`."
            )
            return

        try:
            raw_df = read_btc_csv(uploaded_file)
        except IngestionError as exc:
            st.error(str(exc))
            return

        source_label = uploaded_file.name
    else:
        st.markdown(
            "Use either a direct CSV URL or a Kaggle dataset URL "
            "such as https://www.kaggle.com/datasets/owner/dataset."
        )

        with st.form("dataset_link_form"):
            dataset_link = st.text_input(
                "Dataset link",
                placeholder="https://www.kaggle.com/datasets/mczielinski/bitcoin-historical-data",
            )
            kaggle_csv_filename = st.text_input(
                "Kaggle CSV file name (optional)",
                placeholder="example: btcusd_1-min_data.csv",
                help="Use this when the Kaggle dataset has multiple CSV files.",
            )
            fetch_link = st.form_submit_button("Fetch dataset link", type="primary")

        if fetch_link:
            try:
                raw_df = read_btc_csv_from_link(
                    dataset_link=dataset_link,
                    kaggle_csv_filename=kaggle_csv_filename.strip() or None,
                )
            except IngestionError as exc:
                st.error(str(exc))
                st.session_state.pop("raw_df_from_link", None)
                st.session_state.pop("raw_df_link_source", None)
            else:
                st.session_state["raw_df_from_link"] = raw_df
                st.session_state["raw_df_link_source"] = dataset_link.strip()

        raw_df = st.session_state.get("raw_df_from_link")
        source_label = st.session_state.get("raw_df_link_source", "")

        if raw_df is None:
            st.info("Insert a dataset link and click Fetch dataset link to continue.")
            return

    st.success(f"Loaded {len(raw_df):,} rows and {len(raw_df.columns):,} columns.")
    if source_label:
        st.caption(f"Source: {source_label}")

    st.subheader("Ingested Dataset")
    st.caption("Preview and inspect the data immediately after ingestion.")

    max_preview_rows = max(1, min(500, len(raw_df)))
    default_preview_rows = min(50, max_preview_rows)
    preview_rows = st.slider(
        "Rows to display",
        min_value=1,
        max_value=max_preview_rows,
        value=default_preview_rows,
    )
    st.dataframe(raw_df.head(preview_rows), use_container_width=True)

    full_table_placeholder = st.empty()
    show_full_table = st.checkbox("Show full ingested dataset (paginated)", value=False)
    if show_full_table:
        with full_table_placeholder.container():
            total_rows = len(raw_df)
            page_size = st.select_slider(
                "Rows per page",
                options=[100, 250, 500, 1000, 5000],
                value=500,
            )
            total_pages = max(1, (total_rows + page_size - 1) // page_size)

            page = st.number_input(
                "Page",
                min_value=1,
                max_value=total_pages,
                value=1,
                step=1,
            )
            start_idx = (page - 1) * page_size
            end_idx = min(start_idx + page_size, total_rows)

            st.caption(f"Showing rows {start_idx + 1:,} to {end_idx:,} of {total_rows:,}")
            st.dataframe(raw_df.iloc[start_idx:end_idx], use_container_width=True)
    else:
        full_table_placeholder.empty()

    raw_csv = raw_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download raw ingested dataset",
        data=raw_csv,
        file_name="btc_ingested_raw.csv",
        mime="text/csv",
    )

    try:
        timestamp_candidates = get_timestamp_candidates(raw_df)
        default_timestamp = detect_timestamp_column(raw_df)
    except IngestionError as exc:
        st.error(str(exc))
        return

    price_candidates = get_price_candidates(raw_df)
    if not price_candidates:
        st.error(
            "No numeric price columns were detected. "
            "Please provide a dataset that includes Close/Open/High/Low or a numeric price column."
        )
        return

    default_price = detect_default_price_column(price_candidates) or price_candidates[0]

    st.subheader("Data Configuration")
    st.caption("Detected columns can be overridden before preparation.")

    left_col, right_col = st.columns(2)

    with left_col:
        timestamp_column = st.selectbox(
            "Timestamp column",
            options=timestamp_candidates,
            index=timestamp_candidates.index(default_timestamp),
        )

    with right_col:
        price_column = st.selectbox(
            "Price column",
            options=price_candidates,
            index=price_candidates.index(default_price),
            help="Recommended: Close, Open, High, or Low.",
        )

    fill_missing_days = st.checkbox(
        "Fill missing trading days (daily data only)",
        value=True,
        help="If gaps exist in daily data, rows are inserted and price is time-interpolated.",
    )

    if st.button("Validate and Prepare Dataset", type="primary"):
        try:
            result = prepare_btc_history(
                raw_df,
                timestamp_column=timestamp_column,
                price_column=price_column,
                fill_missing_days=fill_missing_days,
            )
        except IngestionError as exc:
            st.error(str(exc))
            return

        metric_a, metric_b, metric_c, metric_d = st.columns(4)
        metric_a.metric("Rows (cleaned)", f"{len(result.data):,}")
        metric_b.metric("Dropped rows", f"{result.dropped_rows:,}")
        metric_c.metric("Duplicates removed", f"{result.duplicate_timestamps_removed:,}")
        metric_d.metric("Missing days filled", f"{result.inserted_missing_days:,}")

        if result.warnings:
            st.subheader("Validation Notes")
            for warning in result.warnings:
                st.warning(warning)
        else:
            st.success("Validation checks passed without warnings.")

        st.subheader("Prepared BTC Dataset")
        st.dataframe(result.data.head(40), use_container_width=True)

        start_date = result.data["timestamp"].min().date()
        end_date = result.data["timestamp"].max().date()
        st.caption(f"Prepared date range: {start_date} to {end_date}")

        prepared_csv = result.data.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download prepared dataset",
            data=prepared_csv,
            file_name="btc_prepared_for_forecasting.csv",
            mime="text/csv",
        )

        st.session_state["prepared_btc_data"] = result.data
        st.info(
            "Prepared data is stored in session state as `prepared_btc_data` "
            "for future forecasting modules."
        )


if __name__ == "__main__":
    main()
