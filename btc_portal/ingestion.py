from __future__ import annotations

from dataclasses import dataclass
import io
from pathlib import Path
from typing import Any, Iterable, Literal, cast
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import urlopen

import pandas as pd

_TIMESTAMP_HINTS = ("date", "timestamp", "datetime", "time", "ts")
_PRICE_HINTS = ("close", "open", "high", "low", "price")
_PREFERRED_PRICE_ORDER = ("close", "open", "high", "low")


class IngestionError(ValueError):
    """Raised when an uploaded CSV cannot be parsed as BTC price history."""


@dataclass(slots=True)
class IngestionResult:
    data: pd.DataFrame
    warnings: list[str]
    dropped_rows: int
    duplicate_timestamps_removed: int
    inserted_missing_days: int
    was_sorted: bool


def read_btc_csv(uploaded_file: object) -> pd.DataFrame:
    """Load CSV bytes from a Streamlit upload object with encoding fallback."""
    file_obj = cast(Any, uploaded_file)

    try:
        dataframe = pd.read_csv(file_obj)
    except UnicodeDecodeError:
        if hasattr(file_obj, "seek"):
            file_obj.seek(0)
        dataframe = pd.read_csv(file_obj, encoding="latin1")
    except Exception as exc:  # pragma: no cover - surfaced in Streamlit UI
        raise IngestionError(f"Could not read CSV file: {exc}") from exc

    if dataframe.empty:
        raise IngestionError("CSV file is empty.")

    return dataframe


def read_btc_csv_from_link(dataset_link: str, kaggle_csv_filename: str | None = None) -> pd.DataFrame:
    """Load BTC CSV data from a direct URL or a Kaggle dataset URL."""
    cleaned_link = dataset_link.strip()
    if not cleaned_link:
        raise IngestionError("Dataset link is empty.")

    kaggle_slug = _parse_kaggle_dataset_slug(cleaned_link)
    if kaggle_slug:
        return _read_kaggle_dataset_csv(kaggle_slug, kaggle_csv_filename)

    return _read_remote_csv(cleaned_link)


def get_timestamp_candidates(dataframe: pd.DataFrame, min_valid_ratio: float = 0.60) -> list[str]:
    ranked_columns: list[tuple[str, float]] = []

    for column in dataframe.columns:
        parsed = _parse_datetime_series(dataframe[column])
        valid_ratio = parsed.notna().mean()
        if valid_ratio < min_valid_ratio:
            continue

        score = valid_ratio
        lower_name = str(column).lower()
        if any(hint in lower_name for hint in _TIMESTAMP_HINTS):
            score += 0.25

        ranked_columns.append((column, score))

    ranked_columns.sort(key=lambda item: item[1], reverse=True)
    return [column for column, _ in ranked_columns]


def detect_timestamp_column(dataframe: pd.DataFrame) -> str:
    candidates = get_timestamp_candidates(dataframe)
    if not candidates:
        raise IngestionError(
            "No timestamp column detected. Use a Kaggle-style CSV that includes Date or Timestamp."
        )
    return candidates[0]


def get_price_candidates(dataframe: pd.DataFrame, min_valid_ratio: float = 0.60) -> list[str]:
    numeric_candidates: list[tuple[str, float, float]] = []

    for column in dataframe.columns:
        numeric_values = _parse_price_series(dataframe[column])
        valid_ratio = numeric_values.notna().mean()
        if valid_ratio < min_valid_ratio:
            continue

        lower_name = str(column).lower()
        hint_score = 1.0 if any(token in lower_name for token in _PRICE_HINTS) else 0.0
        preference_score = _preferred_price_rank_score(lower_name)
        total_score = preference_score + hint_score + valid_ratio
        numeric_candidates.append((column, total_score, hint_score))

    if not numeric_candidates:
        return []

    numeric_candidates.sort(key=lambda item: item[1], reverse=True)
    has_named_price_columns = any(item[2] > 0 for item in numeric_candidates)
    if has_named_price_columns:
        ordered = [item[0] for item in numeric_candidates if item[2] > 0]
        fallback = [item[0] for item in numeric_candidates if item[2] == 0]
        return ordered + fallback

    return [item[0] for item in numeric_candidates]


def detect_default_price_column(candidates: Iterable[str]) -> str | None:
    candidate_list = list(candidates)
    if not candidate_list:
        return None

    lowered_pairs = [(column.lower(), column) for column in candidate_list]
    for preferred in _PREFERRED_PRICE_ORDER:
        for lowered, original in lowered_pairs:
            if preferred == lowered or preferred in lowered:
                return original

    return candidate_list[0]


def prepare_btc_history(
    dataframe: pd.DataFrame,
    timestamp_column: str,
    price_column: str,
    *,
    fill_missing_days: bool = True,
) -> IngestionResult:
    warnings: list[str] = []

    timestamps = _parse_datetime_series(dataframe[timestamp_column])
    prices = _parse_price_series(dataframe[price_column])

    cleaned = pd.DataFrame({"timestamp": timestamps, "price_usd": prices})
    invalid_mask = cleaned["timestamp"].isna() | cleaned["price_usd"].isna()
    dropped_rows = int(invalid_mask.sum())

    if dropped_rows:
        warnings.append(f"Dropped {dropped_rows} rows with invalid timestamp or price values.")

    cleaned = cleaned.loc[~invalid_mask].copy()
    if cleaned.empty:
        raise IngestionError("No valid rows remain after parsing timestamp and price columns.")

    if cleaned["timestamp"].dt.tz is not None:
        cleaned["timestamp"] = cleaned["timestamp"].dt.tz_convert("UTC").dt.tz_localize(None)

    was_sorted = bool(cleaned["timestamp"].is_monotonic_increasing)
    if not was_sorted:
        warnings.append("Input data was not chronological. Rows were sorted by timestamp.")

    cleaned.sort_values("timestamp", inplace=True, kind="stable")

    duplicate_count = int(cleaned.duplicated(subset="timestamp").sum())
    if duplicate_count:
        warnings.append(f"Removed {duplicate_count} duplicate timestamps (kept latest row).")
        cleaned = cleaned.drop_duplicates(subset="timestamp", keep="last")

    cleaned["is_imputed"] = False
    inserted_missing_days = 0

    if fill_missing_days and len(cleaned) >= 3:
        median_step = cleaned["timestamp"].diff().median()
        if isinstance(median_step, pd.Timedelta) and median_step >= pd.Timedelta(hours=18):
            daily = (
                cleaned.assign(_day=cleaned["timestamp"].dt.normalize())
                .drop_duplicates(subset="_day", keep="last")
                .set_index("_day")
            )

            daily_index = pd.DatetimeIndex(daily.index)
            full_days = pd.date_range(daily_index.min(), daily_index.max(), freq="D")
            missing_days = full_days.difference(daily_index)
            inserted_missing_days = int(len(missing_days))

            if inserted_missing_days:
                daily = daily.reindex(full_days)
                daily["is_imputed"] = daily["timestamp"].isna()
                daily["timestamp"] = daily.index
                daily["price_usd"] = daily["price_usd"].interpolate(method="time").ffill().bfill()
                warnings.append(
                    f"Inserted {inserted_missing_days} missing trading days and interpolated price values."
                )

            cleaned = daily.reset_index(drop=True)[["timestamp", "price_usd", "is_imputed"]]
        else:
            warnings.append("Intraday frequency detected; missing-day filling was skipped.")

    cleaned.reset_index(drop=True, inplace=True)

    return IngestionResult(
        data=cleaned,
        warnings=warnings,
        dropped_rows=dropped_rows,
        duplicate_timestamps_removed=duplicate_count,
        inserted_missing_days=inserted_missing_days,
        was_sorted=was_sorted,
    )


def _preferred_price_rank_score(column_name: str) -> float:
    for rank, preferred in enumerate(_PREFERRED_PRICE_ORDER):
        if preferred == column_name or preferred in column_name:
            return 2.0 - (rank * 0.25)
    return 0.0


def _parse_datetime_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_datetime(series, errors="coerce", utc=True)

    if pd.api.types.is_numeric_dtype(series):
        numeric_series = pd.to_numeric(series, errors="coerce")
        return _parse_numeric_epoch(numeric_series)

    string_series = series.astype(str).str.strip().mask(series.isna())
    parsed = pd.to_datetime(string_series, errors="coerce", utc=True)

    numeric_fallback = pd.to_numeric(string_series, errors="coerce")
    if numeric_fallback.notna().mean() >= 0.60:
        parsed_fallback = _parse_numeric_epoch(numeric_fallback)
        if parsed_fallback.notna().mean() > parsed.notna().mean():
            return parsed_fallback

    return parsed


def _parse_numeric_epoch(numeric_series: pd.Series) -> pd.Series:
    non_null = numeric_series.dropna()
    if non_null.empty:
        return _empty_datetime_series(numeric_series.index)

    median_abs = float(non_null.abs().median())
    candidate_units: list[Literal["s", "ms", "us", "ns"]] = []

    if 1e9 <= median_abs <= 1e10:
        candidate_units.append("s")
    if 1e12 <= median_abs <= 1e13:
        candidate_units.append("ms")
    if 1e15 <= median_abs <= 1e16:
        candidate_units.append("us")
    if 1e18 <= median_abs <= 1e19:
        candidate_units.append("ns")

    if not candidate_units:
        return _empty_datetime_series(numeric_series.index)

    best = _empty_datetime_series(numeric_series.index)
    best_ratio = -1.0

    for unit in candidate_units:
        parsed = pd.to_datetime(cast(Any, numeric_series), errors="coerce", unit=unit, utc=True)
        ratio = parsed.notna().mean()
        if ratio > best_ratio:
            best = parsed
            best_ratio = ratio

    return best


def _empty_datetime_series(index: pd.Index) -> pd.Series:
    return pd.Series(pd.NaT, index=index, dtype="datetime64[ns, UTC]")


def _parse_price_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")

    cleaned = (
        series.astype(str)
        .str.replace(r"[\$,]", "", regex=True)
        .str.replace(r"\s+", "", regex=True)
    )
    return pd.to_numeric(cleaned, errors="coerce")


def _read_remote_csv(dataset_link: str) -> pd.DataFrame:
    normalized_link = dataset_link
    if "://" not in normalized_link:
        normalized_link = f"https://{normalized_link}"

    try:
        with urlopen(normalized_link, timeout=45) as response:
            payload = response.read()
            content_type = response.headers.get("Content-Type", "")
    except URLError as exc:
        raise IngestionError(f"Could not download dataset link: {exc}") from exc
    except Exception as exc:
        raise IngestionError(f"Failed to fetch dataset link: {exc}") from exc

    if not payload:
        raise IngestionError("Dataset link returned an empty response.")

    if "text/html" in content_type.lower():
        raise IngestionError(
            "Dataset link returned HTML instead of CSV. "
            "Use a direct CSV URL or a Kaggle dataset URL."
        )

    return _read_csv_bytes(payload, source_label=dataset_link)


def _read_csv_bytes(payload: bytes, *, source_label: str) -> pd.DataFrame:
    buffer = io.BytesIO(payload)
    try:
        dataframe = pd.read_csv(buffer)
    except UnicodeDecodeError:
        buffer.seek(0)
        dataframe = pd.read_csv(buffer, encoding="latin1")
    except Exception as exc:
        raise IngestionError(f"Could not parse CSV from {source_label}: {exc}") from exc

    if dataframe.empty:
        raise IngestionError(f"CSV from {source_label} is empty.")

    return dataframe


def _parse_kaggle_dataset_slug(dataset_link: str) -> tuple[str, str] | None:
    parsed = urlparse(dataset_link if "://" in dataset_link else f"https://{dataset_link}")
    host = parsed.netloc.lower()
    if "kaggle.com" not in host:
        return None

    path_parts = [part for part in parsed.path.split("/") if part]
    if len(path_parts) < 3 or path_parts[0].lower() != "datasets":
        return None

    owner = path_parts[1].strip()
    dataset = path_parts[2].strip()
    if not owner or not dataset:
        return None

    return owner, dataset


def _read_kaggle_dataset_csv(kaggle_slug: tuple[str, str], kaggle_csv_filename: str | None) -> pd.DataFrame:
    owner, dataset = kaggle_slug

    try:
        import kagglehub
    except ImportError as exc:
        raise IngestionError(
            "Kaggle links require kagglehub. Install it with: pip install kagglehub"
        ) from exc

    dataset_ref = f"{owner}/{dataset}"
    try:
        dataset_dir = Path(kagglehub.dataset_download(dataset_ref))
    except Exception as exc:
        raise IngestionError(
            "Could not download Kaggle dataset. "
            "Make sure Kaggle API credentials are configured on this machine."
        ) from exc

    csv_files = sorted(dataset_dir.rglob("*.csv"))
    if not csv_files:
        raise IngestionError("Downloaded Kaggle dataset has no CSV files.")

    selected_file = _select_kaggle_csv_file(csv_files, kaggle_csv_filename)

    try:
        dataframe = pd.read_csv(selected_file)
    except UnicodeDecodeError:
        dataframe = pd.read_csv(selected_file, encoding="latin1")
    except Exception as exc:
        raise IngestionError(f"Could not parse Kaggle CSV '{selected_file.name}': {exc}") from exc

    if dataframe.empty:
        raise IngestionError(f"Kaggle CSV '{selected_file.name}' is empty.")

    return dataframe


def _select_kaggle_csv_file(csv_files: list[Path], kaggle_csv_filename: str | None) -> Path:
    if kaggle_csv_filename:
        wanted = kaggle_csv_filename.strip().replace("\\", "/").lower()
        for csv_file in csv_files:
            file_name = csv_file.name.lower()
            file_path = str(csv_file).replace("\\", "/").lower()
            if file_name == wanted or file_path.endswith(wanted):
                return csv_file

        available = ", ".join(path.name for path in csv_files[:8])
        raise IngestionError(
            f"Kaggle CSV '{kaggle_csv_filename}' was not found. Available files: {available}"
        )

    if len(csv_files) == 1:
        return csv_files[0]

    ranked = sorted(
        csv_files,
        key=lambda item: (_score_kaggle_csv_name(item.name), _safe_file_size(item)),
        reverse=True,
    )
    best_match = ranked[0]

    if _score_kaggle_csv_name(best_match.name) == 0:
        available = ", ".join(path.name for path in csv_files[:8])
        raise IngestionError(
            "Kaggle dataset contains multiple CSV files. "
            "Please provide the file name in the optional input field. "
            f"Available files: {available}"
        )

    return best_match


def _score_kaggle_csv_name(filename: str) -> int:
    lowered = filename.lower()
    score = 0

    if "btc" in lowered or "bitcoin" in lowered:
        score += 4
    if "historical" in lowered or "history" in lowered:
        score += 2
    if "price" in lowered or "ohlc" in lowered or "market" in lowered:
        score += 1

    return score


def _safe_file_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0
