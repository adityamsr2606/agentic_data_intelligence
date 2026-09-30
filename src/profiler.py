from __future__ import annotations

import ipaddress
import json
import re
from typing import Any

import numpy as np
import pandas as pd

from src.config import settings
from src.models import (
    ColumnProfile,
    DataAsset,
    DataProfile,
    DocumentProfile,
    PatternResult,
    TableProfile,
)


# -------------------------------------------------------------------
# COMMON STRUCTURAL PATTERNS
# -------------------------------------------------------------------

EMAIL_PATTERN = re.compile(
    r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$"
)

URL_PATTERN = re.compile(
    r"^https?://[^\s]+$",
    re.IGNORECASE,
)

UUID_PATTERN = re.compile(
    r"^[0-9a-fA-F]{8}-"
    r"[0-9a-fA-F]{4}-"
    r"[1-5][0-9a-fA-F]{3}-"
    r"[89abAB][0-9a-fA-F]{3}-"
    r"[0-9a-fA-F]{12}$"
)

INTEGER_PATTERN = re.compile(
    r"^[+-]?\d+$"
)

DECIMAL_PATTERN = re.compile(
    r"^[+-]?(?:\d+(?:\.\d+)?|\.\d+)$"
)

DATE_HINT_PATTERN = re.compile(
    r"[-/:]|"
    r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)"
    r"[a-z]*\b",
    re.IGNORECASE,
)


# -------------------------------------------------------------------
# SAFE VALUE HELPERS
# -------------------------------------------------------------------

def _safe_value(value: Any) -> Any:
    """
    Convert pandas, NumPy and complex values into values that can
    safely be stored inside profiling results.
    """

    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    if isinstance(value, (dict, list, tuple, set)):
        return json.dumps(
            value,
            default=str,
            ensure_ascii=False,
            sort_keys=True,
        )

    return value


def _safe_string(value: Any) -> str:
    """
    Convert a value into a stable string representation.

    This is useful for nested objects such as dictionaries and lists,
    which cannot always be counted directly by pandas.
    """

    if isinstance(value, (dict, list, tuple, set)):
        return json.dumps(
            value,
            default=str,
            ensure_ascii=False,
            sort_keys=True,
        )

    return str(value)


def _clean_non_null(
    series: pd.Series,
) -> pd.Series:
    """
    Return only non-null values without modifying the source column.
    """

    return series[
        ~series.isna()
    ]


# -------------------------------------------------------------------
# BASIC COLUMN INFORMATION
# -------------------------------------------------------------------

def _sample_values(
    series: pd.Series,
    limit: int = 10,
) -> list[Any]:
    """
    Return a small sample of non-null values from a column.
    """

    non_null = _clean_non_null(
        series
    )

    if non_null.empty:
        return []

    samples: list[Any] = []

    for value in non_null.head(limit):
        samples.append(
            _safe_value(value)
        )

    return samples


def _distinct_count(
    series: pd.Series,
) -> int:
    """
    Count distinct non-null values safely.
    """

    non_null = _clean_non_null(
        series
    )

    if non_null.empty:
        return 0

    safe_series = non_null.map(
        _safe_string
    )

    return int(
        safe_series.nunique(
            dropna=True
        )
    )


def _top_values(
    series: pd.Series,
    limit: int = 5,
) -> dict[str, int]:
    """
    Return the most frequently occurring values.
    """

    non_null = _clean_non_null(
        series
    )

    if non_null.empty:
        return {}

    safe_series = non_null.map(
        _safe_string
    )

    counts = (
        safe_series
        .value_counts()
        .head(limit)
    )

    return {
        str(value): int(count)
        for value, count in counts.items()
    }


# -------------------------------------------------------------------
# GENERIC TYPE INFERENCE
# -------------------------------------------------------------------

def _match_ratio(
    values: pd.Series,
    matcher,
) -> float:
    """
    Calculate the proportion of values matching a condition.
    """

    if values.empty:
        return 0.0

    matched = 0

    for value in values:
        try:
            if matcher(value):
                matched += 1
        except Exception:
            continue

    return matched / len(values)


def _looks_like_boolean(
    values: pd.Series,
) -> float:
    accepted_values = {
        "true",
        "false",
        "yes",
        "no",
        "y",
        "n",
    }

    return _match_ratio(
        values,
        lambda value: (
            str(value)
            .strip()
            .lower()
            in accepted_values
        ),
    )


def _looks_like_integer(
    values: pd.Series,
) -> float:
    return _match_ratio(
        values,
        lambda value: bool(
            INTEGER_PATTERN.fullmatch(
                str(value).strip()
            )
        ),
    )


def _looks_like_number(
    values: pd.Series,
) -> float:
    return _match_ratio(
        values,
        lambda value: bool(
            DECIMAL_PATTERN.fullmatch(
                str(value).strip()
            )
        ),
    )


def _looks_like_datetime(
    values: pd.Series,
) -> float:
    """
    Detect date-like data based on values rather than column names.

    A date hint is required first so that ordinary integer IDs are not
    accidentally interpreted as timestamps.
    """

    if values.empty:
        return 0.0

    strings = (
        values
        .astype(str)
        .str.strip()
    )

    date_candidates = strings[
        strings.str.contains(
            DATE_HINT_PATTERN,
            regex=True,
            na=False,
        )
    ]

    if len(date_candidates) < (
        len(strings) * 0.5
    ):
        return 0.0

    parsed = pd.to_datetime(
        strings,
        errors="coerce",
        format="mixed",
    )

    return float(
        parsed.notna().mean()
    )


def _infer_type(
    series: pd.Series,
) -> str:
    """
    Infer the physical type of a column using its values.
    """

    if series.empty:
        return "unknown"

    non_null = _clean_non_null(
        series
    )

    if non_null.empty:
        return "unknown"

    if pd.api.types.is_bool_dtype(
        series
    ):
        return "boolean"

    if pd.api.types.is_integer_dtype(
        series
    ):
        return "integer"

    if pd.api.types.is_float_dtype(
        series
    ):
        return "float"

    if pd.api.types.is_datetime64_any_dtype(
        series
    ):
        return "datetime"

    if pd.api.types.is_timedelta64_dtype(
        series
    ):
        return "timedelta"

    sample = non_null.head(
        min(
            500,
            len(non_null),
        )
    )

    complex_count = sum(
        isinstance(
            value,
            (dict, list, tuple, set),
        )
        for value in sample
    )

    if (
        complex_count
        / len(sample)
    ) >= 0.8:
        return "complex"

    if _looks_like_boolean(
        sample
    ) >= 0.95:
        return "boolean"

    if _looks_like_integer(
        sample
    ) >= 0.95:
        return "integer"

    if _looks_like_number(
        sample
    ) >= 0.95:
        return "float"

    if _looks_like_datetime(
        sample
    ) >= 0.90:
        return "datetime"

    return "string"


# -------------------------------------------------------------------
# NUMERIC PROFILING
# -------------------------------------------------------------------

def _numeric_series(
    series: pd.Series,
) -> pd.Series:
    """
    Convert values into numeric values where possible.
    """

    return pd.to_numeric(
        series,
        errors="coerce",
    )


def _numeric_statistics(
    series: pd.Series,
) -> dict[str, Any]:
    """
    Calculate descriptive statistics and two independent forms of
    outlier evidence: IQR and Median Absolute Deviation.

    Outliers are reported as evidence only. They are not automatically
    treated as bad data.
    """

    numeric = (
        _numeric_series(series)
        .dropna()
    )

    if numeric.empty:
        return {}

    minimum = float(
        numeric.min()
    )

    maximum = float(
        numeric.max()
    )

    mean = float(
        numeric.mean()
    )

    median = float(
        numeric.median()
    )

    standard_deviation = (
        numeric.std()
    )

    if pd.isna(
        standard_deviation
    ):
        standard_deviation = 0.0

    first_quartile = float(
        numeric.quantile(
            0.25
        )
    )

    third_quartile = float(
        numeric.quantile(
            0.75
        )
    )

    iqr = (
        third_quartile
        - first_quartile
    )

    iqr_lower_bound = (
        first_quartile
        - 1.5 * iqr
    )

    iqr_upper_bound = (
        third_quartile
        + 1.5 * iqr
    )

    iqr_outliers = numeric[
        (numeric < iqr_lower_bound)
        | (numeric > iqr_upper_bound)
    ]

    # Median Absolute Deviation gives us another robust
    # signal, especially when the dataset is small.
    absolute_deviation = (
        numeric - median
    ).abs()

    mad = float(
        absolute_deviation.median()
    )

    if mad > 0:
        modified_z_scores = (
            0.6745
            * absolute_deviation
            / mad
        )

        mad_outliers = numeric[
            modified_z_scores > 3.5
        ]

    else:
        mad_outliers = numeric.iloc[
            0:0
        ]

    numeric_count = len(
        numeric
    )

    return {
        "minimum": minimum,
        "maximum": maximum,
        "mean": mean,
        "median": median,
        "standard_deviation": float(
            standard_deviation
        ),

        "first_quartile": (
            first_quartile
        ),
        "third_quartile": (
            third_quartile
        ),
        "interquartile_range": float(
            iqr
        ),

        "negative_count": int(
            (numeric < 0).sum()
        ),
        "zero_count": int(
            (numeric == 0).sum()
        ),
        "positive_count": int(
            (numeric > 0).sum()
        ),

        "iqr_lower_bound": float(
            iqr_lower_bound
        ),
        "iqr_upper_bound": float(
            iqr_upper_bound
        ),
        "iqr_outlier_count": int(
            len(iqr_outliers)
        ),
        "iqr_outlier_percentage": round(
            (
                len(iqr_outliers)
                / numeric_count
            )
            * 100,
            4,
        ),

        "median_absolute_deviation": (
            mad
        ),
        "mad_outlier_count": int(
            len(mad_outliers)
        ),
        "mad_outlier_percentage": round(
            (
                len(mad_outliers)
                / numeric_count
            )
            * 100,
            4,
        ),
    }


# -------------------------------------------------------------------
# TEXT PROFILING
# -------------------------------------------------------------------

def _text_statistics(
    series: pd.Series,
) -> dict[str, Any]:
    """
    Calculate general string statistics.
    """

    non_null = _clean_non_null(
        series
    )

    if non_null.empty:
        return {}

    strings = non_null.map(
        _safe_string
    )

    stripped = (
        strings.str.strip()
    )

    lengths = (
        strings.str.len()
    )

    return {
        "minimum_length": int(
            lengths.min()
        ),
        "maximum_length": int(
            lengths.max()
        ),
        "average_length": round(
            float(
                lengths.mean()
            ),
            4,
        ),
        "empty_string_count": int(
            (stripped == "").sum()
        ),
        "whitespace_only_count": int(
            (
                (strings != "")
                & (stripped == "")
            ).sum()
        ),
    }


# -------------------------------------------------------------------
# DATETIME PROFILING
# -------------------------------------------------------------------

def _datetime_statistics(
    series: pd.Series,
) -> dict[str, Any]:
    """
    Calculate basic datetime statistics.
    """

    parsed = pd.to_datetime(
        series,
        errors="coerce",
        format="mixed",
    )

    valid_dates = (
        parsed.dropna()
    )

    if valid_dates.empty:
        return {}

    invalid_count = int(
        parsed.isna().sum()
        - series.isna().sum()
    )

    return {
        "earliest": (
            valid_dates
            .min()
            .isoformat()
        ),
        "latest": (
            valid_dates
            .max()
            .isoformat()
        ),
        "valid_datetime_count": int(
            len(valid_dates)
        ),
        "invalid_datetime_count": max(
            invalid_count,
            0,
        ),
    }


# -------------------------------------------------------------------
# STRUCTURAL PATTERN ANALYSIS
# -------------------------------------------------------------------

def _is_ip_address(
    value: str,
) -> bool:
    try:
        ipaddress.ip_address(
            value.strip()
        )

        return True

    except ValueError:
        return False


def _detect_patterns(
    series: pd.Series,
) -> list[PatternResult]:
    """
    Detect common structural patterns from values.

    Pattern detection only provides evidence. It does not create
    business rules.
    """

    non_null = _clean_non_null(
        series
    )

    if non_null.empty:
        return []

    sample = non_null.head(
        min(
            1000,
            len(non_null),
        )
    )

    strings = (
        sample
        .map(_safe_string)
        .str.strip()
    )

    checks = {
        "email": (
            lambda value: bool(
                EMAIL_PATTERN.fullmatch(
                    value
                )
            )
        ),

        "url": (
            lambda value: bool(
                URL_PATTERN.fullmatch(
                    value
                )
            )
        ),

        "uuid": (
            lambda value: bool(
                UUID_PATTERN.fullmatch(
                    value
                )
            )
        ),

        "integer_string": (
            lambda value: bool(
                INTEGER_PATTERN.fullmatch(
                    value
                )
            )
        ),

        "decimal_string": (
            lambda value: bool(
                DECIMAL_PATTERN.fullmatch(
                    value
                )
            )
        ),

        "ip_address": (
            _is_ip_address
        ),
    }

    results: list[
        PatternResult
    ] = []

    for (
        pattern_name,
        matcher,
    ) in checks.items():

        matched = 0

        for value in strings:
            try:
                if matcher(value):
                    matched += 1
            except Exception:
                continue

        if matched == 0:
            continue

        results.append(
            PatternResult(
                pattern_name=(
                    pattern_name
                ),
                matched_values=(
                    matched
                ),
                checked_values=(
                    len(strings)
                ),
                match_percentage=round(
                    (
                        matched
                        / len(strings)
                    )
                    * 100,
                    4,
                ),
            )
        )

    return results


# -------------------------------------------------------------------
# COLUMN PROFILING
# -------------------------------------------------------------------

def _profile_column(
    column_name: str,
    series: pd.Series,
) -> ColumnProfile:
    """
    Build the complete profile of one column.
    """

    row_count = int(
        len(series)
    )

    null_count = int(
        series.isna().sum()
    )

    if row_count:
        null_percentage = (
            null_count
            / row_count
        ) * 100
    else:
        null_percentage = 0.0

    distinct_count = (
        _distinct_count(
            series
        )
    )

    non_null_count = (
        row_count
        - null_count
    )

    if non_null_count:
        distinct_percentage = (
            distinct_count
            / non_null_count
        ) * 100
    else:
        distinct_percentage = 0.0

    inferred_type = (
        _infer_type(
            series
        )
    )

    numeric_statistics = {}
    text_statistics = {}
    datetime_statistics = {}

    if inferred_type in {
        "integer",
        "float",
    }:
        numeric_statistics = (
            _numeric_statistics(
                series
            )
        )

    if inferred_type in {
        "string",
        "complex",
    }:
        text_statistics = (
            _text_statistics(
                series
            )
        )

    if inferred_type == "datetime":
        datetime_statistics = (
            _datetime_statistics(
                series
            )
        )

    return ColumnProfile(
        column_name=str(
            column_name
        ),

        pandas_type=str(
            series.dtype
        ),

        inferred_type=(
            inferred_type
        ),

        row_count=row_count,

        null_count=(
            null_count
        ),

        null_percentage=round(
            null_percentage,
            4,
        ),

        distinct_count=(
            distinct_count
        ),

        distinct_percentage=round(
            distinct_percentage,
            4,
        ),

        sample_values=(
            _sample_values(
                series,
                limit=10,
            )
        ),

        most_frequent_values=(
            _top_values(
                series,
                limit=5,
            )
        ),

        numeric_statistics=(
            numeric_statistics
        ),

        text_statistics=(
            text_statistics
        ),

        datetime_statistics=(
            datetime_statistics
        ),

        patterns=(
            _detect_patterns(
                series
            )
        ),
    )


# -------------------------------------------------------------------
# TABLE PROFILING
# -------------------------------------------------------------------

def _safe_duplicate_count(
    dataframe: pd.DataFrame,
) -> int:
    """
    Count duplicate records even if some cells contain complex objects.
    """

    if dataframe.empty:
        return 0

    safe_frame = (
        dataframe.copy()
    )

    for column in safe_frame.columns:
        safe_frame[column] = (
            safe_frame[
                column
            ].map(
                _safe_value
            )
        )

    return int(
        safe_frame
        .duplicated(
            keep="first"
        )
        .sum()
    )


def _profile_table(
    table_name: str,
    dataframe: pd.DataFrame,
) -> TableProfile:
    """
    Build a complete profile for one table.
    """

    row_count = int(
        len(dataframe)
    )

    column_count = int(
        len(
            dataframe.columns
        )
    )

    duplicate_count = (
        _safe_duplicate_count(
            dataframe
        )
    )

    if row_count:
        duplicate_percentage = (
            duplicate_count
            / row_count
        ) * 100
    else:
        duplicate_percentage = 0.0

    column_profiles: list[
        ColumnProfile
    ] = []

    for column in dataframe.columns:

        profile = (
            _profile_column(
                column_name=str(
                    column
                ),
                series=dataframe[
                    column
                ],
            )
        )

        column_profiles.append(
            profile
        )

    return TableProfile(
        table_name=(
            table_name
        ),

        row_count=(
            row_count
        ),

        column_count=(
            column_count
        ),

        duplicate_row_count=(
            duplicate_count
        ),

        duplicate_percentage=round(
            duplicate_percentage,
            4,
        ),

        columns=(
            column_profiles
        ),
    )


# -------------------------------------------------------------------
# DOCUMENT PROFILING
# -------------------------------------------------------------------

def _profile_document(
    document_name: str,
    text: str,
) -> DocumentProfile:
    """
    Build basic statistics for unstructured document text.
    """

    text = text or ""

    lines = (
        text.splitlines()
    )

    non_empty_lines = [
        line
        for line in lines
        if line.strip()
    ]

    words = re.findall(
        r"\S+",
        text,
    )

    preview_limit = min(
        settings.max_document_chars,
        1000,
    )

    preview = (
        text[
            :preview_limit
        ]
        .strip()
    )

    return DocumentProfile(
        document_name=(
            document_name
        ),

        character_count=(
            len(text)
        ),

        word_count=(
            len(words)
        ),

        line_count=(
            len(lines)
        ),

        non_empty_line_count=(
            len(
                non_empty_lines
            )
        ),

        preview=(
            preview
        ),
    )


# -------------------------------------------------------------------
# MAIN PROFILING ENTRY POINT
# -------------------------------------------------------------------

def profile_asset(
    asset: DataAsset,
) -> DataProfile:
    """
    Profile every structured table and unstructured document contained
    in a DataAsset.

    This is the main profiling function used by the rest of the system.
    """

    table_profiles: list[
        TableProfile
    ] = []

    for (
        table_name,
        dataframe,
    ) in asset.tables.items():

        table_profiles.append(
            _profile_table(
                table_name,
                dataframe,
            )
        )

    document_profiles: list[
        DocumentProfile
    ] = []

    for (
        document_name,
        text,
    ) in asset.documents.items():

        document_profiles.append(
            _profile_document(
                document_name,
                text,
            )
        )

    return DataProfile(
        source_name=(
            asset.source_path.name
        ),

        source_type=(
            asset.source_type
        ),

        table_count=(
            len(
                table_profiles
            )
        ),

        document_count=(
            len(
                document_profiles
            )
        ),

        tables=(
            table_profiles
        ),

        documents=(
            document_profiles
        ),
    )