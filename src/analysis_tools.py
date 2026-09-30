from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.config import settings
from src.models import (
    DataAsset,
    ToolRequest,
    ToolResult,
)


def _to_python(value: Any) -> Any:
    """
    Convert pandas and NumPy values into normal Python values.
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

    if isinstance(value, dict):
        return {
            str(key): _to_python(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            _to_python(item)
            for item in value
        ]

    return value


def _get_table(
    asset: DataAsset,
    table_name: str | None,
) -> pd.DataFrame:
    if not table_name:
        raise ValueError(
            "A table name is required for this tool."
        )

    if table_name not in asset.tables:
        raise ValueError(
            f"Table '{table_name}' was not found."
        )

    return asset.tables[
        table_name
    ]


def _get_column(
    dataframe: pd.DataFrame,
    column_name: str | None,
) -> pd.Series:
    if not column_name:
        raise ValueError(
            "A column name is required for this tool."
        )

    if column_name not in dataframe.columns:
        raise ValueError(
            f"Column '{column_name}' was not found."
        )

    return dataframe[
        column_name
    ]


def get_tool_catalog() -> list[dict[str, Any]]:
    """
    Describe the tools that the reasoning model is allowed to request.
    """

    return [
        {
            "name": "list_tables",
            "description": (
                "List all available structured tables and documents "
                "inside the loaded data source."
            ),
        },
        {
            "name": "table_overview",
            "description": (
                "Get row count, column count, datatypes and basic "
                "table-level information."
            ),
            "requires": [
                "table_name",
            ],
        },
        {
            "name": "sample_rows",
            "description": (
                "Return a small deterministic sample of records from "
                "a table for semantic understanding."
            ),
            "requires": [
                "table_name",
            ],
            "parameters": {
                "limit": "optional integer",
            },
        },
        {
            "name": "column_distribution",
            "description": (
                "Inspect the most frequent values and their frequency "
                "for a specific column."
            ),
            "requires": [
                "table_name",
                "column_name",
            ],
            "parameters": {
                "limit": "optional integer",
            },
        },
        {
            "name": "missingness_analysis",
            "description": (
                "Calculate null and empty-value statistics for every "
                "column in a table."
            ),
            "requires": [
                "table_name",
            ],
        },
        {
            "name": "duplicate_analysis",
            "description": (
                "Investigate duplicate records. A subset of columns can "
                "optionally be provided through parameters.columns."
            ),
            "requires": [
                "table_name",
            ],
            "parameters": {
                "columns": (
                    "optional list of column names"
                ),
            },
        },
        {
            "name": "uniqueness_analysis",
            "description": (
                "Measure whether one column or a combination of columns "
                "behaves like a unique identifier."
            ),
            "requires": [
                "table_name",
            ],
            "parameters": {
                "columns": (
                    "list of columns, or use column_name"
                ),
            },
        },
        {
            "name": "correlation_analysis",
            "description": (
                "Calculate numeric correlation either between two "
                "columns or across all numeric columns."
            ),
            "requires": [
                "table_name",
            ],
            "parameters": {
                "method": (
                    "pearson, spearman or kendall"
                ),
            },
        },
        {
            "name": "outlier_analysis",
            "description": (
                "Investigate numeric outliers using both IQR and "
                "Median Absolute Deviation."
            ),
            "requires": [
                "table_name",
                "column_name",
            ],
        },
        {
            "name": "column_pair_analysis",
            "description": (
                "Compare two columns for equality, null overlap, "
                "and numeric correlation when possible."
            ),
            "requires": [
                "table_name",
                "column_name",
                "second_column_name",
            ],
        },
        {
            "name": "date_range_analysis",
            "description": (
                "Parse a column as datetime and inspect its date range, "
                "invalid values and future values."
            ),
            "requires": [
                "table_name",
                "column_name",
            ],
        },
        {
            "name": "document_excerpt",
            "description": (
                "Read a controlled portion of an unstructured document "
                "for semantic analysis."
            ),
            "parameters": {
                "document_name": "required",
                "start": "optional integer",
                "length": "optional integer",
            },
        },
    ]


def _list_tables(
    asset: DataAsset,
) -> dict[str, Any]:
    tables = []

    for table_name, dataframe in asset.tables.items():
        tables.append(
            {
                "name": table_name,
                "rows": int(
                    len(dataframe)
                ),
                "columns": int(
                    len(
                        dataframe.columns
                    )
                ),
            }
        )

    documents = []

    for document_name, text in asset.documents.items():
        documents.append(
            {
                "name": document_name,
                "characters": len(
                    text or ""
                ),
            }
        )

    return {
        "tables": tables,
        "documents": documents,
    }


def _table_overview(
    asset: DataAsset,
    request: ToolRequest,
) -> dict[str, Any]:
    dataframe = _get_table(
        asset,
        request.table_name,
    )

    duplicate_count = int(
        dataframe.astype(
            str
        ).duplicated().sum()
    )

    columns = []

    for column in dataframe.columns:
        series = dataframe[
            column
        ]

        columns.append(
            {
                "name": str(
                    column
                ),
                "dtype": str(
                    series.dtype
                ),
                "null_count": int(
                    series.isna().sum()
                ),
                "non_null_count": int(
                    series.notna().sum()
                ),
            }
        )

    return {
        "row_count": int(
            len(dataframe)
        ),
        "column_count": int(
            len(dataframe.columns)
        ),
        "duplicate_rows": (
            duplicate_count
        ),
        "columns": columns,
    }


def _sample_rows(
    asset: DataAsset,
    request: ToolRequest,
) -> dict[str, Any]:
    dataframe = _get_table(
        asset,
        request.table_name,
    )

    requested_limit = request.parameters.get(
        "limit",
        settings.max_sample_rows,
    )

    try:
        requested_limit = int(
            requested_limit
        )
    except (TypeError, ValueError):
        requested_limit = (
            settings.max_sample_rows
        )

    limit = max(
        1,
        min(
            requested_limit,
            settings.max_sample_rows,
        ),
    )

    if dataframe.empty:
        return {
            "rows": [],
            "returned_rows": 0,
        }

    if len(dataframe) <= limit:
        sample = dataframe.copy()

    else:
        sample = dataframe.sample(
            n=limit,
            random_state=42,
        ).sort_index()

    records = []

    for _, row in sample.iterrows():
        record = {}

        for column in dataframe.columns:
            record[str(column)] = (
                _to_python(
                    row[column]
                )
            )

        records.append(record)

    return {
        "returned_rows": len(
            records
        ),
        "rows": records,
    }


def _column_distribution(
    asset: DataAsset,
    request: ToolRequest,
) -> dict[str, Any]:
    dataframe = _get_table(
        asset,
        request.table_name,
    )

    series = _get_column(
        dataframe,
        request.column_name,
    )

    limit = request.parameters.get(
        "limit",
        10,
    )

    try:
        limit = int(limit)
    except (TypeError, ValueError):
        limit = 10

    limit = max(
        1,
        min(
            limit,
            50,
        ),
    )

    safe_series = (
        series
        .dropna()
        .map(str)
    )

    counts = (
        safe_series
        .value_counts()
        .head(limit)
    )

    values = []

    for value, count in counts.items():
        values.append(
            {
                "value": value,
                "count": int(
                    count
                ),
                "percentage": round(
                    (
                        count
                        / len(series)
                    )
                    * 100,
                    4,
                )
                if len(series)
                else 0.0,
            }
        )

    return {
        "column": (
            request.column_name
        ),
        "row_count": int(
            len(series)
        ),
        "null_count": int(
            series.isna().sum()
        ),
        "distinct_count": int(
            safe_series.nunique()
        ),
        "top_values": values,
    }


def _missingness_analysis(
    asset: DataAsset,
    request: ToolRequest,
) -> dict[str, Any]:
    dataframe = _get_table(
        asset,
        request.table_name,
    )

    row_count = len(
        dataframe
    )

    results = []

    for column in dataframe.columns:
        series = dataframe[
            column
        ]

        null_count = int(
            series.isna().sum()
        )

        empty_count = 0

        if (
            pd.api.types.is_object_dtype(
                series
            )
            or pd.api.types.is_string_dtype(
                series
            )
        ):
            non_null = (
                series.dropna()
                .astype(str)
            )

            empty_count = int(
                (
                    non_null.str.strip()
                    == ""
                ).sum()
            )

        missing_count = (
            null_count
            + empty_count
        )

        results.append(
            {
                "column": str(
                    column
                ),
                "null_count": (
                    null_count
                ),
                "empty_string_count": (
                    empty_count
                ),
                "total_missing": (
                    missing_count
                ),
                "missing_percentage": round(
                    (
                        missing_count
                        / row_count
                    )
                    * 100,
                    4,
                )
                if row_count
                else 0.0,
            }
        )

    results.sort(
        key=lambda item: (
            item[
                "missing_percentage"
            ]
        ),
        reverse=True,
    )

    return {
        "row_count": int(
            row_count
        ),
        "columns": results,
    }


def _duplicate_analysis(
    asset: DataAsset,
    request: ToolRequest,
) -> dict[str, Any]:
    dataframe = _get_table(
        asset,
        request.table_name,
    )

    requested_columns = (
        request.parameters.get(
            "columns"
        )
    )

    subset = None

    if requested_columns:
        if not isinstance(
            requested_columns,
            list,
        ):
            raise ValueError(
                "parameters.columns must be a list."
            )

        missing_columns = [
            column
            for column in requested_columns
            if column
            not in dataframe.columns
        ]

        if missing_columns:
            raise ValueError(
                "Unknown duplicate-analysis columns: "
                + ", ".join(
                    missing_columns
                )
            )

        subset = requested_columns

    safe_frame = dataframe.copy()

    columns_to_convert = (
        subset
        if subset
        else list(
            safe_frame.columns
        )
    )

    for column in columns_to_convert:
        safe_frame[column] = (
            safe_frame[
                column
            ].map(str)
        )

    duplicate_mask = (
        safe_frame.duplicated(
            subset=subset,
            keep=False,
        )
    )

    duplicate_rows = (
        dataframe[
            duplicate_mask
        ]
    )

    sample_records = []

    for index, row in (
        duplicate_rows
        .head(10)
        .iterrows()
    ):
        record = {
            "_row_index": int(
                index
            )
            if isinstance(
                index,
                (int, np.integer),
            )
            else str(index)
        }

        for column in dataframe.columns:
            record[str(column)] = (
                _to_python(
                    row[column]
                )
            )

        sample_records.append(
            record
        )

    duplicate_record_count = int(
        duplicate_mask.sum()
    )

    duplicated_after_first = int(
        safe_frame.duplicated(
            subset=subset,
            keep="first",
        ).sum()
    )

    return {
        "columns_used": (
            subset
            if subset
            else list(
                dataframe.columns
            )
        ),
        "duplicate_records": (
            duplicate_record_count
        ),
        "duplicate_rows_after_first": (
            duplicated_after_first
        ),
        "duplicate_percentage": round(
            (
                duplicate_record_count
                / len(dataframe)
            )
            * 100,
            4,
        )
        if len(dataframe)
        else 0.0,
        "sample_duplicates": (
            sample_records
        ),
    }


def _uniqueness_analysis(
    asset: DataAsset,
    request: ToolRequest,
) -> dict[str, Any]:
    dataframe = _get_table(
        asset,
        request.table_name,
    )

    columns = request.parameters.get(
        "columns"
    )

    if not columns:
        if request.column_name:
            columns = [
                request.column_name
            ]
        else:
            raise ValueError(
                "Provide column_name or parameters.columns."
            )

    if not isinstance(
        columns,
        list,
    ):
        raise ValueError(
            "parameters.columns must be a list."
        )

    missing_columns = [
        column
        for column in columns
        if column
        not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            "Unknown columns: "
            + ", ".join(
                missing_columns
            )
        )

    subset = (
        dataframe[
            columns
        ]
        .copy()
    )

    for column in columns:
        subset[column] = (
            subset[
                column
            ].map(str)
        )

    row_count = len(
        subset
    )

    unique_count = int(
        len(
            subset.drop_duplicates()
        )
    )

    duplicate_count = (
        row_count
        - unique_count
    )

    null_rows = int(
        dataframe[
            columns
        ]
        .isna()
        .any(axis=1)
        .sum()
    )

    return {
        "columns": columns,
        "row_count": int(
            row_count
        ),
        "unique_combinations": (
            unique_count
        ),
        "duplicate_combinations": (
            duplicate_count
        ),
        "rows_with_null": (
            null_rows
        ),
        "uniqueness_percentage": round(
            (
                unique_count
                / row_count
            )
            * 100,
            4,
        )
        if row_count
        else 0.0,
        "is_fully_unique": (
            duplicate_count == 0
            and null_rows == 0
            and row_count > 0
        ),
    }


def _correlation_analysis(
    asset: DataAsset,
    request: ToolRequest,
) -> dict[str, Any]:
    dataframe = _get_table(
        asset,
        request.table_name,
    )

    method = str(
        request.parameters.get(
            "method",
            "pearson",
        )
    ).lower()

    allowed_methods = {
        "pearson",
        "spearman",
        "kendall",
    }

    if method not in allowed_methods:
        raise ValueError(
            f"Unsupported correlation method: {method}"
        )

    if (
        request.column_name
        and request.second_column_name
    ):
        first = pd.to_numeric(
            _get_column(
                dataframe,
                request.column_name,
            ),
            errors="coerce",
        )

        second = pd.to_numeric(
            _get_column(
                dataframe,
                request.second_column_name,
            ),
            errors="coerce",
        )

        valid = pd.DataFrame(
            {
                "first": first,
                "second": second,
            }
        ).dropna()

        if len(valid) < 2:
            return {
                "correlation": None,
                "valid_pairs": int(
                    len(valid)
                ),
                "message": (
                    "Not enough numeric pairs."
                ),
            }

        correlation = (
            valid[
                "first"
            ].corr(
                valid[
                    "second"
                ],
                method=method,
            )
        )

        return {
            "first_column": (
                request.column_name
            ),
            "second_column": (
                request.second_column_name
            ),
            "method": method,
            "valid_pairs": int(
                len(valid)
            ),
            "correlation": (
                float(correlation)
                if pd.notna(
                    correlation
                )
                else None
            ),
        }

    numeric = dataframe.select_dtypes(
        include=[
            "number",
        ]
    )

    if len(
        numeric.columns
    ) < 2:
        return {
            "method": method,
            "relationships": [],
            "message": (
                "Fewer than two numeric columns were found."
            ),
        }

    matrix = numeric.corr(
        method=method
    )

    relationships = []

    columns = list(
        matrix.columns
    )

    for first_index in range(
        len(columns)
    ):
        for second_index in range(
            first_index + 1,
            len(columns),
        ):
            first_column = columns[
                first_index
            ]

            second_column = columns[
                second_index
            ]

            value = matrix.loc[
                first_column,
                second_column,
            ]

            if pd.isna(value):
                continue

            relationships.append(
                {
                    "first_column": (
                        first_column
                    ),
                    "second_column": (
                        second_column
                    ),
                    "correlation": float(
                        value
                    ),
                }
            )

    relationships.sort(
        key=lambda item: abs(
            item["correlation"]
        ),
        reverse=True,
    )

    return {
        "method": method,
        "relationships": (
            relationships[:25]
        ),
    }


def _outlier_analysis(
    asset: DataAsset,
    request: ToolRequest,
) -> dict[str, Any]:
    dataframe = _get_table(
        asset,
        request.table_name,
    )

    series = pd.to_numeric(
        _get_column(
            dataframe,
            request.column_name,
        ),
        errors="coerce",
    )

    valid = (
        series.dropna()
    )

    if valid.empty:
        return {
            "column": (
                request.column_name
            ),
            "message": (
                "No numeric values were available."
            ),
        }

    q1 = float(
        valid.quantile(
            0.25
        )
    )

    q3 = float(
        valid.quantile(
            0.75
        )
    )

    iqr = (
        q3 - q1
    )

    lower = (
        q1
        - 1.5 * iqr
    )

    upper = (
        q3
        + 1.5 * iqr
    )

    iqr_mask = (
        (series < lower)
        | (series > upper)
    )

    median = float(
        valid.median()
    )

    absolute_deviation = (
        valid - median
    ).abs()

    mad = float(
        absolute_deviation.median()
    )

    mad_indexes: list[Any] = []

    if mad > 0:
        full_deviation = (
            series - median
        ).abs()

        modified_z = (
            0.6745
            * full_deviation
            / mad
        )

        mad_indexes = list(
            dataframe.index[
                modified_z > 3.5
            ]
        )

    iqr_indexes = list(
        dataframe.index[
            iqr_mask.fillna(
                False
            )
        ]
    )

    def build_samples(
        indexes: list[Any],
    ) -> list[dict[str, Any]]:
        samples = []

        for index in indexes[:10]:
            samples.append(
                {
                    "row_index": (
                        int(index)
                        if isinstance(
                            index,
                            (int, np.integer),
                        )
                        else str(index)
                    ),
                    "value": (
                        _to_python(
                            dataframe.loc[
                                index,
                                request.column_name,
                            ]
                        )
                    ),
                }
            )

        return samples

    return {
        "column": (
            request.column_name
        ),
        "numeric_count": int(
            len(valid)
        ),
        "median": median,
        "iqr": {
            "lower_bound": float(
                lower
            ),
            "upper_bound": float(
                upper
            ),
            "outlier_count": int(
                len(
                    iqr_indexes
                )
            ),
            "samples": build_samples(
                iqr_indexes
            ),
        },
        "mad": {
            "median_absolute_deviation": (
                mad
            ),
            "outlier_count": int(
                len(
                    mad_indexes
                )
            ),
            "samples": build_samples(
                mad_indexes
            ),
        },
    }


def _column_pair_analysis(
    asset: DataAsset,
    request: ToolRequest,
) -> dict[str, Any]:
    dataframe = _get_table(
        asset,
        request.table_name,
    )

    first = _get_column(
        dataframe,
        request.column_name,
    )

    second = _get_column(
        dataframe,
        request.second_column_name,
    )

    both_null = int(
        (
            first.isna()
            & second.isna()
        ).sum()
    )

    first_only_null = int(
        (
            first.isna()
            & second.notna()
        ).sum()
    )

    second_only_null = int(
        (
            first.notna()
            & second.isna()
        ).sum()
    )

    comparable = (
        first.notna()
        & second.notna()
    )

    comparable_count = int(
        comparable.sum()
    )

    equal_count = 0

    if comparable_count:
        equal_count = int(
            (
                first[
                    comparable
                ].astype(str)
                == second[
                    comparable
                ].astype(str)
            ).sum()
        )

    numeric_first = pd.to_numeric(
        first,
        errors="coerce",
    )

    numeric_second = pd.to_numeric(
        second,
        errors="coerce",
    )

    numeric_pairs = pd.DataFrame(
        {
            "first": (
                numeric_first
            ),
            "second": (
                numeric_second
            ),
        }
    ).dropna()

    correlation = None

    if len(
        numeric_pairs
    ) >= 2:
        value = (
            numeric_pairs[
                "first"
            ].corr(
                numeric_pairs[
                    "second"
                ]
            )
        )

        if pd.notna(value):
            correlation = float(
                value
            )

    return {
        "first_column": (
            request.column_name
        ),
        "second_column": (
            request.second_column_name
        ),
        "both_null": both_null,
        "first_only_null": (
            first_only_null
        ),
        "second_only_null": (
            second_only_null
        ),
        "comparable_rows": (
            comparable_count
        ),
        "equal_values": (
            equal_count
        ),
        "equality_percentage": round(
            (
                equal_count
                / comparable_count
            )
            * 100,
            4,
        )
        if comparable_count
        else None,
        "numeric_pair_count": int(
            len(
                numeric_pairs
            )
        ),
        "numeric_correlation": (
            correlation
        ),
    }


def _date_range_analysis(
    asset: DataAsset,
    request: ToolRequest,
) -> dict[str, Any]:
    dataframe = _get_table(
        asset,
        request.table_name,
    )

    series = _get_column(
        dataframe,
        request.column_name,
    )

    parsed = pd.to_datetime(
        series,
        errors="coerce",
        format="mixed",
    )

    valid = parsed.dropna()

    original_non_null = int(
        series.notna().sum()
    )

    invalid_count = (
        original_non_null
        - len(valid)
    )

    if valid.empty:
        return {
            "column": (
                request.column_name
            ),
            "valid_dates": 0,
            "invalid_dates": int(
                invalid_count
            ),
            "message": (
                "No valid dates could be parsed."
            ),
        }

    now = pd.Timestamp.now()

    if valid.dt.tz is not None:
        now = now.tz_localize(
            valid.dt.tz
        )

    future_count = int(
        (valid > now).sum()
    )

    return {
        "column": (
            request.column_name
        ),
        "valid_dates": int(
            len(valid)
        ),
        "invalid_dates": int(
            invalid_count
        ),
        "earliest": (
            valid.min().isoformat()
        ),
        "latest": (
            valid.max().isoformat()
        ),
        "future_value_count": (
            future_count
        ),
        "future_value_percentage": round(
            (
                future_count
                / len(valid)
            )
            * 100,
            4,
        ),
    }


def _document_excerpt(
    asset: DataAsset,
    request: ToolRequest,
) -> dict[str, Any]:
    document_name = (
        request.parameters.get(
            "document_name"
        )
    )

    if not document_name:
        raise ValueError(
            "parameters.document_name is required."
        )

    if (
        document_name
        not in asset.documents
    ):
        raise ValueError(
            f"Document '{document_name}' was not found."
        )

    text = (
        asset.documents[
            document_name
        ]
        or ""
    )

    start = request.parameters.get(
        "start",
        0,
    )

    length = request.parameters.get(
        "length",
        settings.max_document_chars,
    )

    try:
        start = max(
            0,
            int(start),
        )
    except (TypeError, ValueError):
        start = 0

    try:
        length = int(
            length
        )
    except (TypeError, ValueError):
        length = (
            settings.max_document_chars
        )

    length = max(
        1,
        min(
            length,
            settings.max_document_chars,
        ),
    )

    excerpt = text[
        start:
        start + length
    ]

    return {
        "document_name": (
            document_name
        ),
        "start": start,
        "returned_characters": len(
            excerpt
        ),
        "total_characters": len(
            text
        ),
        "has_more": (
            start
            + len(excerpt)
            < len(text)
        ),
        "text": excerpt,
    }


TOOL_FUNCTIONS = {
    "list_tables": (
        lambda asset, request:
        _list_tables(asset)
    ),
    "table_overview": (
        _table_overview
    ),
    "sample_rows": (
        _sample_rows
    ),
    "column_distribution": (
        _column_distribution
    ),
    "missingness_analysis": (
        _missingness_analysis
    ),
    "duplicate_analysis": (
        _duplicate_analysis
    ),
    "uniqueness_analysis": (
        _uniqueness_analysis
    ),
    "correlation_analysis": (
        _correlation_analysis
    ),
    "outlier_analysis": (
        _outlier_analysis
    ),
    "column_pair_analysis": (
        _column_pair_analysis
    ),
    "date_range_analysis": (
        _date_range_analysis
    ),
    "document_excerpt": (
        _document_excerpt
    ),
}


def execute_tool(
    asset: DataAsset,
    request: ToolRequest,
) -> ToolResult:
    """
    Execute one whitelisted analytical tool requested by the agent.

    The language model never gets direct Python execution access.
    """

    tool_name = (
        request.tool_name
        .strip()
        .lower()
    )

    if tool_name not in TOOL_FUNCTIONS:
        return ToolResult(
            tool_name=(
                request.tool_name
            ),
            success=False,
            table_name=(
                request.table_name
            ),
            column_name=(
                request.column_name
            ),
            message=(
                f"Unknown tool: {request.tool_name}"
            ),
        )

    try:
        tool_function = (
            TOOL_FUNCTIONS[
                tool_name
            ]
        )

        result = tool_function(
            asset,
            request,
        )

        return ToolResult(
            tool_name=tool_name,
            success=True,
            table_name=(
                request.table_name
            ),
            column_name=(
                request.column_name
            ),
            result=_to_python(
                result
            ),
        )

    except Exception as exc:
        return ToolResult(
            tool_name=tool_name,
            success=False,
            table_name=(
                request.table_name
            ),
            column_name=(
                request.column_name
            ),
            message=str(
                exc
            ),
        )