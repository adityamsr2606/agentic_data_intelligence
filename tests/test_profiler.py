from __future__ import annotations

import pandas as pd

from src.ingestion import load_asset
from src.profiler import profile_asset


# -------------------------------------------------------------------
# TEST DATA
# -------------------------------------------------------------------

def _create_profile_csv(tmp_path):
    source = tmp_path / "profile_sample.csv"

    source.write_text(
        "abc_code,measurement,event_time,contact\n"
        "X001,18.5,2026-01-10,user1@example.com\n"
        "X002,20.1,2026-01-11,user2@example.com\n"
        "X003,999.0,2026-01-12,bad-email\n"
        "X004,,2026-01-13,user4@example.com\n"
        "X004,,2026-01-13,user4@example.com\n",
        encoding="utf-8",
    )

    return source


def _column(
    profile,
    table_name: str,
    column_name: str,
):
    table = next(
        table
        for table in profile.tables
        if table.table_name == table_name
    )

    return next(
        column
        for column in table.columns
        if column.column_name == column_name
    )


# -------------------------------------------------------------------
# TABLE PROFILE
# -------------------------------------------------------------------

def test_table_profile_counts(tmp_path):
    source = _create_profile_csv(
        tmp_path
    )

    asset = load_asset(
        source
    )

    profile = profile_asset(
        asset
    )

    assert profile.table_count == 1
    assert profile.document_count == 0

    table = profile.tables[0]

    assert table.row_count == 5
    assert table.column_count == 4

    assert (
        table.duplicate_row_count
        == 1
    )

    assert (
        table.duplicate_percentage
        == 20.0
    )


# -------------------------------------------------------------------
# STRING / DISTINCTNESS
# -------------------------------------------------------------------

def test_code_column_profile(
    tmp_path,
):
    source = _create_profile_csv(
        tmp_path
    )

    profile = profile_asset(
        load_asset(
            source
        )
    )

    column = _column(
        profile,
        "profile_sample",
        "abc_code",
    )

    assert (
        column.inferred_type
        == "string"
    )

    assert column.row_count == 5
    assert column.null_count == 0
    assert column.null_percentage == 0.0

    assert column.distinct_count == 4
    assert (
        column.distinct_percentage
        == 80.0
    )

    assert "X001" in (
        column.sample_values
    )


# -------------------------------------------------------------------
# NUMERIC PROFILE
# -------------------------------------------------------------------

def test_numeric_column_profile(
    tmp_path,
):
    source = _create_profile_csv(
        tmp_path
    )

    profile = profile_asset(
        load_asset(
            source
        )
    )

    column = _column(
        profile,
        "profile_sample",
        "measurement",
    )

    assert (
        column.inferred_type
        == "float"
    )

    assert column.row_count == 5
    assert column.null_count == 2
    assert (
        column.null_percentage
        == 40.0
    )

    assert column.distinct_count == 3
    assert (
        column.distinct_percentage
        == 100.0
    )

    statistics = (
        column.numeric_statistics
    )

    assert statistics[
        "minimum"
    ] == 18.5

    assert statistics[
        "maximum"
    ] == 999.0

    assert statistics[
        "median"
    ] == 20.1

    assert statistics[
        "iqr_outlier_count"
    ] == 0

    assert statistics[
        "mad_outlier_count"
    ] == 1


# -------------------------------------------------------------------
# DATETIME PROFILE
# -------------------------------------------------------------------

def test_datetime_column_profile(
    tmp_path,
):
    source = _create_profile_csv(
        tmp_path
    )

    profile = profile_asset(
        load_asset(
            source
        )
    )

    column = _column(
        profile,
        "profile_sample",
        "event_time",
    )

    assert (
        column.inferred_type
        == "datetime"
    )

    assert column.null_count == 0

    statistics = (
        column.datetime_statistics
    )

    assert (
        statistics[
            "valid_datetime_count"
        ]
        == 5
    )

    assert (
        statistics[
            "invalid_datetime_count"
        ]
        == 0
    )

    assert (
        statistics[
            "earliest"
        ]
        == "2026-01-10T00:00:00"
    )

    assert (
        statistics[
            "latest"
        ]
        == "2026-01-13T00:00:00"
    )


# -------------------------------------------------------------------
# PATTERN DETECTION
# -------------------------------------------------------------------

def test_email_pattern_detection(
    tmp_path,
):
    source = _create_profile_csv(
        tmp_path
    )

    profile = profile_asset(
        load_asset(
            source
        )
    )

    column = _column(
        profile,
        "profile_sample",
        "contact",
    )

    email_pattern = next(
        pattern
        for pattern in column.patterns
        if pattern.pattern_name == "email"
    )

    assert (
        email_pattern.checked_values
        == 5
    )

    assert (
        email_pattern.matched_values
        == 4
    )

    assert (
        email_pattern.match_percentage
        == 80.0
    )


# -------------------------------------------------------------------
# DECIMAL STRING PATTERN
# -------------------------------------------------------------------

def test_numeric_pattern_detection(
    tmp_path,
):
    source = _create_profile_csv(
        tmp_path
    )

    profile = profile_asset(
        load_asset(
            source
        )
    )

    column = _column(
        profile,
        "profile_sample",
        "measurement",
    )

    pattern = next(
        pattern
        for pattern in column.patterns
        if (
            pattern.pattern_name
            == "decimal_string"
        )
    )

    assert (
        pattern.checked_values
        == 3
    )

    assert (
        pattern.matched_values
        == 3
    )

    assert (
        pattern.match_percentage
        == 100.0
    )


# -------------------------------------------------------------------
# OUTLIER PRINCIPLE
# -------------------------------------------------------------------

def test_outlier_is_profiled_not_rejected(
    tmp_path,
):
    """
    Profiling should describe statistical unusualness only.

    It must not mutate or remove the original 999.0 observation.
    """

    source = _create_profile_csv(
        tmp_path
    )

    asset = load_asset(
        source
    )

    original = (
        asset.tables[
            "profile_sample"
        ]
        .copy()
    )

    profile_asset(
        asset
    )

    pd.testing.assert_frame_equal(
        asset.tables[
            "profile_sample"
        ],
        original,
    )

    assert (
        asset.tables[
            "profile_sample"
        ]
        .iloc[2][
            "measurement"
        ]
        == 999.0
    )