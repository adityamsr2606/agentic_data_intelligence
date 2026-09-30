from __future__ import annotations

import ipaddress
import re
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from src.models import (
    DataAsset,
    DataQualityRule,
    RuleExecutionResult,
)


# -------------------------------------------------------------------
# GENERIC PATTERNS
# -------------------------------------------------------------------

PATTERNS: dict[str, re.Pattern[str]] = {
    "email": re.compile(
        r"^[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@"
        r"[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+$"
    ),
    "url": re.compile(
        r"^https?://[^\s]+$",
        re.IGNORECASE,
    ),
    "uuid": re.compile(
        r"^[0-9a-fA-F]{8}-"
        r"[0-9a-fA-F]{4}-"
        r"[1-5][0-9a-fA-F]{3}-"
        r"[89abAB][0-9a-fA-F]{3}-"
        r"[0-9a-fA-F]{12}$"
    ),
    "integer_string": re.compile(
        r"^[+-]?\d+$"
    ),
    "decimal_string": re.compile(
        r"^[+-]?(?:\d+(?:\.\d+)?|\.\d+)$"
    ),
}


SUPPORTED_RULE_TYPES = {
    "type_check",
    "pattern_match",
    "not_null",
    "unique",
    "range",
    "allowed_values",
    "length",
    "datetime_parse",
    "not_future",
    "duplicate_rows",
}


# -------------------------------------------------------------------
# ROW PARTITION RESULT
# -------------------------------------------------------------------

@dataclass
class RowPartitionResult:
    """
    Valid and rejected source rows after deterministic rule execution.

    Invalid tables retain the original row values plus DQ metadata.
    """

    valid_tables: dict[str, pd.DataFrame]
    invalid_tables: dict[str, pd.DataFrame]
    valid_row_count: int
    rejected_row_count: int


# -------------------------------------------------------------------
# COMMON HELPERS
# -------------------------------------------------------------------

def _get_dataframe(
    asset: DataAsset,
    rule: DataQualityRule,
) -> pd.DataFrame:
    dataframe = asset.tables.get(
        rule.table_name
    )

    if dataframe is None:
        raise ValueError(
            f"Unknown table '{rule.table_name}'."
        )

    return dataframe


def _validate_columns(
    dataframe: pd.DataFrame,
    rule: DataQualityRule,
) -> None:
    missing = [
        column
        for column in rule.columns
        if column not in dataframe.columns
    ]

    if missing:
        raise ValueError(
            "Rule references unknown column(s): "
            + ", ".join(missing)
        )


def _single_column(
    dataframe: pd.DataFrame,
    rule: DataQualityRule,
) -> pd.Series:
    if len(rule.columns) != 1:
        raise ValueError(
            f"Rule '{rule.rule_type}' requires exactly one column."
        )

    return dataframe[
        rule.columns[0]
    ]


def _failed_positions(
    mask: pd.Series,
) -> list[int]:
    """
    Return zero-based dataframe row positions.

    Row positions are used instead of dataframe index labels so
    rejection metadata remains stable after dataframe transformations.
    """

    return (
        np.flatnonzero(
            mask.to_numpy(
                dtype=bool
            )
        )
        .astype(int)
        .tolist()
    )


def _result(
    rule: DataQualityRule,
    rows_checked: int,
    failed_positions: list[int],
    passed_message: str,
    failed_message: str,
) -> RuleExecutionResult:
    failed_rows = len(
        failed_positions
    )

    pass_percentage = (
        100.0
        if rows_checked == 0
        else round(
            (
                (
                    rows_checked
                    - failed_rows
                )
                / rows_checked
            )
            * 100,
            2,
        )
    )

    return RuleExecutionResult(
        rule_id=rule.rule_id,
        status=(
            "passed"
            if failed_rows == 0
            else "failed"
        ),
        rows_checked=rows_checked,
        failed_rows=failed_rows,
        pass_percentage=pass_percentage,
        failed_row_indexes=(
            failed_positions
        ),
        message=(
            passed_message
            if failed_rows == 0
            else failed_message
        ),
    )


def _skip_result(
    rule: DataQualityRule,
    message: str,
) -> RuleExecutionResult:
    return RuleExecutionResult(
        rule_id=rule.rule_id,
        status="skipped",
        rows_checked=0,
        failed_rows=0,
        pass_percentage=100.0,
        failed_row_indexes=[],
        message=message,
    )


def _error_result(
    rule: DataQualityRule,
    message: str,
) -> RuleExecutionResult:
    return RuleExecutionResult(
        rule_id=rule.rule_id,
        status="error",
        rows_checked=0,
        failed_rows=0,
        pass_percentage=0.0,
        failed_row_indexes=[],
        message=message,
    )


# -------------------------------------------------------------------
# TYPE CHECK
# -------------------------------------------------------------------

def _execute_type_check(
    dataframe: pd.DataFrame,
    rule: DataQualityRule,
) -> RuleExecutionResult:
    series = _single_column(
        dataframe,
        rule,
    )

    expected_type = str(
        rule.parameters.get(
            "expected_type",
            "",
        )
    ).strip().lower()

    applicable = (
        series.notna()
    )

    checked = int(
        applicable.sum()
    )

    if checked == 0:
        return _result(
            rule,
            0,
            [],
            "No non-null values were available to check.",
            "",
        )

    failures = pd.Series(
        False,
        index=series.index,
    )

    values = (
        series.loc[
            applicable
        ]
    )

    if expected_type in {
        "number",
        "numeric",
        "float",
    }:
        converted = pd.to_numeric(
            values,
            errors="coerce",
        )

        failures.loc[
            applicable
        ] = converted.isna()

    elif expected_type in {
        "integer",
        "int",
    }:
        converted = pd.to_numeric(
            values,
            errors="coerce",
        )

        failures.loc[
            applicable
        ] = (
            converted.isna()
            | (
                converted.notna()
                & (
                    converted.mod(1)
                    != 0
                )
            )
        )

    elif expected_type in {
        "datetime",
        "date",
    }:
        converted = pd.to_datetime(
            values,
            errors="coerce",
            format="mixed",
        )

        failures.loc[
            applicable
        ] = converted.isna()

    elif expected_type in {
        "boolean",
        "bool",
    }:
        allowed_strings = {
            "true",
            "false",
            "yes",
            "no",
            "y",
            "n",
            "1",
            "0",
        }

        def is_boolean(
            value: object,
        ) -> bool:
            if isinstance(
                value,
                (
                    bool,
                    np.bool_,
                ),
            ):
                return True

            if isinstance(
                value,
                (
                    int,
                    np.integer,
                ),
            ):
                return value in {
                    0,
                    1,
                }

            if isinstance(
                value,
                str,
            ):
                return (
                    value
                    .strip()
                    .lower()
                    in allowed_strings
                )

            return False

        failures.loc[
            applicable
        ] = ~values.map(
            is_boolean
        )

    elif expected_type in {
        "string",
        "text",
    }:
        failures.loc[
            applicable
        ] = ~values.map(
            lambda value: isinstance(
                value,
                str,
            )
        )

    else:
        raise ValueError(
            "Unsupported expected type "
            f"'{expected_type}'."
        )

    failed = (
        _failed_positions(
            failures
        )
    )

    return _result(
        rule,
        checked,
        failed,
        (
            "All checked values satisfy the "
            f"expected '{expected_type}' type."
        ),
        (
            "One or more values do not satisfy the "
            f"expected '{expected_type}' type."
        ),
    )


# -------------------------------------------------------------------
# PATTERN MATCH
# -------------------------------------------------------------------

def _contains_alphabetic_character(
    value: object,
) -> bool:
    """
    Return True when a value contains at least one Unicode alphabetic
    character.

    This deliberately avoids English-only name validation.

    Examples that pass:
        Rahul Sharma
        Mary-Jane O'Connor
        José García
        李明
        A. Kumar

    Examples that fail:
        900113
        12345
        ---
        whitespace only
    """

    text = str(
        value
    ).strip()

    if not text:
        return False

    return any(
        character.isalpha()
        for character in text
    )


def _execute_pattern_match(
    dataframe: pd.DataFrame,
    rule: DataQualityRule,
) -> RuleExecutionResult:
    series = _single_column(
        dataframe,
        rule,
    )

    pattern_name = str(
        rule.parameters.get(
            "pattern_name",
            "",
        )
    ).strip().lower()

    applicable = (
        series.notna()
    )

    checked = int(
        applicable.sum()
    )

    failures = pd.Series(
        False,
        index=series.index,
    )

    values = (
        series.loc[
            applicable
        ]
    )

    if pattern_name == "ip":

        def valid_ip(
            value: object,
        ) -> bool:
            try:
                ipaddress.ip_address(
                    str(
                        value
                    ).strip()
                )

                return True

            except ValueError:
                return False

        failures.loc[
            applicable
        ] = ~values.map(
            valid_ip
        )

    elif (
        pattern_name
        == "contains_letters"
    ):
        failures.loc[
            applicable
        ] = ~values.map(
            _contains_alphabetic_character
        )

    else:
        pattern = (
            PATTERNS.get(
                pattern_name
            )
        )

        if pattern is None:
            raise ValueError(
                "Unsupported pattern "
                f"'{pattern_name}'."
            )

        failures.loc[
            applicable
        ] = ~(
            values.astype(str)
            .str.fullmatch(
                pattern
            )
            .fillna(False)
        )

    failed = (
        _failed_positions(
            failures
        )
    )

    return _result(
        rule,
        checked,
        failed,
        (
            "All checked values match the "
            f"'{pattern_name}' pattern."
        ),
        (
            "One or more values do not match the "
            f"'{pattern_name}' pattern."
        ),
    )


# -------------------------------------------------------------------
# NULLABILITY
# -------------------------------------------------------------------

def _execute_not_null(
    dataframe: pd.DataFrame,
    rule: DataQualityRule,
) -> RuleExecutionResult:
    series = _single_column(
        dataframe,
        rule,
    )

    failures = (
        series.isna()
    )

    failed = (
        _failed_positions(
            failures
        )
    )

    return _result(
        rule,
        len(series),
        failed,
        "No null values were found.",
        "One or more null values were found.",
    )


# -------------------------------------------------------------------
# UNIQUENESS
# -------------------------------------------------------------------

def _execute_unique(
    dataframe: pd.DataFrame,
    rule: DataQualityRule,
) -> RuleExecutionResult:
    series = _single_column(
        dataframe,
        rule,
    )

    applicable = (
        series.notna()
    )

    failures = pd.Series(
        False,
        index=series.index,
    )

    failures.loc[
        applicable
    ] = (
        series.loc[
            applicable
        ]
        .duplicated(
            keep=False
        )
    )

    failed = (
        _failed_positions(
            failures
        )
    )

    return _result(
        rule,
        int(
            applicable.sum()
        ),
        failed,
        "All checked non-null values are unique.",
        "Duplicate non-null values were found.",
    )


# -------------------------------------------------------------------
# RANGE
# -------------------------------------------------------------------

def _execute_range(
    dataframe: pd.DataFrame,
    rule: DataQualityRule,
) -> RuleExecutionResult:
    series = _single_column(
        dataframe,
        rule,
    )

    minimum = (
        rule.parameters.get(
            "min"
        )
    )

    maximum = (
        rule.parameters.get(
            "max"
        )
    )

    if (
        minimum is None
        and maximum is None
    ):
        raise ValueError(
            "Range rule requires min, max, or both."
        )

    # Current rule_generator.py emits one `inclusive` parameter.
    # Legacy inclusive_min / inclusive_max are still accepted so older
    # rules do not suddenly become incompatible.
    inclusive = (
        rule.parameters.get(
            "inclusive"
        )
    )

    if isinstance(
        inclusive,
        bool,
    ):
        inclusive_min = inclusive
        inclusive_max = inclusive

    else:
        inclusive_min = bool(
            rule.parameters.get(
                "inclusive_min",
                True,
            )
        )

        inclusive_max = bool(
            rule.parameters.get(
                "inclusive_max",
                True,
            )
        )

    applicable = (
        series.notna()
    )

    values = (
        series.loc[
            applicable
        ]
    )

    numeric = pd.to_numeric(
        values,
        errors="coerce",
    )

    failures = pd.Series(
        False,
        index=series.index,
    )

    local_failures = (
        numeric.isna()
    )

    if minimum is not None:
        minimum_value = float(
            minimum
        )

        if inclusive_min:
            local_failures |= (
                numeric
                < minimum_value
            )

        else:
            local_failures |= (
                numeric
                <= minimum_value
            )

    if maximum is not None:
        maximum_value = float(
            maximum
        )

        if inclusive_max:
            local_failures |= (
                numeric
                > maximum_value
            )

        else:
            local_failures |= (
                numeric
                >= maximum_value
            )

    failures.loc[
        applicable
    ] = local_failures

    failed = (
        _failed_positions(
            failures
        )
    )

    return _result(
        rule,
        int(
            applicable.sum()
        ),
        failed,
        "All checked values are within the allowed range.",
        "One or more values are outside the allowed range.",
    )


# -------------------------------------------------------------------
# ALLOWED VALUES
# -------------------------------------------------------------------

def _execute_allowed_values(
    dataframe: pd.DataFrame,
    rule: DataQualityRule,
) -> RuleExecutionResult:
    series = _single_column(
        dataframe,
        rule,
    )

    allowed = (
        rule.parameters.get(
            "values"
        )
    )

    if not isinstance(
        allowed,
        list,
    ):
        raise ValueError(
            "allowed_values rule requires a list parameter named "
            "'values'."
        )

    applicable = (
        series.notna()
    )

    failures = pd.Series(
        False,
        index=series.index,
    )

    failures.loc[
        applicable
    ] = ~(
        series.loc[
            applicable
        ]
        .isin(
            allowed
        )
    )

    failed = (
        _failed_positions(
            failures
        )
    )

    return _result(
        rule,
        int(
            applicable.sum()
        ),
        failed,
        "All checked values are allowed.",
        "One or more values are not in the allowed set.",
    )


# -------------------------------------------------------------------
# STRING LENGTH
# -------------------------------------------------------------------

def _execute_length(
    dataframe: pd.DataFrame,
    rule: DataQualityRule,
) -> RuleExecutionResult:
    series = _single_column(
        dataframe,
        rule,
    )

    # Current rule_generator.py uses min_length / max_length.
    # min / max remain accepted for compatibility with older rules.
    minimum = (
        rule.parameters.get(
            "min_length",
            rule.parameters.get(
                "min"
            ),
        )
    )

    maximum = (
        rule.parameters.get(
            "max_length",
            rule.parameters.get(
                "max"
            ),
        )
    )

    if (
        minimum is None
        and maximum is None
    ):
        raise ValueError(
            "Length rule requires min_length, max_length, or both."
        )

    applicable = (
        series.notna()
    )

    lengths = (
        series.loc[
            applicable
        ]
        .astype(str)
        .str.len()
    )

    local_failures = pd.Series(
        False,
        index=lengths.index,
    )

    if minimum is not None:
        local_failures |= (
            lengths
            < int(
                minimum
            )
        )

    if maximum is not None:
        local_failures |= (
            lengths
            > int(
                maximum
            )
        )

    failures = pd.Series(
        False,
        index=series.index,
    )

    failures.loc[
        applicable
    ] = local_failures

    failed = (
        _failed_positions(
            failures
        )
    )

    return _result(
        rule,
        int(
            applicable.sum()
        ),
        failed,
        "All checked values satisfy the length constraint.",
        "One or more values violate the length constraint.",
    )


# -------------------------------------------------------------------
# DATETIME
# -------------------------------------------------------------------

def _execute_datetime_parse(
    dataframe: pd.DataFrame,
    rule: DataQualityRule,
) -> RuleExecutionResult:
    series = _single_column(
        dataframe,
        rule,
    )

    applicable = (
        series.notna()
    )

    parsed = pd.to_datetime(
        series.loc[
            applicable
        ],
        errors="coerce",
        format="mixed",
    )

    failures = pd.Series(
        False,
        index=series.index,
    )

    failures.loc[
        applicable
    ] = parsed.isna()

    failed = (
        _failed_positions(
            failures
        )
    )

    return _result(
        rule,
        int(
            applicable.sum()
        ),
        failed,
        "All checked values can be parsed as datetime values.",
        "One or more values cannot be parsed as datetime values.",
    )


def _execute_not_future(
    dataframe: pd.DataFrame,
    rule: DataQualityRule,
) -> RuleExecutionResult:
    series = _single_column(
        dataframe,
        rule,
    )

    applicable = (
        series.notna()
    )

    parsed = pd.to_datetime(
        series.loc[
            applicable
        ],
        errors="coerce",
        format="mixed",
        utc=True,
    )

    now = pd.Timestamp.now(
        tz="UTC"
    )

    local_failures = (
        parsed.isna()
        | (
            parsed
            > now
        )
    )

    failures = pd.Series(
        False,
        index=series.index,
    )

    failures.loc[
        applicable
    ] = local_failures

    failed = (
        _failed_positions(
            failures
        )
    )

    return _result(
        rule,
        int(
            applicable.sum()
        ),
        failed,
        "All checked datetime values are valid and not in the future.",
        "One or more datetime values are invalid or in the future.",
    )


# -------------------------------------------------------------------
# DUPLICATE ROWS
# -------------------------------------------------------------------

def _execute_duplicate_rows(
    dataframe: pd.DataFrame,
    rule: DataQualityRule,
) -> RuleExecutionResult:
    subset = (
        rule.columns
        if rule.columns
        else None
    )

    failures = (
        dataframe.duplicated(
            subset=subset,
            keep=False,
        )
    )

    failed = (
        _failed_positions(
            failures
        )
    )

    return _result(
        rule,
        len(
            dataframe
        ),
        failed,
        "No duplicate rows were found.",
        "One or more duplicate rows were found.",
    )


# -------------------------------------------------------------------
# EXECUTOR REGISTRY
# -------------------------------------------------------------------

EXECUTORS = {
    "type_check": (
        _execute_type_check
    ),
    "pattern_match": (
        _execute_pattern_match
    ),
    "not_null": (
        _execute_not_null
    ),
    "unique": (
        _execute_unique
    ),
    "range": (
        _execute_range
    ),
    "allowed_values": (
        _execute_allowed_values
    ),
    "length": (
        _execute_length
    ),
    "datetime_parse": (
        _execute_datetime_parse
    ),
    "not_future": (
        _execute_not_future
    ),
    "duplicate_rows": (
        _execute_duplicate_rows
    ),
}


# -------------------------------------------------------------------
# RULE EXECUTION API
# -------------------------------------------------------------------

def execute_rule(
    asset: DataAsset,
    rule: DataQualityRule,
) -> RuleExecutionResult:
    """
    Execute one rule deterministically.

    Individual rule errors are returned as structured results rather
    than terminating every other rule.
    """

    try:
        if (
            rule.rule_type
            not in SUPPORTED_RULE_TYPES
        ):
            return _skip_result(
                rule,
                (
                    "Unsupported rule type "
                    f"'{rule.rule_type}'."
                ),
            )

        dataframe = (
            _get_dataframe(
                asset,
                rule,
            )
        )

        _validate_columns(
            dataframe,
            rule,
        )

        executor = (
            EXECUTORS[
                rule.rule_type
            ]
        )

        return executor(
            dataframe,
            rule,
        )

    except Exception as exc:
        return _error_result(
            rule,
            str(
                exc
            ),
        )


def execute_rules(
    asset: DataAsset,
    rules: Iterable[
        DataQualityRule
    ],
) -> list[
    RuleExecutionResult
]:
    """
    Execute rules independently so one malformed rule does not prevent
    the remaining rules from running.
    """

    return [
        execute_rule(
            asset,
            rule,
        )
        for rule in rules
    ]


# -------------------------------------------------------------------
# VALID / INVALID ROW PARTITIONING
# -------------------------------------------------------------------

def _build_rule_lookup(
    rules: Iterable[
        DataQualityRule
    ],
) -> dict[
    str,
    DataQualityRule,
]:
    lookup: dict[
        str,
        DataQualityRule,
    ] = {}

    for rule in rules:

        if (
            rule.rule_id
            in lookup
        ):
            raise ValueError(
                "Duplicate DQ rule id "
                f"'{rule.rule_id}'."
            )

        lookup[
            rule.rule_id
        ] = rule

    return lookup


def _build_result_lookup(
    results: Iterable[
        RuleExecutionResult
    ],
) -> dict[
    str,
    RuleExecutionResult,
]:
    lookup: dict[
        str,
        RuleExecutionResult,
    ] = {}

    for result in results:

        if (
            result.rule_id
            in lookup
        ):
            raise ValueError(
                "Duplicate DQ execution result for rule "
                f"'{result.rule_id}'."
            )

        lookup[
            result.rule_id
        ] = result

    return lookup


def _rejection_reason(
    rule: DataQualityRule,
    result: RuleExecutionResult,
) -> str:
    columns = (
        ", ".join(
            rule.columns
        )
        if rule.columns
        else "row"
    )

    return (
        f"{rule.rule_type} [{columns}]: "
        f"{result.message}"
    )


def partition_rows(
    asset: DataAsset,
    rules: Iterable[
        DataQualityRule
    ],
    results: Iterable[
        RuleExecutionResult
    ],
    *,
    blocking_severities: set[str] | None = None,
    fail_on_rule_error: bool = True,
) -> RowPartitionResult:
    """
    Split source rows into valid and invalid datasets.

    By default, critical/high/medium failures are blocking.

    Low-severity rules remain findings but do not reject rows.

    A rejected row retains:
      - original zero-based source row position
      - failed rule count
      - failed rule ids
      - human-readable rejection reasons

    The function is completely generic and does not know anything
    about email, measurements, dates or any particular dataset.
    """

    if blocking_severities is None:
        blocking_severities = {
            "critical",
            "high",
            "medium",
        }

    normalized_blocking = {
        severity
        .strip()
        .lower()
        for severity
        in blocking_severities
    }

    rules_list = list(
        rules
    )

    results_list = list(
        results
    )

    rule_lookup = (
        _build_rule_lookup(
            rules_list
        )
    )

    result_lookup = (
        _build_result_lookup(
            results_list
        )
    )

    missing_results = (
        set(
            rule_lookup
        )
        - set(
            result_lookup
        )
    )

    if missing_results:
        raise ValueError(
            "Missing DQ execution result(s) for: "
            + ", ".join(
                sorted(
                    missing_results
                )
            )
        )

    # table -> row position -> rule ids / reasons
    rejected: dict[
        str,
        dict[
            int,
            dict[
                str,
                list[str],
            ],
        ],
    ] = {}

    for rule in rules_list:

        severity = (
            rule.severity
            .strip()
            .lower()
        )

        if (
            severity
            not in normalized_blocking
        ):
            continue

        result = (
            result_lookup[
                rule.rule_id
            ]
        )

        if (
            result.status
            == "error"
        ):
            if fail_on_rule_error:
                raise ValueError(
                    "Cannot safely partition rows because "
                    f"blocking rule '{rule.rule_id}' failed "
                    f"to execute: {result.message}"
                )

            continue

        if (
            result.status
            in {
                "passed",
                "skipped",
            }
        ):
            continue

        if (
            result.status
            != "failed"
        ):
            continue

        dataframe = (
            asset.tables.get(
                rule.table_name
            )
        )

        if dataframe is None:
            raise ValueError(
                "DQ rule references unknown table "
                f"'{rule.table_name}'."
            )

        reason = (
            _rejection_reason(
                rule,
                result,
            )
        )

        table_rejections = (
            rejected.setdefault(
                rule.table_name,
                {},
            )
        )

        for position in (
            result.failed_row_indexes
        ):

            position = int(
                position
            )

            if (
                position < 0
                or position
                >= len(
                    dataframe
                )
            ):
                raise ValueError(
                    "DQ result contains an invalid row "
                    f"position {position} for table "
                    f"'{rule.table_name}'."
                )

            row_info = (
                table_rejections.setdefault(
                    position,
                    {
                        "rule_ids": [],
                        "reasons": [],
                    },
                )
            )

            if (
                rule.rule_id
                not in row_info[
                    "rule_ids"
                ]
            ):
                row_info[
                    "rule_ids"
                ].append(
                    rule.rule_id
                )

            if (
                reason
                not in row_info[
                    "reasons"
                ]
            ):
                row_info[
                    "reasons"
                ].append(
                    reason
                )

    valid_tables: dict[
        str,
        pd.DataFrame,
    ] = {}

    invalid_tables: dict[
        str,
        pd.DataFrame,
    ] = {}

    valid_row_count = 0
    rejected_row_count = 0

    for (
        table_name,
        dataframe,
    ) in asset.tables.items():

        table_rejections = (
            rejected.get(
                table_name,
                {},
            )
        )

        rejected_positions = sorted(
            table_rejections
        )

        rejected_set = set(
            rejected_positions
        )

        valid_positions = [
            position
            for position
            in range(
                len(
                    dataframe
                )
            )
            if position
            not in rejected_set
        ]

        valid_dataframe = (
            dataframe.iloc[
                valid_positions
            ]
            .copy()
            .reset_index(
                drop=True
            )
        )

        valid_tables[
            table_name
        ] = valid_dataframe

        valid_row_count += len(
            valid_dataframe
        )

        if rejected_positions:

            invalid_dataframe = (
                dataframe.iloc[
                    rejected_positions
                ]
                .copy()
                .reset_index(
                    drop=True
                )
            )

            invalid_dataframe[
                "_dq_source_row_position"
            ] = (
                rejected_positions
            )

            invalid_dataframe[
                "_dq_failed_rule_count"
            ] = [
                len(
                    table_rejections[
                        position
                    ][
                        "rule_ids"
                    ]
                )
                for position
                in rejected_positions
            ]

            invalid_dataframe[
                "_dq_failed_rule_ids"
            ] = [
                table_rejections[
                    position
                ][
                    "rule_ids"
                ]
                for position
                in rejected_positions
            ]

            invalid_dataframe[
                "_dq_rejection_reasons"
            ] = [
                table_rejections[
                    position
                ][
                    "reasons"
                ]
                for position
                in rejected_positions
            ]

        else:

            invalid_dataframe = (
                dataframe.iloc[
                    0:0
                ]
                .copy()
            )

            invalid_dataframe[
                "_dq_source_row_position"
            ] = pd.Series(
                dtype="int64"
            )

            invalid_dataframe[
                "_dq_failed_rule_count"
            ] = pd.Series(
                dtype="int64"
            )

            invalid_dataframe[
                "_dq_failed_rule_ids"
            ] = pd.Series(
                dtype="object"
            )

            invalid_dataframe[
                "_dq_rejection_reasons"
            ] = pd.Series(
                dtype="object"
            )

        invalid_tables[
            table_name
        ] = invalid_dataframe

        rejected_row_count += len(
            invalid_dataframe
        )

    return RowPartitionResult(
        valid_tables=valid_tables,
        invalid_tables=invalid_tables,
        valid_row_count=(
            valid_row_count
        ),
        rejected_row_count=(
            rejected_row_count
        ),
    )