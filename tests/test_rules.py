from __future__ import annotations

from src.ingestion import load_asset
from src.models import DataQualityRule
from src.profiler import profile_asset
from src.rule_executor import (
    execute_rule,
    execute_rules,
    partition_rows,
)
from src.rule_generator import generate_rules
from src.semantic_analyzer import analyze_semantics


# -------------------------------------------------------------------
# TEST DATA
# -------------------------------------------------------------------

def _create_rule_csv(tmp_path):
    source = tmp_path / "rule_sample.csv"

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


# -------------------------------------------------------------------
# MANUAL DETERMINISTIC RULES
# -------------------------------------------------------------------

def test_email_pattern_rule(tmp_path):
    asset = load_asset(
        _create_rule_csv(
            tmp_path
        )
    )

    rule = DataQualityRule(
        rule_id="email_test",
        table_name="rule_sample",
        columns=["contact"],
        rule_type="pattern_match",
        parameters={
            "pattern_name": "email"
        },
        severity="medium",
        confidence=1.0,
        reason="Test rule.",
        generated_by="deterministic",
    )

    result = execute_rule(
        asset,
        rule,
    )

    assert result.status == "failed"
    assert result.rows_checked == 5
    assert result.failed_rows == 1

    assert (
        result.failed_row_indexes
        == [2]
    )

    assert (
        result.pass_percentage
        == 80.0
    )


def test_numeric_type_rule_ignores_nulls(
    tmp_path,
):
    asset = load_asset(
        _create_rule_csv(
            tmp_path
        )
    )

    rule = DataQualityRule(
        rule_id="numeric_test",
        table_name="rule_sample",
        columns=["measurement"],
        rule_type="type_check",
        parameters={
            "expected_type": "number"
        },
        severity="medium",
        confidence=1.0,
        reason="Test rule.",
        generated_by="deterministic",
    )

    result = execute_rule(
        asset,
        rule,
    )

    assert result.status == "passed"

    # Only non-null values are checked.
    assert result.rows_checked == 3
    assert result.failed_rows == 0

    assert (
        result.pass_percentage
        == 100.0
    )


def test_datetime_parse_rule(
    tmp_path,
):
    asset = load_asset(
        _create_rule_csv(
            tmp_path
        )
    )

    rule = DataQualityRule(
        rule_id="datetime_test",
        table_name="rule_sample",
        columns=["event_time"],
        rule_type="datetime_parse",
        parameters={},
        severity="medium",
        confidence=1.0,
        reason="Test rule.",
        generated_by="deterministic",
    )

    result = execute_rule(
        asset,
        rule,
    )

    assert result.status == "passed"
    assert result.rows_checked == 5
    assert result.failed_rows == 0


# -------------------------------------------------------------------
# UNIQUE RULE
# -------------------------------------------------------------------

def test_unique_rule_marks_all_duplicates(
    tmp_path,
):
    asset = load_asset(
        _create_rule_csv(
            tmp_path
        )
    )

    rule = DataQualityRule(
        rule_id="unique_test",
        table_name="rule_sample",
        columns=["abc_code"],
        rule_type="unique",
        parameters={},
        severity="high",
        confidence=1.0,
        reason="Test rule.",
        generated_by="business",
    )

    result = execute_rule(
        asset,
        rule,
    )

    assert result.status == "failed"

    # Both X004 rows must be marked.
    assert (
        result.failed_row_indexes
        == [3, 4]
    )


# -------------------------------------------------------------------
# RANGE RULE
# -------------------------------------------------------------------

def test_range_rule_detects_outside_value(
    tmp_path,
):
    asset = load_asset(
        _create_rule_csv(
            tmp_path
        )
    )

    rule = DataQualityRule(
        rule_id="range_test",
        table_name="rule_sample",
        columns=["measurement"],
        rule_type="range",
        parameters={
            "min": 0,
            "max": 100,
        },
        severity="high",
        confidence=1.0,
        reason="Explicit test constraint.",
        generated_by="business",
    )

    result = execute_rule(
        asset,
        rule,
    )

    assert result.status == "failed"
    assert result.rows_checked == 3
    assert result.failed_rows == 1

    assert (
        result.failed_row_indexes
        == [2]
    )


# -------------------------------------------------------------------
# UNSUPPORTED RULE
# -------------------------------------------------------------------

def test_unsupported_rule_is_skipped(
    tmp_path,
):
    asset = load_asset(
        _create_rule_csv(
            tmp_path
        )
    )

    rule = DataQualityRule(
        rule_id="unsupported_test",
        table_name="rule_sample",
        columns=["abc_code"],
        rule_type="made_up_rule",
        parameters={},
        severity="low",
        confidence=1.0,
        reason="Test only.",
        generated_by="business",
    )

    result = execute_rule(
        asset,
        rule,
    )

    assert result.status == "skipped"
    assert result.failed_rows == 0


# -------------------------------------------------------------------
# PARTITIONING
# -------------------------------------------------------------------

def test_partition_rows_rejects_failed_medium_rule(
    tmp_path,
):
    asset = load_asset(
        _create_rule_csv(
            tmp_path
        )
    )

    rule = DataQualityRule(
        rule_id="email_partition",
        table_name="rule_sample",
        columns=["contact"],
        rule_type="pattern_match",
        parameters={
            "pattern_name": "email"
        },
        severity="medium",
        confidence=1.0,
        reason="Test rule.",
        generated_by="deterministic",
    )

    result = execute_rule(
        asset,
        rule,
    )

    partition = partition_rows(
        asset,
        [rule],
        [result],
    )

    assert (
        partition.valid_row_count
        == 4
    )

    assert (
        partition.rejected_row_count
        == 1
    )

    valid = (
        partition.valid_tables[
            "rule_sample"
        ]
    )

    rejected = (
        partition.invalid_tables[
            "rule_sample"
        ]
    )

    assert "X003" not in (
        valid["abc_code"].tolist()
    )

    assert (
        rejected.iloc[0][
            "abc_code"
        ]
        == "X003"
    )

    assert (
        rejected.iloc[0][
            "_dq_source_row_position"
        ]
        == 2
    )

    assert (
        rejected.iloc[0][
            "_dq_failed_rule_count"
        ]
        == 1
    )


def test_low_severity_failure_does_not_reject_row(
    tmp_path,
):
    asset = load_asset(
        _create_rule_csv(
            tmp_path
        )
    )

    rule = DataQualityRule(
        rule_id="low_email_test",
        table_name="rule_sample",
        columns=["contact"],
        rule_type="pattern_match",
        parameters={
            "pattern_name": "email"
        },
        severity="low",
        confidence=1.0,
        reason="Low severity test.",
        generated_by="deterministic",
    )

    result = execute_rule(
        asset,
        rule,
    )

    assert result.status == "failed"

    partition = partition_rows(
        asset,
        [rule],
        [result],
    )

    assert (
        partition.valid_row_count
        == 5
    )

    assert (
        partition.rejected_row_count
        == 0
    )


# -------------------------------------------------------------------
# GENERATED RULES
# -------------------------------------------------------------------

def test_generated_rules_are_grounded(
    tmp_path,
):
    asset = load_asset(
        _create_rule_csv(
            tmp_path
        )
    )

    profile = profile_asset(
        asset
    )

    semantics = analyze_semantics(
        asset,
        profile,
    )

    rules = generate_rules(
        asset,
        profile,
        semantics,
    )

    rule_types = {
        rule.rule_type
        for rule in rules
    }

    assert "pattern_match" in rule_types
    assert "datetime_parse" in rule_types
    assert "type_check" in rule_types

    # The sample contains no business document establishing a range.
    assert "range" not in rule_types

    # Nor is there evidence that event_time must never be future.
    assert "not_future" not in rule_types


# -------------------------------------------------------------------
# GENERATED RULE EXECUTION
# -------------------------------------------------------------------

def test_generated_rules_execute_without_errors(
    tmp_path,
):
    asset = load_asset(
        _create_rule_csv(
            tmp_path
        )
    )

    profile = profile_asset(
        asset
    )

    semantics = analyze_semantics(
        asset,
        profile,
    )

    rules = generate_rules(
        asset,
        profile,
        semantics,
    )

    results = execute_rules(
        asset,
        rules,
    )

    assert len(results) == len(rules)

    assert all(
        result.status
        in {
            "passed",
            "failed",
        }
        for result in results
    )

    assert not any(
        result.status
        == "error"
        for result in results
    )