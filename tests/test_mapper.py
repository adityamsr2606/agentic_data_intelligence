from __future__ import annotations

from src.ingestion import load_asset
from src.mapper import (
    MappingDecision,
    generate_mappings,
)
from src.models import (
    ColumnSemanticInsight,
    DatasetUnderstanding,
    TargetColumn,
    TargetSchema,
    TargetTable,
)
from src.profiler import profile_asset


# -------------------------------------------------------------------
# TEST DATA
# -------------------------------------------------------------------

def _create_mapping_csv(tmp_path):
    source = tmp_path / "mapping_sample.csv"

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


def _semantics():
    return DatasetUnderstanding(
        detected_domain="unknown",
        dataset_description=(
            "Structured test dataset."
        ),
        column_insights=[
            ColumnSemanticInsight(
                table_name="mapping_sample",
                column_name="abc_code",
                semantic_type="code",
                description="",
                confidence=0.8,
                evidence=[
                    "String code values."
                ],
            ),
            ColumnSemanticInsight(
                table_name="mapping_sample",
                column_name="measurement",
                semantic_type="numeric_value",
                description="",
                confidence=0.9,
                evidence=[
                    "Numeric values."
                ],
            ),
            ColumnSemanticInsight(
                table_name="mapping_sample",
                column_name="event_time",
                semantic_type="datetime",
                description="",
                confidence=0.9,
                evidence=[
                    "Datetime values."
                ],
            ),
            ColumnSemanticInsight(
                table_name="mapping_sample",
                column_name="contact",
                semantic_type="email",
                description="",
                confidence=0.7,
                evidence=[
                    "Email-like values."
                ],
            ),
        ],
    )


def _real_target_schema():
    return TargetSchema(
        tables=[
            TargetTable(
                table_name="quality_records",
                columns=[
                    TargetColumn(
                        column_name="abc_code",
                        data_type="VARCHAR",
                        nullable=False,
                        primary_key=False,
                        description="",
                    ),
                    TargetColumn(
                        column_name="measurement_value",
                        data_type="DOUBLE PRECISION",
                        nullable=True,
                        primary_key=False,
                        description="numeric measurement",
                    ),
                    TargetColumn(
                        column_name="event_time",
                        data_type="TIMESTAMP",
                        nullable=False,
                        primary_key=False,
                        description="",
                    ),
                    TargetColumn(
                        column_name="contact_email",
                        data_type="VARCHAR",
                        nullable=True,
                        primary_key=False,
                        description="email address",
                    ),
                ],
            )
        ]
    )


def _mapping_for(
    mappings,
    source_column: str,
):
    return next(
        mapping
        for mapping in mappings
        if (
            mapping.source_column
            == source_column
        )
    )


# -------------------------------------------------------------------
# EXACT MATCHES
# -------------------------------------------------------------------

def test_exact_name_matches_are_automatic(
    tmp_path,
    monkeypatch,
):
    asset = load_asset(
        _create_mapping_csv(
            tmp_path
        )
    )

    profile = profile_asset(
        asset
    )

    # Ambiguous fields do not need an actual LLM call for this test.
    monkeypatch.setattr(
        "src.mapper._resolve_with_ai",
        lambda payloads: [],
    )

    mappings = generate_mappings(
        asset,
        profile,
        _semantics(),
        _real_target_schema(),
    )

    abc = _mapping_for(
        mappings,
        "abc_code",
    )

    event = _mapping_for(
        mappings,
        "event_time",
    )

    assert (
        abc.target_table
        == "quality_records"
    )

    assert (
        abc.target_column
        == "abc_code"
    )

    assert (
        abc.requires_review
        is False
    )

    assert (
        abc.transformation
        == "direct"
    )

    assert (
        event.target_column
        == "event_time"
    )

    assert (
        event.requires_review
        is False
    )


# -------------------------------------------------------------------
# AMBIGUOUS FALLBACK
# -------------------------------------------------------------------

def test_ambiguous_mapping_requires_review(
    tmp_path,
    monkeypatch,
):
    asset = load_asset(
        _create_mapping_csv(
            tmp_path
        )
    )

    profile = profile_asset(
        asset
    )

    monkeypatch.setattr(
        "src.mapper._resolve_with_ai",
        lambda payloads: [],
    )

    mappings = generate_mappings(
        asset,
        profile,
        _semantics(),
        _real_target_schema(),
    )

    contact = _mapping_for(
        mappings,
        "contact",
    )

    measurement = _mapping_for(
        mappings,
        "measurement",
    )

    assert (
        contact.target_column
        == "contact_email"
    )

    assert (
        contact.requires_review
        is True
    )

    assert (
        measurement.target_column
        == "measurement_value"
    )

    assert (
        measurement.requires_review
        is True
    )


# -------------------------------------------------------------------
# AI CANDIDATE VALIDATION
# -------------------------------------------------------------------

def test_ai_cannot_invent_target_column(
    tmp_path,
    monkeypatch,
):
    asset = load_asset(
        _create_mapping_csv(
            tmp_path
        )
    )

    profile = profile_asset(
        asset
    )

    def fake_ai(
        payloads,
    ):
        return [
            MappingDecision(
                source_table="mapping_sample",
                source_column="contact",
                should_map=True,
                target_table="quality_records",
                target_column="invented_email_column",
                confidence=1.0,
                reason=(
                    "Invented target for test."
                ),
            )
        ]

    monkeypatch.setattr(
        "src.mapper._resolve_with_ai",
        fake_ai,
    )

    mappings = generate_mappings(
        asset,
        profile,
        _semantics(),
        _real_target_schema(),
    )

    contact = _mapping_for(
        mappings,
        "contact",
    )

    # The invented target must be discarded. The guarded deterministic
    # fallback may only return a real candidate.
    assert (
        contact.target_column
        == "contact_email"
    )

    assert (
        contact.target_column
        != "invented_email_column"
    )

    assert (
        contact.requires_review
        is True
    )


# -------------------------------------------------------------------
# AI MAY CHOOSE ONLY A REAL CANDIDATE
# -------------------------------------------------------------------

def test_valid_ai_choice_is_still_guarded(
    tmp_path,
    monkeypatch,
):
    asset = load_asset(
        _create_mapping_csv(
            tmp_path
        )
    )

    profile = profile_asset(
        asset
    )

    def fake_ai(
        payloads,
    ):
        decisions = []

        for payload in payloads:
            if (
                payload[
                    "source_column"
                ]
                == "contact"
            ):
                decisions.append(
                    MappingDecision(
                        source_table=(
                            "mapping_sample"
                        ),
                        source_column=(
                            "contact"
                        ),
                        should_map=True,
                        target_table=(
                            "quality_records"
                        ),
                        target_column=(
                            "contact_email"
                        ),
                        confidence=0.95,
                        reason=(
                            "Email semantic evidence "
                            "supports this candidate."
                        ),
                    )
                )

        return decisions

    monkeypatch.setattr(
        "src.mapper._resolve_with_ai",
        fake_ai,
    )

    mappings = generate_mappings(
        asset,
        profile,
        _semantics(),
        _real_target_schema(),
    )

    contact = _mapping_for(
        mappings,
        "contact",
    )

    assert (
        contact.target_column
        == "contact_email"
    )

    # AI confidence alone is intentionally not enough to bypass the
    # deterministic evidence threshold.
    assert (
        contact.confidence
        < 0.95
    )


# -------------------------------------------------------------------
# NO TARGET SCHEMA
# -------------------------------------------------------------------

def test_empty_target_schema_returns_no_mappings(
    tmp_path,
):
    asset = load_asset(
        _create_mapping_csv(
            tmp_path
        )
    )

    profile = profile_asset(
        asset
    )

    mappings = generate_mappings(
        asset,
        profile,
        _semantics(),
        TargetSchema(
            tables=[]
        ),
    )

    assert mappings == []


# -------------------------------------------------------------------
# MAPPINGS REFERENCE ONLY REAL TARGETS
# -------------------------------------------------------------------

def test_all_generated_targets_exist(
    tmp_path,
    monkeypatch,
):
    asset = load_asset(
        _create_mapping_csv(
            tmp_path
        )
    )

    profile = profile_asset(
        asset
    )

    target = (
        _real_target_schema()
    )

    monkeypatch.setattr(
        "src.mapper._resolve_with_ai",
        lambda payloads: [],
    )

    mappings = generate_mappings(
        asset,
        profile,
        _semantics(),
        target,
    )

    valid_targets = {
        (
            table.table_name,
            column.column_name,
        )
        for table in target.tables
        for column in table.columns
    }

    assert mappings

    assert all(
        (
            mapping.target_table,
            mapping.target_column,
        )
        in valid_targets
        for mapping in mappings
    )