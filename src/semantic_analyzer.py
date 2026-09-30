from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.llm_client import llm_client
from src.models import (
    ColumnProfile,
    ColumnSemanticInsight,
    DataAsset,
    DataProfile,
    DatasetUnderstanding,
)


# -------------------------------------------------------------------
# SEMANTIC VOCABULARY
# -------------------------------------------------------------------

ALLOWED_SEMANTIC_TYPES = {
    "identifier",
    "identifier_candidate",
    "code",
    "numeric_measure",
    "numeric_value",
    "datetime",
    "date",
    "email",
    "url",
    "ip_address",
    "boolean",
    "category",
    "free_text",
    "name",
    "address",
    "phone",
    "currency",
    "percentage",
    "unknown",
}


SEMANTIC_ALIASES = {
    "numeric": "numeric_value",
    "number": "numeric_value",
    "float": "numeric_value",
    "integer": "numeric_value",
    "decimal": "numeric_value",

    "measurement": "numeric_measure",
    "numeric_measurement": "numeric_measure",

    "timestamp": "datetime",
    "date_time": "datetime",

    "email_address": "email",

    "id": "identifier_candidate",

    "text": "free_text",
    "string": "free_text",

    "ip": "ip_address",
}


# -------------------------------------------------------------------
# JSON-SAFE VALUES
# -------------------------------------------------------------------

def _safe_value(
    value: Any,
) -> Any:
    """
    Convert pandas and NumPy values into JSON-safe Python values.
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
            str(key): _safe_value(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [
            _safe_value(item)
            for item in value
        ]

    return value


# -------------------------------------------------------------------
# COMPACT MODEL CONTEXT
# -------------------------------------------------------------------

def _compact_numeric_stats(
    profile: ColumnProfile,
) -> dict[str, Any]:
    """
    Keep only numeric statistics that may help semantic reasoning.
    """

    stats = (
        profile.numeric_statistics
    )

    if not stats:
        return {}

    keys = [
        "min",
        "max",
        "mean",
        "median",
        "negative_count",
        "zero_count",
        "mad_outlier_count",
    ]

    return {
        key: _safe_value(
            stats[key]
        )
        for key in keys
        if key in stats
    }


def _compact_datetime_stats(
    profile: ColumnProfile,
) -> dict[str, Any]:
    """
    Keep only useful datetime statistics.
    """

    stats = (
        profile.datetime_statistics
    )

    if not stats:
        return {}

    keys = [
        "earliest",
        "latest",
        "valid_datetime_count",
        "invalid_datetime_count",
    ]

    return {
        key: _safe_value(
            stats[key]
        )
        for key in keys
        if key in stats
    }


def _compact_patterns(
    profile: ColumnProfile,
) -> list[dict[str, Any]]:
    """
    Keep pattern name and match percentage only.
    """

    return [
        {
            "name": (
                pattern.pattern_name
            ),
            "match_percentage": (
                pattern.match_percentage
            ),
        }
        for pattern in profile.patterns
        if pattern.match_percentage > 0
    ]


def _build_compact_context(
    asset: DataAsset,
    profile: DataProfile,
) -> dict[str, Any]:
    """
    Build a compact evidence package for the local model.

    Detailed profiling remains in Python instead of consuming the
    model's context window.
    """

    tables: list[
        dict[str, Any]
    ] = []

    profile_lookup = {
        table.table_name: table
        for table in profile.tables
    }

    for (
        table_name,
        dataframe,
    ) in asset.tables.items():

        table_profile = (
            profile_lookup.get(
                table_name
            )
        )

        if table_profile is None:
            continue

        columns: list[
            dict[str, Any]
        ] = []

        for column_profile in (
            table_profile.columns
        ):

            columns.append(
                {
                    "name": (
                        column_profile
                        .column_name
                    ),
                    "type": (
                        column_profile
                        .inferred_type
                    ),
                    "null_percentage": (
                        column_profile
                        .null_percentage
                    ),
                    "distinct_percentage": (
                        column_profile
                        .distinct_percentage
                    ),
                    "samples": [
                        _safe_value(value)
                        for value
                        in column_profile
                        .sample_values[:5]
                    ],
                    "patterns": (
                        _compact_patterns(
                            column_profile
                        )
                    ),
                    "numeric": (
                        _compact_numeric_stats(
                            column_profile
                        )
                    ),
                    "datetime": (
                        _compact_datetime_stats(
                            column_profile
                        )
                    ),
                }
            )

        tables.append(
            {
                "name": (
                    table_name
                ),
                "rows": (
                    table_profile.row_count
                ),
                "duplicate_percentage": (
                    table_profile
                    .duplicate_percentage
                ),
                "columns": (
                    columns
                ),
            }
        )

    documents: list[
        dict[str, Any]
    ] = []

    for (
        document_name,
        text,
    ) in asset.documents.items():

        text = text or ""

        documents.append(
            {
                "name": (
                    document_name
                ),
                "excerpt": (
                    text[:2000]
                ),
            }
        )

    return {
        "source_name": (
            asset.source_path.name
        ),
        "source_type": (
            asset.source_type
        ),
        "tables": (
            tables
        ),
        "documents": (
            documents
        ),
    }


# -------------------------------------------------------------------
# MODEL INSTRUCTIONS
# -------------------------------------------------------------------

SEMANTIC_SYSTEM_PROMPT = """
You analyze the semantic meaning of unknown data.

Use only the evidence supplied.

Rules:

1. Never invent tables, columns, values, statistics or business facts.

2. Return exactly one semantic insight for every supplied column.

3. Use only real table and column names from the evidence.

4. A code-like field is not automatically a unique identifier.

5. Use "identifier" only when evidence shows:
   - 100% distinct values
   - 0% missing values

6. Otherwise use "identifier_candidate" or "code".

7. A date column does not automatically mean event logs.

8. An email column does not automatically mean customer or user data.

9. A numeric field does not automatically mean telemetry, salary,
   price or temperature.

10. If the overall business domain is not clearly supported, use:
    "unknown"

11. Do not generate data-quality rules.

12. Do not generate database mappings.

13. Do not decide that an outlier is invalid.

14. Use conservative confidence scores.

Preferred semantic types:

identifier
identifier_candidate
code
numeric_measure
numeric_value
datetime
date
email
url
ip_address
boolean
category
free_text
name
address
phone
currency
percentage
unknown
"""


# -------------------------------------------------------------------
# PROFILE LOOKUPS
# -------------------------------------------------------------------

def _find_column_profile(
    profile: DataProfile,
    table_name: str,
    column_name: str,
) -> ColumnProfile | None:
    """
    Find deterministic profile evidence for one column.
    """

    for table in profile.tables:

        if (
            table.table_name
            != table_name
        ):
            continue

        for column in table.columns:

            if (
                column.column_name
                == column_name
            ):
                return column

    return None


def _pattern_percentage(
    profile: ColumnProfile,
    pattern_name: str,
) -> float | None:
    """
    Return observed pattern coverage for one column.
    """

    for pattern in profile.patterns:

        if (
            pattern.pattern_name
            == pattern_name
        ):
            return (
                pattern
                .match_percentage
            )

    return None


# -------------------------------------------------------------------
# SEMANTIC NORMALIZATION
# -------------------------------------------------------------------

def _normalize_semantic_type(
    value: str,
) -> str:
    """
    Normalize model-generated semantic labels into our controlled
    vocabulary.
    """

    normalized = (
        value
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )

    normalized = (
        SEMANTIC_ALIASES.get(
            normalized,
            normalized,
        )
    )

    if (
        normalized
        not in ALLOWED_SEMANTIC_TYPES
    ):
        return "unknown"

    return normalized


# -------------------------------------------------------------------
# DETERMINISTIC EVIDENCE
# -------------------------------------------------------------------

def _build_profile_evidence(
    column: ColumnProfile,
) -> list[str]:
    """
    Build factual evidence directly from deterministic profiling.

    This ensures every semantic interpretation has auditable evidence
    even when the model leaves its evidence list empty.
    """

    evidence = [
        (
            "Deterministic inferred type: "
            f"{column.inferred_type}."
        ),
        (
            "Null percentage: "
            f"{column.null_percentage}%."
        ),
        (
            "Distinct percentage: "
            f"{column.distinct_percentage}%."
        ),
    ]

    for pattern in (
        column.patterns
    ):

        if (
            pattern.match_percentage
            <= 0
        ):
            continue

        evidence.append(
            (
                f"{pattern.pattern_name} pattern matched "
                f"{pattern.match_percentage}% "
                "of checked values."
            )
        )

    if column.sample_values:

        sample_text = ", ".join(
            str(
                _safe_value(
                    value
                )
            )
            for value
            in column.sample_values[:3]
        )

        evidence.append(
            (
                "Representative values: "
                f"{sample_text}."
            )
        )

    return evidence


# -------------------------------------------------------------------
# FALLBACK SEMANTIC INFERENCE
# -------------------------------------------------------------------

def _fallback_semantic(
    column: ColumnProfile | None,
) -> tuple[
    str,
    float,
    list[str],
]:
    """
    Produce a conservative interpretation when the model misses a real
    column or produces an unusable reference.
    """

    if column is None:
        return (
            "unknown",
            0.40,
            [
                (
                    "No reliable semantic evidence "
                    "was available."
                )
            ],
        )

    evidence = (
        _build_profile_evidence(
            column
        )
    )

    email_match = (
        _pattern_percentage(
            column,
            "email",
        )
        or 0.0
    )

    if email_match >= 60.0:
        return (
            "email",
            min(
                0.90,
                email_match / 100,
            ),
            evidence,
        )

    url_match = (
        _pattern_percentage(
            column,
            "url",
        )
        or 0.0
    )

    if url_match >= 80.0:
        return (
            "url",
            min(
                0.95,
                url_match / 100,
            ),
            evidence,
        )

    ip_match = (
        _pattern_percentage(
            column,
            "ip",
        )
        or 0.0
    )

    if ip_match >= 80.0:
        return (
            "ip_address",
            min(
                0.95,
                ip_match / 100,
            ),
            evidence,
        )

    uuid_match = (
        _pattern_percentage(
            column,
            "uuid",
        )
        or 0.0
    )

    if uuid_match >= 90.0:

        is_unique = (
            column.distinct_percentage
            == 100.0
            and column.null_count
            == 0
        )

        semantic_type = (
            "identifier"
            if is_unique
            else "identifier_candidate"
        )

        return (
            semantic_type,
            min(
                0.95,
                uuid_match / 100,
            ),
            evidence,
        )

    if (
        column.inferred_type
        == "datetime"
    ):
        return (
            "datetime",
            0.90,
            evidence,
        )

    if (
        column.inferred_type
        in {
            "integer",
            "float",
        }
    ):
        return (
            "numeric_value",
            0.80,
            evidence,
        )

    if (
        column.inferred_type
        == "boolean"
    ):
        return (
            "boolean",
            0.90,
            evidence,
        )

    if (
        column.inferred_type
        == "string"
    ):
        return (
            "free_text",
            0.60,
            evidence,
        )

    return (
        "unknown",
        0.40,
        evidence,
    )


# -------------------------------------------------------------------
# SELF-REPAIR OF MODEL OUTPUT
# -------------------------------------------------------------------

def _repair_output(
    asset: DataAsset,
    profile: DataProfile,
    understanding: DatasetUnderstanding,
) -> DatasetUnderstanding:
    """
    Repair incorrect table references without guessing invented or
    ambiguous columns.
    """

    real_columns = {
        table_name: {
            str(column)
            for column
            in dataframe.columns
        }
        for (
            table_name,
            dataframe,
        ) in asset.tables.items()
    }

    locations: dict[
        str,
        list[str],
    ] = {}

    for (
        table_name,
        columns,
    ) in real_columns.items():

        for column_name in columns:

            locations.setdefault(
                column_name,
                [],
            ).append(
                table_name
            )

    repaired: list[
        ColumnSemanticInsight
    ] = []

    seen: set[
        tuple[str, str]
    ] = set()

    for insight in (
        understanding.column_insights
    ):

        table_name = (
            insight.table_name
        )

        column_name = (
            insight.column_name
        )

        if (
            table_name
            in real_columns
            and column_name
            in real_columns[
                table_name
            ]
        ):
            real_table = (
                table_name
            )

        else:

            possible_tables = (
                locations.get(
                    column_name,
                    [],
                )
            )

            if (
                len(
                    possible_tables
                )
                != 1
            ):
                continue

            real_table = (
                possible_tables[0]
            )

        pair = (
            real_table,
            column_name,
        )

        if pair in seen:
            continue

        insight.table_name = (
            real_table
        )

        repaired.append(
            insight
        )

        seen.add(
            pair
        )

    # Any real column missed by the model receives a safe deterministic
    # fallback instead of causing the whole pipeline to fail.
    for (
        table_name,
        dataframe,
    ) in asset.tables.items():

        for raw_column in (
            dataframe.columns
        ):

            column_name = str(
                raw_column
            )

            pair = (
                table_name,
                column_name,
            )

            if pair in seen:
                continue

            column_profile = (
                _find_column_profile(
                    profile,
                    table_name,
                    column_name,
                )
            )

            (
                semantic_type,
                confidence,
                evidence,
            ) = (
                _fallback_semantic(
                    column_profile
                )
            )

            repaired.append(
                ColumnSemanticInsight(
                    table_name=(
                        table_name
                    ),
                    column_name=(
                        column_name
                    ),
                    semantic_type=(
                        semantic_type
                    ),
                    description=(
                        "Conservative semantic interpretation "
                        "generated from deterministic profiling."
                    ),
                    confidence=(
                        confidence
                    ),
                    evidence=(
                        evidence
                    ),
                )
            )

            seen.add(
                pair
            )

    understanding.column_insights = (
        repaired
    )

    return understanding


# -------------------------------------------------------------------
# DOMAIN GROUNDING
# -------------------------------------------------------------------

def _domain_tokens(
    domain: str,
) -> list[str]:
    """
    Extract meaningful tokens from a proposed domain.
    """

    ignored = {
        "data",
        "dataset",
        "datasets",
        "record",
        "records",
        "table",
        "tables",
        "file",
        "files",
        "information",
    }

    tokens = re.findall(
        r"[A-Za-z0-9]+",
        domain.lower(),
    )

    return [
        token
        for token in tokens
        if (
            len(token) > 2
            and token
            not in ignored
        )
    ]


def _domain_is_grounded(
    asset: DataAsset,
    domain: str,
) -> bool:
    """
    Require multiple structural clues or explicit document evidence
    before accepting a specific business domain.
    """

    normalized_domain = (
        domain
        .strip()
        .lower()
    )

    if (
        not normalized_domain
        or normalized_domain
        == "unknown"
    ):
        return True

    tokens = (
        _domain_tokens(
            normalized_domain
        )
    )

    if not tokens:
        return False

    structural_names = [
        Path(
            asset.source_path
        ).stem.lower()
    ]

    for (
        table_name,
        dataframe,
    ) in asset.tables.items():

        structural_names.append(
            table_name.lower()
        )

        structural_names.extend(
            str(column).lower()
            for column
            in dataframe.columns
        )

    structural_matches = 0

    for name in (
        structural_names
    ):

        if any(
            token in name
            for token in tokens
        ):
            structural_matches += 1

    if structural_matches >= 2:
        return True

    # Explicit text can provide stronger domain evidence.
    for text in (
        asset.documents.values()
    ):

        lowered_text = (
            text or ""
        ).lower()

        matching_tokens = sum(
            1
            for token in tokens
            if token in lowered_text
        )

        if matching_tokens >= 1:
            return True

    return False


def _build_generic_description(
    understanding: DatasetUnderstanding,
) -> str:
    """
    Produce a useful factual description when the business domain is
    unknown.
    """

    if not (
        understanding
        .column_insights
    ):
        return (
            "Dataset with no structured columns "
            "available for semantic classification."
        )

    parts = [
        (
            f"{insight.column_name} "
            f"({insight.semantic_type})"
        )
        for insight
        in understanding.column_insights
    ]

    return (
        "Structured dataset containing "
        + ", ".join(parts)
        + "."
    )


# -------------------------------------------------------------------
# DETERMINISTIC GUARDRAILS
# -------------------------------------------------------------------

def _apply_guardrails(
    asset: DataAsset,
    profile: DataProfile,
    understanding: DatasetUnderstanding,
) -> DatasetUnderstanding:
    """
    Apply deterministic safeguards after semantic inference.

    The model proposes meaning, while deterministic profiling remains
    the authority for measurable facts.
    """

    for insight in (
        understanding.column_insights
    ):

        insight.semantic_type = (
            _normalize_semantic_type(
                insight.semantic_type
            )
        )

        column = (
            _find_column_profile(
                profile,
                insight.table_name,
                insight.column_name,
            )
        )

        if column is None:
            continue

        # Always attach deterministic evidence.
        profile_evidence = (
            _build_profile_evidence(
                column
            )
        )

        for evidence_item in (
            profile_evidence
        ):

            if (
                evidence_item
                not in insight.evidence
            ):
                insight.evidence.append(
                    evidence_item
                )

        # Strong identifiers must actually be complete and unique.
        if (
            insight.semantic_type
            == "identifier"
        ):

            is_unique = (
                column.distinct_percentage
                == 100.0
                and column.null_count
                == 0
            )

            if not is_unique:

                insight.semantic_type = (
                    "identifier_candidate"
                )

                insight.confidence = min(
                    insight.confidence,
                    0.85,
                )

                uniqueness_evidence = (
                    "The field cannot be treated as a strong "
                    "identifier because deterministic profiling "
                    "does not establish both complete uniqueness "
                    "and zero missing values."
                )

                if (
                    uniqueness_evidence
                    not in insight.evidence
                ):
                    insight.evidence.append(
                        uniqueness_evidence
                    )

        # Email confidence cannot exceed observed structural coverage.
        if (
            insight.semantic_type
            == "email"
        ):

            match = (
                _pattern_percentage(
                    column,
                    "email",
                )
            )

            if match is None:

                insight.confidence = min(
                    insight.confidence,
                    0.60,
                )

            else:

                insight.confidence = min(
                    insight.confidence,
                    max(
                        0.50,
                        match / 100,
                    ),
                )

        # Datetime semantics must agree with deterministic inference.
        if (
            insight.semantic_type
            in {
                "datetime",
                "date",
            }
            and column.inferred_type
            != "datetime"
        ):

            insight.confidence = min(
                insight.confidence,
                0.60,
            )

        # Numeric semantics must agree with deterministic inference.
        if (
            insight.semantic_type
            in {
                "numeric_measure",
                "numeric_value",
                "currency",
                "percentage",
            }
            and column.inferred_type
            not in {
                "integer",
                "float",
            }
        ):

            insight.confidence = min(
                insight.confidence,
                0.60,
            )

    # Reject unsupported specific business domains.
    if not (
        _domain_is_grounded(
            asset,
            understanding.detected_domain,
        )
    ):
        understanding.detected_domain = (
            "unknown"
        )

    # Unknown domains should always receive a useful factual
    # description instead of filenames or speculative context.
    if (
        understanding.detected_domain
        .strip()
        .lower()
        == "unknown"
    ):
        understanding.dataset_description = (
            _build_generic_description(
                understanding
            )
        )

    return understanding


# -------------------------------------------------------------------
# OUTPUT VALIDATION
# -------------------------------------------------------------------

def _validate_output(
    asset: DataAsset,
    understanding: DatasetUnderstanding,
) -> None:
    """
    Ensure every real structured column appears exactly once.
    """

    expected = {
        (
            table_name,
            str(column),
        )
        for (
            table_name,
            dataframe,
        ) in asset.tables.items()
        for column in dataframe.columns
    }

    received: set[
        tuple[str, str]
    ] = set()

    for insight in (
        understanding.column_insights
    ):

        pair = (
            insight.table_name,
            insight.column_name,
        )

        if pair not in expected:
            raise ValueError(
                "Semantic output contains "
                f"an invalid reference: {pair}"
            )

        if pair in received:
            raise ValueError(
                "Semantic output contains "
                f"a duplicate reference: {pair}"
            )

        received.add(
            pair
        )

    missing = (
        expected
        - received
    )

    if missing:

        readable = ", ".join(
            f"{table}.{column}"
            for (
                table,
                column,
            ) in sorted(
                missing
            )
        )

        raise ValueError(
            "Semantic analysis is missing: "
            f"{readable}"
        )


# -------------------------------------------------------------------
# MAIN SEMANTIC ANALYSIS
# -------------------------------------------------------------------

def analyze_semantics(
    asset: DataAsset,
    profile: DataProfile,
) -> DatasetUnderstanding:
    """
    Perform compact semantic analysis using the local LLM.

    Model output is repaired, validated and constrained using
    deterministic profiling before being returned.
    """

    context = (
        _build_compact_context(
            asset,
            profile,
        )
    )

    prompt = (
        "Analyze this data source.\n\n"
        "Determine a conservative dataset domain and one semantic "
        "type for every structured column.\n\n"
        "Use 'unknown' for the domain when evidence is insufficient.\n\n"
        "Do not invent tables, columns, statistics or business context.\n\n"
        "Return semantic interpretations only. Do not generate data "
        "quality rules or mappings.\n\n"
        "EVIDENCE:\n"
        + json.dumps(
            context,
            ensure_ascii=False,
            default=str,
            separators=(
                ",",
                ":",
            ),
        )
    )

    understanding = (
        llm_client.chat_structured(
            system_prompt=(
                SEMANTIC_SYSTEM_PROMPT
            ),
            user_prompt=(
                prompt
            ),
            response_model=(
                DatasetUnderstanding
            ),
            retries=1,
        )
    )

    # Safely repair incorrect table references and fill model omissions.
    understanding = (
        _repair_output(
            asset,
            profile,
            understanding,
        )
    )

    # Ensure every result references the real loaded schema.
    _validate_output(
        asset,
        understanding,
    )

    # Apply final deterministic grounding.
    understanding = (
        _apply_guardrails(
            asset,
            profile,
            understanding,
        )
    )

    return understanding