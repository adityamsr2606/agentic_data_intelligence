from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from pydantic import BaseModel, Field

from src.llm_client import LLMError, llm_client
from src.models import (
    ColumnProfile,
    DataAsset,
    DataProfile,
    DataQualityRule,
    DatasetUnderstanding,
)


# -------------------------------------------------------------------
# SUPPORTED RULE VOCABULARY
# -------------------------------------------------------------------

ALLOWED_RULE_TYPES = {
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


ALLOWED_EXPECTED_TYPES = {
    "string",
    "integer",
    "number",
    "boolean",
    "datetime",
}


ALLOWED_PATTERNS = {
    "email",
    "url",
    "uuid",
    "ip",
    "integer_string",
    "decimal_string",

    # Internal deterministic structural pattern.
    # It is intentionally broad: a name-like value only needs to
    # contain at least one alphabetic character.
    "contains_letters",
}


PATTERN_ALIASES = {
    "ip_address": "ip",
    "integer": "integer_string",
    "decimal": "decimal_string",
}


# These rules can represent real business requirements and therefore
# require explicit supporting evidence before an AI proposal is kept.
BUSINESS_SENSITIVE_RULES = {
    "not_null",
    "unique",
    "range",
    "allowed_values",
    "length",
    "not_future",
    "duplicate_rows",
}


MAX_AGENTIC_RULES = 8


# -------------------------------------------------------------------
# SMALL AI RESPONSE MODEL
# -------------------------------------------------------------------

class RuleProposal(BaseModel):
    """
    Lightweight rule suggestion returned by the local model.

    The model does not control rule IDs, severity or generated_by.
    """

    table_name: str

    columns: list[str] = Field(
        min_length=1,
        max_length=8,
    )

    rule_type: str

    parameters: dict[
        str,
        Any,
    ] = Field(
        default_factory=dict
    )

    confidence: float = Field(
        ge=0.0,
        le=1.0,
    )

    reason: str = ""


class RuleProposalResponse(BaseModel):
    """
    Compact structured response returned by Qwen.
    """

    rules: list[
        RuleProposal
    ] = Field(
        default_factory=list,
        max_length=MAX_AGENTIC_RULES,
    )


# -------------------------------------------------------------------
# PROFILE HELPERS
# -------------------------------------------------------------------

def _find_column_profile(
    profile: DataProfile,
    table_name: str,
    column_name: str,
) -> ColumnProfile | None:
    """
    Find deterministic profiling evidence for a real column.
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


def _semantic_lookup(
    understanding: DatasetUnderstanding,
) -> dict[
    tuple[str, str],
    tuple[str, float],
]:
    """
    Map each real column to semantic type and confidence.
    """

    return {
        (
            insight.table_name,
            insight.column_name,
        ): (
            insight.semantic_type,
            insight.confidence,
        )
        for insight
        in understanding.column_insights
    }


def _compact_patterns(
    column: ColumnProfile,
) -> list[dict[str, Any]]:
    """
    Keep only meaningful pattern evidence.
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
        for pattern
        in column.patterns
        if pattern.checked_values > 0
    ]


# -------------------------------------------------------------------
# COMPACT AI CONTEXT
# -------------------------------------------------------------------

def _build_rule_context(
    asset: DataAsset,
    profile: DataProfile,
    understanding: DatasetUnderstanding,
) -> dict[str, Any]:
    """
    Build a small evidence package for the local model.

    Detailed profiling remains inside Python.
    """

    semantic_map = (
        _semantic_lookup(
            understanding
        )
    )

    tables: list[
        dict[str, Any]
    ] = []

    for table in profile.tables:

        columns: list[
            dict[str, Any]
        ] = []

        for column in table.columns:

            (
                semantic_type,
                semantic_confidence,
            ) = semantic_map.get(
                (
                    table.table_name,
                    column.column_name,
                ),
                (
                    "unknown",
                    0.0,
                ),
            )

            columns.append(
                {
                    "name": (
                        column.column_name
                    ),
                    "type": (
                        column.inferred_type
                    ),
                    "null_pct": (
                        column.null_percentage
                    ),
                    "distinct_pct": (
                        column.distinct_percentage
                    ),
                    "semantic": (
                        semantic_type
                    ),
                    "semantic_confidence": (
                        semantic_confidence
                    ),
                    "patterns": (
                        _compact_patterns(
                            column
                        )
                    ),
                    "samples": (
                        column.sample_values[:3]
                    ),
                }
            )

        tables.append(
            {
                "name": (
                    table.table_name
                ),
                "rows": (
                    table.row_count
                ),
                "duplicate_pct": (
                    table.duplicate_percentage
                ),
                "columns": (
                    columns
                ),
            }
        )

    documents: list[
        dict[str, str]
    ] = []

    for (
        document_name,
        text,
    ) in asset.documents.items():

        text = (
            text
            or ""
        )

        if not text.strip():
            continue

        documents.append(
            {
                "name": (
                    document_name
                ),
                "excerpt": (
                    text[:1200]
                ),
            }
        )

    return {
        "domain": (
            understanding.detected_domain
        ),
        "tables": tables,
        "documents": documents,
    }


# -------------------------------------------------------------------
# RULE CREATION
# -------------------------------------------------------------------

def _make_rule(
    *,
    table_name: str,
    columns: list[str],
    rule_type: str,
    parameters: dict[str, Any],
    severity: str,
    confidence: float,
    reason: str,
    generated_by: str,
) -> DataQualityRule:
    """
    Create a rule before its stable ID is assigned.
    """

    return DataQualityRule(
        rule_id="pending",
        table_name=table_name,
        columns=columns,
        rule_type=rule_type,
        parameters=parameters,
        severity=severity,
        confidence=confidence,
        reason=reason,
        generated_by=generated_by,
    )


# -------------------------------------------------------------------
# DETERMINISTIC STRUCTURAL RULES
# -------------------------------------------------------------------

def _deterministic_structural_rules(
    profile: DataProfile,
    understanding: DatasetUnderstanding,
) -> list[DataQualityRule]:
    """
    Generate rules that follow directly from strong semantic evidence.

    No dataset-specific column names are hard-coded.
    """

    rules: list[
        DataQualityRule
    ] = []

    semantic_map = (
        _semantic_lookup(
            understanding
        )
    )

    for table in profile.tables:

        for column in table.columns:

            (
                semantic_type,
                semantic_confidence,
            ) = semantic_map.get(
                (
                    table.table_name,
                    column.column_name,
                ),
                (
                    "unknown",
                    0.0,
                ),
            )

            if semantic_type == "email":

                rules.append(
                    _make_rule(
                        table_name=(
                            table.table_name
                        ),
                        columns=[
                            column.column_name
                        ],
                        rule_type=(
                            "pattern_match"
                        ),
                        parameters={
                            "pattern_name": "email"
                        },
                        severity="medium",
                        confidence=max(
                            0.60,
                            semantic_confidence,
                        ),
                        reason=(
                            "The column is classified as "
                            "email data, so email-format "
                            "validation is relevant."
                        ),
                        generated_by=(
                            "deterministic"
                        ),
                    )
                )

            elif semantic_type == "url":

                rules.append(
                    _make_rule(
                        table_name=(
                            table.table_name
                        ),
                        columns=[
                            column.column_name
                        ],
                        rule_type=(
                            "pattern_match"
                        ),
                        parameters={
                            "pattern_name": "url"
                        },
                        severity="medium",
                        confidence=max(
                            0.60,
                            semantic_confidence,
                        ),
                        reason=(
                            "The column is classified "
                            "as URL data."
                        ),
                        generated_by=(
                            "deterministic"
                        ),
                    )
                )

            elif semantic_type == "ip_address":

                rules.append(
                    _make_rule(
                        table_name=(
                            table.table_name
                        ),
                        columns=[
                            column.column_name
                        ],
                        rule_type=(
                            "pattern_match"
                        ),
                        parameters={
                            "pattern_name": "ip"
                        },
                        severity="medium",
                        confidence=max(
                            0.60,
                            semantic_confidence,
                        ),
                        reason=(
                            "The column is classified "
                            "as IP-address data."
                        ),
                        generated_by=(
                            "deterministic"
                        ),
                    )
                )

            elif (
                semantic_type == "name"
                and semantic_confidence >= 0.75
            ):

                rules.append(
                    _make_rule(
                        table_name=(
                            table.table_name
                        ),
                        columns=[
                            column.column_name
                        ],
                        rule_type=(
                            "pattern_match"
                        ),
                        parameters={
                            "pattern_name": (
                                "contains_letters"
                            )
                        },
                        severity="medium",
                        confidence=(
                            semantic_confidence
                        ),
                        reason=(
                            "The column is strongly classified as "
                            "name-like data. Values must therefore "
                            "contain at least one alphabetic "
                            "character. This deliberately avoids a "
                            "language-specific name format."
                        ),
                        generated_by=(
                            "deterministic"
                        ),
                    )
                )

            elif semantic_type in {
                "datetime",
                "date",
            }:

                rules.append(
                    _make_rule(
                        table_name=(
                            table.table_name
                        ),
                        columns=[
                            column.column_name
                        ],
                        rule_type=(
                            "datetime_parse"
                        ),
                        parameters={},
                        severity="medium",
                        confidence=max(
                            0.70,
                            semantic_confidence,
                        ),
                        reason=(
                            "The column is classified "
                            "as date/time data."
                        ),
                        generated_by=(
                            "deterministic"
                        ),
                    )
                )

            elif semantic_type in {
                "numeric_measure",
                "numeric_value",
                "currency",
                "percentage",
            }:

                rules.append(
                    _make_rule(
                        table_name=(
                            table.table_name
                        ),
                        columns=[
                            column.column_name
                        ],
                        rule_type=(
                            "type_check"
                        ),
                        parameters={
                            "expected_type": "number"
                        },
                        severity="medium",
                        confidence=max(
                            0.70,
                            semantic_confidence,
                        ),
                        reason=(
                            "The semantic interpretation "
                            "indicates numeric data."
                        ),
                        generated_by=(
                            "deterministic"
                        ),
                    )
                )

            elif semantic_type == "boolean":

                rules.append(
                    _make_rule(
                        table_name=(
                            table.table_name
                        ),
                        columns=[
                            column.column_name
                        ],
                        rule_type=(
                            "type_check"
                        ),
                        parameters={
                            "expected_type": "boolean"
                        },
                        severity="medium",
                        confidence=max(
                            0.70,
                            semantic_confidence,
                        ),
                        reason=(
                            "The semantic interpretation "
                            "indicates boolean data."
                        ),
                        generated_by=(
                            "deterministic"
                        ),
                    )
                )

    return rules


# -------------------------------------------------------------------
# AGENTIC RULE GENERATION
# -------------------------------------------------------------------

RULE_GENERATION_PROMPT = """
Suggest a small number of evidence-backed data-quality checks.

You are proposing checks, not declaring business requirements.

Allowed rule types:

type_check
pattern_match
not_null
unique
range
allowed_values
length
datetime_parse
not_future
duplicate_rows

Parameter formats:

type_check:
{"expected_type":"string|integer|number|boolean|datetime"}

pattern_match:
{"pattern_name":"email|url|uuid|ip|integer_string|decimal_string"}

not_null:
{}

unique:
{}

range:
{"min":number|null,"max":number|null,"inclusive":true}

allowed_values:
{"values":[...]}

length:
{"min_length":integer|null,"max_length":integer|null}

datetime_parse:
{}

not_future:
{}

duplicate_rows:
{}

Rules:

- Use only real tables and columns.
- Never generate Python, SQL or regex code.
- Do not turn observed minimum or maximum values into range rules.
- Do not infer allowed values from sample values.
- Do not infer length constraints from sample lengths.
- Zero current nulls do not prove that a field is required.
- High cardinality does not prove uniqueness.
- Outliers do not automatically represent invalid data.
- Dates do not automatically require not_future.
- Business constraints require explicit source/document evidence.
- Prefer structural checks with strong semantic evidence.
- Return fewer rules when evidence is weak.
- Maximum 8 proposals.
"""


def _generate_agentic_proposals(
    asset: DataAsset,
    profile: DataProfile,
    understanding: DatasetUnderstanding,
) -> list[RuleProposal]:
    """
    Ask Qwen only for lightweight rule proposals.
    """

    context = (
        _build_rule_context(
            asset,
            profile,
            understanding,
        )
    )

    prompt = (
        "Suggest useful data-quality checks supported "
        "by this evidence.\n"
        "Do not repeat equivalent checks.\n\n"
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

    response = (
        llm_client.chat_structured(
            system_prompt=(
                RULE_GENERATION_PROMPT
            ),
            user_prompt=(
                prompt
            ),
            response_model=(
                RuleProposalResponse
            ),
            retries=0,
            num_predict=450,
        )
    )

    return (
        response.rules[
            :MAX_AGENTIC_RULES
        ]
    )


# -------------------------------------------------------------------
# PARAMETER VALIDATION
# -------------------------------------------------------------------

def _is_number(
    value: Any,
) -> bool:
    return (
        isinstance(
            value,
            (
                int,
                float,
            ),
        )
        and not isinstance(
            value,
            bool,
        )
    )


def _validate_parameters(
    rule_type: str,
    raw_parameters: dict[str, Any],
) -> dict[str, Any] | None:
    """
    Validate every parameter before a rule becomes executable.
    """

    parameters = dict(
        raw_parameters
        or {}
    )

    if rule_type == "type_check":

        expected_type = str(
            parameters.get(
                "expected_type",
                "",
            )
        ).strip().lower()

        if (
            expected_type
            not in ALLOWED_EXPECTED_TYPES
        ):
            return None

        return {
            "expected_type": expected_type
        }

    if rule_type == "pattern_match":

        pattern_name = str(
            parameters.get(
                "pattern_name",
                "",
            )
        ).strip().lower()

        pattern_name = (
            PATTERN_ALIASES.get(
                pattern_name,
                pattern_name,
            )
        )

        if (
            pattern_name
            not in ALLOWED_PATTERNS
        ):
            return None

        return {
            "pattern_name": pattern_name
        }

    if rule_type in {
        "not_null",
        "unique",
        "datetime_parse",
        "not_future",
        "duplicate_rows",
    }:
        return {}

    if rule_type == "range":

        minimum = (
            parameters.get(
                "min"
            )
        )

        maximum = (
            parameters.get(
                "max"
            )
        )

        if (
            minimum is None
            and maximum is None
        ):
            return None

        if (
            minimum is not None
            and not _is_number(
                minimum
            )
        ):
            return None

        if (
            maximum is not None
            and not _is_number(
                maximum
            )
        ):
            return None

        if (
            minimum is not None
            and maximum is not None
            and minimum > maximum
        ):
            return None

        inclusive = (
            parameters.get(
                "inclusive",
                True,
            )
        )

        if not isinstance(
            inclusive,
            bool,
        ):
            return None

        return {
            "min": minimum,
            "max": maximum,
            "inclusive": inclusive,
        }

    if rule_type == "allowed_values":

        values = (
            parameters.get(
                "values"
            )
        )

        if (
            not isinstance(
                values,
                list,
            )
            or not values
            or len(values) > 100
        ):
            return None

        for value in values:

            if isinstance(
                value,
                (
                    dict,
                    list,
                    set,
                    tuple,
                ),
            ):
                return None

        return {
            "values": values
        }

    if rule_type == "length":

        minimum = (
            parameters.get(
                "min_length"
            )
        )

        maximum = (
            parameters.get(
                "max_length"
            )
        )

        if (
            minimum is None
            and maximum is None
        ):
            return None

        if (
            minimum is not None
            and (
                not isinstance(
                    minimum,
                    int,
                )
                or isinstance(
                    minimum,
                    bool,
                )
                or minimum < 0
            )
        ):
            return None

        if (
            maximum is not None
            and (
                not isinstance(
                    maximum,
                    int,
                )
                or isinstance(
                    maximum,
                    bool,
                )
                or maximum < 0
            )
        ):
            return None

        if (
            minimum is not None
            and maximum is not None
            and minimum > maximum
        ):
            return None

        return {
            "min_length": minimum,
            "max_length": maximum,
        }

    return None


# -------------------------------------------------------------------
# REFERENCE VALIDATION
# -------------------------------------------------------------------

def _validate_reference(
    asset: DataAsset,
    table_name: str,
    columns: list[str],
    rule_type: str,
) -> bool:
    """
    Reject invented tables and columns.
    """

    dataframe = (
        asset.tables.get(
            table_name
        )
    )

    if dataframe is None:
        return False

    if not columns:
        return False

    if (
        rule_type
        != "duplicate_rows"
        and len(columns) != 1
    ):
        return False

    if (
        len(
            set(
                columns
            )
        )
        != len(
            columns
        )
    ):
        return False

    valid_columns = {
        str(column)
        for column
        in dataframe.columns
    }

    return all(
        column in valid_columns
        for column
        in columns
    )


# -------------------------------------------------------------------
# BUSINESS-EVIDENCE VALIDATION
# -------------------------------------------------------------------

def _document_text(
    asset: DataAsset,
) -> str:
    """
    Combine supplied document text for deterministic evidence checks.
    """

    return "\n".join(
        (
            text
            or ""
        ).lower()
        for text
        in asset.documents.values()
        if (
            text
            and text.strip()
        )
    )


def _column_tokens(
    columns: list[str],
) -> list[str]:
    """
    Extract useful words from column names for generic document matching.
    """

    tokens: list[str] = []

    for column in columns:

        parts = re.findall(
            r"[a-zA-Z0-9]+",
            column.lower(),
        )

        tokens.extend(
            part
            for part in parts
            if len(part) >= 3
        )

    return tokens


def _document_mentions_columns(
    document: str,
    columns: list[str],
) -> bool:
    """
    Require at least one relevant column reference in supporting text.
    """

    tokens = (
        _column_tokens(
            columns
        )
    )

    if not tokens:
        return False

    return any(
        token in document
        for token in tokens
    )


def _numbers_in_document(
    document: str,
) -> set[str]:
    """
    Extract normalized numeric tokens from documentation.
    """

    return set(
        re.findall(
            r"-?\d+(?:\.\d+)?",
            document,
        )
    )


def _explicit_business_support(
    asset: DataAsset,
    proposal: RuleProposal,
    parameters: dict[str, Any],
) -> bool:
    """
    Determine whether an AI-proposed business constraint has explicit
    supporting evidence in supplied documentation.

    Current observed data alone is never enough.
    """

    if (
        proposal.rule_type
        not in BUSINESS_SENSITIVE_RULES
    ):
        return True

    document = (
        _document_text(
            asset
        )
    )

    if not document:
        return False

    column_mentioned = (
        _document_mentions_columns(
            document,
            proposal.columns,
        )
    )

    if not column_mentioned:
        return False

    rule_type = (
        proposal.rule_type
    )

    if rule_type == "not_null":

        terms = {
            "required",
            "mandatory",
            "must not be null",
            "cannot be null",
            "not null",
            "must be present",
        }

        return any(
            term in document
            for term in terms
        )

    if rule_type == "unique":

        terms = {
            "unique",
            "primary key",
            "no duplicates",
            "must be distinct",
            "cannot repeat",
        }

        return any(
            term in document
            for term in terms
        )

    if rule_type == "not_future":

        terms = {
            "must not be future",
            "cannot be future",
            "not in the future",
            "past dates only",
            "historical dates only",
        }

        return any(
            term in document
            for term in terms
        )

    if rule_type == "duplicate_rows":

        terms = {
            "no duplicate",
            "no duplicates",
            "duplicate rows are not allowed",
            "rows must be unique",
        }

        return any(
            term in document
            for term in terms
        )

    if rule_type == "range":

        document_numbers = (
            _numbers_in_document(
                document
            )
        )

        minimum = (
            parameters.get(
                "min"
            )
        )

        maximum = (
            parameters.get(
                "max"
            )
        )

        required_numbers: list[
            str
        ] = []

        if minimum is not None:
            required_numbers.append(
                str(minimum)
            )

        if maximum is not None:
            required_numbers.append(
                str(maximum)
            )

        if not required_numbers:
            return False

        return all(
            value in document_numbers
            for value in required_numbers
        )

    if rule_type == "length":

        document_numbers = (
            _numbers_in_document(
                document
            )
        )

        minimum = (
            parameters.get(
                "min_length"
            )
        )

        maximum = (
            parameters.get(
                "max_length"
            )
        )

        required_numbers = []

        if minimum is not None:
            required_numbers.append(
                str(minimum)
            )

        if maximum is not None:
            required_numbers.append(
                str(maximum)
            )

        length_terms = {
            "length",
            "characters",
            "character",
            "digits",
            "digit",
        }

        has_length_language = any(
            term in document
            for term in length_terms
        )

        return (
            has_length_language
            and bool(
                required_numbers
            )
            and all(
                value in document_numbers
                for value in required_numbers
            )
        )

    if rule_type == "allowed_values":

        values = (
            parameters.get(
                "values",
                [],
            )
        )

        if not values:
            return False

        return all(
            str(
                value
            ).lower()
            in document
            for value in values
        )

    return False


# -------------------------------------------------------------------
# DETERMINISTIC SEVERITY
# -------------------------------------------------------------------

def _assign_severity(
    rule_type: str,
) -> str:
    """
    Severity is controlled by Python rather than the LLM.
    """

    if rule_type in {
        "unique",
        "not_null",
    }:
        return "high"

    if rule_type in {
        "type_check",
        "pattern_match",
        "range",
        "allowed_values",
        "length",
        "datetime_parse",
        "not_future",
        "duplicate_rows",
    }:
        return "medium"

    return "low"


# -------------------------------------------------------------------
# AI PROPOSAL -> SAFE RULE
# -------------------------------------------------------------------

def _proposal_to_rule(
    asset: DataAsset,
    proposal: RuleProposal,
) -> DataQualityRule | None:
    """
    Convert a lightweight AI proposal into a validated application rule.
    """

    rule_type = (
        proposal.rule_type
        .strip()
        .lower()
    )

    if (
        rule_type
        not in ALLOWED_RULE_TYPES
    ):
        return None

    table_name = (
        proposal.table_name
        .strip()
    )

    columns = [
        column.strip()
        for column
        in proposal.columns
        if column.strip()
    ]

    if not (
        _validate_reference(
            asset,
            table_name,
            columns,
            rule_type,
        )
    ):
        return None

    parameters = (
        _validate_parameters(
            rule_type,
            proposal.parameters,
        )
    )

    if parameters is None:
        return None

    # "contains_letters" is deliberately reserved for the
    # deterministic semantic-name path above. The language model
    # cannot independently apply this rule to arbitrary text columns.
    if (
        rule_type
        == "pattern_match"
        and parameters.get(
            "pattern_name"
        )
        == "contains_letters"
    ):
        return None

    if (
        proposal.confidence
        < 0.55
    ):
        return None

    # Business-sensitive constraints must come from explicit source
    # documentation, not from observed distributions or AI intuition.
    if (
        rule_type
        in BUSINESS_SENSITIVE_RULES
        and not _explicit_business_support(
            asset,
            proposal,
            parameters,
        )
    ):
        return None

    reason = (
        proposal.reason.strip()
        or (
            "Agentic AI suggested this check "
            "from supplied evidence."
        )
    )

    return (
        _make_rule(
            table_name=table_name,
            columns=columns,
            rule_type=rule_type,
            parameters=parameters,
            severity=(
                _assign_severity(
                    rule_type
                )
            ),
            confidence=(
                proposal.confidence
            ),
            reason=reason,
            generated_by=(
                "agentic_ai"
            ),
        )
    )


# -------------------------------------------------------------------
# STABLE RULE IDS
# -------------------------------------------------------------------

def _rule_identity(
    rule: DataQualityRule,
) -> str:
    """
    Build stable identity independent of generated wording.
    """

    payload = {
        "table": (
            rule.table_name
        ),
        "columns": (
            sorted(
                rule.columns
            )
        ),
        "type": (
            rule.rule_type
        ),
        "parameters": (
            rule.parameters
        ),
    }

    return json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        default=str,
        separators=(
            ",",
            ":",
        ),
    )


def _assign_rule_id(
    rule: DataQualityRule,
) -> None:
    """
    Generate a deterministic SHA-based rule ID.
    """

    identity = (
        _rule_identity(
            rule
        )
    )

    digest = (
        hashlib.sha256(
            identity.encode(
                "utf-8"
            )
        )
        .hexdigest()[:12]
    )

    rule.rule_id = (
        f"dq_{digest}"
    )


# -------------------------------------------------------------------
# DEDUPLICATION
# -------------------------------------------------------------------

def _deduplicate_rules(
    rules: list[DataQualityRule],
) -> list[DataQualityRule]:
    """
    Keep the strongest version of equivalent rules.

    Deterministic rules win confidence ties.
    """

    best: dict[
        str,
        DataQualityRule,
    ] = {}

    for rule in rules:

        identity = (
            _rule_identity(
                rule
            )
        )

        existing = (
            best.get(
                identity
            )
        )

        if existing is None:

            best[
                identity
            ] = rule

            continue

        if (
            rule.confidence
            > existing.confidence
        ):

            best[
                identity
            ] = rule

        elif (
            rule.confidence
            == existing.confidence
            and rule.generated_by
            == "deterministic"
        ):

            best[
                identity
            ] = rule

    result = list(
        best.values()
    )

    result.sort(
        key=lambda rule: (
            rule.table_name,
            rule.columns,
            rule.rule_type,
        )
    )

    for rule in result:

        _assign_rule_id(
            rule
        )

    return result


# -------------------------------------------------------------------
# PUBLIC API
# -------------------------------------------------------------------

def generate_rules(
    asset: DataAsset,
    profile: DataProfile,
    understanding: DatasetUnderstanding,
) -> list[DataQualityRule]:
    """
    Generate safe, evidence-backed data-quality rules.

    Final rules combine deterministic structural checks and validated
    Agentic-AI proposals.

    Business-sensitive AI rules are accepted only when explicit source
    documentation supports them.
    """

    deterministic_rules = (
        _deterministic_structural_rules(
            profile,
            understanding,
        )
    )

    agentic_rules: list[
        DataQualityRule
    ] = []

    try:

        proposals = (
            _generate_agentic_proposals(
                asset,
                profile,
                understanding,
            )
        )

    except LLMError:

        # AI enrichment is optional. The pipeline can still run using
        # deterministic structural checks.
        proposals = []

    for proposal in proposals:

        rule = (
            _proposal_to_rule(
                asset,
                proposal,
            )
        )

        if rule is not None:

            agentic_rules.append(
                rule
            )

    return (
        _deduplicate_rules(
            deterministic_rules
            + agentic_rules
        )
    )