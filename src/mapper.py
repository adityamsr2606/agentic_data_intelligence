from __future__ import annotations

import json
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any

from pydantic import BaseModel, Field

from src.config import settings
from src.llm_client import LLMError, llm_client
from src.models import (
    ColumnProfile,
    DataAsset,
    DataProfile,
    DatasetUnderstanding,
    MappingSuggestion,
    TargetColumn,
    TargetSchema,
)


MAX_CANDIDATES_PER_SOURCE = 4
AI_BATCH_SIZE = 10

STRONG_MAPPING_MARGIN = 0.12
MIN_CANDIDATE_SCORE = 0.35
FALLBACK_REVIEW_SCORE = 0.55


# -------------------------------------------------------------------
# INTERNAL MODELS
# -------------------------------------------------------------------

@dataclass
class MappingCandidate:
    """
    Deterministically scored source-to-target candidate.
    """

    target_table: str
    target_column: str
    target_data_type: str
    target_nullable: bool
    target_primary_key: bool
    target_description: str

    name_score: float
    type_score: float
    semantic_score: float
    table_score: float
    total_score: float


class MappingDecision(BaseModel):
    """
    Small decision returned by the local model.

    The model may only choose from candidates supplied by Python.
    """

    source_table: str
    source_column: str

    should_map: bool

    target_table: str = ""
    target_column: str = ""

    confidence: float = Field(
        ge=0.0,
        le=1.0,
    )

    reason: str = ""


class MappingDecisionResponse(BaseModel):
    decisions: list[
        MappingDecision
    ] = Field(
        default_factory=list
    )


# -------------------------------------------------------------------
# NAME NORMALIZATION
# -------------------------------------------------------------------

def _split_words(
    value: str,
) -> list[str]:
    """
    Normalize snake_case, spaces, punctuation and camelCase into tokens.
    """

    value = re.sub(
        r"([a-z0-9])([A-Z])",
        r"\1 \2",
        value,
    )

    value = re.sub(
        r"[^A-Za-z0-9]+",
        " ",
        value,
    )

    return [
        token.lower()
        for token in value.split()
        if token.strip()
    ]


def _compact_name(
    value: str,
) -> str:
    return "".join(
        _split_words(
            value
        )
    )


def _name_similarity(
    source: str,
    target: str,
) -> float:
    """
    Compare names using both token overlap and character similarity.
    """

    source_tokens = set(
        _split_words(
            source
        )
    )

    target_tokens = set(
        _split_words(
            target
        )
    )

    source_compact = (
        _compact_name(
            source
        )
    )

    target_compact = (
        _compact_name(
            target
        )
    )

    if (
        source_compact
        and source_compact
        == target_compact
    ):
        return 1.0

    sequence_score = (
        SequenceMatcher(
            None,
            source_compact,
            target_compact,
        )
        .ratio()
        if source_compact
        and target_compact
        else 0.0
    )

    if (
        source_tokens
        and target_tokens
    ):
        intersection = len(
            source_tokens
            & target_tokens
        )

        union = len(
            source_tokens
            | target_tokens
        )

        token_score = (
            intersection
            / union
            if union
            else 0.0
        )

    else:
        token_score = 0.0

    return round(
        max(
            sequence_score,
            token_score,
        ),
        4,
    )


# -------------------------------------------------------------------
# TYPE COMPATIBILITY
# -------------------------------------------------------------------

def _target_type_family(
    data_type: str,
) -> str:
    """
    Normalize common SQL/database types into generic families.
    """

    value = (
        data_type
        .strip()
        .lower()
    )

    if "uuid" in value:
        return "uuid"

    if any(
        token in value
        for token in {
            "timestamp",
            "datetime",
        }
    ):
        return "datetime"

    if (
        "date" in value
        and "update" not in value
    ):
        return "date"

    if any(
        token in value
        for token in {
            "bool",
            "boolean",
        }
    ):
        return "boolean"

    if (
        "int" in value
        and "interval" not in value
    ):
        return "integer"

    if any(
        token in value
        for token in {
            "numeric",
            "decimal",
            "float",
            "double",
            "real",
            "number",
            "money",
        }
    ):
        return "number"

    if any(
        token in value
        for token in {
            "char",
            "text",
            "string",
            "clob",
            "varchar",
        }
    ):
        return "string"

    return "unknown"


def _source_type_family(
    inferred_type: str,
) -> str:
    value = (
        inferred_type
        .strip()
        .lower()
    )

    if value == "integer":
        return "integer"

    if value in {
        "float",
        "number",
        "numeric",
    }:
        return "number"

    if value == "datetime":
        return "datetime"

    if value == "date":
        return "date"

    if value == "boolean":
        return "boolean"

    if value == "string":
        return "string"

    return "unknown"


def _type_compatibility(
    source_type: str,
    target_type: str,
) -> float:
    """
    Score whether source and target physical types can reasonably map.
    """

    source_family = (
        _source_type_family(
            source_type
        )
    )

    target_family = (
        _target_type_family(
            target_type
        )
    )

    if (
        source_family
        == target_family
        and source_family
        != "unknown"
    ):
        return 1.0

    if {
        source_family,
        target_family,
    } <= {
        "integer",
        "number",
    }:
        return 0.90

    if (
        source_family == "date"
        and target_family == "datetime"
    ):
        return 0.90

    if (
        source_family == "datetime"
        and target_family == "date"
    ):
        return 0.75

    if (
        source_family == "string"
        and target_family == "uuid"
    ):
        return 0.65

    if (
        target_family == "string"
        and source_family
        not in {
            "unknown",
            "string",
        }
    ):
        return 0.55

    if (
        source_family == "unknown"
        or target_family == "unknown"
    ):
        return 0.40

    return 0.10


# -------------------------------------------------------------------
# SEMANTIC COMPATIBILITY
# -------------------------------------------------------------------

SEMANTIC_HINTS = {
    "identifier": {
        "id",
        "identifier",
        "key",
        "uuid",
    },
    "identifier_candidate": {
        "id",
        "identifier",
        "key",
        "code",
    },
    "code": {
        "code",
        "key",
        "identifier",
    },
    "email": {
        "email",
        "mail",
    },
    "url": {
        "url",
        "uri",
        "link",
        "website",
    },
    "ip_address": {
        "ip",
        "ipaddress",
    },
    "datetime": {
        "datetime",
        "timestamp",
        "time",
        "date",
    },
    "date": {
        "date",
        "day",
    },
    "boolean": {
        "flag",
        "boolean",
        "enabled",
        "active",
    },
    "name": {
        "name",
        "title",
        "label",
    },
    "address": {
        "address",
        "street",
    },
    "phone": {
        "phone",
        "mobile",
        "telephone",
    },
    "currency": {
        "amount",
        "price",
        "cost",
        "currency",
        "revenue",
    },
    "percentage": {
        "percent",
        "percentage",
        "ratio",
        "rate",
    },
    "category": {
        "category",
        "class",
        "type",
    },
}


def _semantic_compatibility(
    semantic_type: str,
    target_column: TargetColumn,
) -> float:
    """
    Use semantic classification only as supporting evidence.
    """

    hints = (
        SEMANTIC_HINTS.get(
            semantic_type,
            set(),
        )
    )

    if not hints:
        return 0.0

    text = (
        f"{target_column.column_name} "
        f"{target_column.description}"
    )

    tokens = set(
        _split_words(
            text
        )
    )

    compact = (
        _compact_name(
            text
        )
    )

    if any(
        hint in tokens
        for hint in hints
    ):
        return 1.0

    if any(
        hint in compact
        for hint in hints
        if len(hint) >= 4
    ):
        return 0.75

    return 0.0


# -------------------------------------------------------------------
# PROFILE / SEMANTIC LOOKUPS
# -------------------------------------------------------------------

def _semantic_lookup(
    understanding: DatasetUnderstanding,
) -> dict[
    tuple[str, str],
    tuple[str, float],
]:

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


def _find_source_profile(
    profile: DataProfile,
    table_name: str,
    column_name: str,
) -> ColumnProfile | None:

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


# -------------------------------------------------------------------
# CANDIDATE SCORING
# -------------------------------------------------------------------

def _candidate_score(
    *,
    source_table: str,
    source_column: str,
    source_profile: ColumnProfile,
    semantic_type: str,
    semantic_confidence: float,
    target_table: str,
    target_column: TargetColumn,
) -> MappingCandidate:
    """
    Score one source-column / target-column pair deterministically.
    """

    name_score = (
        _name_similarity(
            source_column,
            target_column.column_name,
        )
    )

    type_score = (
        _type_compatibility(
            source_profile.inferred_type,
            target_column.data_type,
        )
    )

    semantic_score = (
        _semantic_compatibility(
            semantic_type,
            target_column,
        )
        * semantic_confidence
    )

    table_score = (
        _name_similarity(
            source_table,
            target_table,
        )
    )

    total_score = (
        0.50
        * name_score

        + 0.25
        * type_score

        + 0.15
        * semantic_score

        + 0.10
        * table_score
    )

    return MappingCandidate(
        target_table=target_table,
        target_column=(
            target_column.column_name
        ),
        target_data_type=(
            target_column.data_type
        ),
        target_nullable=(
            target_column.nullable
        ),
        target_primary_key=(
            target_column.primary_key
        ),
        target_description=(
            target_column.description
        ),
        name_score=round(
            name_score,
            4,
        ),
        type_score=round(
            type_score,
            4,
        ),
        semantic_score=round(
            semantic_score,
            4,
        ),
        table_score=round(
            table_score,
            4,
        ),
        total_score=round(
            total_score,
            4,
        ),
    )


def _build_candidates(
    profile: DataProfile,
    understanding: DatasetUnderstanding,
    target_schema: TargetSchema,
) -> dict[
    tuple[str, str],
    list[MappingCandidate],
]:
    """
    Build the top deterministic target candidates for every source
    column.
    """

    semantic_map = (
        _semantic_lookup(
            understanding
        )
    )

    candidates: dict[
        tuple[str, str],
        list[MappingCandidate],
    ] = {}

    for table in profile.tables:

        for source_column in (
            table.columns
        ):

            (
                semantic_type,
                semantic_confidence,
            ) = semantic_map.get(
                (
                    table.table_name,
                    source_column.column_name,
                ),
                (
                    "unknown",
                    0.0,
                ),
            )

            scored: list[
                MappingCandidate
            ] = []

            for target_table in (
                target_schema.tables
            ):

                for target_column in (
                    target_table.columns
                ):

                    candidate = (
                        _candidate_score(
                            source_table=(
                                table.table_name
                            ),
                            source_column=(
                                source_column.column_name
                            ),
                            source_profile=(
                                source_column
                            ),
                            semantic_type=(
                                semantic_type
                            ),
                            semantic_confidence=(
                                semantic_confidence
                            ),
                            target_table=(
                                target_table.table_name
                            ),
                            target_column=(
                                target_column
                            ),
                        )
                    )

                    if (
                        candidate.total_score
                        >= MIN_CANDIDATE_SCORE
                    ):
                        scored.append(
                            candidate
                        )

            scored.sort(
                key=lambda candidate: (
                    candidate.total_score
                ),
                reverse=True,
            )

            candidates[
                (
                    table.table_name,
                    source_column.column_name,
                )
            ] = scored[
                :MAX_CANDIDATES_PER_SOURCE
            ]

    return candidates


# -------------------------------------------------------------------
# TRANSFORMATION DECISION
# -------------------------------------------------------------------

def _transformation_for(
    source_type: str,
    target_type: str,
) -> str:
    """
    Keep transformations declarative.

    Arbitrary Python or SQL is never generated here.
    """

    source_family = (
        _source_type_family(
            source_type
        )
    )

    target_family = (
        _target_type_family(
            target_type
        )
    )

    if (
        source_family
        == target_family
    ):
        return "direct"

    if (
        source_family == "integer"
        and target_family == "number"
    ):
        return "direct"

    return (
        f"cast:{target_type}"
    )


# -------------------------------------------------------------------
# DETERMINISTIC STRONG MATCHES
# -------------------------------------------------------------------

def _is_strong_match(
    candidates: list[MappingCandidate],
) -> bool:
    """
    Accept clear deterministic matches without an LLM call.
    """

    if not candidates:
        return False

    top = (
        candidates[0]
    )

    if (
        top.total_score
        < settings.min_mapping_confidence
    ):
        return False

    if len(candidates) == 1:
        return True

    margin = (
        top.total_score
        - candidates[1].total_score
    )

    exact_name = (
        top.name_score
        == 1.0
    )

    return (
        exact_name
        or margin
        >= STRONG_MAPPING_MARGIN
    )


def _deterministic_mapping(
    profile: DataProfile,
    source_table: str,
    source_column: str,
    candidate: MappingCandidate,
) -> MappingSuggestion:
    """
    Convert a clear deterministic candidate into a mapping.
    """

    source_profile = (
        _find_source_profile(
            profile,
            source_table,
            source_column,
        )
    )

    if source_profile is None:
        raise ValueError(
            "Source profile disappeared while "
            "building a mapping."
        )

    return MappingSuggestion(
        source_table=source_table,
        source_column=source_column,
        target_table=(
            candidate.target_table
        ),
        target_column=(
            candidate.target_column
        ),
        transformation=(
            _transformation_for(
                source_profile.inferred_type,
                candidate.target_data_type,
            )
        ),
        confidence=(
            candidate.total_score
        ),
        reason=(
            "Strong deterministic match based on "
            "column name, datatype, semantic meaning "
            "and table context."
        ),
        requires_review=False,
    )


# -------------------------------------------------------------------
# AGENTIC AMBIGUITY RESOLUTION
# -------------------------------------------------------------------

MAPPING_SYSTEM_PROMPT = """
Resolve ambiguous source-to-target mappings.

Python has already created a short list of valid target candidates.

Rules:

1. You may choose ONLY a candidate supplied for that source column.

2. Never invent a table or column.

3. If none of the candidates is sufficiently supported, set
   should_map=false.

4. Column-name similarity is evidence, but not proof.

5. Datatype compatibility is evidence.

6. Semantic meaning is evidence.

7. Samples are examples only.

8. Do not invent business relationships.

9. Do not generate Python, SQL or transformation code.

10. Prefer no mapping over an unsupported mapping.

11. Confidence must reflect the supplied evidence.
"""


def _candidate_payload(
    candidate: MappingCandidate,
) -> dict[str, Any]:

    return {
        "target_table": (
            candidate.target_table
        ),
        "target_column": (
            candidate.target_column
        ),
        "data_type": (
            candidate.target_data_type
        ),
        "nullable": (
            candidate.target_nullable
        ),
        "primary_key": (
            candidate.target_primary_key
        ),
        "description": (
            candidate.target_description
        ),
        "deterministic_score": (
            candidate.total_score
        ),
    }


def _build_ai_source_payload(
    asset: DataAsset,
    profile: DataProfile,
    understanding: DatasetUnderstanding,
    source_table: str,
    source_column: str,
    candidates: list[MappingCandidate],
) -> dict[str, Any]:

    column_profile = (
        _find_source_profile(
            profile,
            source_table,
            source_column,
        )
    )

    if column_profile is None:
        return {}

    semantic_map = (
        _semantic_lookup(
            understanding
        )
    )

    (
        semantic_type,
        semantic_confidence,
    ) = semantic_map.get(
        (
            source_table,
            source_column,
        ),
        (
            "unknown",
            0.0,
        ),
    )

    dataframe = (
        asset.tables.get(
            source_table
        )
    )

    samples: list[Any] = []

    if (
        dataframe is not None
        and source_column
        in dataframe.columns
    ):
        samples = (
            dataframe[
                source_column
            ]
            .dropna()
            .head(3)
            .tolist()
        )

    return {
        "source_table": (
            source_table
        ),
        "source_column": (
            source_column
        ),
        "source_type": (
            column_profile.inferred_type
        ),
        "semantic_type": (
            semantic_type
        ),
        "semantic_confidence": (
            semantic_confidence
        ),
        "samples": samples,
        "candidates": [
            _candidate_payload(
                candidate
            )
            for candidate
            in candidates
        ],
    }


def _resolve_with_ai(
    payloads: list[
        dict[str, Any]
    ],
) -> list[MappingDecision]:
    """
    Resolve a small batch of ambiguous mappings.
    """

    if not payloads:
        return []

    prompt = (
        "Resolve these ambiguous mappings using only "
        "the supplied candidate lists.\n\n"
        "SOURCE CANDIDATES:\n"
        + json.dumps(
            payloads,
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
                MAPPING_SYSTEM_PROMPT
            ),
            user_prompt=(
                prompt
            ),
            response_model=(
                MappingDecisionResponse
            ),
            retries=0,
            num_predict=500,
        )
    )

    return response.decisions


# -------------------------------------------------------------------
# AI DECISION VALIDATION
# -------------------------------------------------------------------

def _candidate_lookup(
    candidates: list[
        MappingCandidate
    ],
) -> dict[
    tuple[str, str],
    MappingCandidate,
]:

    return {
        (
            candidate.target_table,
            candidate.target_column,
        ): candidate
        for candidate
        in candidates
    }


def _validated_ai_mapping(
    profile: DataProfile,
    source_table: str,
    source_column: str,
    candidates: list[MappingCandidate],
    decision: MappingDecision,
) -> MappingSuggestion | None:
    """
    Convert an AI decision into a mapping only if it selected a real
    candidate provided by Python.
    """

    if not (
        decision.should_map
    ):
        return None

    if (
        decision.source_table
        != source_table
        or decision.source_column
        != source_column
    ):
        return None

    lookup = (
        _candidate_lookup(
            candidates
        )
    )

    candidate = (
        lookup.get(
            (
                decision.target_table,
                decision.target_column,
            )
        )
    )

    if candidate is None:
        return None

    source_profile = (
        _find_source_profile(
            profile,
            source_table,
            source_column,
        )
    )

    if source_profile is None:
        return None

    # AI confidence cannot completely override weak deterministic
    # evidence.
    confidence = round(
        (
            0.70
            * candidate.total_score
        )
        + (
            0.30
            * decision.confidence
        ),
        4,
    )

    requires_review = (
        confidence
        < settings.min_mapping_confidence
        or candidate.total_score
        < 0.60
    )

    reason = (
        decision.reason.strip()
        or (
            "Agentic AI selected this target from "
            "the deterministic candidate list."
        )
    )

    return MappingSuggestion(
        source_table=source_table,
        source_column=source_column,
        target_table=(
            candidate.target_table
        ),
        target_column=(
            candidate.target_column
        ),
        transformation=(
            _transformation_for(
                source_profile.inferred_type,
                candidate.target_data_type,
            )
        ),
        confidence=confidence,
        reason=reason,
        requires_review=(
            requires_review
        ),
    )


# -------------------------------------------------------------------
# FALLBACK
# -------------------------------------------------------------------

def _fallback_mapping(
    profile: DataProfile,
    source_table: str,
    source_column: str,
    candidates: list[MappingCandidate],
) -> MappingSuggestion | None:
    """
    Preserve a useful candidate for manual review when the local model
    is unavailable.

    Weak candidates are omitted instead of guessed.
    """

    if not candidates:
        return None

    candidate = (
        candidates[0]
    )

    if (
        candidate.total_score
        < FALLBACK_REVIEW_SCORE
    ):
        return None

    source_profile = (
        _find_source_profile(
            profile,
            source_table,
            source_column,
        )
    )

    if source_profile is None:
        return None

    return MappingSuggestion(
        source_table=source_table,
        source_column=source_column,
        target_table=(
            candidate.target_table
        ),
        target_column=(
            candidate.target_column
        ),
        transformation=(
            _transformation_for(
                source_profile.inferred_type,
                candidate.target_data_type,
            )
        ),
        confidence=(
            candidate.total_score
        ),
        reason=(
            "Best deterministic candidate. "
            "Manual review is required because "
            "the mapping was not sufficiently clear."
        ),
        requires_review=True,
    )


# -------------------------------------------------------------------
# PUBLIC API
# -------------------------------------------------------------------

def generate_mappings(
    asset: DataAsset,
    profile: DataProfile,
    understanding: DatasetUnderstanding,
    target_schema: TargetSchema,
) -> list[MappingSuggestion]:
    """
    Generate source-to-target mapping suggestions.

    Clear mappings are accepted deterministically.

    Ambiguous mappings are narrowed to a short candidate list before
    Agentic AI evaluates them.

    Unsupported mappings are omitted rather than invented.
    """

    if not target_schema.tables:
        return []

    candidate_map = (
        _build_candidates(
            profile,
            understanding,
            target_schema,
        )
    )

    mappings: list[
        MappingSuggestion
    ] = []

    ambiguous: list[
        tuple[
            str,
            str,
            list[MappingCandidate],
        ]
    ] = []

    for (
        source_key,
        candidates,
    ) in candidate_map.items():

        (
            source_table,
            source_column,
        ) = source_key

        if (
            _is_strong_match(
                candidates
            )
        ):
            mappings.append(
                _deterministic_mapping(
                    profile,
                    source_table,
                    source_column,
                    candidates[0],
                )
            )

        elif candidates:
            ambiguous.append(
                (
                    source_table,
                    source_column,
                    candidates,
                )
            )

    # Resolve ambiguous fields in small batches so large schemas do
    # not create one huge local-model prompt.
    for batch_start in range(
        0,
        len(ambiguous),
        AI_BATCH_SIZE,
    ):

        batch = ambiguous[
            batch_start:
            batch_start
            + AI_BATCH_SIZE
        ]

        payloads = [
            _build_ai_source_payload(
                asset,
                profile,
                understanding,
                source_table,
                source_column,
                candidates,
            )
            for (
                source_table,
                source_column,
                candidates,
            ) in batch
        ]

        payloads = [
            payload
            for payload in payloads
            if payload
        ]

        try:
            decisions = (
                _resolve_with_ai(
                    payloads
                )
            )

        except LLMError:
            decisions = []

        decision_lookup = {
            (
                decision.source_table,
                decision.source_column,
            ): decision
            for decision
            in decisions
        }

        for (
            source_table,
            source_column,
            candidates,
        ) in batch:

            decision = (
                decision_lookup.get(
                    (
                        source_table,
                        source_column,
                    )
                )
            )

            mapping = None

            if decision is not None:
                mapping = (
                    _validated_ai_mapping(
                        profile,
                        source_table,
                        source_column,
                        candidates,
                        decision,
                    )
                )

            if (
                mapping is None
            ):
                mapping = (
                    _fallback_mapping(
                        profile,
                        source_table,
                        source_column,
                        candidates,
                    )
                )

            if mapping is not None:
                mappings.append(
                    mapping
                )

    mappings.sort(
        key=lambda item: (
            item.source_table,
            item.source_column,
        )
    )

    return mappings