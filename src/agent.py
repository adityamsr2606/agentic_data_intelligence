from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from src.analysis_tools import (
    execute_tool,
    get_tool_catalog,
)
from src.config import settings
from src.llm_client import LLMError, llm_client
from src.models import (
    AgentDecision,
    DataAsset,
    DataProfile,
    DatasetUnderstanding,
    ToolRequest,
    ToolResult,
)


# -------------------------------------------------------------------
# AGENT RESULT
# -------------------------------------------------------------------

@dataclass
class AgentRunResult:
    """
    Complete audit trail for one Agentic AI reasoning run.
    """

    final_decision: AgentDecision
    tool_results: list[ToolResult]
    decisions: list[AgentDecision]
    rounds_used: int


# -------------------------------------------------------------------
# CONTEXT LIMITS
# -------------------------------------------------------------------

MAX_CONTEXT_COLUMNS = 60
MAX_CONTEXT_DOCUMENTS = 10

MAX_TOOLS_PER_ROUND = 4

MAX_TOOL_LIST_ITEMS = 20
MAX_TOOL_DICT_ITEMS = 30
MAX_TOOL_STRING_LENGTH = 2500

MAX_AGENT_SAMPLE_ROWS = 10
MAX_AGENT_DISTRIBUTION_ITEMS = 15
MAX_AGENT_DOCUMENT_CHARS = 5000


# -------------------------------------------------------------------
# SYSTEM PROMPT
# -------------------------------------------------------------------

AGENT_SYSTEM_PROMPT = """
You are the reasoning and investigation component of an Agentic Data
Intelligence system.

Your job is to understand an unknown dataset using supplied evidence
and, when necessary, request deterministic analytical tools.

You are NOT allowed to calculate important statistics mentally when a
tool can calculate them.

You are NOT allowed to invent:
- row counts
- null counts
- percentages
- correlations
- outliers
- duplicate counts
- ranges
- business rules
- schema relationships
- column meanings
- data-quality violations

Evidence rules:

1. Treat deterministic profiler output as factual evidence.

2. Treat successful tool results as factual evidence.

3. Semantic classifications are interpretations and must be described
   with appropriate confidence.

4. Samples are examples, not proof of a rule.

5. Outliers are unusual observations, not automatically bad data.

6. Duplicates are observations, not automatically violations unless
   business evidence establishes uniqueness.

7. Do not claim a business domain when the evidence is insufficient.

8. Do not create data-quality rules here. A separate guarded component
   handles rule generation.

9. Do not create source-to-target mappings here. A separate guarded
   component handles mapping.

10. Never request arbitrary code, shell commands, SQL execution,
    database writes, file changes, internet access, or tools outside
    the supplied tool catalog.

Tool behavior:

- Request tools only when they can resolve a meaningful uncertainty.
- Do not request the same investigation repeatedly.
- Prefer the smallest useful tool call.
- Stop investigating when the available evidence is sufficient.
- A failed tool is evidence that the requested investigation could not
  be completed; do not pretend it succeeded.

Hypotheses:

- A hypothesis must be supported by evidence.
- Clearly separate interpretation from measured fact.
- Keep confidence conservative.
- Every hypothesis should include evidence explaining why it exists.
- It is valid to return no hypotheses if there is not enough evidence.

Finalization:

- Set ready_to_finalize=true when additional tools are unlikely to
  materially improve the understanding.
- When ready_to_finalize=true, normally return no new tool requests.
- understanding MUST contain a concise, evidence-grounded summary.
- Never return an empty understanding when finalizing.
"""


# -------------------------------------------------------------------
# PROFILE CONTEXT
# -------------------------------------------------------------------

def _compact_patterns(
    column: Any,
) -> list[dict[str, Any]]:
    patterns: list[dict[str, Any]] = []

    for pattern in getattr(
        column,
        "patterns",
        [],
    ):
        patterns.append(
            {
                "name": pattern.pattern_name,
                "match_percentage": (
                    pattern.match_percentage
                ),
                "checked_values": (
                    pattern.checked_values
                ),
            }
        )

    patterns.sort(
        key=lambda item: item.get(
            "match_percentage",
            0,
        ),
        reverse=True,
    )

    return patterns[:3]


def _compact_numeric_statistics(
    statistics: dict[str, Any],
) -> dict[str, Any]:
    if not statistics:
        return {}

    wanted = {
        "min",
        "max",
        "mean",
        "median",
        "q1",
        "q3",
        "iqr",
        "iqr_outlier_count",
        "mad",
        "mad_outlier_count",
    }

    return {
        key: value
        for key, value in statistics.items()
        if key in wanted
    }


def _compact_datetime_statistics(
    statistics: dict[str, Any],
) -> dict[str, Any]:
    if not statistics:
        return {}

    wanted = {
        "min",
        "max",
        "valid_count",
        "invalid_count",
    }

    return {
        key: value
        for key, value in statistics.items()
        if key in wanted
    }


def _compact_profile(
    profile: DataProfile,
) -> dict[str, Any]:
    """
    Build compact deterministic evidence for the local model.
    """

    tables: list[
        dict[str, Any]
    ] = []

    columns_used = 0

    for table in profile.tables:
        table_context: dict[
            str,
            Any,
        ] = {
            "table_name": table.table_name,
            "row_count": table.row_count,
            "column_count": table.column_count,
            "duplicate_row_count": (
                table.duplicate_row_count
            ),
            "duplicate_percentage": (
                table.duplicate_percentage
            ),
            "columns": [],
        }

        for column in table.columns:
            if (
                columns_used
                >= MAX_CONTEXT_COLUMNS
            ):
                break

            column_context = {
                "column_name": (
                    column.column_name
                ),
                "inferred_type": (
                    column.inferred_type
                ),
                "null_percentage": (
                    column.null_percentage
                ),
                "distinct_percentage": (
                    column.distinct_percentage
                ),
                "patterns": (
                    _compact_patterns(
                        column
                    )
                ),
            }

            numeric = (
                _compact_numeric_statistics(
                    column.numeric_statistics
                )
            )

            if numeric:
                column_context[
                    "numeric_statistics"
                ] = numeric

            datetime_statistics = (
                _compact_datetime_statistics(
                    column.datetime_statistics
                )
            )

            if datetime_statistics:
                column_context[
                    "datetime_statistics"
                ] = datetime_statistics

            table_context[
                "columns"
            ].append(
                column_context
            )

            columns_used += 1

        tables.append(
            table_context
        )

        if (
            columns_used
            >= MAX_CONTEXT_COLUMNS
        ):
            break

    documents: list[
        dict[str, Any]
    ] = []

    for document in profile.documents[
        :MAX_CONTEXT_DOCUMENTS
    ]:
        documents.append(
            {
                "document_name": (
                    document.document_name
                ),
                "character_count": (
                    document.character_count
                ),
                "word_count": (
                    document.word_count
                ),
                "line_count": (
                    document.line_count
                ),
                "preview": (
                    document.preview[:600]
                ),
            }
        )

    total_columns = sum(
        table.column_count
        for table in profile.tables
    )

    return {
        "source_name": profile.source_name,
        "source_type": profile.source_type,
        "table_count": profile.table_count,
        "document_count": (
            profile.document_count
        ),
        "tables": tables,
        "documents": documents,
        "context_columns_included": (
            columns_used
        ),
        "total_profiled_columns": (
            total_columns
        ),
    }


# -------------------------------------------------------------------
# SEMANTIC CONTEXT
# -------------------------------------------------------------------

def _compact_semantics(
    understanding: DatasetUnderstanding,
) -> dict[str, Any]:
    insights: list[
        dict[str, Any]
    ] = []

    for insight in (
        understanding.column_insights[
            :MAX_CONTEXT_COLUMNS
        ]
    ):
        insights.append(
            {
                "table_name": (
                    insight.table_name
                ),
                "column_name": (
                    insight.column_name
                ),
                "semantic_type": (
                    insight.semantic_type
                ),
                "confidence": (
                    insight.confidence
                ),
            }
        )

    return {
        "detected_domain": (
            understanding.detected_domain
        ),
        "dataset_description": (
            understanding.dataset_description
        ),
        "column_insights": (
            insights
        ),
    }


# -------------------------------------------------------------------
# TOOL RESULT COMPACTION
# -------------------------------------------------------------------

def _compact_value(
    value: Any,
    depth: int = 0,
) -> Any:
    if depth >= 5:
        return str(
            value
        )[
            :MAX_TOOL_STRING_LENGTH
        ]

    if isinstance(
        value,
        str,
    ):
        return value[
            :MAX_TOOL_STRING_LENGTH
        ]

    if isinstance(
        value,
        dict,
    ):
        compact: dict[
            str,
            Any,
        ] = {}

        for index, (
            key,
            item,
        ) in enumerate(
            value.items()
        ):
            if (
                index
                >= MAX_TOOL_DICT_ITEMS
            ):
                compact[
                    "_truncated"
                ] = True
                break

            compact[
                str(
                    key
                )
            ] = _compact_value(
                item,
                depth + 1,
            )

        return compact

    if isinstance(
        value,
        (
            list,
            tuple,
        ),
    ):
        compact_list = [
            _compact_value(
                item,
                depth + 1,
            )
            for item in value[
                :MAX_TOOL_LIST_ITEMS
            ]
        ]

        if (
            len(value)
            > MAX_TOOL_LIST_ITEMS
        ):
            compact_list.append(
                {
                    "_truncated_items": (
                        len(value)
                        - MAX_TOOL_LIST_ITEMS
                    )
                }
            )

        return compact_list

    return value


def _compact_tool_result(
    result: ToolResult,
) -> dict[str, Any]:
    return {
        "tool_name": result.tool_name,
        "success": result.success,
        "table_name": (
            result.table_name
        ),
        "column_name": (
            result.column_name
        ),
        "result": (
            _compact_value(
                result.result
            )
        ),
        "message": result.message,
    }


# -------------------------------------------------------------------
# TOOL REQUEST SAFETY
# -------------------------------------------------------------------

def _normalize_positive_integer(
    value: Any,
    *,
    default: int,
    maximum: int,
) -> int:
    try:
        parsed = int(
            value
        )

    except (
        TypeError,
        ValueError,
    ):
        return default

    if parsed <= 0:
        return default

    return min(
        parsed,
        maximum,
    )


def _normalize_tool_request(
    request: ToolRequest,
) -> ToolRequest:
    """
    Clamp potentially large read requests before execution.
    """

    parameters = dict(
        request.parameters
        or {}
    )

    if (
        request.tool_name
        == "sample_rows"
    ):
        parameters[
            "limit"
        ] = (
            _normalize_positive_integer(
                parameters.get(
                    "limit"
                ),
                default=5,
                maximum=(
                    MAX_AGENT_SAMPLE_ROWS
                ),
            )
        )

    elif (
        request.tool_name
        == "column_distribution"
    ):
        parameters[
            "limit"
        ] = (
            _normalize_positive_integer(
                parameters.get(
                    "limit"
                ),
                default=10,
                maximum=(
                    MAX_AGENT_DISTRIBUTION_ITEMS
                ),
            )
        )

    elif (
        request.tool_name
        == "document_excerpt"
    ):
        try:
            start = int(
                parameters.get(
                    "start",
                    0,
                )
                or 0
            )

        except (
            TypeError,
            ValueError,
        ):
            start = 0

        parameters[
            "start"
        ] = max(
            0,
            start,
        )

        parameters[
            "length"
        ] = (
            _normalize_positive_integer(
                parameters.get(
                    "length"
                ),
                default=2000,
                maximum=(
                    MAX_AGENT_DOCUMENT_CHARS
                ),
            )
        )

    return ToolRequest(
        tool_name=request.tool_name,
        table_name=request.table_name,
        column_name=request.column_name,
        second_column_name=(
            request.second_column_name
        ),
        parameters=parameters,
    )


def _request_fingerprint(
    request: ToolRequest,
) -> str:
    payload = {
        "tool_name": request.tool_name,
        "table_name": (
            request.table_name
        ),
        "column_name": (
            request.column_name
        ),
        "second_column_name": (
            request.second_column_name
        ),
        "parameters": (
            request.parameters
        ),
    }

    return json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        default=str,
    )


def _new_tool_requests(
    requests: list[
        ToolRequest
    ],
    used_fingerprints: set[str],
) -> list[
    ToolRequest
]:
    accepted: list[
        ToolRequest
    ] = []

    for request in requests:
        normalized = (
            _normalize_tool_request(
                request
            )
        )

        fingerprint = (
            _request_fingerprint(
                normalized
            )
        )

        if (
            fingerprint
            in used_fingerprints
        ):
            continue

        used_fingerprints.add(
            fingerprint
        )

        accepted.append(
            normalized
        )

        if (
            len(
                accepted
            )
            >= MAX_TOOLS_PER_ROUND
        ):
            break

    return accepted


# -------------------------------------------------------------------
# PROMPT BUILDING
# -------------------------------------------------------------------

def _build_round_prompt(
    *,
    asset: DataAsset,
    profile_context: dict[
        str,
        Any,
    ],
    semantic_context: dict[
        str,
        Any,
    ],
    tool_catalog: Any,
    tool_results: list[
        ToolResult
    ],
    round_number: int,
    max_rounds: int,
) -> str:
    evidence = [
        _compact_tool_result(
            result
        )
        for result in tool_results
    ]

    final_round = (
        round_number
        >= max_rounds
    )

    if final_round:
        instructions = (
            "This is the FINAL reasoning round. "
            "Do not request additional tools. "
            "Set ready_to_finalize=true. "
            "The understanding field MUST contain a concise "
            "evidence-grounded summary and MUST NOT be empty."
        )

    else:
        instructions = (
            "Decide whether the current evidence is sufficient. "
            "If an important uncertainty can be resolved by a "
            "whitelisted tool, request the smallest useful tool call. "
            "Always provide a concise current understanding."
        )

    payload = {
        "round": round_number,
        "maximum_rounds": (
            max_rounds
        ),
        "source": {
            "path": str(
                asset.source_path
            ),
            "source_type": (
                asset.source_type
            ),
            "table_names": list(
                asset.tables.keys()
            ),
            "document_names": list(
                asset.documents.keys()
            ),
        },
        "deterministic_profile": (
            profile_context
        ),
        "semantic_interpretation": (
            semantic_context
        ),
        "available_tools": (
            tool_catalog
        ),
        "tool_evidence_so_far": (
            evidence
        ),
        "round_instruction": (
            instructions
        ),
    }

    return (
        "Investigate the dataset using only the supplied "
        "evidence.\n\n"
        + json.dumps(
            payload,
            ensure_ascii=False,
            default=str,
            separators=(
                ",",
                ":",
            ),
        )
    )


# -------------------------------------------------------------------
# FINAL DECISION GUARDRAILS
# -------------------------------------------------------------------

def _fallback_final_decision(
    understanding: DatasetUnderstanding,
    tool_results: list[
        ToolResult
    ],
) -> AgentDecision:
    """
    Build a deterministic final summary if the model fails to provide
    a usable final decision.

    No new claims are invented here.
    """

    summary = (
        understanding.dataset_description
        or (
            "Deterministic profiling and semantic analysis "
            "were completed for the supplied source."
        )
    ).strip()

    domain = (
        understanding.detected_domain
        or "unknown"
    ).strip()

    if (
        domain
        and domain.lower()
        != "unknown"
    ):
        summary += (
            f" The semantic analysis suggests the "
            f"dataset domain may be '{domain}'."
        )

    if tool_results:
        successful = [
            result
            for result in tool_results
            if result.success
        ]

        failed = [
            result
            for result in tool_results
            if not result.success
        ]

        summary += (
            f" The agent executed {len(tool_results)} "
            "deterministic analytical tool call(s), "
            f"with {len(successful)} successful and "
            f"{len(failed)} unsuccessful."
        )

        successful_names = []

        for result in successful:
            if (
                result.tool_name
                not in successful_names
            ):
                successful_names.append(
                    result.tool_name
                )

        if successful_names:
            summary += (
                " Successful investigation tools: "
                + ", ".join(
                    successful_names
                )
                + "."
            )

    return AgentDecision(
        understanding=summary,
        hypotheses=[],
        tool_requests=[],
        ready_to_finalize=True,
    )


def _has_valid_understanding(
    decision: AgentDecision,
) -> bool:
    """
    A final agent response must contain meaningful understanding text.
    """

    text = (
        decision.understanding
        or ""
    ).strip()

    return len(
        text
    ) >= 10


def _finalize_decision(
    decision: AgentDecision,
    understanding: DatasetUnderstanding,
    tool_results: list[
        ToolResult
    ],
) -> AgentDecision:
    """
    Never allow an empty or unusable decision to become the final
    agent output.
    """

    if not (
        _has_valid_understanding(
            decision
        )
    ):
        return (
            _fallback_final_decision(
                understanding,
                tool_results,
            )
        )

    return decision.model_copy(
        update={
            "tool_requests": [],
            "ready_to_finalize": True,
        }
    )


# -------------------------------------------------------------------
# PUBLIC AGENT LOOP
# -------------------------------------------------------------------

def run_agentic_analysis(
    asset: DataAsset,
    profile: DataProfile,
    understanding: DatasetUnderstanding,
    *,
    max_rounds: int | None = None,
) -> AgentRunResult:
    """
    Run a bounded observe-reason-act-observe Agentic AI loop.

    The LLM decides which evidence it needs.

    Python controls:
    - available tools
    - tool arguments
    - execution
    - duplicate requests
    - tool budget
    - maximum rounds
    - final-output validation
    """

    configured_max = int(
        settings.max_agent_rounds
    )

    rounds_limit = (
        configured_max
        if max_rounds is None
        else int(
            max_rounds
        )
    )

    if rounds_limit <= 0:
        raise ValueError(
            "max_rounds must be greater than zero."
        )

    rounds_limit = min(
        rounds_limit,
        configured_max,
    )

    if (
        not asset.tables
        and not asset.documents
    ):
        decision = AgentDecision(
            understanding=(
                "The source contains no analyzable "
                "tables or documents."
            ),
            hypotheses=[],
            tool_requests=[],
            ready_to_finalize=True,
        )

        return AgentRunResult(
            final_decision=decision,
            tool_results=[],
            decisions=[
                decision
            ],
            rounds_used=1,
        )

    profile_context = (
        _compact_profile(
            profile
        )
    )

    semantic_context = (
        _compact_semantics(
            understanding
        )
    )

    tool_catalog = (
        get_tool_catalog()
    )

    tool_results: list[
        ToolResult
    ] = []

    decisions: list[
        AgentDecision
    ] = []

    used_fingerprints: set[
        str
    ] = set()

    for round_number in range(
        1,
        rounds_limit + 1,
    ):
        prompt = (
            _build_round_prompt(
                asset=asset,
                profile_context=(
                    profile_context
                ),
                semantic_context=(
                    semantic_context
                ),
                tool_catalog=(
                    tool_catalog
                ),
                tool_results=(
                    tool_results
                ),
                round_number=(
                    round_number
                ),
                max_rounds=(
                    rounds_limit
                ),
            )
        )

        try:
            decision = (
                llm_client.chat_structured(
                    system_prompt=(
                        AGENT_SYSTEM_PROMPT
                    ),
                    user_prompt=prompt,
                    response_model=(
                        AgentDecision
                    ),
                    retries=0,
                    num_predict=650,
                )
            )

        except LLMError:
            fallback = (
                _fallback_final_decision(
                    understanding,
                    tool_results,
                )
            )

            decisions.append(
                fallback
            )

            return AgentRunResult(
                final_decision=(
                    fallback
                ),
                tool_results=(
                    tool_results
                ),
                decisions=(
                    decisions
                ),
                rounds_used=(
                    round_number
                ),
            )

        decisions.append(
            decision
        )

        # -----------------------------------------------------------
        # Final round
        # -----------------------------------------------------------

        if (
            round_number
            == rounds_limit
        ):
            final_decision = (
                _finalize_decision(
                    decision,
                    understanding,
                    tool_results,
                )
            )

            decisions[
                -1
            ] = final_decision

            return AgentRunResult(
                final_decision=(
                    final_decision
                ),
                tool_results=(
                    tool_results
                ),
                decisions=(
                    decisions
                ),
                rounds_used=(
                    round_number
                ),
            )

        # -----------------------------------------------------------
        # Validate requested actions
        # -----------------------------------------------------------

        requested = (
            _new_tool_requests(
                decision.tool_requests,
                used_fingerprints,
            )
        )

        # -----------------------------------------------------------
        # Agent intentionally finished
        # -----------------------------------------------------------

        if (
            decision.ready_to_finalize
            and not requested
        ):
            final_decision = (
                _finalize_decision(
                    decision,
                    understanding,
                    tool_results,
                )
            )

            decisions[
                -1
            ] = final_decision

            return AgentRunResult(
                final_decision=(
                    final_decision
                ),
                tool_results=(
                    tool_results
                ),
                decisions=(
                    decisions
                ),
                rounds_used=(
                    round_number
                ),
            )

        # -----------------------------------------------------------
        # No useful action requested
        # -----------------------------------------------------------

        if not requested:
            final_decision = (
                _finalize_decision(
                    decision,
                    understanding,
                    tool_results,
                )
            )

            decisions[
                -1
            ] = final_decision

            return AgentRunResult(
                final_decision=(
                    final_decision
                ),
                tool_results=(
                    tool_results
                ),
                decisions=(
                    decisions
                ),
                rounds_used=(
                    round_number
                ),
            )

        # -----------------------------------------------------------
        # Execute only whitelisted deterministic tools
        # -----------------------------------------------------------

        for request in requested:
            result = (
                execute_tool(
                    asset,
                    request,
                )
            )

            tool_results.append(
                result
            )

    # Defensive fallback. Normal execution should return from inside
    # the bounded loop.
    fallback = (
        _fallback_final_decision(
            understanding,
            tool_results,
        )
    )

    decisions.append(
        fallback
    )

    return AgentRunResult(
        final_decision=fallback,
        tool_results=tool_results,
        decisions=decisions,
        rounds_used=rounds_limit,
    )