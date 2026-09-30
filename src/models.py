from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, Field


# -------------------------------------------------------------------
# INGESTION MODELS
# -------------------------------------------------------------------

@dataclass
class DataAsset:
    """
    Internal representation of any file loaded into the system.

    Structured data is stored inside tables.
    Unstructured text is stored inside documents.
    """

    source_path: Path
    source_type: str

    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    documents: dict[str, str] = field(default_factory=dict)


# -------------------------------------------------------------------
# PROFILING MODELS
# -------------------------------------------------------------------

class PatternResult(BaseModel):
    pattern_name: str

    matched_values: int = 0
    checked_values: int = 0

    match_percentage: float = 0.0


class ColumnProfile(BaseModel):
    column_name: str

    pandas_type: str
    inferred_type: str

    row_count: int

    null_count: int
    null_percentage: float

    distinct_count: int
    distinct_percentage: float

    sample_values: list[Any] = Field(default_factory=list)

    most_frequent_values: dict[str, int] = Field(
        default_factory=dict
    )

    numeric_statistics: dict[str, Any] = Field(
        default_factory=dict
    )

    text_statistics: dict[str, Any] = Field(
        default_factory=dict
    )

    datetime_statistics: dict[str, Any] = Field(
        default_factory=dict
    )

    patterns: list[PatternResult] = Field(
        default_factory=list
    )


class TableProfile(BaseModel):
    table_name: str

    row_count: int
    column_count: int

    duplicate_row_count: int
    duplicate_percentage: float

    columns: list[ColumnProfile] = Field(
        default_factory=list
    )


class DocumentProfile(BaseModel):
    document_name: str

    character_count: int
    word_count: int

    line_count: int
    non_empty_line_count: int

    preview: str = ""


class DataProfile(BaseModel):
    source_name: str
    source_type: str

    table_count: int
    document_count: int

    tables: list[TableProfile] = Field(
        default_factory=list
    )

    documents: list[DocumentProfile] = Field(
        default_factory=list
    )


# -------------------------------------------------------------------
# AGENTIC TOOL MODELS
# -------------------------------------------------------------------

class ToolRequest(BaseModel):
    """
    A request made by the reasoning model for additional analysis.
    """

    tool_name: str

    table_name: str | None = None
    column_name: str | None = None
    second_column_name: str | None = None

    parameters: dict[str, Any] = Field(
        default_factory=dict
    )


class ToolResult(BaseModel):
    tool_name: str

    success: bool

    table_name: str | None = None
    column_name: str | None = None

    result: Any = None
    message: str = ""


class AnalysisHypothesis(BaseModel):
    statement: str

    confidence: float = Field(
        ge=0.0,
        le=1.0,
    )

    evidence: str = ""


class AgentDecision(BaseModel):
    """
    Returned by the local LLM after each reasoning round.
    """

    understanding: str = ""

    hypotheses: list[AnalysisHypothesis] = Field(
        default_factory=list
    )

    tool_requests: list[ToolRequest] = Field(
        default_factory=list
    )

    ready_to_finalize: bool = False


# -------------------------------------------------------------------
# SEMANTIC UNDERSTANDING
# -------------------------------------------------------------------

class ColumnSemanticInsight(BaseModel):
    table_name: str
    column_name: str

    semantic_type: str

    description: str = ""

    confidence: float = Field(
        ge=0.0,
        le=1.0,
    )

    evidence: list[str] = Field(
        default_factory=list
    )


class DatasetUnderstanding(BaseModel):
    detected_domain: str = "unknown"

    dataset_description: str = ""

    column_insights: list[ColumnSemanticInsight] = Field(
        default_factory=list
    )


# -------------------------------------------------------------------
# DATA QUALITY MODELS
# -------------------------------------------------------------------

class DataQualityRule(BaseModel):
    rule_id: str

    table_name: str
    columns: list[str]

    rule_type: str

    parameters: dict[str, Any] = Field(
        default_factory=dict
    )

    severity: Literal[
        "critical",
        "high",
        "medium",
        "low",
    ] = "medium"

    confidence: float = Field(
        ge=0.0,
        le=1.0,
    )

    reason: str

    generated_by: Literal[
        "deterministic",
        "agentic_ai",
        "business",
    ] = "agentic_ai"


class RuleExecutionResult(BaseModel):
    rule_id: str

    status: Literal[
        "passed",
        "failed",
        "skipped",
        "error",
    ]

    rows_checked: int = 0
    failed_rows: int = 0

    pass_percentage: float | None = None

    failed_row_indexes: list[int] = Field(
        default_factory=list
    )

    message: str = ""


class QualityFinding(BaseModel):
    title: str

    description: str

    category: str

    severity: Literal[
        "critical",
        "high",
        "medium",
        "low",
        "info",
    ]

    table_name: str | None = None
    column_name: str | None = None

    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
    )

    evidence: list[str] = Field(
        default_factory=list
    )


# -------------------------------------------------------------------
# SOURCE-TO-TARGET MAPPING MODELS
# -------------------------------------------------------------------

class TargetColumn(BaseModel):
    column_name: str

    data_type: str

    nullable: bool = True
    primary_key: bool = False

    description: str = ""


class TargetTable(BaseModel):
    table_name: str

    columns: list[TargetColumn] = Field(
        default_factory=list
    )


class TargetSchema(BaseModel):
    tables: list[TargetTable] = Field(
        default_factory=list
    )


class MappingSuggestion(BaseModel):
    source_table: str
    source_column: str

    target_table: str
    target_column: str

    transformation: str = "direct"

    confidence: float = Field(
        ge=0.0,
        le=1.0,
    )

    reason: str

    requires_review: bool = False


# -------------------------------------------------------------------
# UNSTRUCTURED DOCUMENT MODELS
# -------------------------------------------------------------------

class ExtractedDocumentTable(BaseModel):
    table_name: str

    columns: list[str] = Field(
        default_factory=list
    )

    rows: list[list[Any]] = Field(
        default_factory=list
    )


class DocumentExtraction(BaseModel):
    document_type: str = "unknown"

    summary: str = ""

    extracted_fields: dict[str, Any] = Field(
        default_factory=dict
    )

    tables: list[ExtractedDocumentTable] = Field(
        default_factory=list
    )


# -------------------------------------------------------------------
# FINAL ASSESSMENT
# -------------------------------------------------------------------

class AssessmentResult(BaseModel):
    source_name: str
    source_type: str

    summary: str

    detected_domain: str = "unknown"

    profile: DataProfile

    semantic_insights: list[ColumnSemanticInsight] = Field(
        default_factory=list
    )

    findings: list[QualityFinding] = Field(
        default_factory=list
    )

    generated_rules: list[DataQualityRule] = Field(
        default_factory=list
    )

    rule_results: list[RuleExecutionResult] = Field(
        default_factory=list
    )

    mappings: list[MappingSuggestion] = Field(
        default_factory=list
    )

    recommendations: list[str] = Field(
        default_factory=list
    )

    quality_score: float | None = None