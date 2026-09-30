from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from src.agent import AgentRunResult, run_agentic_analysis
from src.config import settings
from src.database import (
    DatabaseError,
    DatabaseLoadResult,
    inspect_target_schema,
    load_transformed_data,
)
from src.document_extractor import extract_unstructured_data
from src.ingestion import load_asset
from src.mapper import generate_mappings
from src.models import (
    AssessmentResult,
    DataAsset,
    MappingSuggestion,
    TargetSchema,
)
from src.profiler import profile_asset
from src.report import (
    build_assessment,
    save_assessment_report,
)
from src.rule_executor import (
    RowPartitionResult,
    execute_rules,
    partition_rows,
)
from src.rule_generator import generate_rules
from src.semantic_analyzer import analyze_semantics
from src.transformer import (
    MappingKey,
    transform_data,
)


# -------------------------------------------------------------------
# RESULT
# -------------------------------------------------------------------

@dataclass
class PipelineRunResult:
    """
    Final result of one complete pipeline run.
    """

    assessment: AssessmentResult
    partition: RowPartitionResult
    agent: AgentRunResult

    mappings: list[MappingSuggestion]

    report_files: dict[str, Path]
    clean_files: dict[str, Path]

    database_load: DatabaseLoadResult | None

    elapsed_seconds: float


# -------------------------------------------------------------------
# INPUT PREPARATION
# -------------------------------------------------------------------

def _prepare_asset(
    source_path: str | Path,
) -> DataAsset:
    """
    Load any supported source.

    When document text is present, the guarded document extractor is
    allowed to create additional source-grounded structured tables.
    """

    asset = load_asset(
        source_path
    )

    if not asset.documents:
        return asset

    extraction = (
        extract_unstructured_data(
            asset
        )
    )

    return (
        extraction.structured_asset
    )


# -------------------------------------------------------------------
# CLEAN DATASET
# -------------------------------------------------------------------

def _safe_file_part(
    value: str,
    *,
    fallback: str,
) -> str:
    """
    Convert a source or table name into a safe output filename part.
    """

    cleaned = (
        re.sub(
            r"[^A-Za-z0-9_-]+",
            "_",
            value,
        )
        .strip("_")
        .lower()
    )

    return (
        cleaned
        or fallback
    )


def _save_clean_datasets(
    source_name: str,
    partition: RowPartitionResult,
) -> dict[str, Path]:
    """
    Save all DQ-valid rows as clean CSV files.

    Clean datasets contain only original source columns and values.
    DQ rejection metadata is kept only in the rejected-data path and
    is never added to clean records.
    """

    clean_directory = (
        Path(
            settings.report_directory
        )
        / "clean_data"
    )

    clean_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    source_stem = (
        _safe_file_part(
            Path(
                source_name
            ).stem,
            fallback="dataset",
        )
    )

    valid_tables = (
        partition.valid_tables
    )

    multiple_tables = (
        len(
            valid_tables
        )
        > 1
    )

    clean_files: dict[
        str,
        Path,
    ] = {}

    for (
        table_name,
        dataframe,
    ) in valid_tables.items():

        if multiple_tables:
            safe_table_name = (
                _safe_file_part(
                    table_name,
                    fallback="table",
                )
            )

            file_name = (
                f"{source_stem}_"
                f"{safe_table_name}_clean.csv"
            )

        else:
            file_name = (
                f"{source_stem}_clean.csv"
            )

        output_path = (
            clean_directory
            / file_name
        )

        dataframe.to_csv(
            output_path,
            index=False,
            encoding="utf-8",
        )

        clean_files[
            table_name
        ] = output_path

    return clean_files


# -------------------------------------------------------------------
# MAPPING APPROVALS
# -------------------------------------------------------------------

def _parse_mapping_reference(
    value: str,
) -> MappingKey:
    """
    Parse:

        source_table.source_column=target_table.target_column
    """

    if "=" not in value:
        raise ValueError(
            "Mapping approval must use "
            "'source_table.source_column="
            "target_table.target_column'."
        )

    source, target = (
        value.split(
            "=",
            1,
        )
    )

    if (
        "." not in source
        or "." not in target
    ):
        raise ValueError(
            "Mapping approval must include both table and column "
            "names on each side."
        )

    source_table, source_column = (
        source.rsplit(
            ".",
            1,
        )
    )

    target_table, target_column = (
        target.rsplit(
            ".",
            1,
        )
    )

    values = [
        source_table,
        source_column,
        target_table,
        target_column,
    ]

    if any(
        not item.strip()
        for item in values
    ):
        raise ValueError(
            f"Invalid mapping approval '{value}'."
        )

    return (
        source_table.strip(),
        source_column.strip(),
        target_table.strip(),
        target_column.strip(),
    )


def _parse_mapping_approvals(
    values: list[str] | None,
) -> set[MappingKey]:
    approvals: set[
        MappingKey
    ] = set()

    for value in (
        values
        or []
    ):
        approvals.add(
            _parse_mapping_reference(
                value
            )
        )

    return approvals


def _mapping_key(
    mapping: MappingSuggestion,
) -> MappingKey:
    return (
        mapping.source_table,
        mapping.source_column,
        mapping.target_table,
        mapping.target_column,
    )


def _validate_mapping_approvals(
    mappings: list[
        MappingSuggestion
    ],
    approvals: set[
        MappingKey
    ],
) -> None:
    """
    Prevent a CLI approval from referring to a mapping that was never
    suggested by the pipeline.
    """

    available = {
        _mapping_key(
            mapping
        )
        for mapping in mappings
    }

    unknown = (
        approvals
        - available
    )

    if unknown:
        formatted = [
            (
                f"{source_table}.{source_column}="
                f"{target_table}.{target_column}"
            )
            for (
                source_table,
                source_column,
                target_table,
                target_column,
            ) in sorted(
                unknown
            )
        ]

        raise ValueError(
            "Approval references mapping(s) that were not generated: "
            + ", ".join(
                formatted
            )
        )


def _pending_review_mappings(
    mappings: list[
        MappingSuggestion
    ],
    approvals: set[
        MappingKey
    ],
) -> list[
    MappingSuggestion
]:
    return [
        mapping
        for mapping in mappings
        if (
            mapping.requires_review
            and _mapping_key(
                mapping
            )
            not in approvals
        )
    ]


# -------------------------------------------------------------------
# VALID-ASSET CREATION
# -------------------------------------------------------------------

def _valid_asset(
    asset: DataAsset,
    partition: RowPartitionResult,
) -> DataAsset:
    """
    Build a canonical asset containing only DQ-valid rows.

    Documents remain available for audit/provenance.
    """

    return DataAsset(
        source_path=(
            asset.source_path
        ),
        source_type=(
            asset.source_type
        ),
        tables=(
            partition.valid_tables
        ),
        documents=(
            asset.documents
        ),
    )


# -------------------------------------------------------------------
# TARGET SCHEMA
# -------------------------------------------------------------------

def _load_target_schema(
    target_tables: list[str] | None,
) -> TargetSchema | None:
    if not target_tables:
        return None

    return (
        inspect_target_schema(
            table_names=(
                target_tables
            )
        )
    )


# -------------------------------------------------------------------
# PIPELINE
# -------------------------------------------------------------------

def run_pipeline(
    source_path: str | Path,
    *,
    target_tables: list[str] | None = None,
    approved_review_mappings: set[
        MappingKey
    ] | None = None,
    load_database: bool = False,
    max_agent_rounds: int | None = None,
) -> PipelineRunResult:
    """
    Run the complete Agentic Data Intelligence pipeline.

    Database loading remains explicitly opt-in even when the database
    itself is enabled in .env.
    """

    started = time.perf_counter()

    approvals = (
        approved_review_mappings
        or set()
    )

    # ---------------------------------------------------------------
    # 1. UNIVERSAL INGESTION / DOCUMENT EXTRACTION
    # ---------------------------------------------------------------

    asset = (
        _prepare_asset(
            source_path
        )
    )

    # ---------------------------------------------------------------
    # 2. DETERMINISTIC PROFILING
    # ---------------------------------------------------------------

    profile = (
        profile_asset(
            asset
        )
    )

    # ---------------------------------------------------------------
    # 3. SEMANTIC UNDERSTANDING
    # ---------------------------------------------------------------

    semantics = (
        analyze_semantics(
            asset,
            profile,
        )
    )

    # ---------------------------------------------------------------
    # 4. AGENTIC INVESTIGATION
    # ---------------------------------------------------------------

    agent = (
        run_agentic_analysis(
            asset,
            profile,
            semantics,
            max_rounds=(
                max_agent_rounds
            ),
        )
    )

    # ---------------------------------------------------------------
    # 5. DYNAMIC DQ RULE GENERATION
    # ---------------------------------------------------------------

    rules = (
        generate_rules(
            asset,
            profile,
            semantics,
        )
    )

    # ---------------------------------------------------------------
    # 6. DETERMINISTIC DQ EXECUTION
    # ---------------------------------------------------------------

    rule_results = (
        execute_rules(
            asset,
            rules,
        )
    )

    # ---------------------------------------------------------------
    # 7. VALID / INVALID PARTITIONING
    # ---------------------------------------------------------------

    partition = (
        partition_rows(
            asset,
            rules,
            rule_results,
        )
    )

    # ---------------------------------------------------------------
    # 8. CLEAN CSV GENERATION
    # ---------------------------------------------------------------

    clean_files = (
        _save_clean_datasets(
            profile.source_name,
            partition,
        )
    )

    # ---------------------------------------------------------------
    # 9. OPTIONAL SOURCE-TARGET MAPPING
    # ---------------------------------------------------------------

    target_schema = (
        _load_target_schema(
            target_tables
        )
    )

    mappings: list[
        MappingSuggestion
    ] = []

    if target_schema is not None:
        mappings = (
            generate_mappings(
                asset,
                profile,
                semantics,
                target_schema,
            )
        )

        _validate_mapping_approvals(
            mappings,
            approvals,
        )

    # ---------------------------------------------------------------
    # 10. DATABASE LOAD
    # ---------------------------------------------------------------

    database_load: (
        DatabaseLoadResult
        | None
    ) = None

    if load_database:
        if target_schema is None:
            raise ValueError(
                "Database loading requires at least one "
                "--target-table."
            )

        if not mappings:
            raise ValueError(
                "Database loading was requested, but no "
                "source-to-target mappings were generated."
            )

        unresolved = (
            _pending_review_mappings(
                mappings,
                approvals,
            )
        )

        if unresolved:
            readable = [
                (
                    f"{mapping.source_table}."
                    f"{mapping.source_column} -> "
                    f"{mapping.target_table}."
                    f"{mapping.target_column} "
                    f"(confidence={mapping.confidence:.4f})"
                )
                for mapping in unresolved
            ]

            raise ValueError(
                "Database loading was blocked because mapping(s) "
                "still require explicit review:\n- "
                + "\n- ".join(
                    readable
                )
            )

        valid_asset = (
            _valid_asset(
                asset,
                partition,
            )
        )

        transformed = (
            transform_data(
                valid_asset,
                mappings,
                target_schema,
                approved_review_mappings=(
                    approvals
                ),
            )
        )

        if not transformed:
            raise ValueError(
                "Database loading was requested, but no "
                "approved transformed tables were produced."
            )

        database_load = (
            load_transformed_data(
                transformed,
                target_schema,
            )
        )

    # ---------------------------------------------------------------
    # 11. FINAL ASSESSMENT
    # ---------------------------------------------------------------

    assessment = (
        build_assessment(
            profile,
            semantics,
            rules,
            rule_results,
            mappings=mappings,
            partition=partition,
            agent_understanding=(
                agent
                .final_decision
                .understanding
            ),
        )
    )

    # ---------------------------------------------------------------
    # 12. REPORTS
    # ---------------------------------------------------------------

    report_files = (
        save_assessment_report(
            assessment
        )
    )

    elapsed = round(
        time.perf_counter()
        - started,
        2,
    )

    return PipelineRunResult(
        assessment=assessment,
        partition=partition,
        agent=agent,
        mappings=mappings,
        report_files=(
            report_files
        ),
        clean_files=(
            clean_files
        ),
        database_load=(
            database_load
        ),
        elapsed_seconds=(
            elapsed
        ),
    )


# -------------------------------------------------------------------
# CLI OUTPUT
# -------------------------------------------------------------------

def _mapping_summary(
    mappings: list[
        MappingSuggestion
    ],
) -> list[
    dict[str, object]
]:
    return [
        {
            "source": (
                f"{mapping.source_table}."
                f"{mapping.source_column}"
            ),
            "target": (
                f"{mapping.target_table}."
                f"{mapping.target_column}"
            ),
            "confidence": (
                mapping.confidence
            ),
            "requires_review": (
                mapping.requires_review
            ),
        }
        for mapping in mappings
    ]


def _result_summary(
    result: PipelineRunResult,
) -> dict[str, object]:
    database_summary = None

    if (
        result.database_load
        is not None
    ):
        database_summary = {
            "success": (
                result.database_load.success
            ),
            "rows_inserted": (
                result
                .database_load
                .total_rows_inserted
            ),
            "schema": (
                result
                .database_load
                .schema_name
            ),
        }

    return {
        "source": (
            result.assessment.source_name
        ),
        "source_type": (
            result.assessment.source_type
        ),
        "quality_score": (
            result.assessment.quality_score
        ),
        "tables_profiled": (
            result
            .assessment
            .profile
            .table_count
        ),
        "documents_profiled": (
            result
            .assessment
            .profile
            .document_count
        ),
        "agent_rounds": (
            result.agent.rounds_used
        ),
        "agent_tools_executed": len(
            result.agent.tool_results
        ),
        "dq_rules": len(
            result
            .assessment
            .generated_rules
        ),
        "valid_rows": (
            result
            .partition
            .valid_row_count
        ),
        "rejected_rows": (
            result
            .partition
            .rejected_row_count
        ),
        "clean_datasets": {
            table_name: str(
                path
            )
            for (
                table_name,
                path,
            ) in (
                result
                .clean_files
                .items()
            )
        },
        "mappings": (
            _mapping_summary(
                result.mappings
            )
        ),
        "database_load": (
            database_summary
        ),
        "reports": {
            key: str(
                path
            )
            for (
                key,
                path,
            ) in (
                result
                .report_files
                .items()
            )
        },
        "elapsed_seconds": (
            result.elapsed_seconds
        ),
    }


# -------------------------------------------------------------------
# ARGUMENT PARSER
# -------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Agentic Data Intelligence pipeline for structured and "
            "unstructured data."
        )
    )

    parser.add_argument(
        "source",
        help=(
            "Path to the source CSV, Excel, JSON, XML, Parquet, "
            "TXT, LOG, Markdown, PDF or DOCX file."
        ),
    )

    parser.add_argument(
        "--target-table",
        action="append",
        dest="target_tables",
        help=(
            "Existing PostgreSQL target table to inspect and map. "
            "Repeat this option for multiple target tables."
        ),
    )

    parser.add_argument(
        "--approve-mapping",
        action="append",
        dest="approved_mappings",
        help=(
            "Explicitly approve a reviewed mapping using "
            "'source_table.source_column="
            "target_table.target_column'. "
            "Repeat for multiple approvals."
        ),
    )

    parser.add_argument(
        "--load-db",
        action="store_true",
        help=(
            "Load DQ-valid transformed rows into PostgreSQL. "
            "Without this flag, no database write is performed."
        ),
    )

    parser.add_argument(
        "--max-agent-rounds",
        type=int,
        default=None,
        help=(
            "Maximum agent reasoning rounds for this run. "
            "Cannot exceed MAX_AGENT_ROUNDS from configuration."
        ),
    )

    return parser


# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------

def main() -> int:
    parser = (
        _build_parser()
    )

    args = (
        parser.parse_args()
    )

    try:
        approvals = (
            _parse_mapping_approvals(
                args.approved_mappings
            )
        )

        result = (
            run_pipeline(
                source_path=(
                    args.source
                ),
                target_tables=(
                    args.target_tables
                ),
                approved_review_mappings=(
                    approvals
                ),
                load_database=(
                    args.load_db
                ),
                max_agent_rounds=(
                    args.max_agent_rounds
                ),
            )
        )

        print(
            json.dumps(
                _result_summary(
                    result
                ),
                indent=2,
                ensure_ascii=False,
                default=str,
            )
        )

        return 0

    except (
        ValueError,
        FileNotFoundError,
        DatabaseError,
    ) as exc:
        print(
            f"Pipeline failed: {exc}",
            file=sys.stderr,
        )

        return 1

    except KeyboardInterrupt:
        print(
            "Pipeline interrupted.",
            file=sys.stderr,
        )

        return 130


if __name__ == "__main__":
    raise SystemExit(
        main()
    )