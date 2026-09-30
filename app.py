from __future__ import annotations

import hashlib
import html
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from src.database import database_status
from src.main import run_pipeline


# -------------------------------------------------------------------
# PAGE
# -------------------------------------------------------------------

st.set_page_config(
    page_title="Agentic Data Intelligence",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -------------------------------------------------------------------
# DESIGN
# -------------------------------------------------------------------

st.markdown(
    """
    <style>
        .block-container {
            max-width: 1500px;
            padding-top: 2rem;
            padding-bottom: 4rem;
        }

        .main-title {
            font-size: 2.35rem;
            font-weight: 750;
            letter-spacing: -0.035em;
            margin-bottom: 0.25rem;
        }

        .main-subtitle {
            color: #9a9d96;
            font-size: 1rem;
            margin-bottom: 1.8rem;
        }

        .section-note {
            color: #92958f;
            font-size: 0.9rem;
        }

        /* -----------------------------------------------------------
           SUMMARY BOX
        ----------------------------------------------------------- */

        .status-box {
            border: 1px solid rgba(255, 255, 255, 0.12);
            border-radius: 12px;
            padding: 16px 18px;
            background: rgba(255, 255, 255, 0.035);
            margin: 18px 0 22px 0;
            line-height: 1.65;
            color: inherit;
        }

        /* -----------------------------------------------------------
           METRICS
        ----------------------------------------------------------- */

        div[data-testid="stMetric"] {
            border: 1px solid rgba(255, 255, 255, 0.12);
            border-radius: 12px;
            padding: 14px 16px;
            background: rgba(255, 255, 255, 0.045);
        }

        div[data-testid="stMetric"] label {
            color: #a9ada7 !important;
        }

        div[data-testid="stMetricValue"] {
            color: #f4f4f1 !important;
        }

        div[data-testid="stMetricDelta"] {
            color: #c8cbc5 !important;
        }

        /* -----------------------------------------------------------
           FILE UPLOADER
        ----------------------------------------------------------- */

        div[data-testid="stFileUploader"] {
            border-radius: 12px;
        }

        /* -----------------------------------------------------------
           BUTTONS
        ----------------------------------------------------------- */

        .stButton > button {
            border-radius: 9px;
            font-weight: 650;
        }

        /* -----------------------------------------------------------
           DATAFRAMES
        ----------------------------------------------------------- */

        div[data-testid="stDataFrame"] {
            border-radius: 10px;
            overflow: hidden;
        }

        /* -----------------------------------------------------------
           TABS
        ----------------------------------------------------------- */

        button[data-baseweb="tab"] {
            font-weight: 600;
        }

        /* -----------------------------------------------------------
           CODE
        ----------------------------------------------------------- */

        code {
            font-size: 0.88em;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# -------------------------------------------------------------------
# CONSTANTS
# -------------------------------------------------------------------

SUPPORTED_TYPES = [
    "csv",
    "tsv",
    "xlsx",
    "xlsm",
    "xls",
    "json",
    "jsonl",
    "ndjson",
    "xml",
    "parquet",
    "txt",
    "log",
    "md",
    "pdf",
    "docx",
]


# -------------------------------------------------------------------
# HELPERS
# -------------------------------------------------------------------

def _uploaded_file_signature(
    file_bytes: bytes,
    target_tables: list[str],
) -> str:
    digest = hashlib.sha256()

    digest.update(
        file_bytes
    )

    for table in sorted(
        target_tables
    ):
        digest.update(
            table.encode(
                "utf-8"
            )
        )

    return digest.hexdigest()


def _save_temporary_upload(
    uploaded_file,
    directory: str,
) -> Path:
    """
    Preserve the original filename while keeping uploaded data
    temporary.
    """

    safe_name = Path(
        uploaded_file.name
    ).name

    destination = (
        Path(directory)
        / safe_name
    )

    destination.write_bytes(
        uploaded_file.getvalue()
    )

    return destination


def _parse_target_tables(
    value: str,
) -> list[str]:
    tables = [
        item.strip()
        for item in value.split(",")
        if item.strip()
    ]

    return list(
        dict.fromkeys(
            tables
        )
    )


def _profile_dataframe(
    result,
) -> pd.DataFrame:
    rows = []

    for table in (
        result.assessment.profile.tables
    ):
        for column in table.columns:
            patterns = ", ".join(
                (
                    f"{pattern.pattern_name} "
                    f"({pattern.match_percentage:.1f}%)"
                )
                for pattern
                in column.patterns
            )

            numeric = (
                column.numeric_statistics
                or {}
            )

            datetime_stats = (
                column.datetime_statistics
                or {}
            )

            rows.append(
                {
                    "Table": table.table_name,
                    "Column": column.column_name,
                    "Type": column.inferred_type,
                    "Rows": column.row_count,
                    "Null %": column.null_percentage,
                    "Distinct %": (
                        column.distinct_percentage
                    ),
                    "Min": numeric.get(
                        "minimum"
                    ),
                    "Max": numeric.get(
                        "maximum"
                    ),
                    "Median": numeric.get(
                        "median"
                    ),
                    "MAD outliers": numeric.get(
                        "mad_outlier_count"
                    ),
                    "Earliest": (
                        datetime_stats.get(
                            "earliest"
                        )
                    ),
                    "Latest": (
                        datetime_stats.get(
                            "latest"
                        )
                    ),
                    "Patterns": patterns,
                }
            )

    return pd.DataFrame(
        rows
    )


def _table_overview_dataframe(
    result,
) -> pd.DataFrame:
    rows = []

    for table in (
        result.assessment.profile.tables
    ):
        rows.append(
            {
                "Table": table.table_name,
                "Rows": table.row_count,
                "Columns": table.column_count,
                "Duplicate rows": (
                    table.duplicate_row_count
                ),
                "Duplicate %": (
                    table.duplicate_percentage
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def _rules_dataframe(
    result,
) -> pd.DataFrame:
    execution_lookup = {
        execution.rule_id: execution
        for execution
        in result.assessment.rule_results
    }

    rows = []

    for rule in (
        result.assessment.generated_rules
    ):
        execution = (
            execution_lookup.get(
                rule.rule_id
            )
        )

        rows.append(
            {
                "Rule": rule.rule_type,
                "Table": rule.table_name,
                "Column(s)": ", ".join(
                    rule.columns
                ),
                "Severity": rule.severity,
                "Confidence": rule.confidence,
                "Generated by": (
                    rule.generated_by
                ),
                "Status": (
                    execution.status
                    if execution
                    else "not executed"
                ),
                "Rows checked": (
                    execution.rows_checked
                    if execution
                    else None
                ),
                "Failed rows": (
                    execution.failed_rows
                    if execution
                    else None
                ),
                "Pass %": (
                    execution.pass_percentage
                    if execution
                    else None
                ),
                "Reason": rule.reason,
            }
        )

    return pd.DataFrame(
        rows
    )


def _findings_dataframe(
    result,
) -> pd.DataFrame:
    rows = []

    for finding in (
        result.assessment.findings
    ):
        rows.append(
            {
                "Finding": finding.title,
                "Category": finding.category,
                "Severity": finding.severity,
                "Table": finding.table_name,
                "Column": finding.column_name,
                "Confidence": (
                    finding.confidence
                ),
                "Description": (
                    finding.description
                ),
                "Evidence": " | ".join(
                    str(item)
                    for item
                    in finding.evidence
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def _mappings_dataframe(
    result,
) -> pd.DataFrame:
    rows = []

    for mapping in result.mappings:
        rows.append(
            {
                "Source": (
                    f"{mapping.source_table}."
                    f"{mapping.source_column}"
                ),
                "Target": (
                    f"{mapping.target_table}."
                    f"{mapping.target_column}"
                ),
                "Transformation": (
                    mapping.transformation
                ),
                "Confidence": (
                    mapping.confidence
                ),
                "Requires review": (
                    mapping.requires_review
                ),
                "Reason": mapping.reason,
            }
        )

    return pd.DataFrame(
        rows
    )


def _mapping_key(
    mapping,
) -> tuple[str, str, str, str]:
    return (
        mapping.source_table,
        mapping.source_column,
        mapping.target_table,
        mapping.target_column,
    )


def _show_report_downloads(
    result,
) -> None:
    columns = st.columns(
        2
    )

    json_path = (
        result.report_files.get(
            "json"
        )
    )

    html_path = (
        result.report_files.get(
            "html"
        )
    )

    if (
        json_path
        and Path(
            json_path
        ).exists()
    ):
        with columns[0]:
            st.download_button(
                label="Download JSON report",
                data=Path(
                    json_path
                ).read_bytes(),
                file_name=Path(
                    json_path
                ).name,
                mime="application/json",
                use_container_width=True,
            )

    if (
        html_path
        and Path(
            html_path
        ).exists()
    ):
        with columns[1]:
            st.download_button(
                label="Download HTML report",
                data=Path(
                    html_path
                ).read_bytes(),
                file_name=Path(
                    html_path
                ).name,
                mime="text/html",
                use_container_width=True,
            )


# -------------------------------------------------------------------
# SESSION STATE
# -------------------------------------------------------------------

if (
    "analysis_result"
    not in st.session_state
):
    st.session_state[
        "analysis_result"
    ] = None

if (
    "analysis_signature"
    not in st.session_state
):
    st.session_state[
        "analysis_signature"
    ] = None

if (
    "database_load_signature"
    not in st.session_state
):
    st.session_state[
        "database_load_signature"
    ] = None

if (
    "database_load_result"
    not in st.session_state
):
    st.session_state[
        "database_load_result"
    ] = None


# -------------------------------------------------------------------
# SIDEBAR
# -------------------------------------------------------------------

with st.sidebar:
    st.header(
        "Pipeline configuration"
    )

    st.caption(
        "Analysis never writes to PostgreSQL. "
        "Database loading requires a separate explicit action."
    )

    max_rounds = st.slider(
        "Maximum agent rounds",
        min_value=1,
        max_value=4,
        value=2,
        step=1,
        help=(
            "The Agentic AI can stop earlier when "
            "additional investigation is unnecessary."
        ),
    )

    st.divider()

    st.subheader(
        "Target database"
    )

    target_input = st.text_input(
        "Target table(s)",
        placeholder="quality_records",
        help=(
            "Optional. Separate multiple existing PostgreSQL "
            "tables with commas. Supplying a target enables "
            "source-to-target mapping."
        ),
    )

    target_tables = (
        _parse_target_tables(
            target_input
        )
    )

    db = database_status()

    st.write(
        "Database access:",
        "Enabled"
        if db["enabled"]
        else "Disabled",
    )

    st.write(
        "Database configured:",
        "Yes"
        if db["configured"]
        else "No",
    )

    st.write(
        "Schema:",
        db["schema"],
    )


# -------------------------------------------------------------------
# HEADER
# -------------------------------------------------------------------

st.markdown(
    '<div class="main-title">'
    "Agentic Data Intelligence"
    "</div>",
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="main-subtitle">'
    "Profile, investigate, validate, map and prepare structured "
    "or unstructured data using deterministic analytics and "
    "guarded local Agentic AI."
    "</div>",
    unsafe_allow_html=True,
)


# -------------------------------------------------------------------
# UPLOAD
# -------------------------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload a dataset or document",
    type=SUPPORTED_TYPES,
    help=(
        "Structured: CSV, TSV, Excel, JSON, JSONL, XML, "
        "Parquet. Unstructured: TXT, LOG, Markdown, PDF, DOCX."
    ),
)

if uploaded_file is None:
    st.info(
        "Upload a file above to start the Data Profiler Agent."
    )

    st.stop()


file_bytes = (
    uploaded_file.getvalue()
)

signature = (
    _uploaded_file_signature(
        file_bytes,
        target_tables,
    )
)

file_columns = st.columns(
    [
        2,
        1,
        1,
    ]
)

with file_columns[0]:
    st.write(
        f"**File:** {uploaded_file.name}"
    )

with file_columns[1]:
    st.write(
        f"**Size:** "
        f"{len(file_bytes) / 1024:.1f} KB"
    )

with file_columns[2]:
    st.write(
        f"**Target mapping:** "
        f"{'Enabled' if target_tables else 'Off'}"
    )


analyze = st.button(
    "Analyze dataset",
    type="primary",
    use_container_width=True,
)


# -------------------------------------------------------------------
# RUN ANALYSIS
# -------------------------------------------------------------------

if analyze:
    try:
        with st.spinner(
            "Running ingestion, profiling, Agentic AI and "
            "data-quality analysis..."
        ):
            with tempfile.TemporaryDirectory() as directory:
                source_path = (
                    _save_temporary_upload(
                        uploaded_file,
                        directory,
                    )
                )

                result = (
                    run_pipeline(
                        source_path=source_path,
                        target_tables=(
                            target_tables
                            or None
                        ),
                        approved_review_mappings=set(),
                        load_database=False,
                        max_agent_rounds=(
                            max_rounds
                        ),
                    )
                )

        st.session_state[
            "analysis_result"
        ] = result

        st.session_state[
            "analysis_signature"
        ] = signature

        st.session_state[
            "database_load_result"
        ] = None

        st.session_state[
            "database_load_signature"
        ] = None

        st.success(
            "Analysis completed successfully."
        )

    except Exception as exc:
        st.error(
            f"Analysis failed: {exc}"
        )

        st.stop()


# -------------------------------------------------------------------
# CURRENT RESULT
# -------------------------------------------------------------------

result = st.session_state[
    "analysis_result"
]

if result is None:
    st.stop()


if (
    st.session_state[
        "analysis_signature"
    ]
    != signature
):
    st.warning(
        "The uploaded file or target-table configuration changed. "
        "Click Analyze dataset again before using these results."
    )

    st.stop()


# -------------------------------------------------------------------
# SUMMARY
# -------------------------------------------------------------------

st.divider()

st.subheader(
    "Assessment overview"
)

metric_columns = st.columns(
    6
)

metric_columns[0].metric(
    "Quality score",
    f"{result.assessment.quality_score:.2f}",
)

metric_columns[1].metric(
    "Tables",
    result.assessment.profile.table_count,
)

metric_columns[2].metric(
    "DQ rules",
    len(
        result.assessment.generated_rules
    ),
)

metric_columns[3].metric(
    "Valid rows",
    result.partition.valid_row_count,
)

metric_columns[4].metric(
    "Rejected rows",
    result.partition.rejected_row_count,
)

metric_columns[5].metric(
    "Agent rounds",
    result.agent.rounds_used,
)

st.caption(
    "Quality score is based only on validated rules that were "
    "actually executed; it does not imply that unknown business "
    "requirements have been satisfied."
)

safe_summary = html.escape(
    result.assessment.summary
)

st.markdown(
    f"""
    <div class="status-box">
        {safe_summary}
    </div>
    """,
    unsafe_allow_html=True,
)


# -------------------------------------------------------------------
# MAIN TABS
# -------------------------------------------------------------------

(
    profiler_tab,
    agent_tab,
    dq_tab,
    rejected_tab,
    clean_tab,
    mapping_tab,
    report_tab,
) = st.tabs(
    [
        "Data Profiler",
        "Agentic Analysis",
        "Data Quality",
        "Rejected Rows",
        "Clean Dataset",
        "Source → Target",
        "Report",
    ]
)


# -------------------------------------------------------------------
# DATA PROFILER
# -------------------------------------------------------------------

with profiler_tab:
    st.subheader(
        "Dataset overview"
    )

    overview = (
        _table_overview_dataframe(
            result
        )
    )

    st.dataframe(
        overview,
        use_container_width=True,
        hide_index=True,
    )

    st.subheader(
        "Column profiling"
    )

    profile_frame = (
        _profile_dataframe(
            result
        )
    )

    st.dataframe(
        profile_frame,
        use_container_width=True,
        hide_index=True,
    )

    documents = (
        result.assessment
        .profile
        .documents
    )

    if documents:
        st.subheader(
            "Document sources"
        )

        for document in documents:
            with st.expander(
                document.document_name
            ):
                st.write(
                    f"Characters: "
                    f"{document.character_count:,}"
                )

                st.write(
                    f"Words: "
                    f"{document.word_count:,}"
                )

                st.text(
                    document.preview
                )


# -------------------------------------------------------------------
# AGENT
# -------------------------------------------------------------------

with agent_tab:
    st.subheader(
        "Agent understanding"
    )

    st.write(
        result.agent
        .final_decision
        .understanding
    )

    agent_columns = st.columns(
        3
    )

    agent_columns[0].metric(
        "Reasoning rounds",
        result.agent.rounds_used,
    )

    agent_columns[1].metric(
        "Tools executed",
        len(
            result.agent.tool_results
        ),
    )

    agent_columns[2].metric(
        "Hypotheses",
        len(
            result.agent
            .final_decision
            .hypotheses
        ),
    )

    if (
        result.agent
        .final_decision
        .hypotheses
    ):
        st.subheader(
            "Evidence-backed hypotheses"
        )

        for hypothesis in (
            result.agent
            .final_decision
            .hypotheses
        ):
            st.write(
                f"**{hypothesis.statement}**"
            )

            st.caption(
                f"Confidence: "
                f"{hypothesis.confidence:.2f}"
            )

            for evidence in (
                hypothesis.evidence
            ):
                st.write(
                    f"- {evidence}"
                )

    if result.agent.tool_results:
        st.subheader(
            "Agent tool activity"
        )

        for tool in (
            result.agent.tool_results
        ):
            with st.expander(
                f"{tool.tool_name} "
                f"— {'success' if tool.success else 'failed'}"
            ):
                st.json(
                    tool.model_dump(
                        mode="json"
                    )
                )


# -------------------------------------------------------------------
# DATA QUALITY
# -------------------------------------------------------------------

with dq_tab:
    st.subheader(
        "Generated and executed rules"
    )

    rules_frame = (
        _rules_dataframe(
            result
        )
    )

    if rules_frame.empty:
        st.info(
            "No validated data-quality rules were generated."
        )

    else:
        st.dataframe(
            rules_frame,
            use_container_width=True,
            hide_index=True,
        )

    st.subheader(
        "Findings"
    )

    findings = (
        _findings_dataframe(
            result
        )
    )

    if findings.empty:
        st.success(
            "No validated quality findings were generated."
        )

    else:
        st.dataframe(
            findings,
            use_container_width=True,
            hide_index=True,
        )


# -------------------------------------------------------------------
# REJECTED ROWS
# -------------------------------------------------------------------

with rejected_tab:
    if (
        result.partition
        .rejected_row_count
        == 0
    ):
        st.success(
            "No rows were rejected by blocking data-quality rules."
        )

    else:
        st.warning(
            f"{result.partition.rejected_row_count} row(s) "
            "were blocked from the downstream clean-data path."
        )

        for (
            table_name,
            dataframe,
        ) in (
            result.partition
            .invalid_tables
            .items()
        ):
            if dataframe.empty:
                continue

            st.subheader(
                table_name
            )

            st.dataframe(
                dataframe,
                use_container_width=True,
                hide_index=True,
            )


# -------------------------------------------------------------------
# CLEAN DATASET
# -------------------------------------------------------------------

with clean_tab:
    st.subheader(
        "Clean Dataset"
    )

    valid_rows = (
        result.partition.valid_row_count
    )

    rejected_rows = (
        result.partition.rejected_row_count
    )

    total_rows = (
        valid_rows
        + rejected_rows
    )

    retention_rate = (
        (
            valid_rows
            / total_rows
        )
        * 100
        if total_rows > 0
        else 0.0
    )

    clean_metrics = st.columns(
        3
    )

    clean_metrics[0].metric(
        "Clean rows",
        valid_rows,
    )

    clean_metrics[1].metric(
        "Rejected rows",
        rejected_rows,
    )

    clean_metrics[2].metric(
        "Retention rate",
        f"{retention_rate:.1f}%",
    )

    st.caption(
        "The clean dataset contains only rows that passed all "
        "blocking data-quality rules. Original source values are "
        "preserved; the system does not silently repair or invent "
        "replacement values."
    )

    clean_files = getattr(
        result,
        "clean_files",
        {},
    )

    if not clean_files:
        st.warning(
            "No clean CSV was generated for this analysis. "
            "If this result was created before the clean-dataset "
            "feature was added, click Analyze dataset again."
        )

    else:
        for (
            table_name,
            clean_file,
        ) in clean_files.items():
            clean_path = Path(
                clean_file
            )

            st.divider()

            st.subheader(
                table_name
            )

            if not clean_path.exists():
                st.warning(
                    f"The clean CSV for '{table_name}' "
                    "could not be found on disk."
                )

                continue

            try:
                clean_frame = (
                    pd.read_csv(
                        clean_path
                    )
                )

            except pd.errors.EmptyDataError:
                clean_frame = (
                    pd.DataFrame()
                )

            st.write(
                f"**Generated file:** `{clean_path.name}`"
            )

            if clean_frame.empty:
                st.info(
                    "The clean CSV was generated, but it contains "
                    "no valid data rows."
                )

            else:
                st.dataframe(
                    clean_frame,
                    use_container_width=True,
                    hide_index=True,
                )

            st.download_button(
                label=(
                    f"Download {clean_path.name}"
                ),
                data=(
                    clean_path.read_bytes()
                ),
                file_name=(
                    clean_path.name
                ),
                mime="text/csv",
                use_container_width=True,
                key=(
                    f"download_clean_"
                    f"{table_name}_"
                    f"{signature}"
                ),
            )


# -------------------------------------------------------------------
# MAPPING
# -------------------------------------------------------------------

with mapping_tab:
    if not target_tables:
        st.info(
            "Enter an existing PostgreSQL target table in the "
            "sidebar and run the analysis again to generate "
            "source-to-target mappings."
        )

    elif not result.mappings:
        st.warning(
            "No supported mappings were generated for the "
            "selected target schema."
        )

    else:
        st.subheader(
            "Mapping suggestions"
        )

        mapping_frame = (
            _mappings_dataframe(
                result
            )
        )

        st.dataframe(
            mapping_frame,
            use_container_width=True,
            hide_index=True,
        )

        review_mappings = [
            mapping
            for mapping in result.mappings
            if mapping.requires_review
        ]

        approved_keys: set[
            tuple[
                str,
                str,
                str,
                str,
            ]
        ] = set()

        if review_mappings:
            st.subheader(
                "Manual review"
            )

            st.caption(
                "These mappings were not strong enough for "
                "automatic approval."
            )

            for index, mapping in enumerate(
                review_mappings
            ):
                approved = st.checkbox(
                    (
                        f"{mapping.source_table}."
                        f"{mapping.source_column} → "
                        f"{mapping.target_table}."
                        f"{mapping.target_column} "
                        f"({mapping.confidence:.4f})"
                    ),
                    key=(
                        f"mapping_approval_{index}_"
                        f"{signature}"
                    ),
                )

                if approved:
                    approved_keys.add(
                        _mapping_key(
                            mapping
                        )
                    )

        unresolved_count = (
            len(
                review_mappings
            )
            - len(
                approved_keys
            )
        )

        st.divider()

        st.subheader(
            "PostgreSQL load"
        )

        if not db["enabled"]:
            st.warning(
                "DATABASE_ENABLED is false. "
                "Database loading is unavailable."
            )

        elif not db["configured"]:
            st.warning(
                "DATABASE_URL is not configured."
            )

        else:
            st.caption(
                "Only DQ-valid rows are eligible for loading. "
                "Rejected rows remain excluded."
            )

            confirmation = st.checkbox(
                (
                    "I reviewed the mappings and understand "
                    "that this action inserts valid rows into "
                    "the configured PostgreSQL database."
                ),
                key=(
                    f"db_confirmation_{signature}"
                ),
            )

            already_loaded = (
                st.session_state[
                    "database_load_signature"
                ]
                == signature
            )

            if already_loaded:
                st.success(
                    "This file/target combination was already "
                    "loaded during the current application session."
                )

            if unresolved_count > 0:
                st.warning(
                    f"{unresolved_count} reviewed mapping(s) "
                    "still need explicit approval."
                )

            load_disabled = (
                not confirmation
                or unresolved_count > 0
                or already_loaded
            )

            load_clicked = st.button(
                "Load valid rows to PostgreSQL",
                disabled=load_disabled,
                use_container_width=True,
            )

            if load_clicked:
                try:
                    with st.spinner(
                        "Revalidating the pipeline and loading "
                        "DQ-valid rows transactionally..."
                    ):
                        with tempfile.TemporaryDirectory() as directory:
                            source_path = (
                                _save_temporary_upload(
                                    uploaded_file,
                                    directory,
                                )
                            )

                            loaded_result = (
                                run_pipeline(
                                    source_path=source_path,
                                    target_tables=(
                                        target_tables
                                    ),
                                    approved_review_mappings=(
                                        approved_keys
                                    ),
                                    load_database=True,
                                    max_agent_rounds=(
                                        max_rounds
                                    ),
                                )
                            )

                    st.session_state[
                        "database_load_result"
                    ] = loaded_result.database_load

                    st.session_state[
                        "database_load_signature"
                    ] = signature

                    st.success(
                        "Database transaction completed successfully."
                    )

                except Exception as exc:
                    st.error(
                        f"Database load failed: {exc}"
                    )

            load_result = (
                st.session_state[
                    "database_load_result"
                ]
            )

            if (
                load_result is not None
                and st.session_state[
                    "database_load_signature"
                ]
                == signature
            ):
                load_columns = (
                    st.columns(
                        3
                    )
                )

                load_columns[0].metric(
                    "Load status",
                    (
                        "Success"
                        if load_result.success
                        else "Failed"
                    ),
                )

                load_columns[1].metric(
                    "Rows inserted",
                    (
                        load_result
                        .total_rows_inserted
                    ),
                )

                load_columns[2].metric(
                    "Schema",
                    (
                        load_result
                        .schema_name
                    ),
                )


# -------------------------------------------------------------------
# REPORT
# -------------------------------------------------------------------

with report_tab:
    st.subheader(
        "Executive summary"
    )

    st.write(
        result.assessment.summary
    )

    st.write(
        "**Detected domain:**",
        result.assessment.detected_domain,
    )

    st.subheader(
        "Recommendations"
    )

    for recommendation in (
        result.assessment
        .recommendations
    ):
        st.write(
            f"- {recommendation}"
        )

    st.subheader(
        "Download assessment"
    )

    _show_report_downloads(
        result
    )

    with st.expander(
        "View raw assessment JSON"
    ):
        st.json(
            result.assessment.model_dump(
                mode="json"
            )
        )


# -------------------------------------------------------------------
# FOOTER
# -------------------------------------------------------------------

st.divider()

st.caption(
    "Agentic Data Intelligence · deterministic profiling and "
    "validation with guarded local-model reasoning."
)