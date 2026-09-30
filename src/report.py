from __future__ import annotations

import html
import json
import re
from pathlib import Path
from typing import Iterable

from src.config import (
    ensure_runtime_directories,
    settings,
)
from src.models import (
    AssessmentResult,
    DataProfile,
    DataQualityRule,
    DatasetUnderstanding,
    MappingSuggestion,
    QualityFinding,
    RuleExecutionResult,
)
from src.rule_executor import RowPartitionResult


# -------------------------------------------------------------------
# SCORE
# -------------------------------------------------------------------

SEVERITY_WEIGHTS = {
    "critical": 4.0,
    "high": 3.0,
    "medium": 2.0,
    "low": 1.0,
}


def _quality_score(
    rules: list[DataQualityRule],
    results: list[RuleExecutionResult],
) -> float:
    """
    Calculate a transparent rule-based DQ score.

    The score represents performance against rules that were actually
    executed. It is not presented as proof that unknown business rules
    are satisfied.
    """

    rule_lookup = {
        rule.rule_id: rule
        for rule in rules
    }

    weighted_score = 0.0
    total_weight = 0.0

    for result in results:
        if result.status not in {
            "passed",
            "failed",
        }:
            continue

        rule = rule_lookup.get(
            result.rule_id
        )

        if rule is None:
            continue

        weight = SEVERITY_WEIGHTS.get(
            rule.severity,
            1.0,
        )

        weighted_score += (
            result.pass_percentage
            * weight
        )

        total_weight += weight

    if total_weight == 0:
        # No validated executable rule means there is no measured
        # failure. The report explicitly explains that this score is
        # only relative to available validated rules.
        return 100.0

    return round(
        weighted_score
        / total_weight,
        2,
    )


# -------------------------------------------------------------------
# FINDINGS
# -------------------------------------------------------------------

def _rule_findings(
    rules: list[DataQualityRule],
    results: list[RuleExecutionResult],
) -> list[QualityFinding]:
    rule_lookup = {
        rule.rule_id: rule
        for rule in rules
    }

    findings: list[
        QualityFinding
    ] = []

    for result in results:
        rule = rule_lookup.get(
            result.rule_id
        )

        if rule is None:
            continue

        column_name = (
            ", ".join(
                rule.columns
            )
            if rule.columns
            else ""
        )

        if result.status == "failed":
            findings.append(
                QualityFinding(
                    title=(
                        f"Data quality rule failed: "
                        f"{rule.rule_type}"
                    ),
                    description=(
                        f"{result.failed_rows} of "
                        f"{result.rows_checked} checked row(s) "
                        "failed this validated data-quality rule."
                    ),
                    category="data_quality",
                    severity=rule.severity,
                    table_name=rule.table_name,
                    column_name=column_name,
                    confidence=rule.confidence,
                    evidence=[
                        f"Rule ID: {rule.rule_id}",
                        (
                            f"Pass percentage: "
                            f"{result.pass_percentage}%"
                        ),
                        result.message,
                    ],
                )
            )

        elif result.status == "error":
            findings.append(
                QualityFinding(
                    title=(
                        f"Rule execution error: "
                        f"{rule.rule_type}"
                    ),
                    description=(
                        "A validated data-quality rule could not "
                        "be executed successfully."
                    ),
                    category="execution",
                    severity="high",
                    table_name=rule.table_name,
                    column_name=column_name,
                    confidence=1.0,
                    evidence=[
                        f"Rule ID: {rule.rule_id}",
                        result.message,
                    ],
                )
            )

    return findings


def _mapping_findings(
    mappings: list[
        MappingSuggestion
    ],
) -> list[
    QualityFinding
]:
    findings: list[
        QualityFinding
    ] = []

    for mapping in mappings:
        if not mapping.requires_review:
            continue

        findings.append(
            QualityFinding(
                title="Mapping requires review",
                description=(
                    f"Source '{mapping.source_table}."
                    f"{mapping.source_column}' was mapped to "
                    f"'{mapping.target_table}."
                    f"{mapping.target_column}', but the evidence "
                    "was not strong enough for automatic approval."
                ),
                category="mapping",
                severity="medium",
                table_name=(
                    mapping.source_table
                ),
                column_name=(
                    mapping.source_column
                ),
                confidence=(
                    mapping.confidence
                ),
                evidence=[
                    (
                        f"Suggested target: "
                        f"{mapping.target_table}."
                        f"{mapping.target_column}"
                    ),
                    (
                        f"Transformation: "
                        f"{mapping.transformation}"
                    ),
                    mapping.reason,
                ],
            )
        )

    return findings


def _partition_findings(
    partition: RowPartitionResult | None,
) -> list[
    QualityFinding
]:
    if partition is None:
        return []

    findings: list[
        QualityFinding
    ] = []

    for (
        table_name,
        dataframe,
    ) in partition.invalid_tables.items():
        rejected = len(
            dataframe
        )

        if rejected == 0:
            continue

        findings.append(
            QualityFinding(
                title="Rows rejected before loading",
                description=(
                    f"{rejected} row(s) from '{table_name}' "
                    "were prevented from continuing to the "
                    "database load path."
                ),
                category="data_quality",
                severity="high",
                table_name=table_name,
                column_name="",
                confidence=1.0,
                evidence=[
                    (
                        "Rejected rows retain their original "
                        "source row positions and failed rule IDs."
                    )
                ],
            )
        )

    return findings


# -------------------------------------------------------------------
# RECOMMENDATIONS
# -------------------------------------------------------------------

def _build_recommendations(
    rules: list[
        DataQualityRule
    ],
    results: list[
        RuleExecutionResult
    ],
    mappings: list[
        MappingSuggestion
    ],
    partition: RowPartitionResult | None,
) -> list[str]:
    recommendations: list[
        str
    ] = []

    failed_rules = [
        result
        for result in results
        if result.status == "failed"
    ]

    error_rules = [
        result
        for result in results
        if result.status == "error"
    ]

    review_mappings = [
        mapping
        for mapping in mappings
        if mapping.requires_review
    ]

    if failed_rules:
        recommendations.append(
            "Review rejected records and their failed "
            "data-quality rules before correcting or reprocessing "
            "the source data."
        )

    if error_rules:
        recommendations.append(
            "Resolve rule execution errors before treating the "
            "assessment as complete."
        )

    if review_mappings:
        recommendations.append(
            "Manually approve or reject source-to-target mappings "
            "marked for review before database loading."
        )

    if (
        partition is not None
        and partition.rejected_row_count > 0
    ):
        recommendations.append(
            "Keep rejected rows with their DQ metadata for audit, "
            "correction and controlled reprocessing."
        )

    if not rules:
        recommendations.append(
            "No validated data-quality rules were generated. "
            "The absence of rule failures should not be interpreted "
            "as proof that all possible business requirements are "
            "satisfied."
        )

    if not recommendations:
        recommendations.append(
            "No immediate action was identified from the validated "
            "rules and mappings used in this assessment."
        )

    return recommendations


# -------------------------------------------------------------------
# SUMMARY
# -------------------------------------------------------------------

def _build_summary(
    profile: DataProfile,
    understanding: DatasetUnderstanding,
    rules: list[
        DataQualityRule
    ],
    results: list[
        RuleExecutionResult
    ],
    mappings: list[
        MappingSuggestion
    ],
    partition: RowPartitionResult | None,
    agent_understanding: str | None,
) -> str:
    failed = sum(
        1
        for result in results
        if result.status == "failed"
    )

    review_mappings = sum(
        1
        for mapping in mappings
        if mapping.requires_review
    )

    rejected = (
        partition.rejected_row_count
        if partition is not None
        else 0
    )

    base = (
        agent_understanding
        or understanding.dataset_description
        or (
            f"Assessment completed for "
            f"'{profile.source_name}'."
        )
    ).strip()

    details = (
        f" Profiled {profile.table_count} table(s) and "
        f"{profile.document_count} document(s). "
        f"Executed {len(results)} validated data-quality "
        f"rule(s), with {failed} failing rule(s). "
        f"Rejected {rejected} row(s) from the guarded load path. "
        f"{review_mappings} mapping(s) require manual review."
    )

    return (
        base
        + details
    )


# -------------------------------------------------------------------
# ASSESSMENT
# -------------------------------------------------------------------

def build_assessment(
    profile: DataProfile,
    understanding: DatasetUnderstanding,
    rules: Iterable[
        DataQualityRule
    ],
    rule_results: Iterable[
        RuleExecutionResult
    ],
    *,
    mappings: Iterable[
        MappingSuggestion
    ] | None = None,
    partition: RowPartitionResult | None = None,
    agent_understanding: str | None = None,
) -> AssessmentResult:
    """
    Build the final source assessment from already validated pipeline
    evidence.

    This function performs no LLM calls and invents no new analytical
    facts.
    """

    rule_list = list(
        rules
    )

    result_list = list(
        rule_results
    )

    mapping_list = list(
        mappings
        or []
    )

    findings = (
        _rule_findings(
            rule_list,
            result_list,
        )
        + _mapping_findings(
            mapping_list
        )
        + _partition_findings(
            partition
        )
    )

    score = (
        _quality_score(
            rule_list,
            result_list,
        )
    )

    recommendations = (
        _build_recommendations(
            rule_list,
            result_list,
            mapping_list,
            partition,
        )
    )

    summary = (
        _build_summary(
            profile,
            understanding,
            rule_list,
            result_list,
            mapping_list,
            partition,
            agent_understanding,
        )
    )

    return AssessmentResult(
        source_name=(
            profile.source_name
        ),
        source_type=(
            profile.source_type
        ),
        summary=summary,
        detected_domain=(
            understanding.detected_domain
        ),
        profile=profile,
        semantic_insights=(
            understanding.column_insights
        ),
        findings=findings,
        generated_rules=rule_list,
        rule_results=result_list,
        mappings=mapping_list,
        recommendations=(
            recommendations
        ),
        quality_score=score,
    )


# -------------------------------------------------------------------
# REPORT FILE NAMES
# -------------------------------------------------------------------

def _safe_report_name(
    source_name: str,
) -> str:
    stem = Path(
        source_name
    ).stem

    cleaned = (
        re.sub(
            r"[^A-Za-z0-9_-]+",
            "_",
            stem,
        )
        .strip("_")
        .lower()
    )

    return (
        cleaned
        or "assessment"
    )


def _safe_table_name(
    table_name: str,
) -> str:
    """
    Create a safe file-name component from a logical table name.

    Unlike _safe_report_name(), this does not treat dots in table
    names as filename extensions.
    """

    cleaned = (
        re.sub(
            r"[^A-Za-z0-9_-]+",
            "_",
            table_name,
        )
        .strip("_")
        .lower()
    )

    return (
        cleaned
        or "table"
    )


# -------------------------------------------------------------------
# JSON REPORT
# -------------------------------------------------------------------

def _save_json_report(
    assessment: AssessmentResult,
    output_path: Path,
) -> None:
    payload = (
        assessment.model_dump(
            mode="json"
        )
    )

    output_path.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )


# -------------------------------------------------------------------
# HTML REPORT
# -------------------------------------------------------------------

def _severity_badge(
    severity: str,
) -> str:
    return (
        f'<span class="severity {html.escape(severity)}">'
        f'{html.escape(severity.upper())}'
        "</span>"
    )


def _render_findings(
    findings: list[
        QualityFinding
    ],
) -> str:
    if not findings:
        return (
            '<div class="empty">'
            "No validated findings were generated."
            "</div>"
        )

    sections: list[
        str
    ] = []

    for finding in findings:
        evidence = "".join(
            (
                "<li>"
                + html.escape(
                    str(item)
                )
                + "</li>"
            )
            for item in finding.evidence
        )

        location = (
            finding.table_name
            or ""
        )

        if finding.column_name:
            location += (
                "."
                + finding.column_name
            )

        sections.append(
            f"""
            <article class="finding">
                <div class="finding-head">
                    <h3>{html.escape(finding.title)}</h3>
                    {_severity_badge(finding.severity)}
                </div>

                <p>
                    {html.escape(finding.description)}
                </p>

                <p class="meta">
                    Category: {html.escape(finding.category)}
                    · Location: {html.escape(location or "N/A")}
                    · Confidence: {finding.confidence:.2f}
                </p>

                <ul>
                    {evidence}
                </ul>
            </article>
            """
        )

    return "\n".join(
        sections
    )


def _render_rules(
    rules: list[
        DataQualityRule
    ],
    results: list[
        RuleExecutionResult
    ],
) -> str:
    if not rules:
        return (
            '<div class="empty">'
            "No validated rules were generated."
            "</div>"
        )

    result_lookup = {
        result.rule_id: result
        for result in results
    }

    rows: list[
        str
    ] = []

    for rule in rules:
        result = result_lookup.get(
            rule.rule_id
        )

        status = (
            result.status
            if result is not None
            else "not executed"
        )

        pass_percentage = (
            f"{result.pass_percentage:.2f}%"
            if result is not None
            else "N/A"
        )

        rows.append(
            "<tr>"
            f"<td>{html.escape(rule.rule_id)}</td>"
            f"<td>{html.escape(rule.table_name)}</td>"
            f"<td>{html.escape(', '.join(rule.columns))}</td>"
            f"<td>{html.escape(rule.rule_type)}</td>"
            f"<td>{html.escape(rule.severity)}</td>"
            f"<td>{html.escape(status)}</td>"
            f"<td>{html.escape(pass_percentage)}</td>"
            "</tr>"
        )

    return (
        """
        <div class="table-wrap">
        <table>
            <thead>
                <tr>
                    <th>Rule ID</th>
                    <th>Table</th>
                    <th>Column(s)</th>
                    <th>Rule</th>
                    <th>Severity</th>
                    <th>Status</th>
                    <th>Pass %</th>
                </tr>
            </thead>

            <tbody>
        """
        + "".join(
            rows
        )
        + """
            </tbody>
        </table>
        </div>
        """
    )


def _render_mappings(
    mappings: list[
        MappingSuggestion
    ],
) -> str:
    if not mappings:
        return (
            '<div class="empty">'
            "No source-to-target mappings were supplied."
            "</div>"
        )

    rows: list[
        str
    ] = []

    for mapping in mappings:
        rows.append(
            "<tr>"
            f"<td>{html.escape(mapping.source_table)}</td>"
            f"<td>{html.escape(mapping.source_column)}</td>"
            f"<td>{html.escape(mapping.target_table)}</td>"
            f"<td>{html.escape(mapping.target_column)}</td>"
            f"<td>{html.escape(mapping.transformation)}</td>"
            f"<td>{mapping.confidence:.4f}</td>"
            f"<td>{'Yes' if mapping.requires_review else 'No'}</td>"
            "</tr>"
        )

    return (
        """
        <div class="table-wrap">
        <table>
            <thead>
                <tr>
                    <th>Source table</th>
                    <th>Source column</th>
                    <th>Target table</th>
                    <th>Target column</th>
                    <th>Transformation</th>
                    <th>Confidence</th>
                    <th>Review</th>
                </tr>
            </thead>

            <tbody>
        """
        + "".join(
            rows
        )
        + """
            </tbody>
        </table>
        </div>
        """
    )


def _render_recommendations(
    recommendations: list[str],
) -> str:
    return "".join(
        (
            "<li>"
            + html.escape(
                recommendation
            )
            + "</li>"
        )
        for recommendation
        in recommendations
    )


def _save_html_report(
    assessment: AssessmentResult,
    output_path: Path,
) -> None:
    findings_html = (
        _render_findings(
            assessment.findings
        )
    )

    rules_html = (
        _render_rules(
            assessment.generated_rules,
            assessment.rule_results,
        )
    )

    mappings_html = (
        _render_mappings(
            assessment.mappings
        )
    )

    recommendations_html = (
        _render_recommendations(
            assessment.recommendations
        )
    )

    content = f"""<!DOCTYPE html>
<html lang="en">

<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">

<title>Data Intelligence Assessment</title>

<style>
    * {{
        box-sizing: border-box;
    }}

    body {{
        margin: 0;
        font-family:
            Inter,
            Segoe UI,
            Arial,
            sans-serif;
        background: #f5f5f2;
        color: #20211f;
        line-height: 1.55;
    }}

    main {{
        max-width: 1180px;
        margin: 0 auto;
        padding: 44px 24px 80px;
    }}

    header {{
        margin-bottom: 30px;
    }}

    h1 {{
        margin: 0 0 6px;
        font-size: 34px;
    }}

    h2 {{
        margin-top: 0;
        font-size: 21px;
    }}

    h3 {{
        margin: 0;
        font-size: 16px;
    }}

    .subtitle {{
        color: #666861;
    }}

    .grid {{
        display: grid;
        grid-template-columns:
            repeat(auto-fit, minmax(180px, 1fr));
        gap: 14px;
        margin: 24px 0;
    }}

    .metric,
    section {{
        background: #ffffff;
        border: 1px solid #dedfd9;
        border-radius: 12px;
    }}

    .metric {{
        padding: 18px;
    }}

    .metric-label {{
        color: #70726c;
        font-size: 13px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }}

    .metric-value {{
        margin-top: 4px;
        font-size: 24px;
        font-weight: 700;
    }}

    section {{
        margin-top: 18px;
        padding: 22px;
    }}

    .finding {{
        border-top: 1px solid #e8e8e3;
        padding: 16px 0;
    }}

    .finding:first-child {{
        border-top: 0;
        padding-top: 0;
    }}

    .finding-head {{
        display: flex;
        justify-content: space-between;
        gap: 12px;
        align-items: center;
    }}

    .severity {{
        border: 1px solid #cfd0ca;
        border-radius: 999px;
        padding: 3px 8px;
        font-size: 11px;
        font-weight: 700;
    }}

    .meta {{
        color: #70726c;
        font-size: 13px;
    }}

    .table-wrap {{
        overflow-x: auto;
    }}

    table {{
        width: 100%;
        border-collapse: collapse;
        font-size: 13px;
    }}

    th,
    td {{
        text-align: left;
        padding: 10px;
        border-bottom: 1px solid #e7e7e2;
        vertical-align: top;
    }}

    th {{
        background: #f2f2ee;
        font-weight: 650;
    }}

    .empty {{
        color: #777970;
        font-style: italic;
    }}

    footer {{
        margin-top: 30px;
        color: #777970;
        font-size: 12px;
    }}

    @media (max-width: 640px) {{
        main {{
            padding: 24px 14px 50px;
        }}

        h1 {{
            font-size: 27px;
        }}

        section {{
            padding: 16px;
        }}
    }}
</style>
</head>

<body>

<main>

<header>
    <h1>Agentic Data Intelligence Assessment</h1>

    <div class="subtitle">
        {html.escape(assessment.source_name)}
        · {html.escape(assessment.source_type)}
    </div>
</header>

<div class="grid">

    <div class="metric">
        <div class="metric-label">
            Quality score
        </div>

        <div class="metric-value">
            {assessment.quality_score:.2f}
        </div>
    </div>

    <div class="metric">
        <div class="metric-label">
            Tables
        </div>

        <div class="metric-value">
            {assessment.profile.table_count}
        </div>
    </div>

    <div class="metric">
        <div class="metric-label">
            DQ rules
        </div>

        <div class="metric-value">
            {len(assessment.generated_rules)}
        </div>
    </div>

    <div class="metric">
        <div class="metric-label">
            Findings
        </div>

        <div class="metric-value">
            {len(assessment.findings)}
        </div>
    </div>

</div>

<section>
    <h2>Executive summary</h2>

    <p>
        {html.escape(assessment.summary)}
    </p>

    <p>
        <strong>Detected domain:</strong>
        {html.escape(assessment.detected_domain)}
    </p>

    <p class="meta">
        Quality score is calculated only from validated rules that
        were actually executed. It does not represent unknown or
        unstated business requirements.
    </p>
</section>

<section>
    <h2>Findings</h2>

    {findings_html}
</section>

<section>
    <h2>Data-quality rules</h2>

    {rules_html}
</section>

<section>
    <h2>Source-to-target mappings</h2>

    {mappings_html}
</section>

<section>
    <h2>Recommendations</h2>

    <ul>
        {recommendations_html}
    </ul>
</section>

<footer>
    Generated by the Agentic Data Intelligence pipeline using
    deterministic validation and guarded local-model reasoning.
</footer>

</main>

</body>
</html>
"""

    output_path.write_text(
        content,
        encoding="utf-8",
    )


# -------------------------------------------------------------------
# PUBLIC REPORT API
# -------------------------------------------------------------------

def save_assessment_report(
    assessment: AssessmentResult,
) -> dict[str, Path]:
    """
    Save the final assessment as JSON and optionally HTML.
    """

    ensure_runtime_directories()

    report_directory = Path(
        settings.report_directory
    )

    report_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    name = (
        _safe_report_name(
            assessment.source_name
        )
    )

    outputs: dict[
        str,
        Path,
    ] = {}

    json_path = (
        report_directory
        / f"{name}_assessment.json"
    )

    _save_json_report(
        assessment,
        json_path,
    )

    outputs[
        "json"
    ] = json_path

    if settings.generate_html_report:
        html_path = (
            report_directory
            / f"{name}_assessment.html"
        )

        _save_html_report(
            assessment,
            html_path,
        )

        outputs[
            "html"
        ] = html_path

    return outputs


# -------------------------------------------------------------------
# CLEAN DATASET
# -------------------------------------------------------------------

def save_clean_datasets(
    source_name: str,
    partition: RowPartitionResult,
) -> dict[str, Path]:
    """
    Save DQ-valid rows as clean CSV files for downstream
    data-engineering work.

    One CSV file is generated for each source table.

    Clean CSV files contain only the valid source rows. Rejection
    metadata is kept out of these datasets.
    """

    ensure_runtime_directories()

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
        _safe_report_name(
            source_name
        )
    )

    outputs: dict[
        str,
        Path,
    ] = {}

    valid_tables = (
        partition.valid_tables
    )

    multiple_tables = (
        len(
            valid_tables
        )
        > 1
    )

    for (
        table_name,
        dataframe,
    ) in valid_tables.items():

        if multiple_tables:
            safe_table = (
                _safe_table_name(
                    table_name
                )
            )

            file_name = (
                f"{source_stem}_"
                f"{safe_table}_clean.csv"
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

        outputs[
            table_name
        ] = output_path

    return outputs