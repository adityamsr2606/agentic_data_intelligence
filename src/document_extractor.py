from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.config import settings
from src.llm_client import LLMError, llm_client
from src.models import (
    DataAsset,
    DocumentExtraction,
    ExtractedDocumentTable,
)


# -------------------------------------------------------------------
# RESULT
# -------------------------------------------------------------------

@dataclass
class UnstructuredExtractionResult:
    """
    Result of converting document text into validated structured data.
    """

    extractions: dict[str, DocumentExtraction]
    structured_asset: DataAsset
    extracted_table_count: int
    extracted_field_count: int


# -------------------------------------------------------------------
# LIMITS
# -------------------------------------------------------------------

CHUNK_SIZE = 6000
CHUNK_OVERLAP = 300

MAX_TABLES_PER_CHUNK = 8
MAX_COLUMNS_PER_TABLE = 30
MAX_ROWS_PER_TABLE = 200

MAX_FIELD_COUNT = 60


# -------------------------------------------------------------------
# EXTRACTION PROMPT
# -------------------------------------------------------------------

EXTRACTION_SYSTEM_PROMPT = """
You are the structured extraction component of an Agentic Data
Intelligence system.

Extract structured information only from the supplied document text.

This is information extraction, not creative interpretation.

STRICT RULES:

1. Extract only values explicitly present in the supplied text.

2. Never invent:
   - names
   - identifiers
   - dates
   - amounts
   - measurements
   - emails
   - phone numbers
   - addresses
   - categories
   - rows
   - missing values
   - business rules

3. Do not calculate or derive new values.

4. Do not correct spelling or normalize values unless the normalized
   value is also explicitly present in the source.

5. If the document contains clear named fields, place them in
   extracted_fields.

6. If the document contains repeated records or an explicit table,
   return them under tables.

7. Every table row must contain the same number of values as the
   table's columns.

8. Preserve source values as closely as possible.

9. A blank table cell may be represented as null.

10. Do not turn ordinary narrative sentences into artificial tables
    unless the text clearly contains repeated structured records.

11. If no reliable structured information exists, return empty fields
    and tables.

12. document_type may describe the apparent type of document, but use
    "unknown" when evidence is insufficient.

13. summary must remain short and factual.

14. Never generate Python, SQL, database commands or transformation
    expressions.
"""


# -------------------------------------------------------------------
# TEXT NORMALIZATION
# -------------------------------------------------------------------

def _normalize_text(
    value: str,
) -> str:
    """
    Normalize whitespace for conservative grounding comparisons.
    """

    return re.sub(
        r"\s+",
        " ",
        value,
    ).strip()


def _normalize_identifier(
    value: str,
    *,
    fallback: str,
) -> str:
    """
    Create a database/dataframe-safe generic identifier.
    """

    cleaned = (
        re.sub(
            r"[^A-Za-z0-9]+",
            "_",
            str(value),
        )
        .strip("_")
        .lower()
    )

    if not cleaned:
        cleaned = fallback

    if cleaned[0].isdigit():
        cleaned = (
            f"field_{cleaned}"
        )

    return cleaned[:100]


def _unique_names(
    values: list[str],
) -> list[str]:
    """
    Make normalized dataframe column names unique.
    """

    result: list[str] = []
    used: dict[str, int] = {}

    for index, value in enumerate(
        values,
        start=1,
    ):
        base = (
            _normalize_identifier(
                value,
                fallback=f"column_{index}",
            )
        )

        count = (
            used.get(
                base,
                0,
            )
        )

        if count == 0:
            name = base

        else:
            name = (
                f"{base}_{count + 1}"
            )

        used[
            base
        ] = count + 1

        while name in result:
            used[
                base
            ] += 1

            name = (
                f"{base}_{used[base]}"
            )

        result.append(
            name
        )

    return result


def _label_is_grounded(
    label: str,
    source_text: str,
) -> bool:
    """
    Return True only when a proposed schema label is explicitly
    supported by words appearing in the document.

    This prevents the LLM from turning an interpretation such as
    'measurement' or 'value' into persistent schema when the source
    never actually supplied that header.
    """

    label_words = re.findall(
        r"[A-Za-z0-9]+",
        str(label).lower(),
    )

    if not label_words:
        return False

    source_words = re.findall(
        r"[A-Za-z0-9]+",
        source_text.lower(),
    )

    if len(label_words) == 1:
        return (
            label_words[0]
            in source_words
        )

    width = len(
        label_words
    )

    for start in range(
        len(source_words)
        - width
        + 1
    ):
        if (
            source_words[
                start:
                start + width
            ]
            == label_words
        ):
            return True

    return False


# -------------------------------------------------------------------
# DOCUMENT CHUNKING
# -------------------------------------------------------------------

def _chunk_document(
    text: str,
) -> list[str]:
    """
    Split longer documents into overlapping bounded chunks.

    settings.max_document_chars remains the project-level safety cap.
    """

    text = (
        text
        or ""
    ).strip()

    if not text:
        return []

    maximum = int(
        settings.max_document_chars
    )

    text = text[
        :maximum
    ]

    if len(text) <= CHUNK_SIZE:
        return [
            text
        ]

    chunks: list[str] = []

    start = 0

    while start < len(text):
        end = min(
            start + CHUNK_SIZE,
            len(text),
        )

        chunk = (
            text[
                start:end
            ]
            .strip()
        )

        if chunk:
            chunks.append(
                chunk
            )

        if end >= len(text):
            break

        start = max(
            end - CHUNK_OVERLAP,
            start + 1,
        )

    return chunks


# -------------------------------------------------------------------
# SOURCE-GROUNDING VALIDATION
# -------------------------------------------------------------------

def _numeric_values_in_text(
    text: str,
) -> list[float]:
    """
    Collect numeric literals so numeric LLM output can be checked
    against actual source values.
    """

    tokens = re.findall(
        r"(?<![A-Za-z0-9])"
        r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)"
        r"(?![A-Za-z0-9])",
        text,
    )

    numbers: list[float] = []

    for token in tokens:
        try:
            value = float(
                token
            )

            if math.isfinite(
                value
            ):
                numbers.append(
                    value
                )

        except ValueError:
            continue

    return numbers


def _numeric_is_grounded(
    value: int | float,
    source_numbers: list[float],
) -> bool:
    try:
        numeric_value = float(
            value
        )

    except (
        TypeError,
        ValueError,
    ):
        return False

    if not math.isfinite(
        numeric_value
    ):
        return False

    return any(
        math.isclose(
            numeric_value,
            source_value,
            rel_tol=1e-9,
            abs_tol=1e-12,
        )
        for source_value
        in source_numbers
    )


def _value_is_grounded(
    value: Any,
    normalized_source: str,
    normalized_source_lower: str,
    source_numbers: list[float],
) -> bool:
    """
    Conservative hallucination guard.

    Extracted values must be traceable back to the source document.
    """

    if value is None:
        # Null is allowed for an explicitly blank table cell.
        return True

    if isinstance(
        value,
        bool,
    ):
        literal = (
            "true"
            if value
            else "false"
        )

        return (
            literal
            in normalized_source_lower
        )

    if isinstance(
        value,
        (
            int,
            float,
        ),
    ) and not isinstance(
        value,
        bool,
    ):
        return (
            _numeric_is_grounded(
                value,
                source_numbers,
            )
        )

    if isinstance(
        value,
        (
            dict,
            list,
            tuple,
            set,
        ),
    ):
        # Nested structures cannot be reliably traced as one atomic
        # extracted value.
        return False

    text = (
        _normalize_text(
            str(
                value
            )
        )
    )

    if not text:
        return False

    return (
        text.casefold()
        in normalized_source_lower
    )


# -------------------------------------------------------------------
# EXTRACTION VALIDATION
# -------------------------------------------------------------------

def _validate_fields(
    fields: dict[str, Any],
    source_text: str,
) -> dict[str, Any]:
    """
    Keep only scalar fields directly supported by the document.

    Field names are also required to be grounded in the source. If the
    model extracts a valid value but invents a label, a deterministic
    fallback field name is used.
    """

    normalized_source = (
        _normalize_text(
            source_text
        )
    )

    normalized_source_lower = (
        normalized_source.casefold()
    )

    source_numbers = (
        _numeric_values_in_text(
            source_text
        )
    )

    validated: dict[
        str,
        Any,
    ] = {}

    used_names: set[
        str
    ] = set()

    for index, (
        key,
        value,
    ) in enumerate(
        fields.items(),
        start=1,
    ):
        if (
            len(validated)
            >= MAX_FIELD_COUNT
        ):
            break

        if not (
            _value_is_grounded(
                value,
                normalized_source,
                normalized_source_lower,
                source_numbers,
            )
        ):
            continue

        if (
            _label_is_grounded(
                str(
                    key
                ),
                source_text,
            )
        ):
            proposed_name = (
                str(
                    key
                )
            )

        else:
            proposed_name = (
                f"field_{index}"
            )

        base_name = (
            _normalize_identifier(
                proposed_name,
                fallback=(
                    f"field_{index}"
                ),
            )
        )

        name = (
            base_name
        )

        suffix = 2

        while (
            name
            in used_names
        ):
            name = (
                f"{base_name}_{suffix}"
            )

            suffix += 1

        used_names.add(
            name
        )

        validated[
            name
        ] = value

    return validated


def _validate_table(
    table: ExtractedDocumentTable,
    source_text: str,
    table_number: int,
) -> ExtractedDocumentTable | None:
    """
    Validate one AI-proposed document table before it becomes a
    dataframe.

    Values must be grounded in the document.

    AI-generated table and column names are retained only when the
    labels themselves occur in the source. Otherwise deterministic
    positional names are used.
    """

    if not table.columns:
        return None

    if (
        len(
            table.columns
        )
        > MAX_COLUMNS_PER_TABLE
    ):
        return None

    grounded_columns: list[
        str
    ] = []

    for index, column in enumerate(
        table.columns,
        start=1,
    ):
        proposed = str(
            column
        )

        if (
            _label_is_grounded(
                proposed,
                source_text,
            )
        ):
            grounded_columns.append(
                proposed
            )

        else:
            grounded_columns.append(
                f"column_{index}"
            )

    columns = (
        _unique_names(
            grounded_columns
        )
    )

    normalized_source = (
        _normalize_text(
            source_text
        )
    )

    normalized_source_lower = (
        normalized_source.casefold()
    )

    source_numbers = (
        _numeric_values_in_text(
            source_text
        )
    )

    rows: list[
        list[Any]
    ] = []

    for row in table.rows[
        :MAX_ROWS_PER_TABLE
    ]:
        if (
            len(row)
            != len(columns)
        ):
            continue

        grounded = all(
            _value_is_grounded(
                value,
                normalized_source,
                normalized_source_lower,
                source_numbers,
            )
            for value
            in row
        )

        if not grounded:
            continue

        rows.append(
            list(
                row
            )
        )

    if not rows:
        return None

    proposed_table_name = str(
        table.table_name
    )

    if (
        _label_is_grounded(
            proposed_table_name,
            source_text,
        )
    ):
        table_name = (
            _normalize_identifier(
                proposed_table_name,
                fallback=(
                    f"extracted_table_{table_number}"
                ),
            )
        )

    else:
        table_name = (
            f"extracted_table_{table_number}"
        )

    return ExtractedDocumentTable(
        table_name=table_name,
        columns=columns,
        rows=rows,
    )


def _validate_extraction(
    extraction: DocumentExtraction,
    source_text: str,
) -> DocumentExtraction:
    """
    Apply deterministic grounding checks to all AI-produced structure.
    """

    fields = (
        _validate_fields(
            extraction.extracted_fields,
            source_text,
        )
    )

    tables: list[
        ExtractedDocumentTable
    ] = []

    for index, table in enumerate(
        extraction.tables[
            :MAX_TABLES_PER_CHUNK
        ],
        start=1,
    ):
        validated = (
            _validate_table(
                table,
                source_text,
                index,
            )
        )

        if validated is not None:
            tables.append(
                validated
            )

    document_type = (
        extraction.document_type
        or "unknown"
    ).strip()

    if not document_type:
        document_type = "unknown"

    summary = (
        extraction.summary
        or ""
    ).strip()

    return DocumentExtraction(
        document_type=(
            document_type[:100]
        ),
        summary=(
            summary[:500]
        ),
        extracted_fields=fields,
        tables=tables,
    )


# -------------------------------------------------------------------
# LOCAL MODEL EXTRACTION
# -------------------------------------------------------------------

def _extract_chunk(
    document_name: str,
    text: str,
    chunk_number: int,
    total_chunks: int,
) -> DocumentExtraction:
    prompt = (
        f"Document: {document_name}\n"
        f"Chunk: {chunk_number}/{total_chunks}\n\n"
        "Extract only explicitly supported structured information "
        "from the following source text.\n\n"
        "SOURCE TEXT:\n"
        f"{text}"
    )

    extraction = (
        llm_client.chat_structured(
            system_prompt=(
                EXTRACTION_SYSTEM_PROMPT
            ),
            user_prompt=prompt,
            response_model=(
                DocumentExtraction
            ),
            retries=0,
            num_predict=900,
        )
    )

    return (
        _validate_extraction(
            extraction,
            text,
        )
    )


# -------------------------------------------------------------------
# CHUNK MERGING
# -------------------------------------------------------------------

def _values_equal(
    first: Any,
    second: Any,
) -> bool:
    """
    Compare extracted field values conservatively.
    """

    try:
        return (
            json.dumps(
                first,
                sort_keys=True,
                ensure_ascii=False,
                default=str,
            )
            == json.dumps(
                second,
                sort_keys=True,
                ensure_ascii=False,
                default=str,
            )
        )

    except TypeError:
        return (
            str(
                first
            )
            == str(
                second
            )
        )


def _merge_fields(
    extractions: list[
        DocumentExtraction
    ],
) -> dict[str, Any]:
    """
    Merge fields across chunks.

    Conflicting values for the same field are removed rather than
    guessed.
    """

    values: dict[
        str,
        Any,
    ] = {}

    conflicts: set[
        str
    ] = set()

    for extraction in extractions:
        for (
            key,
            value,
        ) in (
            extraction
            .extracted_fields
            .items()
        ):
            if key in conflicts:
                continue

            if key not in values:
                values[
                    key
                ] = value

                continue

            if not (
                _values_equal(
                    values[
                        key
                    ],
                    value,
                )
            ):
                values.pop(
                    key,
                    None,
                )

                conflicts.add(
                    key
                )

    return values


def _table_signature(
    table: ExtractedDocumentTable,
) -> tuple[
    str,
    tuple[str, ...],
]:
    return (
        table.table_name,
        tuple(
            table.columns
        ),
    )


def _row_fingerprint(
    row: list[Any],
) -> str:
    return json.dumps(
        row,
        ensure_ascii=False,
        default=str,
        separators=(
            ",",
            ":",
        ),
    )


def _merge_tables(
    extractions: list[
        DocumentExtraction
    ],
) -> list[
    ExtractedDocumentTable
]:
    """
    Merge chunk tables only when both table name and columns match.
    """

    grouped: dict[
        tuple[
            str,
            tuple[str, ...],
        ],
        list[list[Any]],
    ] = {}

    seen_rows: dict[
        tuple[
            str,
            tuple[str, ...],
        ],
        set[str],
    ] = {}

    for extraction in extractions:
        for table in extraction.tables:
            signature = (
                _table_signature(
                    table
                )
            )

            grouped.setdefault(
                signature,
                [],
            )

            seen_rows.setdefault(
                signature,
                set(),
            )

            for row in table.rows:
                fingerprint = (
                    _row_fingerprint(
                        row
                    )
                )

                if (
                    fingerprint
                    in seen_rows[
                        signature
                    ]
                ):
                    continue

                seen_rows[
                    signature
                ].add(
                    fingerprint
                )

                grouped[
                    signature
                ].append(
                    row
                )

    tables: list[
        ExtractedDocumentTable
    ] = []

    for (
        table_name,
        columns,
    ), rows in grouped.items():
        if not rows:
            continue

        tables.append(
            ExtractedDocumentTable(
                table_name=(
                    table_name
                ),
                columns=list(
                    columns
                ),
                rows=rows[
                    :MAX_ROWS_PER_TABLE
                ],
            )
        )

    return tables


def _merge_chunk_extractions(
    extractions: list[
        DocumentExtraction
    ],
) -> DocumentExtraction:
    if not extractions:
        return DocumentExtraction()

    document_type = "unknown"

    for extraction in extractions:
        candidate = (
            extraction.document_type
            or ""
        ).strip()

        if (
            candidate
            and candidate.lower()
            != "unknown"
        ):
            document_type = (
                candidate
            )
            break

    summaries = []

    for extraction in extractions:
        summary = (
            extraction.summary
            or ""
        ).strip()

        if (
            summary
            and summary
            not in summaries
        ):
            summaries.append(
                summary
            )

    summary = (
        " ".join(
            summaries
        )
    )[:500]

    return DocumentExtraction(
        document_type=(
            document_type
        ),
        summary=summary,
        extracted_fields=(
            _merge_fields(
                extractions
            )
        ),
        tables=(
            _merge_tables(
                extractions
            )
        ),
    )


# -------------------------------------------------------------------
# DATAFRAME CREATION
# -------------------------------------------------------------------

def _unique_table_name(
    base_name: str,
    existing: set[str],
) -> str:
    name = (
        base_name
    )

    suffix = 2

    while name in existing:
        name = (
            f"{base_name}_{suffix}"
        )

        suffix += 1

    existing.add(
        name
    )

    return name


def _extraction_to_tables(
    document_name: str,
    extraction: DocumentExtraction,
    existing_names: set[str],
) -> dict[
    str,
    pd.DataFrame,
]:
    """
    Convert validated extraction output into canonical DataFrames.
    """

    result: dict[
        str,
        pd.DataFrame,
    ] = {}

    document_prefix = (
        _normalize_identifier(
            document_name.rsplit(
                ".",
                1,
            )[0],
            fallback="document",
        )
    )

    if extraction.extracted_fields:
        field_table_name = (
            _unique_table_name(
                f"{document_prefix}_fields",
                existing_names,
            )
        )

        result[
            field_table_name
        ] = pd.DataFrame(
            [
                extraction
                .extracted_fields
            ]
        )

    for index, table in enumerate(
        extraction.tables,
        start=1,
    ):
        base_name = (
            _normalize_identifier(
                (
                    f"{document_prefix}_"
                    f"{table.table_name}"
                ),
                fallback=(
                    f"{document_prefix}_table_{index}"
                ),
            )
        )

        table_name = (
            _unique_table_name(
                base_name,
                existing_names,
            )
        )

        result[
            table_name
        ] = pd.DataFrame(
            table.rows,
            columns=(
                table.columns
            ),
        )

    return result


# -------------------------------------------------------------------
# PUBLIC API
# -------------------------------------------------------------------

def extract_unstructured_data(
    asset: DataAsset,
) -> UnstructuredExtractionResult:
    """
    Convert unstructured document text into validated canonical tables.

    Existing tables already extracted by ingestion are preserved.

    Only values that survive deterministic source-grounding checks are
    allowed into the resulting DataFrames.
    """

    if not asset.documents:
        return UnstructuredExtractionResult(
            extractions={},
            structured_asset=asset,
            extracted_table_count=0,
            extracted_field_count=0,
        )

    extractions: dict[
        str,
        DocumentExtraction,
    ] = {}

    structured_tables = dict(
        asset.tables
    )

    existing_names = set(
        structured_tables
    )

    extracted_table_count = 0
    extracted_field_count = 0

    for (
        document_name,
        text,
    ) in asset.documents.items():
        chunks = (
            _chunk_document(
                text
            )
        )

        if not chunks:
            extractions[
                document_name
            ] = (
                DocumentExtraction()
            )

            continue

        chunk_extractions: list[
            DocumentExtraction
        ] = []

        for chunk_number, chunk in enumerate(
            chunks,
            start=1,
        ):
            try:
                extraction = (
                    _extract_chunk(
                        document_name=(
                            document_name
                        ),
                        text=chunk,
                        chunk_number=(
                            chunk_number
                        ),
                        total_chunks=(
                            len(
                                chunks
                            )
                        ),
                    )
                )

            except LLMError:
                # Do not fabricate fallback fields if the model is
                # unavailable. The raw document remains preserved.
                continue

            chunk_extractions.append(
                extraction
            )

        merged = (
            _merge_chunk_extractions(
                chunk_extractions
            )
        )

        extractions[
            document_name
        ] = merged

        extracted_field_count += len(
            merged.extracted_fields
        )

        generated_tables = (
            _extraction_to_tables(
                document_name,
                merged,
                existing_names,
            )
        )

        extracted_table_count += len(
            generated_tables
        )

        structured_tables.update(
            generated_tables
        )

    structured_asset = DataAsset(
        source_path=(
            asset.source_path
        ),
        source_type=(
            f"{asset.source_type}_structured"
        ),
        tables=structured_tables,
        documents=(
            asset.documents
        ),
    )

    return UnstructuredExtractionResult(
        extractions=extractions,
        structured_asset=(
            structured_asset
        ),
        extracted_table_count=(
            extracted_table_count
        ),
        extracted_field_count=(
            extracted_field_count
        ),
    )