from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import pandas as pd
import pymupdf
from docx import Document

from src.models import DataAsset


SUPPORTED_FILE_TYPES = {
    ".csv",
    ".tsv",
    ".xlsx",
    ".xlsm",
    ".xls",
    ".json",
    ".jsonl",
    ".ndjson",
    ".parquet",
    ".xml",
    ".txt",
    ".log",
    ".md",
    ".pdf",
    ".docx",
}


# -------------------------------------------------------------------
# NAME NORMALIZATION
# -------------------------------------------------------------------

def _clean_table_name(
    name: str,
) -> str:
    """
    Convert a file, sheet or table name into a clean internal name.
    """

    name = str(
        name
    ).strip().lower()

    name = re.sub(
        r"[^a-z0-9_]+",
        "_",
        name,
    )

    name = re.sub(
        r"_+",
        "_",
        name,
    )

    name = name.strip(
        "_"
    )

    return (
        name
        or "data"
    )


def _make_columns_unique(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Make duplicate or empty column names safe without changing
    the actual dataset values.
    """

    dataframe = (
        dataframe.copy()
    )

    used_names: dict[
        str,
        int,
    ] = {}

    final_columns: list[
        str
    ] = []

    for raw_column in dataframe.columns:
        column = str(
            raw_column
        ).strip()

        if (
            not column
            or column.lower()
            == "nan"
        ):
            column = "unnamed"

        if column not in used_names:
            used_names[
                column
            ] = 1

            final_columns.append(
                column
            )

            continue

        used_names[
            column
        ] += 1

        final_columns.append(
            f"{column}_"
            f"{used_names[column]}"
        )

    dataframe.columns = (
        final_columns
    )

    return dataframe


def _add_table(
    asset: DataAsset,
    table_name: str,
    dataframe: pd.DataFrame,
) -> None:
    """
    Add a dataframe to the asset while keeping table names unique.
    """

    if dataframe is None:
        return

    dataframe = (
        _make_columns_unique(
            dataframe
        )
    )

    clean_name = (
        _clean_table_name(
            table_name
        )
    )

    final_name = clean_name
    counter = 2

    while (
        final_name
        in asset.tables
    ):
        final_name = (
            f"{clean_name}_"
            f"{counter}"
        )

        counter += 1

    asset.tables[
        final_name
    ] = (
        dataframe.reset_index(
            drop=True
        )
    )


# -------------------------------------------------------------------
# LOSSLESS DELIMITED-TEXT HELPERS
# -------------------------------------------------------------------

PLUS_PREFIXED_NUMBER_PATTERN = re.compile(
    r"^\+\d+(?:\.\d+)?$"
)

LEADING_ZERO_INTEGER_PATTERN = re.compile(
    r"^[+-]?0\d+$"
)

LONG_INTEGER_PATTERN = re.compile(
    r"^[+-]?\d{16,}$"
)


def _column_requires_text_preservation(
    raw_series: pd.Series,
) -> bool:
    """
    Detect numeric-looking source values whose textual representation
    carries information that normal pandas type inference can remove.

    Examples:

        +919118410699  -> leading plus is meaningful
        001245         -> leading zeros are meaningful
        12345678901234567890 -> may exceed safe numeric precision

    This is based on the values themselves rather than a hard-coded
    column name such as "phone" or "customer_id".
    """

    values = (
        raw_series
        .dropna()
        .astype(str)
    )

    if values.empty:
        return False

    for value in values:
        candidate = (
            value.strip()
        )

        if not candidate:
            continue

        if (
            PLUS_PREFIXED_NUMBER_PATTERN
            .fullmatch(candidate)
        ):
            return True

        if (
            LEADING_ZERO_INTEGER_PATTERN
            .fullmatch(candidate)
        ):
            return True

        if (
            LONG_INTEGER_PATTERN
            .fullmatch(candidate)
        ):
            return True

    return False


def _restore_lexically_significant_columns(
    dataframe: pd.DataFrame,
    raw_dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Restore source text only for columns where automatic numeric
    inference would destroy meaningful lexical information.

    Most normal numeric columns stay numeric. Identifier-like values
    containing a leading plus, leading zero or very long integer
    representation remain exactly as supplied in the source file.
    """

    if (
        dataframe.shape
        != raw_dataframe.shape
    ):
        return dataframe

    if (
        list(
            dataframe.columns
        )
        != list(
            raw_dataframe.columns
        )
    ):
        return dataframe

    dataframe = (
        dataframe.copy()
    )

    for column in dataframe.columns:
        raw_series = (
            raw_dataframe[
                column
            ]
        )

        if not (
            _column_requires_text_preservation(
                raw_series
            )
        ):
            continue

        preserved = (
            raw_series.astype(
                object
            )
        )

        preserved = (
            preserved.where(
                raw_series.notna(),
                None,
            )
        )

        dataframe[
            column
        ] = preserved

    return dataframe


def _read_delimited_candidate(
    path: Path,
    **read_kwargs: Any,
) -> pd.DataFrame:
    """
    Read a delimited file twice:

    1. normally, so ordinary numeric/datetime inference is retained;
    2. as strings, so original lexical values are available.

    Only columns that would lose meaningful textual information are
    restored from the string-preserving read.
    """

    dataframe = (
        pd.read_csv(
            path,
            **read_kwargs,
        )
    )

    raw_dataframe = (
        pd.read_csv(
            path,
            dtype="string",
            **read_kwargs,
        )
    )

    return (
        _restore_lexically_significant_columns(
            dataframe,
            raw_dataframe,
        )
    )


# -------------------------------------------------------------------
# CSV / TSV
# -------------------------------------------------------------------

def _read_csv(
    path: Path,
    asset: DataAsset,
) -> None:
    """
    Read CSV or TSV files.

    CSV delimiter detection supports common non-comma delimiters.

    Numeric inference is retained for ordinary measurements while
    source representations such as +91 phone numbers, leading-zero
    identifiers and very long numeric identifiers are preserved.
    """

    dataframe: (
        pd.DataFrame
        | None
    ) = None

    last_error: (
        Exception
        | None
    ) = None

    if (
        path.suffix.lower()
        == ".tsv"
    ):
        candidates = [
            {
                "sep": "\t",
            },
            {
                "sep": "\t",
                "encoding": "latin-1",
            },
        ]

    else:
        candidates = [
            {
                "sep": None,
                "engine": "python",
            },
            {
                "sep": None,
                "engine": "python",
                "encoding": "latin-1",
            },
            {},
            {
                "encoding": "latin-1",
            },
        ]

    for read_kwargs in candidates:
        try:
            dataframe = (
                _read_delimited_candidate(
                    path,
                    **read_kwargs,
                )
            )

            break

        except Exception as exc:
            last_error = exc

    if dataframe is None:
        if last_error is not None:
            raise last_error

        raise ValueError(
            f"Unable to read delimited file: {path}"
        )

    _add_table(
        asset,
        path.stem,
        dataframe,
    )


# -------------------------------------------------------------------
# EXCEL
# -------------------------------------------------------------------

def _read_excel(
    path: Path,
    asset: DataAsset,
) -> None:
    """
    Read every sheet from an Excel workbook.
    """

    sheets = (
        pd.read_excel(
            path,
            sheet_name=None,
        )
    )

    for (
        sheet_name,
        dataframe,
    ) in sheets.items():
        _add_table(
            asset,
            sheet_name,
            dataframe,
        )


# -------------------------------------------------------------------
# JSON
# -------------------------------------------------------------------

def _find_json_tables(
    value: Any,
    current_name: str,
    asset: DataAsset,
) -> int:
    """
    Recursively search JSON for lists of records.

    This lets the loader work with JSON where useful tables are nested
    instead of existing only at the root level.
    """

    tables_found = 0

    if isinstance(
        value,
        list,
    ):
        if (
            value
            and all(
                isinstance(
                    item,
                    dict,
                )
                for item in value
            )
        ):
            try:
                dataframe = (
                    pd.json_normalize(
                        value
                    )
                )

                _add_table(
                    asset,
                    current_name,
                    dataframe,
                )

                tables_found += 1

            except Exception:
                pass

        for (
            index,
            item,
        ) in enumerate(
            value
        ):
            if isinstance(
                item,
                (
                    dict,
                    list,
                ),
            ):
                child_name = (
                    f"{current_name}_"
                    f"{index + 1}"
                )

                tables_found += (
                    _find_json_tables(
                        item,
                        child_name,
                        asset,
                    )
                )

    elif isinstance(
        value,
        dict,
    ):
        for (
            key,
            item,
        ) in value.items():
            if isinstance(
                item,
                (
                    dict,
                    list,
                ),
            ):
                child_name = (
                    f"{current_name}_"
                    f"{key}"
                )

                tables_found += (
                    _find_json_tables(
                        item,
                        child_name,
                        asset,
                    )
                )

    return tables_found


def _read_json(
    path: Path,
    asset: DataAsset,
) -> None:
    """
    Read ordinary and nested JSON files.
    """

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = (
            json.load(
                file
            )
        )

    root_name = (
        path.stem
    )

    if isinstance(
        data,
        list,
    ):
        if not data:
            _add_table(
                asset,
                root_name,
                pd.DataFrame(),
            )

            return

        if all(
            isinstance(
                item,
                dict,
            )
            for item in data
        ):
            dataframe = (
                pd.json_normalize(
                    data
                )
            )

            _add_table(
                asset,
                root_name,
                dataframe,
            )

            return

    if isinstance(
        data,
        dict,
    ):
        nested_tables = (
            _find_json_tables(
                data,
                root_name,
                asset,
            )
        )

        scalar_part: dict[
            str,
            Any,
        ] = {}

        for (
            key,
            value,
        ) in data.items():
            if isinstance(
                value,
                list,
            ):
                continue

            scalar_part[
                key
            ] = value

        if scalar_part:
            try:
                dataframe = (
                    pd.json_normalize(
                        scalar_part
                    )
                )

                _add_table(
                    asset,
                    (
                        f"{root_name}_"
                        "metadata"
                    ),
                    dataframe,
                )

            except Exception:
                pass

        if nested_tables:
            return

        dataframe = (
            pd.json_normalize(
                data
            )
        )

        _add_table(
            asset,
            root_name,
            dataframe,
        )

        return

    dataframe = pd.DataFrame(
        [
            {
                "value": data,
            }
        ]
    )

    _add_table(
        asset,
        root_name,
        dataframe,
    )


def _read_json_lines(
    path: Path,
    asset: DataAsset,
) -> None:
    """
    Read JSONL / NDJSON where every line is an independent JSON record.
    """

    dataframe = (
        pd.read_json(
            path,
            lines=True,
        )
    )

    _add_table(
        asset,
        path.stem,
        dataframe,
    )


# -------------------------------------------------------------------
# PARQUET
# -------------------------------------------------------------------

def _read_parquet(
    path: Path,
    asset: DataAsset,
) -> None:
    dataframe = (
        pd.read_parquet(
            path
        )
    )

    _add_table(
        asset,
        path.stem,
        dataframe,
    )


# -------------------------------------------------------------------
# XML
# -------------------------------------------------------------------

def _xml_node_to_dict(
    element: ET.Element,
) -> dict[str, Any]:
    """
    Convert a nested XML element into a dictionary.

    This acts as the fallback when pandas.read_xml cannot flatten
    a more complicated XML structure directly.
    """

    result: dict[
        str,
        Any,
    ] = {}

    for (
        key,
        value,
    ) in element.attrib.items():
        result[
            f"@{key}"
        ] = value

    children = list(
        element
    )

    if not children:
        text = (
            element.text.strip()
            if element.text
            else None
        )

        if result:
            result[
                "value"
            ] = text

            return result

        return {
            "value": text,
        }

    grouped_children: dict[
        str,
        list[ET.Element],
    ] = {}

    for child in children:
        grouped_children.setdefault(
            child.tag,
            [],
        ).append(
            child
        )

    for (
        tag,
        nodes,
    ) in grouped_children.items():
        converted = [
            _xml_node_to_dict(
                node
            )
            for node in nodes
        ]

        if (
            len(
                converted
            )
            == 1
        ):
            result[
                tag
            ] = (
                converted[
                    0
                ]
            )

        else:
            result[
                tag
            ] = converted

    return result


def _read_xml(
    path: Path,
    asset: DataAsset,
) -> None:
    """
    Try the standard pandas XML reader first and then fall back
    to a recursive XML conversion when needed.
    """

    try:
        dataframe = (
            pd.read_xml(
                path
            )
        )

        if (
            dataframe
            is not None
            and not dataframe.empty
        ):
            _add_table(
                asset,
                path.stem,
                dataframe,
            )

            return

    except Exception:
        pass

    tree = (
        ET.parse(
            path
        )
    )

    root = (
        tree.getroot()
    )

    children = list(
        root
    )

    if children:
        records = []

        for child in children:
            record = (
                _xml_node_to_dict(
                    child
                )
            )

            records.append(
                record
            )

        dataframe = (
            pd.json_normalize(
                records
            )
        )

        _add_table(
            asset,
            path.stem,
            dataframe,
        )

        return

    dataframe = pd.DataFrame(
        [
            {
                "value": (
                    root.text.strip()
                    if root.text
                    else None
                )
            }
        ]
    )

    _add_table(
        asset,
        path.stem,
        dataframe,
    )


# -------------------------------------------------------------------
# PLAIN TEXT
# -------------------------------------------------------------------

def _read_plain_text(
    path: Path,
    asset: DataAsset,
) -> None:
    """
    Read text-like files without trying to guess their business meaning.

    Semantic interpretation happens later in the Agentic AI layer.
    """

    try:
        text = (
            path.read_text(
                encoding="utf-8",
            )
        )

    except UnicodeDecodeError:
        text = (
            path.read_text(
                encoding="latin-1",
            )
        )

    asset.documents[
        _clean_table_name(
            path.stem
        )
    ] = text


# -------------------------------------------------------------------
# PDF
# -------------------------------------------------------------------

def _safe_pdf_headers(
    first_row: list[Any],
) -> list[str]:
    """
    Build safe headers for an extracted PDF table.
    """

    headers: list[
        str
    ] = []

    for (
        index,
        value,
    ) in enumerate(
        first_row
    ):
        if value is None:
            headers.append(
                f"column_"
                f"{index + 1}"
            )

            continue

        header = str(
            value
        ).strip()

        if not header:
            header = (
                f"column_"
                f"{index + 1}"
            )

        headers.append(
            header
        )

    return headers


def _read_pdf(
    path: Path,
    asset: DataAsset,
) -> None:
    """
    Extract both normal text and detectable tables from a PDF.
    """

    document = (
        pymupdf.open(
            path
        )
    )

    extracted_text: list[
        str
    ] = []

    try:
        for (
            page_number,
            page,
        ) in enumerate(
            document,
            start=1,
        ):
            page_text = (
                page.get_text(
                    "text"
                )
            )

            if page_text.strip():
                extracted_text.append(
                    page_text
                )

            try:
                table_finder = (
                    page.find_tables()
                )

            except Exception:
                continue

            for (
                table_number,
                table,
            ) in enumerate(
                table_finder.tables,
                start=1,
            ):
                rows = (
                    table.extract()
                )

                if not rows:
                    continue

                usable_rows = [
                    row
                    for row in rows
                    if row
                ]

                if not usable_rows:
                    continue

                max_columns = max(
                    len(
                        row
                    )
                    for row in usable_rows
                )

                if (
                    max_columns
                    == 0
                ):
                    continue

                normalized_rows = []

                for row in rows:
                    current_row = list(
                        row
                        or []
                    )

                    if (
                        len(
                            current_row
                        )
                        < max_columns
                    ):
                        current_row.extend(
                            [None]
                            * (
                                max_columns
                                - len(
                                    current_row
                                )
                            )
                        )

                    normalized_rows.append(
                        current_row[
                            :max_columns
                        ]
                    )

                first_row = (
                    normalized_rows[
                        0
                    ]
                )

                headers = (
                    _safe_pdf_headers(
                        first_row
                    )
                )

                data_rows = (
                    normalized_rows[
                        1:
                    ]
                )

                dataframe = (
                    pd.DataFrame(
                        data_rows,
                        columns=headers,
                    )
                )

                _add_table(
                    asset,
                    (
                        f"{path.stem}_"
                        f"page_{page_number}_"
                        f"table_{table_number}"
                    ),
                    dataframe,
                )

    finally:
        document.close()

    asset.documents[
        _clean_table_name(
            path.stem
        )
    ] = "\n".join(
        extracted_text
    )


# -------------------------------------------------------------------
# DOCX
# -------------------------------------------------------------------

def _read_docx(
    path: Path,
    asset: DataAsset,
) -> None:
    """
    Extract paragraphs and tables from a Word document.
    """

    document = (
        Document(
            path
        )
    )

    paragraphs = []

    for paragraph in document.paragraphs:
        text = (
            paragraph.text.strip()
        )

        if text:
            paragraphs.append(
                text
            )

    asset.documents[
        _clean_table_name(
            path.stem
        )
    ] = "\n".join(
        paragraphs
    )

    for (
        table_number,
        table,
    ) in enumerate(
        document.tables,
        start=1,
    ):
        rows: list[
            list[str]
        ] = []

        for row in table.rows:
            row_values = [
                cell.text.strip()
                for cell
                in row.cells
            ]

            rows.append(
                row_values
            )

        if not rows:
            continue

        max_columns = max(
            len(
                row
            )
            for row in rows
        )

        if (
            max_columns
            == 0
        ):
            continue

        normalized_rows = []

        for row in rows:
            current_row = list(
                row
            )

            if (
                len(
                    current_row
                )
                < max_columns
            ):
                current_row.extend(
                    [""]
                    * (
                        max_columns
                        - len(
                            current_row
                        )
                    )
                )

            normalized_rows.append(
                current_row[
                    :max_columns
                ]
            )

        headers = []

        for (
            index,
            value,
        ) in enumerate(
            normalized_rows[
                0
            ]
        ):
            header = str(
                value
            ).strip()

            if not header:
                header = (
                    f"column_"
                    f"{index + 1}"
                )

            headers.append(
                header
            )

        dataframe = (
            pd.DataFrame(
                normalized_rows[
                    1:
                ],
                columns=headers,
            )
        )

        _add_table(
            asset,
            (
                f"{path.stem}_"
                f"table_{table_number}"
            ),
            dataframe,
        )


# -------------------------------------------------------------------
# PUBLIC INGESTION API
# -------------------------------------------------------------------

def load_asset(
    file_path: str | Path,
) -> DataAsset:
    """
    Main ingestion entry point.

    Every supported input format is converted into the same DataAsset
    structure so downstream profiling and AI logic stay format-agnostic.
    """

    path = (
        Path(
            file_path
        )
        .expanduser()
        .resolve()
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Input file does not exist: "
            f"{path}"
        )

    if not path.is_file():
        raise ValueError(
            f"Input must be a file: "
            f"{path}"
        )

    extension = (
        path.suffix.lower()
    )

    if (
        extension
        not in SUPPORTED_FILE_TYPES
    ):
        supported = ", ".join(
            sorted(
                SUPPORTED_FILE_TYPES
            )
        )

        raise ValueError(
            f"Unsupported file type "
            f"'{extension}'. "
            f"Supported types: "
            f"{supported}"
        )

    asset = DataAsset(
        source_path=path,
        source_type=(
            extension.lstrip(
                "."
            )
        ),
    )

    if extension in {
        ".csv",
        ".tsv",
    }:
        _read_csv(
            path,
            asset,
        )

    elif extension in {
        ".xlsx",
        ".xlsm",
        ".xls",
    }:
        _read_excel(
            path,
            asset,
        )

    elif (
        extension
        == ".json"
    ):
        _read_json(
            path,
            asset,
        )

    elif extension in {
        ".jsonl",
        ".ndjson",
    }:
        _read_json_lines(
            path,
            asset,
        )

    elif (
        extension
        == ".parquet"
    ):
        _read_parquet(
            path,
            asset,
        )

    elif (
        extension
        == ".xml"
    ):
        _read_xml(
            path,
            asset,
        )

    elif extension in {
        ".txt",
        ".log",
        ".md",
    }:
        _read_plain_text(
            path,
            asset,
        )

    elif (
        extension
        == ".pdf"
    ):
        _read_pdf(
            path,
            asset,
        )

    elif (
        extension
        == ".docx"
    ):
        _read_docx(
            path,
            asset,
        )

    return asset