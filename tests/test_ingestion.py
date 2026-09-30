from __future__ import annotations

import json

import pandas as pd
import pytest

from src.ingestion import load_asset


# -------------------------------------------------------------------
# CSV
# -------------------------------------------------------------------

def test_load_csv(tmp_path):
    source = tmp_path / "sample.csv"

    source.write_text(
        "id,name,amount\n"
        "1,Alice,10.5\n"
        "2,Bob,20.0\n",
        encoding="utf-8",
    )

    asset = load_asset(
        source
    )

    assert asset.source_type == "csv"
    assert asset.source_path == source
    assert len(asset.tables) == 1
    assert asset.documents == {}

    dataframe = next(
        iter(
            asset.tables.values()
        )
    )

    assert list(
        dataframe.columns
    ) == [
        "id",
        "name",
        "amount",
    ]

    assert len(
        dataframe
    ) == 2

    assert dataframe.iloc[
        0
    ][
        "name"
    ] == "Alice"


# -------------------------------------------------------------------
# TSV
# -------------------------------------------------------------------

def test_load_tsv(tmp_path):
    source = tmp_path / "sample.tsv"

    source.write_text(
        "id\tstatus\n"
        "1\tPASS\n"
        "2\tREVIEW\n",
        encoding="utf-8",
    )

    asset = load_asset(
        source
    )

    assert asset.source_type == "tsv"

    dataframe = next(
        iter(
            asset.tables.values()
        )
    )

    assert list(
        dataframe.columns
    ) == [
        "id",
        "status",
    ]

    assert len(
        dataframe
    ) == 2


# -------------------------------------------------------------------
# JSON
# -------------------------------------------------------------------

def test_load_json_records(tmp_path):
    source = tmp_path / "sample.json"

    payload = [
        {
            "id": 1,
            "name": "Alice",
        },
        {
            "id": 2,
            "name": "Bob",
        },
    ]

    source.write_text(
        json.dumps(
            payload
        ),
        encoding="utf-8",
    )

    asset = load_asset(
        source
    )

    assert asset.source_type == "json"

    dataframe = next(
        iter(
            asset.tables.values()
        )
    )

    assert len(
        dataframe
    ) == 2

    assert set(
        dataframe.columns
    ) == {
        "id",
        "name",
    }


# -------------------------------------------------------------------
# JSONL
# -------------------------------------------------------------------

def test_load_jsonl(tmp_path):
    source = tmp_path / "sample.jsonl"

    source.write_text(
        '{"id": 1, "value": 10}\n'
        '{"id": 2, "value": 20}\n',
        encoding="utf-8",
    )

    asset = load_asset(
        source
    )

    assert asset.source_type in {
        "jsonl",
        "ndjson",
    }

    dataframe = next(
        iter(
            asset.tables.values()
        )
    )

    assert len(
        dataframe
    ) == 2

    assert list(
        dataframe[
            "value"
        ]
    ) == [
        10,
        20,
    ]


# -------------------------------------------------------------------
# PARQUET
# -------------------------------------------------------------------

def test_load_parquet(tmp_path):
    source = tmp_path / "sample.parquet"

    expected = pd.DataFrame(
        {
            "code": [
                "A",
                "B",
            ],
            "measurement": [
                1.5,
                2.5,
            ],
        }
    )

    expected.to_parquet(
        source,
        index=False,
    )

    asset = load_asset(
        source
    )

    assert asset.source_type == "parquet"

    dataframe = next(
        iter(
            asset.tables.values()
        )
    )

    assert dataframe.shape == (
        2,
        2,
    )

    assert list(
        dataframe[
            "code"
        ]
    ) == [
        "A",
        "B",
    ]


# -------------------------------------------------------------------
# EXCEL
# -------------------------------------------------------------------

def test_load_excel_multiple_sheets(
    tmp_path,
):
    source = tmp_path / "sample.xlsx"

    first = pd.DataFrame(
        {
            "id": [
                1,
                2,
            ],
            "name": [
                "Alice",
                "Bob",
            ],
        }
    )

    second = pd.DataFrame(
        {
            "code": [
                "X",
                "Y",
            ]
        }
    )

    with pd.ExcelWriter(
        source,
        engine="openpyxl",
    ) as writer:
        first.to_excel(
            writer,
            sheet_name="People",
            index=False,
        )

        second.to_excel(
            writer,
            sheet_name="Codes",
            index=False,
        )

    asset = load_asset(
        source
    )

    assert asset.source_type in {
        "xlsx",
        "xlsm",
        "xls",
    }

    assert len(
        asset.tables
    ) == 2

    assert all(
        isinstance(
            dataframe,
            pd.DataFrame,
        )
        for dataframe
        in asset.tables.values()
    )


# -------------------------------------------------------------------
# XML
# -------------------------------------------------------------------

def test_load_xml(tmp_path):
    source = tmp_path / "sample.xml"

    source.write_text(
        """
        <records>
            <record>
                <id>1</id>
                <name>Alice</name>
            </record>
            <record>
                <id>2</id>
                <name>Bob</name>
            </record>
        </records>
        """,
        encoding="utf-8",
    )

    asset = load_asset(
        source
    )

    assert asset.source_type == "xml"

    assert len(
        asset.tables
    ) >= 1

    dataframe = next(
        iter(
            asset.tables.values()
        )
    )

    assert len(
        dataframe
    ) == 2


# -------------------------------------------------------------------
# TEXT
# -------------------------------------------------------------------

@pytest.mark.parametrize(
    (
        "extension",
        "expected_type",
    ),
    [
        (
            ".txt",
            "txt",
        ),
        (
            ".log",
            "log",
        ),
        (
            ".md",
            "md",
        ),
    ],
)
def test_load_plain_document(
    tmp_path,
    extension,
    expected_type,
):
    source = (
        tmp_path
        / f"document{extension}"
    )

    content = (
        "Report ID: RPT-100\n"
        "Status: PASS\n"
    )

    source.write_text(
        content,
        encoding="utf-8",
    )

    asset = load_asset(
        source
    )

    assert (
        asset.source_type
        == expected_type
    )

    assert len(
        asset.documents
    ) == 1

    extracted_text = next(
        iter(
            asset.documents.values()
        )
    )

    assert "RPT-100" in extracted_text
    assert "PASS" in extracted_text


# -------------------------------------------------------------------
# FAILURE CASES
# -------------------------------------------------------------------

def test_missing_file_raises_error(
    tmp_path,
):
    missing = (
        tmp_path
        / "missing.csv"
    )

    with pytest.raises(
        FileNotFoundError
    ):
        load_asset(
            missing
        )


def test_unsupported_extension_raises_error(
    tmp_path,
):
    source = (
        tmp_path
        / "sample.xyz"
    )

    source.write_text(
        "test",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError
    ):
        load_asset(
            source
        )