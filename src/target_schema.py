from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from src.models import TargetSchema


SUPPORTED_SCHEMA_FILE_TYPES = {
    ".json",
}


class TargetSchemaError(ValueError):
    """
    Raised when a target schema cannot be loaded or validated.
    """


def _read_json_file(
    file_path: Path,
) -> dict[str, Any]:
    """
    Read a target-schema JSON file.
    """

    try:
        with file_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            data = json.load(
                file
            )

    except json.JSONDecodeError as exc:
        raise TargetSchemaError(
            "Target schema contains invalid JSON: "
            f"{exc}"
        ) from exc

    except OSError as exc:
        raise TargetSchemaError(
            "Unable to read target schema file: "
            f"{exc}"
        ) from exc

    if not isinstance(
        data,
        dict,
    ):
        raise TargetSchemaError(
            "Target schema JSON must contain "
            "an object at the root level."
        )

    return data


def target_schema_from_dict(
    data: dict[str, Any],
) -> TargetSchema:
    """
    Validate a Python dictionary against the canonical TargetSchema
    model.

    Pydantic remains the single source of truth for the target-schema
    structure.
    """

    if not isinstance(
        data,
        dict,
    ):
        raise TargetSchemaError(
            "Target schema input must be a dictionary."
        )

    try:
        schema = (
            TargetSchema
            .model_validate(
                data
            )
        )

    except ValidationError as exc:
        raise TargetSchemaError(
            "Target schema failed validation:\n"
            f"{exc}"
        ) from exc

    _validate_schema_content(
        schema
    )

    return schema


def load_target_schema(
    file_path: str | Path,
) -> TargetSchema:
    """
    Load and validate a target schema from disk.

    JSON is currently used as the portable schema definition format.

    Live PostgreSQL schema discovery will be handled separately by the
    database layer so this file stays independent of database access.
    """

    path = Path(
        file_path
    ).expanduser()

    if not path.exists():
        raise TargetSchemaError(
            "Target schema file does not exist: "
            f"{path}"
        )

    if not path.is_file():
        raise TargetSchemaError(
            "Target schema path is not a file: "
            f"{path}"
        )

    suffix = (
        path.suffix.lower()
    )

    if (
        suffix
        not in SUPPORTED_SCHEMA_FILE_TYPES
    ):
        raise TargetSchemaError(
            "Unsupported target schema format: "
            f"'{suffix}'. Supported formats: "
            + ", ".join(
                sorted(
                    SUPPORTED_SCHEMA_FILE_TYPES
                )
            )
        )

    data = (
        _read_json_file(
            path
        )
    )

    return (
        target_schema_from_dict(
            data
        )
    )


def _validate_schema_content(
    schema: TargetSchema,
) -> None:
    """
    Perform structural checks that are awkward to express purely through
    individual Pydantic fields.

    No assumptions are made about a particular business domain.
    """

    schema_data = (
        schema.model_dump()
    )

    tables = (
        schema_data.get(
            "tables",
            []
        )
    )

    if not isinstance(
        tables,
        list,
    ):
        raise TargetSchemaError(
            "Target schema must contain a list of tables."
        )

    if not tables:
        raise TargetSchemaError(
            "Target schema must contain at least one target table."
        )

    seen_tables: set[str] = set()

    for table in tables:

        if not isinstance(
            table,
            dict,
        ):
            raise TargetSchemaError(
                "Every target table must be an object."
            )

        table_name = str(
            table.get(
                "table_name",
                "",
            )
        ).strip()

        if not table_name:
            raise TargetSchemaError(
                "Every target table must have a table_name."
            )

        normalized_table_name = (
            table_name.lower()
        )

        if (
            normalized_table_name
            in seen_tables
        ):
            raise TargetSchemaError(
                "Duplicate target table name: "
                f"'{table_name}'."
            )

        seen_tables.add(
            normalized_table_name
        )

        columns = (
            table.get(
                "columns",
                []
            )
        )

        if not isinstance(
            columns,
            list,
        ):
            raise TargetSchemaError(
                "Target table "
                f"'{table_name}' must contain a columns list."
            )

        if not columns:
            raise TargetSchemaError(
                "Target table "
                f"'{table_name}' must contain at least one column."
            )

        seen_columns: set[str] = set()

        for column in columns:

            if not isinstance(
                column,
                dict,
            ):
                raise TargetSchemaError(
                    "Every target column must be an object."
                )

            column_name = str(
                column.get(
                    "column_name",
                    "",
                )
            ).strip()

            if not column_name:
                raise TargetSchemaError(
                    "Every target column in "
                    f"'{table_name}' must have a column_name."
                )

            normalized_column_name = (
                column_name.lower()
            )

            if (
                normalized_column_name
                in seen_columns
            ):
                raise TargetSchemaError(
                    "Duplicate target column "
                    f"'{column_name}' in table "
                    f"'{table_name}'."
                )

            seen_columns.add(
                normalized_column_name
            )


def target_schema_to_dict(
    schema: TargetSchema,
) -> dict[str, Any]:
    """
    Return a JSON-safe dictionary representation of a validated target
    schema.
    """

    return (
        schema.model_dump(
            mode="json"
        )
    )


def describe_target_schema(
    schema: TargetSchema,
) -> dict[str, Any]:
    """
    Build a compact schema summary suitable for mapping and reporting.
    """

    schema_data = (
        target_schema_to_dict(
            schema
        )
    )

    tables = (
        schema_data.get(
            "tables",
            []
        )
    )

    return {
        "table_count": (
            len(
                tables
            )
        ),
        "tables": [
            {
                "table_name": (
                    table.get(
                        "table_name"
                    )
                ),
                "column_count": len(
                    table.get(
                        "columns",
                        []
                    )
                ),
                "columns": [
                    {
                        "column_name": (
                            column.get(
                                "column_name"
                            )
                        ),
                        "data_type": (
                            column.get(
                                "data_type"
                            )
                        ),
                        "nullable": (
                            column.get(
                                "nullable"
                            )
                        ),
                        "primary_key": (
                            column.get(
                                "primary_key"
                            )
                        ),
                        "description": (
                            column.get(
                                "description"
                            )
                        ),
                    }
                    for column in (
                        table.get(
                            "columns",
                            []
                        )
                    )
                ],
            }
            for table in tables
        ],
    }