from __future__ import annotations

import uuid
from typing import Any

import numpy as np
import pandas as pd

from src.models import (
    DataAsset,
    MappingSuggestion,
    TargetColumn,
    TargetSchema,
)


MappingKey = tuple[
    str,
    str,
    str,
    str,
]


# -------------------------------------------------------------------
# MAPPING IDENTITY
# -------------------------------------------------------------------

def mapping_key(
    mapping: MappingSuggestion,
) -> MappingKey:
    """
    Return a stable identity for manual mapping approval.
    """

    return (
        mapping.source_table,
        mapping.source_column,
        mapping.target_table,
        mapping.target_column,
    )


# -------------------------------------------------------------------
# TARGET SCHEMA HELPERS
# -------------------------------------------------------------------

def _target_lookup(
    target_schema: TargetSchema,
) -> dict[
    str,
    dict[str, TargetColumn],
]:
    """
    Build target-table/column lookup from the supplied target schema.
    """

    lookup: dict[
        str,
        dict[str, TargetColumn],
    ] = {}

    for table in target_schema.tables:

        if table.table_name in lookup:
            raise ValueError(
                "Target schema contains duplicate table "
                f"'{table.table_name}'."
            )

        columns: dict[
            str,
            TargetColumn,
        ] = {}

        for column in table.columns:

            if column.column_name in columns:
                raise ValueError(
                    "Target schema contains duplicate column "
                    f"'{table.table_name}.{column.column_name}'."
                )

            columns[
                column.column_name
            ] = column

        lookup[
            table.table_name
        ] = columns

    return lookup


def _target_column_order(
    target_schema: TargetSchema,
) -> dict[str, list[str]]:
    """
    Preserve target-schema column order in transformed DataFrames.
    """

    return {
        table.table_name: [
            column.column_name
            for column
            in table.columns
        ]
        for table
        in target_schema.tables
    }


# -------------------------------------------------------------------
# TYPE NORMALIZATION
# -------------------------------------------------------------------

def _target_type_family(
    data_type: str,
) -> str:
    """
    Normalize common SQL/database types to generic transformation
    families.
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
        value == "date"
        or value.startswith(
            "date "
        )
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
            "varchar",
            "text",
            "string",
            "clob",
        }
    ):
        return "string"

    return "unknown"


# -------------------------------------------------------------------
# CAST HELPERS
# -------------------------------------------------------------------

def _invalid_positions(
    original: pd.Series,
    converted: pd.Series,
) -> list[int]:
    """
    Return zero-based row positions where a non-null source value could
    not be converted.
    """

    invalid = (
        original.notna()
        & converted.isna()
    )

    return (
        np.flatnonzero(
            invalid.to_numpy()
        )
        .astype(int)
        .tolist()
    )


def _raise_cast_error(
    *,
    column_name: str,
    target_type: str,
    positions: list[int],
) -> None:
    """
    Raise a useful error without dumping the entire dataset.
    """

    preview = (
        positions[:10]
    )

    suffix = (
        ""
        if len(positions) <= 10
        else (
            f" and {len(positions) - 10} more"
        )
    )

    raise ValueError(
        f"Cannot safely cast column '{column_name}' "
        f"to '{target_type}'. "
        f"Invalid row position(s): {preview}{suffix}."
    )


def _cast_string(
    series: pd.Series,
) -> pd.Series:
    """
    Convert non-null values to strings while preserving nulls.
    """

    result = series.copy()

    mask = (
        result.notna()
    )

    result.loc[
        mask
    ] = (
        result.loc[
            mask
        ]
        .map(str)
    )

    return result


def _cast_number(
    series: pd.Series,
    target_type: str,
) -> pd.Series:
    converted = pd.to_numeric(
        series,
        errors="coerce",
    )

    invalid = (
        _invalid_positions(
            series,
            converted,
        )
    )

    if invalid:
        _raise_cast_error(
            column_name=(
                str(
                    series.name
                )
            ),
            target_type=target_type,
            positions=invalid,
        )

    return converted


def _cast_integer(
    series: pd.Series,
    target_type: str,
) -> pd.Series:
    converted = pd.to_numeric(
        series,
        errors="coerce",
    )

    invalid_mask = (
        series.notna()
        & (
            converted.isna()
            | (
                converted.notna()
                & (
                    converted.mod(1)
                    != 0
                )
            )
        )
    )

    invalid = (
        np.flatnonzero(
            invalid_mask.to_numpy()
        )
        .astype(int)
        .tolist()
    )

    if invalid:
        _raise_cast_error(
            column_name=(
                str(
                    series.name
                )
            ),
            target_type=target_type,
            positions=invalid,
        )

    # Nullable integer keeps missing values without forcing floats.
    return converted.astype(
        "Int64"
    )


def _parse_boolean(
    value: Any,
) -> bool | None:
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    if isinstance(
        value,
        (
            bool,
            np.bool_,
        ),
    ):
        return bool(
            value
        )

    if isinstance(
        value,
        (
            int,
            np.integer,
        ),
    ):
        if value == 1:
            return True

        if value == 0:
            return False

        raise ValueError

    if isinstance(
        value,
        str,
    ):
        normalized = (
            value
            .strip()
            .lower()
        )

        if normalized in {
            "true",
            "yes",
            "y",
            "1",
        }:
            return True

        if normalized in {
            "false",
            "no",
            "n",
            "0",
        }:
            return False

    raise ValueError


def _cast_boolean(
    series: pd.Series,
    target_type: str,
) -> pd.Series:
    values: list[
        bool | None
    ] = []

    invalid: list[
        int
    ] = []

    for position, value in enumerate(
        series.tolist()
    ):
        try:
            values.append(
                _parse_boolean(
                    value
                )
            )

        except ValueError:
            invalid.append(
                position
            )

            values.append(
                None
            )

    if invalid:
        _raise_cast_error(
            column_name=(
                str(
                    series.name
                )
            ),
            target_type=target_type,
            positions=invalid,
        )

    return pd.Series(
        values,
        index=series.index,
        name=series.name,
        dtype="boolean",
    )


def _cast_datetime(
    series: pd.Series,
    target_type: str,
) -> pd.Series:
    converted = pd.to_datetime(
        series,
        errors="coerce",
    )

    invalid = (
        _invalid_positions(
            series,
            converted,
        )
    )

    if invalid:
        _raise_cast_error(
            column_name=(
                str(
                    series.name
                )
            ),
            target_type=target_type,
            positions=invalid,
        )

    return converted


def _cast_date(
    series: pd.Series,
    target_type: str,
) -> pd.Series:
    converted = (
        _cast_datetime(
            series,
            target_type,
        )
    )

    return converted.dt.date


def _cast_uuid(
    series: pd.Series,
    target_type: str,
) -> pd.Series:
    values: list[
        str | None
    ] = []

    invalid: list[
        int
    ] = []

    for position, value in enumerate(
        series.tolist()
    ):
        if value is None:
            values.append(
                None
            )
            continue

        try:
            if pd.isna(
                value
            ):
                values.append(
                    None
                )
                continue

        except (TypeError, ValueError):
            pass

        try:
            parsed = uuid.UUID(
                str(
                    value
                ).strip()
            )

            values.append(
                str(
                    parsed
                )
            )

        except (
            ValueError,
            AttributeError,
        ):
            invalid.append(
                position
            )

            values.append(
                None
            )

    if invalid:
        _raise_cast_error(
            column_name=(
                str(
                    series.name
                )
            ),
            target_type=target_type,
            positions=invalid,
        )

    return pd.Series(
        values,
        index=series.index,
        name=series.name,
        dtype="object",
    )


def _cast_series(
    series: pd.Series,
    target_type: str,
) -> pd.Series:
    """
    Perform one safe, whitelisted datatype conversion.
    """

    family = (
        _target_type_family(
            target_type
        )
    )

    if family == "string":
        return (
            _cast_string(
                series
            )
        )

    if family == "number":
        return (
            _cast_number(
                series,
                target_type,
            )
        )

    if family == "integer":
        return (
            _cast_integer(
                series,
                target_type,
            )
        )

    if family == "boolean":
        return (
            _cast_boolean(
                series,
                target_type,
            )
        )

    if family == "datetime":
        return (
            _cast_datetime(
                series,
                target_type,
            )
        )

    if family == "date":
        return (
            _cast_date(
                series,
                target_type,
            )
        )

    if family == "uuid":
        return (
            _cast_uuid(
                series,
                target_type,
            )
        )

    raise ValueError(
        "Unsupported target datatype for safe cast: "
        f"'{target_type}'."
    )


# -------------------------------------------------------------------
# TRANSFORMATION EXECUTION
# -------------------------------------------------------------------

def _apply_transformation(
    series: pd.Series,
    transformation: str,
) -> pd.Series:
    """
    Execute only transformations created by our mapper.

    Arbitrary Python, SQL or expressions are never evaluated.
    """

    transformation = (
        transformation
        .strip()
    )

    if transformation == "direct":
        return (
            series.copy()
        )

    prefix = "cast:"

    if transformation.startswith(
        prefix
    ):
        target_type = (
            transformation[
                len(prefix):
            ]
            .strip()
        )

        if not target_type:
            raise ValueError(
                "Cast transformation is missing "
                "its target datatype."
            )

        return (
            _cast_series(
                series,
                target_type,
            )
        )

    raise ValueError(
        "Unsupported transformation "
        f"'{transformation}'."
    )


# -------------------------------------------------------------------
# MAPPING VALIDATION
# -------------------------------------------------------------------

def _accepted_mappings(
    mappings: list[
        MappingSuggestion
    ],
    approved_review_mappings: set[
        MappingKey
    ],
) -> list[
    MappingSuggestion
]:
    """
    Include automatic mappings and explicitly approved review mappings.
    """

    accepted: list[
        MappingSuggestion
    ] = []

    for mapping in mappings:

        if not (
            mapping.requires_review
        ):
            accepted.append(
                mapping
            )

            continue

        if (
            mapping_key(
                mapping
            )
            in approved_review_mappings
        ):
            accepted.append(
                mapping
            )

    return accepted


def _validate_mappings(
    asset: DataAsset,
    mappings: list[
        MappingSuggestion
    ],
    target_lookup: dict[
        str,
        dict[
            str,
            TargetColumn,
        ],
    ],
) -> None:
    """
    Validate every source and target reference before transforming.
    """

    seen_targets: set[
        tuple[str, str]
    ] = set()

    target_sources: dict[
        str,
        set[str],
    ] = {}

    for mapping in mappings:

        dataframe = (
            asset.tables.get(
                mapping.source_table
            )
        )

        if dataframe is None:
            raise ValueError(
                "Mapping references unknown source table "
                f"'{mapping.source_table}'."
            )

        if (
            mapping.source_column
            not in dataframe.columns
        ):
            raise ValueError(
                "Mapping references unknown source column "
                f"'{mapping.source_table}."
                f"{mapping.source_column}'."
            )

        target_columns = (
            target_lookup.get(
                mapping.target_table
            )
        )

        if target_columns is None:
            raise ValueError(
                "Mapping references unknown target table "
                f"'{mapping.target_table}'."
            )

        if (
            mapping.target_column
            not in target_columns
        ):
            raise ValueError(
                "Mapping references unknown target column "
                f"'{mapping.target_table}."
                f"{mapping.target_column}'."
            )

        target_key = (
            mapping.target_table,
            mapping.target_column,
        )

        if (
            target_key
            in seen_targets
        ):
            raise ValueError(
                "Multiple source columns map to the same "
                "target column "
                f"'{mapping.target_table}."
                f"{mapping.target_column}'."
            )

        seen_targets.add(
            target_key
        )

        target_sources.setdefault(
            mapping.target_table,
            set(),
        ).add(
            mapping.source_table
        )

    # Row-position merging across unrelated source tables is unsafe.
    for (
        target_table,
        source_tables,
    ) in target_sources.items():

        if len(
            source_tables
        ) > 1:
            readable = ", ".join(
                sorted(
                    source_tables
                )
            )

            raise ValueError(
                f"Target table '{target_table}' receives "
                "columns from multiple source tables "
                f"({readable}). "
                "An explicit join relationship is required "
                "before these sources can be combined."
            )


# -------------------------------------------------------------------
# TARGET CONSTRAINT VALIDATION
# -------------------------------------------------------------------

def _validate_target_column(
    series: pd.Series,
    target_column: TargetColumn,
    target_table: str,
) -> None:
    """
    Validate mapped target constraints that are explicitly represented
    in TargetColumn.

    Unmapped target columns are intentionally not guessed here because
    the current model does not describe database defaults/generated
    values.
    """

    if (
        not target_column.nullable
        and series.isna().any()
    ):
        failed = (
            np.flatnonzero(
                series.isna()
                .to_numpy()
            )
            .astype(int)
            .tolist()
        )

        raise ValueError(
            f"Target column '{target_table}."
            f"{target_column.column_name}' is non-nullable, "
            f"but transformed data contains nulls at "
            f"row position(s) {failed[:10]}."
        )

    if (
        target_column.primary_key
    ):
        if series.isna().any():
            raise ValueError(
                f"Primary-key target column '{target_table}."
                f"{target_column.column_name}' contains null values."
            )

        duplicate_mask = (
            series.duplicated(
                keep=False
            )
        )

        if duplicate_mask.any():
            failed = (
                np.flatnonzero(
                    duplicate_mask
                    .to_numpy()
                )
                .astype(int)
                .tolist()
            )

            raise ValueError(
                f"Primary-key target column '{target_table}."
                f"{target_column.column_name}' contains duplicates "
                f"at row position(s) {failed[:10]}."
            )


# -------------------------------------------------------------------
# PUBLIC API
# -------------------------------------------------------------------

def transform_data(
    asset: DataAsset,
    mappings: list[
        MappingSuggestion
    ],
    target_schema: TargetSchema,
    approved_review_mappings: set[
        MappingKey
    ] | None = None,
) -> dict[
    str,
    pd.DataFrame,
]:
    """
    Transform approved source-to-target mappings into target-shaped
    DataFrames.

    Rules:

    - mappings not requiring review are accepted automatically
    - reviewed mappings require explicit approval
    - only `direct` and safe `cast:<type>` transformations execute
    - multiple unrelated source tables are never merged by row position
    - target references and mapped constraints are validated
    """

    approvals = (
        approved_review_mappings
        or set()
    )

    target_lookup = (
        _target_lookup(
            target_schema
        )
    )

    column_order = (
        _target_column_order(
            target_schema
        )
    )

    accepted = (
        _accepted_mappings(
            mappings,
            approvals,
        )
    )

    if not accepted:
        return {}

    _validate_mappings(
        asset,
        accepted,
        target_lookup,
    )

    grouped: dict[
        str,
        list[
            MappingSuggestion
        ],
    ] = {}

    for mapping in accepted:

        grouped.setdefault(
            mapping.target_table,
            [],
        ).append(
            mapping
        )

    transformed: dict[
        str,
        pd.DataFrame,
    ] = {}

    for (
        target_table,
        table_mappings,
    ) in grouped.items():

        source_table = (
            table_mappings[0]
            .source_table
        )

        source_dataframe = (
            asset.tables[
                source_table
            ]
        )

        target_dataframe = pd.DataFrame(
            index=source_dataframe.index
        )

        for mapping in table_mappings:

            source_series = (
                source_dataframe[
                    mapping.source_column
                ]
            )

            transformed_series = (
                _apply_transformation(
                    source_series,
                    mapping.transformation,
                )
            )

            target_column = (
                target_lookup[
                    target_table
                ][
                    mapping.target_column
                ]
            )

            transformed_series.name = (
                mapping.target_column
            )

            _validate_target_column(
                transformed_series,
                target_column,
                target_table,
            )

            target_dataframe[
                mapping.target_column
            ] = transformed_series

        # Follow target-schema order rather than mapping order.
        ordered_columns = [
            column
            for column
            in column_order[
                target_table
            ]
            if column
            in target_dataframe.columns
        ]

        target_dataframe = (
            target_dataframe[
                ordered_columns
            ]
            .reset_index(
                drop=True
            )
        )

        transformed[
            target_table
        ] = target_dataframe

    return transformed