from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sqlalchemy import (
    MetaData,
    Table,
    create_engine,
    inspect,
    insert,
    text,
)
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

from src.config import settings
from src.models import (
    TargetColumn,
    TargetSchema,
    TargetTable,
)


# -------------------------------------------------------------------
# DATABASE RESULT MODELS
# -------------------------------------------------------------------

@dataclass
class TableLoadResult:
    """
    Result of loading one transformed dataframe.
    """

    table_name: str
    rows_received: int
    rows_inserted: int
    status: str
    message: str


@dataclass
class DatabaseLoadResult:
    """
    Summary of one transactional database load.
    """

    success: bool
    schema_name: str
    tables: list[TableLoadResult]
    total_rows_inserted: int
    message: str


class DatabaseError(RuntimeError):
    """
    Raised for database configuration, validation or loading failures.
    """


# -------------------------------------------------------------------
# ENGINE CREATION
# -------------------------------------------------------------------

def _require_database_enabled() -> None:
    """
    Prevent accidental database access when loading is disabled.
    """

    if not settings.database_enabled:
        raise DatabaseError(
            "Database access is disabled. "
            "Set DATABASE_ENABLED=true only when database operations "
            "are intentionally required."
        )


def _require_database_url() -> None:
    """
    Ensure database configuration is present before connecting.
    """

    if not (
        settings.database_url
        and settings.database_url.strip()
    ):
        raise DatabaseError(
            "DATABASE_URL is empty."
        )


def get_engine() -> Engine:
    """
    Create the SQLAlchemy engine used by the project.

    The function deliberately creates the engine only when database
    access is enabled.
    """

    _require_database_enabled()
    _require_database_url()

    try:
        return create_engine(
            settings.database_url,
            pool_pre_ping=True,
            future=True,
        )

    except Exception as exc:
        raise DatabaseError(
            "Unable to create database engine: "
            f"{exc}"
        ) from exc


# -------------------------------------------------------------------
# DATABASE STATUS
# -------------------------------------------------------------------

def database_status() -> dict[str, Any]:
    """
    Return configuration status without opening a connection when the
    database feature is disabled.
    """

    return {
        "enabled": (
            settings.database_enabled
        ),
        "configured": bool(
            settings.database_url
            and settings.database_url.strip()
        ),
        "schema": (
            settings.database_schema
        ),
    }


def test_database_connection(
    engine: Engine | None = None,
) -> bool:
    """
    Verify that the configured database is reachable.

    No data is written.
    """

    own_engine = (
        engine is None
    )

    if engine is None:
        engine = (
            get_engine()
        )

    try:
        with engine.connect() as connection:
            result = connection.execute(
                text(
                    "SELECT 1"
                )
            )

            value = (
                result.scalar_one()
            )

            return value == 1

    except SQLAlchemyError as exc:
        raise DatabaseError(
            "Database connection test failed: "
            f"{exc}"
        ) from exc

    finally:
        if own_engine:
            engine.dispose()


# -------------------------------------------------------------------
# DATABASE SCHEMA INTROSPECTION
# -------------------------------------------------------------------

def inspect_target_schema(
    table_names: Iterable[str] | None = None,
    engine: Engine | None = None,
) -> TargetSchema:
    """
    Read the real database schema and convert it into the project's
    TargetSchema model.

    This lets the mapper work against actual PostgreSQL tables rather
    than requiring manually written target schemas.
    """

    own_engine = (
        engine is None
    )

    if engine is None:
        engine = (
            get_engine()
        )

    schema_name = (
        settings.database_schema
    )

    try:
        inspector = (
            inspect(
                engine
            )
        )

        available_tables = set(
            inspector.get_table_names(
                schema=schema_name,
            )
        )

        if table_names is None:
            selected_tables = sorted(
                available_tables
            )

        else:
            requested = {
                str(name)
                for name in table_names
            }

            missing = (
                requested
                - available_tables
            )

            if missing:
                readable = ", ".join(
                    sorted(
                        missing
                    )
                )

                raise DatabaseError(
                    "Target table(s) not found in "
                    f"schema '{schema_name}': "
                    f"{readable}"
                )

            selected_tables = sorted(
                requested
            )

        target_tables: list[
            TargetTable
        ] = []

        for table_name in (
            selected_tables
        ):

            column_metadata = (
                inspector.get_columns(
                    table_name,
                    schema=schema_name,
                )
            )

            primary_key = (
                inspector.get_pk_constraint(
                    table_name,
                    schema=schema_name,
                )
            )

            primary_key_columns = set(
                primary_key.get(
                    "constrained_columns",
                    [],
                )
                or []
            )

            columns: list[
                TargetColumn
            ] = []

            for column in (
                column_metadata
            ):

                column_name = str(
                    column[
                        "name"
                    ]
                )

                description = (
                    column.get(
                        "comment"
                    )
                    or ""
                )

                columns.append(
                    TargetColumn(
                        column_name=(
                            column_name
                        ),
                        data_type=(
                            str(
                                column[
                                    "type"
                                ]
                            )
                        ),
                        nullable=bool(
                            column.get(
                                "nullable",
                                True,
                            )
                        ),
                        primary_key=(
                            column_name
                            in primary_key_columns
                        ),
                        description=(
                            description
                        ),
                    )
                )

            target_tables.append(
                TargetTable(
                    table_name=(
                        table_name
                    ),
                    columns=(
                        columns
                    ),
                )
            )

        return TargetSchema(
            tables=target_tables
        )

    except DatabaseError:
        raise

    except SQLAlchemyError as exc:
        raise DatabaseError(
            "Unable to inspect database schema: "
            f"{exc}"
        ) from exc

    finally:
        if own_engine:
            engine.dispose()


# -------------------------------------------------------------------
# PYTHON VALUE NORMALIZATION
# -------------------------------------------------------------------

def _to_database_value(
    value: Any,
) -> Any:
    """
    Convert pandas / NumPy values into values suitable for SQLAlchemy.
    """

    if value is None:
        return None

    try:
        if pd.isna(
            value
        ):
            return None

    except (
        TypeError,
        ValueError,
    ):
        pass

    if isinstance(
        value,
        np.generic,
    ):
        return value.item()

    if isinstance(
        value,
        pd.Timestamp,
    ):
        return (
            value.to_pydatetime()
        )

    if isinstance(
        value,
        (
            datetime,
            date,
        ),
    ):
        return value

    return value


def _dataframe_records(
    dataframe: pd.DataFrame,
) -> list[dict[str, Any]]:
    """
    Convert a dataframe to clean Python records for SQLAlchemy.
    """

    records: list[
        dict[str, Any]
    ] = []

    for row in dataframe.to_dict(
        orient="records"
    ):

        clean_row = {
            str(column): (
                _to_database_value(
                    value
                )
            )
            for (
                column,
                value,
            ) in row.items()
        }

        records.append(
            clean_row
        )

    return records


# -------------------------------------------------------------------
# TARGET SCHEMA VALIDATION
# -------------------------------------------------------------------

def _target_schema_lookup(
    target_schema: TargetSchema,
) -> dict[
    str,
    set[str],
]:
    """
    Convert TargetSchema into a lightweight validation lookup.
    """

    lookup: dict[
        str,
        set[str],
    ] = {}

    for table in (
        target_schema.tables
    ):

        if (
            table.table_name
            in lookup
        ):
            raise DatabaseError(
                "Target schema contains duplicate table "
                f"'{table.table_name}'."
            )

        lookup[
            table.table_name
        ] = {
            column.column_name
            for column
            in table.columns
        }

    return lookup


def _validate_transformed_data(
    transformed_tables: dict[
        str,
        pd.DataFrame,
    ],
    target_schema: TargetSchema,
) -> None:
    """
    Validate transformed table and column references before opening a
    write transaction.
    """

    target_lookup = (
        _target_schema_lookup(
            target_schema
        )
    )

    for (
        table_name,
        dataframe,
    ) in transformed_tables.items():

        if (
            table_name
            not in target_lookup
        ):
            raise DatabaseError(
                "Transformed output references target table "
                f"'{table_name}', but that table is absent "
                "from TargetSchema."
            )

        valid_columns = (
            target_lookup[
                table_name
            ]
        )

        unknown_columns = {
            str(column)
            for column
            in dataframe.columns
            if str(column)
            not in valid_columns
        }

        if unknown_columns:
            readable = ", ".join(
                sorted(
                    unknown_columns
                )
            )

            raise DatabaseError(
                "Transformed output contains unknown "
                f"column(s) for '{table_name}': "
                f"{readable}"
            )


def _validate_database_targets(
    engine: Engine,
    transformed_tables: dict[
        str,
        pd.DataFrame,
    ],
) -> None:
    """
    Confirm that every target table and inserted column really exists
    in the database.
    """

    schema_name = (
        settings.database_schema
    )

    inspector = (
        inspect(
            engine
        )
    )

    available_tables = set(
        inspector.get_table_names(
            schema=schema_name,
        )
    )

    for (
        table_name,
        dataframe,
    ) in transformed_tables.items():

        if (
            table_name
            not in available_tables
        ):
            raise DatabaseError(
                "Database target table "
                f"'{schema_name}.{table_name}' "
                "does not exist."
            )

        database_columns = {
            str(
                column[
                    "name"
                ]
            )
            for column
            in inspector.get_columns(
                table_name,
                schema=schema_name,
            )
        }

        missing_columns = {
            str(column)
            for column
            in dataframe.columns
            if str(column)
            not in database_columns
        }

        if missing_columns:
            readable = ", ".join(
                sorted(
                    missing_columns
                )
            )

            raise DatabaseError(
                "Database table "
                f"'{schema_name}.{table_name}' "
                "is missing mapped column(s): "
                f"{readable}"
            )


# -------------------------------------------------------------------
# INSERT HELPERS
# -------------------------------------------------------------------

def _batched(
    records: list[
        dict[str, Any]
    ],
    batch_size: int,
) -> Iterable[
    list[
        dict[str, Any]
    ]
]:
    """
    Yield deterministic insert batches.
    """

    for start in range(
        0,
        len(records),
        batch_size,
    ):
        yield records[
            start:
            start
            + batch_size
        ]


def _reflect_table(
    engine: Engine,
    table_name: str,
) -> Table:
    """
    Reflect an existing target table using SQLAlchemy rather than
    constructing raw SQL identifiers.
    """

    metadata = (
        MetaData()
    )

    return Table(
        table_name,
        metadata,
        schema=(
            settings.database_schema
        ),
        autoload_with=engine,
    )


# -------------------------------------------------------------------
# DATABASE LOADING
# -------------------------------------------------------------------

def load_transformed_data(
    transformed_tables: dict[
        str,
        pd.DataFrame,
    ],
    target_schema: TargetSchema,
    *,
    batch_size: int = 1000,
    engine: Engine | None = None,
) -> DatabaseLoadResult:
    """
    Insert transformed target-shaped DataFrames into existing database
    tables.

    Safety guarantees:

    - DATABASE_ENABLED must be true
    - target tables must already exist
    - target columns must already exist
    - no table replacement
    - no truncate
    - no delete
    - no arbitrary SQL
    - all inserts occur inside one transaction
    - any failure rolls back the complete load
    """

    _require_database_enabled()
    _require_database_url()

    if batch_size <= 0:
        raise ValueError(
            "batch_size must be greater than zero."
        )

    if not transformed_tables:
        return DatabaseLoadResult(
            success=True,
            schema_name=(
                settings.database_schema
            ),
            tables=[],
            total_rows_inserted=0,
            message=(
                "No transformed tables were supplied."
            ),
        )

    _validate_transformed_data(
        transformed_tables,
        target_schema,
    )

    own_engine = (
        engine is None
    )

    if engine is None:
        engine = (
            get_engine()
        )

    table_results: list[
        TableLoadResult
    ] = []

    total_inserted = 0

    try:
        _validate_database_targets(
            engine,
            transformed_tables,
        )

        reflected_tables = {
            table_name: (
                _reflect_table(
                    engine,
                    table_name,
                )
            )
            for table_name
            in transformed_tables
        }

        # engine.begin() creates one transaction.
        # Any exception automatically rolls back everything.
        with engine.begin() as connection:

            for (
                table_name,
                dataframe,
            ) in transformed_tables.items():

                rows_received = len(
                    dataframe
                )

                if dataframe.empty:

                    table_results.append(
                        TableLoadResult(
                            table_name=(
                                table_name
                            ),
                            rows_received=0,
                            rows_inserted=0,
                            status="skipped",
                            message=(
                                "The transformed dataframe "
                                "contains no rows."
                            ),
                        )
                    )

                    continue

                records = (
                    _dataframe_records(
                        dataframe
                    )
                )

                target_table = (
                    reflected_tables[
                        table_name
                    ]
                )

                inserted = 0

                for batch in (
                    _batched(
                        records,
                        batch_size,
                    )
                ):

                    connection.execute(
                        insert(
                            target_table
                        ),
                        batch,
                    )

                    inserted += len(
                        batch
                    )

                total_inserted += (
                    inserted
                )

                table_results.append(
                    TableLoadResult(
                        table_name=(
                            table_name
                        ),
                        rows_received=(
                            rows_received
                        ),
                        rows_inserted=(
                            inserted
                        ),
                        status="loaded",
                        message=(
                            f"Inserted {inserted} row(s)."
                        ),
                    )
                )

        return DatabaseLoadResult(
            success=True,
            schema_name=(
                settings.database_schema
            ),
            tables=(
                table_results
            ),
            total_rows_inserted=(
                total_inserted
            ),
            message=(
                "Database load completed successfully."
            ),
        )

    except DatabaseError:
        raise

    except SQLAlchemyError as exc:
        raise DatabaseError(
            "Database load failed. "
            "The transaction was rolled back. "
            f"Reason: {exc}"
        ) from exc

    except Exception as exc:
        raise DatabaseError(
            "Database load failed. "
            "The transaction was rolled back. "
            f"Reason: {exc}"
        ) from exc

    finally:
        if own_engine:
            engine.dispose()