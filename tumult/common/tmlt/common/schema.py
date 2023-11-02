"""Module wrapping different representations of data schema."""

# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 Tumult Labs
# 
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
# 
#        http://www.apache.org/licenses/LICENSE-2.0
# 
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
from enum import Enum
from typing import Dict, Iterator, List, Optional, Sequence, cast

import numpy as np
from pyspark.sql import types as spark_types

from tmlt.common.configuration import Categorical, Config

SPARK_TO_PY_TYPE = {
    spark_types.LongType(): int,
    spark_types.IntegerType(): int,
    spark_types.ShortType(): int,
    spark_types.ByteType(): int,
    spark_types.DoubleType(): float,
    spark_types.FloatType(): float,
    spark_types.StringType(): str,
    spark_types.BooleanType(): bool,
}
"""Mapping from Spark type to Python type."""

DEFAULT_SPARK_TYPE = {
    int: spark_types.LongType(),
    float: spark_types.DoubleType(),
    str: spark_types.StringType(),
    bool: spark_types.BooleanType(),
}
"""Mapping from Python type to default Spark type."""

PD_TO_PY_TYPE = {
    np.dtype("int8"): int,
    np.dtype("int16"): int,
    np.dtype("int32"): int,
    np.dtype("int64"): int,
    np.dtype("float32"): float,
    np.dtype("float64"): float,
    np.dtype("object"): str,
    np.dtype("bool"): bool,
}
"""Mapping from Pandas type to Python type."""

DEFAULT_PD_TYPE = {
    int: np.dtype("int64"),
    float: np.dtype("float64"),
    str: np.dtype("object"),
    bool: np.dtype("bool"),
}
"""Mapping from Python type to default Pandas type."""


class PyKind(Enum):
    """Enumerating `kinds` of python types."""

    NUM = 1
    STRING = 2
    BOOL = 3


PY_KINDS = {float: PyKind.NUM, int: PyKind.NUM, str: PyKind.STRING, bool: PyKind.BOOL}
r"""Mapping of Python type to a category in :class:`PyKind`.

Note:
    This is required to allow schema compatibility checking with `pd.DataFrame`\ s
    as integer columns are cast to float types when a `NaN` value exists in an
    integer column (because `NaN`\ s can not be represented with NumPy integer types).
"""


class Schema:
    """A schema wrapper around Pandas and Spark schemas."""

    def __init__(
        self,
        column_types: Optional[Dict[str, type]] = None,
        spark_schema: Optional[spark_types.StructType] = None,
        pd_schema: Optional[Dict[str, np.dtype]] = None,
    ):
        """Construtor.

        Args:
            column_types: mapping from column name to python type.
            spark_schema: StructType schema containing Spark type for each column.
                        If not provided, default Spark types shall be used
                        as defined in :data:`DEFAULT_SPARK_TYPE` dictionary.
            pd_schema: mapping from column name to np.dtype. If not provided, default
                        NumPy types are used as defined in :data:`DEFAULT_PD_TYPE`.
        """
        if all(schema is None for schema in [column_types, spark_types, pd_schema]):
            raise ValueError("At least one schema must be provided.")

        self._column_types = column_types
        self._spark_schema = None
        self._pd_schema = None
        if spark_schema is not None:
            if len(set(spark_schema.names)) != len(spark_schema):
                raise ValueError(
                    f"Column name appears more than once in {spark_schema.names}"
                )
            self._spark_schema = spark_schema
            inferred_py_types = _infer_from_spark(spark_schema)
            if self._column_types:
                assert self._column_types == inferred_py_types
            else:
                self._column_types = inferred_py_types

        if pd_schema is not None:
            self._pd_schema = pd_schema
            inferred_py_types = _infer_from_pd(pd_schema)
            if self._column_types:
                assert self._column_types == inferred_py_types
            else:
                self._column_types = inferred_py_types

    def __eq__(self, other: object) -> bool:
        """Returns whether schemas are equal.

        Args:
            other: Schema that we are comparing against.
        """
        if not isinstance(other, Schema):
            return False

        return self.column_types == other.column_types

    @property
    def columns(self) -> List[str]:
        """Returns column names of all columns in the schema."""
        return list(self.column_types)

    @property
    def column_types(self) -> Dict[str, type]:
        """Returns a map from column names to python types."""
        assert self._column_types is not None
        return self._column_types

    @property
    def spark_schema(self) -> spark_types.StructType:
        """Returns corresponding Spark schema."""
        if self._spark_schema is None:
            self._spark_schema = self._to_spark_schema()
        return self._spark_schema

    @property
    def pd_schema(self) -> Dict[str, np.dtype]:
        """Return corresponding Pandas schema."""
        if self._pd_schema is None:
            self._pd_schema = self._to_pd_schema()
        return self._pd_schema

    @classmethod
    def from_py_types(cls, col_types: Dict[str, type]) -> "Schema":
        """Returns a Schema from a mapping of column name to python type.

        Args:
            col_types: mapping from column name to python type
        """
        return cls(column_types=col_types)

    @classmethod
    def from_spark_schema(cls, spark_schema: spark_types.StructType) -> "Schema":
        """Returns a Schema from a Spark schema.

        Args:
            spark_schema: Spark schema to be used.
        """
        return cls(spark_schema=spark_schema)

    @classmethod
    def from_pd_schema(cls, pd_schema: Dict[str, np.dtype]) -> "Schema":
        """Returns a Schema from a Pandas schema.

        Args:
            pd_schema: Pandas schema to be used.
        """
        return cls(pd_schema=pd_schema)

    @classmethod
    def from_config_object(cls, config: Config) -> "Schema":
        """Returns a Schema from a Config object.

        Args:
            config: Config to convert to a Schema.
        """
        col_types = {attr.column: attr.dtype for attr in config}
        return cls(column_types=col_types)

    def compatible_with(
        self, other_schema: "Schema", columns: Optional[Sequence[str]] = None
    ) -> bool:
        """Returns true if common columns have same *python* types.

        Args:
            other_schema: Schema to be checked against.
            columns: Columns to check compatibility on. If None, common columns are
                 checked.
        """
        common_columns = set(self.columns).intersection(other_schema.columns)
        if columns is None:
            columns = list(common_columns)
        else:
            if not set(columns) <= set(common_columns):
                raise ValueError("Supplied columns must be common to both schemas.")
        return all(
            self.column_types[col] == other_schema.column_types[col] for col in columns
        )

    def compatible_kinds_with(self, other_schema: "Schema") -> bool:
        """Returns true if common columns have same *kind*.

        Args:
            other_schema: Schema to be checked against.
        """
        return all(
            PY_KINDS[self.column_types[col]] == PY_KINDS[other_schema.column_types[col]]
            for col in set(self.columns).intersection(other_schema.columns)
        )

    def assert_column_numeric(self, col_name: str) -> None:
        """Raise exception if `col_name` is nonexistent or non-numeric.

        Args:
            col_name: Name of column to be checked.
        """
        if not col_name in self.columns:
            raise ValueError(f"Column {col_name} does not exist.")

        if PY_KINDS[self.column_types[col_name]] != PyKind.NUM:
            raise ValueError(f"Column {col_name} is not numeric.")

    def _to_spark_schema(self) -> spark_types.StructType:
        """Returns corresponding Spark schema using default conversion."""
        spark_fields = [
            spark_types.StructField(col_name, DEFAULT_SPARK_TYPE[col_type])
            for col_name, col_type in self.column_types.items()
        ]
        return spark_types.StructType(spark_fields)

    def _to_pd_schema(self) -> Dict[str, np.dtype]:
        """Returns corresponding Pandas schema using default conversion."""
        return {
            col_name: cast(np.dtype, DEFAULT_PD_TYPE[col_type])
            for col_name, col_type in self.column_types.items()
        }

    def __iter__(self) -> Iterator[str]:
        """Returns an iterator over the columns in the Schema."""
        return iter(self.columns)

    def __add__(self, other_schema: "Schema") -> "Schema":
        """Returns a new Schema with columns added from other_schema.

        Args:
            other_schema: Schema to be added.
        """
        if set(self.columns) & set(other_schema.columns):
            raise ValueError("Cannot add intersecting schemas")

        combined_spark_schema = spark_types.StructType(
            list(self.spark_schema) + list(other_schema.spark_schema)
        )
        combined_pd_schema = self.pd_schema.copy()
        combined_pd_schema.update(other_schema.pd_schema)
        return Schema(
            {**self.column_types, **other_schema.column_types},
            spark_schema=combined_spark_schema,
            pd_schema=combined_pd_schema,
        )


def _infer_from_spark(spark_schema: spark_types.StructType) -> Dict[str, type]:
    """Converts a Spark schema to a mapping from column names to python types.

    Args:
        spark_schema: StructType object to be used for inference.
    """
    schema_dict = dict(
        map(
            lambda struct_field: (struct_field.name, struct_field.dataType),
            spark_schema,
        )
    )
    return {
        name: SPARK_TO_PY_TYPE[spark_type] for name, spark_type in schema_dict.items()
    }


def _infer_from_pd(pd_schema: Dict[str, np.dtype]) -> Dict[str, type]:
    """Returns a mapping from column names to python types.

    Args:
        pd_schema: Mapping from column names to NumPy types.
    """
    return {name: PD_TO_PY_TYPE[pd_type] for name, pd_type in pd_schema.items()}


def check_partition_attributes(grouping_attributes: Config, schema: Schema) -> None:
    """Check if `grouping_attributes` have valid names and types.

    The following conditions are checked:

    * The `grouping_attributes` are present in `schema`.
    * The `grouping_attribtues` are all :class:`~tmlt.common.configuration.Categorical`.
    * The columns in `schema` are string type.

    Args:
        grouping_attributes: The attributes to group by data on.
        schema: The schema of the protected DataFrame to partition.

    Raises:
        ValueError: If one of the checked conditions fail.
    """
    for attribute in grouping_attributes:
        if attribute.column not in schema:
            raise ValueError(f"{attribute.column} does not exist")
        if not isinstance(attribute, Categorical):
            raise ValueError(f"Cannot partition on attribute of type {type(attribute)}")
        if attribute.dtype != schema.column_types[attribute.column]:
            raise ValueError("Attribute dtypes do not match schema")
