"""Unit tests for :mod:`~tmlt.common.schema`."""

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

import unittest

import numpy as np
from pyspark.sql import types as spark_types

from tmlt.common.configuration import (
    CategoricalStr,
    Config,
    Continuous,
    Discrete,
    Splits,
)
from tmlt.common.schema import Schema


class TestSchema(unittest.TestCase):
    """Tests :class:`~tmlt.common.schema.Schema`"""

    def test_from_spark_schema(self):
        """Tests correct Schema is created using a spark schema"""
        spark_schema = spark_types.StructType(
            [
                spark_types.StructField("A", spark_types.StringType()),
                spark_types.StructField("B", spark_types.IntegerType()),
                spark_types.StructField("C", spark_types.LongType()),
                spark_types.StructField("D", spark_types.FloatType()),
                spark_types.StructField("E", spark_types.DoubleType()),
            ]
        )
        expected_schema_types = {"A": str, "B": int, "C": int, "D": float, "E": float}
        schema = Schema.from_spark_schema(spark_schema)

        self.assertEqual(expected_schema_types, schema.column_types)

    def test_from_pd_schema(self):
        """Tests correct Schema is created using a pandas schema."""
        pd_schema = {
            "A": np.dtype("int8"),
            "B": np.dtype("int16"),
            "C": np.dtype("int32"),
            "D": np.dtype("int64"),
            "E": np.dtype("float32"),
            "F": np.dtype("float64"),
            "G": np.dtype("object"),
        }

        expected_schema_types = {
            "A": int,
            "B": int,
            "C": int,
            "D": int,
            "E": float,
            "F": float,
            "G": str,
        }
        schema = Schema.from_pd_schema(pd_schema)

        self.assertEqual(expected_schema_types, schema.column_types)

    def test_from_config_object(self):
        """Tests correct Schema is created using a Config object."""
        config = Config(
            [
                CategoricalStr("A", ["str1", "str2"]),
                Discrete("B", 2, (10, 12)),
                Continuous("C", 3, (10, 20)),
                Splits("D", [0, 2, 4, 6, 8, 10]),
            ]
        )
        expected_schema_type = {"A": str, "B": int, "C": float, "D": float}
        schema = Schema.from_config_object(config)
        self.assertEqual(expected_schema_type, schema.column_types)

    def test_to_spark_schema(self):
        """Tests _to_spark_schema creates correct spark schema"""
        cols = {"A": int, "B": float, "C": str}
        schema = Schema.from_py_types(cols)
        expected_spark_schema = spark_types.StructType(
            [
                spark_types.StructField("A", spark_types.LongType()),
                spark_types.StructField("B", spark_types.DoubleType()),
                spark_types.StructField("C", spark_types.StringType()),
            ]
        )
        self.assertEqual(schema.spark_schema, expected_spark_schema)

    def test_to_pd_schema(self):
        """Tests _to_pd_schema creates correct spark schema"""
        cols = {"A": int, "B": float, "C": str}
        schema = Schema.from_py_types(cols)
        expected_pd_schema = {
            "A": np.dtype("int64"),
            "B": np.dtype("float64"),
            "C": np.dtype("object"),
        }
        self.assertEqual(schema.pd_schema, expected_pd_schema)

    def test_compatible_with_invalid_columns(self):
        """Tests compatible_with raises error when columns are not common."""
        schema_1 = Schema.from_py_types({"A": float, "B": float})
        schema_2 = Schema.from_py_types({"B": float, "C": str})
        with self.assertRaisesRegex(
            ValueError, "Supplied columns must be common to both schemas."
        ):
            schema_1.compatible_with(schema_2, ["B", "C"])
