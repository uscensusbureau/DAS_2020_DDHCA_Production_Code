"""A module that contains testing infrastructure for testing with Spark.
"""

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

import logging
import shutil
import unittest
from typing import Any, Sequence

import pandas as pd
from pyspark.sql import SparkSession


class PySparkTest(unittest.TestCase):
    """Create a pyspark testing base class for all tests.

    All the unit test methods in the same test class
    can share or reuse the same spark context.
    """

    _spark: SparkSession

    @property
    def spark(self) -> SparkSession:
        """Returns the spark session."""
        return self._spark

    @classmethod
    def suppress_py4j_logging(cls):
        """Remove noise in the logs irrelevant to testing."""
        print("Calling PySparkTest:suppress_py4j_logging")
        logger = logging.getLogger("py4j")
        # This is to silence py4j.java_gateway: DEBUG logs.
        logger.setLevel(logging.ERROR)

    @classmethod
    def setUpClass(cls):
        """Setup SparkSession."""
        cls.suppress_py4j_logging()
        print("Setting up spark session.")
        spark = (
            SparkSession.builder.appName(cls.__name__)
            .master("local[4]")
            .config("spark.sql.warehouse.dir", "/tmp/hive_tables")
            .config("spark.hadoop.fs.defaultFS", "file:///")
            .config("spark.eventLog.enabled", "false")
            .config("spark.driver.allowMultipleContexts", "true")
            .config("spark.ui.showConsoleProgress", "false")
            .config("spark.sql.execution.arrow.pyspark.enabled", "true")
            .config("spark.default.parallelism", "5")  
            .config("spark.memory.offHeap.enabled", "true")
            .config("spark.memory.offHeap.size", "16g")
            .getOrCreate()
        )
        # This is to silence pyspark logs.
        spark.sparkContext.setLogLevel("OFF")
        cls._spark = spark

    @classmethod
    def tearDownClass(cls):
        """Tears down SparkSession."""
        print("Tearing down spark session")
        shutil.rmtree("/tmp/hive_tables", ignore_errors=True)
        cls._spark.stop()

    @classmethod
    def assert_frame_equal_with_sort(
        cls,
        first_df: pd.DataFrame,
        second_df: pd.DataFrame,
        sort_columns: Sequence[str] = None,
        **kwargs: Any,
    ):
        """Asserts that the two data frames are equal.

        Wrapper around pandas test function. Both dataframes are sorted
        since the ordering in Spark is not guaranteed.

        Args:
            first_df: First dataframe to compare.
            second_df: Second dataframe to compare.
            sort_columns: Names of column to sort on. By default sorts by all columns.
            **kwargs: Keyword arguments that will be passed to assert_frame_equal().
        """
        if sorted(first_df.columns) != sorted(second_df.columns):
            raise AssertionError(
                "Dataframes must have matching columns. "
                f"first_df: {sorted(first_df.columns)}. "
                f"second_df: {sorted(second_df.columns)}."
            )
        if first_df.empty and second_df.empty:
            return
        if sort_columns is None:
            sort_columns = list(first_df.columns)
        if sort_columns:
            first_df = first_df.set_index(sort_columns).sort_index().reset_index()
            second_df = second_df.set_index(sort_columns).sort_index().reset_index()
        pd.testing.assert_frame_equal(first_df, second_df, **kwargs)
