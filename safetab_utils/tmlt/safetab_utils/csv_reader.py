"""SafeTab csv reader module for synthetic data."""

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

# pylint: disable=no-name-in-module

import functools
import os
from typing import List

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col, lit

from tmlt.analytics._schema import Schema, analytics_to_spark_schema
from tmlt.common.io_helpers import read_csv
from tmlt.safetab_utils.input_schemas import (
    GEO_SCHEMA,
    PERSON_SCHEMA,
    POP_GROUP_TOTAL_SCHEMA,
    UNIT_SCHEMA,
)
from tmlt.safetab_utils.reader_interface import (
    AbstractSafeTabHReader,
    AbstractSafeTabPReader,
)

GEO_FILENAME = "GRF-C.txt"
"""Filename for the geo df."""

UNIT_FILENAME = "household-records.txt"
"""Filename for the unit df."""

PERSON_FILENAME = "person-records.txt"
"""Filename for the person df."""

POP_GROUP_TOTAL_FILENAME = "pop-group-totals.txt"
"""Filename for the pop group totals df."""


class CSVPReader(AbstractSafeTabPReader):
    """The reader for csv files."""

    def __init__(self, config_path: str, state_filter: List[str]):
        """Sets up csv reader.

        Args:
            config_path: path to directory containing geo, unit, person files. a.k.a.
                the data path.
            state_filter: list of state codes to keep
        """
        self.config_path = config_path
        self.states = state_filter
        self.spark = SparkSession.builder.getOrCreate()

    def get_geo_df(self) -> DataFrame:
        """Returns geo df filtered per state_filter."""
        grfc_filename = os.path.join(self.config_path, GEO_FILENAME)
        schema = analytics_to_spark_schema(Schema(GEO_SCHEMA))
        geo_df = self.spark.createDataFrame(
            read_csv(
                grfc_filename, delimiter="|", dtype=str, usecols=list(GEO_SCHEMA.keys())
            ),
            schema=schema,
        )
        return geo_df.filter(col("TABBLKST").isin(self.states))

    def get_person_df(self) -> DataFrame:
        """Return person df filtered per state_filter."""
        schema = analytics_to_spark_schema(Schema(PERSON_SCHEMA))
        person_df = self.spark.read.csv(
            os.path.join(self.config_path, PERSON_FILENAME),
            schema=schema,
            header=True,
            sep="|",
        )
        return person_df.filter(col("TABBLKST").isin(self.states))


class CSVHReader(AbstractSafeTabHReader):
    """The reader for csv files."""

    def __init__(self, config_path: str, state_filter: List[str]):
        """Sets up csv reader.

        Args:
            config_path: path to directory containing geo, unit, person files. a.k.a
                the data path.
            state_filter: list of state codes to keep
        """
        self.config_path = config_path
        self.states = state_filter
        self.spark = SparkSession.builder.getOrCreate()
        self.geo_df = None

    def get_geo_df(self) -> DataFrame:
        """Returns geo df filtered per state_filter."""

        if self.geo_df:
            return self.geo_df

        grfc_filename = os.path.join(self.config_path, GEO_FILENAME)
        schema = analytics_to_spark_schema(Schema(GEO_SCHEMA))
        geo_df = self.spark.createDataFrame(
            read_csv(
                grfc_filename, delimiter="|", dtype=str, usecols=list(GEO_SCHEMA.keys())
            ),
            schema=schema,
        )

        self.geo_df = geo_df.filter(col("TABBLKST").isin(self.states))
        return self.geo_df

    def get_unit_df(self) -> DataFrame:
        """Returns unit df filtered per state_filter."""
        schema = analytics_to_spark_schema(Schema(UNIT_SCHEMA))
        unit_df = self.spark.read.csv(
            os.path.join(self.config_path, UNIT_FILENAME),
            schema=schema,
            header=True,
            sep="|",
        )
        return unit_df.filter(col("TABBLKST").isin(self.states))

    def get_pop_group_details_df(self) -> DataFrame:
        """Returns pop group details df filtered per state_filter."""
        schema = analytics_to_spark_schema(Schema(POP_GROUP_TOTAL_SCHEMA))
        pop_group_total_df = self.spark.read.csv(
            os.path.join(self.config_path, POP_GROUP_TOTAL_FILENAME),
            schema=schema,
            header=True,
            sep="|",
        )
        # This filters all regions except the AIANNH ones which can overlap states.
        # AIANNH is filtered below based on which AIANNH values pass the state
        # filter from the GRF-C.
        state_filter = col("REGION_TYPE").isin(["USA"]) | (
            ~col("REGION_TYPE").isin(["USA", "AIANNH"])
            & functools.reduce(
                lambda x, y: x | y,
                [col("REGION_ID").startswith(state) for state in self.states],
                lit(False),
            )
        )

        pop_group_non_aiannh = pop_group_total_df.filter(state_filter)

        # This filters for all AIANNH values that can exist in the states passed into
        # the state filter. It then inner joins onto the pop groups to leave only
        # possible aiannh values.

        aiannh_values = (
            self.get_geo_df()
            .withColumnRenamed("AIANNHCE", "REGION_ID")
            .select("REGION_ID")
            .distinct()
            .withColumn("REGION_TYPE", lit("AIANNH"))
        )
        aiannh_pop_groups = pop_group_total_df.join(
            aiannh_values, on=["REGION_ID", "REGION_TYPE"]
        )

        return pop_group_non_aiannh.union(aiannh_pop_groups)
