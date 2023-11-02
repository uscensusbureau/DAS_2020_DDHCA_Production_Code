"""Create ground truth t1 and t2 counts for SafeTab-P using pure spark computation.

Rather than selecting a subset of the possible counts for t2, it releases all possible
t2 counts for any population group that may have t2 counts released.
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
import argparse
import json
import logging
import os
import tempfile
from typing import Callable, Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import (  # pylint: disable=no-name-in-module
    array,
    col,
    explode,
    expr,
    lit,
    pandas_udf,
    when,
)
from pyspark.sql.types import (
    ArrayType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)
from smart_open import open  # pylint: disable=redefined-builtin

from tmlt.common.io_helpers import is_s3_path
from tmlt.safetab_p.paths import ALT_INPUT_CONFIG_DIR_SAFETAB_P, setup_input_config_dir
from tmlt.safetab_p.postprocessing import T1_COLUMNS, T2_COLUMNS
from tmlt.safetab_p.safetab_p_analytics import BINS, CONFIG_PARAMS_P
from tmlt.safetab_utils.characteristic_iterations import IterationManager
from tmlt.safetab_utils.input_validation import validate_input
from tmlt.safetab_utils.regions import (
    BLOCK_COLUMNS,
    REGION_TYPES,
    preprocess_geography_df,
    validate_state_filter_us,
)
from tmlt.safetab_utils.utils import (
    READER_FLAG,
    STATE_FILTER_FLAG,
    create_augmenting_map,
    safetab_input_reader,
)

WORKLOAD_SIZE_TO_STAT_LEVEL = {1: "0", 10: "1", 20: "2", 48: "3"}
"""A dictionary mapping the number of queries in a workload to the workload level."""


def create_ground_truth_p(
    parameters_path: str,
    data_path: str,
    output_path: str,
    config_path: str,
    safetab_output_paths: Optional[List[str]] = None,
    overwrite_config: Optional[Dict] = None,
    us_or_puerto_rico: str = "US",
    append: bool = False,
) -> None:
    """Save ground truth counts.

    If safetab_output_paths is provided, the paths are used to determine the workloads
    that the ground truth should be computed for. This can save time and space over
    computing the most detailed workload for every population group.

    Args:
        parameters_path: The location of the parameters directory.
        data_path: If csv reader, the location of CEF files.
            If cef reader, the file path to the reader config.
        output_path: The location to save t1 and t2. Appends to existing
            files if they exist.
        config_path: The location of the directory containing the schema files.
        safetab_output_paths: The optional list of safetab outputs used to create the
            ground truth.
        overwrite_config: Optional partial config that will overwrite any values for
            matching keys in the config that is read from config.json.
        us_or_puerto_rico: Whether to tabulate for the 50 states + DC ("US") or
            Puerto Rico ("PR").
        append: Whether to append to existing files, or overwrite.
    """
    spark_local_mode = (
        SparkSession.builder.getOrCreate().conf.get("spark.master").startswith("local")
    )
    if (is_s3_path(parameters_path) or is_s3_path(output_path)) and spark_local_mode:
        raise RuntimeError(
            "Reading and writing to and from s3"
            " is not supported when running Spark in local mode."
        )

    logger = logging.getLogger(__name__)  # pylint: disable=redefined-outer-name
    logger.info("Starting ground truth execution...")
    logger.info("with the following parameters:")
    with open(os.path.join(parameters_path, "config.json"), "r") as f:
        config_json = json.load(f)
    if overwrite_config is not None:
        for key, value in overwrite_config.items():
            if key in config_json:
                config_json[key] = value
            else:
                raise KeyError(key)
    for key in CONFIG_PARAMS_P:
        if key not in [
            # us_or_puerto_rico is logged instead.
            "run_us",
            "run_pr",
            # This isn't used for the nonprivate algorithm.
            "privacy_defn",
        ]:
            logger.info("\t%s: %s", key, config_json[key])
    logger.info("\tus_or_puerto_rico: %s", us_or_puerto_rico)
    logger.info(
        "Privacy budgets are only used to avoid tabulating groups"
        " with zero budget assigned."
    )

    # Read privacy budget allocation. If any (geography level, iteration level) has no
    # budget assigned, SafeTab will not produce a count, so we should not produce a
    # target count either. Note that these budgets are not used beyond determining which
    # population group levels to tabulate.
    if us_or_puerto_rico == "US":
        privacy_budget_allocation: Mapping[Tuple[str, str], float] = {
            ("USA", "1"): config_json["privacy_budget_p_level_1_usa"],
            ("USA", "2"): config_json["privacy_budget_p_level_2_usa"],
            ("STATE", "1"): config_json["privacy_budget_p_level_1_state"],
            ("STATE", "2"): config_json["privacy_budget_p_level_2_state"],
            ("COUNTY", "1"): config_json["privacy_budget_p_level_1_county"],
            ("COUNTY", "2"): config_json["privacy_budget_p_level_2_county"],
            ("TRACT", "1"): config_json["privacy_budget_p_level_1_tract"],
            ("TRACT", "2"): config_json["privacy_budget_p_level_2_tract"],
            ("PLACE", "1"): config_json["privacy_budget_p_level_1_place"],
            ("PLACE", "2"): config_json["privacy_budget_p_level_2_place"],
            ("AIANNH", "1"): config_json["privacy_budget_p_level_1_aiannh"],
            ("AIANNH", "2"): config_json["privacy_budget_p_level_2_aiannh"],
        }
    else:
        privacy_budget_allocation = {
            ("PR-STATE", "1"): config_json["privacy_budget_p_level_1_pr_state"],
            ("PR-STATE", "2"): config_json["privacy_budget_p_level_2_pr_state"],
            ("PR-COUNTY", "1"): config_json["privacy_budget_p_level_1_pr_county"],
            ("PR-COUNTY", "2"): config_json["privacy_budget_p_level_2_pr_county"],
            ("PR-TRACT", "1"): float(config_json["privacy_budget_p_level_1_pr_tract"]),
            ("PR-TRACT", "2"): float(config_json["privacy_budget_p_level_2_pr_tract"]),
            ("PR-PLACE", "1"): float(config_json["privacy_budget_p_level_1_pr_place"]),
            ("PR-PLACE", "2"): float(config_json["privacy_budget_p_level_2_pr_place"]),
        }

    # Validate state filtering.
    if us_or_puerto_rico == "US" and validate_state_filter_us(
        config_json[STATE_FILTER_FLAG]
    ):
        state_filter = config_json[STATE_FILTER_FLAG]
    else:
        state_filter = ["72"]

    spark = SparkSession.builder.getOrCreate()
    input_reader = safetab_input_reader(
        reader=config_json[READER_FLAG],
        data_path=data_path,
        state_filter=state_filter,
        program="safetab-p",
    )
    person_sdf = input_reader.get_person_df()

    # Create flat map function and apply it.
    iteration_manager = IterationManager(parameters_path, config_json["max_race_codes"])
    _, _, flatmap = iteration_manager.create_add_iterations_flat_map()
    person_sdf = person_sdf.rdd.flatMap(create_augmenting_map(flatmap)).toDF()

    # Create dataframe of pop groups. Start with geography df.
    pop_group_sdf = preprocess_geography_df(
        input_reader,
        us_or_puerto_rico=us_or_puerto_rico,
        input_config_dir_path=config_path,
    )
    if us_or_puerto_rico == "US":
        stack_expr = (
            "stack(6, 'USA', USA, 'STATE', STATE, 'COUNTY', COUNTY, 'TRACT', TRACT,"
            " 'PLACE', PLACE, 'AIANNH', AIANNH) as (REGION_TYPE, REGION_ID)"
        )
    else:
        stack_expr = (
            "stack(4, 'PR-STATE', `PR-STATE`, 'PR-COUNTY', `PR-COUNTY`, 'PR-TRACT',"
            " `PR-TRACT`, 'PR-PLACE', `PR-PLACE`) as (REGION_TYPE, REGION_ID)"
        )
    pop_group_sdf = pop_group_sdf.select(
        *(col(column) for column in BLOCK_COLUMNS), expr(stack_expr)
    ).where((col("REGION_ID") != "NULL"))

    # Add iteration codes and stat levels.
    if safetab_output_paths is not None:
        logger.info("Reading stat levels from safetab output...")
        stat_sdf = get_stat_level_from_safetab_output(safetab_output_paths)
        # This is a right join beacuse some population groups (i.e. those with no budget
        # assigned) will not be tabulated at all and therefore should be dropped from
        # the pop group sdf.
        pop_group_sdf = pop_group_sdf.join(
            stat_sdf, on=["REGION_TYPE", "REGION_ID"], how="right"
        )
    else:
        # Otherwise, use the most detailed stat level for non TOTAL_ONLY iterations, and
        # use the total count for TOTAL_ONLY iterations. Additionally, drop pop groups
        # with no budget assigned.
        iteration_detail_sdf = spark.createDataFrame(
            iteration_manager.get_iteration_df()[
                ["ITERATION_CODE", "DETAILED_ONLY", "COARSE_ONLY", "LEVEL"]
            ]
        )
        # Filter out region_type, level pairs with zero budget assigned.
        iteration_detail_sdf = iteration_detail_sdf.withColumn(
            "region_type_array",
            array(
                *(lit(region_type) for region_type in REGION_TYPES[us_or_puerto_rico])
            ),
        ).withColumn("REGION_TYPE", explode(col("region_type_array")))
        for (region_type, level), budget in privacy_budget_allocation.items():
            if budget == 0:
                iteration_detail_sdf = iteration_detail_sdf.where(
                    ~((col("LEVEL") == level) & (col("REGION_TYPE") == region_type))
                )
        # Filter out detailed only iterations at the county, tract,
        # place and aiannh level, and coarse only iterations at the usa
        # and state level.
        iteration_detail_sdf = iteration_detail_sdf.where(
            ~(col("REGION_TYPE").isin("USA", "STATE") & (col("COARSE_ONLY") == "True"))
        ).where(
            ~(
                col("REGION_TYPE").isin("COUNTY", "TRACT", "PLACE", "AIANNH")
                & (col("DETAILED_ONLY") == "True")
            )
        )

        stat_sdf = iteration_detail_sdf.withColumn(
            "STAT_LEVEL",
            when(col("DETAILED_ONLY") == "True", lit("0")).otherwise(lit("3")),
        )
        # This is a right join to drop the iteration_code, region_type pairs that
        # shouldn't be tabulated.
        pop_group_sdf = pop_group_sdf.join(stat_sdf, on="REGION_TYPE", how="right")

    # Compute results.
    t1 = (
        person_sdf.withColumn("COUNT", lit(1))
        .join(
            pop_group_sdf,
            on=["TABBLKST", "TABBLKCOU", "TABTRACTCE", "TABBLK", "ITERATION_CODE"],
            how="right",
        )
        .fillna({"COUNT": 0})
        .groupBy("REGION_ID", "REGION_TYPE", "ITERATION_CODE")
        .agg({"COUNT": "sum"})
        .withColumnRenamed("sum(COUNT)", "COUNT")
        .select(*T1_COLUMNS)
    )
    t2: Optional[DataFrame] = None
    for stat_level, age_bucket in [(1, "age4"), (2, "age9"), (3, "age23")]:
        # Skip the first bucket ([0,4]) for age23 since this bucket is shared between
        # age23 and age9.
        bins = (
            range(1, len(BINS[age_bucket]) + 1)
            if age_bucket == "age23"
            else range(len(BINS[age_bucket]) + 1)
        )
        full_domain_sdf = (
            pop_group_sdf.where(col("STAT_LEVEL") >= stat_level)
            .withColumn(age_bucket, explode(array(*(lit(i) for i in bins))))
            .withColumn("QSEX", explode(array(lit("1"), lit("2"))))
        )
        result_sdf = (
            person_sdf.withColumn("COUNT", lit(1))
            .withColumn(age_bucket, _get_digitizing_map(BINS[age_bucket])(col("QAGE")))
            .join(
                full_domain_sdf,
                on=[
                    "TABBLKST",
                    "TABBLKCOU",
                    "TABTRACTCE",
                    "TABBLK",
                    "ITERATION_CODE",
                    "QSEX",
                    age_bucket,
                ],
                how="right",
            )
            .fillna({"COUNT": 0})
            .groupBy("REGION_ID", "REGION_TYPE", "ITERATION_CODE", "QSEX", age_bucket)
            .agg({"COUNT": "sum"})
            .withColumn(
                "age_tuple", _get_age_range_map(BINS[age_bucket])(col(age_bucket))
            )
            .select(
                col("REGION_ID"),
                col("REGION_TYPE"),
                col("ITERATION_CODE"),
                col("age_tuple")[0].alias("AGESTART"),
                col("age_tuple")[1].alias("AGEEND"),
                col("QSEX").alias("SEX"),
                col("sum(COUNT)").alias("COUNT"),
            )
        )
        if t2 is None:
            # Male count and female count for every characteristic iteration eligible
            # for T2 statistics is output in T2.
            # Note that sex_marginal calculation should hapen only once; since there is
            # a for loop on age_bucket - if we do this on final t2 output,
            # we get count*3
            sex_marginal = (
                result_sdf.groupBy("REGION_ID", "REGION_TYPE", "ITERATION_CODE", "SEX")
                .agg({"COUNT": "sum"})
                .withColumnRenamed("sum(COUNT)", "COUNT")
                .withColumn("AGESTART", lit("*"))
                .withColumn("AGEEND", lit("*"))
                .select(*T2_COLUMNS)
            )
            t2 = result_sdf.unionAll(sex_marginal)
        else:
            t2 = t2.unionAll(result_sdf)

    t1.repartition(1).write.csv(
        os.path.join(output_path, "t1"),
        sep="|",
        mode="append" if append else "overwrite",
        header=True,
    )
    # help out mypy
    assert t2 is not None
    t2.repartition(1).write.csv(
        os.path.join(output_path, "t2"),
        sep="|",
        mode="append" if append else "overwrite",
        header=True,
    )
    # Write headers to a file. Spark doesn't write headers for empty files, so no
    # headers will exist if all files are empty.
    with open(os.path.join(output_path, "t1", "headers.csv"), "w") as f:
        f.write("REGION_ID|REGION_TYPE|ITERATION_CODE|COUNT")
    with open(os.path.join(output_path, "t2", "headers.csv"), "w") as f:
        f.write("REGION_ID|REGION_TYPE|ITERATION_CODE|AGESTART|AGEEND|SEX|COUNT")
    logger.info("Tabulating target counts completed successfully.")


def get_stat_level_from_safetab_output(safetab_output_paths: List[str]) -> DataFrame:
    """Return a dataframe containing the maximum stat levels used in all t2 outputs.

    Return a spark dataframe mapping population group to STAT_LEVEL. STAT_LEVEL contains
    the maximum stat level for that population group over all tables T2 in the provided
    safetab outputs. If the population group does not appear in T2, the STAT_LEVEL is 0.

    Args:
        safetab_output_paths: the list of paths containing the output of SafeTab.
    """

    @pandas_udf(StringType())
    def get_level(counts: pd.Series) -> pd.Series:
        return counts.map(WORKLOAD_SIZE_TO_STAT_LEVEL)

    spark = SparkSession.builder.getOrCreate()
    stat_level_schema = StructType(
        [
            StructField("REGION_ID", StringType()),
            StructField("REGION_TYPE", StringType()),
            StructField("ITERATION_CODE", StringType()),
            StructField("STAT_LEVEL", StringType()),
        ]
    )
    stat_levels: DataFrame = spark.createDataFrame([], stat_level_schema)
    for safetab_output_path in safetab_output_paths:
        t2 = (
            spark.read.csv(
                os.path.join(safetab_output_path, "t2"), sep="|", header=True
            )
            .select("REGION_ID", "REGION_TYPE", "ITERATION_CODE")
            .groupBy("REGION_ID", "REGION_TYPE", "ITERATION_CODE")
            .agg({"*": "count"})
            .select(
                col("REGION_ID"),
                col("REGION_TYPE"),
                col("ITERATION_CODE"),
                get_level(col("count(1)")).alias("STAT_LEVEL"),
            )
        )
        stat_levels = stat_levels.unionAll(t2)

        t1 = (
            spark.read.csv(
                os.path.join(safetab_output_path, "t1"), sep="|", header=True
            )
            .select("REGION_ID", "REGION_TYPE", "ITERATION_CODE")
            .withColumn("STAT_LEVEL", lit("0"))
        )
        stat_levels = stat_levels.unionAll(t1)

    stat_levels = (
        stat_levels.groupBy("REGION_ID", "REGION_TYPE", "ITERATION_CODE")
        .agg({"STAT_LEVEL": "max"})
        .withColumnRenamed("max(STAT_LEVEL)", "STAT_LEVEL")
    )

    return stat_levels


def run_target_counts_p(
    parameters_path: str,
    data_path: str,
    output_path: str,
    overwrite_config: Optional[Dict] = None,
) -> None:
    """Run the SafeTab-p target counts algorithm.

    First validates input files, and builds the expected domain of
    `person-records.txt` from files such as `GRF-C.txt`. See :mod:`.input_validation`
    for more details.

    .. warning::
        During validation, `person-records.txt` is checked against the expected domain,
        to make sure that the input files are consistent.

    Args:
        parameters_path: The location of the parameters directory.
        data_path: If csv reader, the location of CEF files.
            If cef reader, the file path to the reader config.
        output_path: The location to save t1 and t2.
        overwrite_config: Optional partial config that will overwrite any values for
            matching keys in the config that is read from config.json.
    """
    setup_input_config_dir()

    us_or_puerto_rico_values = []
    if overwrite_config is None:
        overwrite_config = dict()
    with open(os.path.join(parameters_path, "config.json"), "r") as f:
        config_json = json.load(f)
        config_json.update(overwrite_config)
    if config_json["run_us"]:
        us_or_puerto_rico_values.append("US")
    if config_json["run_pr"]:
        us_or_puerto_rico_values.append("PR")
    if not us_or_puerto_rico_values:
        raise ValueError(
            "Invalid config: At least one of 'run_us', 'run_pr' must be True."
        )

    with tempfile.TemporaryDirectory() as updated_config_dir:
        # Find states used in this execuition to validate input.
        state_filter = []
        if "US" in us_or_puerto_rico_values and validate_state_filter_us(
            config_json[STATE_FILTER_FLAG]
        ):
            state_filter += config_json[STATE_FILTER_FLAG]
        if "PR" in us_or_puerto_rico_values:
            state_filter += ["72"]

        if validate_input(
            parameters_path=parameters_path,
            input_data_configs_path=ALT_INPUT_CONFIG_DIR_SAFETAB_P,
            output_path=updated_config_dir,
            program="safetab-p",
            input_reader=safetab_input_reader(
                reader=config_json[READER_FLAG],
                data_path=data_path,
                state_filter=state_filter,
                program="safetab-p",
            ),
            state_filter=state_filter,
        ):
            for us_or_puerto_rico in us_or_puerto_rico_values:
                create_ground_truth_p(
                    parameters_path=parameters_path,
                    data_path=data_path,
                    output_path=output_path,
                    config_path=updated_config_dir,
                    overwrite_config=overwrite_config,
                    us_or_puerto_rico=us_or_puerto_rico,
                    # If US and PR are run together, append PR results to existing US
                    # results.
                    append=(
                        (us_or_puerto_rico == "PR")
                        and ("US" in us_or_puerto_rico_values)
                    ),
                )


def _get_digitizing_map(bins: List[int]) -> Callable[[pd.Series], pd.Series]:
    """Create a udf to digitize counts into bins.

    Args:
        bins: The list of bins.
    """

    @pandas_udf(IntegerType())
    def digitizing_map(counts: pd.Series) -> pd.Series:
        return pd.Series(np.digitize(counts, bins))

    return digitizing_map


def _get_age_range_map(bins: List[int]) -> Callable[[pd.Series], pd.Series]:
    """Create a map to get bin ranges from bin indices.

    Args:
        bins: The list of bins.
    """
    age_range_dict = {
        x: [0 if x == 0 else bins[x - 1], 115 if x == len(bins) else bins[x] - 1]
        for x in range(len(bins) + 1)
    }

    @pandas_udf(ArrayType(IntegerType()))
    def age_range_map(age_buckets: pd.Series) -> pd.Series:
        return age_buckets.map(age_range_dict)

    return age_range_map


def main(arglst: List[str] = None) -> None:
    """Main function.

    Args:
        arglst: Optional. List of args usually passed to commandline.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)

    parser = argparse.ArgumentParser(
        description="Create target counts for SafeTab-P."
    )  # pylint: disable-msg=C0103

    # Standard args
    parser.add_argument(
        "-i",
        "--input",
        dest="parameters_path",
        help="The directory the input files are stored.",
        required=True,
        type=str,
    )
    parser.add_argument(
        "-r",
        "--reader",
        dest="data_path",
        help=(
            "string used by the reader. The string is interpreted as an "
            "input csv files directory path "
            "for a csv reader or as a reader config file path for a cef reader."
        ),
        required=True,
        type=str,
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="output_path",
        help="The directory to store the output in.",
        required=True,
        type=str,
    )

    args = parser.parse_args(arglst)
    run_target_counts_p(
        parameters_path=args.parameters_path,
        data_path=args.data_path,
        output_path=args.output_path,
    )


if __name__ == "__main__":
    main()
