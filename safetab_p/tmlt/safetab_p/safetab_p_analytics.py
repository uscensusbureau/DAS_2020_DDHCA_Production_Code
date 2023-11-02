"""SafeTab-P Algorithm using analytics_cli.

Runs a differentially private mechanism to create t1 and t2.
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
import functools
import itertools
import json
import logging
import os
import re
import tempfile
from fractions import Fraction
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, cast

import numpy as np
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import (  # pylint: disable=no-name-in-module
    col,
    lit,
    split,
    udf,
    when,
)
from pyspark.sql.types import ArrayType, LongType, StringType, StructField, StructType
from smart_open import open  # pylint: disable=redefined-builtin

from tmlt.analytics._noise_info import _inverse_cdf
from tmlt.analytics.keyset import KeySet
from tmlt.analytics.privacy_budget import PrivacyBudget, PureDPBudget, RhoZCDPBudget
from tmlt.analytics.query_builder import ColumnType, QueryBuilder
from tmlt.analytics.query_expr import CountMechanism
from tmlt.analytics.session import Session
from tmlt.common.configuration import CategoricalStr
from tmlt.common.io_helpers import is_s3_path
from tmlt.safetab_p.paths import (
    ALT_INPUT_CONFIG_DIR_SAFETAB_P,
    get_safetab_p_output_configs,
    setup_input_config_dir,
    setup_safetab_p_output_config_dir,
)
from tmlt.safetab_p.postprocessing import T1_COLUMNS, T2_COLUMNS, postprocess_dfs
from tmlt.safetab_utils.characteristic_iterations import IterationManager
from tmlt.safetab_utils.config_validation import CONFIG_PARAMS_P, validate_config_values
from tmlt.safetab_utils.input_schemas import GEO_SCHEMA, PERSON_SCHEMA
from tmlt.safetab_utils.input_validation import validate_input
from tmlt.safetab_utils.output_validation import validate_output
from tmlt.safetab_utils.regions import (
    REGION_GRANULARITY_MAPPING,
    REGION_TYPES,
    preprocess_geography_df,
    validate_state_filter_us,
)
from tmlt.safetab_utils.utils import (
    READER_FLAG,
    STATE_FILTER_FLAG,
    safetab_input_reader,
)

BINS = {
    "age4": [18, 45, 65],
    "age9": [5, 18, 25, 35, 45, 55, 65, 75],
    "age23": [
        5,
        10,
        15,
        18,
        20,
        21,
        22,
        25,
        30,
        35,
        40,
        45,
        50,
        55,
        60,
        62,
        65,
        67,
        70,
        75,
        80,
        85,
    ],
}
"""Bins for QAGE split into groups."""

STAT_TO_AGE_COL = {0: "*", 1: "age4", 2: "age9", 3: "age23"}
"""Dict to map stat_levels to age columns."""


T2_SCHEMA = StructType(
    [
        StructField(column_name, StringType())
        for column_name in T2_COLUMNS
        if column_name != "COUNT"
    ]
    + [StructField("COUNT", LongType())]
)
"""Spark schema for T2 output columns."""


def execute_plan_p_analytics(
    parameters_path: str,
    data_path: str,
    output_path: str,
    config_path: str,
    overwrite_config: Optional[Dict] = None,
    us_or_puerto_rico: str = "US",
    append: bool = False,
    should_validate_private_output: bool = False,
) -> None:
    """Run the SafeTab-P algorithm and save the results.

    Args:
        parameters_path: The location of the config and the race/ethnicity files.
        data_path: If csv reader, the location of input files.
            If cef reader, the file path to the reader config.
        output_path: The location to save t1 and t2 output files.
        config_path: The location of the directory containing the schema files.
        overwrite_config: Optional partial config that will overwrite values from
            the config file. All keys in the overwrite config must be present in the
            config file.
        us_or_puerto_rico: Whether to tabulate for the 50 states + DC ("US") or
            Puerto Rico ("PR").
        append: Whether to append to existing files, or overwrite.
        should_validate_private_output: If True, validate
            private output after tabulations.
    """
    spark_local_mode = (
        SparkSession.builder.getOrCreate().conf.get("spark.master").startswith("local")
    )
    if (is_s3_path(parameters_path) or is_s3_path(output_path)) and spark_local_mode:
        raise RuntimeError(
            "Reading and writing to and from s3"
            " is not supported when running Spark in local mode."
        )

    logger = logging.getLogger(__name__)
    logger.info("Starting SafeTab-P execution...")
    logger.info("with the following parameters:.")
    with open(os.path.join(parameters_path, "config.json"), "r") as f:
        config_json = json.load(f)
    if overwrite_config is not None:
        for key, value in overwrite_config.items():
            if key in config_json:
                config_json[key] = value
            else:
                raise KeyError(key)
    for key in CONFIG_PARAMS_P:
        if key not in ["run_us", "run_pr"]:  # us_or_puerto_rico is logged instead.
            logger.info("\t%s: %s", key, config_json[key])
    logger.info("\tus_or_puerto_rico: %s", us_or_puerto_rico)
    if "zero_suppression_chance" in config_json:
        logger.info(
            f"\tzero_suppression_chance: {config_json['zero_suppression_chance']}"
        )

    validate_config_values(config_json, "safetab-p", [us_or_puerto_rico])

    # Read and store the privacy budget values and the fraction of budget that should be
    # used in stage 1 of the algorithm.
    if us_or_puerto_rico == "US":
        budget_floats: Mapping[Tuple[str, str], float] = {
            ("USA", "1"): float(config_json["privacy_budget_p_level_1_usa"]),
            ("USA", "2"): float(config_json["privacy_budget_p_level_2_usa"]),
            ("STATE", "1"): float(config_json["privacy_budget_p_level_1_state"]),
            ("STATE", "2"): float(config_json["privacy_budget_p_level_2_state"]),
            ("COUNTY", "1"): float(config_json["privacy_budget_p_level_1_county"]),
            ("COUNTY", "2"): float(config_json["privacy_budget_p_level_2_county"]),
            ("TRACT", "1"): float(config_json["privacy_budget_p_level_1_tract"]),
            ("TRACT", "2"): float(config_json["privacy_budget_p_level_2_tract"]),
            ("PLACE", "1"): float(config_json["privacy_budget_p_level_1_place"]),
            ("PLACE", "2"): float(config_json["privacy_budget_p_level_2_place"]),
            ("AIANNH", "1"): float(config_json["privacy_budget_p_level_1_aiannh"]),
            ("AIANNH", "2"): float(config_json["privacy_budget_p_level_2_aiannh"]),
        }
    else:
        budget_floats = {
            ("PR-STATE", "1"): float(config_json["privacy_budget_p_level_1_pr_state"]),
            ("PR-STATE", "2"): float(config_json["privacy_budget_p_level_2_pr_state"]),
            ("PR-COUNTY", "1"): float(
                config_json["privacy_budget_p_level_1_pr_county"]
            ),
            ("PR-COUNTY", "2"): float(
                config_json["privacy_budget_p_level_2_pr_county"]
            ),
            ("PR-TRACT", "1"): float(config_json["privacy_budget_p_level_1_pr_tract"]),
            ("PR-TRACT", "2"): float(config_json["privacy_budget_p_level_2_pr_tract"]),
            ("PR-PLACE", "1"): float(config_json["privacy_budget_p_level_1_pr_place"]),
            ("PR-PLACE", "2"): float(config_json["privacy_budget_p_level_2_pr_place"]),
        }
    stage_1_budget_fraction = Fraction(config_json["privacy_budget_p_stage_1_fraction"])
    total_budget_value = sum(budget_floats.values())

    (total_budget, noise_mechanism) = (
        (PureDPBudget(total_budget_value), CountMechanism.LAPLACE)
        if config_json["privacy_defn"] == "puredp"
        else (RhoZCDPBudget(total_budget_value), CountMechanism.GAUSSIAN)
    )
    logger.info("Total budget: %s", str(total_budget))

    logger.info("Getting data...")
    # Validate state filtering
    if us_or_puerto_rico == "US" and validate_state_filter_us(
        config_json[STATE_FILTER_FLAG]
    ):
        state_filter = config_json[STATE_FILTER_FLAG]
    else:
        state_filter = ["72"]

    # LOAD DATA
    # Get person records dataframe
    input_reader = safetab_input_reader(
        reader=config_json[READER_FLAG],
        data_path=data_path,
        state_filter=state_filter,
        program="safetab-p",
    )
    person_df = input_reader.get_person_df()

    # Preprocess geography_df.
    # Contains one column for each column in REGION_TYPES[us_or_puerto_rico] and shares
    # BLOCK_COLUMNS with person-records.txt.
    geo_df = preprocess_geography_df(
        input_reader,
        us_or_puerto_rico=us_or_puerto_rico,
        input_config_dir_path=config_path,
    )

    logger.info("Creating the session...")
    # CREATE THE SESSION
    session = Session.from_dataframe(
        privacy_budget=total_budget, source_id="person_source", dataframe=person_df
    )
    session.add_public_dataframe("geo_source", geo_df)

    logger.info("Creating the root query builder...")
    # CREATE THE ROOT QUERY BUILDERS.
    # Do a public join with the geo dataframe and create age bins.
    root_builder = (
        QueryBuilder(source_id="person_source")
        .join_public(public_table="geo_source")
        .map(
            lambda row: {
                "age4": int(np.digitize(row["QAGE"], BINS["age4"])),
                "age9": int(np.digitize(row["QAGE"], BINS["age9"])),
                "age23": int(np.digitize(row["QAGE"], BINS["age23"])),
            },
            new_column_types={
                "age4": ColumnType.INTEGER,
                "age9": ColumnType.INTEGER,
                "age23": ColumnType.INTEGER,
            },
            augment=True,
        )
        .select(
            columns=["age4", "age9", "age23"]
            + [
                col
                for col in PERSON_SCHEMA
                if col not in GEO_SCHEMA
                and col not in ["QAGE", "HOUSEHOLDER", "CENRACE"]
            ]
            + REGION_TYPES[us_or_puerto_rico]
        )
    )
    session.create_view(root_builder, "root", cache=True)

    # Create a different root for each iteration and region_type.
    iteration_manager = IterationManager(parameters_path, config_json["max_race_codes"])
    # Get df with mapping of iteration codes to detailed_only
    spark = SparkSession.builder.getOrCreate()
    detail_df = spark.createDataFrame(
        iteration_manager.get_iteration_df()[["ITERATION_CODE", "DETAILED_ONLY"]]
    )
    session.add_public_dataframe("detail_source", detail_df)

    # Get domains needed
    age_domains = {
        "age4": list(range(4)),
        "age9": list(range(9)),
        "age23": list(range(23)),
    }
    region_domains = {}
    for region_type in REGION_TYPES[us_or_puerto_rico]:
        domain = list(
            geo_df.filter(col(region_type) != "NULL")
            .select(region_type)
            .distinct()
            .toPandas()[region_type]
        )
        region_domains[region_type] = domain

    pop_group_domains: Dict[Tuple[str, str], List] = {}
    detailed_only_result_sdfs: List[DataFrame] = []
    non_detailed_only_result_sdfs: List[DataFrame] = []
    for region_type, iteration_level in itertools.product(
        REGION_TYPES[us_or_puerto_rico], ["1", "2"]
    ):
        granularity = REGION_GRANULARITY_MAPPING[region_type]
        # Create race-code to characteristic iteration function
        if granularity == "coarse":
            flat_map_container = iteration_manager.create_add_pop_groups_flat_map(
                detailed_only="False",
                coarse_only="both",
                level=iteration_level,
                region_type=region_type,
                region_domain=region_domains[region_type],
            )
        else:
            flat_map_container = iteration_manager.create_add_pop_groups_flat_map(
                detailed_only="both",
                coarse_only="False",
                level=iteration_level,
                region_type=region_type,
                region_domain=region_domains[region_type],
            )

        iteration_and_region_builder = (
            QueryBuilder("root")
            .flat_map(
                flat_map_container.flat_map,
                max_num_rows=flat_map_container.sensitivity,
                new_column_types={"POP_GROUP": ColumnType.VARCHAR},
                augment=True,
                grouping=True,
            )
            .map(
                lambda row: {"ITERATION_CODE": row["POP_GROUP"].split(",")[1]},
                new_column_types={"ITERATION_CODE": ColumnType.VARCHAR},
                augment=True,
            )
            .join_public(public_table="detail_source")
            .select(
                columns=["age23", "age9", "age4", "QSEX"]
                + REGION_TYPES[us_or_puerto_rico]
                + ["POP_GROUP"]
                + ["DETAILED_ONLY"]
            )
        )
        region_identifier = region_type.replace("-", "")
        session.create_view(
            iteration_and_region_builder,
            f"safetab_{iteration_level}_{region_identifier}",
            cache=True,
        )

        partition_budget_value = budget_floats[(region_type, iteration_level)]
        partition_budget = (
            PureDPBudget(partition_budget_value)
            if isinstance(total_budget, PureDPBudget)
            else RhoZCDPBudget(partition_budget_value)
        )
        detail_sessions = session.partition_and_create(
            f"safetab_{iteration_level}_{region_identifier}",
            privacy_budget=partition_budget,
            column="DETAILED_ONLY",
            splits={"detailed_only": "True", "non_detailed_only": "False"},
        )

        detailed_only_session = detail_sessions["detailed_only"]
        non_detailed_only_session = detail_sessions["non_detailed_only"]

        pop_group_domains[(iteration_level, region_type)] = cast(
            CategoricalStr, flat_map_container.output_domain["POP_GROUP"]
        ).values

        for k in detail_sessions:
            if k == "detailed_only":
                detailed_only_result_sdfs += detail_only_queries(
                    logger=logger,
                    region_domains=region_domains,
                    budget_floats=budget_floats,
                    total_budget=total_budget,
                    detailed_only_session=detailed_only_session,
                    pop_group_domains=pop_group_domains,
                    iteration_manager=iteration_manager,
                    noise_mechanism=noise_mechanism,
                    config_json=config_json,
                    spark=spark,
                    region_type=region_type,
                    iteration_level=iteration_level,
                )
                detailed_only_session.stop()
            else:
                # Do the non-detailed-only query
                non_detailed_only_result_sdfs += non_detail_only_queries(
                    logger=logger,
                    region_domains=region_domains,
                    budget_floats=budget_floats,
                    total_budget=total_budget,
                    non_detailed_only_session=non_detailed_only_session,
                    pop_group_domains=pop_group_domains,
                    iteration_manager=iteration_manager,
                    noise_mechanism=noise_mechanism,
                    config_json=config_json,
                    spark=spark,
                    stage_1_budget_fraction=stage_1_budget_fraction,
                    age_domains=age_domains,
                    iteration_level=iteration_level,
                    region_type=region_type,
                )
                non_detailed_only_session.stop()
        session.delete_view(f"safetab_{iteration_level}_{region_identifier}")

    session.delete_view("root")

    result_sdfs = detailed_only_result_sdfs + non_detailed_only_result_sdfs
    t1_t2_sdf = functools.reduce(DataFrame.union, result_sdfs)
    t1_t2_sdf.cache()
    t1_sdf = t1_t2_sdf.where("AGESTART='*' AND AGEEND='*' AND SEX='*'").select(
        *T1_COLUMNS
    )
    t2_sdf = t1_t2_sdf.where("NOT(AGESTART='*' AND AGEEND='*' AND SEX='*')").select(
        *T2_COLUMNS
    )

    # Total population for every characteristic iteration is output in T1
    t2_total_sdf = (
        t2_sdf.groupBy("REGION_ID", "REGION_TYPE", "ITERATION_CODE")
        .agg({"COUNT": "sum"})
        .withColumnRenamed("sum(COUNT)", "COUNT")
        .select(*T1_COLUMNS)
    )
    t1_combined_sdf = t1_sdf.unionAll(t2_total_sdf)

    # Male count and female count for every characteristic iteration eligible for T2
    # statistics is output in T2.
    t2_sex_marginal = (
        t2_sdf.groupBy("REGION_ID", "REGION_TYPE", "ITERATION_CODE", "SEX")
        .agg({"COUNT": "sum"})
        .withColumnRenamed("sum(COUNT)", "COUNT")
        .withColumn("AGESTART", lit("*"))
        .withColumn("AGEEND", lit("*"))
        .select(*T2_COLUMNS)
    )
    t2_combined_sdf = t2_sdf.unionAll(t2_sex_marginal)

    t1_combined_sdf.repartition(1).write.csv(
        os.path.join(output_path, "t1"),
        sep="|",
        mode="append" if append else "overwrite",
        header=True,
    )
    t2_combined_sdf.repartition(1).write.csv(
        os.path.join(output_path, "t2"),
        sep="|",
        mode="append" if append else "overwrite",
        header=True,
    )
    t1_t2_sdf.unpersist()
    # Write headers to a file. Spark doesn't write headers for empty files, so no
    # headers will exist if all files are empty.
    with open(os.path.join(output_path, "t1", "headers.csv"), "w") as f:
        f.write("|".join(t1_combined_sdf.columns))
    with open(os.path.join(output_path, "t2", "headers.csv"), "w") as f:
        f.write("|".join(t2_combined_sdf.columns))

    if should_validate_private_output:
        if not validate_output(
            output_sdfs={"t1": t1_combined_sdf, "t2": t2_combined_sdf},
            expected_output_configs=get_safetab_p_output_configs(),
            state_filter=state_filter,
            allow_negative_counts_flag=config_json["allow_negative_counts"],
        ):
            logger.error("SafeTab-P output validation failed. Exiting...")
            raise RuntimeError("Output validation Failed.")
    logger.info("SafeTab-P completed successfully.")


def detail_only_queries(
    logger: logging.Logger,
    region_domains: Dict[str, List[str]],
    budget_floats: Mapping[Tuple[str, str], float],
    total_budget: PrivacyBudget,
    detailed_only_session: Session,
    pop_group_domains: Dict[Tuple[str, str], List],
    iteration_manager: IterationManager,
    noise_mechanism: CountMechanism,
    config_json: Dict[str, Any],
    spark: SparkSession,
    region_type: str,
    iteration_level: str,
) -> List[DataFrame]:
    """Perform all queries for detailed iterations.

    Detailed iterations are those for which DETAILED_ONLY = True. Detailed iterations
    are not eligible to recieve a T2 breakdown, and are only tabulated for nation
    and state-level regions.
    """
    result_sdfs: List[DataFrame] = []
    logger.info(
        f"Evaluating part 1 for region_type: {region_type}, "
        f"iteration_level: {iteration_level}."
    )

    if (
        budget_floats[(region_type, iteration_level)] == 0
        or len(region_domains[region_type]) == 0
    ):
        return result_sdfs

    # Get necessary variables for query building and evaluation.
    budget_value = budget_floats[(region_type, iteration_level)]
    budget = (
        PureDPBudget(budget_value)
        if isinstance(total_budget, PureDPBudget)
        else RhoZCDPBudget(budget_value)
    )

    # NON-ADAPTIVE, DETAILED-ONLY QUERY
    # Create a query to evaluate the population group size.
    detailed_only_pop_group_domain = [
        pop_group
        for pop_group in pop_group_domains[(iteration_level, region_type)]
        if iteration_manager.is_detailed_only(pop_group.split(",")[1])
    ]
    if not detailed_only_pop_group_domain:
        return result_sdfs

    detailed_only_query = (
        QueryBuilder("detailed_only")
        .groupby(KeySet.from_dict({"POP_GROUP": detailed_only_pop_group_domain}))
        .count(name="COUNT", mechanism=noise_mechanism)
    )

    # Evaluate.
    detailed_only_query_answer = (
        detailed_only_session.evaluate(
            query_expr=detailed_only_query, privacy_budget=budget
        )
        .withColumn("REGION_TYPE", lit(region_type))
        .withColumn("REGION_ID", split(col("POP_GROUP"), ",").getItem(0))
        .withColumn("ITERATION_CODE", split(col("POP_GROUP"), ",").getItem(1))
        .drop("POP_GROUP")
        .drop(region_type)
        .withColumn("QAGE", lit("*"))
        .withColumn("QSEX", lit("*"))
    )
    noise_info = detailed_only_session._noise_info(  # pylint: disable=protected-access
        query_expr=detailed_only_query, privacy_budget=budget
    )
    if "zero_suppression_chance" in config_json and region_type not in (
        "USA",
        "STATE",
        "PR-STATE",
    ):
        suppression_threshold = _get_suppression_threshold(
            noise_info=noise_info[0],
            zero_suppression_chance=config_json["zero_suppression_chance"],
        )

        logger.info(
            f"Thresholding T1 counts at {region_type}, iteration level "
            f"{iteration_level}, detailed iterations. Threshold: "
            f"{suppression_threshold}"
        )
        detailed_only_query_answer = detailed_only_query_answer.filter(
            col("COUNT") > lit(suppression_threshold)
        )
    if not config_json["allow_negative_counts"]:
        detailed_only_query_answer = detailed_only_query_answer.withColumn(
            "COUNT", when(col("COUNT") < 0, 0).otherwise(col("COUNT"))
        )
    result_sdfs.append(
        spark.createDataFrame(
            postprocess_dfs([detailed_only_query_answer.toPandas()]), T2_SCHEMA
        )
    )
    return result_sdfs


def non_detail_only_queries(
    logger: logging.Logger,
    region_domains: Dict[str, List[str]],
    budget_floats: Mapping[Tuple[str, str], float],
    total_budget: PrivacyBudget,
    non_detailed_only_session: Session,
    pop_group_domains: Dict[Tuple[str, str], List],
    iteration_manager: IterationManager,
    noise_mechanism: CountMechanism,
    config_json: Dict[str, Any],
    spark: SparkSession,
    stage_1_budget_fraction: Fraction,
    age_domains: Dict[str, List[int]],
    iteration_level: str,
    region_type: str,
) -> List[DataFrame]:
    """Perform all queries for common and other itations.

    Common and other iterations are those for which DETAILED_ONLY = False. These
    iterations are all eligible to recieve T2 breakdowns.
    """
    result_sdfs: List[DataFrame] = []
    logger.info(
        "Evaluating part 2 for region_type: %s, iteration_level: %s.",
        region_type,
        iteration_level,
    )

    region_identifier = region_type.replace("-", "")

    if (
        budget_floats[(region_type, iteration_level)] == 0
        or len(region_domains[region_type]) == 0
    ):
        return result_sdfs

    # Get necessary variables for query building and evaluation.
    # We do budget-splitting math with floating-points, which can be imprecise.
    # However, Analytics automatically detects and corrects budgets that slightly
    # exceed the total, so there is no risk of problems from attempted over-spending.
    budget_1_value = (
        budget_floats[(region_type, iteration_level)] * stage_1_budget_fraction
    )
    budget_2_value = budget_floats[(region_type, iteration_level)] * (
        1 - stage_1_budget_fraction
    )
    budget_1 = (
        PureDPBudget(budget_1_value)
        if isinstance(total_budget, PureDPBudget)
        else RhoZCDPBudget(budget_1_value)
    )
    budget_2 = (
        PureDPBudget(budget_2_value)
        if isinstance(total_budget, PureDPBudget)
        else RhoZCDPBudget(budget_2_value)
    )

    # ADAPTIVE, NON-DETAILED-ONLY QUERY
    non_detailed_only_pop_group_domain = [
        pop_group
        for pop_group in pop_group_domains[(iteration_level, region_type)]
        if not iteration_manager.is_detailed_only(pop_group.split(",")[1])
    ]
    if not non_detailed_only_pop_group_domain:
        return result_sdfs

    # First part of the adaptive query. Finding the stat-levels.
    non_detailed_only_total_query = (
        QueryBuilder("non_detailed_only")
        .groupby(KeySet.from_dict({"POP_GROUP": non_detailed_only_pop_group_domain}))
        .count(mechanism=noise_mechanism)
    )

    # Evaluate the query and obtain the stat-levels based on thresholds.
    non_detailed_result = non_detailed_only_session.evaluate(
        query_expr=non_detailed_only_total_query, privacy_budget=budget_1
    )

    thresholds = config_json["thresholds_p"]
    non_detailed_only_total_answer = non_detailed_result.toPandas()
    non_detailed_only_total_answer["stat_level"] = np.digitize(
        non_detailed_only_total_answer["count"],
        thresholds[f"({region_type}, {iteration_level})"],
    )
    stat_level_df = spark.createDataFrame(
        non_detailed_only_total_answer[["POP_GROUP", "stat_level"]]
    )
    non_detailed_only_session.add_public_dataframe(
        f"stat_level_source_{region_identifier}_{iteration_level}", stat_level_df
    )

    # Second part of the adaptive query. Finding the records by stat-level.
    # Combine column to make groupby easier after filtering by stat-level
    records_by_stat_level = (
        QueryBuilder("non_detailed_only")
        .join_public(
            public_table=f"stat_level_source_{region_identifier}_{iteration_level}"
        )
        .select(["age23", "age9", "age4", "stat_level", "QSEX", "POP_GROUP"])
    )
    non_detailed_only_session.create_view(
        records_by_stat_level, "records_by_stat_level", cache=True
    )
    stat_level_sessions = non_detailed_only_session.partition_and_create(
        "records_by_stat_level",
        privacy_budget=budget_2,
        column="stat_level",
        splits={
            "stat_level_0": 0,
            "stat_level_1": 1,
            "stat_level_2": 2,
            "stat_level_3": 3,
        },
    )

    stat_level_queries = {}
    for age_column, stat_level in [("age23", 3), ("age9", 2), ("age4", 1)]:
        # Get the domain of the stat_level.
        stat_level_pop_groups = list(
            non_detailed_only_total_answer.loc[
                non_detailed_only_total_answer["stat_level"] == stat_level, "POP_GROUP",
            ]
        )
        # If there are no POP_GROUPS at this stat level, move on
        if not stat_level_pop_groups:
            continue

        # Create the query for this stat-level.
        stat_level_query = (
            QueryBuilder(f"stat_level_{stat_level}")
            .groupby(
                KeySet.from_dict(
                    {
                        "POP_GROUP": stat_level_pop_groups,
                        age_column: age_domains[age_column],
                        "QSEX": ["1", "2"],
                    }
                )
            )
            .count(name="COUNT", mechanism=noise_mechanism)
        )
        stat_level_queries[stat_level] = stat_level_query

    # Create the query for the stat-level not covered above.
    stat_level_0_pop_groups = list(
        non_detailed_only_total_answer.loc[
            non_detailed_only_total_answer["stat_level"] == 0, "POP_GROUP"
        ]
    )
    if stat_level_0_pop_groups:
        remaining_pop_groups_total = (
            QueryBuilder("stat_level_0")  # stat_level below all thresholds
            .groupby(KeySet.from_dict({"POP_GROUP": stat_level_0_pop_groups}))
            .count(name="COUNT", mechanism=noise_mechanism)
        )
        stat_level_queries[0] = remaining_pop_groups_total

    # Evaluate stat_level_queries.
    non_detailed_only_results = {}
    for k, stat_level_session in stat_level_sessions.items():
        m = re.match(r"stat_level_(\d)", k)
        if m is None:
            logger.warning(f"Unable to determine stat level of {k}")
            stat_level_session.stop()
            continue
        stat_level = int(m.group(1))
        if stat_level == 0 and not stat_level_0_pop_groups:
            stat_level_session.stop()
            continue
        query = stat_level_queries.get(stat_level)
        if query is None:
            stat_level_session.stop()
            continue
        result = (
            stat_level_session.evaluate(query_expr=query, privacy_budget=budget_2)
            .withColumn("REGION_TYPE", lit(region_type))
            .withColumn("REGION_ID", split(col("POP_GROUP"), ",").getItem(0))
            .withColumn("ITERATION_CODE", split(col("POP_GROUP"), ",").getItem(1))
            .drop("POP_GROUP")
        )
        if (
            "zero_suppression_chance" in config_json
            and stat_level == 0
            and region_type not in ("USA", "STATE", "PR-STATE")
        ):
            noise_info = (
                stat_level_session._noise_info(  # pylint: disable=protected-access
                    query_expr=query, privacy_budget=budget_2
                )
            )
            suppression_threshold = _get_suppression_threshold(
                noise_info=noise_info[0],
                zero_suppression_chance=config_json["zero_suppression_chance"],
            )

            logger.info(
                f"Thresholding T1 counts at {region_type}, iteration level "
                f"{iteration_level}, common/other iterations. Threshold: "
                f"{suppression_threshold}"
            )
            result = result.filter(col("COUNT") > lit(suppression_threshold))
        if not config_json["allow_negative_counts"]:
            result = result.withColumn(
                "COUNT", when(col("COUNT") < 0, 0).otherwise(col("COUNT"))
            )
        non_detailed_only_results[stat_level] = result
        stat_level_session.stop()

    # Age column postprocessing
    for stat_level in stat_level_queries:
        age_column = STAT_TO_AGE_COL[stat_level]
        if age_column == "*":
            non_detailed_only_results[stat_level] = (
                non_detailed_only_results[stat_level]
                .withColumn("QAGE", lit("*"))
                .withColumn("QSEX", lit("*"))
            )
        else:
            non_detailed_only_results[stat_level] = (
                non_detailed_only_results[stat_level]
                .withColumn(
                    age_column, get_age_low_high_udf(BINS[age_column])(col(age_column))
                )
                .withColumnRenamed(age_column, "QAGE")
            )
        result_sdfs.append(
            spark.createDataFrame(
                postprocess_dfs([non_detailed_only_results[stat_level].toPandas()]),
                T2_SCHEMA,
            )
        )
    non_detailed_only_session.delete_view("records_by_stat_level")
    return result_sdfs


def run_plan_p_analytics(
    parameters_path: str,
    data_path: str,
    output_path: str,
    overwrite_config: Optional[Dict] = None,
    should_validate_private_output: bool = False,
) -> None:
    """Entry point for SafeTab-P algorithm.

    First validates input files, and builds the expected domain of
    `person-records.txt` from files such as `GRF-C.txt`. See :mod:`.input_validation`
    for more details.

    .. warning::
        During validation, `person-records.txt` is checked against the expected domain,
        to make sure that the input files are consistent.

    Args:
        parameters_path: The location of the config and the race/ethnicity files.
        data_path: If csv reader, the location of input files.
            If cef reader, the file path to the reader config.
        output_path: The location to save t1 and t2.
        overwrite_config: Optional partial config that will overwrite any values for
            matching keys in the config that is read from config.json.
        should_validate_private_output: If True, validate private output after
            tabulations.
    """
    setup_input_config_dir()
    setup_safetab_p_output_config_dir()

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

    validate_config_values(config_json, "safetab-p", us_or_puerto_rico_values)

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
                execute_plan_p_analytics(
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
                    should_validate_private_output=should_validate_private_output,
                )


def get_age_low_high_udf(bins: List) -> Callable[[int], List[int]]:
    """UDF for returning low and high bounds of the age bin.

    Args:
        bins: The list of bins.
    """

    def get_age_low_high(age_bin: int, bins: List) -> List[int]:
        """Returns the low and high bounds of the age bin.

        Args:
            age_bin: The age bin we are finding the bounds of.
            bins: The list of bins.
                (eg. [18, 45, 65] is bins for [(0-18),(18-45),(45-65),(65-115)])
        """
        if age_bin == 0:
            low, high = 0, bins[age_bin]
        elif age_bin == len(bins):
            low, high = bins[age_bin - 1], 115
        else:
            low, high = bins[age_bin - 1], bins[age_bin]
        return [low, high]

    return udf(lambda age_bin: get_age_low_high(age_bin, bins), ArrayType(LongType()))


def _get_suppression_threshold(
    noise_info: Dict[str, Any], zero_suppression_chance: float
) -> float:
    """Returns the suppression threshold for a given query and privacy definition.

    Args:
        noise_info: The noise info for a measurement.
        zero_suppression_chance: The probability a zero-count statistic is removed by
        the threshold.
    """
    noise_scale = noise_info["noise_parameter"]
    # When noise is off, we don't threshold. We also allow
    # zero_suppression_chance == 0 to disable thresholding.
    if zero_suppression_chance == 0.0:
        return -float("inf")
    if noise_scale == 0:
        return 0
    return _inverse_cdf(noise_info, zero_suppression_chance)
