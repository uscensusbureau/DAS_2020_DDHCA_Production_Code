"""Validates output files of SafeTab-P/-H.

Also creates updated versions of output config files which are used for further
validation.

See `Appendix A` for a description of each file.
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

# pylint: disable=no-name-in-module

import logging
from typing import Mapping, Sequence

from pyspark.sql.dataframe import DataFrame
from pyspark.sql.functions import col

from tmlt.common.configuration import Config
from tmlt.common.validation import validate_spark_df
from tmlt.safetab_utils.validation import check_pop_group_for_invalid_states


def validate_output(
    output_sdfs: Mapping[str, DataFrame],
    expected_output_configs: Mapping[str, Config],
    state_filter: Sequence[str],
    allow_negative_counts_flag: bool,
) -> bool:
    """Check whether all outputs from algorithm execution are as expected.

    Args:
        output_sdfs: A map where keys are output subdirectory names, and values are
            data to be written to that subdirectory.
        expected_output_configs: A set of configs describing all expected output files.
            Keys should be the expected output subdirectory, and values should be the
            config.
        state_filter: List of states included.
        allow_negative_counts_flag: If True, outputs can can negative values in
            COUNT column.
    """
    logger = logging.getLogger(__name__)
    logger.info("Validating outputs ...")

    logger.info("Starting output validation...")
    logger.info("Checking that required output folders are present in output path...")

    expected_output_subdirs = expected_output_configs.keys()

    actual_output_subdirs = list(
        {tabulation for tabulation in output_sdfs if output_sdfs[tabulation]}
    )

    if len(actual_output_subdirs) > len(set(expected_output_subdirs)):
        extra_subdirs = sorted(
            set(actual_output_subdirs) - set(expected_output_subdirs)
        )
        extra_subdirs_str = "{" + ", ".join(extra_subdirs) + "}"
        logger.error(
            f"Invalid output: Additional output folders present {extra_subdirs_str}."
        )
        return False

    missing_subdirs = sorted(set(expected_output_subdirs) - set(actual_output_subdirs))

    if missing_subdirs:
        missing_subdirs_str = "{" + ", ".join(missing_subdirs) + "}"
        logger.error(
            f"Invalid output: missing required output folders {missing_subdirs_str}."
        )
        return False
    logger.info("All required output folders present.")

    logger.info(
        "Outputs are checked for expected formats as per Appendix A output spec..."
    )
    okay = True
    for output_file in actual_output_subdirs:
        okay &= validate_spark_df(
            output_file,
            output_sdfs[output_file],
            expected_output_configs[output_file],
            unexpected_column_strategy="error",
            check_column_order=False,
        )
    if not okay:
        logger.error("Invalid output: Not as per expected format. See above.")
        return False
    logger.info("All generated outputs are as per prior expectation.")

    if state_filter is not None:
        logger.info(
            "Outputs are checked to ensure appropriate states are tabulated as per the "
            "config state_filter_us and run_pr flags..."
        )
        okay = True
        for output_file in actual_output_subdirs:
            if len(output_sdfs[output_file].head(1)) > 0:
                okay &= check_pop_group_for_invalid_states(
                    output_sdfs[output_file], state_filter
                )

        if not okay:
            logger.error("Invalid output: Output does not have state filters applied.")
            return False

    logger.info(
        "Outputs are checked to ensure SafeTab applies non-negative postprocessing"
        " as per config allow_negative_counts flag..."
    )
    okay = True
    for output_file in actual_output_subdirs:
        # If allow_negative_counts_flag=False, then check it does not contain negatives.
        # If it is True, it may or may not contain negative counts.
        if len(output_sdfs[output_file].head(1)) > 0:
            if not allow_negative_counts_flag:
                contains_negatives = not (
                    output_sdfs[output_file].filter(col("COUNT") < 0).rdd.isEmpty()
                )
                if contains_negatives:
                    okay &= False
    if not okay:
        logger.error(
            "Invalid output: Negative counts are not allowed, but were found in the"
            " output."
        )
        return False

    logger.info("Output validation successful. All output files are as expected.")
    return True
