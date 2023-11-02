"""Functions for managing regions.

See `GRF-C.txt`, t1, t2, t3, and t4 in `Appendix A`
for more information.
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

import os
from typing import List, Union

from pyspark.sql.dataframe import DataFrame
from pyspark.sql.functions import (  # pylint: disable=no-name-in-module
    col,
    concat,
    lit,
    when,
)

from tmlt.common.configuration import Config
from tmlt.safetab_utils.csv_reader import CSVHReader, CSVPReader
from tmlt.safetab_utils.utils import STATE_FILTER_FLAG

try:
    from phsafe_safetab_reader.safetab_cef_reader import CEFHReader, CEFPReader
except ImportError:
    try:
        from phsafe_safetab_reader.safetab_cef_reader import CEFReader as CEFHReader
        from phsafe_safetab_reader.safetab_cef_reader import CEFReader as CEFPReader
    except ImportError:
        try:
            from tmlt.mock_cef_reader.safetab_cef_reader import (  # pylint: disable=ungrouped-imports
                CEFHReader,
                CEFPReader,
            )
        except ImportError:
            raise ImportError(
                "Failed to import CEFPReader & CEFHReader from module "
                "safetab_cef_reader. Please verify the safetab_cef_reader module has "
                "been placed in your python path."
            )

BLOCK_COLUMNS = ["TABBLKST", "TABBLKCOU", "TABTRACTCE", "TABBLK"]
"""Columns in `GRF-C.txt` that define blocks.

These columns are shared between `GRF-C.txt` and the person/household record files.
"""

REGION_TYPES = {
    "US": ["USA", "STATE", "COUNTY", "TRACT", "PLACE", "AIANNH"],
    "PR": ["PR-STATE", "PR-COUNTY", "PR-TRACT", "PR-PLACE"],
}
"""Region types for "US" and "PR"."""


REGION_GRANULARITY_MAPPING = {
    "USA": "detailed",
    "STATE": "detailed",
    "COUNTY": "coarse",
    "TRACT": "coarse",
    "PLACE": "coarse",
    "AIANNH": "coarse",
    "PR-STATE": "detailed",
    "PR-COUNTY": "coarse",
    "PR-TRACT": "coarse",
    "PR-PLACE": "coarse",
}
"""Mapping of region_type to the granularity required."""

NONE_AIANNH_CODE = "9999"
NONE_PLACE_CODE = "99999"
"""These AIANNH and Place codes correspond to a location that is not within an
AIANNH area of Place. We do not produce counts for these codes."""


def validate_state_filter_us(state_filter_us: List[str]) -> bool:
    """Returns True if state_filter_us is valid for a US run.

    Validates that the state_filter_us value has the following properties:
     * Is a list of strings
     * Contains no duplicate values
     * All elements are valid FIPS codes for the States and the
     District of Columbia
     * FIPS Codes for Outlying Areas of the United States and the
     Freely Associated States are considered invalid

    Args:
        state_filter_us: States to include in US run, as read in
            from the config file.

    Raises:
        ValueError: When the given state_filter_us is not valid.
    """
    VALID_STATE_CODE_NUMS = set(range(1, 57))
    VALID_STATE_CODE_NUMS -= {3, 7, 14, 43, 52}
    VALID_STATE_CODES = {f"{i:02d}" for i in VALID_STATE_CODE_NUMS}

    if not state_filter_us:
        raise ValueError(
            f"Invalid config: expected '{STATE_FILTER_FLAG}' to not be empty for US run"
        )

    if not isinstance(state_filter_us, list):
        raise ValueError(
            f"Invalid config: expected '{STATE_FILTER_FLAG}' to have type list, not"
            f" {type(state_filter_us).__name__}"
        )

    bad_types = {type(e) for e in state_filter_us if not isinstance(e, str)}
    if bad_types:
        bad_types_str = "{" + ", ".join(t.__name__ for t in bad_types) + "}"
        raise ValueError(
            f"Invalid config: expected '{STATE_FILTER_FLAG}' elements to have type str,"
            f" not {bad_types_str}"
        )

    if len(state_filter_us) != len(set(state_filter_us)):
        raise ValueError(
            f"Invalid config: '{STATE_FILTER_FLAG}' list contains duplicate values"
        )

    bad_state_codes = sorted(set(state_filter_us) - VALID_STATE_CODES)
    if bad_state_codes:
        bad_state_codes_str = "{" + ", ".join(bad_state_codes) + "}"
        raise ValueError(
            f"Invalid config: '{STATE_FILTER_FLAG}' contains invalid codes:"
            f" {bad_state_codes_str}"
        )
    return True


def preprocess_geography_df(
    input_reader: Union[CSVHReader, CSVPReader, CEFHReader, CEFPReader],
    us_or_puerto_rico: str,
    input_config_dir_path: str,
) -> DataFrame:
    """Return a df with a column for each region type.

    Each column of the dataframe corresponds to a region type, and contains a list
    of combined geography codes that uniquely identify a geographic entity at that
    geography level.

    Args:
        input_reader: The reader to read the grfc file.
        us_or_puerto_rico: Whether to return the geography_df for the
            50 states + DC ("US") or Puerto Rico ("PR"). The returned dataframe contains
            columns for the :data:`BLOCK_COLUMNS` and the appropriate
            :data:`REGION_TYPES`.
        input_config_dir_path: The path containing input config files. Must contain
            `GRF-C.json`.
    """
    usecols = BLOCK_COLUMNS + ["PLACEFP", "AIANNHCE"]
    grfc_df = input_reader.get_geo_df().select(usecols)

    # Make sure that we don't use any columns that we don't validate. See #311.
    validated_columns = Config.load_json(
        os.path.join(input_config_dir_path, "GRF-C.json")
    ).columns
    assert sorted(usecols) == sorted(validated_columns)

    if us_or_puerto_rico == "PR":
        grfc_df = grfc_df.withColumn("PR-STATE", lit("72"))
        grfc_df = grfc_df.withColumn("PR-COUNTY", concat(lit("72"), col("TABBLKCOU")))
        grfc_df = grfc_df.withColumn(
            "PR-TRACT", concat(lit("72"), col("TABBLKCOU"), col("TABTRACTCE"))
        )
        grfc_df = grfc_df.withColumn(
            "PR-PLACE",
            when(col("PLACEFP") == NONE_PLACE_CODE, "NULL").otherwise(
                concat(lit("72"), col("PLACEFP"))
            ),
        )
    else:
        grfc_df = grfc_df.withColumn("USA", lit("1"))
        grfc_df = grfc_df.withColumn("STATE", col("TABBLKST"))
        grfc_df = grfc_df.withColumn(
            "COUNTY", concat(col("TABBLKST"), col("TABBLKCOU"))
        )
        grfc_df = grfc_df.withColumn(
            "TRACT", concat(col("TABBLKST"), col("TABBLKCOU"), col("TABTRACTCE"))
        )
        grfc_df = grfc_df.withColumn(
            "PLACE",
            when(col("PLACEFP") == NONE_PLACE_CODE, "NULL").otherwise(
                concat(col("TABBLKST"), col("PLACEFP"))
            ),
        )

        grfc_df = grfc_df.withColumn(
            "AIANNH",
            when(col("AIANNHCE") == NONE_AIANNH_CODE, "NULL").otherwise(
                col("AIANNHCE")
            ),
        )
    return grfc_df.select(BLOCK_COLUMNS + REGION_TYPES[us_or_puerto_rico])
