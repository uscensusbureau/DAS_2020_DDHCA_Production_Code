"""Validates input files to SafeTab.

Also creates updated versions of input file schemas which are used for further
validation and by the main SafeTab algorithm.

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
import os
from typing import List, Optional, Union

from pyspark.sql.dataframe import DataFrame
from pyspark.sql.functions import concat

from tmlt.common.configuration import Config
from tmlt.common.io_helpers import read_csv
from tmlt.common.validation import update_config, validate_directory, validate_spark_df
from tmlt.safetab_utils.characteristic_iterations import (
    IterationFilter,
    IterationManager,
)
from tmlt.safetab_utils.csv_reader import (
    GEO_FILENAME,
    PERSON_FILENAME,
    POP_GROUP_TOTAL_FILENAME,
    UNIT_FILENAME,
    CSVHReader,
    CSVPReader,
)
from tmlt.safetab_utils.paths import INPUT_FILES_SAFETAB_H, INPUT_FILES_SAFETAB_P
from tmlt.safetab_utils.validation import check_pop_group_for_invalid_states

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

PERSON_NAME = "person-records"
"""Name of the person records file for logging purposes."""

UNIT_NAME = "household-records"
"""Name of the household records file for logging purposes."""

GEO_NAME = "GRF-C"
"""Name of the geography records file for logging purposes."""

POP_GROUP_NAME = "pop-group-counts"
"""Name of the population group counts file for logging purposes."""

RACE_COLUMNS = ["QRACE2", "QRACE3", "QRACE4", "QRACE5", "QRACE6", "QRACE7", "QRACE8"]
"""The names of the columns in the persons table that contain race codes."""


def _get_iteration_code_domain(parameters_path: str) -> List[str]:
    """Return the domain of characteristic iterations.

    Args:
        parameters_path: The path containing the input files.
    """
    filenames = [
        "race-characteristic-iterations.txt",
        "ethnicity-characteristic-iterations.txt",
    ]
    iteration_code_domain: List[str] = []
    for filename in filenames:
        filename = os.path.join(parameters_path, filename)
        iteration_df = read_csv(filename, sep="|", dtype=str)
        iteration_code_domain.extend(iteration_df["ITERATION_CODE"])
    return iteration_code_domain


def _get_race_eth_code_domain(parameters_path: str) -> List[str]:
    """Return the domain of race and ethnicity codes.

    Args:
        parameters_path: The path containing the config and Race/Ethnicity files.
    """
    filename = os.path.join(parameters_path, "race-and-ethnicity-codes.txt")
    race_code_df = read_csv(filename, sep="|", dtype=str)
    race_eth_code_domain = list(race_code_df["RACE_ETH_CODE"])
    return race_eth_code_domain


def _check_race_and_ethnicity_iteration_codes_are_disjoint(
    parameters_path: str,
) -> bool:
    """Log an error and return False if race and ethnicity iteration codes overlap.

    Args:
        parameters_path: The path containing the config and Race/Ethnicity files.
    """
    okay = True

    race_filename = os.path.join(parameters_path, "race-characteristic-iterations.txt")
    race_iteration_df = read_csv(race_filename, sep="|", dtype=str)
    race_iteration_domain = set(race_iteration_df["ITERATION_CODE"])

    ethnicity_filename = os.path.join(
        parameters_path, "ethnicity-characteristic-iterations.txt"
    )
    ethnicity_iteration_df = read_csv(ethnicity_filename, sep="|", dtype=str)
    ethnicity_iteration_domain = set(ethnicity_iteration_df["ITERATION_CODE"])

    domain_overlap = sorted(
        race_iteration_domain.intersection(ethnicity_iteration_domain)
    )
    if len(domain_overlap) > 0:
        logging.getLogger(__name__).error(
            "%s appeared as race iteration codes and as ethnicity iteration codes."
            " Expected race and ethnicity iteration codes to be disjoint.",
            domain_overlap,
        )
        okay = False
    return okay


def _check_for_race_codes_after_first_null(sdf: DataFrame) -> bool:
    """Log an error and return False if there are race codes after a 'Null'.

    Args:
        sdf: The input dataframe.
    """
    okay = True
    invalid_pattern = r"^.*Null.*\d\d\d\d.*$"
    sdf_combined_col = sdf.withColumn("combined_race_columns", concat(*RACE_COLUMNS))
    num_invalid = sdf_combined_col.filter(
        sdf_combined_col["combined_race_columns"].rlike(invalid_pattern)
    ).count()
    if num_invalid:
        logging.getLogger(__name__).error(
            "%d records had race codes after a 'Null'.", num_invalid
        )
        okay = False
    return okay


def _check_for_invalid_iteration_metadata(parameters_path: str) -> bool:
    """Return False if any iterations have invalid metadata.

    Rules checked:

    * not (DETAILED_ONLY == 'True' and COARSE_ONLY == 'True')
    * LEVEL == 0 <==> DETAILED_ONLY == 'Null' <==> COARSE_ONLY == 'Null'

    See `race-characteristic-iterations.txt` for more information.

    Args:
        parameters_path: Directory containing the config and Race/Ethnicity files.
    """
    logger = logging.getLogger(__name__)
    filenames = [
        "race-characteristic-iterations.txt",
        "ethnicity-characteristic-iterations.txt",
    ]
    okay = True
    for filename in filenames:
        filename = os.path.join(parameters_path, filename)
        iteration_df = read_csv(filename, sep="|", dtype=str)

        num_invalid = sum(
            (iteration_df["DETAILED_ONLY"] == "True")
            & (iteration_df["COARSE_ONLY"] == "True")
        )
        if num_invalid:
            logger.error(
                "%d of %d iterations had DETAILED_ONLY and COARSE_ONLY as 'True' in %s",
                num_invalid,
                len(iteration_df),
                filename,
            )
            okay = False

        detailed_null_mask = iteration_df["DETAILED_ONLY"] == "Null"
        coarse_null_mask = iteration_df["COARSE_ONLY"] == "Null"
        level_0_mask = iteration_df["LEVEL"] == "0"
        num_invalid = len(iteration_df) - sum(
            (detailed_null_mask == level_0_mask) & (coarse_null_mask == level_0_mask)
        )
        if num_invalid:
            logger.error(
                "%d of %d iterations did not have matching LEVEL == '0', "
                "DETAILED_ONLY == 'Null', and COARSE_ONLY == 'Null' in %s",
                num_invalid,
                len(iteration_df),
                filename,
            )
            okay = False
    return okay


def _check_iteration_hierarchy_height(parameters_path: str) -> bool:
    """Checks that the mapping of race and ethnicity codes to iterations makes sense.

    In particular, each case code can only be mapped to one iteration for each
    combination of:
    - Level (1 or 2)
    - Aloneness ("Alone", or "Alone or in combination")
    - Type. Either ("Common"), or ("Detailed" or "Other)

    For example, a particular race code could map to (at maximum):
    - A level 1, alone, coarse iteration.
    - A level 1, alone, detailed iteration.
    - A level 1, alone or in combination, coarse iteration.
    - A level 1, alone or in combination, detailed iteration.
    - A level 2, alone, coarse iteration.
    - A level 2, alone, detailed iteration.
    - A level 2, alone or in combination, coarse iteration.
    - A level 2, alone or in combination, detailed iteration.

    An ethnicity code (which can only be alone) can only map to:
    - A level 1, coarse iteration.
    - A level 1, detailed iteration.
    - A level 2, coarse iteration.
    - A level 2, detailed iteration.

    Mapping to two iterations of any of the above categories would be invalid,
    however. It would also be invalid for a code to map to a common iteration
    as well as a detailed or coarse iteration with all other properties the same.

    Args:
        parameters_path: Directory containing the config and Race/Ethnicity files.
    """
    okay = True
    # max race codes not used, so is set to a dummy value
    iteration_manager = IterationManager(parameters_path, max_race_codes=0)
    # pylint: disable=protected-access
    for race_eth in ["race", "ethnicity"]:
        max_level = (
            iteration_manager._race_iterations_df["LEVEL"].astype(int).max()
            if race_eth == "race"
            else iteration_manager._ethnicity_iterations_df["LEVEL"].astype(int).max()
        )

        if race_eth == "race":
            combinations = [
                {
                    "race_eth_type": "race",
                    "level": f"{level}",
                    "alone": alone,
                    "detailed_only": detailed_only,
                    "coarse_only": coarse_only,
                }
                for level in range(1, max_level + 1)
                for alone in ("True", "False")
                for detailed_only, coarse_only in [("False", "both"), ("both", "False")]
            ]
        else:
            assert race_eth == "ethnicity"
            combinations = [
                {
                    "race_eth_type": "ethnicity",
                    "level": f"{level}",
                    "detailed_only": detailed_only,
                    "coarse_only": coarse_only,
                }
                for level in range(1, max_level + 1)
                for detailed_only, coarse_only in [("False", "both"), ("both", "False")]
            ]
        for combo in combinations:
            bad_codes = {}
            for (
                race_code,
                iteration_codes,
            ) in iteration_manager.get_race_eth_code_to_iterations(
                IterationFilter(**combo)
            )[
                1
            ].items():
                if len(iteration_codes) > 1:
                    bad_codes[race_code] = iteration_codes
            if bad_codes:
                okay = False
                tabulation_level_message = (
                    "national and state"
                    if combo["detailed_only"] == "both"
                    else "sub-state"
                )
                alone_message = (
                    f" and ALONE={combo['alone']}" if "alone" in combo else ""
                )
                error_message = (
                    f"The following {combo['race_eth_type']} codes are mapped to"
                    " multiple iteration codes that are all labeled as"
                    f" LEVEL={combo['level']}{alone_message}, and are all tabulated at"
                    f" the {tabulation_level_message} level. SafeTab expects each code"
                    " to only be mapped to one such iteration code.\n\nThe list of"
                    f" errors is formatted as [{combo['race_eth_type']} code]:"
                    " [iteration codes].\n"
                )
                error_list = ""
                for race_code, iteration_codes in bad_codes.items():
                    error_list += (
                        race_code + ": " + ", ".join(sorted(iteration_codes)) + "\n"
                    )
                logging.getLogger(__name__).error(error_message + error_list)
    return okay


def _check_t1_pop_groups_are_unique(pop_group_sdf: DataFrame) -> bool:
    """Check that no pop groups appear multiple times in the T1 input.

    Args:
        pop_group_sdf: Spark DataFrame containing the T1 Counts.
    """
    okay = True
    pop_group_counts = pop_group_sdf.groupby(
        ["REGION_TYPE", "REGION_ID", "ITERATION_CODE"]
    ).count()
    pop_group_counts = pop_group_counts.filter(pop_group_counts["count"] > 1)
    if pop_group_counts.count() > 0:
        okay = False
        bad_pop_groups = pop_group_counts.select(
            "REGION_TYPE", "REGION_ID", "ITERATION_CODE"
        ).collect()
        bad_pop_groups = sorted(
            bad_pop_groups,
            key=lambda pop_group: (
                pop_group.REGION_TYPE,
                pop_group.REGION_ID,
                pop_group.ITERATION_CODE,
            ),
        )
        bad_pop_groups_str = ", ".join(
            f"(REGION_TYPE={pop_group.REGION_TYPE}, "
            f"REGION_ID={pop_group.REGION_ID}, "
            f"ITERATION_CODE={pop_group.ITERATION_CODE})"
            for pop_group in bad_pop_groups
        )
        logging.getLogger(__name__).error(
            "The following population groups appeared more than once in "
            f"{POP_GROUP_NAME}: [{bad_pop_groups_str}]"
        )
    return okay


def update_configs(
    parameters_path: str,
    input_data_configs_path: str,
    output_path: str,
    records_filename: str,
    state_filter: List[str],
):
    """Create new complete configuration files from input files.

    A new config is created for each file in :data:`INPUT_FILES`, now using the
    full domain information from files such as
    `race-characteristic-iterations.txt`.

    Args:
        parameters_path: Directory containing the config and Race/Ethnicity files.
        input_data_configs_path: Directory containing input data configs that specify
        the expected formats for the input files. Expects a config with the same name,
        but a .json extension for each input csv.
        output_path: Directory to store new config files.
        records_filename: Name of the person/household records file.
        state_filter: list of state codes in input df
    """
    iteration_code_domain = _get_iteration_code_domain(parameters_path)
    race_eth_code_domain = _get_race_eth_code_domain(parameters_path)

    update_config(
        input_data_configs_path=input_data_configs_path,
        output_path=output_path,
        file_root="GRF-C",
        attribute_to_domain_dict={"TABBLKST": state_filter},
    )
    update_config(
        input_data_configs_path=input_data_configs_path,
        output_path=output_path,
        file_root="ethnicity-characteristic-iterations",
        attribute_to_domain_dict={"ITERATION_CODE": iteration_code_domain},
    )
    update_config(
        input_data_configs_path=input_data_configs_path,
        output_path=output_path,
        file_root=records_filename,
        attribute_to_domain_dict={
            "TABBLKST": state_filter,
            "QRACE1": race_eth_code_domain,
            "QRACE2": ["Null"] + race_eth_code_domain,
            "QRACE3": ["Null"] + race_eth_code_domain,
            "QRACE4": ["Null"] + race_eth_code_domain,
            "QRACE5": ["Null"] + race_eth_code_domain,
            "QRACE6": ["Null"] + race_eth_code_domain,
            "QRACE7": ["Null"] + race_eth_code_domain,
            "QRACE8": ["Null"] + race_eth_code_domain,
            "QSPAN": race_eth_code_domain,
        },
    )
    if os.path.exists(os.path.join(input_data_configs_path, "pop-group-totals.json")):
        update_config(
            input_data_configs_path=input_data_configs_path,
            output_path=output_path,
            file_root="pop-group-totals",
            # No validation logic was updated, due to a multi-column constraint.
            # See check_pop_group_for_invalid_states for filter logic.
            attribute_to_domain_dict={},
        )
    update_config(
        input_data_configs_path=input_data_configs_path,
        output_path=output_path,
        file_root="race-and-ethnicity-code-to-iteration",
        attribute_to_domain_dict={
            "ITERATION_CODE": iteration_code_domain,
            "RACE_ETH_CODE": race_eth_code_domain,
        },
    )
    update_config(
        input_data_configs_path=input_data_configs_path,
        output_path=output_path,
        file_root="race-and-ethnicity-codes",
        attribute_to_domain_dict={"RACE_ETH_CODE": race_eth_code_domain},
    )
    update_config(
        input_data_configs_path=input_data_configs_path,
        output_path=output_path,
        file_root="race-characteristic-iterations",
        attribute_to_domain_dict={"ITERATION_CODE": iteration_code_domain},
    )
    logger = logging.getLogger(__name__)
    logger.info("Updated configs can be found at %s.", output_path)


def validate_input(
    input_reader: Union[CSVPReader, CSVHReader, CEFPReader, CEFHReader],
    parameters_path: str,
    input_data_configs_path: str,
    output_path: str,
    program: str,
    state_filter: List[str],
) -> bool:
    """Return whether all input files are consistent and as expected.

    Validates using a three step process

    1. Uses our prior knowledge to validate the input files using prebuilt
       schemas. For some columns we know the full domain (for instance QAGE and
       QSEX), for other columns we only know the expected format. Finally, some
       columns that are not used are ignored entirely (for instance in the
       `GRF-C.txt`).
    2. Creates new schemas updated with domain information contained in the
       input files.
    3. Validate again using updated schemas.

    Args:
        input_reader: The reader that reads and filters the input files.
        parameters_path: Directory containing the input files.
        input_data_configs_path: Directory containing config files specifying the
            expected formats for the input files. Expects a config with the
            same name, but a .json extension for each input csv.
        output_path: Location to save the updated schemas.
        program: Allowed options are 'safetab-p' and 'safetab-h'.
        state_filter: list of state codes in input df.
    """
    try:
        os.mkdir(output_path)
    except FileExistsError:
        pass
    logger = logging.getLogger(__name__)

    if program not in {"safetab-p", "safetab-h"}:
        raise ValueError(f"program must be 'safetab-p' or 'safetab-h', not {program}")

    # Load input dataframes and their config files
    person_sdf: Optional[DataFrame] = None
    person_config: Optional[Config] = None
    unit_sdf: Optional[DataFrame] = None
    unit_config: Optional[Config] = None
    pop_group_sdf: Optional[DataFrame] = None
    pop_group_config: Optional[Config] = None
    if program == "safetab-p":
        # help out mypy
        assert isinstance(input_reader, (CSVPReader, CEFPReader))
        person_sdf = input_reader.get_person_df()
        person_config = Config.load_json(
            os.path.join(input_data_configs_path, "person-records.json")
        )
    else:
        # help out mypy
        assert isinstance(input_reader, (CSVHReader, CEFHReader))
        unit_sdf = input_reader.get_unit_df()
        unit_config = Config.load_json(
            os.path.join(input_data_configs_path, "household-records.json")
        )
        pop_group_sdf = input_reader.get_pop_group_details_df()
        pop_group_config = Config.load_json(
            os.path.join(input_data_configs_path, "pop-group-totals.json")
        )
    geo_sdf = input_reader.get_geo_df()
    geo_config = Config.load_json(os.path.join(input_data_configs_path, "GRF-C.json"))

    logger.info(
        "Starting phase 1 of input validation. Checking files "
        "against prior expectations..."
    )
    INPUT_FILES = (
        INPUT_FILES_SAFETAB_P if program == "safetab-p" else INPUT_FILES_SAFETAB_H
    )
    # Check input files other than person, household, grfc.
    okay = validate_directory(
        parameters_path,
        input_data_configs_path,
        relative_filenames=[
            input_file
            for input_file in INPUT_FILES
            if input_file
            not in [
                GEO_FILENAME,
                PERSON_FILENAME,
                UNIT_FILENAME,
                POP_GROUP_TOTAL_FILENAME,
            ]
        ],
        delimiter="|",
        extension="txt",
        unexpected_column_strategy="error",
        check_column_order=True,
    )
    # Check input dataframes
    okay &= (
        validate_spark_df(
            PERSON_NAME,
            person_sdf,
            person_config,
            unexpected_column_strategy="error",
            check_column_order=True,
            allow_empty=False,
        )
        if person_sdf is not None and person_config is not None
        else okay
    )
    okay &= (
        validate_spark_df(
            UNIT_NAME,
            unit_sdf,
            unit_config,
            unexpected_column_strategy="error",
            check_column_order=True,
            allow_empty=False,
        )
        if unit_sdf is not None and unit_config is not None
        else okay
    )
    okay &= (
        validate_spark_df(
            POP_GROUP_NAME,
            pop_group_sdf,
            pop_group_config,
            unexpected_column_strategy="error",
            check_column_order=True,
            allow_empty=False,
        )
        if pop_group_sdf is not None and pop_group_config is not None
        else okay
    )
    # Notice that the GRF-C uses "ignore" instead of "error" for unexpected columns.
    # Unlike other files, the GRF-C contains columns that aren't validated or used.
    okay &= validate_spark_df(
        GEO_NAME,
        geo_sdf,
        geo_config,
        unexpected_column_strategy="ignore",
        check_column_order=True,
        allow_empty=False,
    )
    okay &= (
        _check_for_race_codes_after_first_null(person_sdf)
        if person_sdf is not None
        else okay
    )
    okay &= (
        _check_for_race_codes_after_first_null(unit_sdf)
        if unit_sdf is not None
        else okay
    )
    okay &= _check_iteration_hierarchy_height(parameters_path)
    okay &= _check_for_invalid_iteration_metadata(parameters_path)
    okay &= _check_race_and_ethnicity_iteration_codes_are_disjoint(parameters_path)
    okay &= (
        _check_t1_pop_groups_are_unique(pop_group_sdf)
        if pop_group_sdf is not None
        else okay
    )
    if not okay:
        logger.error("Errors found in phase 1. See above.")
        return False
    logger.info("Phase 1 successful.")

    logger.info(
        "Starting phase 2 of input validation. Creating new updated "
        "schemas based on input files..."
    )
    update_configs(
        parameters_path=parameters_path,
        input_data_configs_path=input_data_configs_path,
        output_path=output_path,
        records_filename=PERSON_NAME if person_sdf is not None else UNIT_NAME,
        state_filter=state_filter,
    )
    logger.info("Phase 2 successful.")

    logger.info(
        "Starting phase 3 of input validation. Checking files "
        "against updated schemas..."
    )
    # Check input files other than person, household, grfc.
    okay = validate_directory(
        parameters_path,
        output_path,
        relative_filenames=[
            input_file
            for input_file in INPUT_FILES
            if input_file
            not in [
                GEO_FILENAME,
                PERSON_FILENAME,
                UNIT_FILENAME,
                POP_GROUP_TOTAL_FILENAME,
            ]
        ],
        delimiter="|",
        extension="txt",
        unexpected_column_strategy="ignore",
        check_column_order=True,
    )
    # Check input dataframes
    updated_person_config: Optional[Config] = None
    updated_unit_config: Optional[Config] = None
    updated_pop_group_config: Optional[Config] = None
    if program == "safetab-p":
        updated_person_config = Config.load_json(
            os.path.join(output_path, "person-records.json")
        )
    else:
        updated_unit_config = Config.load_json(
            os.path.join(output_path, "household-records.json")
        )
        updated_pop_group_config = Config.load_json(
            os.path.join(output_path, "pop-group-totals.json")
        )
    updated_geo_config = Config.load_json(os.path.join(output_path, "GRF-C.json"))
    okay &= (
        validate_spark_df(
            PERSON_NAME,
            person_sdf,
            updated_person_config,
            unexpected_column_strategy="ignore",
            check_column_order=True,
        )
        if person_sdf is not None and updated_person_config is not None
        else okay
    )
    okay &= (
        validate_spark_df(
            UNIT_NAME,
            unit_sdf,
            updated_unit_config,
            unexpected_column_strategy="ignore",
            check_column_order=True,
        )
        if unit_sdf is not None and updated_unit_config is not None
        else okay
    )
    okay &= (
        validate_spark_df(
            POP_GROUP_NAME,
            pop_group_sdf,
            updated_pop_group_config,
            unexpected_column_strategy="ignore",
            check_column_order=True,
        )
        if pop_group_sdf is not None and updated_pop_group_config is not None
        else okay
    )
    okay &= (
        check_pop_group_for_invalid_states(pop_group_sdf, state_filter)
        if pop_group_sdf is not None
        else okay
    )
    okay &= validate_spark_df(
        GEO_NAME,
        geo_sdf,
        updated_geo_config,
        unexpected_column_strategy="ignore",
        check_column_order=True,
    )
    if not okay:
        logger.error("Errors found in phase 3. See above.")
        return False
    logger.info("Phase 3 successful. All files are as expected.")
    return True
