"""Validate against a serialized :class:`~tmlt.common.configuration.Config`.

Also contains utilities for updating config files, which is often done after
initial validation.
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
import os
from typing import Callable, Dict, List, Optional, Sequence, Type, Union

import pandas as pd
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import BooleanType

from tmlt.common.configuration import (
    Attribute,
    Categorical,
    CategoricalStr,
    Config,
    Unrestricted,
)
from tmlt.common.io_helpers import read_csv
from tmlt.common.schema import Schema

MAX_INVALID_VALUES_LOGGED = 1000
"""The maximum number of invalid values to log in :func:`validate_spark_df`."""


def validate_file(
    input_filename: str,
    config: Config,
    delimiter: str = ",",
    unexpected_column_strategy: str = "error",
    check_column_order: bool = False,
) -> bool:
    """Validates whether the input file conforms to the config.

    Args:
        input_filename: Name of the file to validate.
        config: A Config object.
        delimiter: The delimiter used to separate tokens in the input file.
        unexpected_column_strategy: How to handle columns that are  not in
            the config. Allowed options are 'ignore', 'warn', and 'error'.
            Defaults to 'error'.
        check_column_order: Whether to check that the order of the columns in the csv
            file match the order of the columns in the config.
    """
    dtypes = {attribute.column: attribute.dtype for attribute in config}
    try:
        found_columns = list(
            read_csv(input_filename, delimiter=delimiter, dtype=dtypes, nrows=1).columns
        )
    except (IOError, ValueError) as e:
        logging.exception(e)
        return False

    # Create schema with all columns for Spark.
    all_column_config = Config(
        [
            config[column] if column in config.columns else Unrestricted(column)
            for column in found_columns
        ]
    )
    all_column_schema = Schema.from_config_object(all_column_config)

    # Load data.
    logging.info("Loading %s...", input_filename)
    spark = SparkSession.builder.getOrCreate()
    sdf = spark.read.csv(
        input_filename,
        schema=all_column_schema.spark_schema,
        mode="FAILFAST",
        enforceSchema=False,
        sep=delimiter,
        header=True,
    )
    return validate_spark_df(
        sdf_name=input_filename,
        sdf=sdf,
        config=config,
        unexpected_column_strategy=unexpected_column_strategy,
        check_column_order=check_column_order,
    )


def validate_spark_df(
    sdf_name: str,
    sdf: DataFrame,
    config: Config,
    unexpected_column_strategy: str = "error",
    check_column_order: bool = False,
    found_columns: Optional[Sequence] = None,
    allow_empty: bool = True,
) -> bool:
    """Validates whether the input dataframe conforms to the config.

    Args:
        sdf_name: A unique name or filename to identify the sdf for logging purposes.
        sdf: A Spark DataFrame.
        config: A config object for the sdf.
        unexpected_column_strategy: How to handle columns that are  not in
            the config. Allowed options are 'ignore', 'warn', and 'error'.
            Defaults to 'error'.
        check_column_order: Whether to check that the order of the columns in the csv
            file match the order of the columns in the config.
        found_columns: Optional. Columns found in the csv for the sdf.
            This parameter is used when this function is called in
            :func:`validate_file`.
        allow_empty: Whether to throw a validation error on an empty dataframe.
            Defaults to True (allow empty dataframes).
    """
    actual_columns: Sequence  # appease mypy
    if found_columns:
        actual_columns = found_columns
    else:
        actual_columns = sdf.columns

    # Check whether the dataframe is empty.
    if (not allow_empty) and (sdf.first() is None):
        logging.error(
            f"{sdf_name} is empty. This fails validation as records are expected."
        )
        logging.error(
            """If the input dataset is not empty, there may be a problem with how it is
            being read, or an intervening filter step may be removing all of the
            records."""
        )
        return False

    # Handle missing columns
    if not all(expected_column in actual_columns for expected_column in config.columns):
        message = (
            "Missing columns."
            f" Expected columns: {config.columns}."
            f" Actual columns: {actual_columns}."
        )
        logging.error(message)
        logging.error("%s failed schema validation.", sdf_name)
        return False

    sdf = sdf.select(*config.columns)

    def create_check_column(attribute: Attribute) -> Callable[[pd.Series], pd.Series]:
        """Return a udf for validating pd.Series.

        Args:
            attribute: An attribute that can validate pd.Series.
        """

        @pandas_udf(BooleanType())
        def check_column(values: pd.Series) -> pd.Series:
            """Return whether all values are valid.

            Args:
                values: The pd.Series to validate against attribute.
            """
            results = pd.Series([False] * len(values), index=values.index)
            okay_index = attribute.validate(
                values, out_of_bounds_strategy="remove"
            ).index
            results[okay_index] = True
            return results

        return check_column

    found_invalid_values = False
    for attribute in config:
        column = attribute.column
        check_column = create_check_column(attribute)
        invalid_sdf = sdf.filter(~check_column(column))
        invalid_values = sorted(
            [
                row[column]
                for row in invalid_sdf.select(column)
                .distinct()
                .take(MAX_INVALID_VALUES_LOGGED)
            ]
        )
        if invalid_values:
            message = f"Invalid values found in {column}: {invalid_values}"
            if len(invalid_values) == MAX_INVALID_VALUES_LOGGED:
                message += (
                    f" (only the first {MAX_INVALID_VALUES_LOGGED} values are shown)"
                )
            logging.error(message)
            found_invalid_values = True
    if found_invalid_values:
        logging.error("%s failed schema validation.", sdf_name)
        return False

    # Handle unexpected columns.
    if unexpected_column_strategy not in {"ignore", "warn", "error"}:
        raise ValueError(
            "unexpected_column_strategy must be 'ignore', 'warn', or 'error', "
            f"not {unexpected_column_strategy}"
        )
    if unexpected_column_strategy != "ignore":
        unexpected_columns = set(actual_columns) - set(config.columns)
        if unexpected_columns:
            message = f"Unexpected columns found: {sorted(unexpected_columns)}"
            if unexpected_column_strategy == "warn":
                logging.warning(message)
            else:
                assert unexpected_column_strategy == "error"
                logging.error(message)
                logging.error("%s failed schema validation.", sdf_name)
                return False

    if check_column_order:
        expected_column_order = [attribute.column for attribute in config]
        found_column_order: List[str] = []
        for column in actual_columns:
            if column in expected_column_order:
                found_column_order.append(column)

        if found_column_order != expected_column_order:
            message = (
                "Columns are out of order."
                f" Expected order: {expected_column_order}."
                f" Actual order: {found_column_order}."
            )
            logging.error(message)
            logging.error("%s failed schema validation.", sdf_name)
            return False

    logging.info("%s passed schema validation.", sdf_name)
    return True


def validate_directory(
    input_path: str,
    input_data_configs_path: str,
    relative_filenames: Optional[Sequence[str]] = None,
    delimiter: str = ",",
    extension: str = "csv",
    unexpected_column_strategy: str = "error",
    check_column_order: bool = False,
) -> bool:
    """Return whether all files in directory conform to their configs.

    Assumes a mirrored structure between `input_path` and `input_data_configs_path`. The
    user can either specify a set of files to check, or the set of files to
    check will be deduced from the set of configs found in the config directory.

    Example:
        For the following directory structure ::

            /some/path
            |-- input
            |   |-- foo.txt
            │   |-- bar/baz.txt
            |   |-- qux.txt
            |-- resources
                |-- config
                    |-- input
                        |-- foo.json
                        |-- bar/baz.json
                        |-- README.md

        The following two calls are equivalent ::

            validate_directory(
                input_path='/some/path/input',
                input_data_configs_path='/some/path/resources/config/input',
                relative_filenames=['foo.txt', 'bar/baz.txt'],
                extension='txt')

            validate_directory(
                input_path='/some/path/input',
                input_data_configs_path='/some/path/resources/config/input',
                extension='txt')

        There is no config for the 'qux' file, so it will not be validated.
        If `qux` were specified in the list of relative filenames, this directory
        structure would produce an error.

    Args:
        input_path: Directory containing the files to validate.
        input_data_configs_path: Directory containing tmlt.common config files
            specifying the expected formats for the input files. Expects a config with
            the same name, but a .json extension for each input csv.
        relative_filenames: A list of relative filenames. The extension must
            match `extension`. If None, it recursively searches for files to
            validate based on the location of .json files in the
            `input_data_configs_path`.
        delimiter: The delimiter used to separate tokens in the input files.
        extension: The file extension for the csv files. Default is "csv".
        unexpected_column_strategy: How to handle columns that are  not in
            the config. Allowed options are 'ignore', 'warn', and 'error'.
            Defaults to 'error'.
        check_column_order: Whether to check that the order of the columns in the csv
            file match the order of the columns in the config.
    """
    if relative_filenames is None:
        relative_filenames = []
        for dirpath, _, filenames in os.walk(input_data_configs_path):
            path = os.path.relpath(dirpath, input_data_configs_path)
            for filename in filenames:
                name_part, extension_part = os.path.splitext(filename)
                if extension_part == ".json":
                    relative_filename = os.path.join(path, name_part)
                    relative_filenames.append(f"{relative_filename}.{extension}")
    relative_filenames_without_extension = []
    for relative_filename in relative_filenames:
        name_part, extension_part = os.path.splitext(relative_filename)
        if not extension_part == "." + extension:
            raise ValueError(
                f'Input filenames must end with ".{extension}", not {relative_filename}'
            )
        relative_filenames_without_extension.append(name_part)

    okay = True
    for filename in relative_filenames_without_extension:
        config_filename = os.path.join(input_data_configs_path, f"{filename}.json")
        config = Config.load_json(config_filename)
        filename = os.path.join(input_path, f"{filename}.{extension}")
        okay &= validate_file(
            input_filename=filename,
            config=config,
            delimiter=delimiter,
            unexpected_column_strategy=unexpected_column_strategy,
            check_column_order=check_column_order,
        )
    return okay


def update_config_object(
    config: Config,
    attribute_to_domain_dict: Dict[str, Sequence[Union[str, int]]],
    attribute_type: Type[Categorical] = CategoricalStr,
) -> Config:
    # noqa: D417, D214 ; suppress incorrect warnings on nested note section.
    """Create new config, updated with actual attributes.

    Args:
        config: tmlt.common Config.
        attribute_to_domain_dict: Dictionary from name of a attribute to
            the domain to use for that attribute. Replaces existing attributes
            with the same name. Attributes that are included in the dictionary,
            but are not found in the config to update are ignored.

            Note:
                This is because the order of config columns often matters,
                so adding attributes from a dictionary is dangerous.
        attribute_type: Attribute object name that represents the type of config
            being created. This could be
            :class:`~tmlt.common.configuration.CategoricalStr` or
            :class:`~tmlt.common.configuration.CategoricalInt`.
    """
    updated_attributes: List[Attribute] = []
    for attribute in config:
        if attribute.column in attribute_to_domain_dict:
            domain = attribute_to_domain_dict[attribute.column]
            updated_attributes.append(attribute_type(attribute.column, domain))
        else:
            updated_attributes.append(attribute)
    return Config(updated_attributes)


def update_config(
    input_data_configs_path: str,
    output_path: str,
    file_root: str,
    attribute_to_domain_dict: Dict[str, Sequence[Union[str, int]]],
    attribute_type: Type[Categorical] = CategoricalStr,
):  # noqa: D417, D214 ; suppress incorrect warnings due to nested note section.
    """Save a new config updated with actual attributes.

    This function, along with :func:`validate_directory` helps with a
    validation workflow where you have the following three directories

    1. A directory containing input csv files.
    2. A directory containing initial serialized Config objects.
    3. A directory to store updated Config objects.

    The purpose of the second directory is to do the initial validation, for
    instance making sure the input columns are in the expected format.

    The purpose of the third directory is to do a second round of validation,
    which checks that the various input files are consistent. It is often the
    case that some of the input files further specify the schema of other
    files. This is exactly the use case this method is intended for, for the
    special case of :class:`~tmlt.common.configuration.CategoricalStr` attributes.

    The full process usually requires project specific details, but
    generally follows these steps

    1. In the first phase the input files are validated against the initial
       config files. (Using :func:`validate_directory`)
    2. In the second phase, the schemas are updated using information contained
       in the various input files. This is often highly project dependant, but
       may use this function.
    3. In the third phase, the input files are validated again, this time
       against the updated configs.

    Args:
        input_data_configs_path: Location for the original configs.
        output_path: Location to save the updated configs.
        file_root: Filename of the input file the configs correspond to.
        attribute_to_domain_dict: Dictionary from name of a attribute to
            the domain to use for that attribute. Replaces existing attributes
            with the same name. Attributes that are included in the dictionary,
            but are not found in the config to update are ignored.

            Note:
                This is because the order of config columns often
            matters, so adding attributes from a dictionary is dangerous.
        attribute_type: Attribute object name that represents the type of config
            being created. This could be
            :class:`~tmlt.common.configuration.CategoricalStr` or
            :class:`~tmlt.common.configuration.CategoricalInt`.
    """
    config_filename = os.path.join(input_data_configs_path, f"{file_root}.json")
    config = Config.load_json(config_filename)
    updated_config = update_config_object(
        config, attribute_to_domain_dict, attribute_type
    )
    updated_config.save_json(os.path.join(output_path, f"{file_root}.json"))
