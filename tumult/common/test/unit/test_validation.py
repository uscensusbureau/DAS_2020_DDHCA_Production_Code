"""Unit tests for :mod:`~tmlt.common.validation`."""

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
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, call, patch

import pandas as pd
from parameterized import parameterized_class

from tmlt.common.configuration import CategoricalStr, Config, Unrestricted
from tmlt.common.pyspark_test_tools import PySparkTest
from tmlt.common.validation import (
    MAX_INVALID_VALUES_LOGGED,
    update_config,
    validate_directory,
    validate_file,
    validate_spark_df,
)


class TestValidation(PySparkTest):
    """TestCase for validation."""

    def setUp(self):
        """Set up test."""
        self.input_dir = tempfile.mkdtemp()
        self.okay_filename = os.path.join(self.input_dir, "okay.csv")
        okay_df = pd.DataFrame(
            {"A": ["a1", "a1", "a2", "a2"], "B": ["b1", "b2", "b1", "b2"]}
        )
        okay_df.to_csv(self.okay_filename, index=False)

        self.not_okay_filename = os.path.join(self.input_dir, "not/okay.csv")
        not_okay_df = pd.DataFrame(
            {"C": ["c1", "c1", "c2", "c2", "c3"], "D": ["d1", "d2", "d4", "d2", "d2"]}
        )
        os.mkdir(os.path.join(self.input_dir, "not"))
        not_okay_df.to_csv(self.not_okay_filename, index=False)

        self.config_dir = tempfile.mkdtemp()
        self.okay_config_filename = os.path.join(self.config_dir, "okay.json")
        okay_config = Config(
            [Unrestricted("A", pattern=r"a\d$"), CategoricalStr("B", ["b1", "b2"])]
        )
        okay_config.save_json(self.okay_config_filename)

        self.not_okay_config_filename = os.path.join(self.config_dir, "not/okay.json")
        not_okay_config = Config(
            [CategoricalStr("C", ["c1", "c2"]), CategoricalStr("D", ["d1", "d2"])]
        )
        os.mkdir(os.path.join(self.config_dir, "not"))
        not_okay_config.save_json(self.not_okay_config_filename)

    def tearDown(self):
        """Tear down test."""
        shutil.rmtree(self.input_dir)
        shutil.rmtree(self.config_dir)

    @patch("tmlt.common.validation.logging")
    def test_validate_file_okay(self, mock_logger: MagicMock):
        """Validate_file returns True for properly formatted files.

        Args:
            mock_logger: mocked logger to be used for checking logged validation info.
        """
        config = Config.load_json(self.okay_config_filename)
        okay = validate_file(input_filename=self.okay_filename, config=config)
        mock_logger.info.assert_has_calls(
            [
                call("Loading %s...", self.okay_filename),
                call("%s passed schema validation.", self.okay_filename),
            ]
        )
        self.assertTrue(okay)

    @patch("tmlt.common.validation.logging")
    def test_validate_file_unexpected_values(self, mock_logger: MagicMock):
        """Validate_file returns False if unexpected values are found.

        Args:
            mock_logger: mocked logger to be used for checking logged validation info.
        """
        config = Config.load_json(self.not_okay_config_filename)
        okay = validate_file(input_filename=self.not_okay_filename, config=config)
        mock_logger.error.assert_has_calls(
            [
                call("Invalid values found in C: ['c3']"),
                call("Invalid values found in D: ['d4']"),
                call("%s failed schema validation.", self.not_okay_filename),
            ]
        )
        self.assertFalse(okay)

    @patch("tmlt.common.validation.logging")
    def test_validate_file_many_unexpected_values(self, mock_logger: MagicMock):
        """Validate_file does not show all unexpected values if there are too many.

        In particular, it shows the first
        :data:`~tmlt.common.validation.MAX_INVALID_VALUES_LOGGED` distinct
        values.

        Args:
            mock_logger: mocked logger to be used for checking logged validation info.
        """
        config = Config([CategoricalStr("A", ["a1", "a2"])])

        invalid_values = sorted([f"b{i}" for i in range(MAX_INVALID_VALUES_LOGGED)])
        data = "\n".join(["A"] + invalid_values)
        with open(self.not_okay_filename, "w") as f:
            f.write(data)

        okay = validate_file(input_filename=self.not_okay_filename, config=config)
        mock_logger.error.assert_has_calls(
            [
                call(
                    f"Invalid values found in A: {invalid_values} "
                    f"(only the first {MAX_INVALID_VALUES_LOGGED} values are shown)"
                ),
                call("%s failed schema validation.", self.not_okay_filename),
            ]
        )
        self.assertFalse(okay)

    def test_validate_directory_okay(self):
        """Validate_directory returns True for properly formatted files."""
        okay = validate_directory(
            input_path=self.input_dir,
            input_data_configs_path=self.config_dir,
            relative_filenames=["okay.csv"],
        )
        self.assertTrue(okay)

    def test_validate_directory_not_okay(self):
        """Validate_directory returns False for improperly formatted files."""
        okay = validate_directory(
            input_path=self.input_dir, input_data_configs_path=self.config_dir
        )
        self.assertFalse(okay)

    def test_validate_directory_catches_bad_filename(self):
        """validate_directory returns False for a file with the wrong extension."""
        with self.assertRaisesRegex(ValueError, 'must end with ".csv"'):
            validate_directory(
                input_path=self.input_dir,
                input_data_configs_path=self.config_dir,
                relative_filenames=["bad.jpg"],
                extension="csv",
            )

    def test_handle_unexpected_columns_ignore(self):
        """Validate_file can ignore unexpected columns."""
        config = Config.load_json(self.okay_config_filename)
        config = config[["A"]]
        okay = validate_file(
            input_filename=self.okay_filename,
            config=config,
            unexpected_column_strategy="ignore",
        )
        self.assertTrue(okay)

    def test_handle_bad_column_order_no_check(self):
        """Validate_file does not throw error with out-of-order when check is false."""
        config = Config.load_json(self.okay_config_filename)
        config = config[["B", "A"]]
        okay = validate_file(input_filename=self.okay_filename, config=config)
        self.assertTrue(okay)

    @patch("tmlt.common.validation.logging")
    def test_handle_unexpected_columns_warn(self, mock_logger: MagicMock):
        """Validate_file can warn if unexpected columns are found.

        Args:
            mock_logger: mocked logger to be used for checking logged warnings.
        """
        config = Config.load_json(self.okay_config_filename)
        config = config[["A"]]
        okay = validate_file(
            input_filename=self.okay_filename,
            config=config,
            unexpected_column_strategy="warn",
        )
        mock_logger.warning.assert_called_with("Unexpected columns found: ['B']")
        self.assertTrue(okay)

    @patch("tmlt.common.validation.logging")
    def test_handle_unexpected_columns_error(self, mock_logger: MagicMock):
        """Validate_file can log an error if unexpected columns are found.

        Args:
            mock_logger: mocked logger to be used for checking logged errors.
        """
        config = Config.load_json(self.okay_config_filename)
        config = config[["A"]]
        okay = validate_file(
            input_filename=self.okay_filename,
            config=config,
            unexpected_column_strategy="error",
        )
        mock_logger.error.assert_has_calls(
            [
                call("Unexpected columns found: ['B']"),
                call("%s failed schema validation.", self.okay_filename),
            ]
        )
        self.assertFalse(okay)

    @patch("tmlt.common.validation.logging")
    def test_handle_bad_column_order_with_check(self, mock_logger: MagicMock):
        """Validate_file throws error with out-of-order when check is True.

        Args:
            mock_logger: mocked logger to be used for checking logged errors.
        """
        config = Config.load_json(self.okay_config_filename)
        config = config[["B", "A"]]
        okay = validate_file(
            input_filename=self.okay_filename, config=config, check_column_order=True
        )
        mock_logger.error.assert_has_calls(
            [
                call(
                    "Columns are out of order."
                    " Expected order: ['B', 'A']."
                    " Actual order: ['A', 'B']."
                ),
                call("%s failed schema validation.", self.okay_filename),
            ]
        )
        self.assertFalse(okay)

    @patch("tmlt.common.validation.logging")
    def test_missing_column(self, mock_logger: MagicMock):
        """Validate_file throws error when a column is not found.

        Args:
            mock_logger: mocked logger to be used for checking logged errors.
        """
        config = Config.load_json(self.okay_config_filename)
        config = config + Config([CategoricalStr("C", ["c1", "c2"])])
        okay = validate_file(input_filename=self.okay_filename, config=config)
        mock_logger.error.assert_has_calls(
            [
                call(
                    "Missing columns."
                    f" Expected columns: {config.columns}."
                    " Actual columns: ['A', 'B']."
                ),
                call("%s failed schema validation.", self.okay_filename),
            ]
        )
        self.assertFalse(okay)


class TestUpdateConfig(unittest.TestCase):
    """TestCase for :tmlt.common.validation.update_config`."""

    def setUp(self):
        """Create temporary directories."""
        self.config_dir = tempfile.mkdtemp()
        self.output_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Remove temporary directories."""
        shutil.rmtree(self.config_dir)
        shutil.rmtree(self.output_dir)

    def test_update_config(self):
        """Update config correctly creates new, updated config."""
        config_filename = os.path.join(self.config_dir, "test.json")
        config = Config(
            [Unrestricted("A", pattern=r"a\d$"), Unrestricted("B", pattern=r"b\d$")]
        )
        config.save_json(config_filename)

        update_config(
            input_data_configs_path=self.config_dir,
            output_path=self.output_dir,
            file_root="test",
            attribute_to_domain_dict={"A": ["a1", "a2"]},
        )

        expected_config = Config(
            [CategoricalStr("A", ["a1", "a2"]), Unrestricted("B", pattern=r"b\d$")]
        )
        updated_filename = os.path.join(self.output_dir, "test.json")
        actual_config = Config.load_json(updated_filename)
        self.assertEqual(actual_config, expected_config)


SAFETAB_P_GRF_C_SCHEMA = """{
    "marshalled": true,
    "class": "tmlt.common.configuration.Config",
    "init_params": {
        "attributes": [
            {
                "marshalled": true,
                "class": "tmlt.common.configuration.Unrestricted",
                "init_params": {
                    "column": "TABBLKST",
                    "pattern": "\\\\d{2}$"
                }
            },
            {
                "marshalled": true,
                "class": "tmlt.common.configuration.Unrestricted",
                "init_params": {
                    "column": "TABBLKCOU",
                    "pattern": "\\\\d{3}$"
                }
            },
            {
                "marshalled": true,
                "class": "tmlt.common.configuration.Unrestricted",
                "init_params": {
                    "column": "TABTRACTCE",
                    "pattern": "\\\\d{6}$"
                }
            },
            {
                "marshalled": true,
                "class": "tmlt.common.configuration.Unrestricted",
                "init_params": {
                    "column": "TABBLK",
                    "pattern": "\\\\d{4}$"
                }
            },
            {
                "marshalled": true,
                "class": "tmlt.common.configuration.Unrestricted",
                "init_params": {
                    "column": "PLACEFP",
                    "pattern": "\\\\d{5}$"
                }
            },
            {
                "marshalled": true,
                "class": "tmlt.common.configuration.Unrestricted",
                "init_params": {
                    "column": "AIANNHCE",
                    "pattern": "\\\\d{4}$"
                }
            }
        ]
    }
}
"""


@parameterized_class(
    [
        {
            # Valid Spark DataFrame SafeTab-P
            "data": """TABBLKST|TABBLKCOU|TABTRACTCE|PLACEFP|TABBLK|AIANNHCE
                             01|      001|    000001|  22800|  0001|    0001
                             72|      031|    050901|  99999|  2040|    9999
                             02|      002|    000002|  99999|  0001|    9999
                             11|      001|    000100|  99999|  3004|    9999
            """,
            "file_name": "GRF-C",
            "valid_status": True,
            "config_str": SAFETAB_P_GRF_C_SCHEMA,
            "allow_empty": False,
        },
        {
            # Invalid Columns in SafeTab-P DataFrame
            "data": """TABBLKST||TABTRACTCE|PLACEFP|TABBLK|AIANNHCE
                             01|001| 000001|  22800|  0001|    0001
            """,  # Missing TABBLKCOU
            "file_name": "GRF-C",
            "valid_status": False,
            "config_str": SAFETAB_P_GRF_C_SCHEMA,
            "allow_empty": False,
        },
        {
            # No data in SafeTab-P DataFrame
            "data": """TABBLKST|TABBLKCOU|TABTRACTCE|PLACEFP|TABBLK|AIANNHCE
            """,
            "file_name": "GRF-C",
            "valid_status": False,
            "config_str": SAFETAB_P_GRF_C_SCHEMA,
            "allow_empty": False,
        },
        {
            # No data in SafeTab-P DataFrame, but empty allowed
            "data": """TABBLKST|TABBLKCOU|TABTRACTCE|PLACEFP|TABBLK|AIANNHCE
            """,
            "file_name": "GRF-C",
            "valid_status": True,
            "config_str": SAFETAB_P_GRF_C_SCHEMA,
            "allow_empty": True,
        },
    ]
)
class TestValidateSparkDF(PySparkTest):
    """Test to ensure that the validate_spark_df function works as expected."""

    data: str
    file_name: str
    valid_status: bool
    config_str: str
    allow_empty: bool

    def setUp(self):
        # Setup temporary directory and config file
        self.temp_dir = tempfile.mkdtemp()
        self.config_filename = os.path.join(self.temp_dir, "config.json")
        with open(self.config_filename, "w") as f:
            f.write(self.config_str)

        self.config = Config.load_json(self.config_filename)

        # Create Spark DataFrame
        self.data_file = os.path.join(self.temp_dir, "data_file.txt")
        with open(self.data_file, "w") as f:
            f.write(self.data.replace(" ", ""))
        self.df = self.spark.read.csv(self.data_file, header=True, sep="|")

    def test_validate_spark_df(self):
        """Test that validate_spark_df returns the expected value."""
        self.assertEqual(
            validate_spark_df(
                self.file_name, self.df, self.config, allow_empty=self.allow_empty
            ),
            self.valid_status,
        )
