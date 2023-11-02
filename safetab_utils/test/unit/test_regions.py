"""Unit test for :mod:`tmlt.safetab_utils.regions`."""

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
import io
import os
import pkgutil
import shutil
import tempfile
from pathlib import Path
from typing import List

import pandas as pd
from parameterized import parameterized

from tmlt.common.configuration import Config, Unrestricted
from tmlt.core.utils.testing import PySparkTest
from tmlt.safetab_utils.csv_reader import GEO_FILENAME
from tmlt.safetab_utils.regions import preprocess_geography_df, validate_state_filter_us
from tmlt.safetab_utils.utils import STATE_FILTER_FLAG, safetab_input_reader


class TestRegions(PySparkTest):
    """TestCase for :mod:`tmlt.safetab_utils.regions`."""

    def setUp(self):
        """Set up test."""
        self.test_dir = tempfile.mkdtemp()
        json_file = Path(os.path.join(self.test_dir, "GRF-C.json"))
        json_file.write_bytes(
            pkgutil.get_data("tmlt.safetab_utils", "resources/test/GRF-C.json")
        )
        geo_filename = os.path.join(self.test_dir, GEO_FILENAME)
        pd.read_csv(
            io.BytesIO(
                pkgutil.get_data(  # type: ignore
                    "tmlt.safetab_utils", "resources/test/GRF-C.txt"
                )
            ),
            encoding="utf8",
            delimiter="|",
            dtype=str,
        ).to_csv(geo_filename, index=False, sep="|")

    def tearDown(self) -> None:
        """Cleans up temp directory."""
        shutil.rmtree(self.test_dir)

    def test_preprocess_geography_df_us(self):
        """geography_df is correct for test input, for us_or_puerto_rico="US"."""
        expected_geography_df = pd.DataFrame(
            {
                "TABBLKST": ["01"] * 8 + ["02"] * 8,
                "TABBLKCOU": (["001"] * 4 + ["002"] * 4) * 2,
                "TABTRACTCE": (["000001"] * 2 + ["000002"] * 2) * 4,
                "TABBLK": ["0001", "0002"] * 8,
                "USA": ["1"] * 16,
                "STATE": ["01"] * 8 + ["02"] * 8,
                "COUNTY": ["01001"] * 4 + ["01002"] * 4 + ["02001"] * 4 + ["02002"] * 4,
                "TRACT": (
                    ["01001000001"] * 2
                    + ["01001000002"] * 2
                    + ["01002000001"] * 2
                    + ["01002000002"] * 2
                    + ["02001000001"] * 2
                    + ["02001000002"] * 2
                    + ["02002000001"] * 2
                    + ["02002000002"] * 2
                ),
                "PLACE": ["0122800"] * 8 + ["0222800"] * 8,
                "AIANNH": ["0001"] * 4 + ["0002"] * 6 + ["0003"] * 6,
            }
        )
        actual_geography_df = preprocess_geography_df(
            safetab_input_reader(
                reader="csv",
                data_path=self.test_dir,
                state_filter=["01", "02"],
                program="safetab-h",
            ),
            us_or_puerto_rico="US",
            input_config_dir_path=self.test_dir,
        ).toPandas()

        pd.testing.assert_frame_equal(
            actual_geography_df, expected_geography_df, check_like=True
        )

    def test_preprocess_geography_df_pr(self):
        """geography_df is correct for test input, for us_or_puerto_rico="PR"."""
        expected_geography_df = pd.DataFrame(
            {
                "TABBLKST": ["72"],
                "TABBLKCOU": ["031"],
                "TABTRACTCE": ["050901"],
                "TABBLK": ["2040"],
                "PR-COUNTY": ["72031"],
                "PR-STATE": ["72"],
                "PR-TRACT": ["72031050901"],
                "PR-PLACE": ["NULL"],
            }
        )
        actual_geography_df = preprocess_geography_df(
            safetab_input_reader(
                reader="csv",
                data_path=self.test_dir,
                state_filter=["72"],
                program="safetab-h",
            ),
            us_or_puerto_rico="PR",
            input_config_dir_path=self.test_dir,
        ).toPandas()

        pd.testing.assert_frame_equal(
            actual_geography_df, expected_geography_df, check_like=True
        )

    def test_grfc_unexpected_columns(self):
        """create_geography_df checks that we validated every column we are using.

        This test patches in a fake geography config that it thinks would have been used
        during input validation so that it does not include one of the columns that are
        are actually used in the `GRF-C.txt`.

        See `test_input_validation.test_unexpected_columns_grfc_okay` for more
        information.
        """
        temporary_config_directory = tempfile.TemporaryDirectory()
        columns = [
            "TABBLKST",
            "TABBLKCOU",
            "TABTRACTCE",
            "PLACEFP",
            "TABBLK",
        ]  # Missing "AIANNHCE"
        grfc_config = Config([Unrestricted(column) for column in columns])
        filename = os.path.join(temporary_config_directory.name, "GRF-C.json")
        grfc_config.save_json(filename)

        with self.assertRaises(AssertionError):
            preprocess_geography_df(
                safetab_input_reader(
                    reader="csv",
                    data_path=self.test_dir,
                    state_filter=["01", "02", "11"],
                    program="safetab-h",
                ),
                us_or_puerto_rico="US",
                input_config_dir_path=temporary_config_directory.name,
            )

    def test_validate_state_filter_us(self):
        """Tests that the US state filter does not contain unexpected values."""
        state_filter_us = ["01", "02", "11"]
        okay = validate_state_filter_us(state_filter_us)
        self.assertTrue(okay)

    # Remember to escape opening braces when they contain a number (e.g. \{03})
    # so that they are not interpreted as regex repetition markers.
    @parameterized.expand(
        [
            (
                [],
                f"Invalid config: expected '{STATE_FILTER_FLAG}' to not be empty for "
                "US run",
            ),
            (
                {"01"},
                f"Invalid config: expected '{STATE_FILTER_FLAG}' to have type list,"
                " not set",
            ),
            (
                [10],
                f"Invalid config: expected '{STATE_FILTER_FLAG}' elements to have type"
                " str, not {int}",
            ),
            (
                ["01", "02", "11", "01"],
                f"Invalid config: '{STATE_FILTER_FLAG}' list contains duplicate values",
            ),
            (
                ["3", "57", "43"],
                rf"Invalid config: '{STATE_FILTER_FLAG}' contains invalid codes: \{{3,"
                r" 43, 57}",
            ),
            (
                ["random"],
                rf"Invalid config: '{STATE_FILTER_FLAG}' contains invalid codes:"
                r" \{random}",
            ),
            (
                ["01", "02", "011", "14", "60", "64", "72", "78"],
                rf"Invalid config: '{STATE_FILTER_FLAG}' contains invalid codes:"
                r" \{011, 14, 60, 64, 72, 78}",
            ),
        ]
    )
    def test_invalid_state_filter_us(
        self, state_filter_us: List[str], error_msg_regex: str
    ):
        """Tests that the US state filter does not contain unexpected values."""
        with self.assertRaisesRegex(ValueError, error_msg_regex):
            _ = validate_state_filter_us(state_filter_us)
