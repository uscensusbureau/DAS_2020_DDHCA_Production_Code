"""Unit test for :mod:`tmlt.safetab_utils.characteristic_iterations`."""

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
import unittest
from collections import defaultdict
from typing import DefaultDict, Dict, Set

import pandas as pd
from parameterized import parameterized

from tmlt.common.configuration import Row
from tmlt.safetab_utils.characteristic_iterations import (
    IterationFilter,
    IterationManager,
)

race_eth_codes_to_iterations_testcases = list(
    pd.read_csv(
        io.BytesIO(
            pkgutil.get_data(  # type: ignore
                "tmlt.safetab_utils",
                "resources/test/race_eth_codes_to_iterations_testcases.txt",
            )
        ),
        encoding="utf8",
        delimiter="|",
        dtype=str,
    ).itertuples(index=False)
)
"""Test cases that have been vetted by the Census.

Used in :class:`.TestRaceEthCodesToIterations`.
"""


def _write_testdata_to_temp(temp_dir: str, filename: str):
    filepath = os.path.join(temp_dir, filename)
    pd.read_csv(
        io.BytesIO(
            pkgutil.get_data(  # type: ignore
                "tmlt.safetab_utils", f"resources/test/{filename}"
            )
        ),
        encoding="utf-8",
        delimiter="|",
        dtype=str,
    ).to_csv(filepath, index=False, sep="|")


class TestIterationManager(unittest.TestCase):
    """TestCase for
    :class:`tmlt.safetab_utils.characteristic_iterations.IterationFilter`."""

    input_dir: str
    iteration_manager: IterationManager

    @classmethod
    def setUpClass(cls) -> None:
        """Create an IterationManager for tests."""

        cls.input_dir = tempfile.mkdtemp()
        _write_testdata_to_temp(cls.input_dir, "race-characteristic-iterations.txt")
        _write_testdata_to_temp(
            cls.input_dir, "ethnicity-characteristic-iterations.txt"
        )
        _write_testdata_to_temp(
            cls.input_dir, "race-and-ethnicity-code-to-iteration.txt"
        )
        cls.iteration_manager = IterationManager(cls.input_dir, max_race_codes=3)

    @classmethod
    def tearDownClass(cls) -> None:
        """Cleans up temp directory."""
        shutil.rmtree(cls.input_dir)

    @parameterized.expand(
        [
            (
                {},
                {
                    "2001",
                    "3001",
                    "4001",
                    "7000",
                    "7100",
                    "7400",
                    "1130",
                    "1140",
                    "1150",
                    "1810",
                    "1820",
                    "1850",
                    "2250",
                    "2260",
                    "2380",
                    "2930",
                    "2940",
                    "3060",
                    "8000",
                    "8010",
                    "8020",
                    "8030",
                    "8040",
                    "8050",
                    "9000",
                    "9010",
                    "9020",
                    "9030",
                    "9040",
                    "9050",
                    "3010",
                    "3011",
                    "3040",
                    "3042",
                    "3080",
                    "3081",
                },
            ),
            (
                {"race_eth_type": "race", "alone": "True", "detailed_only": "True"},
                {"4001", "8000", "8010", "8020", "8030", "8040", "8050"},
            ),
            (
                {
                    "race_eth_type": "ethnicity",
                    "detailed_only": "False",
                    "coarse_only": "False",
                },
                {"3010", "3011"},
            ),
            (
                {"race_eth_type": "race", "coarse_only": "False", "level": "1"},
                {
                    "2001",
                    "7100",
                    "1130",
                    "1820",
                    "2250",
                    "2930",
                    "8000",
                    "8020",
                    "8040",
                    "9000",
                    "9020",
                    "9040",
                },
            ),
        ]
    )
    def test_get_iteration_codes(
        self, kwargs: Dict[str, str], expected_iterations: Set[str]
    ):
        """IterationFilter returns the expected iterations.

        Args:
            kwargs: Keyword arguments to pass to IterationFilter.
            expected_iterations: The iterations we expect get_iteration_codes to return.
        """
        iteration_filter = IterationFilter(**kwargs)
        actual_iterations = set(
            self.iteration_manager.get_iteration_codes(iteration_filter)
        )
        self.assertEqual(actual_iterations, expected_iterations)

    @parameterized.expand(
        [
            (
                {"QRACE1": "1011", "QRACE2": "1012", "QRACE3": "1031", "QSPAN": "1010"},
                {"2001", "1130"},
            ),
            (
                {"QRACE1": "1001", "QRACE2": "1020", "QSPAN": "2020"},
                {"2001", "1130", "3010"},
            ),
        ]
    )
    def test_create_add_iterations_flat_map(
        self, row: Row, expected_iterations: Set[str]
    ):
        """Create_add_iterations_flat_map is correct for test input.

        Only tests using coarse_only=False and level="1".

        Args:
            row: The QRACE and QSPAN values for a record.
            expected_iterations: The expected iterations matching the record.
        """
        (
            sensitivity,
            config,
            add_iterations_flat_map,
        ) = self.iteration_manager.create_add_iterations_flat_map(
            coarse_only="False", level="1"
        )
        input_row: DefaultDict = defaultdict(lambda: "Null")
        input_row.update(row)
        output_rows = add_iterations_flat_map(input_row)
        actual_iterations = {output_row["ITERATION_CODE"] for output_row in output_rows}
        self.assertEqual(sensitivity, 4)
        self.assertEqual(actual_iterations, expected_iterations)
        self.assertEqual(config.domain_size, 14)

    @parameterized.expand(
        [
            (
                {
                    "USA": "01",
                    "QRACE1": "1011",
                    "QRACE2": "1012",
                    "QRACE3": "1031",
                    "QSPAN": "1010",
                },
                {"01,2001", "01,1130"},
            ),
            (
                {"USA": "01", "QRACE1": "1001", "QRACE2": "1020", "QSPAN": "2020"},
                {"01,2001", "01,1130", "01,3010"},
            ),
        ]
    )
    def test_create_add_pop_groups_flat_map(
        self, row: Row, expected_pop_groups: Set[str]
    ):
        """Create_add_pop_groups_flat_map is correct for test input.

        Only tests using coarse_only=False, level="1", and region_type="USA".

        Args:
            row: The QRACE, QSPAN, and USA values for a record.
            expected_pop_groups: The expected pop_groups matching the record.
        """
        (
            sensitivity,
            config,
            add_iterations_flat_map,
        ) = self.iteration_manager.create_add_pop_groups_flat_map(
            coarse_only="False", level="1", region_type="USA", region_domain=["01"]
        )
        input_row: DefaultDict = defaultdict(lambda: "Null")
        input_row.update(row)
        output_rows = add_iterations_flat_map(input_row)
        actual_pop_groups = {output_row["POP_GROUP"] for output_row in output_rows}
        self.assertEqual(sensitivity, 4)
        self.assertEqual(actual_pop_groups, expected_pop_groups)
        self.assertEqual(config.domain_size, 14)


class TestRaceEthCodesToIterations(unittest.TestCase):
    """Creates tests from :data:`.race_eth_codes_to_iterations_testcases`."""

    parameters_path: str
    race_eth_code_to_name: pd.DataFrame
    iteration_code_to_name: pd.DataFrame
    detailed_race_eth_codes_to_iterations: pd.DataFrame
    coarse_race_eth_codes_to_iterations: pd.DataFrame

    @classmethod
    def setUpClass(cls):
        """Load code_to_name input files."""
        cls.parameters_path = tempfile.mkdtemp()
        _write_testdata_to_temp(
            cls.parameters_path, "race-and-ethnicity-code-to-iteration.txt"
        )
        _write_testdata_to_temp(cls.parameters_path, "race-and-ethnicity-codes.txt")
        race_eth_code_to_name = pd.read_csv(
            os.path.join(cls.parameters_path, "race-and-ethnicity-codes.txt"),
            sep="|",
            dtype=str,
        )
        cls.race_eth_code_to_name = race_eth_code_to_name.set_index(
            "RACE_ETH_CODE", drop=True
        )
        _write_testdata_to_temp(
            cls.parameters_path, "ethnicity-characteristic-iterations.txt"
        )
        ethnicity_iteration_code_to_name = pd.read_csv(
            os.path.join(
                cls.parameters_path, "ethnicity-characteristic-iterations.txt"
            ),
            sep="|",
            dtype=str,
        )[["ITERATION_CODE", "ITERATION_NAME"]]
        _write_testdata_to_temp(
            cls.parameters_path, "race-characteristic-iterations.txt"
        )
        race_iteration_code_to_name = pd.read_csv(
            os.path.join(cls.parameters_path, "race-characteristic-iterations.txt"),
            sep="|",
            dtype=str,
        )[["ITERATION_CODE", "ITERATION_NAME"]]
        iteration_code_to_name = pd.concat(
            [ethnicity_iteration_code_to_name, race_iteration_code_to_name]
        )
        cls.iteration_code_to_name = iteration_code_to_name.set_index(
            "ITERATION_CODE", drop=True
        )
        iteration_manager = IterationManager(cls.parameters_path, max_race_codes=3)
        (
            cls.detailed_sensitivity,
            cls.detailed_race_eth_codes_to_iterations,
        ) = iteration_manager.create_race_eth_codes_to_iterations(coarse_only="False")
        (
            cls.coarse_sensitivity,
            cls.coarse_race_eth_codes_to_iterations,
        ) = iteration_manager.create_race_eth_codes_to_iterations(detailed_only="False")

    @parameterized.expand(race_eth_codes_to_iterations_testcases)
    def test_race_eth_codes_to_iterations_testcases(
        self,
        race_codes_str: str,
        ethnicity_code: str,
        detailed_iteration_codes_str: str,
        coarse_iteration_codes_str: str,
    ):
        """race_eth_codes_to_iterations agrees with census test cases.

        Test cases contained in :data:`.race_eth_codes_to_iterations_testcases`

        Args:
            race_codes_str: A comma separated string of race codes.
            ethnicity_code: An ethnicity code.
            detailed_iteration_codes_str: A comma separated string of expected detailed
                iteration codes.
            coarse_iteration_codes_str: A comma separated string of expected coarse
                    iteration codes.
        """
        race_codes = race_codes_str.split(",")
        expected_detailed_iteration_codes = set(detailed_iteration_codes_str.split(","))
        expected_coarse_iteration_codes = set(coarse_iteration_codes_str.split(","))
        race_names = list(self.race_eth_code_to_name.loc[race_codes, "RACE_ETH_NAME"])
        expected_detailed_iteration_names = list(
            self.iteration_code_to_name.loc[
                expected_detailed_iteration_codes, "ITERATION_NAME"
            ]
        )
        expected_coarse_iteration_names = list(
            self.iteration_code_to_name.loc[
                expected_coarse_iteration_codes, "ITERATION_NAME"
            ]
        )
        print("Race and ethnicity codes")
        for i, (race_code, race_name) in enumerate(zip(race_codes, race_names)):
            print(f"\tRACE{i + 1}: {race_code} ({race_name})")
        ethnicity_name = self.race_eth_code_to_name.loc[ethnicity_code]["RACE_ETH_NAME"]
        print(f"\tQSPAN: {ethnicity_code} ({ethnicity_name})")
        print("Expected Detailed (COARSE_ONLY=False) Iterations")
        for iteration_code, iteration_name in zip(
            expected_detailed_iteration_codes, expected_detailed_iteration_names
        ):
            print(f"\t{iteration_code} ({iteration_name})")
        actual_detailed_iteration_codes = set(
            TestRaceEthCodesToIterations.detailed_race_eth_codes_to_iterations(
                race_codes, ethnicity_code
            )
        )
        actual_detailed_iteration_names = list(
            self.iteration_code_to_name.loc[
                actual_detailed_iteration_codes, "ITERATION_NAME"
            ]
        )
        print("Actual Detailed Iterations")
        for iteration_code, iteration_name in zip(
            actual_detailed_iteration_codes, actual_detailed_iteration_names
        ):
            print(f"\t{iteration_code} ({iteration_name})")
        print()
        self.assertEqual(
            actual_detailed_iteration_codes, expected_detailed_iteration_codes
        )
        print("Expected Coarse (DETAILED_ONLY=False) Iterations")
        for iteration_code, iteration_name in zip(
            expected_coarse_iteration_codes, expected_coarse_iteration_names
        ):
            print(f"\t{iteration_code} ({iteration_name})")
        actual_coarse_iteration_codes = set(
            TestRaceEthCodesToIterations.coarse_race_eth_codes_to_iterations(
                race_codes, ethnicity_code
            )
        )
        actual_coarse_iteration_names = list(
            self.iteration_code_to_name.loc[
                actual_coarse_iteration_codes, "ITERATION_NAME"
            ]
        )
        print("Actual Coarse Iterations")
        for iteration_code, iteration_name in zip(
            actual_coarse_iteration_codes, actual_coarse_iteration_names
        ):
            print(f"\t{iteration_code} ({iteration_name})")
        print()
        self.assertEqual(actual_coarse_iteration_codes, expected_coarse_iteration_codes)

    def test_sensitivity(self):
        """race_eth_codes_to_iterations has the correct sensitivity."""
        self.assertEqual(self.detailed_sensitivity, 8)
        self.assertEqual(self.coarse_sensitivity, 8)

    @classmethod
    def tearDownClass(cls) -> None:
        """Cleans up temp directory."""
        shutil.rmtree(cls.parameters_path)
