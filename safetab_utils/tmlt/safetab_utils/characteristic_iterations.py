"""Functions for managing race codes and characteristic iterations."""

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

import itertools
import os
from collections import defaultdict
from typing import (
    Callable,
    DefaultDict,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Set,
    Tuple,
)

import pandas as pd

from tmlt.common.configuration import CategoricalStr, Config, Row
from tmlt.common.io_helpers import read_csv


class FlatMapContainer(NamedTuple):
    """A container for a flatmap function and associated information."""

    sensitivity: int
    output_domain: Config
    flat_map: Callable[[Row], List[Row]]


class IterationFilter:
    """Specifies a subset of all characteristic iterations."""

    def __init__(
        self,
        race_eth_type: str = "both",
        alone: str = "both",
        detailed_only: str = "both",
        coarse_only: str = "both",
        level: str = "both",
    ):
        """Constructor.

        Args:
            race_eth_type: Whether to include only race iteration codes, only ethnicity
                iteration codes, or both. Allowed options are "race", "ethnicity", and
                "both".
            alone: Whether to include only race iteration codes where ALONE is equal to
                "True" or "False". Allowed options are "True", "False", and "both". If
                "True" or "False", race_eth_type must be "race".
            detailed_only: Whether to include only iteration codes where DETAILED_ONLY
                is equal to "True" or "False". Allowed options are "True", "False", and
                "both". If "True", coarse_only cannot also be "True".
            coarse_only: Whether to include only iteration codes where COARSE_ONLY
                is equal to "True" or "False". Allowed options are "True", "False", and
                "both". If "True", detailed_only cannot also be "True".
            level: Whether to include only iteration codes where LEVEL is equal to "1"
                or "2". Allowed options are "1", "2", and "both".
        """
        if detailed_only == "True" and coarse_only == "True":
            raise ValueError("detailed_only and coarse_only cannot both be 'True'.")
        if alone != "both":
            if race_eth_type != "race":
                raise ValueError(
                    "race_eth_type must be 'race' if alone != 'both', not "
                    f"'{race_eth_type}'"
                )
        if race_eth_type not in {"race", "ethnicity", "both"}:
            raise ValueError(
                "race_eth_type must be 'race', 'ethnicity', or 'both', not "
                f"'{race_eth_type}'"
            )
        self._race_eth_type = race_eth_type

        if alone not in {"True", "False", "both"}:
            raise ValueError(f"alone must be 'True', 'False', or 'both', not '{alone}'")
        self._alone = alone

        if detailed_only not in {"True", "False", "both"}:
            raise ValueError(
                "detailed_only must be 'True', 'False', or 'both', not "
                f"'{detailed_only}'"
            )
        self._detailed_only = detailed_only

        if coarse_only not in {"True", "False", "both"}:
            raise ValueError(
                f"coarse_only must be 'True', 'False', or 'both', not {coarse_only}"
            )
        self._coarse_only = coarse_only

        if level not in {"1", "2", "both"}:
            raise ValueError(f"level must be '1', '2', or 'both', not {level}")
        self._level = level

    def __call__(
        self, race_iterations_df: pd.DataFrame, ethnicity_iterations_df: pd.DataFrame
    ) -> pd.Series:
        """Return a series of iterations which met the filters.

        Args:
            race_iterations_df: `race_characteristic_iterations.txt.` data.
            ethnicity_iterations_df: `ethnicity_characteristic_iterations.txt` data.
        """
        if self._race_eth_type == "race":
            df = race_iterations_df
        elif self._race_eth_type == "ethnicity":
            df = ethnicity_iterations_df
        else:
            df = pd.concat(
                [race_iterations_df.drop(columns=["ALONE"]), ethnicity_iterations_df],
                ignore_index=True,
            )
        for column, value in [
            ("ALONE", self._alone),
            ("DETAILED_ONLY", self._detailed_only),
            ("COARSE_ONLY", self._coarse_only),
        ]:
            if value in ["True", "False"]:
                df = df[df[column] == value]
        if self._level in ["1", "2"]:
            df = df[df["LEVEL"] == self._level]
        return df["ITERATION_CODE"]


class IterationManager:
    """Helper class for managing race, ethnicity, and characteristic iteration codes."""

    def __init__(self, parameters_path: str, max_race_codes: int):
        """Constructor.

        Args:
            parameters_path: The path containing the config and race/ethnicity files.
            max_race_codes: The maximum number of race codes for any one record.
        """
        self._max_race_codes = max_race_codes
        self._race_iterations_df = read_csv(
            os.path.join(parameters_path, "race-characteristic-iterations.txt"),
            delimiter="|",
            dtype=str,
        )
        self._ethnicity_iterations_df = read_csv(
            os.path.join(parameters_path, "ethnicity-characteristic-iterations.txt"),
            delimiter="|",
            dtype=str,
        )
        # Ignore level 0 iterations
        self._race_iterations_df = self._race_iterations_df[
            self._race_iterations_df["LEVEL"] != "0"
        ]
        self._ethnicity_iterations_df = self._ethnicity_iterations_df[
            self._ethnicity_iterations_df["LEVEL"] != "0"
        ]
        self._race_eth_codes_to_iteration_df = read_csv(
            os.path.join(parameters_path, "race-and-ethnicity-code-to-iteration.txt"),
            delimiter="|",
            dtype=str,
        )
        # Store detailed only iterations for fast lookup.
        self._detailed_only_iterations = frozenset(
            self.get_iteration_codes(IterationFilter(detailed_only="True"))
        )

    def get_iteration_codes(self, iteration_filter: IterationFilter) -> pd.Series:
        """Return the list of iteration codes accepted by the given filter.

        Args:
            iteration_filter: An iteration filter specifying a subset of all possible
                characteristic iterations.
        """
        return iteration_filter(self._race_iterations_df, self._ethnicity_iterations_df)

    def get_race_eth_code_to_iterations(
        self, iteration_filter: IterationFilter
    ) -> Tuple[int, DefaultDict[str, Set[str]]]:
        """Return a mapping from an race/ethnicity code to characteristic iterations.

        Args:
            iteration_filter: An iteration filter specifying a subset of all possible
                characteristic iterations.

        Returns:
            A tuple containing

            * :math:`h`: The number of levels in the characteristic iteration
              hierarchy, restricted to the specified iterations.
            * A defaultdict from race/ethnicity code to characteristic iterations.
        """
        iterations_df = pd.DataFrame(
            iteration_filter(self._race_iterations_df, self._ethnicity_iterations_df)
        )
        df = self._race_eth_codes_to_iteration_df.merge(iterations_df)
        df = df.set_index("RACE_ETH_CODE")
        if len(df) == 0:
            h = 0
        else:
            h = max(df.index.value_counts())
        race_eth_code_to_iterations = defaultdict(
            set, df["ITERATION_CODE"].groupby(level=0).apply(set)
        )
        return h, race_eth_code_to_iterations

    def create_race_eth_codes_to_iterations(
        self,
        detailed_only: str = "both",
        coarse_only: str = "both",
        level: str = "both",
    ) -> Tuple[int, Callable[[Sequence[str], str], List[str]]]:
        """Return a function from race codes and an ethnicity code to iterations.

        Args:
            detailed_only: Whether to include only iteration codes where DETAILED_ONLY
                is equal to "True" or "False". Allowed options are "True", "False", and
                "both". If "True", coarse_only cannot also be "True".
            coarse_only: Whether to include only iteration codes where COARSE_ONLY
                is equal to "True" or "False". Allowed options are "True", "False", and
                "both". If "True", detailed_only cannot also be "True".
            level: Whether to include only iteration codes where LEVEL is equal to "1"
                or "2". Allowed options are "1", "2", and "both".

        Returns:
            A tuple containing

            * The sensitivity of the mapping (the maximum number of iterations
                possible to output). This is calculated based on the input files.
            * A function from race codes and an ethnicity code to iterations.
        """
        h_r, race_to_alone_iterations = self.get_race_eth_code_to_iterations(
            IterationFilter(
                race_eth_type="race",
                alone="True",
                detailed_only=detailed_only,
                coarse_only=coarse_only,
                level=level,
            )
        )
        (
            _,
            race_to_alone_or_in_combination_iterations,
        ) = self.get_race_eth_code_to_iterations(
            IterationFilter(
                race_eth_type="race",
                alone="False",
                detailed_only=detailed_only,
                coarse_only=coarse_only,
                level=level,
            )
        )
        h_e, ethnicity_to_iterations = self.get_race_eth_code_to_iterations(
            IterationFilter(
                race_eth_type="ethnicity",
                detailed_only=detailed_only,
                coarse_only=coarse_only,
                level=level,
            )
        )
        # This assumes that race_to_iterations_alone and
        # race_to_iterations_alone_or_in_combination have the exact same hierarchy
        # structure. See #91 for more information.
        sensitivity = h_r * max(2, self._max_race_codes) + h_e

        def race_eth_codes_to_iterations(
            race_codes: Sequence[str], ethnicity_code: str
        ) -> List[str]:
            """Return a list of characteristic iteration codes.

            Args:
                race_codes: A list of race codes corresponding to a record.
                ethnicity_code: The ethnicity code corresponding to a record.
            """
            if len(race_codes) > self._max_race_codes:
                raise ValueError(
                    f"At most max_race_codes={self._max_race_codes} race "
                    "codes can be provided."
                )
            alone_iterations: Optional[Set[str]] = None
            alone_or_in_combination_iterations: Set[str] = set()

            for race_code in race_codes:
                alone = race_to_alone_iterations[race_code]
                alone_or_in_combination = race_to_alone_or_in_combination_iterations[
                    race_code
                ]
                if alone_iterations is None:
                    alone_iterations = alone.copy()
                else:
                    alone_iterations &= alone
                alone_or_in_combination_iterations |= alone_or_in_combination

            ethnicity_iterations = ethnicity_to_iterations[ethnicity_code]

            # Assertion is to satisfy mypy typechecking. alone_iterations is only None
            # here if race_codes is empty, which should never happen.
            assert alone_iterations is not None
            all_iterations = list(
                alone_iterations
                | alone_or_in_combination_iterations
                | ethnicity_iterations
            )
            return all_iterations

        return sensitivity, race_eth_codes_to_iterations

    def create_add_iterations_flat_map(
        self,
        detailed_only: str = "both",
        coarse_only: str = "both",
        level: str = "both",
    ) -> FlatMapContainer:
        """Return a flat map function to add iterations to `person-records.txt`.

        The flat map adds one column called "ITERATION_CODE".

        The sensitivity of the flat map is :math:`h_r*max(2, r) + h_e`, where
        :math:`h_r` is the number of levels in the characteristic iteration
        hierarchy (restricted to indicated iteration type) for race (ethnicity for
        :math:`h_e`), and :math:`r` is the maximum number of race codes for an
        individual.

        Args:
            detailed_only: Whether to include only iteration codes where DETAILED_ONLY
                is equal to "True" or "False". Allowed options are "True", "False", and
                "both". If "True", coarse_only cannot also be "True".
            coarse_only: Whether to include only iteration codes where COARSE_ONLY
                is equal to "True" or "False". Allowed options are "True", "False", and
                "both". If "True", detailed_only cannot also be "True".
            level: Whether to include only iteration codes where LEVEL is equal to "1"
                or "2". Allowed options are "1", "2", and "both".

        Returns:
            A tuple containing

            * The sensitivity of the flat map.
            * A config for the output of the flat map.
            * The flat map described above.
        """
        (
            sensitivity,
            race_eth_codes_to_iterations,
        ) = self.create_race_eth_codes_to_iterations(
            detailed_only=detailed_only, coarse_only=coarse_only, level=level
        )

        def add_iterations_flat_map(row: Row) -> List[Row]:
            """Perform a flat map on a particular row from `person-records.txt`.

            Args:
                row: A row from `person-records.txt` as a namedtuple.
            """
            race_codes = [row[f"QRACE{i + 1}"] for i in range(self._max_race_codes)]
            race_codes = [race_code for race_code in race_codes if race_code != "Null"]
            ethnicity_code = row["QSPAN"]
            all_iteration_codes = race_eth_codes_to_iterations(
                race_codes, ethnicity_code
            )
            rows = []
            for iteration_code in all_iteration_codes:
                rows.append({"ITERATION_CODE": str(iteration_code)})
            return rows

        config = Config(
            [
                CategoricalStr(
                    "ITERATION_CODE",
                    self.get_iteration_codes(
                        IterationFilter(
                            detailed_only=detailed_only,
                            coarse_only=coarse_only,
                            level=level,
                        )
                    ),
                )
            ]
        )
        return FlatMapContainer(sensitivity, config, add_iterations_flat_map)

    def create_add_pop_groups_flat_map(
        self,
        region_type: str,
        region_domain: List[str],
        detailed_only: str = "both",
        coarse_only: str = "both",
        level: str = "both",
    ) -> FlatMapContainer:
        """Return a flat map function to add pop groups to `person-records.txt`.

        The flat map adds one column called "POP_GROUP". It has the format
        f"{region_id},{iteration_code}".

        The sensitivity of the flat map is :math:`h_r*max(2, r) + h_e`, where
        :math:`h_r` is the number of levels in the characteristic iteration
        hierarchy (restricted to indicated iteration type) for race (ethnicity for
        :math:`h_e`), and :math:`r` is the maximum number of race codes for an
        individual.

        Args:
            region_type: The region type column which is used to append region ids to
                iteration codes to get the population group.
            region_domain: The domain of the region type column.
            detailed_only: Whether to include only iteration codes where DETAILED_ONLY
                is equal to "True" or "False". Allowed options are "True", "False", and
                "both". If "True", coarse_only cannot also be "True".
            coarse_only: Whether to include only iteration codes where COARSE_ONLY
                is equal to "True" or "False". Allowed options are "True", "False", and
                "both". If "True", detailed_only cannot also be "True".
            level: Whether to include only iteration codes where LEVEL is equal to "1"
                or "2". Allowed options are "1", "2", and "both".

        Returns:
            A tuple containing

            * The sensitivity of the flat map.
            * A config for the output of the flat map.
            * The flat map described above.
        """
        (
            sensitivity,
            race_eth_codes_to_iterations,
        ) = self.create_race_eth_codes_to_iterations(
            detailed_only=detailed_only, coarse_only=coarse_only, level=level
        )

        def add_pop_groups_flat_map(row: Row) -> List[Row]:
            """Perform a flat map on a particular row from `person-records.txt`.

            Args:
                row: A row from `person-records.txt` as a namedtuple.
            """
            race_codes = [row[f"QRACE{i + 1}"] for i in range(self._max_race_codes)]
            race_codes = [race_code for race_code in race_codes if race_code != "Null"]
            ethnicity_code = row["QSPAN"]
            all_iteration_codes = race_eth_codes_to_iterations(
                race_codes, ethnicity_code
            )
            rows = []
            for iteration_code in all_iteration_codes:
                rows.append({"POP_GROUP": f"{row[region_type]},{str(iteration_code)}"})
            return rows

        config = Config(
            [
                CategoricalStr(
                    "POP_GROUP",
                    [
                        f"{region_id},{iteration_code}"
                        for iteration_code, region_id in itertools.product(
                            self.get_iteration_codes(
                                IterationFilter(
                                    detailed_only=detailed_only,
                                    coarse_only=coarse_only,
                                    level=level,
                                )
                            ),
                            region_domain,
                        )
                    ],
                )
            ]
        )
        return FlatMapContainer(sensitivity, config, add_pop_groups_flat_map)

    def is_detailed_only(self, iteration_code: str) -> bool:
        """Return True if the iteration code is detailed_only, otherwise return False.

        Args:
            iteration_code: The iteration code to test.
        """
        return iteration_code in self._detailed_only_iterations

    def get_iteration_df(self) -> pd.DataFrame:
        """Return the combined iteration df."""
        return pd.concat(
            [
                self._race_iterations_df.drop(columns=["ALONE"]),
                self._ethnicity_iterations_df,
            ],
            ignore_index=True,
        )
