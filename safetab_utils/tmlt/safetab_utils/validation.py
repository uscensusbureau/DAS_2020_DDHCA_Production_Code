"""Validates input and output files of SafeTab-P/-H."""

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

import functools
import logging
from typing import Sequence

from pyspark.sql.dataframe import DataFrame
from pyspark.sql.functions import col, lit


def check_pop_group_for_invalid_states(
    sdf: DataFrame, state_filter: Sequence[str]
) -> bool:
    """Return False if any records in population groups have invalid states.

    Rules checked:

    * "USA" and "AIANNH" are always included.
    * Otherwise, the prefix of the REGION_ID must be in state_filter.

    See `pop-group-totals.txt` for more information.

    Args:
        sdf: The input dataframe.
        state_filter: The list of states to filter on.
    """
    if len(state_filter) == 0:
        return True
    num_invalid_states = sdf.filter(
        ~col("REGION_TYPE").isin(["USA", "AIANNH"])
        & ~functools.reduce(
            lambda x, y: x | y,
            [col("REGION_ID").startswith(state) for state in state_filter],
            lit(False),
        )
    ).count()
    if num_invalid_states > 0:
        logging.getLogger(__name__).error(
            "Invalid input: Input does not have state filters applied."
        )
        return False
    return True
