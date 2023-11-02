"""Functions for postprocessing output files."""

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

from typing import List

import pandas as pd

T1_COLUMNS = ["REGION_ID", "REGION_TYPE", "ITERATION_CODE", "COUNT"]
"""Expected columns, in order, for t1."""

T2_COLUMNS = [
    "REGION_ID",
    "REGION_TYPE",
    "ITERATION_CODE",
    "AGESTART",
    "AGEEND",
    "SEX",
    "COUNT",
]
"""Expected columns, in order, for t2."""


def postprocess_dfs(dfs: List[pd.DataFrame]) -> pd.DataFrame:
    """Returns a postprocessed df matching the format of t2.

    This is used for postprocessing both t1 and t2 dataframes. To use this on t1
    dataframes, first add the missing columns (QAGE, QSEX), and then remove them
    from the dataframe returned by this function.

    Expected input format is a list of data frames with the following columns

    * REGION_ID: In the correct format
    * REGION_TYPE: In the correct format
    * ITERATION_CODE: In the correct format
    * QAGE: As a tuple (AGE_START, AGE_END + 1), or "*"
    * QSEX: In the correct format, but needs to be renamed to SEX
    * COUNT: In the correct format

    Args:
        dfs: List of data frames in expected input format.
    """
    if not dfs:
        return pd.DataFrame(columns=T2_COLUMNS)
    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df["AGESTART"] = "*"
    combined_df["AGEEND"] = "*"
    age_included_index = combined_df.index[combined_df["QAGE"] != "*"]
    age_included_df = pd.DataFrame(
        data=combined_df.loc[age_included_index, "QAGE"].values.tolist(),
        index=age_included_index,
        columns=["AGESTART", "AGEEND"],
    )
    age_included_df.loc[age_included_df["AGEEND"] != 115, "AGEEND"] -= 1
    age_included_df["AGESTART"] = age_included_df["AGESTART"].astype(str)
    age_included_df["AGEEND"] = age_included_df["AGEEND"].astype(str)
    combined_df.loc[age_included_index, ["AGESTART", "AGEEND"]] = age_included_df
    combined_df["SEX"] = combined_df["QSEX"]
    return combined_df[T2_COLUMNS]
