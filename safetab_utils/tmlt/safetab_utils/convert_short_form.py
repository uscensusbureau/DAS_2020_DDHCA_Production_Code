"""Module for converting short form iteration mapping to long form.

`Appendix A` contains details on the long form iteration mapping format
(`race-and-ethnicity-code-to-iteration.txt`). The details on the short form
format appear in `Appendix F`.
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

# Experimenting with black formatter
# (See https://gitlab.com/tumult-labs/tumult/-/issues/158)
# pylint: disable=bad-continuation

import argparse
import re
from typing import List

import pandas as pd


def validate_short_form(input_name: str):
    """Validate the short form iteration mapping file.

    This function implements the grammar defined in `Appendix F`. It validates
    every line of the input file, excluding the first line, and raises an
    exception if a line fails to validate.

    Args:
        input_name: Short form iteration mapping file name.
    """
    race_code = r"[0-9]{4}"
    iteration_code = r"[ ]*" + r"[0-9]{1,4}" + r"[ ]*"
    race_code_range = r"(?:" + race_code + r":" + race_code + r")"
    race_code_term = r"(?:" + race_code + r"|" + race_code_range + r")"
    race_code_list = (
        r"(?:"
        + r"(?:"
        + r"[ ]*"
        + race_code_term
        + r")"
        + r"(?:"
        + r","
        + r"[ ]*"
        + race_code_term
        + r")*"
        + r"[ ]*"
        + r")"
    )
    iteration_mapping = r"\A" + iteration_code + r"\|" + race_code_list + r"\n\Z"

    pattern = re.compile(iteration_mapping)
    with open(input_name, "r") as f:
        next(f)
        for line_num, line in enumerate(f):
            match = pattern.match(line)
            if match is None:
                raise ValueError(
                    f"Line {line_num+2} in {input_name} does not match the spec: {line}"
                )


def convert_short_form(input_name: str, race_codes_name: str, output_name: str):
    """Convert short form iteration mapping to long form.

    The short form mapping uses ranges in which not all race codes are valid,
    e.g. 1000:1999 even if all of these are not valid race codes. When we
    produce the long form file, we remove race codes that are not found in the
    races codes file.

    Args:
        input_name: Short form iteration mapping file name.
        race_codes_name: Name of file with list of races codes.
        output_name: Long form iteration mapping file name.

    """
    short_form_df = pd.read_csv(
        input_name, sep="|", dtype={"ITERATION_CODE": str, "RACE_ETH_CODE": str}
    )

    race_code_df = pd.read_csv(race_codes_name, sep="|", dtype=str)

    def convert_row(iteration_code: str, qrace_values: str) -> pd.DataFrame:
        iteration_code = iteration_code.strip()
        code_terms = qrace_values.split(sep=",")
        code_terms = [x.strip() for x in code_terms]
        codes: List[int] = []
        for code_term in code_terms:
            if ":" in code_term:
                code_range = code_term.split(sep=":")
                codes += list(range(int(code_range[0]), int(code_range[1]) + 1))
            else:
                codes += [int(code_term)]
        return pd.DataFrame(
            [
                {"ITERATION_CODE": iteration_code, "RACE_ETH_CODE": race_code}
                for race_code in codes
            ]
        )

    long_form_df = pd.concat(
        [
            convert_row(iteration_code, qrace_values)
            for iteration_code, qrace_values in zip(
                short_form_df["ITERATION_CODE"], short_form_df["RACE_ETH_CODE"]
            )
        ]
    )

    # remove race codes not in the domain
    long_form_df = long_form_df[
        long_form_df["RACE_ETH_CODE"].isin(race_code_df["RACE_ETH_CODE"].astype(int))
    ]
    long_form_df.to_csv(output_name, index=False, sep="|")


def main():
    """Parse command line arguments and run :func:`convert_short_form`."""

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--short-form",
        dest="input_name",
        help="Name of input short-form race to iteration file.",
        required=True,
    )
    parser.add_argument(
        "--race-codes",
        dest="race_codes_name",
        help="Name of file with race codes.",
        required=True,
    )

    parser.add_argument(
        "--output",
        dest="output_name",
        help="Name of destination long-form race to iteration file.",
        required=True,
    )

    args = parser.parse_args()
    validate_short_form(args.input_name)
    convert_short_form(args.input_name, args.race_codes_name, args.output_name)


if __name__ == "__main__":
    main()
