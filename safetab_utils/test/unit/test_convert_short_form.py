"""Unit test for :mod:`convert_short_form`"""

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
import tempfile
import unittest

import pandas as pd

from tmlt.safetab_utils.convert_short_form import (
    convert_short_form,
    validate_short_form,
)


class TestConvertShortForm(unittest.TestCase):
    """Test short form to long form conversion."""

    def setUp(self):
        input_file, self.input_file_name = tempfile.mkstemp()
        os.close(input_file)
        output_file, self.output_file_name = tempfile.mkstemp()
        os.close(output_file)
        race_codes_file, self.race_codes_file_name = tempfile.mkstemp()
        os.close(race_codes_file)

    def tearDown(self):
        os.remove(self.input_file_name)
        os.remove(self.output_file_name)

    def test_convert_short_form(self):
        """Test short form to long form conversion."""
        with open(self.input_file_name, mode="w") as f:
            f.write("ITERATION_CODE|RACE_ETH_CODE\n1|  1 \n2 |1:2,  3:4   \n3|1:2, 4\n")
        with open(self.race_codes_file_name, mode="w") as f:
            f.write("RACE_ETH_CODE|RACE_ETH_NAME\n1|a\n2|b\n4|d\n")

        convert_short_form(
            self.input_file_name, self.race_codes_file_name, self.output_file_name
        )
        df1 = pd.read_csv(self.output_file_name, sep="|")
        df2 = pd.DataFrame(
            [[1, 1], [2, 1], [2, 2], [2, 4], [3, 1], [3, 2], [3, 4]],
            columns=["ITERATION_CODE", "RACE_ETH_CODE"],
        )
        pd.testing.assert_frame_equal(left=df1, right=df2)

    def test_validate_short_form1(self):
        """Test incorrect short form format throws error."""
        with open(self.input_file_name, mode="w") as f:
            f.write("ITERATION_CODE|RACE_ETH_CODE\n1|  1000 \n2 |1000:200  \n")
        with self.assertRaises(ValueError):
            validate_short_form(self.input_file_name)

    def test_validate_short_form2(self):
        """Test incorrect short form format throws error."""
        with open(self.input_file_name, mode="w") as f:
            f.write("ITERATION_CODE|RACE_ETH_CODE\n1|  1000 \n2 |1000 2000  \n")
        with self.assertRaises(ValueError):
            validate_short_form(self.input_file_name)


if __name__ == "__main__":
    unittest.main()
