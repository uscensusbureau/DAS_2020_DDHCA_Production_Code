"""Tests SafeTab csv reader on toy data."""

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

# pylint: disable=line-too-long

import os
import shutil
import tempfile

from tmlt.core.utils.testing import PySparkTest
from tmlt.safetab_utils.csv_reader import (
    GEO_FILENAME,
    PERSON_FILENAME,
    POP_GROUP_TOTAL_FILENAME,
    UNIT_FILENAME,
    CSVHReader,
    CSVPReader,
)


class TestCSVReader(PySparkTest):
    """Parameterized unit tests for csv reader."""

    def setUp(self):
        """Set up test."""
        self.tmp_dir = tempfile.mkdtemp()

        geo = """TABBLKST|TABBLKCOU|TABTRACTCE|PLACEFP|TABBLK|AIANNHCE\n
01|001|000001|22800|0001|0001\n
72|031|050901|99999|2040|9999\n
02|002|000002|99999|0001|9999\n
11|001|000100|99999|3004|9999"""

        person = """QAGE|QSEX|HOUSEHOLDER|TABBLKST|TABBLKCOU|TABTRACTCE|TABBLK|CENRACE|QRACE1|QRACE2|QRACE3|QRACE4|QRACE5|QRACE6|QRACE7|QRACE8|QSPAN\n
028|1|True|01|001|000001|0001|01|1000|Null|Null|Null|Null|Null|Null|Null|1010\n
079|1|True|72|031|050901|2040|01|1340|Null|Null|Null|Null|Null|Null|Null|5230\n
073|1|True|11|001|000100|3004|01|3310|5454|2322|Null|Null|Null|Null|Null|7680"""

        unit = """TABBLKST|TABBLKCOU|TABTRACTCE|TABBLK|HHRACE|QRACE1|QRACE2|QRACE3|QRACE4|QRACE5|QRACE6|QRACE7|QRACE8|QSPAN|HOUSEHOLD_TYPE|TEN\n
01|001|000001|0002|63|1011|1012|1031|Null|Null|Null|Null|Null|1010|5|0\n
72|031|050901|2040|01|1340|Null|Null|Null|Null|Null|Null|Null|5230|1|1\n
11|001|000100|3004|11|3310|5454|2322|Null|Null|Null|Null|Null|7680|2|1"""

        pop_group = """REGION_ID|REGION_TYPE|ITERATION_CODE|COUNT\n
1|USA|1234|100\n
72|STATE|7394|50\n
11|STATE|2048|74\n
016068476340|TRACT|2937|23"""

        geo_filename = os.path.join(self.tmp_dir, GEO_FILENAME)
        unit_filename = os.path.join(self.tmp_dir, UNIT_FILENAME)
        person_filename = os.path.join(self.tmp_dir, PERSON_FILENAME)
        pop_group_filename = os.path.join(self.tmp_dir, POP_GROUP_TOTAL_FILENAME)

        with open(geo_filename, "w") as f:
            f.write(geo)
        with open(unit_filename, "w") as f:
            f.write(unit)
        with open(person_filename, "w") as f:
            f.write(person)
        with open(pop_group_filename, "w") as f:
            f.write(pop_group)

        self.reader_p = CSVPReader(self.tmp_dir, ["01", "72"])
        self.reader_h = CSVHReader(self.tmp_dir, ["01", "72"])

    def test_read_geo_df(self):
        """Load geo df."""
        df = self.reader_p.get_geo_df()
        self.assertEqual(df.count(), 2)

    def test_read_unit_df(self):
        """Load unit df."""
        df = self.reader_h.get_unit_df()
        self.assertEqual(df.count(), 2)

    def test_read_person_df(self):
        """Load person df."""
        df = self.reader_p.get_person_df()
        self.assertEqual(df.count(), 2)

    def test_state_filter(self):
        """Test filter."""
        reader_h = CSVHReader(self.tmp_dir, ["11"])
        geo_h = reader_h.get_geo_df()
        unit_h = reader_h.get_unit_df()
        pop_group = reader_h.get_pop_group_details_df()
        self.assertEqual(geo_h.count(), 1)
        self.assertEqual(unit_h.count(), 1)
        self.assertEqual(pop_group.count(), 2)
        reader_p = CSVPReader(self.tmp_dir, ["11"])
        geo_h = reader_p.get_geo_df()
        person_h = reader_p.get_person_df()
        self.assertEqual(geo_h.count(), 1)
        self.assertEqual(person_h.count(), 1)

    def tearDown(self) -> None:
        """Cleans up temp directory."""
        shutil.rmtree(self.tmp_dir)
