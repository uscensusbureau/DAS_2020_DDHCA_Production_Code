"""Unit tests for :mod:`tmlt.safetab_utils.utils`."""

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

# pylint: disable=no-self-use

import tempfile
import unittest
from typing import Type

from parameterized import parameterized
from typing_extensions import Literal

from tmlt.safetab_utils.csv_reader import CSVHReader, CSVPReader
from tmlt.safetab_utils.utils import file_empty, safetab_input_reader


class TestUtils(unittest.TestCase):
    """TestCase for :mod:`tmlt.safetab_utils.utils`."""

    @parameterized.expand(
        [("csv", "safetab-h", CSVHReader), ("csv", "safetab-p", CSVPReader)]
    )
    def test_safetab_input_reader(
        self,
        reader: str,
        program: Literal["safetab-h", "safetab-p"],
        reader_object: Type,
    ):
        """safetab_input_reader creates a reader object.

        Args:
            reader: String indicating the type of reader.
            program: Type of program running the reader.
            reader_object: Resulting reader object.
        """
        tempdir = tempfile.TemporaryDirectory()
        state_filter = ["01", "02", "11"]

        Reader = safetab_input_reader(
            reader=reader,
            data_path=tempdir.name,
            state_filter=state_filter,
            program=program,
        )
        self.assertTrue(isinstance(Reader, reader_object))

    @parameterized.expand([("xyz",), (10,)])
    def test_invalid_reader_config(self, reader: str):
        """safetab_input_reader errors on invalid reader type.

        Args:
            reader: String indicating the type of reader.
        """
        tempdir = tempfile.TemporaryDirectory()
        state_filter = ["01", "02", "11"]

        with self.assertRaisesRegex(
            ValueError, "Invalid config: 'reader' must be one of: csv cef"
        ):
            safetab_input_reader(
                reader=reader,
                data_path=tempdir.name,
                state_filter=state_filter,
                program="safetab-h",
            )

    @parameterized.expand([("", True), ("a", False)])
    def test_file_empty(self, content: str, result: bool):
        """file_empty returns true on empty file.

        Args:
            content: string to write to file.
            result: expected outcome of file_empty.
        """
        tmpfile = tempfile.TemporaryFile()
        if content:
            tmpfile.write(content.encode("utf-8"))
            tmpfile.seek(0)
        self.assertTrue(file_empty(tmpfile.name) == result)
