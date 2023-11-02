"""Unit tests for :mod:`tmlt.common.error`."""

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

# pylint: disable=no-member, no-self-use
import os
import tempfile
import unittest
from unittest.mock import Mock

import pandas as pd
from parameterized import parameterized

from tmlt.common.io_helpers import is_s3_path, multi_read_csv, to_csv_with_create_dir


class TestIOHelpers(unittest.TestCase):
    """TestCase for :mod:`tmlt.common.io_helpers`."""

    @parameterized.expand(
        [
            ("s3://example/path/", True),
            ("s3://example/path/to/file.txt", True),
            ("s3a://example/path/starting/with/s3a", True),
            ("example/absolute/path/", False),
            ("example/relative/path/to/file.txt", False),
            ("http://www.example.com/", False),
        ]
    )
    def test_is_s3_path(self, path: str, expected: bool):
        """is_s3_path correctly classifies paths.

        Args:
            path: A string representing a path.
            expected: Whether the path is an s3 path.
        """
        self.assertEqual(is_s3_path(path), expected)

    @parameterized.expand([("output.csv",), ("a/output.csv",), ("a/b/output.csv",)])
    def test_to_csv_with_create_dir(self, filename: str):
        """to_csv_with_create_dir can create any number of new directories.

        Args:
            filename: The file name to create. Includes a prefix of directories
                that need to be created.
        """
        tempdir = tempfile.TemporaryDirectory()
        path = os.path.join(tempdir.name, filename)
        expected_df = pd.DataFrame({"A": ["a1", "a2", "a3"]})
        to_csv_with_create_dir(expected_df, path, sep=".", index=False)
        actual_df = pd.read_csv(path, sep=".")
        pd.testing.assert_frame_equal(actual_df, expected_df)

    def test_to_csv_with_create_dir_doesnt_create_s3_dirs(self):
        """to_csv_with_create_dir doesn't create directories if given a path to s3."""
        path = "s3://example/path"
        mock_df = Mock()
        to_csv_with_create_dir(mock_df, path)
        mock_df.to_csv.assert_called_with(path)
        self.assertFalse(os.path.isdir("s3:"))

    @parameterized.expand([(1,), (2,), (3,)])
    def test_multi_read_csv_multiple_files(self, n_files: int):
        """multi_read_csv works with 1 or more files.

        Args:
            n_files: the number of files to store in the directory.
        """
        tempdir = tempfile.TemporaryDirectory()
        df = pd.DataFrame({"A": ["a1", "a2", "a3"], "B": ["b1", "b2", "b3"]})
        for i in range(n_files):
            filename = os.path.join(tempdir.name, f"{i}.csv")
            df.to_csv(
                filename, index=False, sep="j"
            )  # sep="j" is to test passing kwargs
        expected_df = pd.concat([df] * n_files).reset_index(drop=True)
        actual_df = multi_read_csv(tempdir.name, sep="j")
        pd.testing.assert_frame_equal(actual_df, expected_df)

    def test_multi_read_csv_ignores_empty_and_non_csv(self):
        """multi_read_csv skips files that are empty, or don't end with .csv"""
        tempdir = tempfile.TemporaryDirectory()
        df = pd.DataFrame({"A": ["a1", "a2", "a3"], "B": ["b1", "b2", "b3"]})
        df.to_csv(os.path.join(tempdir.name, "bad-extension.txt"), index=False)
        expected_df = df.rename(columns={"A": "C", "B": "D"})
        expected_df.to_csv(os.path.join(tempdir.name, "valid.csv"), index=False)

        with open(os.path.join(tempdir.name, "empty.csv"), "w"):
            pass

        expected_directory_contents = {"bad-extension.txt", "empty.csv", "valid.csv"}
        actual_directory_contents = set(os.listdir(tempdir.name))
        self.assertEqual(actual_directory_contents, expected_directory_contents)

        actual_df = multi_read_csv(tempdir.name)
        pd.testing.assert_frame_equal(actual_df, expected_df)
