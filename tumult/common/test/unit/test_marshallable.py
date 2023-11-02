"""Tests Marshallable objects."""

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
import shutil
import tempfile
import unittest
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, Optional

import nose
import numpy as np
import pandas as pd
from parameterized import parameterized
from scipy import sparse

from tmlt.common.marshallable import Item, Marshallable


class ExampleClass(Marshallable):
    """Class to test Marshallable functionality."""

    @Marshallable.SaveInitParams
    def __init__(self, a: Any, b: Any, key1: Any = None, key2: Any = None):
        """Constructor."""
        self.a = a
        self.b = b
        self.keys = [key1, key2]

    def __eq__(self, other: Any) -> bool:
        """Return if other is equal to self.

        Args:
            other: Another object to compare against.
        """
        if not isinstance(other, type(self)):
            return False
        return self.a == other.a and self.b == other.b and self.keys == other.keys


@dataclass
class ExampleDataClass:
    """Dataclass to test Marshallable functionality."""

    my_int: int
    my_str: str
    my_nested_example_class: ExampleClass
    my_nested_example_dataclass: Optional["ExampleDataClass"]


def serde(item: Item) -> Item:
    """Return item after serializing and deserializing.

    Args:
        item: An Item.
    """
    return Marshallable.deserialize(Marshallable.serialize(item))


def save_load(test_class: Marshallable, filename: str) -> Item:
    """Return item after saving to and loading from a file.

    Args:
        test_class: A marshallable object.
        filename: A file to use for marshalling and unmarshalling.
    """
    test_class.save_json(filename)
    return Marshallable.load_json(filename)


def assert_dictionary_equal(test_case: unittest.TestCase, a: Dict, b: Dict):
    """Assert that two dictionaries are equal, checked recursively.

    Args:
        test_case: The test case to fail if a and b are not equal.
        a: A dictionary.
        b: A dictionary.
    """
    test_case.assertEqual(len(a), len(b))
    for key, value in a.items():
        test_case.assertIn(key, b)
        other_value = b[key]
        if isinstance(value, dict) and isinstance(other_value, dict):
            assert_dictionary_equal(test_case, value, other_value)
        else:
            test_case.assertEqual(value, other_value)


class TestMarshallable(unittest.TestCase):
    """Class to test Marshallable objects."""

    def setUp(self):
        """Setup test."""
        self.A = ExampleClass(a=3, b="5", key2="c")
        self.tmp_dir = tempfile.mkdtemp()
        self.tmp_filename = os.path.join(self.tmp_dir, "test.json")

    def tearDown(self):
        """Tear down test."""
        shutil.rmtree(self.tmp_dir)

    @parameterized.expand(
        [
            (13,),
            (True,),
            (3.14,),
            ("test",),
            (u"unicode",),
            (OrderedDict({"a": 1, "b": 2}),),
            ((3, 2),),
            (["a", "b"],),
            (None,),
        ]
    )
    def test_json_primitives(self, item: Item):
        """Tests serializing and deserializing primitives.

        Args:
            item: An Item.
        """
        self.assertEqual(item, serde(item))

    def test_dtype(self):
        """Tests serializing and deserializing numpy data type objects."""
        self.assertEqual(np.dtype("float64"), serde(np.dtype("float64")))

    def test_json_numpy_array(self):  # pylint: disable=no-self-use
        """Tests serializing and deserializing numpy array."""
        np.testing.assert_array_equal(np.ones((3, 2)), serde(np.ones((3, 2))))

    def test_scipy_sparse(self):
        """Tests serializing and deserializing scipy matrix."""
        actual = serde(sparse.eye(3))
        expected = sparse.eye(3)
        self.assertEqual(actual.format, expected.format)
        self.assertEqual(actual.dtype, expected.dtype)
        self.assertEqual(actual.shape, expected.shape)

        # The L1 norm of the different elements is 0
        self.assertEqual(abs(actual != expected).sum(), 0)

    def test_function(self):
        """Tests serializing and deserializing functions."""

        def times2(arg):
            return 2 * arg

        actual = serde(times2)(5)
        expected = 10
        self.assertEqual(actual, expected)

    def test_json_marshallable_class(self):
        """Tests serializing and deserializing persistable data."""
        self.assertEqual(self.A, serde(self.A))

    def test_json_mixed(self):
        """Tests serializing and deserializing json with mixed types."""
        d = {"a": 13, "5": 3.14, "(1, 2)": (3, 2), "b": [1, 3]}
        assert_dictionary_equal(self, d, serde(d))

    def test_json_hierarchical(self):
        """Tests serializing and deserializing nested structures."""
        d = {
            "a": self.A,
            "b": {"b1": [((self.A, [3, 1]),), "b", 3], "b2": {"b21": self.A}},
            "c": (self.A,),
        }
        assert_dictionary_equal(self, d, serde(d))

    def test_save_load(self):
        """Tests save and load functionality."""
        c1 = ExampleClass("a", 3)
        self.assertEqual(c1, save_load(c1, self.tmp_filename))

        c2 = ExampleClass((2, "3"), 3, key1="b", key2=4)
        self.assertEqual(c2, save_load(c2, self.tmp_filename))

    @parameterized.expand(
        [
            (
                pd.DataFrame(
                    {"A": [1, 2, 3], "B": [1.0, 2.0, 3.0], "C": ["A", "B", "C"]}
                ),
            ),
            (
                pd.DataFrame(
                    {"A": [1, 2, None], "B": ["Xyz", "BCA", None], "D": [3, 4, 5]},
                    index=["A", "B", "C"],
                ),
            ),
            (
                pd.DataFrame(
                    {
                        "A": pd.Series([1, 2, 3], dtype=np.int32),
                        "B": pd.Series(["A", "b", "C"], dtype=object),
                        "C": pd.Series(["Z1", "Z2", "Z3"], dtype=np.str_),
                    }
                ),
            ),
        ]
    )
    def test_pd_dataframe(self, df: pd.DataFrame):
        """Tests serializing and deserializing pandas DataFrames."""
        # pylint: disable=no-self-use
        pd.testing.assert_frame_equal(df, serde(df), check_exact=True)

    def test_dataclass(self):
        """Tests serializing and deserializing dataclasses."""
        a = ExampleDataClass(
            1,
            "2",
            ExampleClass(3, "4"),
            ExampleDataClass(5, "6", ExampleClass(7, "8"), None),
        )
        self.assertEqual(a, serde(a))

    @nose.tools.raises(Exception)  
    def test_dictionary_non_string_keys(self):
        """Dictionaries can use non-string keys.

        Note that json cannot use non-string keys for dictionaries.
        """
        expected = {(1, 2): (3, 5), 5: 3, 4: {"str": 4, 5: 3}}
        actual = serde(expected)
        assert_dictionary_equal(self, actual, expected)

    @nose.tools.raises(Exception)  
    def test_dictionary_bad_keys(self):
        """Dictionaries can use keys that could be mistaken for other types."""
        expected = {
            "class": 'Not a marshalled class, just using "class": as a key.',
            "sparse": "Not a scipy.sparse.spmatrix either.",
            "tuple": "Or a tuple.",
        }
        actual = serde(expected)
        assert_dictionary_equal(self, actual, expected)
