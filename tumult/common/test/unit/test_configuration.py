"""Unit tests for :mod:`~tmlt.common.configuration`."""

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

# pylint: skip-file

import re
import unittest
from typing import Sequence

import numpy as np
import pandas as pd
from parameterized import parameterized

from tmlt.common.configuration import (
    CategoricalInt,
    CategoricalStr,
    Config,
    Continuous,
    Discrete,
    Splits,
    Unrestricted,
)


class TestUnrestricted(unittest.TestCase):
    """TestCase for :class:`~tmlt.common.configuration.Unrestricted`."""

    def setUp(self):
        """Setup."""
        self.attribute = Unrestricted("test")
        self.attribute_regex = Unrestricted("test", r"-?\d\d\d")
        self.number_attribute_regex = Unrestricted("test", r"-?\d+(\.\d+)?")
        self.native_values = pd.Series(["000", "001", "002"])
        self.invalid_values = pd.Series(["0", "001", "002"])

    def test_properties(self):
        """Unrestricted attribute properties are as expected."""
        self.assertEqual(self.attribute.column, "test")
        self.assertEqual(self.attribute._pattern, None)
        self.assertEqual(self.attribute_regex._pattern, r"-?\d\d\d")

    def test_domain_not_implemented(self):
        """Unrestricted attribute is without a domain."""
        with self.assertRaises(NotImplementedError):
            self.attribute.domain

    def test_validate_no_regex(self):
        """Vaidate returns expected output for valid attribute values."""
        actual = self.attribute.validate(self.native_values)
        expected = self.native_values
        pd.testing.assert_series_equal(actual, expected)

    def test_validate_regex_raise(self):
        """Validate raises ValueError for values not matching regex."""
        with self.assertRaises(ValueError):
            self.attribute_regex.validate(self.invalid_values)

    def test_validate_regex_drop(self):
        """Validate remove drops rows with values not matching regex."""
        actual = self.attribute_regex.validate(
            self.invalid_values, out_of_bounds_strategy="remove"
        )
        expected = pd.Series(["001", "002"], index=[1, 2])
        pd.testing.assert_series_equal(actual, expected)

    def test_validate_regex_okay(self):
        """Vaidate returns expected output for values matching regex."""
        actual = self.attribute_regex.validate(self.native_values)
        expected = self.native_values
        pd.testing.assert_series_equal(actual, expected)

    def test_validate_no_partial_match(self):
        """Validate should raise or drop if the input only partially matches
        the regex."""
        three_digit_numbers = Unrestricted("test", r"\d\d\d")
        invalid_value = pd.Series(["1234"])
        with self.assertRaises(ValueError):
            three_digit_numbers.validate(invalid_value)
        self.assertTrue(
            three_digit_numbers.validate(
                invalid_value, out_of_bounds_strategy="remove"
            ).empty
        )

    @parameterized.expand(
        [
            (pd.Series([100.1, -101, 200]), pd.Series(["100.1", "-101.0", "200.0"])),
            (pd.Series([100, -101, 200]), pd.Series(["100", "-101", "200"])),
        ]
    )
    def test_validate_cast_values(self, input_values, expected):
        """Validate returns expected output for valid attribute values after casting."""
        actual = self.number_attribute_regex.validate(input_values)
        pd.testing.assert_series_equal(actual, expected)

    def test_standardize_not_implemented(self):
        """Unrestricted.standardize raises NotImplementedError."""
        with self.assertRaises(NotImplementedError):
            self.attribute.standardize(self.native_values)

    def test_unstandardize_not_implemented(self):
        """Unrestricted.unstandardize raises NotImplementedError."""
        with self.assertRaises(NotImplementedError):
            self.attribute.unstandardize(pd.Series([0, 1]))

    def test_equality(self):
        """Equality on Unrestricted attribute is as expected."""
        self.assertEqual(self.attribute, self.attribute)
        self.assertEqual(self.attribute, Unrestricted("test"))

    def test_repr(self):
        """String representation is as expected."""
        actual = repr(self.attribute)
        expected = "Unrestricted('test')"
        self.assertEqual(actual, expected)


class TestContinuous(unittest.TestCase):
    """TestCase for :class:`~tmlt.common.configuration.Continuous`."""

    def setUp(self):
        """Setup."""
        super().setUp()
        self.native_values = pd.Series([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=float)
        self.standardized_values = pd.Series([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
        self.unstandardized_values = pd.Series(
            [
                (0, 2),
                (0, 2),
                (2, 4),
                (2, 4),
                (4, 6),
                (4, 6),
                (6, 8),
                (6, 8),
                (8, 10),
                (8, 10),
            ]
        )
        self.invalid_values = pd.Series([-1, 0, 1, 100], dtype=float)
        self.attribute = Continuous("test", 5, [0, 10])

    def test_invalid_native_domain(self):
        """Raises ValueError for domain with min value greater than max."""
        msg = (
            "Minimum (10.0) must be smaller than maximum (0.0) "
            "(Continuous('test', 5, [10.0, 0.0]))"
        )
        msg = re.escape(msg)
        with self.assertRaisesRegex(ValueError, msg):
            Continuous("test", 5, [10, 0])

    def test_invalid_bin(self):
        """Continuous raises ValueError for zero bin size."""
        msg = (
            "Attribute domain must be an integer greater than or equal to "
            "1, not 0 (Continuous('test', 0, [0.0, 10.0]))"
        )
        msg = re.escape(msg)
        with self.assertRaisesRegex(ValueError, msg):
            Continuous("test", 0, [0, 10])

    def test_properties(self):
        """Continuous attribute properties are as expected."""
        self.assertEqual(self.attribute.column, "test")
        self.assertEqual(self.attribute.minimum, 0)
        self.assertEqual(self.attribute.maximum, 10)
        self.assertEqual(self.attribute.domain, 5)

    def test_standardize(self):
        """Standardize on Continuous attribute returns the expected values."""
        actual = self.attribute.standardize(self.native_values)
        expected = self.standardized_values
        pd.testing.assert_series_equal(actual, expected)

    def test_unstandardize(self):
        """Unstandardize on Continuous attribute returns the expected values."""
        actual = self.attribute.unstandardize(self.standardized_values)
        expected = self.unstandardized_values
        pd.testing.assert_series_equal(actual, expected)

    def test_validate_error(self):
        """Validate error strategy raises ValueError for invalid values."""
        msg = (
            f"Found out of bounds elements in {self.attribute}\n0     -1.0\n3    100.0"
        )
        msg = re.escape(msg)
        with self.assertRaisesRegex(ValueError, msg):
            self.attribute.validate(self.invalid_values, out_of_bounds_strategy="error")

    def test_validate_remove(self):
        """Validate remove strategy drops rows with invalid values."""
        actual = self.attribute.validate(
            self.invalid_values, out_of_bounds_strategy="remove"
        )
        expected = pd.Series([0, 1], index=[1, 2], dtype=float)
        pd.testing.assert_series_equal(actual, expected)

    def test_validate_clip(self):
        """Validate clip strategy returns expected values."""
        actual = self.attribute.validate(
            self.invalid_values, out_of_bounds_strategy="clip"
        )
        expected = pd.Series([0, 0, 1, 10], dtype=float)
        pd.testing.assert_series_equal(actual, expected)

    def test_validate_unknown(self):
        """Validate unknown strategy raises ValueError."""
        msg = 'Unknown out of bounds strategy "unknown"'
        with self.assertRaisesRegex(ValueError, msg):
            self.attribute.validate(
                self.invalid_values, out_of_bounds_strategy="unknown"
            )

    def test_equality(self):
        """Equality on Continuous attribute is as expected."""
        self.assertEqual(self.attribute, Continuous("test", 5, [0, 10]))
        self.assertNotEqual(self.attribute, Continuous("test", 5, [0, 11]))
        self.assertNotEqual(self.attribute, Continuous("test2", 5, [0, 10]))
        self.assertNotEqual(self.attribute, Splits("test", [0, 2, 4, 6, 8, 10]))

    def test_repr(self):
        """String representation is as expected."""
        actual = repr(self.attribute)
        expected = "Continuous('test', 5, [0.0, 10.0])"
        self.assertEqual(actual, expected)


class TestDiscrete(unittest.TestCase):
    """TestCase for :class:`~tmlt.common.configuration.Discrete`."""

    def test_invalid_domain(self):
        """Discrete raises ValueError for invalid domain."""
        msg = (
            "There must be an integer number of bins for each value in the "
            "domain for discrete columns (Discrete('test', 5, [-10, 11])"
        )
        msg = re.escape(msg)
        with self.assertRaisesRegex(ValueError, msg):
            Discrete("test", 5, [-10, 11])

    def test_repr(self):
        """String representation is as expected."""
        actual = repr(Discrete("test", 5, [0, 10]))
        expected = "Discrete('test', 5, [0, 10])"
        self.assertEqual(actual, expected)


class TestSplits(unittest.TestCase):
    """TestCase for :class:`~tmlt.common.configuration.Splits`."""

    def setUp(self):
        """Setup."""
        super().setUp()
        self.native_values = pd.Series([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
        self.standardized_values = pd.Series([0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 3])
        self.unstandardized_values = pd.Series(
            [
                (0, 4),
                (0, 4),
                (0, 4),
                (0, 4),
                (4, 6),
                (4, 6),
                (6, 8),
                (6, 8),
                (8, 10),
                (8, 10),
                (8, 10),
            ]
        )
        self.invalid_values = pd.Series([-1, 0, 1, 100], dtype=float)
        self.attribute = Splits("test", [0, 4, 6, 8, 10])

    def test_invalid_splits(self):
        """Splits raises ValueError for incorrectly sorted values."""
        msg = (
            "Splits should be sorted in ascending order. "
            "(Splits('test', [1, 2, 3, 4, 0]))"
        )
        msg = re.escape(msg)
        with self.assertRaisesRegex(ValueError, msg):
            Splits("test", [1, 2, 3, 4, 0])

    def test_invalid_domain(self):
        """Splits raises ValueError for invalid domain."""
        msg = (
            "Attribute domain must be an integer greater than or equal to "
            "1, not 0 (Splits('test', [0]))"
        )
        msg = re.escape(msg)
        with self.assertRaisesRegex(ValueError, msg):
            Splits("test", [0])

    def test_validate(self):
        """Splits.validate correctly validates a series of values in the domain."""
        self.attribute.validate(self.native_values)

    def test_properties(self):
        """Splits attribute properties are as expected."""
        self.assertEqual(self.attribute.column, "test")
        self.assertEqual(self.attribute.minimum, 0)
        self.assertEqual(self.attribute.maximum, 10)
        self.assertEqual(self.attribute.domain, 4)

    def test_standardize(self):
        """Standardize on Splits attribute returns the expected values."""
        actual = self.attribute.standardize(self.native_values)
        expected = self.standardized_values
        pd.testing.assert_series_equal(actual, expected)

    def test_unstandardize(self):
        """Unstandardize on Splits attribute returns the expected values."""
        actual = self.attribute.unstandardize(self.standardized_values)
        expected = self.unstandardized_values
        pd.testing.assert_series_equal(actual, expected)

    def test_equality(self):
        """Equality on Splits attribute is as expected."""
        self.assertEqual(self.attribute, Splits("test", [0, 4, 6, 8, 10]))
        self.assertNotEqual(self.attribute, Splits("test", [0, 3, 6, 8, 10]))
        self.assertNotEqual(self.attribute, Splits("test2", [0, 4, 6, 8, 10]))
        self.assertNotEqual(self.attribute, Continuous("test", 4, [0, 10]))

    def test_repr(self):
        """String representation is as expected."""
        actual = repr(self.attribute)
        expected = "Splits('test', [0, 4, 6, 8, 10])"
        self.assertEqual(actual, expected)


class TestCategoricalStr(unittest.TestCase):
    """TestCase for :class:`~tmlt.common.configuration.CategoricalStr`."""

    def setUp(self):
        """Setup."""
        super().setUp()
        self.unstandardized_values = pd.Series(["A", "B", "C", "D", "E"])
        self.standardized_values = pd.Series([0, 1, 2, 3, 4])
        self.invalid_values = pd.Series(["A", "B", "F", "D"])
        self.attribute = CategoricalStr("test", ["A", "B", "C", "D", "E"])

    def test_invalid_domain(self):
        """CategoricalStr raises ValueError for invalid domain."""
        msg = (
            "Attribute domain must be an integer greater than or equal to "
            "1, not 0 (CategoricalStr('test', []))"
        )
        msg = re.escape(msg)
        with self.assertRaisesRegex(ValueError, msg):
            CategoricalStr("test", [])

    def test_properties(self):
        """CategoricalStr attribute properties are as expected."""
        self.assertEqual(self.attribute.column, "test")
        self.assertEqual(self.attribute.domain, 5)
        self.assertEqual(self.attribute.dtype, str)

    def test_standardize(self):
        """Standardize on CategoricalStr attribute returns the expected values."""
        actual = self.attribute.standardize(self.unstandardized_values)
        expected = self.standardized_values
        pd.testing.assert_series_equal(actual, expected)

    def test_unstandardize(self):
        """Unstandardize on CategoricalStr attribute returns the expected values."""
        actual = self.attribute.unstandardize(self.standardized_values)
        expected = self.unstandardized_values
        pd.testing.assert_series_equal(actual, expected)

    def test_validate_error(self):
        """Validate error strategy raises ValueError for invalid values."""
        msg = f"Found unexpected values in {self.attribute}\n2    F"
        msg = re.escape(msg)
        with self.assertRaisesRegex(ValueError, msg):
            self.attribute.validate(self.invalid_values, out_of_bounds_strategy="error")

    def test_validate_remove(self):
        """Validate remove strategy drops rows with invalid values."""
        actual = self.attribute.validate(
            self.invalid_values, out_of_bounds_strategy="remove"
        )
        expected = pd.Series(["A", "B", "D"], index=[0, 1, 3])
        pd.testing.assert_series_equal(actual, expected)

    def test_validate_clip(self):
        """Validate clip strategy raises ValueError for CategoricalStr columns."""
        msg = 'Cannot use strategy "clip" with CategoricalStr columns.'
        with self.assertRaisesRegex(ValueError, msg):
            self.attribute.validate(self.invalid_values, out_of_bounds_strategy="clip")

    def test_validate_unknown(self):
        """Validate unknown strategy raises ValueError."""
        msg = 'Unknown out of bounds strategy "unknown"'
        with self.assertRaisesRegex(ValueError, msg):
            self.attribute.validate(
                self.invalid_values, out_of_bounds_strategy="unknown"
            )

    def test_only_str_values(self):
        """CategoricalStr raises ValueError is native values are not strings."""
        msg = "CategoricalStr values must be strings, not [1, 2, 3, 4]"
        msg = re.escape(msg)
        with self.assertRaisesRegex(ValueError, msg):
            CategoricalStr("A", [1, 2, 3, 4])

    def test_equality(self):
        """Equality on CategoricalStr attribute is as expected."""
        self.assertEqual(
            self.attribute, CategoricalStr("test", ["A", "B", "C", "D", "E"])
        )
        self.assertNotEqual(
            self.attribute, CategoricalStr("test", ["A", "B", "C", "D", "F"])
        )
        self.assertNotEqual(
            self.attribute, CategoricalStr("test2", ["A", "B", "C", "D", "E"])
        )
        self.assertNotEqual(self.attribute, Continuous("test", 3, [0, 10]))

    def test_repr(self):
        """String representation is as expected."""
        actual = repr(self.attribute)
        expected = "CategoricalStr('test', ['A', 'B', 'C', 'D', 'E'])"
        self.assertEqual(actual, expected)


class TestCategoricalInt(unittest.TestCase):
    """TestCase for :class:`~tmlt.common.configuration.CategoricalInt`."""

    def setUp(self):
        """Setup."""
        super().setUp()
        self.unstandardized_values = pd.Series([1, 7, 2, 3, 5])
        self.standardized_values = pd.Series([0, 1, 2, 3, 4])
        self.invalid_values = pd.Series([1, 7, 2, 4, 2])
        self.attribute = CategoricalInt("test", [1, 7, 2, 3, 5])

    def test_invalid_domain(self):
        """CategoricalInt raises ValueError for invalid domain."""
        msg = (
            "Attribute domain must be an integer greater than or equal to "
            "1, not 0 (CategoricalInt('test', []))"
        )
        msg = re.escape(msg)
        with self.assertRaisesRegex(ValueError, msg):
            CategoricalInt("test", [])

    def test_properties(self):
        """CategoricalInt attribute properties are as expected."""
        self.assertEqual(self.attribute.column, "test")
        self.assertEqual(self.attribute.domain, 5)
        self.assertEqual(self.attribute.dtype, int)

    def test_standardize(self):
        """Standardize on CategoricalInt attribute returns the expected values."""
        actual = self.attribute.standardize(self.unstandardized_values)
        expected = self.standardized_values
        pd.testing.assert_series_equal(actual, expected)

    def test_unstandardize(self):
        """Unstandardize on CategoricalInt attribute returns the expected values."""
        actual = self.attribute.unstandardize(self.standardized_values)
        expected = self.unstandardized_values
        pd.testing.assert_series_equal(actual, expected)

    def test_validate_error(self):
        """Validate error strategy raises ValueError for invalid values."""
        msg = f"Found unexpected values in {self.attribute}\n3    4"
        msg = re.escape(msg)
        with self.assertRaisesRegex(ValueError, msg):
            self.attribute.validate(self.invalid_values, out_of_bounds_strategy="error")

    def test_validate_remove(self):
        """Validate remove strategy drops rows with invalid values."""
        actual = self.attribute.validate(
            self.invalid_values, out_of_bounds_strategy="remove"
        )
        expected = pd.Series([1, 7, 2, 2], index=[0, 1, 2, 4])
        pd.testing.assert_series_equal(actual, expected)

    def test_validate_clip(self):
        """Validate clip strategy raises ValueError for CategoricalInt columns."""
        msg = 'Cannot use strategy "clip" with CategoricalInt columns.'
        with self.assertRaisesRegex(ValueError, msg):
            self.attribute.validate(self.invalid_values, out_of_bounds_strategy="clip")

    def test_validate_unknown(self):
        """Validate unknown strategy raises ValueError."""
        msg = 'Unknown out of bounds strategy "unknown"'
        with self.assertRaisesRegex(ValueError, msg):
            self.attribute.validate(
                self.invalid_values, out_of_bounds_strategy="unknown"
            )

    def test_only_int_values(self):
        """CategoricalInt raises ValueError is native values are not integers."""
        msg = "CategoricalInt values must be integers, not ['A', 'B', 'C', 'D']"
        msg = re.escape(msg)
        with self.assertRaisesRegex(ValueError, msg):
            CategoricalInt("A", ["A", "B", "C", "D"])

    def test_equality(self):
        """Equality on CategoricalInt attribute is as expected."""
        self.assertEqual(self.attribute, CategoricalInt("test", [1, 7, 2, 3, 5]))
        self.assertNotEqual(self.attribute, CategoricalInt("test", [1, 7, 2, 3, 4]))
        self.assertNotEqual(self.attribute, CategoricalInt("test2", [1, 7, 2, 3, 5]))
        self.assertNotEqual(self.attribute, Continuous("test", 3, [0, 10]))

    def test_repr(self):
        """String representation is as expected."""
        actual = repr(self.attribute)
        expected = "CategoricalInt('test', [1, 7, 2, 3, 5])"
        self.assertEqual(actual, expected)


class TestConfig(unittest.TestCase):
    """TestCase for :class:`~tmlt.common.configuration.Config`."""

    def setUp(self):
        """Setup."""
        super().setUp()
        self.native_df = pd.DataFrame(
            {
                "continuous": [0, 0.1, 0.2, 0.3, 0.4, 0.5],
                "discrete": [-5, -4, -3, -2, -1, 0],
                "splits": [0.1, 1.1, 2.1, 3.1, 4.1, 5.1],
                "categorical": ["A", "B", "A", "C", "D", "E"],
            }
        )
        self.standardized_df = pd.DataFrame(
            {
                "continuous": [0, 0, 1, 1, 2, 2],
                "discrete": [0, 0, 0, 1, 1, 1],
                "splits": [0, 1, 1, 1, 2, 3],
                "categorical": [0, 1, 0, 2, 3, 4],
            }
        )
        self.unstandardized_df = pd.DataFrame(
            {
                "continuous": [
                    (0, 0.2),
                    (0, 0.2),
                    (0.2, 0.4),
                    (0.2, 0.4),
                    (0.4, 0.6),
                    (0.4, 0.6),
                ],
                "discrete": [(-5, -2), (-5, -2), (-5, -2), (-2, 1), (-2, 1), (-2, 1)],
                "splits": [(0, 1), (1, 4), (1, 4), (1, 4), (4, 5), (5, 6)],
                "categorical": ["A", "B", "A", "C", "D", "E"],
            }
        )
        self.invalid_df = pd.DataFrame(
            {
                "continuous": [0, 0.1, 0.2],
                "discrete": [-5, -4, -3],
                "splits": [1.1, 10000000, 0.1],  # Only 1 invalid value.
                "categorical": ["A", "B", "A"],
            }
        )
        self.config = Config(
            [
                Continuous("continuous", 3, [0, 0.6]),
                Discrete("discrete", 2, [-5, 1]),
                Splits("splits", [0, 1, 4, 5, 6]),
                CategoricalStr("categorical", ["A", "B", "C", "D", "E"]),
            ]
        )
        self.continuous = Continuous("continuous", 3, [0, 0.6])
        self.tuples = [
            (0, 0, 0, 0),
            (0, 0, 0, 1),
            (0, 0, 1, 0),
            (1, 1, 1, 1),
            (2, 1, 3, 4),
        ]
        self.indices = [0, 1, 5, 2 * 4 * 5 + 4 * 5 + 5 + 1, 3 * 2 * 4 * 5 - 1]

    @parameterized.expand(
        [
            ("continuous", [0, 0.05, 0.1, 0.15, 0.6]),
            ("discrete", [-5, -4, -3, -2, -1, 0, 1]),
            ("splits", [0, 3, 6]),
        ]
    )
    def test_set_bins(self, column: str, bins: Sequence[float]):
        """set_bins returns a config with an updated attribute."""
        config = self.config.set_bins(column, bins)
        expected_columns = self.config.columns
        actual_columns = config.columns
        self.assertListEqual(actual_columns, expected_columns)

        expected_attribute = Splits(column, bins)
        actual_attribute = config[column]
        self.assertEqual(actual_attribute, expected_attribute)

    def test_set_bins_not_continuous(self):
        """set_bins raises an error if "column" isn't continuous."""
        msg = "Column to update must be Continuous, not CategoricalStr"
        with self.assertRaisesRegex(ValueError, msg):
            self.config.set_bins("categorical", [1, 2, 3])

    def test_set_bins_new_domain(self):
        """set_bins raises an error if the new bins would change the domain."""
        msg = r"set_bins cannot update domain. (0.0, 0.3) != (0.0, 0.6)"
        msg = re.escape(msg)
        with self.assertRaisesRegex(ValueError, msg):
            self.config.set_bins("continuous", [0, 0.2, 0.3])

    def test_not_an_attribute(self):
        """Config raises ValueError for invalid values."""
        msg = "Expected Attribute, not str"
        with self.assertRaisesRegex(ValueError, msg):
            Config(self.config.attributes + ["I am not an attribute"])

    def test_repeated_column(self):
        """Config raises ValueError for repeated column names."""
        msg = 'Column names must be unique, found "test" more than once'
        with self.assertRaisesRegex(ValueError, msg):
            Config([Continuous("test", 10, [0, 10]), Discrete("test", 5, [0, 10])])

    def test_properties(self):
        """Config properties are as expected."""
        self.assertEqual(
            self.config.columns, ["continuous", "discrete", "splits", "categorical"]
        )
        self.assertEqual(self.config.domain, (3, 2, 4, 5))
        self.assertEqual(self.config.domain_size, 3 * 2 * 4 * 5)

    def test_standardize(self):
        """Standardize on Config returns the expected values."""
        actual = self.config.standardize(self.native_df)
        expected = self.standardized_df
        pd.testing.assert_frame_equal(actual, expected)

    def test_unstandardize(self):
        """Unstandardize on Config returns the expected values."""
        actual = self.config.unstandardize(self.standardized_df)
        expected = self.unstandardized_df
        pd.testing.assert_frame_equal(actual, expected)

    def test_validate_error(self):
        """Validate error strategy raises ValueError for invalid values."""
        msg = (
            f"Found out of bounds elements in {self.config['splits']}\n1    10000000.0"
        )
        msg = re.escape(msg)
        with self.assertRaisesRegex(ValueError, msg):
            self.config.validate(self.invalid_df, out_of_bounds_strategy="error")

    def test_validate_remove(self):
        """Validate remove strategy drops rows with invalid values."""
        actual = self.config.validate(self.invalid_df, out_of_bounds_strategy="remove")
        expected = self.invalid_df.iloc[[0, 2]].reset_index(drop=True)
        pd.testing.assert_frame_equal(actual, expected)

    def test_validate_clip(self):
        """Validate clip strategy returns expected values."""
        actual = self.config.validate(self.invalid_df, out_of_bounds_strategy="clip")
        expected = self.invalid_df.copy()
        expected.loc[[1], "splits"] = 6
        pd.testing.assert_frame_equal(actual, expected)

    def test_validate_unknown(self):
        """Validate unknown strategy raises ValueError."""
        msg = 'Unknown out of bounds strategy "unknown"'
        with self.assertRaisesRegex(ValueError, msg):
            self.config.validate(self.invalid_df, out_of_bounds_strategy="unknown")

    def test_validate_extra_columns(self):
        """Config.validate returns expected dataframe."""
        config = Config(self.config.attributes[:2])
        actual = config.validate(self.native_df)
        expected = self.native_df
        pd.testing.assert_frame_equal(actual, expected)

    def test_project(self):
        """Config.project returns new config with projected columns."""
        updated_config = self.config.project(["discrete", "continuous"])
        self.assertEqual(updated_config.domain, (2, 3))
        self.assertEqual(updated_config.columns, ["discrete", "continuous"])

    def test_indices_to_tuples(self):
        """Config.indices_to_tuples returns expected tuples."""
        actual = self.config.indices_to_tuples(self.indices)
        expected = np.array(self.tuples)
        np.testing.assert_array_equal(actual, expected)

    def test_tuples_to_indices(self):
        """Config.tuples_to_indices returns expected indices."""
        actual = self.config.tuples_to_indices(self.tuples)
        expected = np.array(self.indices)
        np.testing.assert_array_equal(actual, expected)

    def test_indices_to_tuples_empty(self):
        """Config.indices_to_tuples on empty indices."""
        actual = self.config.indices_to_tuples([])
        expected = np.empty((0, len(self.config)), dtype=int)
        np.testing.assert_array_equal(actual, expected)

    def test_tuples_to_indices_empty(self):
        """Config.tuples_to_indices on empty tuple."""
        actual = self.config.tuples_to_indices([])
        expected = np.empty((0,), dtype=int)
        np.testing.assert_array_equal(actual, expected)

    def test_magic_get_attribute(self):
        """Test get attribute from Config."""
        actual = self.config["continuous"]
        expected = self.continuous
        self.assertEqual(actual, expected)

    def test_magic_get_config(self):
        """Test get projection from Config."""
        actual = self.config[["continuous"]]
        expected = Config([self.continuous])
        self.assertEqual(actual, expected)

    def test_magic_eq(self):
        """Test that configs have same attributes."""
        actual = self.config[["continuous"]]
        expected = Config([self.continuous])
        self.assertEqual(actual, expected)

    def test_magic_add(self):
        """Test new config having attributes of both configs is returned."""
        actual = self.config[["continuous"]] + self.config[["discrete"]]
        expected = self.config[["continuous", "discrete"]]
        self.assertEqual(actual, expected)

    def test_magic_iter(self):
        """Test that iterator over the attributes in the config."""
        actual = list(self.config)
        expected = self.config.attributes
        self.assertEqual(actual, expected)

    def test_magic_len(self):
        """Test that number of columns in the config is as expected."""
        actual = len(self.config)
        expected = 4
        self.assertEqual(actual, expected)

    def test_repr(self):
        """String representation is as expected."""
        actual = repr(self.config[["continuous", "discrete"]])
        expected = (
            "Config([Continuous('continuous', 3, [0.0, 0.6]), "
            "Discrete('discrete', 2, [-5, 1])])"
        )
        self.assertEqual(actual, expected)

    @parameterized.expand(
        [
            (
                0,
                {
                    "continuous": 0.1,
                    "discrete": 0,
                    "splits": 0.5,
                    "categorical": "B",
                    "unrestricted": "55",
                },
            ),
            (
                1,
                {
                    "continuous": 0.2,
                    "discrete": -5,
                    "splits": 5,
                    "categorical": "E",
                    "unrestricted": "23",
                },
            ),
            (
                2,
                {
                    "continuous": 0.5,
                    "discrete": -2,
                    "splits": 6,
                    "categorical": "D",
                    "unrestricted": "90",
                },
            ),
        ]
    )
    def test_is_valid_True(self, _, row):
        """Config.is_valid correctly identifies valid records."""
        config = Config(
            [
                Continuous("continuous", 3, [0, 0.6]),
                Discrete("discrete", 2, [-5, 1]),
                Splits("splits", [0, 1, 4, 5, 6]),
                CategoricalStr("categorical", ["A", "B", "C", "D", "E"]),
                Unrestricted("unrestricted", r"\d\d"),
            ]
        )
        self.assertTrue(config.is_valid(row))

    @parameterized.expand(
        [
            (
                0,
                {
                    "continuous": 0.6,
                    "discrete": 1,
                    "splits": -1,
                    "categorical": 1,
                    "unrestricted": "555",
                },
            ),
            (
                1,
                {
                    "continuous": -0.1,
                    "discrete": 2,
                    "splits": 7,
                    "categorical": "F",
                    "unrestricted": "5",
                },
            ),
            (
                2,
                {
                    "continuous": 1,
                    "discrete": 5,
                    "splits": 7,
                    "categorical": 25,
                    "unrestricted": "AB",
                },
            ),
        ]
    )
    def test_is_valid_False(self, _, row):
        """Config.is_valid correctly identifies invalid records."""
        config = Config(
            [
                Continuous("continuous", 3, [0, 0.6]),
                Discrete("discrete", 2, [-5, 1]),
                Splits("splits", [0, 1, 4, 5, 6]),
                CategoricalStr("categorical", ["A", "B", "C", "D", "E"]),
                Unrestricted("unrestricted", r"\d\d"),
            ]
        )
        self.assertFalse(config.is_valid(row))
