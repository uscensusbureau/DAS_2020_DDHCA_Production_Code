"""Module for converting between different representations of the data.

This module has three main responsibilities:

1. Going between user interpretable (native) DataFrames and standardized
   DataFrames where each attribute is encoded as integers.
2. Going between the standardized representation of rows and what index
   of the 1d histogram it corresponds to.
3. Validating the config and DataFrame.

Since some operators can change the standardized form, such as changing the
domain or projecting onto a subset of columns, the config has methods for
updating after these changes. It also makes sure that these changes make
sense. For instance, it will raise an error if a user tries to change the
domain of a categorical column.
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

import re
from abc import abstractmethod
from typing import (
    Any,
    Dict,
    Generic,
    Iterator,
    List,
    Sequence,
    Tuple,
    TypeVar,
    overload,
)

import numpy as np
import pandas as pd

from tmlt.common.marshallable import Marshallable

Row = Dict[str, Any]
"""Type alias for a dictionary with string keys."""


class Attribute(Marshallable):
    """Specification for a single attribute in the domain.

    It is responsible for going between standardized and native forms for that
    column. See :class:`Config` for more information.
    """

    def __init__(self, column: str, domain: int, dtype: type = str):
        """Constructor.

        Args:
            column: The name of the column this attribute corresponds to.
            domain: The number of possible unique values for this attribute.
            dtype: The data type of values for this attribute.
        """
        self._column = column
        self._domain = domain
        self._dtype = dtype
        if (domain != int(domain)) | (not domain >= 1):
            raise ValueError(
                "Attribute domain must be an integer greater "
                f"than or equal to 1, not {domain} ({self})"
            )
        if column == "":
            raise ValueError("Cannot use empty string for column name.")
        if column[0] == "_":
            raise ValueError('Column names starting with "_" are reserved.')
        if not isinstance(dtype, type):
            raise TypeError(f"Expected dtype to be type, not {type(dtype)}")

    @property
    def column(self) -> str:
        """Return the name of the column this attribute corresponds to."""
        return self._column

    @property
    def domain(self) -> int:
        """Return the number of possible unique values for this attribute."""
        return self._domain

    @property
    def dtype(self) -> type:
        """Return the type of this attribute."""
        return self._dtype

    @abstractmethod
    def is_valid(self, value: Any) -> bool:
        """Determine whether the value conforms to the attribute's native domain.

        Args:
            value: The value to check.
        """
        raise NotImplementedError()

    @abstractmethod
    def validate(
        self, values: pd.Series, out_of_bounds_strategy: str = "error"
    ) -> pd.Series:
        """Return a new Series which conforms to the attribute's native domain.

        Args:
            values: The Series in human readable form to validate.
            out_of_bounds_strategy: How to handle rows which are outside the
                bounds of this column's domain. The default strategy, "error",
                raises an exception, "remove" eliminates the offending rows,
                and "clip" sets the value to the upper or lower bound of the
                domain.
        """
        raise NotImplementedError()

    @abstractmethod
    def standardize(self, values: pd.Series) -> pd.Series:
        """Return a new Series which is converted to standard form.

        In standard form every value is a nonnegative integer.

        Args:
            values: The Series to convert to standard form.
        """
        raise NotImplementedError()

    @abstractmethod
    def unstandardize(self, values: pd.Series) -> pd.Series:
        """Return a new Series which is converted to a human readable form.

        In standard form every value is a nonnegative integer.

        Args:
            values: The Series to convert from standard form.
        """
        raise NotImplementedError()

    def __eq__(self, other: object) -> bool:
        """Return whether the two attributes are equivalent.

        Args:
            other: Attribute to be compared against.
        """
        if not isinstance(other, self.__class__):
            return False
        return (
            self.column == other.column
            and self.domain == other.domain
            and self.dtype == other.dtype
        )

    @abstractmethod
    def __repr__(self) -> str:
        """Return a string representation of self."""
        raise NotImplementedError()


class Unrestricted(Attribute):
    """An attribute without a domain."""

    @Marshallable.SaveInitParams
    def __init__(self, column: str, pattern: str = None):
        """Constructor.

        Args:
            column: The name of the column this attribute corresponds to.
            pattern: Optional regular expression to validate columns against.
        """
        super().__init__(column, 1, str)
        if pattern is not None and not isinstance(pattern, str):
            raise TypeError(f"Expected pattern to be a str, not {pattern}")
        self._pattern = pattern

    @property
    def domain(self) -> int:
        """Return the number of possible unique values for this attribute."""
        raise NotImplementedError()

    def is_valid(self, value: Any) -> bool:
        """Determine whether the value conforms to the attribute's native domain.

        Args:
            value: The value to check.
        """
        if self._pattern is not None and re.fullmatch(self._pattern, value) is None:
            return False
        return True

    def validate(
        self, values: pd.Series, out_of_bounds_strategy: str = "error"
    ) -> pd.Series:
        """Return a new Series which conforms to the attribute's native domain.

        Args:
            values: The Series in human readable form to validate.
            out_of_bounds_strategy: How to handle rows which are outside the
                bounds of this column's domain. The default strategy, "error",
                raises an exception, "remove" eliminates the offending rows,
                and "clip" sets the value to the upper or lower bound of the
                domain.
        """
        # This is a hack to fix AttributeError: Can only use .str accessor
        # with string values!.
        if len(values) > 0 and any(isinstance(item, (int, float)) for item in values):
            values = values.astype("str")
        if self._pattern is not None:
            unknown = ~values.str.fullmatch(self._pattern)
        else:
            unknown = pd.Series([], dtype=str)
        if any(unknown):
            if out_of_bounds_strategy == "error":
                raise ValueError(
                    f"Found unexpected values in {self}\n{values[unknown]}"
                )
            if out_of_bounds_strategy == "remove":
                return values[~unknown]
            if out_of_bounds_strategy == "clip":
                raise ValueError(
                    f'Cannot use strategy "clip" with {type(self).__name__} columns.'
                )
            raise ValueError(
                f'Unknown out of bounds strategy "{out_of_bounds_strategy}"'
            )
        return values

    def standardize(self, values: pd.Series) -> pd.Series:
        """Return a new Series which is converted to standard form.

        In standard form every value is a nonnegative integer.

        Args:
            values: The Series to convert to standard form.
        """
        raise NotImplementedError()

    def unstandardize(self, values: pd.Series) -> pd.Series:
        """Return a new Series which is converted to a human readable form.

        In standard form every value is a nonnegative integer.

        Args:
            values: The Series to convert from standard form.
        """
        raise NotImplementedError()

    def __eq__(self, other: object) -> bool:
        """Return whether the two attributes are equivalent.

        Args:
            other: Attribute to compare against.
        """
        if not isinstance(other, self.__class__):
            return False
        return self.column == other.column and self._pattern == other._pattern

    def __repr__(self) -> str:
        """Return a string representation of self."""
        if self._pattern is not None:
            return (
                f"{self.__class__.__name__}('{self.column}', pattern={self._pattern})"
            )
        return f"{self.__class__.__name__}('{self.column}')"


class Continuous(Attribute):
    """A numerical attribute which is binned in the standard form."""

    @Marshallable.SaveInitParams
    def __init__(
        self, column: str, bins: int, domain: Tuple[int, int], dtype: type = float
    ):
        """Constructor.

        Args:
            column: The name of the column this attribute corresponds to.
            bins: The number of equally sized bins to divide the domain into.
            domain: A tuple with the minimum, and maximum column values in the
                native domain.
            dtype: The data type of values for this attribute.
        """
        self._minimum = dtype(domain[0])
        self._maximum = dtype(domain[1])

        super().__init__(column, bins, dtype)
        if self._minimum >= self._maximum:
            raise ValueError(
                f"Minimum ({self._minimum}) must be smaller than "
                f"maximum ({self._maximum}) ({self})"
            )

    @property
    def minimum(self) -> int:
        """Return the minimum value in the native domain."""
        return self._minimum

    @property
    def maximum(self) -> int:
        """Return the maximum value in the native domain."""
        return self._maximum

    @property
    def splits(self) -> List[float]:
        """Return list of boundaries between bins.

        For example, [0, 1, 4, 8] specifies
        three bins: [0, 1), [1, 4), and [4, 8).
        """
        bin_width = (self.maximum - self.minimum) / self.domain
        return [self.minimum + i * bin_width for i in range(self.domain + 1)]

    def is_valid(self, value: Any) -> bool:
        """Determine whether the value conforms to the attribute's native domain.

        Args:
            value: The value to check.
        """
        if (value < self.minimum) or (value >= self.maximum):
            return False
        return True

    def validate(
        self, values: pd.Series, out_of_bounds_strategy: str = "error"
    ) -> pd.Series:
        """Return a new Series which conforms to the attribute's native domain.

        Args:
            values: The Series in human readable form to validate.
            out_of_bounds_strategy: How to handle rows which are outside the
                bounds of this column's domain. The default strategy, "error",
                raises an exception, "remove" eliminates the offending rows,
                and "clip" sets the value to the upper or lower bound of the
                domain.
        """
        out_of_bounds = (values < self.minimum) | (values >= self.maximum)
        if any(out_of_bounds):
            if out_of_bounds_strategy == "error":
                raise ValueError(
                    f"Found out of bounds elements in {self}\n{values[out_of_bounds]}"
                )
            if out_of_bounds_strategy == "remove":
                return values[~out_of_bounds]
            if out_of_bounds_strategy == "clip":
                return pd.Series(
                    np.clip(values, self.minimum, self.maximum), index=values.index
                )
            raise ValueError(
                f'Unknown out of bounds strategy "{out_of_bounds_strategy}"'
            )
        return values

    def standardize(self, values: pd.Series) -> pd.Series:
        """Return a new Series which is converted to standard form.

        In standard form every value is a nonnegative integer.

        Args:
            values: The Series to convert to standard form.
        """
        bin_width = (self.maximum - self.minimum) / self.domain
        values = ((values - self.minimum) // bin_width).astype(int)
        # The upper bound of the domain is inclusive.
        values[values == self.domain] = self.domain - 1
        return values

    def unstandardize(self, values: pd.Series) -> pd.Series:
        """Return a new Series which is converted to a human readable form.

        In standard form every value is a nonnegative integer.

        Args:
            values: The Series to convert from standard form.
        """
        bin_width = (self.maximum - self.minimum) / self.domain
        bin_mins = (values * bin_width + self.minimum).astype(self.dtype)
        bin_maxs = (bin_mins + bin_width).astype(self.dtype)
        return pd.Series(zip(bin_mins, bin_maxs), index=values.index)

    def __eq__(self, other: object) -> bool:
        """Return whether the two attributes are equivalent.

        Args:
            other: Attribute to compare against.
        """
        # This will be checked again in the call to super(), but must be checked here to
        # appease mypy.
        if not isinstance(other, self.__class__):
            return False
        return (
            super().__eq__(other)
            and self.minimum == other.minimum
            and self.maximum == other.maximum
        )

    def __repr__(self) -> str:
        """Return a string representation of self."""
        return (
            f"Continuous('{self.column}', {self.domain}, [{self.minimum}, "
            f"{self.maximum}])"
        )


class Discrete(Continuous):
    """An attribute where all values are integer. Divides evenly into bins."""

    @Marshallable.SaveInitParams
    def __init__(self, column: str, bins: int, domain: Tuple[int, int]):
        """Constructor.

        Args:
            column: The name of the column this attribute corresponds to.
            bins: The number of equally sized bins to divide the domain into.
            domain: A tuple with the minimum, and maximum column values in the
                native domain.
        """
        super().__init__(column, bins, domain, dtype=int)
        k = (self.maximum - self.minimum) / bins
        if k % 1 != 0 or not k > 0:
            raise ValueError(
                "There must be an integer number of bins for each value in "
                f"the domain for discrete columns ({self})"
            )

    @property
    def dtype(self) -> type:
        """Return the type of this attribute."""
        return int

    def __repr__(self) -> str:
        """Return a string representation of self."""
        return (
            f"Discrete('{self.column}', {self.domain}, [{self.minimum}, "
            f"{self.maximum}])"
        )


class Splits(Continuous):
    """A continuous domain that is divided into explicit bins."""

    @Marshallable.SaveInitParams
    def __init__(self, column: str, splits: Sequence[float]):
        """Constructor.

        Args:
            column: The name of the column this attribute corresponds to.
            splits: A list of floats sorted in ascending order that specify the
                boundaries of each bin. For example, [0, 1, 4, 8] specifies
                three bins: [0, 1), [1, 4), and [4, 8].
        """
        splits = list(splits)
        self._splits = splits
        super().__init__(column, len(splits) - 1, (min(splits), max(splits)))
        if splits != sorted([float(split) for split in splits]):
            raise ValueError(f"Splits should be sorted in ascending order. ({self})")

    @property
    def splits(self) -> List[float]:
        """Return a list of boundaries between the bins in ascending order."""
        return self._splits

    def is_valid(self, value: Any) -> bool:
        """Determine whether the value conforms to the attribute's native domain.

        Args:
            value: The value to check.
        """
        if (value < self.minimum) or (value > self.maximum):
            return False
        return True

    def validate(
        self, values: pd.Series, out_of_bounds_strategy: str = "error"
    ) -> pd.Series:
        """Return a new Series which conforms to the attribute's native domain.

        This function is very similar to the parent function expect the maximum value is
        valid.

        Args:
            values: The Series in human readable form to validate.
            out_of_bounds_strategy: How to handle rows which are outside the
                bounds of this column's domain. The default strategy, "error",
                raises an exception, "remove" eliminates the offending rows,
                and "clip" sets the value to the upper or lower bound of the
                domain.
        """
        out_of_bounds = (values < self.minimum) | (values > self.maximum)
        if any(out_of_bounds):
            if out_of_bounds_strategy == "error":
                raise ValueError(
                    f"Found out of bounds elements in {self}\n{values[out_of_bounds]}"
                )
            if out_of_bounds_strategy == "remove":
                return values[~out_of_bounds]
            if out_of_bounds_strategy == "clip":
                return pd.Series(
                    np.clip(values, self.minimum, self.maximum), index=values.index
                )
            raise ValueError(
                f'Unknown out of bounds strategy "{out_of_bounds_strategy}"'
            )
        return values

    def standardize(self, values: pd.Series) -> pd.Series:
        """Return a new Series which is converted to standard form.

        In standard form every value is a nonnegative integer.

        Args:
            values: The Series to convert to standard form.
        """
        values = pd.Series(np.digitize(values, self.splits[1:]), index=values.index)
        # The upper bound of the domain is inclusive.
        values[values == self.domain] = self.domain - 1
        return values

    def unstandardize(self, values: pd.Series) -> pd.Series:
        """Return a new Series which is converted to a human readable form.

        In standard form every value is a nonnegative integer.

        Args:
            values: The Series to convert from standard form.
        """
        return pd.Series(
            [(self.splits[i], self.splits[i + 1]) for i in values], index=values.index
        )

    def __eq__(self, other: object) -> bool:
        """Return whether the two attributes are equivalent.

        Args:
            other: Attribute to compare against.
        """
        # This will be checked again in the call to super(), but must be checked here to
        # appease mypy.
        if not isinstance(other, self.__class__):
            return False
        return super().__eq__(other) and self.splits == other.splits

    def __repr__(self) -> str:
        """Return a string representation of self."""
        return f"Splits('{self.column}', {self.splits})"


V = TypeVar("V", int, str)


class Categorical(Attribute, Generic[V]):
    """Base class for attributes with a many to one mapping to standard form."""

    @Marshallable.SaveInitParams
    def __init__(self, column: str, values: Sequence[V]):
        """Constructor.

        Args:
            column: The name of the column this attribute corresponds to.
            values: A list of native values.
        """
        self._value_map: Dict[V, int] = {k: i for i, k in enumerate(values)}
        self._values: List[V] = list(values)
        super().__init__(column, len(values), dtype=self.dtype)
        if len(set(values)) != len(values):
            raise ValueError(f"Categorical values must be unique ({self})")

    @property
    @abstractmethod
    def dtype(self) -> type:
        """Return the type of this attribute."""
        raise NotImplementedError()

    @property
    def value_map(self) -> Dict[V, int]:
        """Return a dictionary from native to standardized."""
        return self._value_map

    @property
    def values(self) -> List[V]:
        """Return a list of values in standardized order."""
        return self._values

    def is_valid(self, value: Any) -> bool:
        """Determine whether the value conforms to the attribute's native domain.

        Args:
            value: The value to check.
        """
        return value in self.value_map

    def validate(
        self, values: pd.Series, out_of_bounds_strategy: str = "error"
    ) -> pd.Series:
        """Return a new Series which conforms to the attribute's native domain.

        Args:
            values: The Series in human readable form to validate.
            out_of_bounds_strategy: How to handle rows which are outside the
                attribute's native domain. The default strategy, "error",
                raises an exception, "remove" eliminates the offending rows,
                and "clip" sets the value to the upper or lower bound of the
                domain.
        """
        # This is a hack to fix a type error that occurs with isin.
        if len(values) > 0 and any(isinstance(item, int) for item in values):
            values = values.astype("int")
        unknown = ~values.isin(self.value_map)
        if any(unknown):
            if out_of_bounds_strategy == "error":
                raise ValueError(
                    f"Found unexpected values in {self}\n{values[unknown]}"
                )
            if out_of_bounds_strategy == "remove":
                return values[~unknown]
            if out_of_bounds_strategy == "clip":
                raise ValueError(
                    f'Cannot use strategy "clip" with {type(self).__name__} columns.'
                )
            raise ValueError(
                f'Unknown out of bounds strategy "{out_of_bounds_strategy}"'
            )
        return values

    def standardize(self, values: pd.Series) -> pd.Series:
        """Return a new Series which is converted to standard form.

        In standard form every value is a nonnegative integer.

        Args:
            values: The Series to convert to standard form.
        """
        return values.map(self.value_map)

    def unstandardize(self, values: pd.Series) -> pd.Series:
        """Return a new Series which is converted to a human readable form.

        In standard form every value is a nonnegative integer.

        Args:
            values: The Series to convert from standard form.
        """
        return values.map(dict(enumerate(self.values)))

    def __eq__(self, other: object) -> bool:
        """Return whether the two attributes are equivalent.

        Args:
            other: Attribute to compare against.
        """
        # This will be checked again in the call to super(), but must be checked here to
        # appease mypy.
        if not isinstance(other, self.__class__):
            return False
        return super().__eq__(other) and self.values == other.values


# pylint: disable=unsubscriptable-object
# https://github.com/PyCQA/pylint/issues/2822
class CategoricalStr(Categorical[str]):
    """A Categorical attribute with string values."""

    @Marshallable.SaveInitParams
    def __init__(self, column: str, values: Sequence[str]):
        """Constructor.

        Args:
            column: The name of the column this attribute corresponds to.
            values: A list of native values.
        """
        if any(k != str(k) for k in values):
            raise ValueError(f"CategoricalStr values must be strings, not {values}")
        super().__init__(column, values)

    @property
    def dtype(self) -> type:
        """Return the type of this attribute."""
        return str

    def __repr__(self) -> str:
        """Return a string representation of self."""
        return f"CategoricalStr('{self.column}', {self.values})"


# pylint: disable=unsubscriptable-object
# https://github.com/PyCQA/pylint/issues/2822
class CategoricalInt(Categorical[int]):
    """A Categorical attribute with integer values."""

    @Marshallable.SaveInitParams
    def __init__(self, column: str, values: Sequence[int]):
        """Constructor.

        Args:
            column: The name of the column this attribute corresponds to.
            values: A list of native values.
        """
        try:
            assert all(k == int(k) for k in values)
        except (ValueError, AssertionError):
            raise ValueError(f"CategoricalInt values must be integers, not {values}")
        super().__init__(column, values)

    @property
    def dtype(self) -> type:
        """Return the type of this attribute."""
        return int

    def __repr__(self) -> str:
        """Return a string representation of self."""
        return f"CategoricalInt('{self.column}', {self.values})"


class Config(Marshallable):
    """Organizes multiple attributes into a single domain configuration."""

    @Marshallable.SaveInitParams
    def __init__(self, attributes: Sequence[Attribute]):
        """Constructor.

        Args:
            attributes: A list of attributes in the desired cannonical order.
        """
        columns = []
        for attribute in attributes:
            if not isinstance(attribute, Attribute):
                raise ValueError(f"Expected Attribute, not {type(attribute).__name__}.")
            if attribute.column in columns:
                raise ValueError(
                    "Column names must be unique, found "
                    f'"{attribute.column}" more than once'
                )
            columns.append(attribute.column)
        self._columns = columns
        self._attributes = list(attributes)
        self._attribute_dict = dict(zip(columns, attributes))

    @property
    def attributes(self) -> List[Attribute]:
        """Return the attributes included in this config."""
        return self._attributes

    @property
    def columns(self) -> List[str]:
        """Return the columns in the cannonical order."""
        return self._columns

    @property
    def domain(self) -> Tuple[int, ...]:
        """Return the standard form of the domain.

        Each element of the domain is the number of distinct values in the
        standard form of each column.
        """
        return tuple(attribute.domain for attribute in self)

    @property
    def domain_size(self) -> int:
        """Return the total number of possible tuples in the domain."""
        # dtype is int, for case where domain is empty.
        return np.prod(self.domain, dtype=int)

    def is_valid(self, row: Row) -> bool:
        """Determine whether row conforms to the config's native domain.

        Args:
            row: the row to validate.
        """
        for attribute in self.attributes:
            if not attribute.is_valid(row[attribute.column]):
                return False
        return True

    def validate(
        self, data_frame: pd.DataFrame, out_of_bounds_strategy: str = "error"
    ) -> pd.DataFrame:
        """Return a new DataFrame that conforms to the config's native domain.

        Ignores columns which are not included in the config, but ensures that
        all columns in the config are present.

        Args:
            data_frame: A DataFrame in native form to validate.
            out_of_bounds_strategy: How to handle rows which are outside of a
                column's domain. The default strategy, "error", raises an
                exception, "remove" eliminates the offending rows, and "clip"
                sets the value to the upper or lower bound of the domain.
        """
        if not self.attributes:
            return data_frame
        index = data_frame.index
        column_values = []
        for column in data_frame.columns:
            if column in self.columns:
                attribute = self[column]
                values = data_frame.loc[index, column]
                values = attribute.validate(values, out_of_bounds_strategy)
                column_values.append(values)
                index = index & values.index
            else:
                column_values.append(data_frame[column])
        new_df = pd.concat(column_values, axis=1, join="inner")
        new_df.columns = data_frame.columns
        return new_df.reset_index(drop=True)

    def add(self, attributes: Sequence[Attribute]) -> "Config":
        """Return a new Config with the attributes appended to the end.

        Args:
            attributes: Attributes to be appended.
        """
        return Config(self.attributes + list(attributes))

    def project(self, columns: Sequence[str]) -> "Config":
        """Return a new Config projected to the given columns.

        Args:
            columns: Column names to project to.
        """
        return Config([self[c] for c in columns])

    def set_bins(self, column: str, bins: Sequence[float]) -> "Config":
        """Return a new Config with an updated binning for the specified column.

        Args:
            column: Name of the attribute to update.
            bins: New binning. Must have same upper and lower bound as existing binning.
        """
        new_binning = Splits(column, bins)
        old_binning = self[column]
        if not isinstance(old_binning, Continuous):
            raise ValueError(
                f"Column to update must be Continuous, not {type(old_binning).__name__}"
            )
        new_domain = (new_binning.minimum, new_binning.maximum)
        old_domain = (old_binning.minimum, old_binning.maximum)
        if new_domain != old_domain:
            raise ValueError(
                f"set_bins cannot update domain. {new_domain} != {old_domain}"
            )
        return Config(
            [
                new_binning if attribute.column is column else attribute
                for attribute in self
            ]
        )

    def standardize(self, data_frame: pd.DataFrame) -> pd.DataFrame:
        """Return a standardized form of the DataFrame.

        In standard form every row is a tuple of nonnegative integer.

        Args:
            data_frame: The DataFrame to convert to standard form.
        """
        data_frame = data_frame.copy()
        for attribute in self:
            column = attribute.column
            data_frame[column] = attribute.standardize(data_frame[column])
        return data_frame

    def unstandardize(self, data_frame: pd.DataFrame) -> pd.DataFrame:
        """Return a human readable form of the DataFrame.

        In standard form every row is a tuple of nonnegative integer.

        Args:
            data_frame: The DataFrame to convert from standard form.
        """
        data_frame = data_frame.copy()
        for attribute in self:
            column = attribute.column
            data_frame[column] = attribute.unstandardize(data_frame[column])
        return data_frame

    def tuples_to_indices(self, tuples: Sequence[Tuple]) -> np.ndarray:
        """Return a list of integers corresponding to the given tuples.

        The integers are the index of each tuple in the 1-d histogram of the
        domain. Uses C-style (row-major) order to determine the index.

        See np.ravel_multi_index for more information.

        Args:
            tuples: A list of tuples in the standardized domain.
        """
        if len(tuples) == 0:
            return np.empty((0,), dtype=int)
        tuples_array = np.array(tuples, dtype=int)
        # Typing for np.ravel_multi_index appears to be unworkable. The example
        # code from the numpy website throws mypy errors.
        return np.ravel_multi_index(tuples_array.transpose(), self.domain)  # type: ignore[call-overload] # pylint: disable=line-too-long

    def indices_to_tuples(self, indices: Sequence[int]) -> np.ndarray:
        """Return a list of tuples corresponding to the given indices.

        The integers are the index of each tuple in the 1-d histogram of the
        domain. Uses C-style (row-major) order to determine the index.

        See np.ravel for more information.

        Args:
            indices: A list of indices for the 1-d histogram of the domain.
        """
        np_indices = np.array(indices, dtype=int)
        return np.transpose(np.unravel_index(np_indices, self.domain))

    def __add__(self, other: "Config") -> "Config":
        """Return a new config with the attributes of both configs.

        Args:
            other: Config to add.
        """
        return self.add(other.attributes)

    def __eq__(self, other: object) -> bool:
        """Return whether the two configs have the same attributes.

        Args:
            other: Config to compare against.
        """
        if not isinstance(other, self.__class__):
            return False
        return self.attributes == other.attributes

    @overload
    def __getitem__(
        self, key: str
    ) -> Attribute:  # noqa: D105 https://github.com/PyCQA/pydocstyle/issues/525
        ...

    @overload
    def __getitem__(
        self, key: List[str]
    ) -> "Config":  # noqa: D105 https://github.com/PyCQA/pydocstyle/issues/525
        ...

    def __getitem__(
        self, key
    ):  # pylint: disable=missing-type-doc, missing-return-type-doc
        """Return a projection if given a list, or an attribute otherwise.

        Args:
            key: column name(s) to project to.
        """
        if isinstance(key, list):
            return self.project(key)
        return self._attribute_dict[key]

    def __iter__(self) -> Iterator[Attribute]:
        """Return an iterator over the attributes in the config."""
        return iter(self.attributes)

    def __len__(self) -> int:
        """Return the number of columns in the config."""
        return len(self.columns)

    def __repr__(self) -> str:
        """Return a string representation of self."""
        return f"Config({self.attributes})"
