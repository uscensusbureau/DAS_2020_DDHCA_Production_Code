"""Utility for marshalling objects."""

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

import base64
import importlib
import inspect
import json
from abc import ABC
from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import fields, is_dataclass
from functools import wraps
from io import BytesIO
from types import FunctionType
from typing import Any, Callable, Dict, List, Type, TypeVar, Union

import numpy as np
import pandas as pd
from scipy import sparse

# https://www.python.org/dev/peps/pep-0484/#annotating-instance-and-class-methods
M = TypeVar("M", bound="Marshallable")
"""Type admitting instances of subclasses of
:class:`~tmlt.common.marshallable.Marshallable`."""

Item = Any
"""Type alias for :class:`~typing.Any`."""

Primitive = Union[int, float, str, bool, None]
"""Primitives."""


def get_class_name(obj: object) -> str:
    """Return the fully qualified class name.

    Example:
        >>> from tmlt.common.schema import Schema
        >>> schema = Schema(column_types={"foo": str})
        >>> get_class_name(schema)
        'tmlt.common.schema.Schema'

    Args:
        obj: Object to get the class name for.
    """
    module_str = obj.__class__.__module__
    class_str = obj.__class__.__name__
    return f"{module_str}.{class_str}"


def get_class(class_name: str) -> type:
    """Return the class from it's class name.

    Example:
        >>> class_name = "tmlt.common.schema.Schema"
        >>> identity_class = get_class(class_name)
        >>> identity_class(column_types={"foo": str}).columns
        ['foo']

    Args:
        class_name: A full class name including module. For example,
            "tmlt.common.marshallable.Marshallable".
    """
    class_name_parts = class_name.split(".")
    class_str = class_name_parts[-1]
    module_str = ".".join(class_name_parts[:-1])
    module = importlib.import_module(module_str)
    return getattr(module, class_str)


class Marshallable(ABC):
    """Marshallable objects can serialized to and from JSON.

    Terminology for the purpose of this class:

    The "marshalled" form of an object is a nested dictionary which can be
    converted to JSON.

    The "serialized" form of an object is a JSON str.
    """

    PRIMITIVES = (int, float, str, bool, int, str, type(None))

    def __init__(self, *args, **kwargs):
        super().__init__(self, *args, **kwargs)

    @staticmethod
    def SaveInitParams(func: Callable) -> Callable:
        """Decorator for Marshallable subclasses.

        Captures the initialization parameters when a marshallable object is
        initialized and stores them to self._init_params.

        The intended use is:

        .. code-block:: python

            @Marshallable.SaveInitParams
            def __init__(self, ...):
                ...

        Args:
            func: The __init__ of a Marshallable class.

        Returns:
            A wrapped __init__ that stores the initialization parameters.
        """
        names = inspect.getfullargspec(func).args

        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # NOTE: This is so that Marshallable objects can call
            # super().__init__ without worrying about overwritting _init_params
            if not hasattr(self, "_init_params"):
                init_params = dict()
                for name, arg in zip(names[1:], args):
                    init_params[name] = arg
                for name, arg in kwargs.items():
                    init_params[name] = arg
                self._init_params = init_params  # pylint: disable=protected-access
            func(self, *args, **kwargs)

        return wrapper

    @property
    def init_params(self) -> Dict:
        """Return the parameters required to initialize a new instance."""
        if not hasattr(self, "_init_params"):
            raise AttributeError(
                "Use @Marshallable.SaveInitParams decorator on the __init__ "
                "of instantiable, Marshallable classes. See Marshallable for "
                "more details."
            )
        self._init_params: Dict  # pylint: disable=attribute-defined-outside-init
        return self._init_params  # pylint: disable=no-member

    @staticmethod
    def marshal(item: Item) -> Union[Dict, Primitive, List]:
        """Return the marshalled form of the item.

        Args:
            item: The item to be marshalled.

        Raises:
            TypeError: If the item is not one of the acceptable types.
                Currently primitives, list, tuple, Marshallable, np.ndarray,
                scipy.sparse, np.dtype, OrderedDict, dict, functions, and
                np.number are accepted.
        """
        if isinstance(item, Marshallable.PRIMITIVES):
            return item
        if isinstance(item, tuple):
            # json.dumps converts tuples to list
            return {"tuple": [Marshallable.marshal(i) for i in item]}
        if isinstance(item, list):
            return [Marshallable.marshal(i) for i in item]
        if isinstance(item, Marshallable):
            return item._marshal()  # pylint: disable=protected-access
        if isinstance(item, np.ndarray):
            return {
                "ndarray": item.tolist(),
                "shape": item.shape,
                "dtype": Marshallable.marshal(item.dtype),
            }
        if isinstance(item, sparse.spmatrix):
            item_format = item.format
            item = item.tocoo()
            return {
                "sparse": Marshallable.marshal(item.data),
                "format": item_format,
                "row": Marshallable.marshal(item.row),
                "col": Marshallable.marshal(item.col),
                "shape": item.shape,
            }
        if isinstance(item, np.dtype):
            return {"datatype": str(item)}
        if isinstance(item, OrderedDict):
            kv_tuples = tuple(item.items())
            return {"ordered_dict": Marshallable.marshal(kv_tuples)}
        if isinstance(item, dict):
            if not all(isinstance(key, str) for key in item):
                raise ValueError(
                    "Marshallable cannot serialize dictionaries with non-string keys"
                )
            return {key: Marshallable.marshal(value) for key, value in item.items()}
        if isinstance(item, FunctionType):
            return {
                "function": item.__name__,
                "source": inspect.getsource(item).strip(),
            }
        if isinstance(item, np.integer):
            return int(item)
        if isinstance(item, np.number):
            return float(item)
        if isinstance(item, pd.DataFrame):
            df_parquet = item.to_parquet(index=True)
            df = base64.b64encode(df_parquet).decode("ascii")
            return {"df": df}
        if is_dataclass(item):
            # Note: This likely won't work for all dataclasses.
            # For example, if there is a __post_init__ that changes the values.
            return {
                "dataclass": get_class_name(item),
                "init_params": Marshallable.marshal(
                    {
                        field.name: getattr(item, field.name)
                        for field in fields(item)
                        if field.init
                    }
                ),
            }

        raise TypeError(f"{type(item)} not supported by Marshallable._marshal.")

    @staticmethod
    def unmarshal(item: Dict) -> Item:
        """Retrieve an item from its marshalled form.

        Args:
            item: The marshalled item to unmarshal.
        """
        if isinstance(item, Marshallable.PRIMITIVES):
            return item
        if isinstance(item, list):
            return [Marshallable.unmarshal(i) for i in item]
        if isinstance(item, dict):
            if "class" in item:
                item_class = get_class(item["class"])
                assert issubclass(item_class, Marshallable)
                return item_class._unmarshal(item)  # pylint: disable=protected-access
            if "tuple" in item:
                unmarshalled = Marshallable.unmarshal(item["tuple"])
                assert isinstance(unmarshalled, Iterable)
                return tuple(unmarshalled)
            if "ndarray" in item:
                dtype = Marshallable.unmarshal(item["dtype"])
                return np.array(item["ndarray"], dtype=dtype).reshape(item["shape"])
            if "sparse" in item:
                data = Marshallable.unmarshal(item["sparse"])
                row = Marshallable.unmarshal(item["row"])
                col = Marshallable.unmarshal(item["col"])
                return sparse.coo_matrix(
                    (data, (row, col)), shape=item["shape"]
                ).asformat(item["format"])
            if "datatype" in item:
                return np.dtype(item["datatype"])
            if "ordered_dict" in item:
                unmarshalled = Marshallable.unmarshal(item["ordered_dict"])
                assert isinstance(unmarshalled, tuple)
                return OrderedDict(unmarshalled)
            if "function" in item:
                function = item["function"]
                source = item["source"]
                exec(source)  # pylint: disable=exec-used
                return locals()[function]
            if "df" in item:
                with BytesIO() as b:
                    b.write(base64.b64decode(item["df"]))
                    return pd.read_parquet(b)
            if "dataclass" in item:
                item_class = get_class(item["dataclass"])
                return item_class(**Marshallable.unmarshal(item["init_params"]))
            return {key: Marshallable.unmarshal(value) for key, value in item.items()}
        raise TypeError(
            f"{item} is not in a supported format for Marshallable.unmarshal."
        )

    def _marshal(self) -> Dict:
        """Return the marshalled form this object."""
        return {
            "marshalled": True,
            "class": get_class_name(self),
            "init_params": Marshallable.marshal(self.init_params),
        }

    @classmethod
    def _unmarshal(cls: Type[M], item: Dict) -> M:
        """Retrieve an item from its marshalled form.

        Args:
            item: The marshalled item to unmarshal.

        Raises:
            TypeError: If the item does not have a matching key for 'class'.
        """
        actual_classname = item["class"]
        expected_classname = f"{cls.__module__}.{cls.__name__}"
        assert (
            actual_classname == expected_classname
        ), f"Expected {expected_classname}, not {actual_classname}!"
        init_params = Marshallable.unmarshal(item["init_params"])
        assert isinstance(init_params, dict)
        return cls(**init_params)

    @staticmethod
    def serialize(item: Item) -> str:
        """Return the JSON string representation for an object.

        Args:
            item: The item to serialize.
        """
        return json.dumps(Marshallable.marshal(item))

    @staticmethod
    def deserialize(json_str: str) -> Item:
        """Return an appropriate object from the JSON string representation.

        Args:
            json_str: The JSON string to deserialize.
        """
        return Marshallable.unmarshal(json.loads(json_str))

    @classmethod
    def load_json(cls: Type[M], filename: str) -> M:
        """Returns unmarshalled object created from JSON file.

        Args:
            filename: The filename of the .json file to load.
        """
        with open(filename) as f:
            config_dict = json.load(f)
        obj = cls.unmarshal(config_dict)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected object of type {type(cls)}, not {type(obj)}")
        return obj

    def save_json(self, filename: str):
        """Saves a marshallable object in JSON format.

        Args:
            filename: The name of the file to save.
        """
        with open(filename, "w") as f:
            json.dump(self.marshal(self), f, indent=4)
