"""Ops arguments and results: values and pixels to and from ``Arg``.

An ``Arg`` is pixels (inline, or a reference to a tensor on a plane) or anything
else as a protobuf ``Value``. Both sides of the wire use these: a client encodes
what it sends and decodes what comes back, a server the reverse.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np
from google.protobuf import struct_pb2

# Imported from the defining modules, not from `biopb.image`: the package
# `__init__` imports this module.
from biopb.image._utils import (
    _is_dask_array,
    deserialize_image_data,
    serialize_from_numpy_to_image_data,
)
from biopb.image.image_data_pb2 import ImageData
from biopb.image.rpc_ops_pb2 import Arg

#: The axes an unlabelled array of each rank is taken to have.
NDIM_LABELS = {
    2: ["Y", "X"],
    3: ["Y", "X", "C"],
    4: ["Z", "Y", "X", "C"],
    5: ["T", "Z", "Y", "X", "C"],
}


def jsonable(value: Any) -> Any:
    """*value* as plain JSON types: numpy values converted, tables as columns.

    Raises ``TypeError`` for a type with no JSON form.
    """
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, np.ndarray):
        # Only complex needs the element walk: a bare complex isn't JSON.
        items = value.tolist()
        return jsonable(items) if value.dtype.kind == "c" else items
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):  # a pandas DataFrame or Series
        try:
            return jsonable(to_dict(orient="list"))
        except TypeError:
            return jsonable(to_dict())
    raise TypeError(f"a value of type {type(value).__name__} is not JSON")


def _fill_value(target: struct_pb2.Value, value: Any) -> None:
    """Set *target* from plain JSON types, field by field.

    Not ``json_format.ParseDict``/``MessageToDict``: those are JSON *text*
    guards, and a ``Value`` on the wire is protobuf binary, where
    ``number_value`` is a double that carries nan and inf unchanged -- an
    argument or a result may legitimately be either.
    """
    if value is None:
        target.null_value = struct_pb2.NULL_VALUE
    elif isinstance(value, bool):
        target.bool_value = value
    elif isinstance(value, (int, float)):
        target.number_value = value
    elif isinstance(value, str):
        target.string_value = value
    elif isinstance(value, dict):
        target.struct_value.Clear()
        for key, item in value.items():
            _fill_value(target.struct_value.fields[key], item)
    else:
        target.list_value.Clear()
        for item in value:
            _fill_value(target.list_value.values.add(), item)


def json_arg(value: Any) -> Arg:
    """*value* as a JSON ``Arg``."""
    arg = Arg()
    _fill_value(arg.json, jsonable(value))
    return arg


def json_value(value: struct_pb2.Value, *, ints: bool = False) -> Any:
    """A ``Value`` as Python.

    A protobuf number is always a double. With *ints*, an integral one comes
    back as an ``int``, so a count the sender wrote as 6 reads as 6, not 6.0.
    """
    kind = value.WhichOneof("kind")
    if kind == "number_value":
        number = value.number_value
        return int(number) if ints and number.is_integer() else number
    if kind == "string_value":
        return value.string_value
    if kind == "bool_value":
        return value.bool_value
    if kind == "struct_value":
        return {
            k: json_value(v, ints=ints) for k, v in value.struct_value.fields.items()
        }
    if kind == "list_value":
        return [json_value(v, ints=ints) for v in value.list_value.values]
    return None


def encode_arg(value: Any, *, dim_labels: Optional[Sequence[str]] = None) -> Arg:
    """*value* as an ``Arg``: an array as inline pixels, anything else as JSON.

    *dim_labels* name an array's axes; unlabelled, they follow ``NDIM_LABELS``
    for its rank. A dask array is computed.
    """
    if isinstance(value, np.ndarray) or _is_dask_array(value):
        array = np.asarray(value)
        labels = list(dim_labels) if dim_labels is not None else None
        if labels is None:
            labels = NDIM_LABELS.get(array.ndim)
        image_data = serialize_from_numpy_to_image_data(array, dim_labels=labels)
        return Arg(eager=image_data.eager_data)
    return json_arg(value)


def decode_arg(arg: Arg, *, ints: bool = True) -> Any:
    """An ``Arg`` as Python: JSON as plain values (see ``json_value`` for
    *ints*), inline pixels as a numpy array, a reference as a dask array.

    Reading a reference needs ``biopb[tensor]``.
    """
    kind = arg.WhichOneof("kind")
    if kind == "json":
        return json_value(arg.json, ints=ints)
    if kind == "eager":
        return deserialize_image_data(ImageData(eager_data=arg.eager))
    if kind == "lazy":
        from biopb.tensor.client import TensorFlightClient

        return TensorFlightClient.tensor_from_pb(arg.lazy)
    raise ValueError("an empty Arg has no value")
