"""A server of the ``Ops`` protocol, from decorated functions.

A server file declares its ops and serves them::

    from biopb_image_base import Tensor, op, serve

    @op(description="Mean intensity and area per label", labels=["measurement"])
    def label_stats(image: Tensor("YX"), labels: Tensor("YX")) -> dict:
        ...

    if __name__ == "__main__":
        serve()

Arguments come from the signature: a parameter annotated ``Tensor(axes)``, or
``Annotated[np.ndarray, Tensor(axes)]`` where the file is type-checked, is a
tensor argument, and every other parameter is a kwarg whose default is
advertised in ``OpInfo.kwargs``. ``axes`` is what the function sees, in that
order; the wrapper puts the input's other axes back on an array result of the
same rank.

``input`` says how pixels arrive:

* ``"eager"`` (default): numpy arrays. The other axes must be singleton, and a
  lazy input above ``EAGER_INPUT_CAP`` is refused rather than pulled whole.
* ``"lazy"``: dask arrays; the function may return one.
* ``"blocks"``: the function is mapped over blocks of the tensor arguments
  with ``map_overlap``, iterating the axes not in ``axes``. For pixelwise ops
  only: the output at a pixel may depend on a neighbourhood the overlap
  covers, and nothing else.

Both are advertised so a caller can decide how to call the op before trying
it: ``input`` as ``OpInfo.input``, and whether the function is a generator as
``OpInfo.streaming``. A process without pyarrow advertises a ``"lazy"`` op as
``"eager"``: it can decode no ``array_id``, so the op takes inline input only.

A single return value is the output ``result``; a tuple gives ``0``, ``1``,
.... An array is a tensor output, inline or on the sink by size, and anything
else is JSON. A function that yields is a streaming op: a yielded string is a
progress event, anything else one item's outputs, and a value it returns is
the final event's outputs.

Where large results go is the deployment's, not the call's: the embedded
tensor server under ``--cache-dir``, else the plane named by
``BIOPB_TENSOR_URL``/``BIOPB_TENSOR_TOKEN``, else inline.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import logging
import math
import os
import re
import sys
import threading
import typing
from concurrent import futures
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import biopb.image as proto
import grpc
import numpy as np
from biopb._web_auth import host_is_public_bind
from biopb.image import (
    deserialize_image_data,
    normalize_array_dims,
    serialize_from_numpy_to_image_data,
)
from google.protobuf import json_format, struct_pb2

from biopb_image_base.common import (
    _MAX_EAGER_SIZE,
    _MAX_MSG_SIZE,
    TokenValidationInterceptor,
    _is_dask_array,
    _pyarrow_available,
    server_context,
)

logger = logging.getLogger(__name__)

#: The largest lazy input an ``"eager"`` op pulls into memory.
EAGER_INPUT_CAP = 2 * 1024**3

#: The token a server checks, when its launcher sets one.
TOKEN_ENV = "BIOPB_ALGORITHM_TOKEN"

_INPUT_MODES = ("eager", "lazy", "blocks")
_SPATIAL_AXES = frozenset("ZYX")

# biopb's ndim -> axis-label convention (see biopb.image._utils).
_NDIM_LABELS = {
    2: ["Y", "X"],
    3: ["Y", "X", "C"],
    4: ["Z", "Y", "X", "C"],
    5: ["T", "Z", "Y", "X", "C"],
}


# =============================================================================
# Declaring ops
# =============================================================================


@dataclass(frozen=True)
class Tensor:
    """Annotates a tensor argument: ``Tensor("YX")`` is a 2D image.

    The string is the axes the function sees, in order. A call rather than a
    subscript, because linters and type checkers read a string in a subscript
    (``Tensor["YX"]``) as a forward reference to a type named ``YX``.
    """

    axes: str

    def __post_init__(self):
        if not isinstance(self.axes, str) or not self.axes:
            raise TypeError(f"Tensor takes an axes string like 'YX', got {self.axes!r}")
        axes = self.axes.upper()
        if len(set(axes)) != len(axes):
            raise TypeError(f"Tensor({axes!r}) repeats an axis")
        object.__setattr__(self, "axes", axes)


@dataclass
class _OpDef:
    name: str
    fn: Callable
    description: str
    labels: List[str]
    input: str
    block_shape: Optional[Dict[str, int]]
    overlap: Dict[str, int]
    dtype: Optional[np.dtype]
    tensors: Dict[str, str] = field(default_factory=dict)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    required: List[str] = field(default_factory=list)
    int_kwargs: List[str] = field(default_factory=list)
    streaming: bool = False

    def _kwargs_text(self) -> str:
        """The non-tensor arguments as they'd appear in a call: a comma-separated
        ``name=default`` per argument, bare ``name`` when required. Documentation
        for a caller -- nothing decodes this back, so a default that has no JSON
        shape (e.g. ``nan``) is exactly as fine here as any other repr."""
        required = set(self.required)
        return ", ".join(
            name if name in required else f"{name}={self.kwargs[name]!r}"
            for name in self.kwargs
        )

    def info(self) -> proto.OpInfo:
        # What this process can serve, not what the op declares: without
        # pyarrow no array_id decodes, so a lazy op takes inline input only.
        advertised = self.input
        if advertised == "lazy" and not _pyarrow_available():
            advertised = "eager"
        info = proto.OpInfo(
            name=self.name,
            description=self.description,
            labels=self.labels,
            kwargs=self._kwargs_text(),
            input=proto.OpInfo.InputMode.Value(advertised.upper()),
            streaming=self.streaming,
        )
        for name, axes in self.tensors.items():
            info.tensors[name].axes = axes
            info.tensors[name].mapped = self.input == "blocks"
        return info


#: Every op declared in this process, by name.
_REGISTRY: Dict[str, _OpDef] = {}


def _annotation(fn: Callable, param: inspect.Parameter) -> Any:
    """A parameter's annotation: evaluated when it is a string, and the
    ``Tensor`` inside an ``Annotated``.

    Under ``from __future__ import annotations`` every annotation is a string.
    Only ``Tensor(...)`` and ``int`` matter here, so one that does not evaluate
    (a name imported under TYPE_CHECKING) is simply not a tensor.
    """
    ann = param.annotation
    if isinstance(ann, str):
        try:
            ann = eval(ann, fn.__globals__)  # noqa: S307 - the server file's own source
        except Exception:
            return inspect.Parameter.empty
    if typing.get_origin(ann) is typing.Annotated:
        base, *metadata = typing.get_args(ann)
        return next((m for m in metadata if isinstance(m, Tensor)), base)
    return ann


def _per_axis(value, axes: str, what: str) -> Dict[str, int]:
    """An int for every axis, a sequence aligned with *axes*, or a dict by axis."""
    if value is None:
        return {}
    if isinstance(value, int):
        return {a: value for a in axes if a in _SPATIAL_AXES}
    if isinstance(value, dict):
        out = {str(k).upper(): int(v) for k, v in value.items()}
    else:
        value = list(value)
        if len(value) != len(axes):
            raise ValueError(f"{what} {value} does not match the axes {axes!r}")
        out = {a: int(v) for a, v in zip(axes, value, strict=True)}
    unknown = set(out) - set(axes)
    if unknown:
        raise ValueError(f"{what} names axes {sorted(unknown)} not in {axes!r}")
    return out


def _define(
    fn: Callable,
    *,
    name: Optional[str],
    description: Optional[str],
    labels: Sequence[str],
    input: str,  # noqa: A002 - the keyword the server file writes
    block_shape,
    overlap,
    dtype,
) -> _OpDef:
    if input not in _INPUT_MODES:
        raise ValueError(f"input must be one of {_INPUT_MODES}, got {input!r}")
    if description is None:
        description = inspect.cleandoc(fn.__doc__ or "").split("\n\n")[0]
    definition = _OpDef(
        name=name or fn.__name__,
        fn=fn,
        description=description,
        labels=list(labels),
        input=input,
        block_shape=None,
        overlap={},
        dtype=np.dtype(dtype) if dtype is not None else None,
        streaming=inspect.isgeneratorfunction(fn),
    )
    for param in inspect.signature(fn).parameters.values():
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            raise ValueError(
                f"{definition.name}: *args and **kwargs cannot be advertised"
            )
        ann = _annotation(fn, param)
        if isinstance(ann, Tensor):
            definition.tensors[param.name] = ann.axes
            continue
        if param.default is param.empty:
            definition.kwargs[param.name] = None
            definition.required.append(param.name)
        else:
            definition.kwargs[param.name] = param.default
        if ann is int or (
            isinstance(param.default, int) and not isinstance(param.default, bool)
        ):
            definition.int_kwargs.append(param.name)

    if input == "blocks":
        axes_set = set(definition.tensors.values())
        if len(axes_set) != 1:
            raise ValueError(
                f"{definition.name}: input='blocks' maps tensors of one shape, so "
                f"every tensor argument declares the same axes (got {sorted(axes_set)})"
            )
        if definition.streaming:
            raise ValueError(f"{definition.name}: a streaming op cannot be 'blocks'")
        axes = axes_set.pop()
        definition.block_shape = _per_axis(block_shape, axes, "block_shape") or None
        definition.overlap = _per_axis(overlap, axes, "overlap")
    elif block_shape is not None or overlap:
        raise ValueError(
            f"{definition.name}: block_shape and overlap apply to input='blocks' only"
        )
    return definition


def op(
    fn: Optional[Callable] = None,
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
    labels: Sequence[str] = (),
    input: str = "eager",  # noqa: A002 - reads naturally at the call site
    block_shape=None,
    overlap=0,
    dtype=None,
):
    """Declare a function as an op; ``serve()`` serves every declared op.

    Args:
        name: The op's name; the function's by default.
        description: One line for a client's listing; the docstring's first
            paragraph by default.
        labels: For grouping, e.g. ``["segmentation"]``.
        input: ``"eager"``, ``"lazy"`` or ``"blocks"``; see the module docs.
        block_shape: ``"blocks"`` only: the block size over the op's axes, as
            one int per axis or a dict by axis. Whole axes by default.
        overlap: ``"blocks"`` only: the halo each block is read with, an int
            for every spatial axis or per axis.
        dtype: ``"blocks"`` only: the output dtype. Without it the first block
            is computed once to learn it.

    The function stays callable as it was.
    """

    def register(f: Callable) -> Callable:
        definition = _define(
            f,
            name=name,
            description=description,
            labels=labels,
            input=input,
            block_shape=block_shape,
            overlap=overlap,
            dtype=dtype,
        )
        existing = _REGISTRY.get(definition.name)
        if existing is not None and existing.fn.__qualname__ != f.__qualname__:
            raise ValueError(f"two ops are named {definition.name!r}")
        _REGISTRY[definition.name] = definition
        f.__biopb_op__ = definition
        return f

    return register(fn) if fn is not None else register


def describe(definitions: Sequence[_OpDef]) -> proto.OpList:
    """The op list a server advertises, with its fingerprint."""
    ops = proto.OpList(
        ops=[d.info() for d in sorted(definitions, key=lambda d: d.name)]
    )
    digest = hashlib.sha256(ops.SerializeToString(deterministic=True)).hexdigest()
    ops.fingerprint = digest[:16]
    return ops


# =============================================================================
# Values
# =============================================================================


#: The key `_jsonable` carries a non-finite float under, and `_from_json` (the
#: kernel's `ops` client, `_process_ops.py`) reads it back from. JSON has no
#: literal for nan/inf/-inf, and `google.protobuf.Value` refuses to serialize
#: one to JSON text (`MessageToDict` raises) -- so a measurement that
#: legitimately returns nan (an empty-input rate, "undefined" not "zero") or
#: inf (an unbounded resolution) would otherwise crash decoding the result,
#: not just lose precision.
NON_FINITE_FLOAT_KEY = "__float__"


def _jsonable(value: Any) -> Any:
    """*value* as plain JSON types: numpy values converted, tables as columns,
    a non-finite float carried as ``{"__float__": "nan"}`` (see
    :data:`NON_FINITE_FLOAT_KEY`)."""
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, float) and not math.isfinite(value):
        return {NON_FINITE_FLOAT_KEY: str(value)}
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, np.ndarray):
        # Only a float array can hold a non-finite value; skip the per-element
        # walk below unless one is actually there. Complex still falls
        # through it, since a bare complex scalar isn't JSON either.
        if value.dtype.kind == "f" and np.isfinite(value).all():
            return value.tolist()
        if value.dtype.kind not in "fc":
            return value.tolist()
        return _jsonable(value.tolist())
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):  # a pandas DataFrame or Series
        try:
            return _jsonable(to_dict(orient="list"))
        except TypeError:
            return _jsonable(to_dict())
    raise TypeError(f"an output of type {type(value).__name__} is not JSON")


def _json_arg(value: Any) -> proto.Arg:
    return proto.Arg(json=json_format.ParseDict(_jsonable(value), struct_pb2.Value()))


@dataclass
class _Pixels:
    """A decoded tensor argument: the array and its axis labels."""

    array: Any
    labels: List[str]


def _decode_pixels(name: str, arg: proto.Arg) -> _Pixels:
    kind = arg.WhichOneof("kind")
    if kind == "eager":
        array = deserialize_image_data(proto.ImageData(eager_data=arg.eager))
        labels = [str(label).upper() for label in arg.eager.dim_labels]
    elif kind == "lazy":
        try:
            from biopb.tensor.client import TensorFlightClient
        except ImportError as exc:
            raise ValueError(
                f"{name}: a lazy input needs biopb-image-base[lazy] on the server"
            ) from exc
        array = TensorFlightClient.tensor_from_pb(arg.lazy)
        descriptor = TensorFlightClient.descriptor_from_pb(arg.lazy)
        labels = [str(label).upper() for label in descriptor.dim_labels]
    else:
        raise ValueError(f"{name} is a tensor argument; got {kind or 'nothing'}")
    if not labels:
        labels = _NDIM_LABELS.get(array.ndim)
        if labels is None:
            raise ValueError(f"{name}: a {array.ndim}D input needs its dim_labels")
    if len(labels) != array.ndim:
        raise ValueError(
            f"{name}: dim_labels {labels} do not match shape {array.shape}"
        )
    return _Pixels(array, list(labels))


def _undo_non_finite(value: Any) -> Any:
    """Restore a `NON_FINITE_FLOAT_KEY`-carried nan/inf/-inf to a real float."""
    if isinstance(value, dict) and set(value) == {NON_FINITE_FLOAT_KEY}:
        return float(value[NON_FINITE_FLOAT_KEY])
    if isinstance(value, list):
        return [_undo_non_finite(v) for v in value]
    if isinstance(value, dict):
        return {k: _undo_non_finite(v) for k, v in value.items()}
    return value


def _decode_kwarg(definition: _OpDef, name: str, arg: proto.Arg) -> Any:
    if arg.WhichOneof("kind") != "json":
        raise ValueError(f"{name} is not a tensor argument of {definition.name}")
    value = _undo_non_finite(json_format.MessageToDict(arg.json))
    if (
        name in definition.int_kwargs
        and isinstance(value, float)
        and value.is_integer()
    ):
        value = int(value)
    return value


def _to_axes(name: str, pixels: _Pixels, axes: str, mode: str):
    """The array as the function sees it: exactly *axes*, in order."""
    try:
        return normalize_array_dims(pixels.array, pixels.labels, list(axes))
    except ValueError as exc:
        raise ValueError(
            f"{name}: the op sees {axes!r} and the input is {''.join(pixels.labels)} "
            f"{tuple(pixels.array.shape)}. With input={mode!r} its other axes must "
            f"be singleton; slice the input, or declare input='blocks' for a "
            f"pixelwise op ({exc})"
        ) from exc


def _restore_axes(result, axes: str, like: Optional[_Pixels]):
    """An array result with the input's other axes put back, and its labels.

    Only a result of the op's own rank is taken to be on its axes; any other
    shape is returned as is.
    """
    if like is None or result.ndim != len(axes):
        return result, None
    return normalize_array_dims(result, list(axes), like.labels), like.labels


# =============================================================================
# Sinks: where a large result goes
# =============================================================================


def _regular_chunks(array):
    """*array* as dask on a regular grid (a smaller last chunk allowed)."""
    import dask.array as da

    if not _is_dask_array(array):
        chunks = da.core.normalize_chunks(
            "auto", array.shape, limit=16 * 1024**2, dtype=array.dtype
        )
        return da.from_array(array, chunks=chunks)
    return array.rechunk(tuple(max(c) for c in array.chunks))


def _field_name(op_name: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_-]+", "-", op_name).strip("-") or "result"
    return f"{stem}-{os.urandom(4).hex()}"


class _InlineSink:
    """Every result in the response."""

    def put(self, array, labels: Optional[List[str]], op_name: str) -> proto.Arg:
        if _is_dask_array(array):
            array = array.compute()
        array = np.asarray(array)
        if array.nbytes > _MAX_MSG_SIZE - 1024**2:
            raise ValueError(
                f"the result is {array.nbytes} bytes, more than one response carries; "
                "serve with --cache-dir, or next to a data plane, to return it by "
                "reference"
            )
        image_data = serialize_from_numpy_to_image_data(array, dim_labels=labels)
        return proto.Arg(eager=image_data.eager_data)


#: How many blocks ``upload_array``'s ``da.store`` may compute and hold in
#: flight at once. Left unset, dask's default threaded scheduler races ahead
#: of the network sink -- unbounded, and with no relation to how many CPUs
#: are free -- so a fine-grained result (``input="blocks"`` at a small
#: ``block_shape``) can pile up many more decoded, overlap-concatenated
#: blocks than the whole array's own size before the slow per-chunk upload
#: drains them (biopb/biopb#1148).
_UPLOAD_WORKERS = 4


def _ambient_scheduler_configured() -> bool:
    """Whether dask would already resolve to something other than its bare
    local default: a scheduler the deployment set itself
    (``dask.config.set``), or an active ``dask.distributed`` cluster.

    Deferred to ``dask.base.get_scheduler`` itself (with no explicit
    scheduler or collection, it checks exactly those two things and nothing
    else) rather than re-checked by hand, so this never drifts from what
    dask actually does. Blindly setting our own scheduler to bound upload
    concurrency would otherwise silently pull every upload off a configured
    cluster and onto local threads instead: ``dask.config.set(scheduler=...)``
    outranks an active distributed ``Client`` in dask's own precedence.
    """
    from dask.base import get_scheduler

    return get_scheduler() is not None


class _PlaneSink(_InlineSink):
    """Small arrays inline; dask and large arrays to the data plane named by
    ``BIOPB_TENSOR_URL``, as tensors of its scratch source."""

    def __init__(self, url: str, token: Optional[str]):
        self._url = url
        self._token = token
        self._client = None
        self._lock = threading.Lock()

    def _connect(self):
        with self._lock:
            if self._client is None:
                from biopb.tensor.client import TensorFlightClient

                self._client = TensorFlightClient(self._url, token=self._token)
            return self._client

    def put(self, array, labels, op_name):
        if not _is_dask_array(array) and np.asarray(array).nbytes <= _MAX_EAGER_SIZE:
            return super().put(array, labels, op_name)
        return proto.Arg(lazy=self._upload(_regular_chunks(array), labels, op_name))

    def _upload(self, array, labels, op_name):
        import dask

        client = self._connect()
        desc = client.setup_array_upload(
            f"cache://scratch/@fields/{_field_name(op_name)}",
            array,
            dim_labels=labels,
        )
        if _ambient_scheduler_configured():
            client.upload_array(desc, array)
        else:
            with dask.config.set(scheduler="threads", num_workers=_UPLOAD_WORKERS):
                client.upload_array(desc, array)
        return client.get_tensor(desc.array_id, output="pb")


class _EmbeddedSink(_PlaneSink):
    """Large results to this process's embedded tensor server, each with its
    own read token and a TTL."""

    def __init__(self, cache):
        super().__init__("", None)
        self._cache = cache

    def _upload(self, array, labels, op_name):
        array_id = self._cache.create_source(
            array, source_name=op_name, dim_labels=labels
        )
        return self._cache.to_serialized_tensor(array_id)


# =============================================================================
# Running a call
# =============================================================================


def _blocks(definition: _OpDef, pixels: Dict[str, _Pixels], kwargs: Dict[str, Any]):
    """Map the op over blocks of its tensor arguments; a dask result."""
    import dask.array as da

    axes = next(iter(definition.tensors.values()))
    names = list(definition.tensors)
    first = pixels[names[0]]
    others = [label for label in first.labels if label not in axes]
    order = others + list(axes)
    arrays = []
    for name in names:
        try:
            arr = normalize_array_dims(pixels[name].array, pixels[name].labels, order)
        except ValueError as exc:
            raise ValueError(f"{name}: {exc}") from exc
        if tuple(arr.shape) != tuple(arrays[0].shape if arrays else arr.shape):
            raise ValueError(
                f"input='blocks' maps tensors of one shape; {name} is "
                f"{tuple(arr.shape)} and {names[0]} {tuple(arrays[0].shape)}"
            )
        arrays.append(arr)

    shape = arrays[0].shape
    block = definition.block_shape or {}
    chunks = tuple(
        1 if i < len(others) else min(block.get(label, shape[i]), shape[i])
        for i, label in enumerate(order)
    )
    arrays = [
        (a if _is_dask_array(a) else da.from_array(a, chunks=chunks)).rechunk(chunks)
        for a in arrays
    ]
    depth = {
        i: definition.overlap.get(label, 0)
        for i, label in enumerate(order)
        if i >= len(others)
    }
    lead = (0,) * len(others)

    def apply(*blocks):
        inner = [b.reshape(b.shape[len(others) :]) for b in blocks]
        out = np.asarray(
            definition.fn(**dict(zip(names, inner, strict=True)), **kwargs)
        )
        if out.shape != inner[0].shape:
            raise ValueError(
                f"input='blocks' is for pixelwise ops: a {inner[0].shape} block "
                f"gave {out.shape}"
            )
        return out.reshape(blocks[0].shape)

    def mapped(dtype):
        return da.map_overlap(
            apply,
            *arrays,
            depth=depth,
            boundary="none",
            dtype=dtype,
            meta=np.empty((0,) * len(order), dtype=dtype),
        )

    if definition.dtype is not None:
        result = mapped(definition.dtype)
    else:
        # One block, computed to learn what the function returns -- an
        # empty-array probe (map_overlap's own default) is not safe for an
        # arbitrary op. Reused as the graph itself when the guess held, rather
        # than building the whole map_overlap a second time: at ~18k blocks
        # for a fine block_shape, that graph is not free (biopb/biopb#1148).
        # A wrong guess relabels the same graph rather than casting it: every
        # block's real bytes are already the true dtype (apply() returns
        # whatever the op actually produces, regardless of the guess passed
        # to build the graph) -- only the graph's own dtype/meta bookkeeping
        # was wrong, so .astype() would pay a real copy of the whole array to
        # fix a label, not a value.
        guess = mapped(arrays[0].dtype)
        dtype = guess.blocks[lead + (0,) * len(axes)].compute().dtype
        result = (
            guess
            if dtype == guess.dtype
            else da.Array(
                guess.dask,
                guess.name,
                guess.chunks,
                meta=np.empty((0,) * len(order), dtype=dtype),
            )
        )
    return normalize_array_dims(result, order, first.labels), first.labels


class _OpsServicer(proto.OpsServicer):
    """``Ops`` over a set of op definitions. Calls run one at a time."""

    def __init__(self, definitions: Sequence[_OpDef], sink, compress: bool):
        self._lock = threading.Lock()
        self._ops = {d.name: d for d in definitions}
        self._oplist = describe(definitions)
        self._sink = sink
        self._compress = compress

    def Describe(self, request, context):
        return self._oplist

    def Call(self, request, context):
        definition = self._ops.get(request.op)
        if definition is None:
            context.abort(
                grpc.StatusCode.NOT_FOUND,
                f"no op {request.op!r}; this server has {sorted(self._ops)}",
            )
        with server_context(context, self._lock):
            yield from self._run(definition, request, context)

    def _arguments(self, definition: _OpDef, request) -> Tuple[Dict, Dict]:
        unknown = set(request.args) - set(definition.tensors) - set(definition.kwargs)
        if unknown:
            raise ValueError(
                f"{definition.name} takes no argument {sorted(unknown)}; it takes "
                f"{sorted(definition.tensors) + sorted(definition.kwargs)}"
            )
        missing = [
            name
            for name in list(definition.tensors) + definition.required
            if name not in request.args
        ]
        if missing:
            raise ValueError(f"{definition.name} needs {missing}")
        pixels = {
            name: _decode_pixels(name, request.args[name])
            for name in definition.tensors
        }
        kwargs = {
            name: _decode_kwarg(definition, name, arg)
            for name, arg in request.args.items()
            if name in definition.kwargs
        }
        return pixels, kwargs

    def _inputs(self, definition: _OpDef, pixels: Dict[str, _Pixels]) -> Dict:
        inputs = {}
        for name, axes in definition.tensors.items():
            arr = _to_axes(name, pixels[name], axes, definition.input)
            if definition.input == "eager" and _is_dask_array(arr):
                if arr.nbytes > EAGER_INPUT_CAP:
                    raise ValueError(
                        f"{name} is {arr.nbytes} bytes, more than the "
                        f"{EAGER_INPUT_CAP} an input='eager' op reads whole; slice "
                        "the input, or the op declares input='lazy' or 'blocks'"
                    )
                arr = arr.compute()
            elif definition.input == "lazy" and not _is_dask_array(arr):
                import dask.array as da

                arr = da.from_array(arr, chunks=arr.shape)
            inputs[name] = arr
        return inputs

    def _event(self, definition, value, relabel, context) -> proto.Event:
        """One event of *value*'s outputs; ``relabel(array)`` answers an array
        output on the input's axes, and its labels."""
        values = value if isinstance(value, tuple) else (value,)
        keys = (
            [str(i) for i in range(len(values))]
            if isinstance(value, tuple)
            else ["result"]
        )
        event = proto.Event()
        compressible = False
        for key, item in zip(keys, values, strict=True):
            if isinstance(item, np.ndarray) or _is_dask_array(item):
                item, labels = relabel(item)
                event.outputs[key].CopyFrom(
                    self._sink.put(item, labels, definition.name)
                )
                if (
                    event.outputs[key].HasField("eager")
                    and np.dtype(item.dtype).kind in "iub"
                ):
                    compressible = True
            else:
                event.outputs[key].CopyFrom(_json_arg(item))
        if self._compress and not compressible:
            context.disable_next_message_compression()
        return event

    def _run(self, definition: _OpDef, request, context):
        pixels, kwargs = self._arguments(definition, request)
        if definition.input == "blocks":
            result, labels = _blocks(definition, pixels, kwargs)
            yield self._event(definition, result, lambda a: (a, labels), context)
            return

        first = next(iter(definition.tensors), None)
        if first is None:

            def relabel(array):
                return array, None
        else:

            def relabel(array):
                return _restore_axes(array, definition.tensors[first], pixels[first])

        inputs = self._inputs(definition, pixels)
        if not definition.streaming:
            result = definition.fn(**inputs, **kwargs)
            yield self._event(definition, result, relabel, context)
            return

        # A yielded string is progress; anything else is one item's outputs.
        # The generator's return value, if any, is the final event's outputs.
        items = definition.fn(**inputs, **kwargs)
        count = 0
        try:
            while True:
                try:
                    item = next(items)
                except StopIteration as stop:
                    result = stop.value
                    break
                if isinstance(item, str):
                    if self._compress:
                        context.disable_next_message_compression()
                    yield proto.Event(progress=item)
                    continue
                count += 1
                event = self._event(definition, item, relabel, context)
                event.progress = str(count)
                yield event
            if result is not None:
                yield self._event(definition, result, relabel, context)
        finally:
            items.close()


# =============================================================================
# Serving
# =============================================================================


def build_server(
    definitions: Sequence[_OpDef],
    *,
    host: str = "127.0.0.1",
    port: int = 0,
    sink=None,
    token: Optional[str] = None,
    workers: int = 4,
) -> Tuple[grpc.Server, int]:
    """A gRPC server of *definitions*, not started, and the port it bound.

    Off loopback, integer outputs are gzipped: label images shrink 25-50x on
    the wire, while on loopback compression costs more than it saves.
    """
    from biopb_image_base.health import add_health_servicer

    remote = host_is_public_bind(host)
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=workers),
        compression=grpc.Compression.Gzip if remote else grpc.Compression.NoCompression,
        interceptors=(TokenValidationInterceptor(token),),
        options=(
            ("grpc.max_receive_message_length", _MAX_MSG_SIZE),
            ("grpc.max_send_message_length", _MAX_MSG_SIZE),
        ),
    )
    servicer = _OpsServicer(definitions, sink or _InlineSink(), compress=remote)
    proto.add_OpsServicer_to_server(servicer, server)
    add_health_servicer(server)
    bound = server.add_insecure_port(f"{host}:{port}")
    if not bound:
        raise OSError(f"could not bind {host}:{port}")
    return server, bound


def _sink_from(args) -> Any:
    if args.cache_dir:
        from biopb_image_base.server import start_embedded_cache

        cache = start_embedded_cache(
            args.cache_dir,
            args.cache_size,
            ip=args.host,
            local=not host_is_public_bind(args.host),
            tensor_port=args.tensor_port,
            tensor_external_location=args.tensor_external_location,
        )
        return _EmbeddedSink(cache) if cache is not None else _InlineSink()
    url = os.environ.get("BIOPB_TENSOR_URL")
    if url:
        return _PlaneSink(url, os.environ.get("BIOPB_TENSOR_TOKEN") or None)
    return _InlineSink()


def serve(
    argv: Optional[Sequence[str]] = None, *, ops: Optional[Sequence] = None
) -> None:
    """Serve the declared ops until the process is stopped.

    ``--describe`` prints the op list as JSON and exits without binding a port.
    The token is ``$BIOPB_ALGORITHM_TOKEN``; a server bound off loopback without
    one mints one and prints it.

    Args:
        argv: The command line; ``sys.argv[1:]`` by default.
        ops: The functions to serve; every ``@op`` in the process by default.
    """
    from biopb_image_base.logging_config import get_log_level_from_env, setup_logging

    parser = argparse.ArgumentParser(description="Serve biopb ops.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument(
        "--describe", action="store_true", help="print the op list and exit"
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--cache-dir", help="return large results through an embedded tensor server"
    )
    parser.add_argument("--cache-size", default="32GB")
    parser.add_argument("--tensor-port", type=int, default=8817)
    parser.add_argument("--tensor-external-location")
    args = parser.parse_args(argv)

    if ops is None:
        definitions = list(_REGISTRY.values())
    else:
        undeclared = [f for f in ops if not hasattr(f, "__biopb_op__")]
        if undeclared:
            raise ValueError(f"not declared with @op: {undeclared}")
        definitions = [f.__biopb_op__ for f in ops]
    if not definitions:
        parser.error("no ops declared; decorate a function with @op")

    if args.describe:
        print(json_format.MessageToJson(describe(definitions)))
        return

    setup_logging(get_log_level_from_env())
    # Under the control, die with it: it passes a parent-death pipe, so a
    # control that dies uncatchably leaves no server holding a GPU.
    from biopb._lifecycle import deathwatch

    deathwatch.install()
    token = os.environ.get(TOKEN_ENV) or None
    if token is None and host_is_public_bind(args.host):
        import secrets

        token = secrets.token_urlsafe(32)
        print(f"Token (send as 'authorization: Bearer <token>'): {token}", flush=True)

    server, port = build_server(
        definitions,
        host=args.host,
        port=args.port,
        sink=_sink_from(args),
        token=token,
        workers=args.workers,
    )
    server.start()
    logger.info(
        "serving %s on %s:%d",
        ", ".join(sorted(d.name for d in definitions)),
        args.host,
        port,
    )
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        server.stop(grace=2).wait()
        sys.exit(0)
