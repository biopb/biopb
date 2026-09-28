"""The kernel's ``ops``: the algorithm plane's ops as callables.

The control names the servers (``biopb.algorithms()``): script entries
it runs under uv, and url entries someone else runs. Every op they advertise
becomes a callable in ``ops``, bound from the op lists the control cached, so
binding starts nothing; a script entry's server starts on the first call to
one of its ops.

A call is one ``Ops.Call`` stream. Arguments go by name; a tensor argument
takes an ``np.ndarray`` (sent inline) or a tensor-server ``array_id`` (sent as
a reference, which the server reads from the plane itself). A tensor result
comes back as an ``np.ndarray``, or as an ``array_id`` when any input was one
or the server returned it by reference: a result already on the kernel's plane
keeps its id, and any other is copied into the plane's scratch source. There
is no call deadline: the call gives up after ``timeout.process_image`` seconds
without an event, and a Stop cancels it on the server.
"""

from __future__ import annotations

import logging
import math
import os
import queue
import re
import threading
import time
from collections.abc import Callable
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import biopb.image as proto
import dask.array as da
import grpc
import numpy as np
from biopb.image.utils import (
    deserialize_image_data,
    serialize_from_numpy_to_image_data,
)
from biopb.tensor._location import same_location
from google.protobuf import json_format, struct_pb2

from .._config import get_setting

logger = logging.getLogger(__name__)

# biopb's ndim -> axis-label convention (see biopb.image.utils).
_NDIM_LABELS = {
    2: ["Y", "X"],
    3: ["Y", "X", "C"],
    4: ["Z", "Y", "X", "C"],
    5: ["T", "Z", "Y", "X", "C"],
}

#: The tensor server's scratch source, which every writable server serves at
#: this fixed id. An upload adds a tensor to a source that already exists, and
#: an op result belongs to no source of the user's, so this is where it goes.
SCRATCH_SOURCE_ID = "scratch"

#: How often a long call prints its progress.
_PROGRESS_EVERY_S = 5.0

#: How long ``ensure`` may take: a first start installs, and may pull torch.
_ENSURE_TIMEOUT = 900.0


def _make_channel(url: str, options=None) -> grpc.Channel:
    """Build a gRPC channel from a ``grpc://`` or ``grpcs://`` URL."""
    parsed = urlparse(url)
    scheme = parsed.scheme.lower()
    target = parsed.netloc or parsed.path
    if not target:
        raise ValueError(f"algorithm server URL has no host: {url!r}")
    if scheme == "grpcs":
        return grpc.secure_channel(
            target, grpc.ssl_channel_credentials(), options=options
        )
    if scheme == "grpc":
        return grpc.insecure_channel(target, options=options)
    raise ValueError(f"algorithm server URL must be grpc:// or grpcs://, got {url!r}")


def _sanitize_name(name: str) -> str:
    return re.sub(r"\W", "_", name) or "op"


# The sentinel a non-finite float (nan/inf/-inf) is carried under -- JSON has
# no literal for one, and `google.protobuf.Value` refuses to serialize one to
# JSON text at all (`MessageToDict` raises), so an argument or a result that
# legitimately is one would otherwise crash rather than lose precision. Mirrors
# `biopb_image_base.ops.NON_FINITE_FLOAT_KEY`, on the server side of this same
# wire protocol; not imported from there; the two sides share no Python code.
_NON_FINITE_FLOAT_KEY = "__float__"


def _jsonable(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return {_NON_FINITE_FLOAT_KEY: str(value)}
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
    return value


def _from_json(value: Any) -> Any:
    """A JSON result with its integral numbers as ints: JSON has only doubles,
    so a count the server sent as 6 arrives as 6.0. Also restores a
    `_NON_FINITE_FLOAT_KEY`-carried nan/inf/-inf to a real float."""
    if isinstance(value, dict) and set(value) == {_NON_FINITE_FLOAT_KEY}:
        return float(value[_NON_FINITE_FLOAT_KEY])
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, list):
        return [_from_json(v) for v in value]
    if isinstance(value, dict):
        return {k: _from_json(v) for k, v in value.items()}
    return value


def _refusal(name: str, exc: grpc.RpcError) -> Exception:
    """The exception an op's failure raises in the kernel: the server's
    message, typed by what went wrong rather than wrapped in gRPC's repr."""
    detail = (exc.details() or "").strip() or exc.code().name
    code = exc.code()
    if code == grpc.StatusCode.INVALID_ARGUMENT:
        return ValueError(f"{name}: {detail}")
    if code == grpc.StatusCode.NOT_FOUND:
        return LookupError(f"{name}: {detail}")
    return RuntimeError(f"{name}: {code.name}: {detail}")


_same_plane = same_location


class _Server:
    """One registry entry the kernel calls: where it is and as whom."""

    def __init__(self, row: dict, channel_options):
        self.name = row["name"]
        self.kind = row["kind"]
        self._options = channel_options
        self._lock = threading.Lock()
        self._url = row.get("url") if row.get("state") == "up" else None
        self._token = row.get("token")
        if self.kind == "url":
            self._url = row.get("url")
        self._stub = None

    def stub(self):
        """The stub and call metadata, starting a script entry if needed."""
        with self._lock:
            if self._url is None:
                from biopb import ensure_algorithm

                row = ensure_algorithm(self.name, timeout=_ENSURE_TIMEOUT)
                if row["state"] != "up":
                    detail = (row.get("error") or "").strip()
                    raise RuntimeError(
                        f"algorithm server {self.name!r} is {row['state']}"
                        + (f":\n{detail}" if detail else "")
                        + f"\nSee ops.status() and ops.logs({self.name!r})."
                    )
                self._url, self._token, self._stub = row["url"], row["token"], None
            if self._stub is None:
                self._stub = proto.OpsStub(_make_channel(self._url, self._options))
            metadata = (
                [("authorization", f"Bearer {self._token}")] if self._token else None
            )
            return self._stub, metadata

    def forget(self) -> bool:
        """Drop a script entry's address, so the next call ensures it again
        (it may have restarted on another port). False for a url entry."""
        with self._lock:
            if self.kind != "script":
                return False
            self._url = self._token = self._stub = None
            return True


class _OpCall:
    """What one op needs to run: its server, its schema, the kernel's plane."""

    def __init__(self, server, info: dict, client_getter, inactivity_timeout):
        self.server = server
        self.info = info
        self.name = info["name"]
        self.tensors = {
            k: v.get("axes", "") for k, v in (info.get("tensors") or {}).items()
        }
        self.client_getter = client_getter
        self.inactivity_timeout = inactivity_timeout

    # --- arguments ------------------------------------------------------- #

    def _tensor(self, name, value, labels, client) -> proto.Arg:
        if isinstance(value, str):
            if client is None:
                raise RuntimeError(
                    f"No tensor server connected; cannot resolve array_id {value!r}."
                )
            return proto.Arg(lazy=client.get_tensor_pb(value))
        arr = np.asarray(value)
        if isinstance(labels, dict):
            labels = labels.get(name)
        labels = list(labels) if labels is not None else _NDIM_LABELS.get(arr.ndim)
        image_data = serialize_from_numpy_to_image_data(arr, dim_labels=labels)
        return proto.Arg(eager=image_data.eager_data)

    def arguments(self, args, kwargs, dim_labels) -> tuple[Dict[str, proto.Arg], bool]:
        if args:
            if len(args) > 1 or len(self.tensors) != 1:
                names = ", ".join(sorted(self.tensors))
                kwargs_text = self.info.get("kwargs") or ""
                if kwargs_text:
                    names = f"{names}, {kwargs_text}" if names else kwargs_text
                raise TypeError(f"{self.name} takes its arguments by name: {names}")
            kwargs = {next(iter(self.tensors)): args[0], **kwargs}
        client = self.client_getter()
        encoded, by_id = {}, False
        for name, value in kwargs.items():
            if name in self.tensors:
                by_id = by_id or isinstance(value, str)
                encoded[name] = self._tensor(name, value, dim_labels, client)
            else:
                encoded[name] = proto.Arg(
                    json=json_format.ParseDict(_jsonable(value), struct_pb2.Value())
                )
        return encoded, by_id

    # --- the stream ------------------------------------------------------ #

    def stream(self, arguments: Dict[str, proto.Arg]) -> List[proto.Event]:
        """The events that carry outputs, reading until the stream ends."""
        for attempt in (1, 2):
            stub, metadata = self.server.stub()
            try:
                return self._read(
                    stub.Call(
                        proto.Call(op=self.name, args=arguments), metadata=metadata
                    )
                )
            except grpc.RpcError as exc:
                # A managed server that restarted is on another port, with
                # another token: find it again, once.
                if (
                    attempt == 1
                    and exc.code()
                    in (grpc.StatusCode.UNAVAILABLE, grpc.StatusCode.UNAUTHENTICATED)
                    and self.server.forget()
                ):
                    continue
                raise
        raise AssertionError("unreachable")

    def _read(self, call) -> List[proto.Event]:
        events: queue.Queue = queue.Queue()

        def pump():
            try:
                for event in call:
                    events.put(("event", event))
                events.put(("end", None))
            except grpc.RpcError as exc:
                events.put(("error", exc))

        threading.Thread(target=pump, name=f"op-{self.name}", daemon=True).start()
        results: List[proto.Event] = []
        last_print = time.monotonic()
        try:
            while True:
                try:
                    kind, item = events.get(timeout=self.inactivity_timeout)
                except queue.Empty:
                    raise TimeoutError(
                        f"{self.name}: no word from the server in "
                        f"{self.inactivity_timeout:g} s (timeout.process_image)"
                    ) from None
                if kind == "end":
                    return results
                if kind == "error":
                    raise item
                if item.outputs:
                    results.append(item)
                elif (
                    item.progress and time.monotonic() - last_print >= _PROGRESS_EVERY_S
                ):
                    print(f"{self.name}: {item.progress}", flush=True)
                    last_print = time.monotonic()
        finally:
            # A Stop (KeyboardInterrupt), a timeout, or an error: the server
            # stops computing for a caller that left. A no-op once it ended.
            call.cancel()

    # --- results --------------------------------------------------------- #

    def _upload(self, client, array) -> str:
        if not isinstance(array, da.Array):
            array = da.from_array(array, chunks=array.shape)
        field = f"{_sanitize_name(self.name)}-{os.urandom(4).hex()}"
        desc = client.add_tensor(f"cache://{SCRATCH_SOURCE_ID}/@fields/{field}", array)
        client.upload_array(desc, array)
        return desc.array_id

    def value(self, arg: proto.Arg, by_id: bool):
        kind = arg.WhichOneof("kind")
        if kind == "json":
            return _from_json(json_format.MessageToDict(arg.json))
        client = self.client_getter()
        if kind == "eager":
            array = deserialize_image_data(proto.ImageData(eager_data=arg.eager))
            return (
                self._upload(client, array) if by_id and client is not None else array
            )
        from biopb.tensor.client import TensorFlightClient

        if client is not None and _same_plane(arg.lazy.location, client.location):
            return TensorFlightClient.descriptor_from_pb(arg.lazy).array_id
        array = TensorFlightClient.tensor_from_pb(arg.lazy)
        if client is None:
            return array.compute()
        return self._upload(client, array)

    def result(self, event: proto.Event, by_id: bool):
        outputs = event.outputs
        if set(outputs) == {"result"}:
            return self.value(outputs["result"], by_id)
        keys = sorted(
            outputs, key=lambda k: (not k.isdigit(), int(k) if k.isdigit() else 0, k)
        )
        return tuple(self.value(outputs[k], by_id) for k in keys)


def _build_op(call: _OpCall) -> Callable:
    info = call.info

    def op(*args, dim_labels=None, **kwargs):
        arguments, by_id = call.arguments(args, kwargs, dim_labels)
        try:
            events = call.stream(arguments)
        except grpc.RpcError as exc:
            raise _refusal(call.name, exc) from None
        values = [call.result(event, by_id) for event in events]
        if not values:
            return None
        return values[0] if len(values) == 1 else values

    tensors = ", ".join(f"{k}: {v}" for k, v in call.tensors.items()) or "none"
    doc = [
        info.get("description") or f"The {call.name} op.",
        "",
        f"Server: {call.server.name} ({call.server.kind} entry)",
        f"Tensor arguments (axes the op sees): {tensors}",
    ]
    if info.get("labels"):
        doc.append(f"Labels: {', '.join(info['labels'])}")
    if info.get("kwargs"):
        doc.append(f"Other arguments (a bare name is required): {info['kwargs']}")
    mode = info.get("input", "EAGER")
    if mode == "LAZY":
        doc.append(
            "Input: lazy -- a large array_id is accepted without a size cap; "
            "the op itself may stay out-of-core."
        )
    elif mode == "BLOCKS":
        doc.append("Input: computed over blocks internally; call it like any other op.")
    else:
        doc.append(
            "Input: eager -- a lazy array_id over ~2GiB is refused, not pulled whole."
        )
    if info.get("streaming"):
        doc.append("Streaming: this call yields more than one result; expect a list.")
    doc += [
        "",
        "Arguments go by name; one tensor argument may also go first.",
        "  A tensor is an np.ndarray (sent inline; axes by ndim: 2D=YX, 3D=YXC,",
        "  4D=ZYXC, 5D=TZYXC, or dim_labels='ZYX' / {name: axes}) or an array_id",
        "  str (the server reads it from the tensor server).",
        "Tensor results are np.ndarray, or array_id str when an input was one or",
        "the server returned a reference. A tuple for several outputs; a list",
        "when a streaming op sent several events.",
    ]
    op.__doc__ = "\n".join(doc)
    op.__name__ = _sanitize_name(call.name)
    op.op_name = call.name
    op.server = call.server.name
    op.labels = list(info.get("labels") or [])
    op.description = info.get("description", "")
    op.kwargs_text = info.get("kwargs") or ""
    op.tensors = dict(call.tensors)
    op.input_mode = mode.lower()
    op.streaming = bool(info.get("streaming"))
    return op


class Ops:
    """The algorithm plane's ops, by name (``ops["gaussian"]``), and as
    attributes where the name is not one of the methods below.

    ``refresh()`` after adding or editing a server file, ``status()`` for each
    server's state, ``logs(name)`` and ``restart(name)`` for a server file the
    control runs. Not a dict, so an op may be named ``items`` or ``get``.
    """

    def __init__(
        self,
        client_getter: Callable[[], object],
        *,
        inactivity_timeout: float = 300.0,
        channel_options=None,
    ):
        self._client_getter = client_getter
        self._inactivity_timeout = inactivity_timeout
        self._options = channel_options
        self._ops: Dict[str, Callable] = {}

    # --- by name --------------------------------------------------------- #

    def __getitem__(self, name):
        return self._ops[name]

    def __iter__(self):
        return iter(self._ops)

    def __len__(self):
        return len(self._ops)

    def __contains__(self, name):
        return name in self._ops

    def __getattr__(self, name):
        ops = self.__dict__.get("_ops", {})
        if name in ops:
            return ops[name]
        raise AttributeError(
            f"no op {name!r}; ops has {sorted(ops)}. After adding a server file, "
            "call ops.refresh()."
        )

    def __dir__(self):
        return sorted(set(super().__dir__()) | set(self._ops))

    def __repr__(self):
        return f"<ops: {', '.join(sorted(self._ops)) or 'none'}>"

    # --- binding --------------------------------------------------------- #

    def bind(self, rows: Optional[List[dict]]) -> None:
        """Bind every op the rows advertise. An op name two servers share is
        bound as ``<server>_<op>`` for both."""
        entries = []
        for row in rows or []:
            if not row.get("ops"):
                continue
            server = _Server(row, self._options)
            for info in row["ops"]:
                entries.append((server, info))
        counts: Dict[str, int] = {}
        for _server, info in entries:
            counts[info["name"]] = counts.get(info["name"], 0) + 1
        ops: Dict[str, Callable] = {}
        for server, info in entries:
            key = _sanitize_name(info["name"])
            if counts[info["name"]] > 1:
                key = _sanitize_name(f"{server.name}_{info['name']}")
            call = _OpCall(server, info, self._client_getter, self._inactivity_timeout)
            ops[key] = _build_op(call)
        self._ops = ops

    # --- the control's verbs --------------------------------------------- #

    def refresh(self) -> str:
        """Have the control install and describe new or edited server files,
        rebind, and say what changed. An entry still installing binds on a
        later refresh."""
        from biopb import refresh_algorithms

        before = set(self._ops)
        rows = refresh_algorithms()
        if rows is None:
            return "No control answered; ops are unchanged."
        self.bind(rows)
        added = sorted(set(self._ops) - before)
        removed = sorted(before - set(self._ops))
        lines = [f"ops: {', '.join(sorted(self._ops)) or 'none'}"]
        if added:
            lines.append(f"added: {', '.join(added)}")
        if removed:
            lines.append(f"removed: {', '.join(removed)}")
        pending = [r["name"] for r in rows if r["state"] in ("new", "installing")]
        if pending:
            lines.append(
                f"still installing: {', '.join(pending)} (refresh again later)"
            )
        failed = [r["name"] for r in rows if r["state"] in ("failed", "invalid")]
        if failed:
            lines.append(f"failed: {', '.join(failed)} (see ops.status())")
        return "\n".join(lines)

    def status(self) -> str:
        """Each server: its kind, state, ops, and the first line of any error."""
        from biopb import algorithms

        rows = algorithms()
        if rows is None:
            return "No control answered: the algorithm plane is unavailable."
        if not rows:
            return (
                "No algorithm servers. Add a server file to "
                "~/.config/biopb/algorithms/ and call ops.refresh()."
            )
        lines = []
        for r in rows:
            names = ", ".join(o["name"] for o in r["ops"]) or "-"
            lines.append(f"{r['name']} ({r['kind']}): {r['state']}; ops: {names}")
            if r.get("error"):
                lines.append("  error: " + r["error"].strip().splitlines()[0])
        return "\n".join(lines)

    def logs(self, name: str, lines: int = 50) -> str:
        """The tail of a server file's log: its install and its server's output."""
        from biopb import algorithm_logs

        return "\n".join(algorithm_logs(name, lines=lines))

    def restart(self, name: str) -> str:
        """Restart a server file's server (picking up an edit), and rebind."""
        from biopb import algorithms, restart_algorithm

        row = restart_algorithm(name, timeout=_ENSURE_TIMEOUT)
        self.bind(algorithms())
        error = (row.get("error") or "").strip()
        return f"{name}: {row['state']}" + (f"\n{error}" if error else "")


def build_ops_from_config(config: dict, client_getter: Callable[[], object]) -> Ops:
    """The kernel's ``ops``, bound from the control's registry.

    The kernel bootstrap, ``workflow_env`` and an exported notebook's
    bootstrap cell all call this, so the wiring is in one place.
    ``client_getter`` is called at op-call time, so the asynchronously
    connecting tensor client is picked up live. With no control, ``ops`` is
    empty and ``ops.status()`` says why.
    """
    from biopb import algorithms

    max_msg_bytes = get_setting(config, "grpc.max_message_size_mb") * 1024 * 1024
    ops = Ops(
        client_getter,
        inactivity_timeout=get_setting(config, "timeout.process_image"),
        channel_options=[
            ("grpc.max_receive_message_length", max_msg_bytes),
            ("grpc.max_send_message_length", max_msg_bytes),
        ],
    )
    try:
        rows = algorithms(timeout=get_setting(config, "timeout.get_op_names"))
    except Exception:  # noqa: BLE001 - an unreadable registry never blocks a kernel
        logger.exception("could not list the algorithm servers")
        rows = None
    ops.bind(rows)
    return ops
