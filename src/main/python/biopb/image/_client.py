"""A client of an algorithm server: the ``Ops`` protocol over gRPC."""

from __future__ import annotations

import queue
import threading
from typing import Any, Callable, Iterator, Mapping, Optional
from urllib.parse import urlparse

import grpc
from google.protobuf import empty_pb2

from biopb.image._arg import decode_arg, encode_arg
from biopb.image.rpc_ops_pb2 import Arg, Call, Event, OpList
from biopb.image.rpc_ops_pb2_grpc import OpsStub

#: How long ``connect`` may wait for the control to bring a registry entry up: a
#: first start installs, and may pull torch.
_ENSURE_TIMEOUT = 900.0


def make_channel(url: str, options=None) -> grpc.Channel:
    """A gRPC channel to a ``grpc://`` or ``grpcs://`` URL."""
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


def op_error(op: str, exc: grpc.RpcError) -> Exception:
    """The exception a failed call raises: the server's message, typed by what
    went wrong rather than wrapped in gRPC's repr. ``ValueError`` for arguments
    the op refuses, ``LookupError`` for an op the server lacks, ``RuntimeError``
    otherwise."""
    code = exc.code()
    detail = (exc.details() or "").strip() or code.name
    if code == grpc.StatusCode.INVALID_ARGUMENT:
        return ValueError(f"{op}: {detail}")
    if code == grpc.StatusCode.NOT_FOUND:
        return LookupError(f"{op}: {detail}")
    return RuntimeError(f"{op}: {code.name}: {detail}")


class OpsClient:
    """One algorithm server. Use :func:`connect` to make one.

    A call has no deadline: ``inactivity_timeout`` bounds the silence between
    events instead, and leaving a call (an error, a ``break``, a timeout)
    cancels it on the server.
    """

    def __init__(
        self,
        url: str,
        *,
        token: Optional[str] = None,
        options=None,
        inactivity_timeout: Optional[float] = None,
    ):
        self.url = url
        self.inactivity_timeout = inactivity_timeout
        self._channel = make_channel(url, options)
        self._stub = OpsStub(self._channel)
        self._metadata = [("authorization", f"Bearer {token}")] if token else None

    def close(self) -> None:
        self._channel.close()

    def __enter__(self) -> OpsClient:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def describe(self, timeout: float = 10.0) -> OpList:
        """The ops the server offers."""
        return self._stub.Describe(
            empty_pb2.Empty(), metadata=self._metadata, timeout=timeout
        )

    def events(self, op: str, args: Mapping[str, Arg]) -> Iterator[Event]:
        """Every event of one call, progress-only ones included, until the
        server ends the stream. Raises ``grpc.RpcError`` for a failed call and
        ``TimeoutError`` after ``inactivity_timeout`` seconds of silence."""
        call = self._stub.Call(Call(op=op, args=dict(args)), metadata=self._metadata)
        items: queue.Queue = queue.Queue()

        def pump():
            try:
                for event in call:
                    items.put(("event", event))
                items.put(("end", None))
            except grpc.RpcError as exc:
                items.put(("error", exc))

        threading.Thread(target=pump, name=f"op-{op}", daemon=True).start()
        try:
            while True:
                try:
                    kind, item = items.get(timeout=self.inactivity_timeout)
                except queue.Empty:
                    raise TimeoutError(
                        f"{op}: no word from the server in "
                        f"{self.inactivity_timeout:g} s"
                    ) from None
                if kind == "end":
                    return
                if kind == "error":
                    raise item
                yield item
        finally:
            # A stop, a timeout or an error: the server stops computing for a
            # caller that left. A no-op once the call ended.
            call.cancel()

    def call(
        self,
        op: str,
        *,
        on_progress: Optional[Callable[[str], None]] = None,
        dim_labels=None,
        **values: Any,
    ) -> Any:
        """Run one op on *values*, by name: arrays go as pixels, anything else
        as JSON (see :func:`encode_arg`; *dim_labels* labels the arrays).

        Returns the op's result: its single output, or a tuple of them in
        order for several; ``None`` for no output. A streaming op returns a list
        with one such value per event. A failed call raises what
        :func:`op_error` says.
        """
        args = {
            name: encode_arg(v, dim_labels=dim_labels) for name, v in values.items()
        }
        results = []
        try:
            for event in self.events(op, args):
                if event.outputs:
                    results.append(_result(event))
                elif event.progress and on_progress is not None:
                    on_progress(event.progress)
        except grpc.RpcError as exc:
            raise op_error(op, exc) from None
        if not results:
            return None
        return results[0] if len(results) == 1 else results


def _result(event: Event) -> Any:
    outputs = event.outputs
    if set(outputs) == {"result"}:
        return decode_arg(outputs["result"])
    keys = sorted(
        outputs, key=lambda k: (not k.isdigit(), int(k) if k.isdigit() else 0, k)
    )
    return tuple(decode_arg(outputs[k]) for k in keys)


def connect(
    target: str,
    *,
    token: Optional[str] = None,
    options=None,
    inactivity_timeout: Optional[float] = None,
) -> OpsClient:
    """A client of the algorithm server at *target*: a ``grpc://`` or
    ``grpcs://`` URL, or the name of a registry entry the control manages.

    A name is brought up through the control first (installing it if needed)
    and takes the server's own token; an explicit *token* overrides it.
    """
    if "://" not in target:
        from biopb import ensure_algorithm

        row = ensure_algorithm(target, timeout=_ENSURE_TIMEOUT)
        if row["state"] != "up":
            detail = (row.get("error") or "").strip()
            raise RuntimeError(
                f"algorithm server {target!r} is {row['state']}"
                + (f":\n{detail}" if detail else "")
            )
        target, token = row["url"], token or row.get("token")
    return OpsClient(
        target, token=token, options=options, inactivity_timeout=inactivity_timeout
    )
