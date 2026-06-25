"""Unit tests for the _ServerContextMeta auto-wrapping of RPC handlers.

These run in-process against a fake gRPC context, so they don't need a live
server. They lock down the backward-compatibility contract:

- RPC handlers are auto-wrapped in _server_context (no boilerplate needed).
- Streaming handlers are wrapped as generators (yield from), so the context
  spans the whole stream.
- An explicit in-body `with self._server_context(context)` still works and
  translates errors exactly once (reentrancy guard).
- A handler that calls context.abort() directly keeps its status code; it is
  not downgraded to INTERNAL.
"""

import inspect

import grpc
import pytest

from biopb_image_base.common import BiopbServicerBase, abort_invalid_argument
from biopb_image_base.mock_servicer import MockServicer


class FakeContext:
    """Minimal stand-in for grpc._server._Context.

    Mirrors the two behaviors the servicer relies on: code() returns None until
    a status is set, and abort() records the status then raises a *bare*
    Exception (exactly as grpc._server._Context.abort does).
    """

    def __init__(self):
        self._code = None
        self._details = None

    def abort(self, code, details):
        self._code = code
        self._details = details
        raise Exception()  # noqa: TRY002 - mimic grpc's bare-Exception abort

    def code(self):
        return self._code


# --- Servicers under test (defined without the boilerplate `with` block) ----


class _RaisesValueError(BiopbServicerBase):
    def RunDetection(self, request, context):
        raise ValueError("bad arg")


class _ExplicitBoilerplate(BiopbServicerBase):
    def RunDetection(self, request, context):
        with self._server_context(context):  # still allowed; becomes pass-through
            raise ValueError("explicit")


class _DirectAbort(BiopbServicerBase):
    def Run(self, request, context):
        abort_invalid_argument(context, "missing image")


class _StreamRaises(BiopbServicerBase):
    def RunStream(self, request_iterator, context):
        yield  # produce one item, then fail
        raise ValueError("stream boom")


def test_rpc_handlers_are_wrapped():
    # Unary and streaming handlers carry the wrap marker...
    assert getattr(MockServicer.RunDetection, "__biopb_wrapped__", False)
    assert getattr(MockServicer.Run, "__biopb_wrapped__", False)
    assert getattr(MockServicer.RunStream, "__biopb_wrapped__", False)
    assert getattr(MockServicer.GetOpNames, "__biopb_wrapped__", False)
    # ...and the streaming wrapper is itself a generator function.
    assert inspect.isgeneratorfunction(MockServicer.RunStream)
    assert inspect.isgeneratorfunction(BiopbServicerBase.RunDetectionStream)


def test_helpers_are_not_wrapped():
    # _server_context / __init__ are not RPC handlers and must be untouched.
    assert not getattr(BiopbServicerBase._server_context, "__biopb_wrapped__", False)
    assert not getattr(BiopbServicerBase.__init__, "__biopb_wrapped__", False)


def test_no_boilerplate_translates_value_error():
    servicer = _RaisesValueError(use_lock=False)
    ctx = FakeContext()
    with pytest.raises(Exception):
        servicer.RunDetection(object(), ctx)
    assert ctx.code() == grpc.StatusCode.INVALID_ARGUMENT


def test_explicit_boilerplate_translates_once():
    # Outer (metaclass) + inner (explicit) wrap: reentrancy => single translation.
    servicer = _ExplicitBoilerplate(use_lock=False)
    ctx = FakeContext()
    with pytest.raises(Exception):
        servicer.RunDetection(object(), ctx)
    assert ctx.code() == grpc.StatusCode.INVALID_ARGUMENT


def test_direct_abort_status_is_preserved():
    # A direct context.abort() must keep INVALID_ARGUMENT, not become INTERNAL.
    servicer = _DirectAbort(use_lock=False)
    ctx = FakeContext()
    with pytest.raises(Exception):
        servicer.Run(object(), ctx)
    assert ctx.code() == grpc.StatusCode.INVALID_ARGUMENT


def test_streaming_handler_translates_errors():
    servicer = _StreamRaises(use_lock=False)
    ctx = FakeContext()
    with pytest.raises(Exception):
        list(servicer.RunStream(iter([]), ctx))
    assert ctx.code() == grpc.StatusCode.INVALID_ARGUMENT
