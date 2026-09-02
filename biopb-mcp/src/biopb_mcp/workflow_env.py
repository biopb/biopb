"""The environment a saved biopb workflow rebuilds for itself.

One public call, because a workflow notebook has to build its handles somehow
and the alternative is what the exported notebook used to carry: a pasted block
importing ``biopb_mcp._config``, ``biopb_mcp._connection`` and
``biopb_mcp.mcp._process_ops`` -- three private modules -- into a document a
user is expected to keep and edit. The dependency on biopb-mcp is total either
way; this makes it supported instead of copied.

It is also what ``verify_workflow`` no longer does for the agent. A scratch
kernel that pre-built ``client`` and ``ops`` verified a document that would not
run for its reader, because the two setups were different code in different
places. Now there is one: the document's own first cell calls this, and the
verification runs it.

No viewer, and there will not be one. The viewer is how an agent shows
something to the person it is working with; a saved workflow is run by someone
already looking at their own screen.
"""

import logging
import sys

logger = logging.getLogger(__name__)


class _Namespace:
    """What the plugin loader binds into: an object with a ``user_ns``.

    The loader was written for an IPython shell and uses nothing else of it, so
    a notebook cell's own globals go in wearing the same shape -- which is what
    lets a workflow run under ``python wf.py`` as well as in a kernel.
    """

    def __init__(self, user_ns):
        self.user_ns = user_ns


class WorkflowEnvError(RuntimeError):
    """The data plane could not be reached, so there is nothing to work on."""


def workflow_env(*, plugins=True, require_client=True):
    """Build a workflow's handles; return ``(client, ops)``.

    *client* is a ``TensorFlightClient`` for this machine's data plane and *ops*
    the ProcessImage callables the config names (an empty dict when it names
    none).

    **It also binds the user's kernel plugins into the notebook's namespace**,
    which is a side effect and is named here because a function that writes to
    your globals should say so. It is what the session kernel does at step 7b,
    and doing it any other way would change how a workflow spells its calls: a
    plugin loaded from ``~/.config/biopb/kernel/rolling_ball.py`` binds the name
    ``rolling_ball``, so a document rewritten from a session keeps
    ``rolling_ball.subtract_background(...)`` rather than reaching through a
    returned container. Pass ``plugins=False`` to skip it.

    They are **the reader's plugins, not the author's** -- this machine's
    ``~/.config/biopb/kernel``, whatever it holds. A workflow calling a plugin
    this machine does not have binds nothing and fails at the call, so what
    loaded is printed: that line is the answer to the `NameError` that follows.

    Raises :class:`WorkflowEnvError` when no data plane can be reached and
    *require_client* is set. Failing here is the point: the alternative is a
    ``None`` client and a cell three steps later blaming the workflow for the
    environment.
    """
    from ._config import load_config
    from ._connection import TensorConnection
    from .mcp._process_ops import build_ops_from_config

    config = load_config()
    conn = TensorConnection()
    conn.auto_connect()
    if require_client and conn.client is None:
        raise WorkflowEnvError(
            "No data plane: "
            + (conn.last_message or "the tensor server could not be reached")
            + ". Start one with `biopb control start`, or set $BIOPB_TENSOR_URL."
        )
    ops = build_ops_from_config(config, lambda: conn.client)
    if plugins:
        _load_plugins(config, _caller_namespace())
    return conn.client, ops


def _caller_namespace():
    """Where a plugin should bind: the notebook's namespace.

    Under a kernel that is the shell's ``user_ns``, asked for by name rather
    than found by walking frames -- so it is still right when the call comes
    from a lab's own helper rather than straight from a cell. With no kernel
    (a workflow run as a plain script) there is no such registry, so it is the
    immediate caller's globals, and calling through a wrapper there would bind
    into the wrapper's module instead.
    """
    try:
        from IPython import get_ipython

        ip = get_ipython()
        if ip is not None:
            return ip.user_ns
    except Exception:  # noqa: BLE001 - no IPython installed, or none running
        pass
    # Two frames up: this function's caller is `workflow_env`, and its caller is
    # the workflow.
    return sys._getframe(2).f_globals


def _load_plugins(config, namespace):
    """Bind the user's kernel plugins into *namespace*, and say what bound.

    Fail-open, like the kernel's own step: a workflow that does not use a plugin
    must not fail because one is broken, and one that does will fail where it
    uses it.
    """
    try:
        from .mcp import _requires
        from .mcp._bootstrap import _load_namespace_plugins

        _load_namespace_plugins(_Namespace(namespace), config)
        bound = _requires._LOADED_FILES + _requires._LOADED_ENTRY_POINTS
        print("kernel plugins:", ", ".join(bound) if bound else "(none)")
    except Exception as exc:  # noqa: BLE001 - a plugin gap is not a failed workflow
        print("kernel plugins not loaded:", exc)
