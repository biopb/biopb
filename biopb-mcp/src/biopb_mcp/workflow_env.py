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

logger = logging.getLogger(__name__)


class WorkflowEnvError(RuntimeError):
    """The data plane could not be reached, so there is nothing to work on."""


def workflow_env(*, plugins=True, require_client=True):
    """Build a workflow's handles; return ``(client, ops)``.

    *client* is a ``TensorFlightClient`` for this machine's data plane and *ops*
    the ProcessImage callables the config names (an empty dict when it names
    none). With *plugins*, the user's kernel plugins are loaded into the calling
    IPython namespace exactly as the session kernel loads them -- so a workflow
    calling ``rolling_ball.subtract_background`` finds the same name it found
    when it was written. Outside IPython that step is skipped.

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
        _load_plugins(config)
    return conn.client, ops


def _load_plugins(config):
    """Bind the user's kernel plugins into the calling namespace, if any.

    Fail-open, like the kernel's own step: a workflow that does not use a plugin
    must not fail because one is broken, and one that does will fail where it
    uses it.
    """
    try:
        from IPython import get_ipython

        ip = get_ipython()
        if ip is None:
            return
        from .mcp._bootstrap import _load_namespace_plugins

        _load_namespace_plugins(ip, config)
    except Exception as exc:  # noqa: BLE001 - a plugin gap is not a failed workflow
        logger.warning("kernel plugins not loaded: %s", exc)
