"""The environment a saved biopb workflow rebuilds for itself.

One public call, because a workflow notebook has to build its handles somehow
and the alternative is a pasted block importing ``biopb_mcp._config`` and
``biopb_mcp.mcp._process_ops`` -- private modules -- into a document a user is
expected to keep and edit. The dependency on biopb-mcp is total either
way; this makes it supported instead of copied.

It is also what ``verify_workflow`` no longer does for the agent. A scratch
kernel that pre-built ``client`` and ``ops`` verified a document that would not
run for its reader, because the two setups were different code in different
places. Now there is one: the document's own first cell calls this, and the
verification runs it.

No viewer, and there will not be one. The viewer is how an agent shows
something to the person it is working with; a saved workflow is run by someone
already looking at their own screen.

What comes back is the *connection* (``biopb.tensor.Connection``), not a
client: a reconnect swaps the client out, so a document derives it per use
(``client = conn.client``) the way the session kernel does before each cell. The connection is also what ``TensorBrowserWidget(viewer,
connection=conn)`` takes.
"""


class WorkflowEnvError(RuntimeError):
    """The data plane could not be reached, so there is nothing to work on."""


def workflow_env(*, require_client=True):
    """Build a workflow's handles; return ``(conn, ops)``.

    *conn* is a connected ``biopb.tensor.Connection`` for this machine's data
    plane and *ops* the algorithm plane's ops, as the session's kernel binds
    them (empty when this machine's control names none). A document that wants
    the session's spelling for the client takes it on the next line::

        conn, ops = workflow_env()
        client = conn.client

    Raises :class:`WorkflowEnvError` when no data plane can be reached and
    *require_client* is set. Failing here is the point: the alternative is a
    ``None`` client and a cell three steps later blaming the workflow for the
    environment. With *require_client* off, *conn* still comes back unconnected,
    which a caller can retry.
    """
    from biopb.tensor import Connection

    from ._config import load_config
    from .mcp._process_ops import build_ops_from_config

    config = load_config()
    conn = Connection()
    conn.connect()
    if require_client and conn.client is None:
        raise WorkflowEnvError(
            "No data plane: "
            + (conn.last_message or "the tensor server could not be reached")
            + ". Start one with `biopb control start`, or set $BIOPB_TENSOR_URL."
        )
    return conn, build_ops_from_config(config, lambda: conn.client)
