"""The initialize-handshake ``instructions``: base guidance, doc index, the rest.

Imports nothing heavy, because two processes compose them: the session's
FastMCP server, per MCP session, and the stdio shim, which answers ``initialize``
itself before any session child exists.
"""

import logging

logger = logging.getLogger(__name__)

# Handed to the client in the initialize handshake (the only handshake-time
# carrier MCP defines). Clients that honor it inject it into the model's
# context from the first turn (compliance is up to the client/agent), so this
# field carries the guidance that must hold on *every* turn — the operation
# guardrails.
BASE_INSTRUCTIONS = (
    "First action of every session: call `start_kernel`. It brings up the kernel "
    "-- the tensor client, `ops` and the kernel plugins, and a napari window where "
    "the session has one -- and blocks until they are ready; nothing "
    "auto-starts, and every other kernel tool fails until it returns. It also "
    "rebuilds a kernel that never started, died, or was torn down by the user "
    "closing the viewer window -- but a kernel that is already up it leaves "
    "alone, so it cannot clear a wedged one: that is `interrupt_kernel` (stop "
    "the job) or `restart_kernel` (hard-restart). A user asking to start, open, "
    "or launch biopb or napari is asking for this tool.\n"
    "\n"
    "This biopb-mcp session drives a child IPython kernel over bioimage data; "
    "`execute_code` runs arbitrary Python in it. Its namespace always holds the "
    "data plane -- `client`, the tensor server -- and the algorithm plane -- "
    "`ops`, the server-side image-processing operations, and the user's kernel "
    "plugin modules. A napari `viewer` is there only when the user has configured "
    "one and this machine can show it: `server_status`'s `## Viewer` says whether "
    "this session has one, and nothing should assume it. The browser page the "
    "control serves (the web viewer) shows the user an image either way. "
    "The index below "
    "lists the docs; read one with `read_doc(id)`, and read the reference docs "
    "it lists before non-trivial work.\n"
    "\n"
    "Operation guardrails (apply on every turn):\n"
    "- Use data from `client`, or from `viewer`'s layers where the session has a "
    "window; avoid the filesystem unless the user explicitly asks.\n"
    '- Browse the catalog with `client.query_sources(sql, format="pandas")` '
    "(server-side DuckDB, the only browse surface); the `sources` columns are source_id, "
    "source_url, source_type, indexed_at, metadata_json, is_resolved, and "
    "`tensors` (a LIST of STRUCT(array_id, dim_labels, shape, dtype), one per "
    "tensor -- query it with UNNEST(tensors) or list_filter, or reach the "
    "source's first tensor as `tensors[1].dtype` -- DuckDB lists are "
    "1-indexed, unlike `tensors[0]` in Python/TS code). Unresolved (cloud / "
    "synced-folder) sources carry an empty `tensors`, so a predicate like "
    "`WHERE tensors[1].dtype = 'uint8'` silently drops them; filter on "
    "`is_resolved` to opt them in/out on "
    "purpose (`WHERE NOT is_resolved` finds what hasn't been resolved yet). "
    "Resolved is not the same as local: assume a cloud or synced-folder source "
    "may not have its bytes on the serving machine, so the first read can be "
    "slow or fail offline. Plan for that -- warn the user before a long read "
    "rather than after it.\n"
    "- Prefer lazy dask operations; only `.compute()` the final result.\n"
    "- Show intermediate results for the user to validate at each step: a layer "
    "on `viewer` where the session has a window, otherwise upload the result and "
    'give them a web-viewer link (`read_doc("web-viewer")`).\n'
    "- Do not assume — ask the user to clarify uncertainties; they know the "
    "data better than you do."
)

# Appended after the index: how to follow a procedure doc, and when to write one.
AUTHORING_INSTRUCTIONS = (
    "A procedure doc opens with a Requirements line; resolve it against "
    "`server_status` before starting, and treat a gap as something to name and "
    'work around rather than a reason to stop (`read_doc("requirements")`). '
    "Never substitute silently -- the user cannot judge a result whose method "
    "they were not told changed.\n"
    "\n"
    "After accomplishing a task worth repeating, ask the user whether it should "
    "become a doc, and write it with `write_doc`.\n"
    "\n"
    "Destructive steps always ask first, whatever a doc says: restarting the "
    "kernel, interrupting a running job, overwriting a layer, or writing files."
)

# The header the rendered index is carried under. The index is inlined here
# rather than fetched on request because every prompted hop loses agents
# (biopb/biopb#894 is the record of agents missing the *first* hop).
INDEX_HEADER = (
    "The doc index follows. It is itself doc `index`: re-read it with "
    '`read_doc("index")` and edit it with `write_doc`. A bullet that opens '
    "with `id:` is a doc entry; write notes as plain prose."
)


def compose() -> str:
    """The handshake text: the base guidance, the rendered index, the rest.

    Composed from :data:`BASE_INSTRUCTIONS` every time rather than appended to,
    so switching a dimension back off cannot leave a stale fragment behind.
    """
    parts = [BASE_INSTRUCTIONS]
    try:
        from ._docs import render_index

        parts.append(f"{INDEX_HEADER}\n\n{render_index()}")
    except Exception:  # pragma: no cover - the store is fail-open everywhere else
        logger.debug(
            "docs: could not render the index for the handshake", exc_info=True
        )
    parts.append(AUTHORING_INSTRUCTIONS)
    return "\n\n".join(parts)
