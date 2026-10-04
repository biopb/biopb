"""What kind of session this process is, by how it was started.

A leaf module: the launcher decides once and chat and the registry record read
the answer, so neither re-derives it from launch flags.

* ``durable``: a session on a dynamic port -- ``biopb mcp view``, or one the
  control launched for the dashboard or for an agent. It publishes itself, runs
  until a person stops it, and serves chat.
* ``direct``: a ``--transport http`` server on a configured port that an agent
  connects to over http. It publishes no session and serves no chat.

Every session serves the stop route, so every row on the dashboard can be
stopped: a shim whose session is stopped sees it stop answering and unbinds.
"""

DURABLE = "durable"
DIRECT = "direct"


def of(view, port):
    """The mode for a launch: ``--view``, and the configured ``--port`` (0 for a
    dynamic one)."""
    return DURABLE if view or port == 0 else DIRECT


def serves_chat(mode):
    """Whether the built-in chat is mounted. The lease keeps it from sharing a
    kernel with an attached agent, so a session an agent may attach to can
    serve it."""
    return mode == DURABLE


def publishes(mode):
    """Whether the session binds a dynamic port and registers itself."""
    return mode == DURABLE
