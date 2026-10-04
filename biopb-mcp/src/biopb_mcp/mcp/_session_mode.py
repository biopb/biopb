"""What kind of session this process is, by how it ends.

A leaf module: the launcher decides once and the observe API, chat and the
registry record read the answer, so none of them re-derive it from launch flags.

* ``shim``: the child of a stdio shim that spawned it. Its shim reaps it, so it
  has no stop route and no chat.
* ``durable``: a session a person owns -- ``biopb mcp view``, or the dashboard's
  "new session". It ends when they stop it. Serves the stop route and chat.
* ``ephemeral``: a session the control launched for an agent. The control ends
  it once it has been free for a grace period, so it serves the stop route (the
  control asks it to end itself) and chat, and is recorded as ephemeral for the
  control to find.
* ``direct``: a ``--transport http`` server on a configured port that an agent
  connects to over http. It is nobody's to end from the web and publishes no
  session.
"""

SHIM = "shim"
DURABLE = "durable"
EPHEMERAL = "ephemeral"
DIRECT = "direct"

#: Set by the control when it launches a session for an agent. Kept in sync with
#: ``biopb_control._control._LIFETIME_ENV`` (a literal on each side: the two
#: packages cannot import each other, and a core constant would raise biopb-mcp's
#: ``biopb`` floor).
ENV_LIFETIME = "BIOPB_SESSION_LIFETIME"


def of(view, shim_owned, port, lifetime=None):
    """The mode for a launch: ``--view``, a shim's port-report file, the
    configured ``--port`` (0 for a dynamic one), the control's lifetime."""
    if shim_owned:
        return SHIM
    if view or port == 0:
        return EPHEMERAL if lifetime == EPHEMERAL else DURABLE
    return DIRECT


def owns_reap(mode):
    """Whether the session serves the stop route: something other than a shim
    ends it, by asking it to end itself."""
    return mode in (DURABLE, EPHEMERAL)


def serves_chat(mode):
    """Whether the built-in chat is mounted. The lease keeps it from sharing a
    kernel with an attached agent, so it no longer depends on there being none."""
    return owns_reap(mode)


def publishes(mode):
    """Whether the session binds a dynamic port and registers itself."""
    return mode != DIRECT
