"""Who is writing this kernel: the one-agent claim, and everyone else's cells.

Runs **in the MCP server process**. Two questions, one subject -- the set of
clients writing to a single namespace and viewer:

* **May I write?** The kernel owns the one-agent claim (``_jobs.submit`` is the
  only thing that can enforce it atomically), but this process mirrors it, for
  the reason spelled out on :data:`_claimed_by`.
* **Who else did?** A person can run cells from the observe page, and the chat
  loop from its own turn, leaving an agent's picture of the namespace stale with
  nothing in its own results to say so. The foreign-activity digest is that
  notice, and its read/ack split is what makes it deferred-never-dropped.

Both are policy about co-writers, so they are one module: a caller asking either
question is asking about the same relation, and "foreign" is defined by
:data:`_local_origin`, which both halves read.
"""

import contextvars
import logging

from . import _app
from ._kernel_rpc import _run_job_call

logger = logging.getLogger(__name__)

# This process's mirror of the kernel's one-agent claim: the last client whose
# code the kernel actually accepted, or None while unclaimed.
#
# The kernel owns the claim -- ``_jobs.submit`` is the choke point and the only
# thing that can enforce it atomically. But ``restart_kernel`` cannot be gated
# from inside the kernel it destroys, and asking the kernel who owns it first is
# both a check-then-act race and *fail-open on a busy kernel*: a round trip that
# comes back "busy" would read as "no owner", and a kernel busy running the
# holder's job is exactly when a stray restart costs the most. Every claim
# passes through this process, so mirroring it here answers the question with no
# round trip and no window.
#
# Set from the kernel's own decision wherever a reply arrives: any submit the
# kernel did not refuse came from the holder, and a refusal names the holder
# outright, so assigning on both keeps the mirror true through a restart that
# happened somewhere else (the observe page's, which clears it explicitly).
#
# **Recorded before the submit is sent, not after.** A reply can be lost while
# the kernel goes on to claim and run the code anyway -- ``execute_interactive``
# hands the request over before it starts its clock, so a timed-out call is still
# queued and executes when the main thread frees up. Setting the mirror only on
# the way back would leave it empty while the kernel is genuinely held, and an
# empty mirror lets a stranger restart the session that just started. The window
# is claimed first and corrected from whatever the kernel says, so the failure
# direction is "held by the client that asked" rather than "held by nobody".
_claimed_by: str | None = None


def _note_claim(writer):
    """Record that the kernel is held by *writer* (ignores ``None``)."""
    global _claimed_by
    if writer is not None:
        _claimed_by = writer


def _presume_claim(writer):
    """Take the claim for *writer* only if this process has not seen one.

    Guarded on "not seen": a client the kernel is about to refuse must never
    overwrite a holder already known here, and it will be corrected by the
    refusal in any case.
    """
    if _claimed_by is None:
        _note_claim(writer)


def clear_claim():
    """Forget the mirrored claim, for a caller that just replaced the kernel."""
    global _claimed_by
    _claimed_by = None


def claim_holder():
    """Who holds this kernel, as far as this process has seen, or ``None``.

    A read for a caller deciding whether an action is worth offering at all --
    the chat pane's engine switch, which would otherwise hand the session to a
    second client that the kernel then refuses on its first cell. Mirrored, so
    it can be stale in the safe direction only: it is set from what the kernel
    actually said (:func:`_note_claim`), and cleared when the kernel is replaced.
    """
    return _claimed_by


# Refusal for a client that does not hold this kernel's one-agent claim
# (_jobs.submit). Shared by every state-changing tool so the agent gets one
# explanation rather than three, and so the recovery named is the same in all of
# them: the person at the machine, never a second agent.
_NOT_OWNER_MSG = (
    "This kernel is already in use by another client{held_by}, and only one "
    "agent runs code in a session. Two of you writing to the same namespace and "
    "viewer would order the writes without either being able to see what the "
    "other believes is there. Reading tools (poll_job, server_status, "
    "take_screenshot, inspect_object) still work, so you can watch. You cannot "
    "take the session over — restarting it is the user's to do, from the observe "
    "page. Tell them what you wanted and let them decide."
)


# Identity for a caller that reaches the tools without an MCP request at all --
# the in-process chat loop, which is a client of this server in every sense that
# matters but arrives as a plain function call. Set for the length of one
# dispatch (``_chat``), so every tool gates it the way it gates a remote client
# instead of letting it through as "no identity".
_local_identity: contextvars.ContextVar = contextvars.ContextVar(
    "biopb_local_identity", default=None
)

# The job origin this caller submits under, which is also the point of view the
# foreign-activity digest is read from: "someone else's cell" is a relation, not
# a property of the cell. Defaults to the remote clients this server was written
# for; the in-process chat loop sets it for the length of a dispatch, beside its
# identity above.
_local_origin: contextvars.ContextVar = contextvars.ContextVar(
    "biopb_local_origin", default="mcp"
)


def _client_identity():
    """``(id, label)`` for the client behind this call, or ``(None, "")``.

    The streamable-http transport mints a per-connection ``mcp-session-id``, so
    two clients reaching one session child are distinguishable even though the
    tool surface itself is stateless — this is the id the kernel's one-agent
    claim is keyed on (``_jobs.submit``). ``clientInfo.name`` from the initialize
    handshake rides along as a label, purely so a refusal can name who holds the
    kernel.

    Read through ``mcp.get_context()`` rather than a ``Context`` tool parameter:
    the parameter form is excluded from the advertised input schema, but it also
    makes the function uncallable without one, and every in-process caller (the
    tests today, an in-process chat loop later) has no request at all. Outside a
    request this yields no identity, which ``submit`` reads as "nothing to claim
    with" and lets through -- unless an in-process caller has announced itself
    through :data:`_local_identity`, which takes precedence over the request
    because it *is* the caller; a loop dispatching tools has no request of its
    own to be found.
    """
    local = _local_identity.get()
    if local is not None:
        return local
    try:
        rc = _app.mcp.get_context().request_context
    except Exception:  # noqa: BLE001 - no request, or an SDK shape we don't know
        return None, ""
    request = getattr(rc, "request", None)
    ident = request.headers.get("mcp-session-id") if request is not None else None
    session = getattr(rc, "session", None)
    if not ident and session is not None:
        # A client that negotiated no transport session still gets one identity
        # per connection: the ServerSession object is per-connection and lives
        # as long as it does, which is all the claim needs.
        ident = f"conn-{id(session):x}"
    params = getattr(session, "client_params", None)
    label = getattr(getattr(params, "clientInfo", None), "name", "") or ""
    return ident, label


def _foreign_digest(host) -> list:
    """The cells run by another writer that the agent has not been told about,
    or ``[]``.

    "Another writer" is relative to :data:`_local_origin`, so the chat loop is
    not handed its own cells.

    A pure read — see :func:`_ack_foreign_digest` for why the ack is a second call.
    Auxiliary, like the window-liveness probe: a kernel that answers with
    anything but the expected list yields no digest rather than breaking the
    result the agent actually asked for.
    """
    digest, _res, _w = _run_job_call(host, "foreign_digest", _local_origin.get())
    if not digest or not isinstance(digest, list):
        return []
    if not all(isinstance(d, dict) and "job_id" in d for d in digest):
        return []
    return digest


def _ack_foreign_digest(host, digest, writer=None) -> None:
    """Retire the *terminal* entries of *digest*, once the note carrying them is
    on its way back to the agent.

    Split from the read because acking inside it consumed notices that were
    never delivered: ``execute_interactive`` sends the request before it starts
    its timeout clock, so a probe that times out is still queued at the kernel
    and runs when the main thread frees up — setting the flag for a note nobody
    received. Acking only after this process has parsed a reply keeps the
    guarantee that a notice is deferred, never dropped.

    Running entries are excluded here rather than in the kernel: they were
    reported as ``running``, which is not the final status the agent is promised,
    so they must stay pending even if they have finished since.

    *writer* is the asking client, passed through so the kernel can refuse an ack
    from a client that does not hold it: a second client's ``poll_job`` may
    *read* the digest, but discharging a notice the holder has not received
    would defeat the exactly-once promise this split exists to keep.
    """
    ids = [d["job_id"] for d in digest if d.get("status") != "running"]
    if ids:
        _run_job_call(host, "ack_foreign_digest", ids, writer=writer)


def _render_foreign_note(digest) -> str:
    """The digest as a line appended to an agent-facing result, or ``""``.

    The agent is not the only writer of this namespace: a person can run code
    from the observe page, through the same job runner (``docs/user-console.md``).
    That leaves the agent's picture of the namespace stale with nothing in its
    own results to say so — hence this note, appended at the same seam as
    ``_window_note``, which is how every other user-attributed fact already
    reaches the agent (``cancel_reason``, ``teardown_reason``).

    Deliberately says *that* something changed, not *what*: the agent is told to
    re-verify, which is cheap and cannot go stale itself. It names **no** job id
    in the instruction either — pointing at one of several invites an agent to
    read that one, call the notice discharged, and never see the rest, which it
    will not be offered again.
    """
    if not digest:
        return ""
    # Older kernels' digest entries carry no origin; they could only ever have
    # been user cells, so read a missing one as "user" rather than dropping the
    # attribution.
    origins = {(d.get("origin") or "user") for d in digest}
    if origins == {"user"}:
        who = "The user"
        listed = ", ".join(f"{d['job_id']} ({d.get('status')})" for d in digest)
    else:
        who = "Another writer"
        listed = ", ".join(
            f"{d['job_id']} ({d.get('status')}, {d.get('origin') or 'user'})"
            for d in digest
        )
    return (
        f"\n\nⓘ {who} ran code in this kernel: "
        f"{listed}. A finished cell is reported once; a running one repeats "
        "until it ends, so a repeat is not a new cell. Read them with poll_job. "
        "Variables and layers may have changed — re-check with dir() / "
        "viewer.layers rather than trusting what you last saw."
    )


def _foreign_activity_note(host) -> str:
    """Read, render, and retire the activity notice, in that order.

    Retiring is the holder's alone (see :func:`_ack_foreign_digest`): a second
    client reaching a read-only tool still gets shown what ran, but does not
    consume the notice out from under the agent actually working here.
    """
    digest = _foreign_digest(host)
    note = _render_foreign_note(digest)
    if note:
        _ack_foreign_digest(host, digest, _client_identity()[0])
    return note
