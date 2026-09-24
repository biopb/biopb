"""Who is writing this kernel: the one-agent claim, and everyone else's cells.

Runs **in the MCP server process**. Two questions, one subject -- the set of
clients writing to a single namespace and viewer:

* **May I write?** This process owns the one-agent claim
  (:func:`take_claim`): every agent's cell enters the kernel through it.
* **Who else did?** A person can run cells from an attached Jupyter client, and
  the chat loop from its own turn, leaving an agent's picture of the namespace
  stale with nothing in its own results to say so. The foreign-activity digest
  is that notice, read from the host's job records (``_job_log``), and its
  read/ack split is what makes it deferred-never-dropped.

Both are policy about co-writers, so they are one module: a caller asking either
question is asking about the same relation, and "foreign" is defined by
:data:`_local_origin`, which both halves read.
"""

import contextvars
import logging
import threading

from . import _app

logger = logging.getLogger(__name__)

# The one-agent claim: the client whose code this kernel runs, or None while
# unclaimed. Decided here (:func:`take_claim`), where every agent's cell enters
# the kernel, so the check and the claim are one step under one lock. It lasts
# one kernel: a restart clears it, and a respawn -- a new generation of the
# host's kernel -- voids it.
_claimed_by: str | None = None
_claimed_gen = None
_claim_label = ""

# Guards :data:`_claimed_by`, and -- the reason it is a lock rather than nothing
# -- is held across a whole restart by :func:`restart`.
#
# The claim is read and written from several threads: the tools await their
# kernel round trips off the event loop, so a cell claiming the kernel and a
# restart clearing it genuinely overlap. The window that matters is the restart:
# gate, replace the kernel, clear the claim is one decision, and a cell that
# lands in the middle of it would have its claim on the *new* kernel wiped by
# the clear that follows -- an empty claim over a held kernel lets a stranger
# restart the session that just started.
#
# Reentrant so a caller holding it can take a claim. Lock order is
# _claim_lock -> KernelHost._lock, and nothing takes them the other way round.
_claim_lock = threading.RLock()


def take_claim(host, writer, label=""):
    """Claim *host*'s kernel for *writer*, or refuse: returns None when
    *writer* holds it (now or already), else the holder's label.

    A caller with ``writer=None`` -- a transport that yields no client id --
    neither claims nor is checked, since there is nothing to tell two of them
    apart with.
    """
    global _claimed_by, _claimed_gen, _claim_label
    with _claim_lock:
        if _claimed_gen is not None and _claimed_gen != host.generation:
            _claimed_by = None  # the kernel it was made on is gone
        if writer is None or _claimed_by == writer:
            return None
        if _claimed_by is None:
            _claimed_by, _claimed_gen, _claim_label = writer, host.generation, label
            return None
        return _claim_label


def _note_claim(writer, label=""):
    """Record that the kernel is held by *writer*, on whichever kernel runs
    now (ignores ``None``)."""
    global _claimed_by, _claimed_gen, _claim_label
    if writer is not None:
        with _claim_lock:
            _claimed_by, _claimed_gen, _claim_label = writer, None, label


def clear_claim():
    """Forget the claim, for a caller that just replaced the kernel."""
    global _claimed_by, _claimed_gen, _claim_label
    with _claim_lock:
        _claimed_by, _claimed_gen, _claim_label = None, None, ""


def claim_holder():
    """Who holds this kernel, as far as this process has seen, or ``None``.

    A read for a caller deciding whether an action is worth offering at all --
    the chat pane's engine switch, which would otherwise hand the session to a
    second client that is then refused on its first cell. A claim on a kernel
    since respawned still reads as held until the next :func:`take_claim`,
    which is the safe direction.
    """
    with _claim_lock:
        return _claimed_by


def stop_refusal(holder, job_origin, writer, origin):
    """Why the client *writer*, of *origin*, may not stop a job of
    *job_origin* on a kernel held by *holder*, as a refusal dict; None when
    it may.

    One rule for the session's jobs and a verification's: a client that does
    not hold the kernel is refused (``not_owner``), and so is a job another
    writer started (``foreign_job``) -- the stop would be silent to them. The
    person at the machine (``origin="user"``) may stop anything, and a caller
    with no identity is not checked for the claim.
    """
    if origin == "user":
        return None
    if writer is not None and holder not in (None, writer):
        return {"refused": "not_owner"}
    if job_origin != origin:
        return {"refused": "foreign_job", "origin": job_origin}
    return None


def restart(host, writer, also_discard=None):
    """Gated restart: replace the kernel and reset the claim, atomically.

    Returns ``(refusal, discarded)``. *refusal* is ``None`` once the kernel is
    back, or the message to hand the caller instead; *discarded* is whatever
    *also_discard* returned, or ``None``. ``host.restart`` failing raises, and
    leaves the claim alone -- the old kernel may still be there.

    **Blocking, and meant to be called off the event loop.** The steps are one
    critical section rather than several statements at a call site: reading the
    holder, replacing the kernel and clearing the claim have to be indivisible,
    or a cell landing between the restart and the clear loses the claim it just
    made on the new kernel (see :data:`_claim_lock`). Holding the lock for the
    seconds a restart takes is the point -- a claim on a kernel that is being
    destroyed is not a claim.

    *also_discard* is the other thing that dies with this kernel: an in-flight
    verification, which runs in a second kernel but holds this session's one job
    slot (``_scratch.discard``). **It runs only once the gate has passed**, which
    is the whole reason it is a parameter here rather than a line before the
    call. Refusing a stranger's restart *after* their call has already destroyed
    the holder's verification is not a refusal -- it hands a client that cannot
    take the session the power to wreck it, which is precisely what the gate
    exists to prevent.

    It runs *before* ``host.restart()`` rather than after, because until the slot
    is released the freshly restarted kernel can accept nothing. A restart that
    then fails has discarded the verification for nothing; the session is in a
    bad state either way, and leaving the slot held would be the worse of the
    two.

    Reads the claim here rather than asking the kernel: a round trip would be
    a check-then-act race *and* fail-open on a busy kernel.
    """
    with _claim_lock:
        held = _claimed_by
        if held is not None and writer is not None and writer != held:
            return _NOT_OWNER_MSG.format(held_by=""), None
        discarded = also_discard() if also_discard is not None else None
        host.restart()
        clear_claim()  # a fresh kernel is unclaimed until someone runs code in it
        return None, discarded


def restart_for_user(host, also_discard=None):
    """Ungated restart, for the person at the machine.

    Never gated on the one-agent claim, which makes it the recovery path for a
    session held by a client that is gone: an agent cannot take a kernel from
    another agent, but the user can always replace it. Same critical section as
    :func:`restart`, and blocking for the same reason. There is no gate for
    *also_discard* to be ordered against here, but it stays inside the lock so a
    claim cannot land between the discard and the clear.
    """
    with _claim_lock:
        discarded = also_discard() if also_discard is not None else None
        host.restart()
        clear_claim()
        return discarded


# Refusal for a client that does not hold this kernel's one-agent claim
# (take_claim). Shared by every state-changing tool so the agent gets one
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
    claim is keyed on (``take_claim``). ``clientInfo.name`` from the initialize
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
    not handed its own cells. A pure read -- see :func:`_ack_foreign_digest`
    for why the ack is a second call.
    """
    return host.jobs.foreign_digest(_local_origin.get())


def _ack_foreign_digest(host, digest, writer=None) -> None:
    """Retire the *terminal* entries of *digest*, once the note carrying them is
    on its way back to the agent.

    Split from the read so a notice is retired only by the call that delivers
    it: a result that never reaches the agent must leave it pending.

    Running entries are excluded: they were reported as ``running``, which is
    not the final status the agent is promised, so they must stay pending even
    if they have finished since.

    **Only the kernel's holder can discharge a notice.** Reading the digest is
    open to anyone -- a second client watching the session is welcome to see
    that a cell ran -- but a bystander's ``poll_job`` acking it would retire a
    notice the holder never received. Decided on the claim; a caller
    with no identity is the in-process case and acks.
    """
    with _claim_lock:
        if writer is not None and _claimed_by not in (None, writer):
            return
    ids = [d["job_id"] for d in digest if d.get("status") != "running"]
    if ids:
        host.jobs.ack_foreign_digest(ids)


def _render_foreign_note(digest) -> str:
    """The digest as a line appended to an agent-facing result, or ``""``.

    The agent is not the only writer of this namespace: a person can run code
    from a Jupyter client attached to the kernel, recorded as a job the same
    way (``docs/jupyter-clients.md``).
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
