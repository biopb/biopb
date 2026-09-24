"""The session kernel's gate on cells from other Jupyter clients.

Runs inside the kernel, as its ``kernel_class`` (``KernelHost._launch`` passes
``--IPKernelApp.kernel_class``). A cell from any client but the host's own is
refused while a job runs on its worker thread, and otherwise runs inline,
announced as an ``origin="user"`` job so the host records it and the agent is
told of it (``docs/jupyter-clients.md``).

The gate is ``do_execute`` rather than a ``pre_run_cell`` callback because
IPython catches and prints what an event callback raises, so a callback cannot
refuse a cell.

The host's control requests land here too (:meth:`GatedKernel.biopb_request`):
stopping a job, a job's status, the graceful close before a kill. They go on
the control channel, which ipykernel serves on its own thread, so none of them
waits behind a cell on the main thread -- the cell being stopped included.
"""

import asyncio
import os

from ipykernel.ipkernel import IPythonKernel

from . import _jobs

# Env var carrying the host client's session id, set by KernelHost._launch
# (literal mirrored there as ENV_HOST_SESSION, kept in sync by this comment).
# Handed over at launch rather than learned from a request, so the gate is armed
# before the connection file lets anyone in, and no cell can re-adopt it.
#
# Not a security boundary: the id rides every iopub message's parent header,
# and a client holding the connection file can already run anything. The gate
# keeps honest clients from writing over each other; it does not stop a
# hostile one.
ENV_HOST_SESSION = "BIOPB_HOST_SESSION"

# ``None`` when the kernel was launched without one, which gates nothing: there
# is no host to tell apart from anyone else.
_host_session = os.environ.get(ENV_HOST_SESSION) or None

# How the refusal names whoever holds the kernel, by job origin (see _jobs._Job).
_HOLDER = {
    "mcp": "the agent",
    "chat": "the chat agent",
    "user": "a user's cell",
}


# The host's request on the control channel, and its reply. The literals are
# mirrored in _kernel_io (kept in sync by this comment).
CONTROL_REQUEST = "biopb_request"
CONTROL_REPLY = "biopb_reply"


def _close_session(ns):
    """Close the tensor client, then release dask: run before every kill.

    So the tensor server sees a clean Flight GOAWAY (and cancels any in-flight
    do_get) instead of discovering the dropped connection only via socket
    teardown after the kill -- the lag that lets a `biopb server stop` right
    after Ctrl-C block on its graceful drain. Dask goes through ``_dask_ctl``
    because a cluster this kernel spun is its to stop: shutting the workers
    down gracefully lets them clean their spill files (biopb/biopb#13).
    Best-effort: a failure here just leaves it to the kill.
    """
    try:
        conn = ns.get("_conn")
        if conn is not None and getattr(conn, "client", None) is not None:
            conn.client.close()
    except Exception:  # noqa: BLE001
        pass
    try:
        ctl = ns.get("_dask_ctl")
        if ctl is not None:
            ctl.shutdown()
    except Exception:  # noqa: BLE001
        pass


def _refusal_text(job):
    what = job["intent"] or job["code"] or "(no source)"
    holder = _HOLDER.get(job["origin"], job["origin"])
    return (
        f"Not run: {job['job_id']} is running for {holder} "
        f"({job['elapsed']:.0f}s so far): {what}\n"
        "Run this cell again once it finishes, or stop it with Stop on the "
        "session's observe page."
    )


class GatedKernel(IPythonKernel):
    """``IPythonKernel`` that refuses or records execute requests it did not
    get from the host (module docstring)."""

    async def do_execute(
        self,
        code,
        silent,
        store_history=True,
        user_expressions=None,
        allow_stdin=False,
        *,
        cell_meta=None,
        cell_id=None,
    ):
        async def run():
            return await super(GatedKernel, self).do_execute(
                code,
                silent,
                store_history,
                user_expressions,
                allow_stdin,
                cell_meta=cell_meta,
                cell_id=cell_id,
            )

        header = self.get_parent("shell").get("header", {})
        session = header.get("session")
        # An empty cell is a client asking for its prompt number or evaluating
        # user_expressions (qtconsole sends both, silently). Decided on the
        # code, not on `silent`: that flag only stops output being broadcast,
        # and silent code runs with full effect. Completion and inspection are
        # other message types and never reach here.
        if not code.strip() or _host_session is None or session == _host_session:
            return await run()

        running = _jobs.running_job()
        if running is not None:
            return self._refuse(running)

        reply = None
        try:
            with _jobs.record_inline(code, request=header.get("msg_id")) as job:
                reply = await run()
                if reply.get("status") == "ok":
                    job.status = "ok"
                else:
                    job.error_text = f"{reply.get('ename')}: {reply.get('evalue')}"
                    # A stop (Stop on the observe page, or the client's own
                    # Ctrl-C) is not a defect, as in _jobs._run.
                    interrupted = reply.get("ename") == "KeyboardInterrupt"
                    job.interrupted = interrupted
                    job.status = "interrupted" if interrupted else "error"
        except KeyboardInterrupt:
            # A stop that arrived after the cell's code had returned, while
            # ipykernel or this gate was wrapping it up. The client still gets
            # a reply: without one it would wait forever.
            if reply is None:
                reply = {
                    "status": "error",
                    "execution_count": self.execution_count,
                    "ename": "KeyboardInterrupt",
                    "evalue": "",
                    "traceback": [],
                    "user_expressions": {},
                    "payload": [],
                }
        return reply

    # -- the host's control requests -----------------------------------------

    control_msg_types = [*IPythonKernel.control_msg_types, CONTROL_REQUEST]

    async def biopb_request(self, stream, ident, parent):
        """Run one of the host's control ops and reply with its result.

        ``content`` is ``{"op": <name>, "args": {...}, "timeout": <s>}``. The op
        runs on a worker thread, bounded by *timeout*: control requests are
        handled one at a time, and ipykernel's own interrupt and shutdown
        requests share that queue. Refused for any session but the host's, as
        the execute gate is.
        """
        content = parent.get("content", {})
        session = parent.get("header", {}).get("session")
        if _host_session is not None and session != _host_session:
            reply = {"status": "error", "evalue": "not the host's session"}
        else:
            ops = {
                "interrupt": _jobs.interrupt,
                "status": _jobs.status,
                "poll": _jobs.poll,
                "close": lambda: _close_session(self.shell.user_ns),
            }
            op = ops.get(content.get("op"))
            if op is None:
                reply = {"status": "error", "evalue": f"no op {content.get('op')!r}"}
            else:
                try:
                    r = await asyncio.wait_for(
                        asyncio.to_thread(op, **content.get("args", {})),
                        content.get("timeout", 10.0),
                    )
                    reply = {"status": "ok", "r": r}
                except Exception as exc:  # noqa: BLE001 - the host reads why
                    reply = {"status": "error", "evalue": repr(exc)}
        self.session.send(stream, CONTROL_REPLY, reply, parent, ident=ident)

    def _refuse(self, running):
        """An error reply, and the same error on iopub: a reply alone renders
        nothing in a notebook cell."""
        text = _refusal_text(running)
        content = {
            "ename": "KernelBusy",
            "evalue": text,
            "traceback": ["KernelBusy: " + text],
        }
        self.send_response(
            self.iopub_socket, "error", content, ident=self._topic("error")
        )
        return {
            "status": "error",
            "execution_count": self.execution_count,
            "user_expressions": {},
            "payload": [],
            **content,
        }
