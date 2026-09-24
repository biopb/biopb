"""The session kernel's gate on cells from other Jupyter clients.

Runs inside the kernel, as its ``kernel_class`` (``__main__`` passes
``--IPKernelApp.kernel_class``). A cell from any client but the host's own is
refused while a job runs on its worker thread, and otherwise runs inline,
announced as an ``origin="user"`` job so the host records it and the agent is
told of it (``docs/jupyter-clients.md``).

The gate is ``do_execute`` rather than a ``pre_run_cell`` callback because
IPython catches and prints what an event callback raises, so a callback cannot
refuse a cell.
"""

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

        with _jobs.record_inline(code, request=header.get("msg_id")) as job:
            reply = await run()
            if reply.get("status") == "ok":
                job.status = "ok"
            else:
                job.error_text = f"{reply.get('ename')}: {reply.get('evalue')}"
                # A client's own Ctrl-C lands here, on the main thread; it is a
                # stop, not a defect, as in _jobs._run.
                interrupted = reply.get("ename") == "KeyboardInterrupt"
                job.interrupted = interrupted
                job.status = "interrupted" if interrupted else "error"
        return reply

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
