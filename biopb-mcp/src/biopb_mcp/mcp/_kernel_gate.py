"""The session kernel's gate on cells from other Jupyter clients.

Runs inside the kernel, as its ``kernel_class`` (``__main__`` passes
``--IPKernelApp.kernel_class``). A cell from any client but the host's own is
refused while a job runs on its worker thread, and otherwise runs inline,
recorded as an ``origin="user"`` job so the agent is told of it
(``docs/jupyter-clients.md``).

The gate is ``do_execute`` rather than a ``pre_run_cell`` callback because
IPython catches and prints what an event callback raises, so a callback cannot
refuse a cell.
"""

from ipykernel.ipkernel import IPythonKernel

from . import _jobs

# The host's client session id, learned from the request that runs
# :func:`adopt_host_session`. ``None`` until then, which gates nothing: the
# kernel cannot tell the host from anyone else before it has been told.
_host_session = None

# How the refusal names whoever holds the kernel, by job origin (see _jobs._Job).
_HOLDER = {
    "mcp": "the agent",
    "chat": "the chat agent",
    "user": "a cell from the observe page",
}


def adopt_host_session():
    """Record the session this request came from as the host's own.

    The host runs this before it marks the kernel ready, so every later request
    from that session is its own and anything else is foreign. Adopting from a
    request rather than passing an id means the host has nothing to generate or
    hand over, and a kernel restart re-adopts with the new client.
    """
    global _host_session
    from IPython import get_ipython

    _host_session = get_ipython().kernel.get_parent("shell")["header"]["session"]


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

        session = self.get_parent("shell").get("header", {}).get("session")
        # `silent` is a client's hidden execution -- completion, inspection --
        # which changes nothing a record would describe.
        if silent or _host_session is None or session == _host_session:
            return await run()

        running = _jobs.running_job()
        if running is not None:
            return self._refuse(running)

        with _jobs.record_inline(code) as job:
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
        content = {
            "ename": "KernelBusy",
            "evalue": _refusal_text(running),
            "traceback": ["KernelBusy: " + _refusal_text(running)],
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
