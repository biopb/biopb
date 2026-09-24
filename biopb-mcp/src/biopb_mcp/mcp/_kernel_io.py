"""The host's connection to its kernel: send from any thread, route every reply.

Runs **in the MCP server process**, owned by ``KernelHost``. One
``ThreadedKernelClient`` per kernel: its IO loop thread owns the zmq sockets,
so any caller thread may send, and every shell reply and iopub message arrives
on that thread, where it is routed to the request it answers by parent
``msg_id``.

That routing is what lets round trips overlap. ``execute_interactive`` reads
iopub itself and discards whatever is not its own request's, so two of them
could not share a client, and the host serialized every call behind one lock.
Here nothing is discarded that someone is waiting for, so each call only waits
for its own reply: the kernel still runs requests one at a time, in arrival
order.

Messages whose parent is no request of ours -- another Jupyter client's cells,
a late reply to a call that already timed out -- are dropped, except that every
``status`` message updates :attr:`KernelChannels.execution_state`, the kernel's
own busy/idle.
"""

import threading
import time

# How long a call waits for its request's iopub ``idle`` once the reply is in.
# The reply travels on the shell socket and the output on iopub, so the reply
# can overtake the last of the output; ``idle`` is published after both and
# marks the output complete. Bounded because iopub is a PUB socket and drops
# rather than blocks -- a call is not failed for a status message it never got.
_IDLE_GRACE = 5.0

# How long each readiness attempt waits for its kernel_info round trip.
_READY_ATTEMPT = 1.0


class KernelGone(Exception):
    """The connection was closed while the call waited: the kernel was torn
    down (shutdown, restart, respawn)."""


class _Call:
    """What one request has received so far."""

    __slots__ = ("msg_id", "stdout", "results", "errors", "reply", "replied", "idle")

    def __init__(self, msg_id):
        self.msg_id = msg_id
        self.stdout = []
        # text/plain of execute_result and display_data, in arrival order
        self.results = []
        # tracebacks of iopub error messages, one list of lines each
        self.errors = []
        self.reply = None
        self.replied = threading.Event()
        self.idle = threading.Event()


class KernelChannels:
    """Shell and iopub to one kernel, shared by every caller thread."""

    def __init__(self, km):
        from jupyter_client.threaded import ThreadedKernelClient

        # km.client() would build whatever km.client_class names; built here so
        # the class is this module's decision. Same session as the manager, so
        # the kernel's gate recognises these requests as the host's.
        self._kc = ThreadedKernelClient(
            parent=km,
            connection_file=km.connection_file,
            **km.get_connection_info(session=True),
        )
        self._calls = {}  # msg_id -> _Call
        self._calls_lock = threading.Lock()
        self._closed = False
        # The kernel's last published execution_state, from any client's
        # request: "busy" while its main thread runs something.
        self.execution_state = "starting"

    def start(self, timeout, is_alive):
        """Open the channels and return once the kernel answers on both.

        Readiness is a kernel_info round trip whose ``idle`` also arrived on
        iopub: a SUB socket drops what is published before its subscription
        reaches the kernel, and an execute sent before that would wait for an
        ``idle`` that was never delivered to it.
        """
        kc = self._kc
        # All of them, though only shell and iopub are used: stop_channels
        # touches every channel, creating any that were never started and then
        # closing their sockets under live streams.
        kc.start_channels()
        # Instance attributes shadow the no-op handlers; set before anything of
        # ours is sent, so no reply can arrive unrouted.
        kc.shell_channel.call_handlers = self._on_shell
        kc.iopub_channel.call_handlers = self._on_iopub
        deadline = time.monotonic() + timeout
        while True:
            call = self._send("kernel_info_request", {})
            try:
                if call.replied.wait(_READY_ATTEMPT) and call.idle.wait(_READY_ATTEMPT):
                    return
            finally:
                self._forget(call)
            if not is_alive():
                raise RuntimeError("Kernel died before replying to kernel_info")
            if time.monotonic() > deadline:
                raise RuntimeError(f"Kernel didn't respond in {timeout:g} seconds")

    def execute(self, code, timeout):
        """Run *code*; return its :class:`_Call` once its reply and output are in.

        Raises ``TimeoutError`` when no reply comes within *timeout*, and
        :class:`KernelGone` when the connection closes first. A request that
        timed out is still queued at the kernel and runs later; whatever it
        sends then has no call to go to and is dropped.
        """
        call = self._send(
            "execute_request",
            {
                "code": code,
                "silent": False,
                "store_history": False,
                "user_expressions": {},
                "allow_stdin": False,
                "stop_on_error": True,
            },
        )
        try:
            if not call.replied.wait(timeout):
                raise TimeoutError
            call.idle.wait(_IDLE_GRACE)
            if call.reply is None:
                raise KernelGone
            return call
        finally:
            self._forget(call)

    def close(self):
        """Stop the channels, failing every call still waiting."""
        with self._calls_lock:
            self._closed = True
            calls = list(self._calls.values())
            self._calls.clear()
        for call in calls:
            call.replied.set()
            call.idle.set()
        self._kc.stop_channels()

    # -- internals, IO loop side ---------------------------------------------

    def _send(self, msg_type, content):
        msg = self._kc.session.msg(msg_type, content)
        call = _Call(msg["header"]["msg_id"])
        with self._calls_lock:
            if self._closed:
                raise KernelGone
            # Registered before the send, so the reply cannot beat it here.
            self._calls[call.msg_id] = call
        self._kc.shell_channel.send(msg)
        return call

    def _forget(self, call):
        with self._calls_lock:
            self._calls.pop(call.msg_id, None)

    def _call_for(self, msg):
        msg_id = (msg.get("parent_header") or {}).get("msg_id")
        with self._calls_lock:
            return self._calls.get(msg_id)

    def _on_shell(self, msg):
        call = self._call_for(msg)
        if call is not None:
            call.reply = msg["content"]
            call.replied.set()

    def _on_iopub(self, msg):
        msg_type = msg["header"]["msg_type"]
        content = msg["content"]
        if msg_type == "status":
            # Global kernel state, tracked whether or not it's this call's --
            # checked before the call lookup below, which every other client's
            # traffic (a foreign cell, comm/widget chatter) would otherwise pay
            # for nothing on this thread.
            state = content.get("execution_state", "")
            self.execution_state = state
            if state != "idle":
                return
            call = self._call_for(msg)
            if call is not None:
                call.idle.set()
            return
        if msg_type not in ("stream", "execute_result", "display_data", "error"):
            return
        call = self._call_for(msg)
        if call is None:
            return
        if msg_type == "stream":
            call.stdout.append(content.get("text", ""))
        elif msg_type in ("execute_result", "display_data"):
            text = content.get("data", {}).get("text/plain", "")
            if text:
                call.results.append(text)
        elif msg_type == "error":
            call.errors.append(content.get("traceback", []))
