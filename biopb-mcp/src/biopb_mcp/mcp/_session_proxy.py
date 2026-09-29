"""A kernel that relays to the running session kernel.

JupyterLab lists only kernels it started, so a kernelspec runs this as its
kernel (``python -m biopb_mcp.mcp._session_proxy -f {connection_file}``). Lab
sees an ordinary kernel on the ports of its own connection file; every message
is re-signed with the session's key and forwarded to the session kernel's
ports, and the replies come back the same way. A notebook thereby runs in the
session's namespace, as any client attached by connection file does.

The session is not Lab's to end: a shutdown request stops this process and is
not forwarded, so closing or restarting the notebook's kernel only detaches.

The session is the newest ``kernel-biopb-*.json`` in the runtime dir whose
shell port answers. With none, the proxy stays up and answers each request with
an error, so a notebook opened first works once the session starts. It keeps its
session until that one stops answering.
"""

import argparse
import asyncio
import glob
import json
import os
import socket
import sys
import traceback
from typing import Optional

import zmq
import zmq.asyncio
from jupyter_client.session import Session

_PATTERN = "kernel-biopb-*.json"
_NO_SESSION = "no running biopb session; start one from the biopb dashboard"
# Set to pin a connection file instead of taking the newest live one.
ENV_CONNECTION_FILE = "BIOPB_SESSION_CONNECTION_FILE"
# A reply that never comes (the session died mid-request) must not pin its
# routing entry forever.
_MAX_PENDING = 1024


def _alive(info: dict) -> bool:
    """Whether something accepts connections on the kernel's shell port."""
    try:
        with socket.create_connection((info["ip"], info["shell_port"]), timeout=1.0):
            return True
    except (OSError, KeyError):
        return False


def _read(path: str) -> Optional[dict]:
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def find_session(runtime_dir: Optional[str] = None) -> Optional[tuple[str, dict]]:
    """The connection file and info of the newest live session, else ``None``.

    A hard-killed session leaves its file behind, so a file alone is not a
    session: the shell port must accept a connection.
    """
    pinned = os.environ.get(ENV_CONNECTION_FILE)
    if pinned:
        info = _read(pinned)
        return (pinned, info) if info and _alive(info) else None
    if runtime_dir is None:
        from jupyter_core.paths import jupyter_runtime_dir

        runtime_dir = jupyter_runtime_dir()
    files = sorted(
        glob.glob(os.path.join(runtime_dir, _PATTERN)),
        key=os.path.getmtime,
        reverse=True,
    )
    for path in files:
        info = _read(path)
        if info and _alive(info):
            return path, info
    return None


def _session(info: dict) -> Session:
    return Session(
        key=info["key"].encode(),
        signature_scheme=info.get("signature_scheme", "hmac-sha256"),
        username="biopb-proxy",
    )


def _url(info: dict, port_name: str) -> str:
    return f"{info['transport']}://{info['ip']}:{info[port_name]}"


class Upstream:
    """Sockets to one session kernel, and the tasks reading its replies."""

    def __init__(self, path: str, info: dict, ctx: zmq.asyncio.Context):
        self.path = path
        self.info = info
        self.session = _session(info)
        self.socks = {}
        for name in ("shell", "control", "stdin"):
            s = ctx.socket(zmq.DEALER)
            # Same identity on all three: the kernel sends an input request on
            # stdin to the identity that sent the execute on shell.
            s.setsockopt(zmq.IDENTITY, self.session.bsession)
            s.connect(_url(info, f"{name}_port"))
            self.socks[name] = s
        s = ctx.socket(zmq.SUB)
        s.setsockopt(zmq.SUBSCRIBE, b"")
        s.connect(_url(info, "iopub_port"))
        self.socks["iopub"] = s

    def close(self):
        for s in self.socks.values():
            s.close(linger=0)


class SessionProxy:
    def __init__(self, connection_file: str):
        info = _read(connection_file)
        if info is None:
            raise SystemExit(f"cannot read connection file {connection_file}")
        self.ctx = zmq.asyncio.Context()
        self.lab = _session(info)
        self.lab_socks = {}
        for name, kind in (
            ("shell", zmq.ROUTER),
            ("control", zmq.ROUTER),
            ("stdin", zmq.ROUTER),
            ("iopub", zmq.PUB),
            ("hb", zmq.REP),
        ):
            s = self.ctx.socket(kind)
            s.bind(_url(info, f"{name}_port"))
            self.lab_socks[name] = s
        self.up: Optional[Upstream] = None
        self.up_tasks: list[asyncio.Task] = []
        # msg_id -> the Lab client's identity frames, to route a reply back.
        self.pending: dict[str, list[bytes]] = {}
        self.stop = asyncio.Event()

    # -- upstream ---------------------------------------------------------

    def _connect(self) -> bool:
        """Attach to the session, keeping the current one while it answers."""
        if self.up is not None and _alive(self.up.info):
            return True
        found = find_session()
        self._drop()
        if found is None:
            return False
        self.up = Upstream(*found, self.ctx)
        self.up_tasks = [
            asyncio.ensure_future(self._relay_up(name))
            for name in ("shell", "control", "stdin", "iopub")
        ]
        return True

    def _drop(self):
        for t in self.up_tasks:
            t.cancel()
        self.up_tasks = []
        if self.up is not None:
            self.up.close()
            self.up = None

    async def _relay_up(self, name: str):
        """Session -> Lab, one channel."""
        up = self.up
        src, dst = up.socks[name], self.lab_socks[name]
        while True:
            frames = await src.recv_multipart()
            try:
                idents, rest = up.session.feed_identities(frames)
                msg = up.session.deserialize(rest)
            except Exception:  # noqa: BLE001 - one bad frame must not end the relay
                continue
            if name == "iopub":
                await dst.send_multipart(
                    self.lab.serialize(msg, idents) + msg["buffers"]
                )
                continue
            parent = msg["parent_header"].get("msg_id")
            route = (
                self.pending.pop(parent, None)
                if name != "stdin"
                else self.pending.get(parent)
            )
            if route is None:
                continue
            await dst.send_multipart(self.lab.serialize(msg, route) + msg["buffers"])

    # -- Lab -> session ---------------------------------------------------

    async def _relay_down(self, name: str):
        sock = self.lab_socks[name]
        while True:
            frames = await sock.recv_multipart()
            try:
                idents, rest = self.lab.feed_identities(frames)
                msg = self.lab.deserialize(rest)
            except Exception:  # noqa: BLE001
                continue
            try:
                await self._handle(name, idents, msg)
            except Exception:  # noqa: BLE001 - one bad request must not end the relay
                traceback.print_exc()

    async def _handle(self, name: str, idents: list, msg: dict):
        kind = msg["header"]["msg_type"]
        if name == "control" and kind == "shutdown_request":
            content = {"status": "ok", "restart": msg["content"].get("restart", False)}
            await self._reply(name, idents, msg, "shutdown_reply", content)
            self.stop.set()
            return
        if not self._connect():
            await self._refuse(name, idents, msg, kind)
            return
        if len(self.pending) >= _MAX_PENDING:
            self.pending.pop(next(iter(self.pending)))
        if name != "stdin":
            self.pending[msg["header"]["msg_id"]] = idents
        await self.up.socks[name].send_multipart(
            self.up.session.serialize(msg) + msg["buffers"]
        )

    # -- answering locally ------------------------------------------------

    async def _reply(self, name, idents, parent, msg_type, content):
        reply = self.lab.msg(msg_type, content, parent=parent["header"])
        await self.lab_socks[name].send_multipart(self.lab.serialize(reply, idents))

    async def _iopub(self, parent, msg_type, content):
        m = self.lab.msg(msg_type, content, parent=parent["header"])
        await self.lab_socks["iopub"].send_multipart(
            self.lab.serialize(m, [msg_type.encode()])
        )

    async def _refuse(self, name, idents, msg, kind):
        """No session: say so, in the shape the client expects.

        Clients wait for the idle that follows a request's reply (Lab's
        readiness check does), so each answer is bracketed by busy and idle.
        """
        await self._iopub(msg, "status", {"execution_state": "busy"})
        if kind == "kernel_info_request":
            reply, content = (
                "kernel_info_reply",
                {
                    "status": "ok",
                    "protocol_version": "5.3",
                    "implementation": "biopb-session-proxy",
                    "implementation_version": "0",
                    "language_info": {"name": "python", "file_extension": ".py"},
                    "banner": _NO_SESSION,
                    "help_links": [],
                },
            )
        else:
            err = {
                "ename": "NoSession",
                "evalue": _NO_SESSION,
                "traceback": [_NO_SESSION],
            }
            if kind == "execute_request":
                await self._iopub(msg, "error", err)
            reply, content = (
                kind[: -len("request")] + "reply",
                {"status": "error", **err},
            )
        await self._reply(name, idents, msg, reply, content)
        await self._iopub(msg, "status", {"execution_state": "idle"})

    # -- run --------------------------------------------------------------

    async def run(self):
        hb = self.lab_socks["hb"]

        async def beat():
            while True:
                await hb.send(await hb.recv())

        tasks = [
            asyncio.ensure_future(beat()),
            *(
                asyncio.ensure_future(self._relay_down(n))
                for n in ("shell", "control", "stdin")
            ),
        ]
        self._connect()
        await self.stop.wait()
        await asyncio.sleep(0.2)  # let the shutdown_reply leave
        for t in tasks:
            t.cancel()
        self._drop()
        for s in self.lab_socks.values():
            s.close(linger=200)
        self.ctx.term()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("-f", "--connection-file", required=True)
    args = p.parse_args(argv)
    asyncio.run(SessionProxy(args.connection_file).run())


if __name__ == "__main__":
    main(sys.argv[1:])
