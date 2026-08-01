"""A real biopb-mcp session, brought up and driven from synchronous test code.

`docs/skill-testing.md` §6 runs at **Tier 2** of §8: a real shim-spawned session
child, a real IPython kernel, a real napari viewer, real dask — and the nine
real tools reached over real MCP. Nothing here stands in for the runtime. That
is the whole point: a hand-written tool surface would put `execute_code`'s
return shape, `server_status`'s report and the `guide://` bodies back into a
transcription, which is precisely the property that disqualified §5 from
gating (§5c).

What this module owns is bring-up, a synchronous façade over the async MCP
client, and three environment facts that have to be *forced* rather than
inherited, because each of them silently changes what a run is testing:

**A display, and a GL context behind it.** `transport.display_mode` defaults to
`auto`, which degrades to a viewer-less kernel when `$DISPLAY` is unset — a
legitimate production mode, so nothing fails. But a §6 run that took it would
be scoring a session in which step 2's "show the user the first and last
frames" *cannot happen*. Worse, `QT_QPA_PLATFORM=offscreen` is not enough on
its own: napari builds, and then `add_image` dies inside vispy's extension
probe, because offscreen Qt has no GL context. So a GL-capable display is a
hard requirement here — a workstation's own, or `xvfb-run -a`. Absent one,
these tests **skip with instructions** rather than run somewhere else.

**No tensor plane.** A developer box often has a data plane up, and then
`client` is live and the agent can wander into whatever that machine's catalog
happens to hold — so a finding might not reproduce anywhere else. The child is
pointed at an unreachable URL instead: `auto_connect` fails best-effort,
`client` lands as ``None``, and the fixture reaches the agent as a napari layer
and nothing else. Every skill's Parameters table already accepts "a layer on
`viewer`" as a source, and a session with no tensor plane is a real
configuration a user can be in, so step 1 still has something true to resolve.

**A config tree of our own.** `XDG_CONFIG_HOME` points at a temp dir, so the
run neither reads the developer's `mcp-config.json` nor their personal
`~/.config/biopb/skills/*.md`. The catalog under test is the shipped one.

Arrays cross the boundary as ``.npy`` files in a shared temp dir, not as base64
inside a tool call: the session child is on this machine, a fixture movie is
several megabytes, and tool output is truncated for the agent's benefit.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import tempfile
import threading
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

#: How long bring-up may take. The kernel imports napari and spins dask, which
#: is seconds on a warm machine and much worse on a cold one.
SPAWN_TIMEOUT = 120.0
KERNEL_TIMEOUT = 300.0
CALL_TIMEOUT = 300.0

#: Printed by kernel-side snippets so the façade can find a path in output that
#: also carries whatever the agent's code decided to print.
SENTINEL = "__BIOPB_SKILL_HARNESS__"

#: An address nothing listens on, to keep `client` at None. Port 1 is
#: privileged and unbound; the connect fails fast rather than hanging.
UNREACHABLE_TENSOR_URL = "grpc://127.0.0.1:1"


class SessionUnavailable(Exception):
    """Bring-up cannot even be attempted here — carries what is missing."""


def why_unavailable() -> str:
    """``""`` when a session can run here, else the reason to skip with.

    Checked before spawning because the failure is slow and unhelpful
    otherwise: the child comes up, the kernel starts, and the first
    `add_image` dies deep inside vispy with an ``AttributeError`` on ``None``.
    """
    if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        return (
            "no display: §6 needs a GL-capable one for napari layers. Run under "
            "`xvfb-run -a -s '-screen 0 1024x768x24'`, or on a desktop session. "
            "QT_QPA_PLATFORM=offscreen alone is NOT enough — vispy needs a GL "
            "context that the offscreen platform does not provide."
        )
    try:
        import mcp  # noqa: F401
    except ImportError:  # pragma: no cover - the mcp extra is a test dep
        return "the `mcp` client SDK is not installed"
    return ""


@dataclass
class ToolResult:
    """One tool call's outcome, flattened to the text an agent would see."""

    name: str
    text: str
    is_error: bool = False

    def __str__(self) -> str:
        return f"{self.name} -> {'ERROR ' if self.is_error else ''}{self.text}"


@dataclass
class ToolSpec:
    """A tool as the server advertises it — the schema an agent is handed."""

    name: str
    description: str
    input_schema: dict


class _LoopThread:
    """An event loop on its own thread, so sync test code can drive an async
    MCP session that must stay open across many calls."""

    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run, name="biopb-skill-mcp", daemon=True
        )
        self._thread.start()

    def _run(self) -> None:
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def submit(self, coro, timeout: float):
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result(timeout)

    def close(self) -> None:
        self.loop.call_soon_threadsafe(self.loop.stop)
        self._thread.join(timeout=10)
        self.loop.close()


@dataclass
class LiveSession:
    """The façade the harness and its tests use. Built by :func:`live_session`."""

    url: str
    session_id: str
    instructions: str
    tools: list[ToolSpec]
    scratch: Path
    _loop: _LoopThread
    _session: Any
    _turn: int = 0
    calls: list[tuple[int, str, dict]] = field(default_factory=list)

    # --- the agent-visible surface -----------------------------------------

    def call(self, name: str, /, **arguments: Any) -> ToolResult:
        """Call a tool exactly as an agent would, and record that it happened.

        The record is what the gate-spy assertions read: which tool, with what,
        and at which conversational turn.
        """
        self.calls.append((self._turn, name, dict(arguments)))
        result = self._loop.submit(
            self._session.call_tool(name, arguments), CALL_TIMEOUT
        )
        text = "\n".join(
            block.text for block in result.content if getattr(block, "text", None)
        )
        return ToolResult(name=name, text=text, is_error=bool(result.isError))

    def read_resource(self, uri: str) -> str:
        from pydantic import AnyUrl

        result = self._loop.submit(
            self._session.read_resource(AnyUrl(uri)), CALL_TIMEOUT
        )
        return "\n".join(c.text for c in result.contents if getattr(c, "text", None))

    # --- harness-side plumbing ---------------------------------------------
    #
    # These are the harness talking to the kernel around the agent, never
    # something the agent does. Kept separate from `call` on purpose: what the
    # agent did is a trace, and setup must not appear in it.

    def setup(self, code: str) -> ToolResult:
        """Run kernel code as the harness, without recording a turn."""
        self.calls.append((-1, "execute_code[setup]", {}))
        result = self._loop.submit(
            self._session.call_tool("execute_code", {"python_code": code}),
            CALL_TIMEOUT,
        )
        text = "\n".join(b.text for b in result.content if getattr(b, "text", None))
        out = ToolResult("execute_code", text, bool(result.isError))
        if out.is_error or "Traceback" in out.text:
            raise SessionUnavailable(f"harness setup failed:\n{out.text}")
        return out

    def put_array(self, name: str, array: np.ndarray) -> None:
        """Bind *array* in the kernel namespace, via a file rather than a
        literal — a fixture movie is megabytes and tool arguments are not."""
        path = self.scratch / f"in-{uuid.uuid4().hex}.npy"
        np.save(path, array)
        self.setup(f"import numpy as _np\n{name} = _np.load({str(path)!r})\ndel _np")

    def get_array(self, expression: str) -> np.ndarray | None:
        """Read an array back out of the kernel, or ``None`` if the expression
        does not evaluate to one. A run that left nothing behind is a normal
        outcome here, not an error."""
        path = self.scratch / f"out-{uuid.uuid4().hex}.npy"
        out = self.setup(
            "import numpy as _np\n"
            "try:\n"
            f"    _v = _np.asarray({expression})\n"
            f"    _np.save({str(path)!r}, _v)\n"
            f"    print({SENTINEL!r}, 'ok', _v.shape)\n"
            "except Exception as _e:\n"
            f"    print({SENTINEL!r}, 'no', type(_e).__name__)\n"
            "finally:\n"
            "    _np = None\n"
        )
        if f"{SENTINEL} ok" not in out.text or not path.is_file():
            return None
        return np.load(path)

    def has_real_viewer(self) -> tuple[bool, str]:
        """Whether `viewer` is a napari viewer and not the headless sentinel."""
        out = self.setup(
            "try:\n"
            "    _n = len(viewer.layers)\n"
            f"    print({SENTINEL!r}, 'viewer', type(viewer).__name__, _n)\n"
            "except Exception as _e:\n"
            f"    print({SENTINEL!r}, 'noviewer', _e)\n"
        )
        return (f"{SENTINEL} viewer" in out.text, out.text.strip())

    def client_is_none(self) -> bool:
        out = self.setup(f"print({SENTINEL!r}, 'client', client is None)")
        return f"{SENTINEL} client True" in out.text


def _write_config(root: Path) -> None:
    """A config tree of our own, so neither the developer's settings nor their
    personal skills reach the child."""
    (root / "biopb").mkdir(parents=True, exist_ok=True)
    (root / "biopb" / "mcp-config.json").write_text(
        json.dumps(
            {
                # One process, no cluster: the fixtures are small and a
                # LocalCluster is the slowest part of bring-up.
                "dask": {"scheduler": "threads"},
                # Nothing watches a web UI during an unattended run.
                "observe": {"enabled": False},
                "transport": {"kind": "http", "display_mode": "auto"},
            }
        ),
        encoding="utf-8",
    )
    # Present but empty: the local-skills dir must exist as *nothing*, so the
    # catalog under test is exactly what the package ships.
    (root / "biopb" / "skills").mkdir(exist_ok=True)


@contextmanager
def live_session() -> Iterator[LiveSession]:
    """Bring a session up, hand back a driver, and reap it on the way out."""
    if reason := why_unavailable():
        raise SessionUnavailable(reason)

    from biopb_mcp._config import load_config
    from biopb_mcp.mcp import _shim

    scratch = Path(tempfile.mkdtemp(prefix="biopb-skill-session-"))
    _write_config(scratch / "config")

    saved = {
        k: os.environ.get(k)
        for k in ("XDG_CONFIG_HOME", "BIOPB_TENSOR_URL", "QT_QPA_PLATFORM")
    }
    os.environ["XDG_CONFIG_HOME"] = str(scratch / "config")
    os.environ["BIOPB_TENSOR_URL"] = UNREACHABLE_TENSOR_URL
    os.environ.pop("QT_QPA_PLATFORM", None)  # a real GL platform, not offscreen

    child = session_id = loop = None
    stop = None
    try:
        child, url, session_id = _shim.spawn_session(
            load_config(), timeout=SPAWN_TIMEOUT
        )
        loop = _LoopThread()
        session, init, tools, stop = _connect(loop, url)
        live = LiveSession(
            url=url,
            session_id=session_id,
            instructions=(init.instructions or "").strip(),
            tools=tools,
            scratch=scratch,
            _loop=loop,
            _session=session,
        )
        ready = live.call("start_kernel")
        if ready.is_error:
            raise SessionUnavailable(f"start_kernel failed: {ready.text}")
        real, detail = live.has_real_viewer()
        if not real:
            raise SessionUnavailable(
                "the kernel came up without a napari viewer, so this run would "
                "score an environment where step 2 cannot happen at all. "
                f"Probe said: {detail}"
            )
        yield live
    finally:
        if loop is not None:
            try:
                # Let the driver leave its context managers before the loop
                # stops, so httpx closes its connection rather than being
                # cancelled mid-flight.
                if stop is not None:
                    loop.loop.call_soon_threadsafe(stop.set)
                loop.close()
            except Exception:  # noqa: BLE001 - teardown must not mask a failure
                pass
        if child is not None:
            _shim._reap_session(child, session_id)
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        shutil.rmtree(scratch, ignore_errors=True)


def _connect(loop: _LoopThread, url: str):
    """Enter the streamable-http client and keep it open on *loop*'s thread.

    The MCP client is a pair of nested async context managers that must stay
    entered for the life of the session, so they are entered inside a driver
    coroutine that then parks until closed — rather than per call, which would
    reinitialise the server on every tool use.
    """
    from mcp import ClientSession
    from mcp.client import streamable_http

    # Renamed in the SDK; the old spelling still works but deprecation-warns.
    # biopb-mcp floors `mcp>=1.20`, which is below the rename, so both are
    # reachable and the fallback is not dead code.
    connect = (
        getattr(streamable_http, "streamable_http_client", None)
        or streamable_http.streamablehttp_client
    )

    ready = threading.Event()
    box: dict[str, Any] = {}

    async def driver():
        try:
            async with connect(url) as (read, write, _):
                async with ClientSession(read, write) as session:
                    box["init"] = await session.initialize()
                    box["tools"] = (await session.list_tools()).tools
                    box["session"] = session
                    box["stop"] = asyncio.Event()
                    ready.set()
                    await box["stop"].wait()
        except BaseException as exc:  # noqa: BLE001 - reported to the caller
            box.setdefault("error", exc)
            ready.set()

    asyncio.run_coroutine_threadsafe(driver(), loop.loop)
    if not ready.wait(timeout=SPAWN_TIMEOUT):
        raise SessionUnavailable(f"MCP client never became ready at {url}")
    if "error" in box:
        raise SessionUnavailable(f"MCP connect failed: {box['error']!r}")

    tools = [
        ToolSpec(t.name, t.description or "", t.inputSchema or {}) for t in box["tools"]
    ]
    return box["session"], box["init"], tools, box["stop"]
