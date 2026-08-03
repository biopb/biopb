"""A real biopb-mcp session, brought up and driven from synchronous test code.

`biopb-mcp/docs/skill-testing.md` §5b: a real shim-spawned session child, a
real IPython kernel, a real napari viewer, real dask — and the nine real tools
reached over real MCP. Nothing here stands in for the runtime. That is the
whole point: a hand-written tool surface would put `execute_code`'s return
shape, `server_status`'s report and the `guide://` bodies into a transcription
that no longer tracks what the runtime does.

What this module owns is bring-up, a synchronous façade over the async MCP
client, and three environment facts that have to be *forced* rather than
inherited, because each of them silently changes what a run is testing:

**A display, and a GL context behind it.** With no `$DISPLAY` the launcher
spawns its own Xvfb and renders the viewer there (`mcp/_xvfb.py`), so a
display-less box with the `xvfb` package installed runs these tests unaided.
What still cannot be conjured is the binary itself (plus Mesa's software GL
behind it): absent both a display and Xvfb, the session child fails fast at
spawn — so these tests **skip with instructions** rather than pay the slow
bring-up for a guaranteed failure. (`QT_QPA_PLATFORM=offscreen` is never a
substitute: napari builds, then `add_image` dies inside vispy's extension
probe, because offscreen Qt has no GL context.)

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
from collections.abc import Iterator, Sequence
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
        from biopb_mcp.mcp import _xvfb

        if not _xvfb.available():
            return (
                "no display and no Xvfb: §5 needs a GL-capable display for "
                "napari layers. Install xvfb (the launcher then provides its "
                "own virtual display) or run on a desktop session. "
                "QT_QPA_PLATFORM=offscreen alone is NOT enough — vispy needs a "
                "GL context that the offscreen platform does not provide."
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


def describe_block(block) -> str:
    """One MCP content block as the text an agent sees. Never silently empty.

    **A dropped block reads as a broken tool.** `take_screenshot` returns an
    `ImageContent` — `.data` and `.mimeType`, no `.text` — so a filter that
    kept only blocks with text handed the agent an empty string for a
    screenshot that had in fact been captured. In a measured run the agent
    built a montage specifically to look at, got nothing back twice, concluded
    "the screenshot tool isn't returning images", and spent the rest of the arm
    working around a tool that was working.

    A text model cannot read the PNG either way; what it can do is tell "no
    image" apart from "an image I cannot see", and only one of those is worth
    building a workaround for. Passing the pixels to a vision-capable agent is
    a separate change — tool results are strings on this API, so an image has
    to travel as its own user message.
    """
    if text := getattr(block, "text", None):
        return text
    if (data := getattr(block, "data", None)) is not None:
        kind = getattr(block, "mimeType", "") or "binary"
        # base64 -> bytes, near enough for a size the agent can reason about.
        kb = max(1, len(data) * 3 // 4 // 1024)
        return f"[{kind}, ~{kb} KB — returned, but not readable as text]"
    return f"[{getattr(block, 'type', 'unknown')} content block]"


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
    #: Whether the curated catalog was offered at all (the ablation arm).
    skills_enabled: bool
    _loop: _LoopThread
    _session: Any
    _turn: int = 0
    calls: list[tuple[int, str, dict]] = field(default_factory=list)

    # --- the agent-visible surface -----------------------------------------

    def call(self, name: str, /, **arguments: Any) -> ToolResult:
        """Call a tool exactly as an agent would, and record that it happened.

        The record is what answers "did it ask before it spent": which tool,
        with what, and at which conversational turn.
        """
        self.calls.append((self._turn, name, dict(arguments)))
        result = self._loop.submit(
            self._session.call_tool(name, arguments), CALL_TIMEOUT
        )
        text = "\n".join(describe_block(block) for block in result.content)
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
        """Whether `viewer` is a live napari viewer with a working canvas."""
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


def _write_config(
    root: Path, *, skills_enabled: bool = True, plugins: Sequence[str] = ()
) -> None:
    """A config tree of our own, so neither the developer's settings nor their
    personal skills reach the child.

    ``skills_enabled=False`` is the **ablation arm**: the ``find_skills`` tool
    stays registered but ``load_catalog()`` returns an empty list, so the agent
    can call it and get nothing back, while the kernel, napari, dask and every
    library stay exactly as they were. That is §5's rule — disclose the
    environment, withhold only the skill — and it is a real shipped
    configuration rather than a hole cut for the test.

    ``plugins`` names kernel plugins the case's skill declares in its
    ``checklist:``. They are seeded into this tree's own ``biopb/kernel/`` from
    the ones biopb-mcp ships, so the loader that runs is the real one — and only
    what a case asks for is present, since a plugin the skill never declared is
    an environment difference nobody chose.
    """
    (root / "biopb").mkdir(parents=True, exist_ok=True)
    if plugins:
        from biopb_mcp import plugins as bundled

        kernel_dir = root / "biopb" / "kernel"
        kernel_dir.mkdir(exist_ok=True)
        source = Path(bundled.__file__).parent
        for name in plugins:
            shutil.copyfile(source / f"{name}.py", kernel_dir / f"{name}.py")
    (root / "biopb" / "mcp-config.json").write_text(
        json.dumps(
            {
                # One process, no cluster: the fixtures are small and a
                # LocalCluster is the slowest part of bring-up.
                "dask": {"scheduler": "threads"},
                # Nothing watches a web UI during an unattended run.
                "observe": {"enabled": False},
                "transport": {"kind": "http"},
                "services": {"skills_enabled": skills_enabled},
            }
        ),
        encoding="utf-8",
    )
    # Present but empty: the local-skills dir must exist as *nothing*, so the
    # catalog under test is exactly what the package ships.
    (root / "biopb" / "skills").mkdir(exist_ok=True)


@contextmanager
def live_session(
    *, skills_enabled: bool = True, plugins: Sequence[str] = ()
) -> Iterator[LiveSession]:
    """Bring a session up, hand back a driver, and reap it on the way out.

    ``skills_enabled=False`` withholds the curated catalog and nothing else
    -- the ablation arm of the benchmark. ``plugins`` seeds the kernel plugins
    a case's skill declares.
    """
    if reason := why_unavailable():
        raise SessionUnavailable(reason)

    from biopb_mcp._config import load_config
    from biopb_mcp.mcp import _shim

    scratch = Path(tempfile.mkdtemp(prefix="biopb-skill-session-"))
    _write_config(scratch / "config", skills_enabled=skills_enabled, plugins=plugins)

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
            skills_enabled=skills_enabled,
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
