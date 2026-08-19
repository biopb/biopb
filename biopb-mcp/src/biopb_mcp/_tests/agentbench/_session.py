"""A real biopb-mcp session, brought up and driven from synchronous test code.

`biopb-mcp/docs/skills.md` §10b: a real shim-spawned session child, a
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

**A config tree of our own.** `BIOPB_CONFIG_HOME` points at a temp dir, so the
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

#: Env the tripwire reads in the child: where to write, and what to watch.
ENV_GUARD_LOG = "BIOPB_GUARD_LOG"
ENV_GUARD_MARKERS = "BIOPB_GUARD_MARKERS"

#: Installed into every child process of a run, as `sitecustomize` on its
#: `PYTHONPATH`. Records — never blocks — reads of anything the harness owns.
#:
#: `execute_code` is arbitrary Python by design, so a run can open the fixture
#: that defines its own answer (`truth["structural_channel"]`, the trajectory,
#: the tolerances, the persona's facts) or the skill markdown an ablated arm is
#: supposed to lack. Both have happened: a measured `skill+asked` arm reached
#: its procedure by walking the installed package and opening
#: `mcp/_skills_data/drift-correction.md`.
#:
#: Recording rather than refusing, on purpose. The agent is curious, not
#: adversarial, and it says what it did in the trace; what the layer actually
#: needs is for a compromised run to be *loud* instead of scoring like a good
#: one. Refusing would also change the environment under test, which §5
#: forbids — "disclose the environment, withhold only the skill" — and would
#: break the session child's own legitimate reads of `_skills_data`. Judgement
#: about which reads matter belongs in the parent, where it is testable, so the
#: hook stays a dumb recorder and writes down who was asking.
_TRIPWIRE = '''\
"""Benchmark tripwire — records reads of harness-owned paths. Not security."""

import json
import os
import sys

_MARKERS = tuple(
    m for m in os.environ.get({markers!r}, "").split(os.pathsep) if m
)
_LOG = os.environ.get({log!r}, "")
_busy = False


def _watch(event, args):
    global _busy
    if _busy or event not in ("open", "os.listdir", "os.scandir"):
        return
    target = args[0] if args else None
    if not isinstance(target, (str, bytes, os.PathLike)):
        return
    try:
        path = os.path.abspath(os.fsdecode(target))
    except Exception:
        return
    if not any(m in path for m in _MARKERS):
        return
    _busy = True
    try:
        # Which process was asking is the whole discrimination: the session
        # child reads `_skills_data` to serve `skill://`, and that is the
        # system working. The kernel is where agent code runs.
        with open(_LOG, "a", encoding="utf-8") as fh:
            fh.write(
                json.dumps(
                    {{
                        "pid": os.getpid(),
                        "event": event,
                        "path": path,
                        "in_kernel": "ipykernel" in sys.modules,
                    }}
                )
                + "\\n"
            )
    except Exception:
        pass
    finally:
        _busy = False


if _MARKERS and _LOG:
    sys.addaudithook(_watch)
'''


def guard_markers() -> list[str]:
    """What a run must not read: the harness's own tree, and the shipped skill
    bodies.

    `_tests/` holds every case's `truth`, its tolerances and its persona — read
    it and any arm passes, with nothing in the result to say so. `_skills_data`
    is narrower: legitimate for the session child to read, and the ablation's
    undoing if the *kernel* reads it.

    Path *fragments*, not absolute roots, because the parent and the child need
    not resolve `biopb_mcp` to the same tree — an editable install re-points it
    through its own finder, and under a git worktree the two diverge outright.
    Rooted at the package directory rather than a bare name, so this says
    "biopb_mcp's tests", not "any directory called `_tests`".
    """
    sep = os.sep
    return [
        f"{sep}biopb_mcp{sep}_tests{sep}",
        f"{sep}biopb_mcp{sep}mcp{sep}_skills_data{sep}",
    ]


#: Built once per process by :func:`staged_package`.
_STAGED: Path | None = None


def staged_package() -> Path:
    """biopb-mcp as it *ships*, unpacked, for the child to import instead of
    the checkout.

    The benchmark runs from a source tree, where `_tests/` sits inside the
    installed package — so `os.path.dirname(biopb_mcp.__file__)` walks straight
    into every case's `truth`, its tolerances and its persona. A measured arm
    made exactly that walk (looking for the skill markdown, which does ship).
    The wheel excludes `_tests` in both of the two places that matter —
    `packages.find` *and* `exclude-package-data` — so building one and putting
    it first on the child's path removes the answer key from the only process
    that could read it, and leaves the child running what users actually have.

    Prepending is enough because biopb-mcp's editable install is a plain path
    `.pth`, and `PYTHONPATH` is processed before site-packages contributes
    those entries — so the staged copy shadows the checkout. It would not
    shadow a *finder*-style editable (`biopb-tensor-server` has one), which is
    why this is asserted at bring-up rather than assumed.

    Loud on failure, never silent: an unstaged run is a run whose numbers can be
    read off a file, and degrading quietly would reintroduce the exact class of
    bug this exists to close.
    """
    global _STAGED
    if _STAGED is not None:
        return _STAGED

    import subprocess
    import zipfile

    from ._fixture import checkout_root

    root = checkout_root()
    if root is None:
        raise SessionUnavailable(
            "no checkout around this module, so there is no workspace to build "
            "the biopb-mcp wheel from"
        )
    out = Path(tempfile.mkdtemp(prefix="biopb-skill-wheel-"))
    try:
        subprocess.run(
            ["uv", "build", "--package", "biopb-mcp", "--wheel", "-o", str(out)],
            cwd=root,
            check=True,
            capture_output=True,
            timeout=300,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        detail = getattr(exc, "stderr", b"") or b""
        raise SessionUnavailable(
            f"could not build the biopb-mcp wheel to isolate the run from its "
            f"own test tree: {exc}\n{detail[-500:].decode('utf-8', 'replace')}"
        ) from exc

    wheels = sorted(out.glob("*.whl"))
    if not wheels:
        raise SessionUnavailable(f"uv build produced no wheel in {out}")
    unpacked = out / "unpacked"
    with zipfile.ZipFile(wheels[-1]) as zf:
        zf.extractall(unpacked)
    if (unpacked / "biopb_mcp" / "_tests").exists():
        raise SessionUnavailable(
            "the built wheel still contains biopb_mcp/_tests — the packaging "
            "excludes have regressed and a run could read its own answer key"
        )
    _STAGED = unpacked
    return _STAGED


#: How long bring-up may take. The kernel imports napari and spins dask, which
#: is seconds on a warm machine and much worse on a cold one.
SPAWN_TIMEOUT = 120.0
KERNEL_TIMEOUT = 300.0
CALL_TIMEOUT = 300.0

#: Printed by kernel-side snippets so the façade can find a path in output that
#: also carries whatever the agent's code decided to print.
SENTINEL = "__BIOPB_HARNESS__"

#: An address nothing listens on, to keep `client` at None for a case presented
#: as `array`. Port 1 is privileged and unbound; the connect fails fast rather
#: than hanging. A case presented on the run's plane passes that plane's url
#: instead — see `live_session(tensor_url=...)`.
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


#: What the **client** contributes, on top of the nine tools the server
#: advertises. Not `@mcp.tool()`s, and deliberately not: they are the harness
#: standing in for a capability every shipped MCP client already has.
#:
#: A skill body and a `guide://` page are MCP **resources**, and a resource is
#: not a tool. `find_skills` returns metadata plus a `uri`, the handshake
#: instructions say to read that uri, and `_bridge` translates *tools* onto a
#: chat-completions API — so before this existed the agent was handed a pointer
#: it had no verb to dereference. Measured on the 2026-08-03 sweep, that cost
#: the benchmark its independent variable: `skill+silent` reached for
#: `pystackreg` because the catalog *metadata* named it in `checklist:`, having
#: never read a line of the procedure, and `skill+asked` got the body only by
#: walking the installed package directory. A broken or empty skill body would
#: have scored the same.
#:
#: The ablation still holds through here, and is not re-implemented: the
#: `skill://` resource resolves through `load_catalog()`, which returns `[]`
#: when `services.skills_enabled` is off, so an ablated run that reads the uri
#: gets "No skill '<id>' in the catalog" — the server's own answer, not one the
#: harness invented.
CLIENT_TOOLS: tuple[ToolSpec, ...] = (
    ToolSpec(
        name="list_resources",
        description=(
            "List the MCP resources this server exposes, including URI "
            "templates. Resources carry reference material rather than "
            "actions: the `guide://` pages and the full body of each "
            "curated skill."
        ),
        input_schema={"type": "object", "properties": {}},
    ),
    ToolSpec(
        name="read_resource",
        description=(
            "Read one MCP resource and return its text. `uri` is a full "
            "resource URI — for example `skill://drift-correction` (the `uri` "
            "field of a `find_skills` result, whose body holds the actual "
            "step-by-step workflow) or `guide://kernel`."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "uri": {
                    "type": "string",
                    "description": "e.g. skill://<id> or guide://<name>",
                }
            },
            "required": ["uri"],
        },
    ),
)

#: Names in :data:`CLIENT_TOOLS`, for dispatch.
CLIENT_TOOL_NAMES = frozenset(t.name for t in CLIENT_TOOLS)


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
    #: Whether the curated catalog was offered at all (`--bench-skills`).
    skills_enabled: bool
    #: Where the tripwire writes. Absent until something is recorded.
    guard_log: Path = Path()
    #: The session child's own pid. It reads `_skills_data` to serve
    #: `skill://`, which is the system working, so `peeked` needs to tell that
    #: process apart from the kernel underneath it.
    child_pid: int | None = None
    _loop: _LoopThread = None  # type: ignore[assignment]
    _session: Any = None
    _turn: int = 0
    calls: list[tuple[int, str, dict]] = field(default_factory=list)

    # --- the agent-visible surface -----------------------------------------

    @property
    def agent_tools(self) -> list[ToolSpec]:
        """Everything the agent can actually do: the server's tools plus the
        client's resource verbs.

        Kept distinct from `tools` — which stays the server's own
        advertisement — because conflating the two is the bug this exists to
        fix. What a server offers and what an agent can reach are different
        lists, and the gap between them was invisible while only one was named.
        """
        return [*self.tools, *CLIENT_TOOLS]

    def call(self, name: str, /, **arguments: Any) -> ToolResult:
        """Call a tool exactly as an agent would, and record that it happened.

        The record is what answers "did it ask before it spent": which tool,
        with what, and at which conversational turn — so a client-side call is
        recorded here too, on the same footing as a server one.
        """
        self.calls.append((self._turn, name, dict(arguments)))
        if name in CLIENT_TOOL_NAMES:
            return self._call_client_tool(name, arguments)
        result = self._loop.submit(
            self._session.call_tool(name, arguments), CALL_TIMEOUT
        )
        text = "\n".join(describe_block(block) for block in result.content)
        return ToolResult(name=name, text=text, is_error=bool(result.isError))

    def _call_client_tool(self, name: str, arguments: dict) -> ToolResult:
        """Serve one of :data:`CLIENT_TOOLS`.

        Errors come back as an error *result*, never an exception: a tool that
        raises out of the conversation loop ends the run, and "that uri does not
        resolve" is something an agent should be able to read and recover from.
        """
        try:
            if name == "list_resources":
                return ToolResult(name, self._list_resources())
            uri = str(arguments.get("uri", "")).strip()
            if not uri:
                return ToolResult(name, "read_resource needs a `uri`.", True)
            return ToolResult(name, self.read_resource(uri))
        except Exception as exc:  # noqa: BLE001 - handed to the agent as text
            return ToolResult(name, f"{name} failed: {exc!r}", True)

    def _list_resources(self) -> str:
        listed = self._loop.submit(self._session.list_resources(), CALL_TIMEOUT)
        templates = self._loop.submit(
            self._session.list_resource_templates(), CALL_TIMEOUT
        )
        lines = [
            f"{r.uri} — {r.description or r.name or ''}".rstrip(" —")
            for r in listed.resources
        ]
        lines += [
            f"{t.uriTemplate} — {t.description or t.name or ''}".rstrip(" —")
            for t in templates.resourceTemplates
        ]
        return "\n".join(lines) or "This server exposes no resources."

    def peeked(self) -> list[dict]:
        """Harness-owned files this run read that it had no business reading.

        Serving `skill://` means the *session child* opens `_skills_data`, and
        `load_catalog` scans the whole directory — the system working. The
        kernel is where agent code runs, so the discriminator is the process,
        and it is applied here rather than in the hook: the recorder stays dumb
        and the judgement stays somewhere a test can reach it.

        Not `"ipykernel" in sys.modules`, which reads true in the session child
        as well and quietly classified every catalog scan as a peek.
        """
        if not self.guard_log.exists():
            return []
        out = []
        for line in self.guard_log.read_text(encoding="utf-8").splitlines():
            try:
                entry = json.loads(line)
            except ValueError:
                continue
            serving = entry.get(
                "pid"
            ) == self.child_pid and "_skills_data" in entry.get("path", "")
            if not serving:
                out.append(entry)
        return out

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

    ``skills_enabled=False`` is what **`--bench-skills=false`** sets: the ``find_skills`` tool
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
    *,
    skills_enabled: bool = True,
    plugins: Sequence[str] = (),
    tensor_url: str = "",
) -> Iterator[LiveSession]:
    """Bring a session up, hand back a driver, and reap it on the way out.

    ``skills_enabled=False`` withholds the curated catalog and nothing else
    -- the ablated half of a skill's delta, `--bench-skills=false`. ``plugins``
    seeds the kernel plugins a case's skill declares.

    ``tensor_url`` is the run's data plane, for a case presented on one. Empty
    -- the usual state -- points the child at an address nothing answers, so
    ``client is None`` and the agent meets the environment every `array` case's
    task prompt describes. Either way the *control* plane is bypassed
    entirely: ``$BIOPB_TENSOR_URL`` is read before it is consulted, which is
    what keeps a benchmark run from touching the developer's own deployment.
    """
    if reason := why_unavailable():
        raise SessionUnavailable(reason)

    from biopb_mcp._config import load_config
    from biopb_mcp.mcp import _shim

    scratch = Path(tempfile.mkdtemp(prefix="biopb-skill-session-"))
    _write_config(scratch / "config", skills_enabled=skills_enabled, plugins=plugins)

    saved = {
        k: os.environ.get(k)
        for k in (
            "BIOPB_CONFIG_HOME",
            "BIOPB_TENSOR_URL",
            "QT_QPA_PLATFORM",
            "PYTHONPATH",
            ENV_GUARD_LOG,
            ENV_GUARD_MARKERS,
        )
    }
    os.environ["BIOPB_CONFIG_HOME"] = str(scratch / "config")
    os.environ["BIOPB_TENSOR_URL"] = tensor_url or UNREACHABLE_TENSOR_URL
    os.environ.pop("QT_QPA_PLATFORM", None)  # a real GL platform, not offscreen

    # The tripwire, inherited by the session child and its kernel. `sitecustomize`
    # is imported by `site` from anywhere on the path, which is what lets this
    # reach both processes without the product knowing it exists. Caveat: it
    # shadows any other `sitecustomize` on the way — none in this venv, and a
    # benchmark run is not a general-purpose environment.
    guard_log = scratch / "guard.jsonl"
    (scratch / "sitecustomize.py").write_text(
        _TRIPWIRE.format(markers=ENV_GUARD_MARKERS, log=ENV_GUARD_LOG),
        encoding="utf-8",
    )
    os.environ[ENV_GUARD_LOG] = str(guard_log)
    os.environ[ENV_GUARD_MARKERS] = os.pathsep.join(guard_markers())

    child = session_id = loop = None
    stop = None
    try:
        # Inside the try, because building the wheel can fail and everything
        # above has already redirected `BIOPB_CONFIG_HOME` and the tensor URL for
        # this whole process. Raising past the `finally` would leave that
        # redirect in place — every later test in the process reading a temp
        # config tree that no longer exists, which is a far worse failure than
        # the one being reported, and a silent one.
        #
        # Order matters: the staged wheel must precede whatever the checkout's
        # editable `.pth` will later append, and `scratch` only has to be
        # somewhere importable for `sitecustomize`.
        os.environ["PYTHONPATH"] = os.pathsep.join(
            [
                str(scratch),
                str(staged_package()),
                *([saved["PYTHONPATH"]] if saved["PYTHONPATH"] else []),
            ]
        )
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
            guard_log=guard_log,
            child_pid=child.pid,
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
