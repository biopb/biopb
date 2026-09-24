"""Shared pytest fixtures for biopb-mcp tests."""

import asyncio
import json
import pathlib

import pytest

from biopb_mcp._config import CONFIG


def call_tool(fn, *args, **kwargs):
    """Drive an ``async def`` MCP tool from a synchronous test.

    The tools are async so their kernel round trips go to a thread rather
    than stalling the process's one event loop (``_kernel_rpc._job_call``).
    The suite stays synchronous and gives each call its own loop -- the
    ``asyncio.run`` convention the chat tests already use.
    """
    return asyncio.run(fn(*args, **kwargs))


def rpc_reply(r, window_alive=True):
    """A kernel ``execute`` result carrying a job call's return value *r*, the
    way ``_kernel_rpc._run_job_call`` reads it back: a ``user_expression``
    holding ``{"r": r, "w": <viewer window alive?>}`` as JSON."""
    payload = json.dumps({"r": r, "w": window_alive})
    return {
        "stdout": "",
        "result_text": "",
        "error_text": "",
        "status": "ok",
        "user_expressions": {
            "rpc": {"status": "ok", "data": {"text/plain": repr(payload)}}
        },
    }


def iopub_event(content):
    """A ``biopb_job`` announcement as the host's iopub listener receives it."""
    from biopb_mcp.mcp._job_log import MSG_TYPE

    return {"header": {"msg_type": MSG_TYPE}, "parent_header": {}, "content": content}


def kernel_snapshot(status="ok", cells=None, error_text="", **job_kw):
    """``_jobs.poll``'s answer, built from a real ``_Job`` so a field added to
    the snapshot reaches every test that settles from one. *cells* is a list of
    ``(code, status)`` making it a verification."""
    from biopb_mcp.mcp import _jobs

    job = _jobs._Job(
        **{"job_id": "job-1", "code": "x = 1", "request": "req-1", **job_kw}
    )
    if cells is not None:
        job.verify = _jobs._Verification("wf", [code for code, _ in cells])
        for cell, (_code, cell_status) in zip(job.verify.cells, cells, strict=True):
            cell.status = cell_status
    job.status = status
    job.error_text = error_text
    return job.snapshot()


class ScriptedJobs:
    """``host.jobs`` for a mock host: the host's job records, scripted.

    Polls are answered from *polls* in order, the last one repeating (none:
    every job is unknown); the running job is *running*, an id or ``None``;
    the job list and the export are *summary* and
    *export*. The foreign-activity digest is *digest*, less what
    has been acked; acks and the point of view each read was made from are
    recorded for the test to assert on.
    """

    def __init__(
        self,
        polls=(),
        digest=(),
        summary=(),
        export=(),
        running=None,
        running_origin="mcp",
        window=None,
    ):
        self._polls = list(polls)
        self._running = running
        self._running_origin = running_origin
        self._window = window
        self._summary = list(summary)
        self._export = list(export)
        self.polled = 0
        self._digest = list(digest)
        self.acked = []
        self.digest_origins = []

    def poll(self, job_id):
        self.polled += 1
        if not self._polls:
            return {"job_id": job_id, "status": "unknown", "error_text": ""}
        snap = self._polls.pop(0) if len(self._polls) > 1 else self._polls[0]
        return {"job_id": job_id, **snap}

    def new_id(self):
        self._ids = getattr(self, "_ids", 0) + 1
        return f"job-{self._ids}"

    def running(self, prefer=None):
        if self._running is None:
            return None
        return {
            "job_id": self._running,
            "status": "running",
            "origin": self._running_origin,
        }

    def running_cell(self):
        return None

    def window_alive(self, job_id):
        return self._window

    def foreign_digest(self, for_origin):
        self.digest_origins.append(for_origin)
        return [d for d in self._digest if d["job_id"] not in self.acked]

    def ack_foreign_digest(self, job_ids):
        self.acked.extend(job_ids)
        return len(job_ids)

    def summary(self):
        return list(self._summary)

    def export(self):
        return list(self._export)


def pytest_addoption(parser):
    """Register the benchmark's run options (`--bench-fixtures`, `--bench-skills`, …).

    Here rather than in `bench/conftest.py` because pytest calls this hook only
    on the conftests it loads at *startup* — the rootdir's and those on the way
    down to the arguments. An option declared any deeper is silently never
    registered, so `pytest _tests` would reject the flag that `pytest
    _tests/bench` accepts.

    The consequence for callers, documented in `bench/README.md`: a `--bench-*`
    flag needs an argument **at or below this directory**. `pytest biopb-mcp`
    and a bare `pytest` from the repo root never load this file at startup and
    reject the flags as unrecognized.

    `bench._options` is stdlib-only for the same reason from the other side:
    this import runs on every pytest invocation in the repo, including the ones
    that never collect a benchmark.
    """
    from .bench._options import add_options

    add_options(parser)


def pytest_configure(config):
    """Reject an unusable benchmark option at startup, in one line.

    Resolution happens again during collection, and an unreadable value raised
    from there arrives as one collection error per module in `bench/` — five
    tracebacks for a typo. Doing it here turns that into pytest's own usage
    error, before anything is collected.

    A stale `BIOPB_BENCH_*` in someone's shell therefore stops a run that was
    never going to collect a benchmark. That is the intended trade: the
    variable is namespaced, it means nothing else, and the failure it prevents
    is a paid run that quietly did the larger thing.
    """
    from .bench._options import BadOption, resolve

    try:
        resolve(config)
    except BadOption as exc:
        raise pytest.UsageError(str(exc)) from exc


@pytest.fixture(autouse=True)
def _isolate_config(monkeypatch, tmp_path):
    """Isolate the config singleton + config dir for every test.

    Two hazards the process-wide ``CONFIG`` singleton (issue #31) introduces for
    tests:

    1. *State leakage* -- the cache persists across tests, so a value loaded (or
       written) in one test would bleed into the next.
    2. *Non-hermeticity* -- call sites now hit ``CONFIG.get(...)``, whose first
       access reads the developer's real ``~/.config/biopb/mcp-config.json``.

    This autouse fixture points ``Path.home()`` at a per-test ``tmp_path`` (so an
    untouched config resolves to defaults), clears inherited ``BIOPB_*`` / ``XDG_*`` env vars
    so tests start from the conventional defaults, and invalidates the cache
    before and after each test. ``monkeypatch`` is function-scoped, so tests that
    set their own ``Path.home`` compose with this -- their setattr runs later and
    wins, sharing the same ``tmp_path``.
    """
    for var in (
        "BIOPB_CONFIG_HOME",
        "BIOPB_STATE_HOME",
        "BIOPB_DATA_HOME",
        "XDG_CONFIG_HOME",
        "XDG_STATE_HOME",
        "XDG_DATA_HOME",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(pathlib.Path, "home", classmethod(lambda cls: tmp_path))
    CONFIG.reload()
    yield
    CONFIG.reload()


@pytest.fixture(autouse=True)
def _isolate_loaded_plugins():
    """Reset the kernel-plugin record between tests.

    ``mcp/_requires.py`` holds what the plugin loader loaded in module state (one
    kernel, one record — see its docstring), so any test that drives a load leaks
    into the next one's ``server_status`` report.
    """
    from biopb_mcp.mcp import _requires

    def clear():
        _requires.record_loaded_plugins()

    clear()
    yield
    clear()
