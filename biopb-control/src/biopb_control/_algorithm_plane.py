"""The algorithm plane: the registry's script entries, run under uv.

A script entry (``~/.config/biopb/algorithms/<name>.py``) is a server file
with a PEP 723 header. The control installs it (``uv lock --script`` then
``uv sync --script``), asks it for its ops (``uv run <file> --describe``) and
caches the answer by the file's hash, so its ops are known without running
it. The server starts on :meth:`AlgorithmPlane.ensure` -- the kernel ensures
on the first call to one of its ops -- and stays up until stopped or the
control exits.

An entry's states:

========== =========================================================
new        never installed at this file's hash; no cached ops
installing ``uv lock`` + ``uv sync`` + ``--describe`` are running
stopped    installed and described, ops cached, no process
starting   spawned, waiting for its port
up         serving
failed     install, describe or start failed; ``error`` holds the log tail
========== =========================================================

A failure before the port binds (a bad header, an import error) is
``failed`` and waits for the next ensure; only a server that was ``up`` and
crashed is restarted, with backoff. A url entry is probed and passed along:
its lifecycle and logs belong to whoever runs it.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import secrets
import shutil
import socket
import subprocess
import threading
import time
from pathlib import Path
from typing import Callable, Optional

from biopb import _algorithms, _locations

from ._supervisor import (
    _BACKOFF_SCHEDULE,
    _HEALTHY_RESET_SECONDS,
    ServiceProcess,
    ServiceSpec,
)

logger = logging.getLogger(__name__)

#: How long an install (``uv lock`` + ``uv sync``) may take: a first install
#: that pulls torch is minutes.
INSTALL_TIMEOUT = 3600.0
#: How long ``--describe`` may take: importing a model's stack is slow.
DESCRIBE_TIMEOUT = 600.0
#: Log lines an ``error`` carries.
ERROR_TAIL_LINES = 20

#: The token a server checks; the runtime's ``TOKEN_ENV``.
TOKEN_ENV = "BIOPB_ALGORITHM_TOKEN"


def find_uv() -> Optional[list[str]]:
    """The uv command: ``$BIOPB_UV``, else ``uv`` on PATH or where the uv
    installer puts it (a desktop-started control may not have it on PATH)."""
    override = os.environ.get("BIOPB_UV")
    if override:
        return [override]
    found = shutil.which("uv")
    if found:
        return [found]
    for candidate in (
        Path.home() / ".local" / "bin" / "uv",
        Path.home() / ".cargo" / "bin" / "uv",
    ):
        if candidate.is_file():
            return [str(candidate)]
    return None


def _file_hash(path: Path) -> Optional[str]:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _tail(path: Path, lines: int) -> list[str]:
    try:
        with open(path, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            fh.seek(max(0, fh.tell() - 64 * 1024))
            text = fh.read().decode("utf-8", errors="replace")
    except OSError:
        return []
    return text.splitlines()[-lines:]


class ScriptEntry(ServiceProcess):
    """One script entry: its install, its cached op list, and its server."""

    def __init__(
        self,
        name: str,
        path: Path,
        *,
        state_dir: Path,
        uv: Optional[list[str]],
        data_plane: Callable[[], Optional[tuple[str, Optional[str]]]],
    ):
        super().__init__()
        self.name = name
        self.path = path
        self._uv = uv or None
        self._data_plane = data_plane
        self._cache_path = state_dir / f"{name}.describe.json"
        self._log = state_dir / f"{name}.log"
        self._lock = threading.RLock()
        self._installer: Optional[threading.Thread] = None
        self._installed = threading.Event()
        self._installed.set()
        # The failure that holds the entry in `failed`, and the file hash it
        # failed at: an install failure is retried by refresh only once the
        # file changes, and by ensure always.
        self._error: Optional[str] = None
        self._failed_hash: Optional[str] = None
        # The running server.
        self._port = 0
        self._token = ""
        self._running_hash: Optional[str] = None
        self._want = False
        self._was_up = False
        self._up_since: Optional[float] = None
        self._failures = 0
        self._restarts = 0
        self._next_attempt_at = 0.0
        self._log_rotated = False

    # --- ServiceProcess -------------------------------------------------- #

    def _service_spec(self) -> ServiceSpec:
        env = os.environ.copy()
        env[TOKEN_ENV] = self._token
        plane = self._data_plane()
        if plane is not None:
            env["BIOPB_TENSOR_URL"] = plane[0]
            if plane[1]:
                env["BIOPB_TENSOR_TOKEN"] = plane[1]
        argv = [*(self._uv or ["uv"]), "run", "--script", str(self.path)]
        argv += ["--host", "127.0.0.1", "--port", str(self._port)]
        return ServiceSpec(
            argv=argv,
            env=env,
            host="127.0.0.1",
            port=self._port,
            log_path=self._log,
            label=f"algorithm {self.name}",
        )

    def _open_log(self):
        self._log_rotated = True
        return super()._open_log()

    # --- install and describe ------------------------------------------- #

    def cached(self) -> Optional[dict]:
        """The cached ``{"hash", "oplist"}``, or ``None``."""
        try:
            data = json.loads(self._cache_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return data if isinstance(data, dict) and "hash" in data else None

    def _append_log(self, text: str) -> None:
        if not self._log_rotated:
            _locations.rotate_log(self._log)
            self._log_rotated = True
        with open(self._log, "a", encoding="utf-8") as fh:
            fh.write(text)

    def _fail(self, what: str, file_hash: Optional[str]) -> None:
        tail = "\n".join(_tail(self._log, ERROR_TAIL_LINES))
        self._error = f"{what}\n{tail}".strip()
        self._failed_hash = file_hash
        logger.warning("algorithm %s: %s", self.name, what)

    def _run_step(
        self, argv: list[str], timeout: float, *, stdout=None
    ) -> subprocess.CompletedProcess:
        self._append_log(f"\n--- control: {' '.join(argv)} ---\n")
        with open(self._log, "ab") as log:
            return subprocess.run(
                argv,
                stdin=subprocess.DEVNULL,
                stdout=stdout if stdout is not None else log,
                stderr=log,
                timeout=timeout,
                check=False,
            )

    def _install(self) -> None:
        """Lock, sync and describe the file as it is now. Runs off the lock."""
        file_hash = _file_hash(self.path)
        try:
            if self._uv is None:
                self._fail(
                    "uv is not installed; set $BIOPB_UV or put uv on PATH", file_hash
                )
                return
            script = ["--script", str(self.path)]
            for verb in ("lock", "sync"):
                done = self._run_step([*self._uv, verb, *script], INSTALL_TIMEOUT)
                if done.returncode != 0:
                    self._fail(f"uv {verb} failed (exit {done.returncode})", file_hash)
                    return
            done = self._run_step(
                [*self._uv, "run", *script, "--describe"],
                DESCRIBE_TIMEOUT,
                stdout=subprocess.PIPE,
            )
            if done.returncode != 0:
                self._fail(f"--describe failed (exit {done.returncode})", file_hash)
                return
            try:
                oplist = json.loads(done.stdout)
            except ValueError:
                self._append_log(done.stdout.decode("utf-8", errors="replace"))
                self._fail("--describe printed no op list", file_hash)
                return
            self._cache_path.write_text(
                json.dumps({"hash": file_hash, "oplist": oplist}), encoding="utf-8"
            )
            self._error = None
            self._failed_hash = None
            logger.info("algorithm %s: installed and described", self.name)
        except subprocess.TimeoutExpired as exc:
            self._fail(
                f"{exc.cmd[1] if len(exc.cmd) > 1 else 'uv'} timed out", file_hash
            )
        except OSError as exc:
            self._fail(f"could not run uv: {exc}", file_hash)
        finally:
            with self._lock:
                self._installer = None
                self._installed.set()

    def start_install(self, *, retry_failed: bool) -> None:
        """Install in the background unless the cache is current, an install
        is running, or it failed at this hash (and *retry_failed* is off)."""
        with self._lock:
            if self._installer is not None:
                return
            file_hash = _file_hash(self.path)
            cached = self.cached()
            if cached is not None and cached["hash"] == file_hash:
                return
            if not retry_failed and self._failed_hash == file_hash and self._error:
                return
            self._error = None
            self._installed.clear()
            self._installer = threading.Thread(
                target=self._install, name=f"install-{self.name}", daemon=True
            )
            self._installer.start()

    # --- the server ------------------------------------------------------ #

    def _stop_locked(self) -> None:
        self._want = False
        proc = self._proc
        self._proc = None
        self._running_hash = None
        if proc is not None:
            self._terminate(proc, timeout=5.0)
        self._close_death_pipe()

    def _spawn_locked(self) -> bool:
        self._port = _free_port()
        self._token = secrets.token_urlsafe(32)
        try:
            self._start_process()
        except OSError as exc:
            self._fail(f"could not start: {exc}", _file_hash(self.path))
            self._want = False
            return False
        self._running_hash = _file_hash(self.path)
        self._was_up = False
        self._up_since = None
        return True

    def ensure(self, wait: float) -> None:
        """Install if the file changed, start if not running, and wait up to
        *wait* seconds for the port. Restarts a server running an older file."""
        deadline = time.monotonic() + wait
        self.start_install(retry_failed=True)
        if not self._installed.wait(max(0.0, deadline - time.monotonic())):
            return  # still installing: the row says so
        with self._lock:
            cached = self.cached()
            if cached is None or cached["hash"] != _file_hash(self.path):
                return  # the install failed: the row carries why
            # A start that failed at this version is retried.
            self._error = None
            self._failed_hash = None
            self._reap_locked()
            if self._proc is not None and self._running_hash != cached["hash"]:
                self._stop_locked()
            self._want = True
            if self._proc is None and not self._spawn_locked():
                return
        while time.monotonic() < deadline:
            with self._lock:
                self.tick_locked()
                if self._proc is None or self._was_up:
                    return
            time.sleep(0.1)

    def stop(self) -> None:
        with self._lock:
            self._stop_locked()
        self._close_child_bindings()
        self._close_log()

    def _reap_locked(self) -> None:
        """Handle a child that exited: a server that was up restarts with
        backoff, one that never bound its port is ``failed``."""
        rc = self._reap_dead_child()
        if rc is None:
            return
        self._running_hash = None
        if self._was_up and self._want:
            self._restarts += 1
            self._failures += 1
            self._error = None
            idx = min(self._failures, len(_BACKOFF_SCHEDULE) - 1)
            self._next_attempt_at = time.monotonic() + _BACKOFF_SCHEDULE[idx]
            logger.warning("algorithm %s exited (code %s); restarting", self.name, rc)
        else:
            self._want = False
            self._fail(f"exited before serving (code {rc})", _file_hash(self.path))
        self._was_up = False
        self._up_since = None

    def tick_locked(self) -> None:
        self._reap_locked()
        if self._proc is not None:
            if self._port_up():
                now = time.monotonic()
                self._was_up = True
                if self._up_since is None:
                    self._up_since = now
                elif now - self._up_since >= _HEALTHY_RESET_SECONDS:
                    self._failures = 0
            return
        if self._want and time.monotonic() >= self._next_attempt_at:
            self._spawn_locked()

    def tick(self) -> None:
        with self._lock:
            self.tick_locked()

    # --- status ---------------------------------------------------------- #

    def state(self) -> str:
        with self._lock:
            if self._installer is not None:
                return "installing"
            if self._error:
                return "failed"
            if self._child_alive():
                return "up" if self._was_up else "starting"
            if self._want:
                return "starting"  # between a crash and its restart
            cached = self.cached()
            if cached is not None and cached["hash"] == _file_hash(self.path):
                return "stopped"
            return "new"

    def row(self) -> dict:
        with self._lock:
            state = self.state()
            cached = self.cached()
            current = cached is not None and (
                cached["hash"] == _file_hash(self.path)
                or cached["hash"] == self._running_hash
            )
            oplist = cached["oplist"] if current else {}
            running = state in ("up", "starting") and self._proc is not None
            return _algorithms.row(
                {"name": self.name, "kind": "script", "url": None},
                url=f"grpc://127.0.0.1:{self._port}" if running else None,
                state=state,
                ops=oplist.get("ops", []),
                fingerprint=oplist.get("fingerprint", ""),
                error=self._error,
                token=self._token if running else None,
                restarts=self._restarts,
            )

    def logs(self, lines: int) -> dict:
        return {
            "name": self.name,
            "path": str(self._log),
            "exists": self._log.exists(),
            "lines": _tail(self._log, lines),
        }


class AlgorithmPlane:
    """Every registry entry: script entries supervised, url entries probed."""

    def __init__(
        self,
        *,
        directory: Optional[Path] = None,
        state_dir: Optional[Path] = None,
        uv: Optional[list[str]] = None,
        data_plane: Callable[[], Optional[tuple[str, Optional[str]]]] = lambda: None,
    ):
        self._directory = directory
        self._state_dir = state_dir
        self._uv = uv if uv is not None else find_uv()
        self._data_plane = data_plane
        self._scripts: dict[str, ScriptEntry] = {}
        self._lock = threading.Lock()

    def _entries(self) -> list[dict]:
        """The registry now, with the script entries' supervisors kept in step:
        a new file gets one, and a removed file's server is stopped."""
        listed = _algorithms.entries(self._directory)
        state_dir = self._state_dir or _locations.algorithms_state_dir()
        gone = []
        with self._lock:
            names = set()
            for entry in listed:
                if entry["kind"] != "script" or entry["error"]:
                    continue
                names.add(entry["name"])
                known = self._scripts.get(entry["name"])
                if known is None or str(known.path) != entry["path"]:
                    self._scripts[entry["name"]] = ScriptEntry(
                        entry["name"],
                        Path(entry["path"]),
                        state_dir=state_dir,
                        uv=self._uv,
                        data_plane=self._data_plane,
                    )
            for name in set(self._scripts) - names:
                gone.append(self._scripts.pop(name))
        for entry in gone:
            entry.stop()
        return listed

    def _script(self, name: str) -> Optional[ScriptEntry]:
        self._entries()
        with self._lock:
            return self._scripts.get(name)

    def _row(self, entry: dict, *, probe: bool, timeout: float) -> dict:
        if entry["error"]:
            return _algorithms.row(entry, state="invalid")
        if entry["kind"] == "script":
            with self._lock:
                script = self._scripts.get(entry["name"])
            if script is not None:
                return script.row()
            return _algorithms.row(entry, state="new")
        if not probe:
            return _algorithms.row(entry)
        return _algorithms.row(
            entry, **_algorithms.probe(entry["url"], timeout=timeout)
        )

    def rows(self, *, probe: bool = True, timeout: float = 4.0) -> list[dict]:
        """A row per entry; url entries probed concurrently when *probe*."""
        listed = self._entries()
        if not listed:
            return []
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=min(8, len(listed))) as pool:
            return list(
                pool.map(lambda e: self._row(e, probe=probe, timeout=timeout), listed)
            )

    def refresh(self) -> list[dict]:
        """Install and describe every new or edited script entry, in the
        background; answer the rows."""
        self._entries()
        with self._lock:
            scripts = list(self._scripts.values())
        for script in scripts:
            script.start_install(retry_failed=False)
        return self.rows()

    def _find(self, name: str) -> tuple[Optional[dict], Optional[ScriptEntry]]:
        entry = next((e for e in self._entries() if e["name"] == name), None)
        with self._lock:
            return entry, self._scripts.get(name)

    def ensure(self, name: str, wait: float) -> dict:
        """Bring a script entry up (waiting up to *wait*), or probe a url entry."""
        entry, script = self._find(name)
        if entry is None:
            raise KeyError(name)
        if script is None:
            return self._row(entry, probe=True, timeout=min(max(wait, 1.0), 4.0))
        script.ensure(wait)
        return script.row()

    def _managed(self, name: str) -> ScriptEntry:
        entry, script = self._find(name)
        if entry is None:
            raise KeyError(name)
        if script is None:
            raise ValueError(
                f"{name} is a url entry: its lifecycle and logs belong to whoever runs it"
            )
        return script

    def stop(self, name: str) -> dict:
        script = self._managed(name)
        script.stop()
        return script.row()

    def restart(self, name: str, wait: float) -> dict:
        script = self._managed(name)
        script.stop()
        script.ensure(wait)
        return script.row()

    def logs(self, name: str, lines: int) -> dict:
        return self._managed(name).logs(lines)

    def tick(self) -> None:
        with self._lock:
            scripts = list(self._scripts.values())
        for script in scripts:
            script.tick()

    def stop_all(self) -> None:
        with self._lock:
            scripts = list(self._scripts.values())
        for script in scripts:
            script.stop()
