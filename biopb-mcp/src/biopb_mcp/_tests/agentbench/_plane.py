"""A data plane for the run, when a case asks its data to arrive lazily.

Some skills are written for data the session never holds: `client.get_tensor`
hands back a lazy dask array over Flight, and the body is about chunk
boundaries, out-of-core routes and what never has to be materialised. Handing
such a skill an in-memory numpy array on a viewer layer measures a different
route than the one it was written for.

**There is no faked-lazy middle ground.** A `da.from_array` over a local file
with ``client is None`` would be a data environment no production session has,
so the skill would still be measured off its own path. A case is presented
either as ``array`` — in-memory numpy on a viewer layer, which is a real thing
an agent meets — or as ``tensor``, which is the real lazy path with a real
server behind it. Neither is a default and neither is a fallback.

Three properties shape what is here.

**Conditional.** No selected case asks for `tensor`, nothing starts, and the
suite behaves exactly as it did. The import of `biopb_tensor_server` is
likewise deferred and optional: biopb-mcp cannot depend on it (it is never on
PyPI), so a machine without it reports `tensor` cases as unavailable rather
than failing them.

**Run-scoped.** One server for the whole pytest invocation: a case is one
session per sample and an invocation is many cases, and a plane per session
would upload the same volumes again every time — and these are the large
fixtures by construction. It is also the production shape, where a durable
plane outlives the sessions that come and go against it.

**Isolated by construction, not by cleanup.** The plane must be writable — the
skills' own steps upload results (`drift-correction` step 7, `stitch-tiles`
step 7) — so an agent can create sources, and there is *no API to drop one*:
`remove_source` refuses any url that is not `dnd://`, and cache adapters are
expected to accumulate until the server stops. So isolation cannot come from
cleaning up between sessions. It comes from the id:

    source_id = f"cache_{sha256(source_name)[:12]}"       (upload_manager.py)

The id an agent sees is a **one-way hash of a name it is never told**. The
adapter keeps the id and the url (`cache://<source_id>`), never the name, so
the name appears nowhere in a descriptor, a layer or the catalog. A fixture
uploaded under a per-run random name therefore cannot be collided with by
accident and cannot be replaced by an agent that only knows the id.

That is an argument, so it is also checked: :meth:`TensorPlane.fingerprint`
samples a corner of the served array, and `bench/_engine` compares it after
every sample. A changed fingerprint does not fail a test — it flags the row, the same
way `read-harness-internals` does, because `execute_code` is arbitrary Python
and the layer's defence is that nothing can happen *quietly*.
"""

from __future__ import annotations

import atexit
import contextlib
import hashlib
import importlib.util
import json
import os
import secrets
import socket
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

#: How long to wait for the plane to answer `SERVING`. Generous: it scans an
#: (empty) source tree and binds before it is ready, and a slow machine timing
#: out here would report as "no tensor cases ran" rather than as a hang.
BOOT_TIMEOUT_S = 60.0


def plane_unavailable() -> str:
    """Why a `tensor`-presented case cannot run here, or ``""``.

    Answered without spawning anything, so a machine that cannot host a plane
    skips those cases the same way it skips on a missing API key.
    """
    for module, why in (
        ("biopb_tensor_server", "the tensor server is not installed in this env"),
        ("pyarrow", "pyarrow is missing, so nothing can talk Flight"),
    ):
        if importlib.util.find_spec(module) is None:
            return f"{why} ({module})"
    return ""


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@dataclass
class TensorPlane:
    """A writable tensor server of our own, for the length of one run."""

    url: str
    root: Path
    process: subprocess.Popen
    #: Random per run, and never sent anywhere. It is the pre-image of every
    #: fixture id this plane serves, which is what makes those ids unforgeable.
    secret: str = field(default_factory=lambda: secrets.token_hex(8))
    _client: Any = None

    @property
    def client(self):
        if self._client is None:
            from biopb.tensor import TensorFlightClient

            self._client = TensorFlightClient(self.url)
        return self._client

    def upload(
        self,
        key: str,
        array: np.ndarray,
        *,
        chunks: Sequence[int] | None = None,
        dim_labels: Sequence[str] | None = None,
    ) -> str:
        """Put one fixture array on the plane and return its ``array_id``.

        *key* names it only within this run: the name actually sent is salted
        with :attr:`secret`, so the id the agent receives is not derivable from
        anything the agent knows.

        ``chunks`` is explicit rather than left to the uploader's default,
        because where laziness is the point the chunking *is* the thing under
        test — a route that only fails at a chunk boundary is not exercised by
        a single-chunk array.
        """
        import dask.array as da

        array = np.asarray(array)
        chunk_shape = tuple(chunks) if chunks else array.shape
        lazy = da.from_array(array, chunks=chunk_shape)
        return self.client.upload_array(
            lazy,
            f"cache:{self.secret}-{key}",
            chunk_shape=list(chunk_shape),
            dim_labels=list(dim_labels) if dim_labels else None,
        )

    def fingerprint(self, array_id: str) -> str:
        """A hash of a corner of what the plane currently serves for *array_id*.

        One small read rather than a pass over the volume: this is checked once
        per sample, and its job is to notice that the bytes changed at all.
        """
        tensor = self.client.get_tensor(array_id)
        corner = tuple(slice(0, min(4, int(n))) for n in tensor.shape)
        sample = np.ascontiguousarray(np.asarray(tensor[corner]))
        return hashlib.sha256(sample.tobytes()).hexdigest()[:16]

    def stop(self) -> None:
        with contextlib.suppress(Exception):
            if self._client is not None:
                self._client.close()
        self._client = None
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                self.process.kill()
                with contextlib.suppress(subprocess.TimeoutExpired):
                    self.process.wait(timeout=2.0)


def _write_plane_config(root: Path) -> Path:
    """A config tree of the plane's own, so the developer's catalog is neither
    read nor written — the same discipline `_session._write_config` applies to
    the MCP config."""
    (root / "cache").mkdir(parents=True, exist_ok=True)
    (root / "write").mkdir(parents=True, exist_ok=True)
    path = root / "biopb.json"
    path.write_text(
        json.dumps(
            {
                "log_level": "WARNING",
                # No sources: everything this plane serves arrives by upload,
                # which is a supported runtime state (an empty catalog boots).
                "sources": [],
                "writable": True,
                "write_dir": str(root / "write"),
                "cache": {"backend": "file", "file_cache_dir": str(root / "cache")},
            }
        ),
        encoding="utf-8",
    )
    return path


def start_plane() -> TensorPlane:
    """Spawn a writable plane on a free port and wait for it to serve."""
    if why := plane_unavailable():
        raise RuntimeError(why)
    root = Path(tempfile.mkdtemp(prefix="biopb-skill-plane-"))
    config, port = _write_plane_config(root), _free_port()
    log = (root / "plane.log").open("ab", buffering=0)
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "biopb_tensor_server.cli",
            "serve",
            "--config",
            str(config),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            # Required, and not merely convenient: several skills' own steps
            # upload their result, and a read-only plane would fail the step
            # rather than measure it.
            "--writable",
        ],
        stdout=log,
        stderr=log,
        # Not the developer's config tree, and not the developer's plane.
        env={**os.environ, "XDG_CONFIG_HOME": str(root / "config")},
    )
    plane = TensorPlane(url=f"grpc://127.0.0.1:{port}", root=root, process=process)
    try:
        _wait_until_serving(plane, port)
    except Exception:
        plane.stop()
        tail = (root / "plane.log").read_text(errors="replace")[-2000:]
        raise RuntimeError(f"the tensor plane did not come up:\n{tail}") from None
    return plane


def _wait_until_serving(plane: TensorPlane, port: int) -> None:
    """Port up, then `SERVING`.

    Both, because the port binds before the source scan finishes: a plane that
    accepts a connection is not yet one that answers about its catalog, and
    uploading into that window is how a run acquires an intermittent failure
    nobody can reproduce.
    """
    deadline = time.monotonic() + BOOT_TIMEOUT_S
    while time.monotonic() < deadline:
        if plane.process.poll() is not None:
            raise RuntimeError(f"the plane exited with {plane.process.returncode}")
        with contextlib.suppress(OSError):
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                break
        time.sleep(0.25)
    else:
        raise RuntimeError(f"nothing listening on {port} after {BOOT_TIMEOUT_S:g}s")

    while time.monotonic() < deadline:
        with contextlib.suppress(Exception):
            if plane.client.health_check().get("status") == "SERVING":
                return
        time.sleep(0.25)
    raise RuntimeError(f"the plane never reported SERVING within {BOOT_TIMEOUT_S:g}s")


_PLANE: TensorPlane | None = None


def ensure_plane() -> TensorPlane:
    """The run's plane, started on first ask.

    Conditional by being lazy: a run whose selected cases all present `array`
    never calls this, and nothing is spawned.
    """
    global _PLANE
    if _PLANE is None:
        _PLANE = start_plane()
        atexit.register(stop_plane)
    return _PLANE


def running_plane() -> TensorPlane | None:
    """The plane if one was started, without starting one."""
    return _PLANE


def stop_plane() -> None:
    """Stop the run's plane, if there is one.

    Cleanup is the process and its temp tree, not a per-session sweep: there is no
    API to drop a `cache://` source, so what an agent uploaded lives until the
    plane stops. That is affordable because it is bounded by the chunk cache,
    and safe because a fixture's id cannot be reached from anything the agent
    was given.
    """
    global _PLANE
    if _PLANE is not None:
        _PLANE.stop()
        _PLANE = None
