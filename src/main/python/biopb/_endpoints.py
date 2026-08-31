"""Control / control-plane endpoint location — shared, stdlib-only.

The control (control plane) exposes a small loopback HTTP control API. Two
independent processes need to agree on where it listens:

- the control itself (``biopb-control``, a separate workspace package), and
- ``biopb-mcp``'s ``_connection``, which asks the control to ensure the data
  plane is up instead of shelling out ``biopb server start`` itself.

Neither can import the other (``biopb-mcp`` cannot import ``biopb-control`` any
more than it can import ``biopb-tensor-server`` — see the
"shared config lives in core biopb SDK" rationale), so the one thing they must
share — the endpoint — lives here in the dependency-light core ``biopb`` SDK,
next to ``_locations`` / ``_config_constraints``. Kept stdlib-only so
importing it never drags in the heavy server/mcp stacks.

This is the anchor of the single-origin web front (see
``biopb-mcp/ARCHITECTURE.md``): the control serves a
Starlette/uvicorn app on this port that serves the built ``web/`` SPA bundle at
its root (dashboard ``/``, dataviewer ``/viewer``, per-session observe
``/session/<id>/observe`` — all React routes of one SPA), answers its own control
API (bare ``/health``; control verbs under ``/api/*``, e.g.
``/api/data_plane/ensure``), and reverse-proxies the supervised tensor server's
HTTP sidecar under a ``/data_plane/*`` namespace (data API at
``/data_plane/api/*``). Each plane owns a path prefix
so the ``/api/*`` namespaces never collide; per-session ``/session/<id>/api/*``
proxies to the session child on the same origin.
"""

import json
import os
import tempfile

# --- the base-port convention --------------------------------------------- #
#
# One number places all three listeners, so a whole deployment moves together --
# which is what lets two users run side-by-side controls on one host (each also
# needs its own ``BIOPB_STATE_HOME``, since the pid / credential / runtime
# records are per-state-dir).
#
# The base and its offsets are **the container's** (``entrypoint.sh``:
# ``BIOPB_BASE_PORT`` default 8810, sidecar = base+4, gRPC = base+5), extended
# with the control at base+3. Deliberately not a second convention: two
# base-port schemes that agree at their defaults and diverge the moment either
# base moves would be indistinguishable in a bug report.
BASE_DEFAULT_PORT = 8810
CONTROL_PORT_OFFSET = 3
SIDECAR_PORT_OFFSET = 4
FLIGHT_PORT_OFFSET = 5


def control_port_for(base_port: int) -> int:
    """The control / browser-origin port for a base (default base -> 8813)."""
    return base_port + CONTROL_PORT_OFFSET


def sidecar_port_for(base_port: int) -> int:
    """The tensor HTTP sidecar port for a base (default base -> 8814)."""
    return base_port + SIDECAR_PORT_OFFSET


def flight_port_for(base_port: int) -> int:
    """The flight gRPC data-plane port for a base (default base -> 8815)."""
    return base_port + FLIGHT_PORT_OFFSET


# Loopback control API. Distinct from the other biopb ports so all four can run
# at once on one host: tensor-server web 8814 / gRPC 8815, MCP /mcp 8765.
CONTROL_DEFAULT_HOST = "127.0.0.1"
CONTROL_DEFAULT_PORT = control_port_for(BASE_DEFAULT_PORT)  # 8813

# --- the runtime (discovery) record --------------------------------------- #
#
# The control's port stopped being a constant when `--base-port` arrived: a
# deployment that moves the base moves the control with it, and a client that
# only knew 8813 would look in the wrong place. So a serving control publishes
# the endpoint it actually bound, and clients read it here.
#
# Precedence for every reader is `BIOPB_CONTROL_*` -> this record -> the 8813
# default. The env var stays on top so an explicit override still wins over a
# discovered value, and the static default remains the answer when nothing is
# running (a probe against it then simply fails to connect, exactly as before).
#
# Written by ``biopb_control._run`` after its bind succeeds and removed on a
# clean stop. A crash leaves it behind, so it is a *hint*, never proof that a
# control is alive -- every consumer already probes ``/health`` or connects.


def _runtime_record() -> dict:
    """The serving control's published endpoint, or ``{}`` if none/unreadable.

    Deliberately forgiving: a missing, truncated, or malformed file means "no
    record", never an exception. This sits under ``control_port()``, which every
    client calls before it can even reach the control -- a hard failure here
    would break discovery entirely rather than degrade to the default.

    ``RuntimeError`` is in the net because locating the file is itself fallible:
    ``Path.home()`` raises it on Windows when the environment carries no
    ``USERPROFILE``/``HOMEPATH`` (a scrubbed-env service, or a test that clears
    ``os.environ``). Nowhere to look is just another way of having no record.
    """
    try:
        from ._locations import control_runtime_file

        with open(control_runtime_file(), encoding="utf-8") as fh:
            rec = json.load(fh)
        return rec if isinstance(rec, dict) else {}
    except (OSError, ValueError, ImportError, RuntimeError):
        return {}


def write_runtime_record(host: str, port: int, pid: int) -> None:
    """Publish the endpoint a control just bound. Best-effort.

    ``pid`` lets ``biopb control status`` tell a live foreground control from a
    record a crashed one left behind -- the distinction the pid file draws for
    daemons and cannot draw for ``control run``. It is stamped with the same
    create-time token the pid file carries, because a pid alone cannot make that
    distinction: only a crash strands a record (a clean stop retracts it), and a
    recycled pid would then read as a control that is still up. ``None`` when the
    platform has no cheap create-time -- readers degrade to liveness there, as
    they do for a legacy bare-pid file.
    """
    from ._lifecycle.proc import process_create_time
    from ._locations import control_runtime_file

    path = control_runtime_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        {
            "host": host,
            "port": port,
            "pid": pid,
            "create_time": process_create_time(pid),
        }
    )
    # Written exactly as the pid file is (daemon.write_pid_file): a *unique*
    # sibling temp, then os.replace. Unique because the temp name is what makes
    # concurrent writers safe -- a fixed one has two publishers truncating and
    # writing the same file before either renames, so the record that lands can
    # be a mix of both. Racing publishers are reachable here: only `control
    # start` takes the start lock, so a foreground `control run` sharing the
    # state dir is not serialized against it. And the temp is unlinked on any
    # failure, so a full disk leaves no debris beside the record it could not
    # replace.
    fd, tmp = tempfile.mkstemp(
        prefix=f".{path.name}-", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(payload)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def remove_runtime_record() -> None:
    """Retract the published endpoint on a clean stop. Best-effort."""
    try:
        from ._locations import control_runtime_file

        control_runtime_file().unlink()
    except (OSError, ImportError, RuntimeError):
        pass


def read_runtime_record() -> dict:
    """The published endpoint record (``{}`` when absent). See :func:`_runtime_record`."""
    return _runtime_record()


def control_host() -> str:
    """The control-API bind/connect host.

    ``BIOPB_CONTROL_HOST`` -> the serving control's published record -> 127.0.0.1.
    """
    env = os.environ.get("BIOPB_CONTROL_HOST")
    if env:
        return env
    host = _runtime_record().get("host")
    return host if isinstance(host, str) and host else CONTROL_DEFAULT_HOST


def control_port() -> int:
    """The control-API port.

    ``BIOPB_CONTROL_PORT`` -> the serving control's published record -> 8813.

    A malformed override or record falls back to the next source rather than
    raising, so a stray value can never wedge a client that only wants to probe
    the control.
    """
    raw = os.environ.get("BIOPB_CONTROL_PORT")
    if raw:
        try:
            return int(raw)
        except ValueError:
            return CONTROL_DEFAULT_PORT
    port = _runtime_record().get("port")
    if isinstance(port, int):
        return port
    return CONTROL_DEFAULT_PORT


def control_base_url() -> str:
    """The control-API base URL, e.g. ``http://127.0.0.1:8813``."""
    return f"http://{control_host()}:{control_port()}"
