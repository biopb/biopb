"""Control / control-plane endpoint location — shared, stdlib-only.

The control (control plane) exposes a small loopback HTTP control API. Two
independent processes need to agree on where it listens:

- the control itself (``biopb-control``, a separate workspace package), and
- its clients (top-level ``biopb`` attributes, backed by this private
  :mod:`biopb._control` package), which ask it where the data plane is and
  to bring it up.

A client cannot import ``biopb-control``, so the endpoint lives here in the
dependency-light core ``biopb`` SDK. Kept stdlib-only so importing it never
drags in the heavy server/mcp stacks.
"""

import json
import os
import tempfile

# Base-port convention
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
        from .._locations import control_runtime_file

        with open(control_runtime_file(), encoding="utf-8") as fh:
            rec = json.load(fh)
        return rec if isinstance(rec, dict) else {}
    except (OSError, ValueError, ImportError, RuntimeError):
        return {}


def write_runtime_record(
    host: str,
    port: int,
    pid: int,
    user_url: str | None = None,
) -> None:
    """Publish the endpoint a control just bound. Best-effort.

    ``user_url`` is how the user's browser reaches it when that is not the bound
    address (behind a reverse proxy); written only when set. See
    :func:`user_base_url`.

    ``pid`` lets ``biopb control status`` tell a live foreground control from a
    record a crashed one left behind -- the distinction the pid file draws for
    daemons and cannot draw for ``control run``. It is stamped with the same
    create-time token the pid file carries, because a pid alone cannot make that
    distinction: only a crash strands a record (a clean stop retracts it), and a
    recycled pid would then read as a control that is still up. ``None`` when the
    platform has no cheap create-time -- readers degrade to liveness there, as
    they do for a legacy bare-pid file.
    """
    from .._locations import control_runtime_file
    from ..lifecycle.proc import process_create_time

    path = control_runtime_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "host": host,
        "port": port,
        "pid": pid,
        "create_time": process_create_time(pid),
    }
    if user_url:
        record["user_url"] = user_url
    payload = json.dumps(record)
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
        from .._locations import control_runtime_file

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


def connect_url(host: str, port: int, scheme: str = "http") -> str:
    """A base URL a client on this machine can connect to.

    A server bound to a wildcard (``0.0.0.0``, ``::``) is dialed over loopback:
    the address it is *bound* to is not one a client can dial (0.0.0.0 is not
    dialable on Windows). An explicit host is used as given, and an IPv6 literal
    is bracketed so the ``:port`` suffix stays unambiguous.
    """
    host = {"0.0.0.0": "127.0.0.1", "::": "::1", "": "127.0.0.1"}.get(host, host)
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    return f"{scheme}://{host}:{port}"


def control_base_url() -> str:
    """The control-API base URL a client on this machine connects to, e.g.
    ``http://127.0.0.1:8813``."""
    return connect_url(control_host(), control_port())


def user_base_url() -> str:
    """Where the *user's browser* reaches the control, for a link handed to them.

    :func:`control_base_url` is where this machine connects, and behind a reverse
    proxy (an Open OnDemand ``/node/<host>/<port>`` route) that is a loopback
    address the user's browser cannot reach. A control told its public form
    (``--url-prefix``, ``--public-origin``) publishes it in its record: origin plus
    prefix, or the bare path when only the prefix is set (a path on whatever site
    the user opened the session from). Otherwise the two are the same.
    """
    published = _runtime_record().get("user_url")
    if isinstance(published, str) and published.strip():
        return published.strip().rstrip("/")
    return control_base_url()
