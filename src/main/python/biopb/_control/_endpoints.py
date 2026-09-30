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
    url_prefix: str | None = None,
    public_origin: str | None = None,
) -> None:
    """Publish the endpoint a control just bound. Best-effort.

    ``url_prefix`` and ``public_origin`` say how the *user's browser* reaches this
    control when that is not the address it bound: the path a reverse proxy
    publishes it under, and the origin (scheme, host, port) the proxy answers on.
    They are written only when set, so a plain local control's record is what it
    always was. See :func:`user_base_url`.

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
    if url_prefix:
        record["url_prefix"] = url_prefix
    if public_origin:
        record["public_origin"] = public_origin
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


# A control bound to a wildcard address is published under that address, which
# names where it listens, not somewhere a client can connect to (0.0.0.0 is not
# dialable on Windows, and reads as a mistake everywhere).
_WILDCARD_CONNECT = {"0.0.0.0": "127.0.0.1", "::": "::1", "[::]": "::1"}


def control_base_url() -> str:
    """The control-API base URL a client on this machine connects to, e.g.
    ``http://127.0.0.1:8813``. A wildcard bind is dialed over loopback."""
    host = control_host()
    host = _WILDCARD_CONNECT.get(host, host)
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    return f"http://{host}:{control_port()}"


def _published(env_name: str, record_key: str) -> str:
    """A published string, ``$env`` outranking the serving control's record."""
    value = os.environ.get(env_name) or _runtime_record().get(record_key)
    return value.strip() if isinstance(value, str) else ""


def public_url_prefix() -> str:
    """The path prefix the user's browser reaches the control under, or ``""``.

    ``BIOPB_URL_PREFIX`` (what the control itself reads), then the serving
    control's record. Canonicalized to ``/a/b``: the control validated it before
    publishing, so this only tidies slashes.
    """
    segments = [s for s in _published("BIOPB_URL_PREFIX", "url_prefix").split("/") if s]
    return "/" + "/".join(segments) if segments else ""


def public_origin() -> str:
    """The origin (``https://host[:port]``) the user's browser reaches the
    control on, or ``""``. ``BIOPB_PUBLIC_ORIGIN``, then the record."""
    return _published("BIOPB_PUBLIC_ORIGIN", "public_origin").rstrip("/")


def user_base_url() -> str:
    """Where the *user's browser* reaches the control, for a link handed to them.

    :func:`control_base_url` is where this machine connects, and behind a reverse
    proxy (an Open OnDemand ``/node/<host>/<port>`` route) it is a loopback
    address the user's browser cannot reach. The control knows the public form
    because it was told it, so it publishes it: origin plus prefix when both are
    set, the bare path when only the prefix is (a path on whatever origin the user
    reached the portal at), and the connect URL when neither is.
    """
    prefix, origin = public_url_prefix(), public_origin()
    if prefix or origin:
        return f"{origin}{prefix}"
    return control_base_url()
