"""The algorithm registry, and a probe of a server of the ``Ops`` protocol.

The registry is a directory, ``~/.config/biopb/algorithms/``. Each file is one
entry, named by its stem; a stem starting with ``_`` is skipped:

- ``<name>.py``, a script entry: a server file the control runs with uv.
- ``<name>.json``, ``{"url": "grpc://host:port"}``: a server someone else runs.
  The control probes it and passes it along; it never starts or stops it.

Reading the registry is stdlib only, so the lean control and ``biopb.control``
can call it; the gRPC probe imports gRPC on first use.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

from biopb import _locations

logger = logging.getLogger(__name__)

# Default per-probe deadline (seconds). Kept short so a dead server does not
# stall its row; statuses() probes concurrently, so this bounds the sweep too.
_DEFAULT_TIMEOUT = 4.0

_UNSAFE_IN_A_NAME = re.compile(r"[^A-Za-z0-9_-]+")


# --------------------------------------------------------------------------- #
# The registry (stdlib only)
# --------------------------------------------------------------------------- #


def registry_dir() -> Path:
    """The registry directory; see :func:`biopb._locations.algorithms_dir`."""
    return _locations.algorithms_dir()


def _url_entry(name: str, path: Path) -> dict:
    entry = {"name": name, "kind": "url", "path": str(path), "url": None, "error": None}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        entry["error"] = f"unreadable: {exc}"
        return entry
    url = data.get("url") if isinstance(data, dict) else None
    if not isinstance(url, str) or not url.strip():
        entry["error"] = 'expected {"url": "grpc://host:port"}'
    else:
        entry["url"] = url.strip()
    return entry


def entries(directory: Optional[Path] = None) -> list[dict]:
    """The registry's entries, by name: ``{name, kind, path, url, error}``.

    ``kind`` is ``"script"`` or ``"url"``. An entry that cannot be used (a
    ``.json`` without a url, or a name taken by both a ``.py`` and a ``.json``)
    carries an ``error`` rather than being dropped, so it shows up to be fixed.
    A missing directory is an empty registry. Never raises.
    """
    directory = directory or registry_dir()
    try:
        paths = sorted(p for p in directory.iterdir() if p.is_file())
    except OSError:
        return []
    by_name: dict[str, dict] = {}
    for path in paths:
        if path.name.startswith("_") or path.suffix not in (".py", ".json"):
            continue
        name = path.stem
        if path.suffix == ".py":
            entry = {
                "name": name,
                "kind": "script",
                "path": str(path),
                "url": None,
                "error": None,
            }
        else:
            entry = _url_entry(name, path)
        if name in by_name:
            by_name[name]["error"] = f"both {name}.py and {name}.json exist"
            continue
        by_name[name] = entry
    return list(by_name.values())


def configured() -> list[dict]:
    """The registry's entries; see :func:`entries`."""
    return entries()


def servers_from_config(config) -> list[str]:
    """The server URLs a biopb-mcp config listed, for the migration.

    Any odd shape reads as an empty list; non-string and blank entries are
    dropped. Never raises.
    """
    if not isinstance(config, dict):
        return []
    services = config.get("services")
    if not isinstance(services, dict):
        return []
    servers = services.get("process_image_servers")
    if not isinstance(servers, list):
        return []
    return [s for s in servers if isinstance(s, str) and s.strip()]


def _name_for(url: str) -> str:
    """An entry name for a server URL: its host and port."""
    try:
        target = urlparse(url).netloc or url
    except ValueError:
        target = url
    return _UNSAFE_IN_A_NAME.sub("-", target).strip("-") or "server"


def migrate_from_mcp_config(directory: Optional[Path] = None) -> list[str]:
    """Move the mcp config's server URLs into url entries; answer the names.

    Runs once: only while the registry directory does not exist, and it creates
    the directory whether or not there was anything to write. The key leaves
    the mcp config, which no longer reads it.
    """
    directory = directory or registry_dir()
    if directory.exists():
        return []
    mcp_config = _locations.mcp_config_path()
    try:
        config = json.loads(mcp_config.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        config = None
    directory.mkdir(parents=True, exist_ok=True)
    written = []
    for url in servers_from_config(config):
        name = _name_for(url)
        while name in written:
            name += "-1"
        (directory / f"{name}.json").write_text(
            json.dumps({"url": url}) + "\n", encoding="utf-8"
        )
        written.append(name)
    services = config.get("services") if isinstance(config, dict) else None
    if isinstance(services, dict) and "process_image_servers" in services:
        del services["process_image_servers"]
        tmp = mcp_config.with_name(mcp_config.name + ".tmp")
        tmp.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        tmp.replace(mcp_config)
    if written:
        logger.info("algorithm registry: migrated %s from the mcp config", written)
    return written


# --------------------------------------------------------------------------- #
# Probing a server (lazy gRPC)
# --------------------------------------------------------------------------- #


def _target(url: str) -> str:
    """A ``host:port`` label for display; the raw URL if it will not parse."""
    try:
        parsed = urlparse(url)
        return parsed.netloc or parsed.path or url
    except ValueError:
        return url


def _scheme(url: str) -> str:
    """The URL scheme lowercased (``grpc`` / ``grpcs``); empty when absent."""
    try:
        return urlparse(url).scheme.lower()
    except ValueError:
        return ""


def _rpc_message(exc) -> str:
    """A compact one-line message from a ``grpc.RpcError`` for display."""
    try:
        code = exc.code().name
    except Exception:  # noqa: BLE001 - best-effort formatting, never re-raise
        code = "RPC_ERROR"
    try:
        detail = (exc.details() or "").strip()
    except Exception:  # noqa: BLE001
        detail = ""
    detail = detail.splitlines()[0] if detail else ""
    return f"{code}: {detail}" if detail else code


def _result(state: str, *, ops=None, fingerprint="", error=None) -> dict:
    return {
        "state": state,
        "ops": list(ops or []),
        "fingerprint": fingerprint,
        "error": error,
    }


def probe(
    url: str, *, token: Optional[str] = None, timeout: float = _DEFAULT_TIMEOUT
) -> dict:
    """Ask the server at ``url`` for its ops (``Ops.Describe``).

    Returns ``{state, ops, fingerprint, error}``, where ``ops`` are the
    ``OpInfo`` messages as JSON dicts:

    - ``up``: it answered.
    - ``unreachable``: nothing answered in time (down, bad host, TLS mismatch).
    - ``error``: it answered with an error, including a server that does not
      implement ``Ops`` (such as one of the retired ``ProcessImage`` protocol).
    - ``invalid``: the URL is not ``grpc://`` or ``grpcs://``.

    Never raises.
    """
    try:
        import grpc
        from google.protobuf import empty_pb2, json_format

        import biopb.image as proto
    except ImportError as exc:  # pragma: no cover - grpc is a base dependency
        return _result("error", error=f"gRPC support unavailable: {exc}")

    try:
        parsed = urlparse(url)
        scheme = (parsed.scheme or "").lower()
        target = parsed.netloc or parsed.path
    except ValueError as exc:
        return _result("invalid", error=f"unparseable URL: {exc}")
    if not target or scheme not in ("grpc", "grpcs"):
        return _result(
            "invalid", error="URL must be grpc://host:port or grpcs://host:port"
        )

    channel = None
    try:
        if scheme == "grpcs":
            channel = grpc.secure_channel(target, grpc.ssl_channel_credentials())
        else:
            channel = grpc.insecure_channel(target)
        metadata = [("authorization", f"Bearer {token}")] if token else None
        try:
            answer = proto.OpsStub(channel).Describe(
                empty_pb2.Empty(), timeout=timeout, metadata=metadata
            )
        except grpc.RpcError as exc:
            code = exc.code()
            if code == grpc.StatusCode.UNIMPLEMENTED:
                return _result(
                    "error",
                    error="does not implement Ops (a ProcessImage server must move to Ops)",
                )
            if code in (grpc.StatusCode.UNAVAILABLE, grpc.StatusCode.DEADLINE_EXCEEDED):
                return _result("unreachable", error=_rpc_message(exc))
            return _result("error", error=_rpc_message(exc))
        listing = json_format.MessageToDict(answer)
        return _result("up", ops=listing.get("ops", []), fingerprint=answer.fingerprint)
    except Exception as exc:  # noqa: BLE001 - probe() never raises; fold into a row
        return _result("error", error=str(exc))
    finally:
        if channel is not None:
            channel.close()


# --------------------------------------------------------------------------- #
# Rows
# --------------------------------------------------------------------------- #


def row(entry: dict, **state) -> dict:
    """One entry's row, as the control and ``biopb image servers`` list it:
    ``{name, kind, url, target, scheme, state, ops, op_count, fingerprint,
    error}``. *state* supplies or overrides the live fields."""
    url = state.pop("url", None) or entry.get("url") or ""
    out = {
        "name": entry["name"],
        "kind": entry["kind"],
        "url": url or None,
        "target": _target(url) if url else entry["name"],
        "scheme": _scheme(url) or "grpc",
        "state": "unknown",
        "ops": [],
        "fingerprint": "",
        "error": entry.get("error"),
    }
    out.update(state)
    out["op_count"] = len(out["ops"])
    return out


def status(entry: dict, *, timeout: float = _DEFAULT_TIMEOUT) -> dict:
    """A url entry's row with a live probe; any other entry as listed."""
    if entry["kind"] != "url" or entry.get("error"):
        return row(entry, state="invalid" if entry.get("error") else "unknown")
    return row(entry, **probe(entry["url"], timeout=timeout))


def statuses(*, timeout: float = _DEFAULT_TIMEOUT) -> list[dict]:
    """A row for every registry entry, url entries probed concurrently.

    A script entry reads ``unknown`` here: its state is the control's, which
    runs it.
    """
    listed = entries()
    if not listed:
        return []
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=min(8, len(listed))) as pool:
        return list(pool.map(lambda e: status(e, timeout=timeout), listed))
