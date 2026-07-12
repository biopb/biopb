"""Inspect the algorithm plane — the configured ``biopb.image`` ProcessImage servers.

The "algorithm plane" is the set of gRPC ``ProcessImage`` servicers an agent's
kernel exposes as callable ``ops``. Their URLs live in the biopb-mcp config today
(``mcp.services.process_image_servers``); each is queried via ``GetOpNames`` so its
ops surface in the kernel. Moving that config under the control is migration
Layer 4 (``biopb-mcp/docs/mcp-dedaemonization-migration.md`` §7), deliberately
*later*.

This module is a **read-only inspector** for the control dashboard: it lists the
configured servers and probes each for liveness + advertised ops. It does **not**
control their lifecycle (start/stop is Layer 4, out of scope) and it never writes
the config. Mirrors ``biopb._agents`` in spirit — a small importable API the lean
control plane can call.

Two layers, on purpose:

- :func:`configured` reads the server URLs straight out of the biopb-mcp config
  file with the stdlib ``json`` module. It reproduces the config *location* and
  *key* rather than importing ``biopb_mcp`` — the control plane must not import the
  mcp package (invariant I2), and this keeps the list cheap and dependency-free.
- :func:`probe` / :func:`statuses` open a gRPC channel and call ``GetOpNames``.
  gRPC and the ``biopb.image`` stubs are part of the base ``biopb`` SDK (already a
  control dependency), so probing stays within the lean-control budget — it drags
  in no napari / dask / Qt. The ``grpc`` import is lazy so a bare
  :func:`configured` never pays for it.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# Default per-probe deadline (seconds). The dashboard probes on demand (a button),
# so a few seconds is fine; kept short so a dead server doesn't stall its row for
# long. Probes run concurrently (see statuses), so this bounds the whole sweep too.
_DEFAULT_TIMEOUT = 4.0


# --------------------------------------------------------------------------- #
# Reading the configured server list (stdlib only)
# --------------------------------------------------------------------------- #


def _config_file() -> Path:
    """The biopb-mcp config file location (``~/.config/biopb-mcp/config.json``).

    Reproduces ``biopb_mcp._config.get_config_path()`` — the same on every platform
    — rather than importing ``biopb_mcp``: the control plane must not import the mcp
    package (invariant I2), and the read is plain JSON. Resolved at call time (not
    cached) so a test that repoints ``Path.home()`` gets an isolated location.
    """
    return Path.home() / ".config" / "biopb-mcp" / "config.json"


def servers_from_config(config) -> list[str]:
    """Extract the ProcessImage server URLs from an already-loaded config mapping.

    The single normalization point for the list, shared by :func:`configured` (the
    control plane, which reads the config file fresh) and the biopb-mcp kernel,
    which passes its live ``CONFIG`` dict here instead of hand-reading the key — so
    "where the algorithm-plane server list lives" is defined in exactly one place.

    Reads ``mcp.services.process_image_servers`` (the nested location), falling back
    to the legacy flat ``mcp.process_image_servers`` the installer once seeded — the
    same precedence ``biopb_mcp._config._migrate_legacy_keys`` applies (nested
    wins). Any non-mapping / missing / oddly-typed shape reads as an empty list;
    non-string and blank entries are dropped. Never raises.
    """
    if not isinstance(config, dict):
        return []
    mcp = config.get("mcp")
    if not isinstance(mcp, dict):
        return []
    services = mcp.get("services")
    if isinstance(services, dict) and "process_image_servers" in services:
        servers = services.get("process_image_servers")
    else:  # legacy flat location (pre-nesting installer / hand-edited config)
        servers = mcp.get("process_image_servers")
    if not isinstance(servers, list):
        return []
    return [s for s in servers if isinstance(s, str) and s.strip()]


def configured() -> list[str]:
    """The configured ProcessImage server URLs, in config order.

    Reads the biopb-mcp config file from disk (stdlib JSON) and normalizes it via
    :func:`servers_from_config`. A missing / malformed / oddly-typed config reads as
    an empty list and never raises: this feeds a status display, not a write.
    """
    path = _config_file()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return servers_from_config(data)


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


def _result(state: str, *, ops=None, op_count=None, error=None, single_op=False):
    """A uniform probe result dict so every branch returns the same shape."""
    ops = list(ops or [])
    return {
        "state": state,
        "ops": ops,
        "op_count": op_count if op_count is not None else len(ops),
        "error": error,
        "single_op": single_op,
    }


def probe(url: str, *, timeout: float = _DEFAULT_TIMEOUT) -> dict:
    """Open a gRPC channel to ``url`` and ask it for its ops.

    Returns ``{state, ops, op_count, error, single_op}``:

    - ``state="serving"`` — ``GetOpNames`` answered (``ops`` lists the names), or the
      server is up but implements no ``GetOpNames`` (a single-op server:
      ``single_op=True``, ``ops=[]``, ``op_count=1``).
    - ``state="unreachable"`` — the channel/RPC could not connect (server down, bad
      host, TLS mismatch, deadline).
    - ``state="error"`` — the server answered but with an unexpected RPC error.
    - ``state="invalid"`` — the URL is not a usable ``grpc://`` / ``grpcs://`` target.
    - ``state="unknown"`` — gRPC support is unexpectedly unavailable in this env.

    Never raises: every failure is folded into the returned dict so one bad server
    never breaks the sweep or the dashboard.
    """
    try:
        # Lazy: grpc + the image stubs are hard biopb deps, but importing them only
        # when a probe actually runs keeps `configured()` (the plain config read)
        # import-light and lets it work even in a stripped env.
        import grpc
        from google.protobuf import empty_pb2

        import biopb.image as proto
    except ImportError as exc:  # pragma: no cover - grpc is a base biopb dependency
        return _result("unknown", error=f"gRPC support unavailable: {exc}")

    scheme = _scheme(url)
    parsed = urlparse(url)
    target = parsed.netloc or parsed.path
    if not target or scheme not in ("grpc", "grpcs"):
        return _result(
            "invalid", error="URL must be grpc://host:port or grpcs://host:port"
        )

    if scheme == "grpcs":
        channel = grpc.secure_channel(target, grpc.ssl_channel_credentials())
    else:
        channel = grpc.insecure_channel(target)
    try:
        stub = proto.ProcessImageStub(channel)
        try:
            response = stub.GetOpNames(empty_pb2.Empty(), timeout=timeout)
        except grpc.RpcError as exc:
            code = exc.code()
            if code == grpc.StatusCode.UNIMPLEMENTED:
                # A single-op server (no GetOpNames) is still serving one nameless
                # op — the same case biopb_mcp._process_ops treats as single-op mode.
                return _result("serving", op_count=1, single_op=True)
            if code in (
                grpc.StatusCode.UNAVAILABLE,
                grpc.StatusCode.DEADLINE_EXCEEDED,
            ):
                return _result("unreachable", error=_rpc_message(exc))
            return _result("error", error=_rpc_message(exc))
        ops = [name for name in response.names if name]
        return _result("serving", ops=ops)
    finally:
        channel.close()


# --------------------------------------------------------------------------- #
# Per-server status + the full sweep
# --------------------------------------------------------------------------- #


def status(url: str, *, timeout: float = _DEFAULT_TIMEOUT) -> dict:
    """One server's row: its identity (``url`` / ``target`` / ``scheme``) plus the
    :func:`probe` result (``state`` / ``ops`` / ``op_count`` / ``error`` /
    ``single_op``)."""
    return {
        "url": url,
        "target": _target(url),
        "scheme": _scheme(url) or "grpc",
        **probe(url, timeout=timeout),
    }


def statuses(*, timeout: float = _DEFAULT_TIMEOUT) -> list[dict]:
    """Status for every configured server, in config order.

    Probes run concurrently — each blocks up to ``timeout`` on a dead server — so
    the sweep is bounded by the slowest single probe, not their sum. Called from the
    control dashboard's threadpool handler.
    """
    urls = configured()
    if not urls:
        return []
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=min(8, len(urls))) as pool:
        # map preserves input order, so rows stay in config order.
        return list(pool.map(lambda u: status(u, timeout=timeout), urls))
