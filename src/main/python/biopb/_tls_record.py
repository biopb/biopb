"""What a local flight plane serves, published for clients on the same machine.

A client on this machine should not have to guess the plane's certificate. It
used to read ``state/biopb/tls/server-cert.pem`` — the pair the plane *mints* —
which is right only when the plane minted one: an operator's ``--tls-cert``
never lands there, so the anchor was missing (or, worse, a stale minted cert
from an earlier run, which fails the handshake instead of failing to open).

So the plane says what it serves, and says it by **fingerprint**. Not by writing
the certificate itself, which would tempt a client into the offline ``ca_pem``
mode that skips the hostname-override probe — and a local client dials loopback,
so a certificate naming only the host's public name needs that probe or every
connection fails hostname verification (biopb/biopb#916).

Keyed by port: this record is consulted only for a local dial, where the host is
one of ``localhost`` / ``127.0.0.1`` / ``::1`` and carries no information, while
the port is what distinguishes two planes sharing a state tree. Nothing
guarantees there is only one — the uid-scoped cache lock refuses a second plane
in the default configuration, but the cache is optional and a plane with
``backend: memory`` takes no lock at all.

Trust boundary: this file is inside the user's own state tree, so writing it
requires already being that user. It is a *hint about identity*, never a
credential, and a wrong entry costs a refused connection rather than a trusted
impostor — the fingerprint is checked against what the server actually presents
on every connect.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from typing import Dict, Optional

from biopb._locations import tls_served_certs

logger = logging.getLogger(__name__)


def _load(path) -> Dict[str, dict]:
    """Read the record; an absent, unreadable or malformed file is empty.

    Never raises: a client that cannot read this falls back to the older path,
    and a plane that cannot read it simply overwrites its own entry.
    """
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (FileNotFoundError, ValueError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def publish(port: int, fingerprint: str) -> None:
    """Record that the plane on *port* serves the leaf with this *fingerprint*.

    Best-effort: a plane that cannot write this still serves fine, and the
    clients that would have read it fall back to the minted-cert path. Failing
    startup over an advisory hint would trade a working deployment for a broken
    one.
    """
    path = tls_served_certs()
    entry = {"fingerprint": fingerprint, "pid": os.getpid(), "updated_at": time.time()}
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        records = _load(path)
        records[str(port)] = entry
        # Atomic replace so a concurrent reader never sees a half-written file.
        fd, tmp = tempfile.mkstemp(
            dir=path.parent, prefix=".tls-served-", suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(records, f, indent=2, sort_keys=True)
            os.replace(tmp, path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
    except OSError as e:
        logger.warning(
            "could not publish the served TLS certificate for port %s (%s); "
            "clients on this machine will fall back to the minted certificate",
            port,
            e,
        )


def lookup(port: int) -> Optional[str]:
    """The fingerprint the plane on *port* published, or ``None`` if it did not.

    ``None`` covers every uninteresting case — no file, no entry for this port,
    a plane too old to publish one — and each means the same thing to a caller:
    fall back, do not fail.
    """
    entry = _load(tls_served_certs()).get(str(port))
    if not isinstance(entry, dict):
        return None
    value = entry.get("fingerprint")
    return value if isinstance(value, str) and value else None


def retract(port: int) -> None:
    """Drop the entry for *port* on a clean shutdown.

    A stale entry is not dangerous — a client that trusts it simply fails to
    verify the next plane, loudly — but leaving one behind means the next plane
    on this port inherits a claim it did not make.
    """
    path = tls_served_certs()
    records = _load(path)
    if records.pop(str(port), None) is None:
        return
    try:
        fd, tmp = tempfile.mkstemp(
            dir=path.parent, prefix=".tls-served-", suffix=".tmp"
        )
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, sort_keys=True)
        os.replace(tmp, path)
    except OSError:
        logger.debug("could not retract the served TLS record for port %s", port)
