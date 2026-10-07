"""What a local flight plane serves, published for clients on the same machine.

The plane records the *fingerprint* of the leaf it serves (not the certificate,
which would push clients into the offline ``ca_pem`` mode that skips the
hostname-override probe a loopback dial needs). Keyed by port, since the host of a
local dial carries no information and several planes may share a state tree.

The file is an identity hint, never a credential: the fingerprint is checked
against what the server presents on every connect, so a wrong entry costs a
refused connection, not a trusted impostor.
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

    Never raises.
    """
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (FileNotFoundError, ValueError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def publish(port: int, fingerprint: str) -> None:
    """Record that the plane on *port* serves the leaf with this *fingerprint*.

    Best-effort: on failure clients fall back to the minted-cert path.
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

    ``None`` (no file, no entry, old plane) means the caller falls back.
    """
    entry = _load(tls_served_certs()).get(str(port))
    if not isinstance(entry, dict):
        return None
    value = entry.get("fingerprint")
    return value if isinstance(value, str) and value else None


def retract(port: int) -> None:
    """Drop the entry for *port* on a clean shutdown.

    Avoids leaving a stale claim for the next plane on this port.
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
