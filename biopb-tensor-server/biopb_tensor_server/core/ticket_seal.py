"""Seals on read tickets (biopb/biopb#1112).

A reference handed to a process the holder does not control used to carry the
connection's own bearer token: the full-access one, good for every source, write
and action, for as long as it lives. A seal replaces it for the one thing the
reference is for. It is a MAC the server computes over what a plan reads --
which tensor, which version, which window of chunks -- and an expiry, so its
holder may read exactly that and nothing else, with no table to keep and no
cleanup to run.

Nothing here decides who may be issued one: GetFlightInfo does, by authorizing
the read it plans first. Nothing here is an action either: a seal opens
``do_get`` and ``chunk_locate`` for chunks and the ``roi`` read, and no
``do_action``.

The key persists in the state tree, so a restart does not invalidate the lazy
arrays clients hold; replacing it revokes every seal outstanding. A tensor's id
is not enough to bind a seal to: it can be reused (a scratch field name), and
the content version is what tells the tensor that had it before from the one
that has it now. The chunk seal gets that through the identity, which carries
the version header; the ROI seal carries the version itself.
"""

from __future__ import annotations

import hmac
import logging
import secrets
import struct
import time
from hashlib import sha256
from typing import Callable, Optional, Sequence

from biopb.tensor.ticket_pb2 import ChunkGrant, RoiGrant

logger = logging.getLogger(__name__)

__all__ = ["DEFAULT_SEAL_TTL", "TicketSealer", "load_seal_key"]

#: How long a seal lasts, in seconds, unless the server is told otherwise. A
#: distributed job may not fetch its last chunk until hours in, and a lazy array
#: held by a long session outlives a shorter one; a day covers both, and the
#: server's own token still reads without it.
DEFAULT_SEAL_TTL = 24 * 3600.0

_MAC_BYTES = 16
_CHUNK_DOMAIN = b"biopb-chunk-seal-1"
_ROI_DOMAIN = b"biopb-roi-seal-1"


def _frame(*parts: bytes) -> bytes:
    """Length-prefix each part, so no two field splits MAC the same bytes."""
    return b"".join(struct.pack(">I", len(part)) + part for part in parts)


def _uints(values: Sequence[int]) -> bytes:
    return struct.pack(f">{len(values)}I", *values)


class TicketSealer:
    """Mints and checks the seals on one server's tickets."""

    def __init__(
        self,
        key: Optional[bytes] = None,
        ttl: Optional[float] = DEFAULT_SEAL_TTL,
        clock: Callable[[], float] = time.time,
    ) -> None:
        """*key* is the secret; a process that is given none makes a fresh one,
        so its seals die with it. *ttl* is seconds, or None / 0 for seals that
        never expire."""
        if key is not None and len(key) < 16:
            raise ValueError("a seal key must be at least 16 bytes")
        self._key = key if key is not None else secrets.token_bytes(32)
        self._ttl = float(ttl) if ttl else 0.0
        self._clock = clock

    @staticmethod
    def new_key() -> bytes:
        return secrets.token_bytes(32)

    # -- chunks ---------------------------------------------------------------

    def _chunk_mac(
        self, identity: bytes, start: Sequence[int], stop: Sequence[int], expires: int
    ) -> bytes:
        body = _frame(
            _CHUNK_DOMAIN,
            identity,
            _uints(start),
            _uints(stop),
            struct.pack(">Q", expires),
        )
        return hmac.new(self._key, body, sha256).digest()[:_MAC_BYTES]

    def _expiry(self) -> int:
        return int(self._clock() + self._ttl) if self._ttl else 0

    def grant_chunks(
        self, identity: bytes, start: Sequence[int], stop: Sequence[int]
    ) -> ChunkGrant:
        """The grant for the chunks in ``[start, stop)`` of *identity*."""
        expires = self._expiry()
        return ChunkGrant(
            window_start=list(start),
            window_stop=list(stop),
            expires_at=expires,
            seal=self._chunk_mac(identity, start, stop, expires),
        )

    def covers(self, identity: bytes, grant: ChunkGrant, index: Sequence[int]) -> bool:
        """Does *grant* authorize reading chunk *index* of *identity*?

        True only for a seal this key made over exactly these fields, not yet
        expired, whose window holds the index.
        """
        if not grant.seal:
            return False
        start, stop = list(grant.window_start), list(grant.window_stop)
        if not (len(start) == len(stop) == len(index)):
            return False
        expected = self._chunk_mac(identity, start, stop, grant.expires_at)
        if not hmac.compare_digest(expected, grant.seal):
            return False
        if grant.expires_at and self._clock() >= grant.expires_at:
            return False
        return all(lo <= i < hi for lo, i, hi in zip(start, index, stop, strict=True))

    # -- annotations ----------------------------------------------------------

    def _roi_mac(self, array_id: str, version: bytes, expires: int) -> bytes:
        body = _frame(
            _ROI_DOMAIN,
            array_id.encode("utf-8"),
            version,
            struct.pack(">Q", expires),
        )
        return hmac.new(self._key, body, sha256).digest()[:_MAC_BYTES]

    def grant_rois(self, array_id: str, content_version: Optional[bytes]) -> RoiGrant:
        """The grant to read *array_id*'s annotations as of *content_version*."""
        version = content_version or b""
        expires = self._expiry()
        return RoiGrant(
            content_version=version,
            expires_at=expires,
            seal=self._roi_mac(array_id, version, expires),
        )

    def covers_rois(
        self,
        array_id: str,
        grant: RoiGrant,
        current_version: Optional[bytes],
    ) -> bool:
        """Does *grant* authorize reading *array_id*'s annotations now?

        *current_version* is the tensor's version today; a seal made for an
        earlier one is refused, so a reused name does not inherit it.
        """
        if not grant.seal:
            return False
        expected = self._roi_mac(array_id, grant.content_version, grant.expires_at)
        if not hmac.compare_digest(expected, grant.seal):
            return False
        if grant.expires_at and self._clock() >= grant.expires_at:
            return False
        return hmac.compare_digest(grant.content_version, current_version or b"")


#: The owner-only file in the state tree that holds the key.
SEAL_KEY_FILE = "ticket-seal.key"


def load_seal_key() -> bytes:
    """The persisted seal key, made and stored on first use.

    Beside the other credentials in the state tree, readable by its owner only.
    Delete the file to revoke every seal outstanding: the next server to start
    makes a new key. An unreadable or too-short file is treated as absent, since
    keeping a key nobody can verify is worse than rotating it.
    """
    from biopb._security.credentials import read_credential, write_credential

    stored = read_credential(SEAL_KEY_FILE)
    if stored:
        try:
            key = bytes.fromhex(stored)
        except ValueError:
            key = b""
        if len(key) >= 16:
            return key
    key = TicketSealer.new_key()
    try:
        write_credential(key.hex(), SEAL_KEY_FILE)
    except OSError as exc:
        # A state tree that cannot be written (a read-only container) costs the
        # seals their survival across a restart, not the server its start.
        logger.warning(
            "could not persist the ticket seal key (%s); seals issued by this "
            "server stop working when it restarts",
            exc,
        )
    return key
