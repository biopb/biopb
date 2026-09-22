"""What a member directory carries, whatever format it is in.

A member is one uploaded tensor's store, under ``<write_dir>/fields/``.
There are two layouts -- an OME-Zarr
image group (``adapters.registered.ZarrMember``) and a segment store
(``adapters.cache_member.CacheMember``) -- and this is the vocabulary they
share: the bookkeeping block, the upload marker, and how a directory is read
back as one format or the other.

It sits below both so either can be imported without the other, and so the boot
sweep can ask a directory what it is without knowing which adapter will open it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from biopb_tensor_server.adapters.zarr import (
    read_zattrs,
    upload_expires_at,
    with_upload_state,
)

__all__ = [
    "MEMBER_ATTR",
    "MEMBER_DESCRIPTOR",
    "member_attrs",
    "member_marker",
    "read_member_version",
    "upload_expires_at",
]

#: The ``biopb`` sub-block a member's metadata file carries, beside the upload
#: marker: ``{"member": {"content_version": "<hex>"}}``. A member's token is its
#: own, not its source's, so publishing one never moves a sibling's chunk ids.
MEMBER_ATTR = "member"

#: A ``cache://`` member's metadata file. It is not a zarr group, so it carries
#: what a ``.zattrs`` carries for the other format -- shape, dtype, grid, axes,
#: the token and the upload marker -- under a name that says as much.
MEMBER_DESCRIPTOR = "descriptor.json"


def member_attrs(
    content_version: bytes, state: str, expires_at: Optional[float] = None
) -> dict:
    """The ``biopb`` block a member's metadata file carries.

    The upload marker and the member's own token, in one block: a crash before
    READY leaves a directory the boot sweep recognizes and removes, and the
    token is what namespaces the member's chunk ids against a name reclaimed
    after a discard. Identical for both formats, so the sweep reads one shape.

    *expires_at* records the member's deadline with the member, so it survives
    the process that granted it (``zarr.with_upload_state``).
    """
    return with_upload_state(
        {"biopb": {MEMBER_ATTR: {"content_version": content_version.hex()}}},
        state,
        expires_at,
    )


def read_member_version(attrs: Any) -> Optional[bytes]:
    """A member's persisted ``content_version``, or None if it records none.

    None is not an error the way a missing ``source_id`` is: it means the
    directory is not a member this server minted, and the caller skips it.
    """
    if not isinstance(attrs, dict):
        return None
    block = (attrs.get("biopb") or {}).get(MEMBER_ATTR) or {}
    try:
        return bytes.fromhex(block["content_version"])
    except (KeyError, TypeError, ValueError):
        return None


def member_marker(store: Path) -> Optional[Dict[str, Any]]:
    """The metadata of the member directory *store*, whichever format it is in.

    ``.zattrs`` for a zarr member, ``descriptor.json`` for a cache one. What the
    boot sweep reads to decide whether a directory was left pending, before any
    adapter is asked to open it -- neither format is privileged there, and a
    directory that is neither answers None.
    """
    zattrs = read_zattrs(store)
    if zattrs is not None:
        return zattrs
    try:
        parsed = json.loads((store / MEMBER_DESCRIPTOR).read_text())
    except (OSError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None
