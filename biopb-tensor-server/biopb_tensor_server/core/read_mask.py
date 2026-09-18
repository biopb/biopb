"""The GetFlightInfo response mask: which optional parts a request asked for.

One reader, shared by everything that honours the mask -- the server's
``get_flight_info``, the adapter base's ``plan_flight_info``, and the remote
proxy that forwards a request upstream -- so the set of legal paths is defined
once and a request cannot mean different things to two of them.

Every path costs I/O the base descriptor does not, which is why the mask is
opt-in throughout and empty means *none*. The three ``with_*`` bools this
replaced defaulted the most expensive part (the read plan) to on, so a describe
had to remember to switch it off; the cheap call is now the one you get by
saying nothing.
"""

from __future__ import annotations

from typing import FrozenSet

import pyarrow.flight as flight
from biopb.tensor.descriptor_pb2 import TensorReadOption

__all__ = [
    "ENDPOINTS",
    "IS_RESIDENT",
    "METADATA_JSON",
    "PYRAMID",
    "READ_MASK_PATHS",
    "UPLOAD_STATUS",
    "read_mask",
]

#: The per-request chunk read plan (O(chunks)). A read asks; a describe does not.
ENDPOINTS = "endpoints"
#: The full OME tree -- megabytes on a per-plane-annotated file.
METADATA_JSON = "metadata_json"
#: The advertised resolution pyramid; sizing a native one opens each level.
PYRAMID = "pyramid"
#: Progress of the upload backing this tensor. The one cheap path.
UPLOAD_STATUS = "upload_status"
#: Whether the bytes are local right now -- a bounded stat walk.
IS_RESIDENT = "is_resident"

READ_MASK_PATHS: FrozenSet[str] = frozenset(
    {ENDPOINTS, METADATA_JSON, PYRAMID, UPLOAD_STATUS, IS_RESIDENT}
)


def read_mask(read_opt: TensorReadOption) -> FrozenSet[str]:
    """The paths *read_opt* asked for, validated.

    An unknown path is refused rather than dropped. A client that asks for a
    part this server does not serve has planned around receiving it, so silence
    would surface later as a missing field with no explanation -- and a typo in
    a path is exactly the case that would otherwise read as "the server chose
    not to fill it".
    """
    paths = frozenset(read_opt.fields.paths)
    unknown = sorted(paths - READ_MASK_PATHS)
    if unknown:
        raise flight.FlightServerError(
            f"tensor_read: unknown field mask path(s) {unknown}. "
            f"Known paths: {sorted(READ_MASK_PATHS)}."
        )
    return paths
