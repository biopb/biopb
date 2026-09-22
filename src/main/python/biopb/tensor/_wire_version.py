"""Chunk wire-protocol version -- the single source of truth both the tensor
server (which stamps it) and the client (which enforces it) import.

``biopb-tensor-server`` depends on ``biopb``, so this constant lives in the core
package and there is exactly one definition; the two sides can never disagree
silently. A breaking change to the chunk ``RecordBatch`` encoding therefore fails
fast at ``GetFlightInfo`` with an actionable message instead of a cryptic decode
error deep in the read path.

Bump ``TENSOR_WIRE_PROTOCOL_VERSION`` ONLY on a breaking change to the chunk
data/shape/dtype encoding (not for additive, back-compatible fields):

- ``v1`` -- typed ``data: list<T>`` per chunk (pre-#293).
- ``v2`` -- unified binary blob + numpy dtype string (biopb/biopb#293); the
  client reconstructs with ``np.frombuffer(bytes, dtype)``.

This module is stdlib-only so it stays cheap to import on every code path.
"""

TENSOR_WIRE_PROTOCOL_VERSION = 2

# Schema-metadata key carrying the server's protocol version, set on the chunk
# schema and the GetFlightInfo schema. Stored as a UTF-8 string on the wire.
WIRE_PROTOCOL_METADATA_KEY = "chunk_wire_protocol"

# The Flight protocol shape -- which descriptors, tickets and put commands the
# server understands. Distinct from the chunk encoding above: this one is
# reported by the ``health`` action (``protocol``) and checked by the SDK before
# its first Flight call, so a shape mismatch fails with an actionable message
# instead of a parse error from the wrong proto.
#
# - ``v1`` -- ``FlightCmd`` with a sentinel ``source_id``, prefix-sniffed tickets,
#   ROI annotations over DoAction (retired).
# - ``v2`` -- ``FlightRequest`` / ``TensorTicket`` / ``PutCommand`` oneofs; the
#   catalog is SQL over DoGet, ROI annotations ride DoGet / DoPut.
#   ``TensorReadOption`` carries a ``FieldMask`` rather than ``with_*`` bools,
#   and every optional part of the response is opt-in -- an empty mask is a
#   describe, where an unset ``with_read_plan`` briefly meant the full plan.
#   The ``upload_status`` and ``is_resident`` actions are gone: both are live
#   per-source reads and ride the descriptor GetFlightInfo returns
#   (biopb/biopb#1048).
#   One ``set_upload_status`` action moves an upload: it climbs PENDING ->
#   READY, with DISCARDED reachable from either, and READY seals it against
#   writes and opens it to reads in one move, so a chunk that was never
#   uploaded reads as zeros. A refused write carries its state in
#   ``extra_info``.
#   Uploads add to a source rather than create one: ``register_source`` mints
#   the source, ``add_tensor`` (was ``create_tensor``, was ``create_source``)
#   adds ``<scheme>://<source_id>/@fields/<name>`` to it and refuses a name
#   that exists, and DoPut carries a ``chunk_ticket`` from the tensor's own
#   GetFlightInfo plan rather than bounds the client chose.
#   Every action takes full access, ``chunk_locate`` included.
#   ``SerializedTensor`` is a serialized ``FlightInfo`` plus location and
#   token; GetFlightInfo stamps the requested ``slice_hint`` on the
#   FlightInfo's ``app_metadata``.
#
# The upload surface is still settling, so it moves
# inside v2 rather than minting a version per revision: the SDKs and the server
# ship from one repo and move together.
FLIGHT_PROTOCOL_VERSION = 2
