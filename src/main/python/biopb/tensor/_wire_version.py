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
