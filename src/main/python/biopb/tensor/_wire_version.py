"""
Protocol versions for the Flight API.

This module is stdlib-only so it stays cheap to import on every code path.
"""

# Chunk wire-protocol version. Bump on a breaking change to the chunk
# data/shape/dtype encoding:
#
# A mismatch fails fast at ``GetFlightInfo`` with an actionable message.
TENSOR_WIRE_PROTOCOL_VERSION = 2

# Schema-metadata key carrying the server's protocol version, set on the chunk
# schema and the GetFlightInfo schema. Stored as a UTF-8 string on the wire.
WIRE_PROTOCOL_METADATA_KEY = "chunk_wire_protocol"

# The Flight protocol version. Distinct from the chunk encoding above.
#
# Reported by the ``health`` action (``protocol``) and checked by the SDK before
# its first Flight call.
FLIGHT_PROTOCOL_VERSION = 2
