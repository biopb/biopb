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

# The Flight protocol version a server speaks. Distinct from the chunk encoding
# above.
#
# Reported by the ``health`` action (``protocol``) and checked by the SDK before
# its first Flight call.
#
# v3 moved what a plan says about the request it answers. A plan's
# ``FlightInfo.app_metadata`` is the whole ``TensorReadOption`` it was asked
# with, and its descriptor no longer echoes the request's ``scale_hint`` and
# ``reduction_method`` back. A v2 plan carried only the requested ``SliceHint``
# there and echoed the rest on the descriptor.
FLIGHT_PROTOCOL_VERSION = 3

# The protocol versions this SDK can talk to. A new SDK reads a v2 server's plans
# (see ``_session._plan_request``); an older SDK compares for equality and so
# refuses a v3 server, which is the intent: it would misread the plan.
SUPPORTED_FLIGHT_PROTOCOLS = frozenset({2, FLIGHT_PROTOCOL_VERSION})

# Schema-metadata key carrying the protocol a plan was written under, so a plan
# handed between processes (a ``SerializedTensor``) says how to read it without a
# connection to ask. Absent means v2.
FLIGHT_PROTOCOL_METADATA_KEY = "flight_protocol"
