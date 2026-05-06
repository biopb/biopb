"""Utilities for encoding and decoding chunk identifiers (chunk_id) used in Flight endpoints."""

import struct
from dataclasses import dataclass
from typing import Tuple

from biopb.tensor.ticket_pb2 import ChunkBounds


@dataclass
class ChunkEndpoint:
    """A chunk with its metadata for Flight endpoint creation.

    Attributes:
        chunk_id: Backend-specific chunk identifier (bytes)
        bounds: Array coordinates (start, stop) for this chunk
    """
    chunk_id: bytes
    bounds: ChunkBounds


def encode_chunk_id(
    array_id: str, 
    backend_data: bytes,     
    split_index: int = 0,
    split_max: int = 1,
) -> bytes:
    """Encode array_id and backend-specific data into chunk_id.

    Format:
    - 4 bytes: array_id length (uint32, big-endian)
    - N bytes: array_id (UTF-8)
    - M bytes: backend_data
    - 2 bytes: split_index (uint16, big-endian)
    - 2 bytes: split_max (uint16, big-endian)

    Args:
        array_id: Tensor identifier
        backend_data: Backend-specific chunk data
        split_index: Index of this split (0-based)
        split_max: Total number of splits for this chunk

    Returns:
        Encoded chunk_id bytes
    """
    array_id_bytes = array_id.encode('utf-8')
    backend_data = backend_data + struct.pack('>H', split_index) + struct.pack('>H', split_max)
    return struct.pack('>I', len(array_id_bytes)) + array_id_bytes + backend_data


def decode_chunk_id(chunk_id: bytes) -> Tuple[str, bytes, int, int]:
    """Decode array_id and backend data from chunk_id.

    Args:
        chunk_id: Encoded chunk identifier

    Returns:
        Tuple of (array_id, backend_data, split_index, split_max)
    """
    array_id_len = struct.unpack('>I', chunk_id[:4])[0]
    array_id = chunk_id[4:4+array_id_len].decode('utf-8')
    
    data = chunk_id[4+array_id_len:]

    split_max = struct.unpack('>H', data[-2:])[0]
    split_index = struct.unpack('>H', data[-4:-2])[0]
    
    backend_data = data[:-4]

    return array_id, backend_data, split_index, split_max


def get_backend_data(chunk_id: bytes) -> bytes:
    """Extract backend-specific data from chunk_id."""
    _, backend_data, _, _ = decode_chunk_id(chunk_id)
    return backend_data
