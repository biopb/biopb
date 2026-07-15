"""MRC adapter for electron-microscopy volumes.

Handles the MRC family (`.mrc`, `.mrcs`, `.rec`, `.st`, `.map`) -- the de-facto
interchange format for cryo-EM / cryo-ET, tomography, and FIB-SEM.

Reader: rosettasciio's ``rsciio.mrc`` parses the header (dim labels, voxel scale,
dtype, extended-header size). But reads are NOT routed through rsciio's dask
array -- MRC is a flat, C-contiguous blob at a fixed byte offset, so this adapter
holds ONE long-lived ``np.memmap`` and slices it per ``get_data(bounds)``. That
serves an arbitrary sub-region touching only the requested pages, with no
per-read memmap reopen (rsciio's dask path re-opens a memmap per block per
compute; see biopb/biopb#94). ``metadata_file=None`` disables rsciio's DE-movie
4D-STEM auto-discovery, guaranteeing the contiguous layout the memmap assumes;
if the on-disk region is too small for that layout the adapter falls back to
rsciio's lazy dask array.

Chunk ID format:
- array_id prefix + whole-array bounds (base class splits oversized single chunk)

Single chunk strategy - base class handles splitting for oversized arrays.
"""

import logging
import threading
from typing import TYPE_CHECKING, Any, List, Optional

import numpy as np
from biopb.tensor.descriptor_pb2 import TensorDescriptor
from biopb.tensor.ticket_pb2 import ChunkBounds

from biopb_tensor_server.core.base import SourceAdapter, TensorAdapter
from biopb_tensor_server.core.discovery import ClaimContext, SourceClaim

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from biopb_tensor_server.core.config import SourceConfig
    from biopb_tensor_server.core.discovery import DiscoveryState

# MRC family extensions. Claimed at the filesystem level; the header decides the
# rest at read time.
MRC_EXTENSIONS = (".mrc", ".mrcs", ".rec", ".st", ".map")

# Standard MRC-2014 header is 1024 bytes; the extended header (NEXT bytes)
# follows, then the raw data.
_MRC_HEADER_BYTES = 1024


class MrcAdapter(SourceAdapter, TensorAdapter):
    """Adapter for MRC electron-microscopy volumes.

    Uses rosettasciio to parse the header and an own ``np.memmap`` for lazy,
    arbitrary-sub-region reads. Single-tensor source.
    """

    _single_tensor_source = True

    @classmethod
    def claim(cls, ctx: ClaimContext, state: "DiscoveryState") -> Optional[SourceClaim]:
        """Claim MRC-family files (.mrc/.mrcs/.rec/.st/.map).

        Pure extension check -- no reader import, no content read (recall-free,
        so a cloud/synced-folder placeholder is not recalled here).
        """
        if not ctx.is_file():
            return None

        name = ctx.name.lower()
        if not name.endswith(MRC_EXTENSIONS):
            return None

        state.try_claim_path(ctx.path_str)
        return SourceClaim(
            source_type="mrc",
            primary_path=ctx.path_str,
            is_remote=ctx.is_remote,
        )

    @classmethod
    def create_from_config(
        cls,
        source: "SourceConfig",
        credentials_config: Optional[Any] = None,
    ) -> "MrcAdapter":
        """Create adapter instance from SourceConfig.

        Reads only the header via rosettasciio (``lazy=True`` never touches the
        data); the pixel bytes are reached later through the memmap.
        """
        from rsciio.mrc import file_reader

        url = str(source.url)
        # metadata_file=None disables rsciio's DE-movie 4D-STEM auto-discovery,
        # keeping the read a plain contiguous MRC (the layout the memmap assumes).
        sig = file_reader(url, lazy=True, metadata_file=None)[0]

        data = sig["data"]  # lazy dask array; gives shape + dtype
        axes = sig["axes"]
        std_header = sig["original_metadata"].get("std_header", {})

        return cls(
            source_id=source.source_id,
            url=url,
            shape=tuple(int(s) for s in data.shape),
            dtype=np.dtype(data.dtype),
            axes=axes,
            std_header=std_header,
            original_metadata=sig["original_metadata"],
            dim_labels=source.dim_labels,
            source_url=url,
            lazy_data=data,
        )

    def __init__(
        self,
        source_id: str,
        url: str,
        shape: tuple,
        dtype: np.dtype,
        axes: List[dict],
        std_header: dict,
        original_metadata: dict,
        dim_labels: Optional[List[str]] = None,
        source_url: Optional[str] = None,
        lazy_data=None,
    ):
        self.source_id = source_id
        self._url = url
        self._shape = shape
        self._dtype = np.dtype(dtype)
        self._axes = axes
        self._original_metadata = original_metadata
        self._io_lock = threading.Lock()

        self._source_url = source_url if source_url else url
        self._source_type = "mrc"

        # Dimension labels: caller override, else the reader's axis names
        # (default z,y,x), else positional.
        if dim_labels:
            self.dim_labels = list(dim_labels)
        else:
            self.dim_labels = [
                str(ax.get("name")) if ax.get("name") else f"dim{i}"
                for i, ax in enumerate(axes)
            ]

        # Own long-lived memmap over the contiguous data region. Falls back to
        # rsciio's lazy dask array if the on-disk region can't back the layout.
        offset = _MRC_HEADER_BYTES + int(std_header.get("NEXT", 0) or 0)
        self._mm = None
        self._lazy_data = lazy_data
        try:
            self._mm = np.memmap(
                url, dtype=self._dtype, mode="r", offset=offset, shape=self._shape
            )
        except (ValueError, OSError):
            # Region too small / unmappable: use rsciio's dask array instead.
            if lazy_data is None:
                raise
            logger.warning(
                "MRC %s: contiguous memmap unavailable; falling back to rsciio "
                "lazy read",
                url,
            )

    def get_tensor_descriptor(self) -> TensorDescriptor:
        return TensorDescriptor(
            array_id=self.array_id,
            dim_labels=self.dim_labels,
            shape=list(self._shape),
            chunk_shape=list(self._shape),  # single chunk; base splits oversize
            dtype=self._dtype.str,
        )

    def list_tensor_descriptors(self) -> List[TensorDescriptor]:
        return [self.get_tensor_descriptor()]

    def get_data(self, bounds: ChunkBounds) -> np.ndarray:
        """Read a sub-region. Slices the own memmap (or the dask fallback)."""
        super().get_data(bounds)
        slices = tuple(
            slice(int(s), int(e))
            for s, e in zip(bounds.start, bounds.stop, strict=True)
        )
        if self._mm is not None:
            # No lock on the fast path: a read-only np.memmap has no shared cursor
            # (unlike a seekable file handle), so indexing computes byte offsets
            # directly and the copy lands in a fresh buffer -- concurrent reads are
            # thread-safe. Skipping the lock lets parallel do_get chunk reads of one
            # MRC source run at once. Copy out so the result is independent of the
            # mapping.
            return np.array(self._mm[slices])
        # Fallback: rsciio's dask array reopens a memmap per block on compute; keep
        # it under the lock (the slow, rarely-taken path).
        with self._io_lock:
            return self._lazy_data[slices].compute()

    def _physical_scale(self) -> Optional[tuple]:
        """Voxel size + unit per dimension, from the reader's axis scales.

        rsciio's MRC axes carry ``scale`` (voxel size, typically nm) and
        ``units`` in dim order, 1:1 with ``dim_labels``.
        """
        scale: List[float] = []
        unit: List[str] = []
        for ax in self._axes:
            try:
                v = float(ax.get("scale") or 0.0)
            except (TypeError, ValueError):
                v = 0.0
            if v > 0:
                scale.append(v)
                unit.append(str(ax.get("units")) if ax.get("units") else "")
            else:
                scale.append(0.0)
                unit.append("")
        if len(scale) != len(self.dim_labels) or not any(scale):
            return None
        return scale, unit

    def get_metadata(self) -> dict:
        """MRC header as a JSON-safe dict (rsciio hex-encodes byte/void fields)."""
        meta = {"format": "mrc"}
        for key in ("std_header", "fei_header"):
            if key in self._original_metadata:
                meta[key] = self._original_metadata[key]
        return meta
