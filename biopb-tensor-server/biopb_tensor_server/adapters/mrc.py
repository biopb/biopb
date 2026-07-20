"""MRC adapter for electron-microscopy volumes.

Handles the MRC family (`.mrc`, `.mrcs`, `.rec`, `.st`, `.map`) -- the de-facto
interchange format for cryo-EM / cryo-ET, tomography, and FIB-SEM.

Reader: rosettasciio's ``rsciio.mrc`` parses the header (dim labels, voxel scale,
dtype, extended-header size). But reads are NOT routed through rsciio's dask
array -- MRC is a flat, C-contiguous blob at a fixed byte offset, so this adapter
maps the data region itself with ``np.memmap`` and slices it per
``get_data(bounds)``, serving an arbitrary sub-region while touching only the
requested pages. ``metadata_file=None`` disables rsciio's DE-movie 4D-STEM
auto-discovery, guaranteeing the contiguous layout the memmap assumes; a file
whose data region cannot back that layout is rejected at registration.

The mapping is created and released **per read** (biopb/biopb#71). A long-lived
mapping pinned the file for as long as the source stayed catalogued, which on
Windows makes the volume undeletable and on POSIX means an unlinked multi-GB
tomogram frees no disk space -- ``ls`` shows it gone, ``df`` disagrees. Mapping
costs ~0.05 ms against a ~34 ms 64 MB chunk read (0.14%) and is O(1) in file
size, so there is nothing to amortise.

There is deliberately **no rsciio-dask fallback** for an unmappable layout. That
array's graph holds a memmap of its own, so the fallback reintroduced exactly the
pin this adapter exists to avoid -- silently, on a path nobody would think to
check. An MRC we cannot map is a registration failure, not a source served with
worse properties than the format's contract promises.

Chunk ID format:
- array_id prefix + whole-array bounds (base class splits oversized single chunk)

Single chunk strategy - base class handles splitting for oversized arrays.
"""

from typing import TYPE_CHECKING, Any, List, Optional

import numpy as np
from biopb.tensor.descriptor_pb2 import TensorDescriptor
from biopb.tensor.ticket_pb2 import ChunkBounds

from biopb_tensor_server.core.base import TensorAdapter
from biopb_tensor_server.core.chunk import content_version_from_path
from biopb_tensor_server.core.discovery import ClaimContext, SourceClaim

if TYPE_CHECKING:
    from biopb_tensor_server.core.config import SourceConfig
    from biopb_tensor_server.core.discovery import DiscoveryState

# MRC family extensions. Claimed at the filesystem level; the header decides the
# rest at read time.
MRC_EXTENSIONS = (".mrc", ".mrcs", ".rec", ".st", ".map")

# Standard MRC-2014 header is 1024 bytes; the extended header (NEXT bytes)
# follows, then the raw data.
_MRC_HEADER_BYTES = 1024


class MrcAdapter(TensorAdapter):
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
    ):
        self.source_id = source_id
        self._url = url
        self._shape = shape
        self._dtype = np.dtype(dtype)
        self._axes = axes
        self._original_metadata = original_metadata

        self._source_url = source_url if source_url else url
        # Cheap content_version from the file's stat signature (#178): O(1),
        # folded into minted chunk_ids so a re-saved file gets a fresh cache
        # namespace. None (unresolved / non-file url) leaves the source unversioned.
        self._content_version = content_version_from_path(self._source_url)
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

        # Offset of the contiguous data region; reads map from here (per read --
        # see the module docstring). Probe the mapping once now so an unmappable
        # layout fails at registration rather than on the first read.
        self._offset = _MRC_HEADER_BYTES + int(std_header.get("NEXT", 0) or 0)
        self._release(self._map())

    def _map(self) -> np.memmap:
        """Map the data region read-only. Caller must ``_release`` the result."""
        return np.memmap(
            self._url,
            dtype=self._dtype,
            mode="r",
            offset=self._offset,
            shape=self._shape,
        )

    @staticmethod
    def _release(mm: np.memmap) -> None:
        """Drop the mapping now, rather than whenever the GC gets to it."""
        underlying = getattr(mm, "_mmap", None)
        if underlying is not None:
            underlying.close()

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
        """Read a sub-region through a fresh memmap."""
        super().get_data(bounds)
        slices = tuple(
            slice(int(s), int(e))
            for s, e in zip(bounds.start, bounds.stop, strict=True)
        )
        # No lock: each read owns its own mapping, and a read-only np.memmap has
        # no shared cursor (unlike a seekable file handle) -- indexing computes
        # byte offsets directly and the copy lands in a fresh buffer. So parallel
        # do_get chunk reads of one MRC source run at once. Copy out so the result
        # outlives the mapping.
        mm = self._map()
        try:
            return np.array(mm[slices])
        finally:
            self._release(mm)

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
