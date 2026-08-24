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

The mapping is kept warm between reads and closed by the shared idle reaper on a
short TTL (:mod:`_handle_reaper`).

biopb/biopb#71 mapped per read instead, on the grounds that "mapping costs
~0.05 ms against a ~34 ms 64 MB chunk read (0.14%) and is O(1) in file size, so
there is nothing to amortise". That prices the ``mmap()`` call, which is indeed
free. What it misses is that a fresh mapping arrives with an *empty page table*,
so the copy re-faults every page it touches even when the bytes are already in
page cache -- a cost proportional to the region read, not to the mapping. Reading
a 1.6 GB volume on its own transfer grid, warm, measured 0.341 s and 49k minor
faults mapping per read against 0.106 s and **zero** through one held mapping:
3.2x, for bytes that never left RAM.

What #71 was actually protecting against is real and is still honoured: a mapping
held for as long as the source stayed catalogued pinned the file, which on
Windows makes the volume undeletable and on POSIX means an unlinked multi-GB
tomogram frees no disk space -- ``ls`` shows it gone, ``df`` disagrees. That is an
argument against holding it *forever*, which was the only option before pools
carried their own TTL. The pin is now bounded to ``_MAPPING_TTL`` seconds past
the last read rather than to the catalog's lifetime.

There is deliberately **no rsciio-dask fallback** for an unmappable layout. That
array's graph holds a memmap of its own, so the fallback reintroduced exactly the
pin this adapter exists to avoid -- silently, on a path nobody would think to
check. An MRC we cannot map is a registration failure, not a source served with
worse properties than the format's contract promises.

Chunk ID format:
- array_id prefix + whole-array bounds (base class splits oversized single chunk)

Single chunk strategy - base class handles splitting for oversized arrays.
"""

import threading
import time
from typing import TYPE_CHECKING, Any, List, Optional, Tuple

import numpy as np
from biopb.tensor.descriptor_pb2 import TensorDescriptor
from biopb.tensor.ticket_pb2 import ChunkBounds

from biopb_tensor_server.adapters._handle_reaper import IdleHandleReaper
from biopb_tensor_server.adapters._scale import axes_scale
from biopb_tensor_server.core.adapter_base import (
    TensorAdapter,
    catalog_entry,
)
from biopb_tensor_server.core.chunk import (
    content_version_from_path,
    default_transfer_chunk_shape,
)
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

# Seconds an idle mapping is kept warm. Short on purpose, and for a different
# reason than the parse-a-directory formats: what a held MRC mapping saves is a
# warm page table, not an expensive open, and that is worth only as long as the
# read that built it -- across an idle gap the next read faults in whatever it
# touches regardless, while the mapping goes on pinning the file. 5 s spans one
# whole-volume read (192 chunks at ~34 ms is ~6.5 s of reads, and the reaper
# re-stamps on every one) and bounds the #71 pin -- Windows undeletable, POSIX
# unlink-frees-nothing -- to seconds past the last read instead of the catalog's
# lifetime.
_MAPPING_TTL = 5.0

# One mapping pins one file. Bounded like every pool, so a catalogued tomogram
# collection cannot hold every volume it has ever served.
_mapping_reaper = IdleHandleReaper(_MAPPING_TTL, "mrc-mapping-reaper", max_handles=8)


class MrcAdapter(TensorAdapter):
    """Adapter for MRC electron-microscopy volumes.

    Uses rosettasciio to parse the header and an own ``np.memmap`` for lazy,
    arbitrary-sub-region reads. Single-tensor source.
    """

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

        # Fences the mapping's open/close against reads. Reads do NOT hold it for
        # their duration (see get_data), so ``_active_reads`` is what actually
        # stops a reap landing under a copy in flight.
        self._io_lock = threading.Lock()
        self._persistent_map: Optional[np.memmap] = None
        self._persistent_last_access = 0.0
        self._active_reads = 0

        # Probe the mapping once now so an unmappable layout fails at
        # registration rather than on the first read. Released immediately: a
        # source that is catalogued but never read should pin nothing.
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
            # A flat MRC volume has no block structure to align to, so the grid
            # is sized from the shape alone (biopb/biopb#809 -- the server no
            # longer splits an oversized declared grid down to the transfer
            # target, only to the Arrow ceiling).
            chunk_shape=default_transfer_chunk_shape(
                self._shape, self._dtype.str, self.dim_labels
            ),
            dtype=self._dtype.str,
        )

    def list_tensor_descriptors(self) -> List[TensorDescriptor]:
        return [catalog_entry(self.get_tensor_descriptor())]

    def get_data(self, bounds: ChunkBounds) -> np.ndarray:
        """Read a sub-region through the source's shared mapping."""
        super().get_data(bounds)
        return self._copy_out(self._bounds_to_slices(bounds))

    def get_decimated_data(
        self, bounds: ChunkBounds, step: Tuple[int, ...]
    ) -> Optional[np.ndarray]:
        """A strided slice of the same mapping: only the picked elements copy.

        Indexing a memmap computes byte offsets, so the step costs nothing to
        express and the copy shrinks by the product of the strides -- a scale-8
        pick of a 3D extent moves 1/512 of the bytes. What it does not shrink is
        what the kernel faults: a stride finer than a page still touches every
        page it steps across, which is why the saving is memcpy-and-cache first
        and I/O only where the stride outruns the readahead.
        """
        super().get_data(bounds)
        return self._copy_out(self._bounds_to_strided_slices(bounds, step))

    def _copy_out(self, slices: Tuple[slice, ...]) -> np.ndarray:
        """Copy ``slices`` out of the shared mapping, counting the read.

        The copy runs OUTSIDE _io_lock, which is the property #71 established
        and this keeps: a read-only np.memmap has no shared cursor (unlike a
        seekable file handle), indexing computes byte offsets directly and the
        copy lands in a fresh buffer, so parallel do_get chunk reads of one MRC
        source still run at once. Sharing the mapping does not change that --
        what it changes is that unmapping is now someone else's decision, so
        the read has to be counted while it is in flight. Copy out (np.array)
        so the result outlives the mapping either way.
        """
        mm = self._begin_read()
        try:
            return np.array(mm[slices])
        finally:
            self._end_read()

    def _begin_read(self) -> np.memmap:
        """Map (or reuse) the data region and mark a read in flight.

        Holds ``_io_lock`` only to hand out the mapping and take the count -- not
        across the copy. Incrementing under the lock is what makes the count
        sound: the reaper tests it under the same lock, so a read that has been
        handed a mapping is always visible before a reap can decide to drop it.
        """
        with self._io_lock:
            if self._persistent_map is None:
                self._persistent_map = self._map()
                # Stamped before register, so this adapter sorts newest and a cap
                # eviction triggered by its own register never picks it.
                self._persistent_last_access = time.monotonic()
                _mapping_reaper.register(self)
            self._persistent_last_access = time.monotonic()
            self._active_reads += 1
            return self._persistent_map

    def _end_read(self) -> None:
        with self._io_lock:
            self._active_reads -= 1
            self._persistent_last_access = time.monotonic()

    def _release_persistent_handle(self) -> None:
        """Drop the shared mapping and permit a later remap.

        The :class:`~biopb_tensor_server.adapters._handle_reaper.ReapableHandle`
        hook. Caller holds ``_io_lock`` (read path / reaper) or is the GC
        finalizer, and has already established ``_active_reads == 0`` -- unmapping
        under a copy in flight would fault, not raise. Safe to call repeatedly.
        """
        mapping, self._persistent_map = self._persistent_map, None
        _mapping_reaper.discard(self)
        if mapping is not None:
            self._release(mapping)

    def close(self) -> None:
        """Release the mapping now rather than waiting for the reaper."""
        with self._io_lock:
            if self._active_reads == 0:
                self._release_persistent_handle()

    def __del__(self):
        try:
            self._release_persistent_handle()
        except Exception:
            pass

    def _physical_scale(self) -> Optional[tuple]:
        """Voxel size + unit per dimension, from the reader's axis scales.

        rsciio's MRC axes carry ``scale`` (voxel size, typically nm) and
        ``units`` in dim order, 1:1 with ``dim_labels``.
        """
        return axes_scale(self._axes, self.dim_labels)

    def get_metadata(self) -> dict:
        """MRC header as a JSON-safe dict (rsciio hex-encodes byte/void fields)."""
        meta = {"format": "mrc"}
        for key in ("std_header", "fei_header"):
            if key in self._original_metadata:
                meta[key] = self._original_metadata[key]
        return meta
