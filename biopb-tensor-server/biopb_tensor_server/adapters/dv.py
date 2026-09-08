"""Native ``mrc``-package adapter for DeltaVision DV files.

BioIO reads a DV through ``bioio-dv``, whose Dask array pays the same
graph-rebuild-per-read cost the other bioio-backed formats do (see
``docs/dask-bypass-benchmarks.md``, biopb/biopb#799 phase 3). Unlike CZI or
plain TIFF, DV has no block structure to over-read around: the file is one
flat, C-contiguous pixel blob at a fixed offset, and ``bioio-dv`` itself reads
it through :mod:`mrc` -- the same package this adapter uses directly. So the
win here is purely the graph-construction overhead, not I/O shape.

:class:`mrc.DVFile` (the actively maintained reader in ``mrc._new``, distinct
from the legacy ``Mrc``/``Mrc2`` classes ``adapters/mrc.py`` uses for plain
cryo-EM MRC) parses the header, exposes ``sizes``/``dtype``/``voxel_size`` and
memory-maps the data region eagerly. Reads slice that mapping directly, the
same held-mapping shape :class:`~biopb_tensor_server.adapters.mrc.MrcAdapter`
uses for its own single-tensor volumes, and for the same reason (biopb/biopb#71
-- a fresh mapping re-faults every page it touches even when the bytes are
already resident).

**Scope.**  Every local DV a well-formed header describes; DVFile validates
the format's own magic (``dvid``) so a malformed file fails at registration.
A remote DV is declined by :meth:`DeltaVisionAdapter.claim`, so it falls to
BioIO's ``DvAdapter`` through the registry -- this module imports nothing from
BioIO and has no fallback path of its own.

**Type naming.**  BioIO's DV adapter already occupies the source type ``"dv"``
(unlike Zeiss, DeltaVision has no separate multi-extension vendor family name
to fall back to the way LSM/CZI left ``"zeiss"`` for BioIO and took fresh
``"lsm"``/``"czi"`` for themselves) -- see ``adapters/__init__.py`` for why the
type string must be exclusive per class (the cloud phase-2 lazy-resolve flow
looks a claimed source back up by this string). This adapter therefore claims
under ``"deltavision"`` rather than colliding with BioIO's ``"dv"``; the class
name follows the same to avoid shadowing ``bioio.DvAdapter`` in the flat
adapter-package namestore (``from .dv import DeltaVisionAdapter``).
"""

import logging
import threading
import time
from typing import TYPE_CHECKING, Any, List, Optional, Tuple

import numpy as np
from biopb.tensor.descriptor_pb2 import TensorDescriptor
from biopb.tensor.ticket_pb2 import ChunkBounds

from biopb_tensor_server.adapters._handle_reaper import IdleHandleReaper
from biopb_tensor_server.adapters._scale import MICRON, scale_by_label
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

logger = logging.getLogger(__name__)

DV_EXTENSION = ".dv"

# Seconds an idle mapping is kept warm -- same reasoning and the same value as
# adapters/mrc.py: what a held mapping saves is a warm page table, not an
# expensive open, and that is only worth as long as the read that built it.
_MAPPING_TTL = 5.0

_mapping_reaper = IdleHandleReaper(_MAPPING_TTL, "dv-mapping-reaper", max_handles=8)


class DeltaVisionAdapter(TensorAdapter):
    """Reads DeltaVision DV volumes through ``mrc.DVFile``. Single-tensor source."""

    SOURCE_TYPE = "deltavision"

    @classmethod
    def claim(cls, ctx: ClaimContext, state: "DiscoveryState") -> Optional[SourceClaim]:
        """Claim a local DV by extension alone, without reading its content.

        Same shape as :class:`~biopb_tensor_server.adapters.czi.CziAdapter`:
        definite even under a cloud root / dehydrated placeholder (claim reads
        nothing, so it cannot trigger a recall), remote is declined so BioIO's
        ``DvAdapter`` can serve it instead.
        """
        if not ctx.is_file() or ctx.is_remote:
            return None
        if not ctx.name.lower().endswith(DV_EXTENSION):
            return None

        state.try_claim_path(ctx.path_str)
        return SourceClaim(
            source_type=cls.SOURCE_TYPE,
            primary_path=ctx.path_str,
            is_remote=False,
        )

    @classmethod
    def create_from_config(
        cls,
        source: "SourceConfig",
        credentials_config: Optional[Any] = None,
    ) -> "DeltaVisionAdapter":
        """Create a native adapter for a local DV file.

        A remote DV never reaches here through discovery -- ``claim`` declines
        it -- and an explicitly configured remote url is refused rather than
        silently rerouted.
        """
        if source.is_remote:
            raise ValueError(f"{cls.__name__} only supports local files")

        url = str(source.url)
        path = url[len("file://") :] if url.startswith("file://") else url

        return cls(path, source.source_id, dim_labels=source.dim_labels)

    def __init__(
        self,
        url: str,
        source_id: str,
        dim_labels: Optional[List[str]] = None,
    ):
        self.source_id = source_id
        self._url = url
        self._source_url = url
        self._source_type = self.SOURCE_TYPE
        self._content_version = content_version_from_path(url)

        # Probe the header once now so a malformed file fails at registration
        # rather than on the first read; the mapping is released immediately
        # (a source that is catalogued but never read should pin nothing).
        import mrc

        with mrc.DVFile(url) as probe:
            self._axes = str(probe.axes)  # e.g. "CTZYX" -- native loop order + YX
            self._shape = tuple(int(probe.sizes[axis]) for axis in self._axes)
            self._dtype = probe.dtype
            self._voxel = probe.voxel_size

        native_labels = list(self._axes)
        if dim_labels and len(dim_labels) != len(native_labels):
            logger.warning(
                "dv: ignoring %d configured dim_labels for %s -- this document "
                "reads as a %d-axis %s array",
                len(dim_labels),
                url,
                len(native_labels),
                "".join(native_labels),
            )
            dim_labels = None
        self.dim_labels = list(dim_labels or native_labels)

        # Fences the mapping's open/close against reads, same protocol as
        # adapters/mrc.py: _active_reads (not _io_lock) is what actually keeps
        # a reap from landing under a copy in flight.
        self._io_lock = threading.Lock()
        self._persistent_handle: Optional[Any] = None
        self._persistent_last_access = 0.0
        self._active_reads = 0

    def get_tensor_descriptor(self) -> TensorDescriptor:
        return TensorDescriptor(
            array_id=self.array_id,
            dim_labels=self.dim_labels,
            # A flat DV volume has no block structure to align to (biopb/biopb#809).
            chunk_shape=default_transfer_chunk_shape(
                list(self._shape), self._dtype.str, self.dim_labels
            ),
            shape=list(self._shape),
            dtype=self._dtype.str,
        )

    def list_tensor_descriptors(self) -> List[TensorDescriptor]:
        return [catalog_entry(self.get_tensor_descriptor())]

    @property
    def read_block_shape(self) -> Optional[Tuple[int, ...]]:
        """None: the mapping has no block structure to align a tile to."""
        return None

    def get_data(self, bounds: ChunkBounds) -> np.ndarray:
        """Read a sub-region through the source's shared mapping."""
        super().get_data(bounds)
        return self._copy_out(self._bounds_to_slices(bounds))

    def get_decimated_data(
        self, bounds: ChunkBounds, step: Tuple[int, ...]
    ) -> Optional[np.ndarray]:
        """A strided slice of the same mapping -- see MrcAdapter for why this
        is cheap: indexing a memmap computes byte offsets, so the copy shrinks
        by the product of the strides."""
        super().get_data(bounds)
        return self._copy_out(self._bounds_to_strided_slices(bounds, step))

    def _copy_out(self, slices: Tuple[slice, ...]) -> np.ndarray:
        """Copy ``slices`` out of the shared mapping, counting the read.

        Runs OUTSIDE ``_io_lock`` (see MrcAdapter._copy_out): a read-only
        memmap has no shared cursor, so parallel reads of one DV source still
        run at once. What has to be fenced is unmapping racing a copy, which
        is what ``_active_reads`` (taken under the lock) protects.
        """
        handle = self._begin_read()
        try:
            return np.array(handle.data[slices])
        finally:
            self._end_read()

    def _open(self) -> Any:
        import mrc

        return mrc.DVFile(self._url)

    def _begin_read(self) -> Any:
        with self._io_lock:
            if self._persistent_handle is None:
                self._persistent_handle = self._open()
                self._persistent_last_access = time.monotonic()
                _mapping_reaper.register(self)
            self._persistent_last_access = time.monotonic()
            self._active_reads += 1
            return self._persistent_handle

    def _end_read(self) -> None:
        with self._io_lock:
            self._active_reads -= 1
            self._persistent_last_access = time.monotonic()

    def _release_persistent_handle(self) -> None:
        """Close the mapping and permit a later reopen.

        The :class:`~biopb_tensor_server.adapters._handle_reaper.ReapableHandle`
        hook. Caller holds ``_io_lock`` (read path / reaper) or is the GC
        finalizer, and has already established ``_active_reads == 0``. Safe to
        call repeatedly.
        """
        handle, self._persistent_handle = self._persistent_handle, None
        _mapping_reaper.discard(self)
        if handle is not None:
            try:
                handle.close()
            except Exception:
                logger.debug("error closing persistent DV mapping", exc_info=True)

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

    def _physical_scale(self) -> Optional[Tuple[List[float], List[str]]]:
        """Voxel size in micrometres, from the DV header's ``dx``/``dy``/``dz``."""
        values = {"x": self._voxel.x, "y": self._voxel.y, "z": self._voxel.z}
        return scale_by_label(self.dim_labels, values, MICRON)

    def get_metadata(self) -> dict:
        """The DV header as a JSON-safe dict; skips the per-frame extended
        header (biopb/biopb#799 -- O(sections), not worth paying for a catalog
        listing nobody asked to see frame-by-frame acquisition metadata in)."""
        try:
            import mrc

            with mrc.DVFile(self._url) as probe:
                header = dict(probe.hdr._asdict())
        except Exception:
            return {"format": "dv"}
        header.pop("blank", None)
        return {"format": "dv", "header": header}


__all__ = ["DeltaVisionAdapter"]
