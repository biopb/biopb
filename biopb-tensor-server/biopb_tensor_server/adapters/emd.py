"""EMD adapter for electron-microscopy datasets.

Handles EMD files (`.emd`) -- an HDF5 container -- in both notable flavors:
Berkeley/NCEM and Velox/ThermoFisher. Reader: rosettasciio's ``rsciio.emd``,
which auto-detects the flavor and returns one *signal* per dataset. Each signal
becomes a within-source tensor (field), so a multi-signal EMD is a multi-tensor
source (the bioio multi-scene model).

Reads go through rsciio's lazy dask array, which for HDF5 forwards the
**native chunk grid** (``da.from_array(dataset, chunks=dataset.chunks)``) -- the
physical/compression layout, so ``chunk_shape`` advertised to clients is the
storage-efficient one, and ``get_data(bounds)`` is a native h5py partial read
(no per-block memmap reopen, unlike the flat-blob MRC case).

Velox lazy support is complete for image data but a TODO for 4D-STEM
spectrum-images (FrameLocationTable); when rsciio returns an eager (non-dask)
array this adapter wraps it with ``dask.array.from_array`` so the read path stays
uniform (a large spectrum-image may load whole into RAM -- logged).

Chunk ID format:
- array_id (= source_id/field) + bounds
"""

import logging
import threading
from typing import TYPE_CHECKING, Any, List, Optional

import dask.array as da
import numpy as np
from biopb.tensor.descriptor_pb2 import TensorDescriptor
from biopb.tensor.ticket_pb2 import ChunkBounds

from biopb_tensor_server.adapters._scale import axes_scale
from biopb_tensor_server.core.base import TensorAdapter
from biopb_tensor_server.core.chunk import content_version_from_path
from biopb_tensor_server.core.discovery import ClaimContext, SourceClaim
from biopb_tensor_server.core.errors import InvalidTensorId, TensorNotFound

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from biopb_tensor_server.core.config import SourceConfig
    from biopb_tensor_server.core.discovery import DiscoveryState


class EmdAdapter(TensorAdapter):
    """Adapter for EMD electron-microscopy files (NCEM and Velox flavors).

    Dual-role, like the bioio adapter:
    - source-level (``signal_index=None``): lists all signals as tensors.
    - tensor-level (``signal_index=int``): reads one signal's data.
    """

    @classmethod
    def claim(cls, ctx: ClaimContext, state: "DiscoveryState") -> Optional[SourceClaim]:
        """Claim `.emd` files by extension (recall-free; no reader import)."""
        if not ctx.is_file():
            return None

        if not ctx.name.lower().endswith(".emd"):
            return None

        state.try_claim_path(ctx.path_str)
        return SourceClaim(
            source_type="emd",
            primary_path=ctx.path_str,
            is_remote=ctx.is_remote,
        )

    @classmethod
    def create_from_config(
        cls,
        source: "SourceConfig",
        credentials_config: Optional[Any] = None,
    ) -> "EmdAdapter":
        """Create source-level adapter. rsciio auto-detects NCEM vs Velox."""
        from rsciio.emd import file_reader

        url = str(source.url)
        # source.dataset (the existing HDF5 "dataset path" field) optionally pins
        # one signal; None means all signals.
        signals = file_reader(url, lazy=True, dataset_path=source.dataset)
        if not signals:
            raise ValueError(f"EMD source {url!r} contained no readable signals")

        # Velox eager-fallback: normalize any non-dask signal to a dask array so
        # the read path is uniform.
        for i, sig in enumerate(signals):
            d = sig["data"]
            if not isinstance(d, da.Array):
                logger.warning(
                    "EMD %s signal %d returned eager (non-lazy) data; wrapping. "
                    "A large spectrum-image may load whole into RAM.",
                    url,
                    i,
                )
                sig["data"] = da.from_array(np.asarray(d))

        return cls(
            source_id=source.source_id,
            url=url,
            signals=signals,
            dim_labels=source.dim_labels,
            source_url=url,
        )

    def __init__(
        self,
        source_id: str,
        url: str,
        signals: List[dict],
        signal_index: Optional[int] = None,
        dim_labels: Optional[List[str]] = None,
        source_url: Optional[str] = None,
        io_lock: Optional[threading.Lock] = None,
    ):
        self.source_id = source_id
        self._url = url
        self._signals = signals
        self.signal_index = signal_index
        self._dim_labels_override = dim_labels
        self._io_lock = io_lock if io_lock is not None else threading.Lock()
        self._tensor_adapters: dict = {}

        self._source_url = source_url if source_url else url
        # Cheap content_version from the file's stat signature (#178): O(1),
        # folded into minted chunk_ids so a re-saved file gets a fresh cache
        # namespace. None (unresolved / non-file url) leaves the source unversioned.
        self._content_version = content_version_from_path(self._source_url)
        self._source_type = "emd"

        if signal_index is not None:
            # Tensor-level: bind this signal's data + axes.
            sig = signals[signal_index]
            self._data = sig["data"]
            self._axes = sig["axes"]
            self._original_metadata = sig.get("original_metadata", {})
            self.dim_labels = self._labels_for(sig)
        else:
            # Source-level: no bound signal; dim_labels is the default for tensors.
            self._data = None
            self._axes = None
            self._original_metadata = None
            self.dim_labels = dim_labels

    def _labels_for(self, sig: dict) -> List[str]:
        """Dim labels for one signal: caller override, else reader axis names."""
        if self._dim_labels_override:
            return list(self._dim_labels_override)
        axes = sig["axes"]
        return [
            str(ax.get("name")) if ax.get("name") else f"dim{i}"
            for i, ax in enumerate(axes)
        ]

    def _field_for(self, index: int) -> str:
        """Within-source field for a signal. The signal index is the field."""
        return str(index)

    def list_tensor_descriptors(self) -> List[TensorDescriptor]:
        """One descriptor per EMD signal, carrying the native HDF5 chunk grid."""
        descriptors = []
        for i, sig in enumerate(self._signals):
            data = sig["data"]
            descriptors.append(
                TensorDescriptor(
                    array_id=f"{self.source_id}/{self._field_for(i)}",
                    dim_labels=self._labels_for(sig),
                    shape=list(data.shape),
                    # rsciio forwards the native HDF5 chunks; chunksize is the
                    # per-dim max (single chunk per grid cell here).
                    chunk_shape=list(data.chunksize),
                    dtype=np.dtype(data.dtype).str,
                )
            )
        return descriptors

    def get_tensor_descriptor(self) -> TensorDescriptor:
        if self.signal_index is not None:
            data = self._data
            return TensorDescriptor(
                array_id=self.array_id,
                dim_labels=self.dim_labels if self.dim_labels else [],
                shape=list(data.shape),
                chunk_shape=list(data.chunksize),
                dtype=np.dtype(data.dtype).str,
            )
        # Source-level: first signal's descriptor.
        return self.list_tensor_descriptors()[0]

    def get_tensor_adapter(self, tensor_id: str) -> "TensorAdapter":
        """Return a tensor-scoped adapter for a specific signal.

        The EMD field is the signal's integer index (``source_id/0``,
        ``source_id/1``, ...). A non-integer field is structurally malformed
        (``InvalidTensorId``); an integer outside the signal range is a
        well-formed id that names no signal (``TensorNotFound``). Both are the
        caller's mistake, terminal -- never the bare ``ValueError`` that would
        leak as ``FlightInternalError`` (issue #378).
        """
        field = self._within_source_field(tensor_id)
        try:
            index = int(field)
        except (TypeError, ValueError) as e:
            raise InvalidTensorId(
                f"Unknown EMD signal: {tensor_id!r}", reason="malformed_tensor_id"
            ) from e
        if not (0 <= index < len(self._signals)):
            raise TensorNotFound(
                f"Unknown EMD signal: {tensor_id!r}", reason="unknown_field"
            )

        if field in self._tensor_adapters:
            return self._tensor_adapters[field]

        adapter = EmdAdapter(
            source_id=self.source_id,
            url=self._url,
            signals=self._signals,
            signal_index=index,
            dim_labels=self._dim_labels_override,
            source_url=self._source_url,
            io_lock=self._io_lock,
        )
        adapter._tensor_name = field
        self._tensor_adapters[field] = adapter
        return adapter

    def get_data(self, bounds: ChunkBounds) -> np.ndarray:
        """Read a sub-region from this signal's dask array (native h5py read)."""
        if self.signal_index is None:
            raise ValueError("Cannot get data from source-level EMD adapter")
        super().get_data(bounds)
        slices = self._bounds_to_slices(bounds)
        with self._io_lock:
            return self._data[slices].compute()

    def _physical_scale(self) -> Optional[tuple]:
        """Voxel size + unit per dimension, from this signal's axis scales."""
        if self._axes is None:
            return None
        return axes_scale(self._axes, self.dim_labels or [])

    def get_metadata(self) -> dict:
        """Source-level EMD metadata, JSON-safe.

        EMD metadata is genuinely per-signal, and the source-level adapter
        (``signal_index is None``) has no bound signal, so this is the bare
        ``{"format": "emd"}`` header stored in the catalog row. Each signal's
        own ``original_metadata`` is served per-tensor via
        :meth:`get_tensor_metadata` (biopb/biopb#253).
        """
        if self._original_metadata is None:
            return {"format": "emd"}
        return {"format": "emd", "original_metadata": self._original_metadata}

    def get_tensor_metadata(self) -> Optional[dict]:
        """This signal's ``original_metadata`` as the delta over the source row.

        Per-signal, merged over the source-level ``{"format": "emd"}`` catalog row
        (so ``"format"`` is not repeated here). ``None`` when this signal carries
        no ``original_metadata``, or on the source-level adapter (no bound signal).
        """
        if self.signal_index is None or self._original_metadata is None:
            return None
        return {"original_metadata": self._original_metadata}
