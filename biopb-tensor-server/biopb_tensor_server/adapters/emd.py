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
(no per-block memmap reopen, unlike the flat-blob MRC case). The HDF5 file stays
open between reads and the shared idle reaper (:mod:`_handle_reaper`) closes it
once idle; the next read reopens it.

Velox lazy support is complete for image data but a TODO for 4D-STEM
spectrum-images (FrameLocationTable); when rsciio returns an eager (non-dask)
array this adapter wraps it with ``dask.array.from_array`` so the read path stays
uniform (a large spectrum-image may load whole into RAM -- logged).

Chunk ID format:
- array_id (= source_id/field) + bounds
"""

import json
import logging
import threading
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import dask.array as da
import numpy as np
from biopb.tensor.descriptor_pb2 import TensorDescriptor
from biopb.tensor.ticket_pb2 import ChunkBounds

from biopb_tensor_server.adapters._handle_reaper import IdleHandleReaper
from biopb_tensor_server.adapters._scale import axes_scale
from biopb_tensor_server.core.adapter_base import TensorAdapter
from biopb_tensor_server.core.chunk import (
    content_version_from_path,
    default_transfer_chunk_shape,
)
from biopb_tensor_server.core.discovery import ClaimContext, SourceClaim
from biopb_tensor_server.core.errors import InvalidTensorId, TensorNotFound

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from biopb_tensor_server.core.config import SourceConfig
    from biopb_tensor_server.core.discovery import DiscoveryState


def _read_signals(url: str) -> List[dict]:
    """Every signal of an EMD file as rsciio reads it lazily, each ``data`` a dask
    array. rsciio auto-detects NCEM vs Velox."""
    from rsciio.emd import file_reader

    signals = file_reader(url, lazy=True)
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
    return signals


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, bytes):
        return value.hex()
    return str(value)


def _json_safe(value: Any) -> Any:
    """*value* as plain JSON types, which is what a payload is made of."""
    return json.loads(json.dumps(value, default=_json_default))


class _NotOpened:
    """One signal's array before the file is opened: what the descriptors and the
    grid are made of (``shape``, ``dtype``, ``chunksize``), none of the data.

    A source rebuilt from its row holds these, with its handle released, so the
    HDF5 container is opened by the first read and not before.
    """

    def __init__(
        self, shape: Tuple[int, ...], dtype: np.dtype, chunksize: Tuple[int, ...]
    ):
        self.shape = shape
        self.dtype = dtype
        self.chunksize = chunksize


# Seconds an idle EMD file is kept open. Short, like MRC: a reopen is ~3.5 ms
# (rsciio re-reads the metadata and rebuilds the dask graph), so what a held file
# saves is that one open across a burst of reads, worth only as long as the burst.
_HANDLE_TTL = 5.0

# One open HDF5 file per source; bounded like every pool so a catalogued EM
# collection cannot hold every file it has ever served.
_handle_reaper = IdleHandleReaper(_HANDLE_TTL, "emd-handle-reaper", max_handles=8)


class _EmdHandle:
    """A source's open HDF5 file, shared by its source and tensor adapters.

    rosettasciio's lazy reader returns dask arrays over live h5py datasets, so the
    file stays open as long as the signals are used. This is the
    :class:`~biopb_tensor_server.adapters._handle_reaper.ReapableHandle`: the
    reaper closes the file once idle, and the next read reopens it and swaps the
    signals' arrays. Everything else a signal carries (axes, shape, chunks,
    metadata) is kept from the first open.

    A source whose signals are not backed by h5py (Velox's eager fallback) holds
    no file, so there is nothing to release.
    """

    def __init__(self, url: str, signals: List[dict], io_lock: threading.Lock):
        self._url = url
        self.signals = signals
        self._io_lock = io_lock
        self._active_reads = 0  # a read holds _io_lock throughout
        self._persistent_last_access = time.monotonic()
        # Signals still ``_NotOpened`` (a source rebuilt from its row) start
        # released: the first read opens the file.
        self._released = any(not isinstance(sig["data"], da.Array) for sig in signals)
        self._holds_file = self._released or bool(self._datasets())
        if self._holds_file and not self._released:
            _handle_reaper.register(self)

    def _datasets(self) -> list:
        """The h5py datasets under the signals' arrays.

        Each sits in a layer of its own; the per-chunk layers are lazy, and walking
        the whole graph would build them (~140 ms for 4k chunks).
        """
        import h5py
        from dask.highlevelgraph import MaterializedLayer

        return [
            node
            for sig in self.signals
            for layer in sig["data"].dask.layers.values()
            if isinstance(layer, MaterializedLayer)
            for node in layer.values()
            if isinstance(node, h5py.Dataset)
        ]

    def array(self, index: int) -> da.Array:
        """Signal *index*'s dask array, reopening the file if it was released.

        Caller holds ``_io_lock``.
        """
        if self._released:
            self._reopen()
        self._persistent_last_access = time.monotonic()
        return self.signals[index]["data"]

    def _reopen(self) -> None:
        fresh = _read_signals(self._url)
        if len(fresh) != len(self.signals):
            raise RuntimeError(
                f"EMD {self._url!r} changed on disk: it had {len(self.signals)} "
                f"signals and now has {len(fresh)}"
            )
        for sig, new in zip(self.signals, fresh, strict=True):
            sig["data"] = new["data"]
        self._released = False
        self._holds_file = bool(self._datasets())
        self._persistent_last_access = time.monotonic()
        if self._holds_file:
            _handle_reaper.register(self)

    def _release_persistent_handle(self) -> None:
        """Close the file and permit a later reopen; safe to call twice.

        The reaper hook. Caller holds ``_io_lock``.
        """
        _handle_reaper.discard(self)
        if self._released or not self._holds_file:
            return
        for dataset in self._datasets():
            if dataset:
                dataset.file.close()
        self._released = True


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
        url = str(source.url)
        return cls(
            source_id=source.source_id,
            url=url,
            signals=_read_signals(url),
            source_url=url,
        )

    @classmethod
    def create_from_payload(
        cls,
        source: "SourceConfig",
        payload: Dict[str, Any],
        metadata: Dict[str, Any],
        credentials_config: Optional[Any] = None,
    ) -> "EmdAdapter":
        """Rebuild from the row's per-signal structure; the HDF5 container is
        opened by the first read of a signal."""
        url = str(source.url)
        signals = [
            {
                "data": _NotOpened(
                    tuple(int(s) for s in entry["shape"]),
                    np.dtype(entry["dtype"]),
                    tuple(int(c) for c in entry["chunksize"]),
                ),
                "axes": entry["axes"],
                "original_metadata": entry["original_metadata"],
            }
            for entry in payload["signals"]
        ]
        return cls(source_id=source.source_id, url=url, signals=signals, source_url=url)

    def catalog_payload(self) -> Optional[Dict[str, Any]]:
        """Each signal's structure, calibration and metadata: what its descriptor,
        grid, scale and per-tensor metadata are made of. Source level only."""
        if self.signal_index is not None:
            return None
        return {
            "signals": [
                {
                    "shape": [int(s) for s in sig["data"].shape],
                    "dtype": np.dtype(sig["data"].dtype).str,
                    "chunksize": [int(c) for c in sig["data"].chunksize],
                    "axes": [
                        {
                            "name": ax.get("name"),
                            "scale": _json_safe(ax.get("scale")),
                            "units": ax.get("units"),
                        }
                        for ax in sig["axes"]
                    ],
                    "original_metadata": _json_safe(sig.get("original_metadata", {})),
                }
                for sig in self._signals
            ]
        }

    def __init__(
        self,
        source_id: str,
        url: str,
        signals: List[dict],
        signal_index: Optional[int] = None,
        source_url: Optional[str] = None,
        io_lock: Optional[threading.Lock] = None,
        handle: Optional[_EmdHandle] = None,
    ):
        self.source_id = source_id
        self._url = url
        self._signals = signals
        self.signal_index = signal_index
        self._io_lock = io_lock if io_lock is not None else threading.Lock()
        self._handle = handle or _EmdHandle(url, signals, self._io_lock)
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
            # Source-level: no bound signal or axis labels.
            self._data = None
            self._axes = None
            self._original_metadata = None
            self.dim_labels = None

    def _labels_for(self, sig: dict) -> List[str]:
        """Dimension labels from one signal's reader axis names."""
        axes = sig["axes"]
        return [
            str(ax.get("name")) if ax.get("name") else f"dim{i}"
            for i, ax in enumerate(axes)
        ]

    def _field_for(self, index: int) -> str:
        """Within-source field for a signal. The signal index is the field."""
        return str(index)

    def list_tensor_descriptors(self) -> List[TensorDescriptor]:
        """One structural entry per EMD signal (no grid -- biopb/biopb#812)."""
        return [
            TensorDescriptor(
                array_id=f"{self.source_id}/{self._field_for(i)}",
                dim_labels=self._labels_for(sig),
                shape=list(sig["data"].shape),
                dtype=np.dtype(sig["data"].dtype).str,
            )
            for i, sig in enumerate(self._signals)
        ]

    def _serving_descriptor(self, index: int) -> TensorDescriptor:
        """The full descriptor for signal ``index``, grid included.

        Sized against that signal's own dtype, labels and native HDF5 chunks --
        rsciio forwards those and ``chunksize`` is the per-dim max (a single
        chunk per grid cell here), which seeds the transfer grid (#809).
        """
        sig = self._signals[index]
        data = sig["data"]
        labels = self._labels_for(sig)
        return TensorDescriptor(
            array_id=f"{self.source_id}/{self._field_for(index)}",
            dim_labels=labels,
            shape=list(data.shape),
            chunk_shape=default_transfer_chunk_shape(
                data.shape,
                np.dtype(data.dtype).str,
                labels,
                native=data.chunksize,
            ),
            dtype=np.dtype(data.dtype).str,
        )

    def get_tensor_descriptor(self) -> TensorDescriptor:
        if self.signal_index is not None:
            desc = self._serving_descriptor(self.signal_index)
            # This adapter's own identity: the bound field name is authoritative
            # over the index-derived one.
            desc.array_id = self.array_id
            return desc
        # Source-level: the default (first) signal, sized from that signal --
        # not read back off the catalog listing, which carries no grid.
        return self._serving_descriptor(0)

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
            source_url=self._source_url,
            io_lock=self._io_lock,
            handle=self._handle,
        )
        adapter._tensor_name = field
        self._tensor_adapters[field] = adapter
        return adapter

    def close(self) -> None:
        """Close the HDF5 file now rather than waiting for the reaper.

        The signals are shared by the source-level adapter and its tensor
        adapters, so closing through any of them closes them all. A later read
        reopens the file.
        """
        with self._io_lock:
            self._handle._release_persistent_handle()

    @property
    def read_block_shape(self) -> Optional[Tuple[int, ...]]:
        """The dask block -- the ``native=`` seed of this field's grid."""
        chunksize = getattr(self._data, "chunksize", None)
        return tuple(int(size) for size in chunksize) if chunksize else None

    def get_data(self, bounds: ChunkBounds) -> np.ndarray:
        """Read a sub-region from this signal's dask array (native h5py read)."""
        if self.signal_index is None:
            raise ValueError("Cannot get data from source-level EMD adapter")
        super().get_data(bounds)
        slices = self._bounds_to_slices(bounds)
        with self._io_lock:
            return self._handle.array(self.signal_index)[slices].compute()

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
