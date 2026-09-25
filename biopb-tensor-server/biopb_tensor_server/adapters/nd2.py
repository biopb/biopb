"""Native ``nd2``-package adapter for Nikon ND2 files.

BioIO reads an ND2 through ``bioio-nd2``, whose Dask array pays the same
graph-rebuild-per-read cost every other bioio-backed format does.
biopb/biopb#797 already bypassed that graph for pixel reads by calling
``nd2.read_frame`` directly from inside BioIO's ``NikonAdapter`` -- but scene
enumeration, dimension labels and metadata still went through BioIO's
``BioImage``/OME conversion. This module finishes the migration
(biopb/biopb#799 phase 3): every fact this adapter reports, structural or
physical, is read from the ``nd2`` package alone.

**One tensor per XY stage position.** ``ND2File.sizes`` folds ``P`` (nd2's
XY-position loop) in as an ordinary loop axis alongside ``T``/``Z``: every
position in an ND2 experiment shares the same T/Z/C/Y/X shape, because the
file's ``experiment`` structure is a single flat nested-loop definition (one
scalar ``count`` per loop, applied to the whole file) with no per-position
sub-loop to make positions vary. That uniformity would make folding ``P`` into
a leading descriptor axis *safe* -- but this adapter still splits each
position into its own tensor, matching the multi-field convention every other
adapter here uses (:class:`~biopb_tensor_server.adapters.lif.LifAdapter`'s
one-tensor-per-image, the BioIO adapter's one-tensor-per-scene, and BioIO's
own ``NikonAdapter``, which this module replaces for local files). A client
that wants one position doesn't want the
others' bytes folded into its shape, and ``dim_labels`` stays the canonical
T/Z/C/Y/X/S rather than carrying a format-specific ``P`` a generic consumer
has no reason to know about.

**Frame addressing.**  ``nd2.read_frame`` decodes one loop coordinate
(P/T/Z) at a time and hands back the full C/Y/X[/S] block for it --
components and RGB samples are baked into the pixel, not looped. This
adapter builds the loop-coordinate -> frame-index map once at registration
(from ``ND2File.loop_indices``) and rejects a file where two frames share a
coordinate (an unrepresented/custom acquisition loop) at that point, rather
than guessing or silently mis-serving one of them -- there is no BioIO
fallback left to catch it.

A remote ND2 is declined by :meth:`Nd2Adapter.claim`, so it falls to BioIO's
``NikonAdapter`` through the registry -- this module imports nothing from
BioIO.
"""

import logging
import math
import threading
import time
from dataclasses import dataclass
from itertools import product
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
from biopb.tensor.descriptor_pb2 import TensorDescriptor
from biopb.tensor.ticket_pb2 import ChunkBounds

from biopb_tensor_server.adapters._handle_reaper import IdleHandleReaper
from biopb_tensor_server.adapters._scale import MICRON, scale_by_label
from biopb_tensor_server.core import chunk as chunk_policy
from biopb_tensor_server.core.adapter_base import (
    TensorAdapter,
    catalog_entry,
)
from biopb_tensor_server.core.chunk import (
    compute_transfer_chunk_size,
    content_version_from_path,
    default_transfer_chunk_shape,
    estimate_chunk_bytes,
)
from biopb_tensor_server.core.discovery import ClaimContext, SourceClaim
from biopb_tensor_server.core.errors import TensorNotFound

if TYPE_CHECKING:
    from biopb_tensor_server.core.config import SourceConfig
    from biopb_tensor_server.core.discovery import DiscoveryState

logger = logging.getLogger(__name__)

ND2_EXTENSION = ".nd2"

# The loop axes nd2 addresses one frame at a time (see the module docstring).
# C and S (RGB) are baked into the frame ``read_frame`` returns, never looped.
_LOOP_AXES = ("P", "T", "Z")
_POSITION_AXIS = "P"

# Same TTL and reasoning as bioio.py's ND2 reader pool: what a held reader
# saves is a warm page table over ``nd2.read_frame``'s mmap view, which is
# only worth as long as the read that built it -- so the TTL is seconds, not
# the long default every reopen-is-cheap format uses.
_READER_TTL = 5.0

_reader_reaper = IdleHandleReaper(_READER_TTL, "nd2-reader-reaper", max_handles=8)


class _Nd2Reader:
    """One ND2 reader, shared by every position adapter of one source.

    The handle is per *file* while adapters are per position -- a reader per
    position would map the same file once per position to serve reads that
    are already serialized behind the one ``_io_lock`` they share.
    """

    def __init__(self, url: str, io_lock: threading.Lock) -> None:
        self._url = url
        # The source's lock, shared with every position adapter -- so the
        # fence the reaper takes is the same one reads hold.
        self._io_lock = io_lock
        self._reader = None
        # Reads hold ``_io_lock`` end to end, so none is ever in flight when
        # the reaper takes it.
        self._active_reads = 0
        self._persistent_last_access = 0.0

    def acquire(self):
        """Open (or reuse) the reader. Caller holds ``_io_lock``."""
        # Stamped before register, so this handle sorts newest and a cap
        # eviction triggered by its own register never picks it.
        self._persistent_last_access = time.monotonic()
        if self._reader is None:
            import nd2

            self._reader = nd2.ND2File(self._url)
            _reader_reaper.register(self)
        return self._reader

    def _release_persistent_handle(self) -> None:
        """Close the reader and permit a later reopen.

        The :class:`~biopb_tensor_server.adapters._handle_reaper.ReapableHandle`
        hook. Caller holds ``_io_lock`` (read path / reaper) or is the GC
        finalizer. Safe to call repeatedly.
        """
        reader, self._reader = self._reader, None
        _reader_reaper.discard(self)
        if reader is not None:
            try:
                reader.close()
            except Exception:
                logger.debug("error closing persistent ND2 reader", exc_info=True)

    def __del__(self):
        try:
            self._release_persistent_handle()
        except Exception:
            pass


@dataclass(frozen=True)
class _Nd2Layout:
    """What one probe of an ND2 file tells the adapter. Built once, at registration."""

    labels: Tuple[str, ...]
    shape: Tuple[int, ...]
    dtype: np.dtype
    voxel_um: Dict[str, Optional[float]]
    ome_summary: dict
    #: loop coordinate (one int per axis in ``_LOOP_AXES`` present in ``labels``)
    #: -> frame index, i.e. what ``read_frame`` takes.
    frame_indices: Dict[Tuple[int, ...], int]

    @property
    def n_positions(self) -> int:
        """Positions to expose as tensors. 1 when the file has no ``P`` axis
        (``ND2File.sizes`` omits size-1 axes)."""
        if _POSITION_AXIS not in self.labels:
            return 1
        return int(self.shape[self.labels.index(_POSITION_AXIS)])

    @property
    def field_labels(self) -> List[str]:
        """This file's native axes with the position loop removed."""
        return [label for label in self.labels if label != _POSITION_AXIS]

    @property
    def field_shape(self) -> List[int]:
        return [
            int(size)
            for label, size in zip(self.labels, self.shape, strict=True)
            if label != _POSITION_AXIS
        ]


def _loop_key(coordinates: Dict[str, int], present: Tuple[str, ...]) -> Tuple[int, ...]:
    return tuple(coordinates.get(axis, 0) for axis in present)


def _ome_summary(reader) -> dict:
    try:
        ome = reader.ome_metadata()
    except Exception:
        return {}
    try:
        return ome.model_dump(mode="json")
    except Exception:
        return {}


def read_layout(path: str) -> _Nd2Layout:
    """Probe an ND2 file and describe what it takes to read it.

    Raises when two frames share a (P, T, Z) loop coordinate: an unrepresented
    or custom acquisition loop this adapter has no way to address unambiguously.
    """
    import nd2

    with nd2.ND2File(path) as reader:
        sizes = reader.sizes  # ordered, singleton axes already dropped
        labels = tuple(sizes.keys())
        shape = tuple(int(size) for size in sizes.values())
        dtype = reader.dtype
        voxel = reader.voxel_size()
        voxel_um = {"x": voxel.x, "y": voxel.y, "z": voxel.z}
        ome_summary = _ome_summary(reader)

        present = tuple(axis for axis in _LOOP_AXES if axis in labels)
        frame_indices: Dict[Tuple[int, ...], int] = {}
        for frame_index, coordinates in enumerate(reader.loop_indices):
            key = _loop_key(coordinates, present)
            if key in frame_indices:
                raise ValueError(
                    f"ND2 {path} has two frames at loop coordinate "
                    f"{dict(zip(present, key, strict=True))}; this file's "
                    "acquisition loop is not addressable frame-by-frame"
                )
            frame_indices[key] = frame_index

    return _Nd2Layout(
        labels=labels,
        shape=shape,
        dtype=dtype,
        voxel_um=voxel_um,
        ome_summary=ome_summary,
        frame_indices=frame_indices,
    )


class Nd2Adapter(TensorAdapter):
    """Reads a Nikon ND2 file through the ``nd2`` package, one tensor per XY
    stage position.

    Dual-role, the same shape :class:`~biopb_tensor_server.adapters.lif.LifAdapter`
    uses:

    - Source-level (``position=None``): manages the file's layout, lists every
      position as its own tensor.
    - Position-level (``position=int``): handles data access for one position.
    """

    SOURCE_TYPE = "nd2"

    @classmethod
    def claim(cls, ctx: ClaimContext, state: "DiscoveryState") -> Optional[SourceClaim]:
        """Claim a local ND2 by extension alone, without reading its content.

        Same shape as :class:`~biopb_tensor_server.adapters.czi.CziAdapter`:
        definite even under a cloud root / dehydrated placeholder, remote is
        declined so BioIO's ``NikonAdapter`` can serve it instead. Whether this
        file's acquisition loop is addressable is decided at construction
        (:func:`read_layout`), not here.
        """
        if not ctx.is_file() or ctx.is_remote:
            return None
        if not ctx.name.lower().endswith(ND2_EXTENSION):
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
    ) -> "Nd2Adapter":
        """Create a native adapter for a local ND2 file."""
        if source.is_remote:
            raise ValueError(f"{cls.__name__} only supports local files")

        url = str(source.url)
        path = url[len("file://") :] if url.startswith("file://") else url

        return cls(
            path,
            source.source_id,
            layout=read_layout(path),
        )

    def __init__(
        self,
        url: str,
        source_id: str,
        layout: _Nd2Layout,
        position: Optional[int] = None,
        io_lock: Optional[threading.Lock] = None,
        shared_handle: Optional["_Nd2Reader"] = None,
    ):
        self.source_id = source_id
        self._url = url
        self._layout = layout
        self._source_url = url
        self._source_type = self.SOURCE_TYPE
        self._content_version = content_version_from_path(url)
        self.position = position

        # Source-level: no bound position, no labels. Position-level: the
        # file's own axes, with P folded away.
        self.dim_labels = None if position is None else list(layout.field_labels)

        # T/Z only: P is fixed for a position-level adapter, never looped.
        self._present_loop_axes = tuple(
            axis
            for axis in _LOOP_AXES
            if axis in layout.labels and axis != _POSITION_AXIS
        )
        if position is not None:
            self._frame_indices, self._frame_shape = self._position_frame_plan(
                layout, position
            )

        # One reader per file, shared by every position adapter -- held warm
        # the same way CziAdapter holds its libCZI reader, since
        # ``nd2.read_frame`` returns a zero-copy view onto the reader's mmap
        # and a reopen re-faults every page a crop touches even when the
        # bytes are already resident.
        self._io_lock = io_lock if io_lock is not None else threading.Lock()
        self._shared_handle: Optional[_Nd2Reader] = shared_handle
        self._tensor_adapters: Dict[str, Nd2Adapter] = {}

    def _position_frame_plan(
        self, layout: "_Nd2Layout", position: int
    ) -> Tuple[Dict[Tuple[int, ...], int], Tuple[int, ...]]:
        """The (T, Z) frame-index lookup and reshape target for one fixed
        position, computed once rather than re-derived on every read.

        ``read_frame`` never returns a ``P`` axis, so both live in field
        space from the start -- no leading size-1 axis is ever created only
        to be sliced back off.
        """
        if _POSITION_AXIS not in layout.labels:
            frame_indices = dict(layout.frame_indices)
        else:
            present = tuple(axis for axis in _LOOP_AXES if axis in layout.labels)
            p_index = present.index(_POSITION_AXIS)
            frame_indices = {
                key[:p_index] + key[p_index + 1 :]: frame_index
                for key, frame_index in layout.frame_indices.items()
                if key[p_index] == position
            }
        frame_shape = tuple(self._native_block(layout.field_labels, layout.field_shape))
        return frame_indices, frame_shape

    # ---- descriptors --------------------------------------------------------

    def _descriptor_for(
        self, position: int, labels: Optional[List[str]] = None
    ) -> TensorDescriptor:
        if labels is None:
            labels = self._layout.field_labels
        shape = self._layout.field_shape
        dtype = self._layout.dtype.str
        return TensorDescriptor(
            array_id=f"{self.source_id}/{_POSITION_AXIS}:{position}",
            dim_labels=labels,
            chunk_shape=self._transfer_chunk_shape(labels, shape, dtype),
            shape=shape,
            dtype=dtype,
        )

    def list_tensor_descriptors(self) -> List[TensorDescriptor]:
        return [
            catalog_entry(self._descriptor_for(position))
            for position in range(self._layout.n_positions)
        ]

    def get_tensor_descriptor(self) -> TensorDescriptor:
        if self.position is not None:
            return self._descriptor_for(self.position, labels=self.dim_labels)
        return self.get_tensor_adapter(f"{_POSITION_AXIS}:0").get_tensor_descriptor()

    def get_tensor_adapter(self, tensor_id: Optional[str]) -> "Nd2Adapter":
        field = self._within_source_field(tensor_id)
        position = self._position_for_field(field)
        cached = self._tensor_adapters.get(field)
        if cached is not None:
            return cached

        adapter = self.__class__(
            self._url,
            self.source_id,
            self._layout,
            position=position,
            io_lock=self._io_lock,
            shared_handle=self._reader_handle(),
        )
        adapter._tensor_name = field
        self._tensor_adapters[field] = adapter
        return adapter

    def _reader_handle(self) -> "_Nd2Reader":
        """This source's shared reader, made on first use.

        Position adapters receive it from the source-level adapter (via
        :meth:`get_tensor_adapter`); one constructed directly -- a test, a
        benchmark -- makes its own.
        """
        if self._shared_handle is None:
            self._shared_handle = _Nd2Reader(self._url, self._io_lock)
        return self._shared_handle

    def _position_for_field(self, field: Optional[str]) -> int:
        if not field or field == self.source_id:
            return 0
        prefix = f"{_POSITION_AXIS}:"
        if field.startswith(prefix):
            try:
                position = int(field[len(prefix) :])
            except ValueError:
                position = -1
            if 0 <= position < self._layout.n_positions:
                return position
        raise TensorNotFound(f"Unknown position: {field}", reason="unknown_field")

    def _native_block(self, labels: List[str], shape: List[int]) -> List[int]:
        """One full C/Y/X[/S] frame: ``read_frame`` addresses T/Z, not those."""
        return [
            1 if label.upper() in {"T", "Z"} else int(size)
            for label, size in zip(labels, shape, strict=True)
        ]

    def _transfer_chunk_shape(
        self, labels: List[str], shape: List[int], dtype: str
    ) -> List[int]:
        """Never split an ND2's component axes -- they are inside the pixel.

        Mirrors ``NikonAdapter._transfer_chunk_shape`` (biopb/biopb#806): a
        frame is materialised as one interleaved block, ``(Y, X, channel, RGB
        component)``, so C and S sit *below* X and a per-channel chunk would
        fault in every page the other components occupy.
        """
        upper_labels = [str(label).upper() for label in labels]
        if len(upper_labels) != len(shape):
            return default_transfer_chunk_shape(shape, dtype, labels)
        unit = [
            int(size) if label in {"C", "S"} else 1
            for label, size in zip(upper_labels, shape, strict=True)
        ]
        if math.prod(unit) <= 1:
            return default_transfer_chunk_shape(
                shape, dtype, labels, native=self._native_block(labels, shape)
            )
        if estimate_chunk_bytes(tuple(unit), dtype) >= (
            chunk_policy.PREFERRED_ARROW_BATCH_BYTES
        ):
            return unit
        return list(
            compute_transfer_chunk_size(tuple(unit), tuple(shape), dtype, upper_labels)
        )

    # ---- reads --------------------------------------------------------------

    @property
    def read_block_shape(self) -> Optional[Tuple[int, ...]]:
        """None: ``read_frame`` returns an mmap view, then this adapter crops."""
        return None

    def get_data(self, bounds: ChunkBounds) -> np.ndarray:
        return self._read(bounds, (1,) * len(bounds.start))

    def get_decimated_data(
        self, bounds: ChunkBounds, step: Tuple[int, ...]
    ) -> Optional[np.ndarray]:
        """The step selects frames on T/Z and strides the mmap view inside one."""
        return self._read(bounds, step)

    def _read(self, bounds: ChunkBounds, step: Tuple[int, ...]) -> np.ndarray:
        if self.position is None:
            raise ValueError("Cannot get data from source-level adapter")

        super().get_data(bounds)  # validate bounds against the descriptor
        labels = [label.upper() for label in self.dim_labels]
        starts = tuple(int(value) for value in bounds.start)
        stops = tuple(int(value) for value in bounds.stop)
        steps = tuple(max(1, int(size)) for size in step)
        output = np.empty(
            tuple(
                len(range(start, stop, size))
                for start, stop, size in zip(starts, stops, steps, strict=True)
            ),
            dtype=self._layout.dtype,
        )

        sequence_axes = [
            axis for axis, label in enumerate(labels) if label in {"T", "Z"}
        ]
        sequence_ranges = [
            range(starts[axis], stops[axis], steps[axis]) for axis in sequence_axes
        ]

        handle = self._reader_handle()
        with self._io_lock:
            reader = handle.acquire()
            try:
                for coordinates in product(*sequence_ranges):
                    coordinate_by_axis = dict(
                        zip(sequence_axes, coordinates, strict=True)
                    )
                    coordinate_by_label = {
                        labels[axis]: coordinate
                        for axis, coordinate in coordinate_by_axis.items()
                    }
                    key = _loop_key(coordinate_by_label, self._present_loop_axes)
                    frame_index = self._frame_indices[key]
                    frame = reader.read_frame(frame_index).reshape(self._frame_shape)

                    source_slices = []
                    output_slices = []
                    for axis, label in enumerate(labels):
                        if label in {"T", "Z"}:
                            index = (coordinate_by_axis[axis] - starts[axis]) // steps[
                                axis
                            ]
                            source_slices.append(slice(0, 1))
                            output_slices.append(slice(index, index + 1))
                        else:
                            source_slices.append(
                                slice(starts[axis], stops[axis], steps[axis])
                            )
                            output_slices.append(slice(None))
                    output[tuple(output_slices)] = frame[tuple(source_slices)]
            except Exception:
                # A half-open reader is not reusable; drop it so the next
                # read (on this or any sibling position adapter) reopens
                # rather than failing on the same handle.
                handle._release_persistent_handle()
                raise
        return output

    def close(self) -> None:
        """Release this source's reader, including its position adapters'.

        They share one, so closing it here closes it for all of them; the
        next read on any of them reopens.
        """
        for adapter in list(self._tensor_adapters.values()):
            adapter.close()
        if self._shared_handle is not None:
            with self._io_lock:
                self._shared_handle._release_persistent_handle()

    # ---- metadata -------------------------------------------------------------

    def _physical_scale(self) -> Optional[Tuple[List[float], List[str]]]:
        if self.position is None:
            return None
        return scale_by_label(self.dim_labels, self._layout.voxel_um, MICRON)

    def get_metadata(self) -> dict:
        return dict(self._layout.ome_summary)


__all__ = ["Nd2Adapter"]
