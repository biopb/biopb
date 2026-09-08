"""Native ``nd2``-package adapter for Nikon ND2 files.

BioIO reads an ND2 through ``bioio-nd2``, whose Dask array pays the same
graph-rebuild-per-read cost every other bioio-backed format does (see
``docs/dask-bypass-benchmarks.md``). biopb/biopb#797 already bypassed that
graph for pixel reads by calling ``nd2.read_frame`` directly from inside
BioIO's ``NikonAdapter`` -- but scene enumeration, dimension labels and
metadata still went through BioIO's ``BioImage``/OME conversion. This module
finishes the migration (biopb/biopb#799 phase 3): every fact this adapter
reports, structural or physical, is read from the ``nd2`` package alone.

**Stage positions are a leading axis, not a scene split.** BioIO models each
XY stage position (nd2's ``P`` loop) as its own scene/tensor, matching its
general per-format convention. The ``nd2`` package does not: ``ND2File.sizes``
folds ``P`` in as an ordinary loop axis alongside ``T``/``Z``, because every
position in an ND2 experiment shares the same T/Z/C/Y/X shape -- there is
nothing scene-like (no independent bounding box, no per-position channel set)
to split apart the way a CZI scene or a LIF image would need to be. Reporting
one array with a leading ``P`` axis is both truer to the format and simpler
than re-deriving BioIO's split, and the normalization contract only requires
Y/X (and a samples axis) last -- any other axis order or extra label is legal
(see :mod:`~biopb_tensor_server.adapters.czi`). So this is a single-tensor
source, unlike ``NikonAdapter``.

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

if TYPE_CHECKING:
    from biopb_tensor_server.core.config import SourceConfig
    from biopb_tensor_server.core.discovery import DiscoveryState

logger = logging.getLogger(__name__)

ND2_EXTENSION = ".nd2"

# The loop axes nd2 addresses one frame at a time (see the module docstring).
# C and S (RGB) are baked into the frame ``read_frame`` returns, never looped.
_LOOP_AXES = ("P", "T", "Z")

# Same TTL and reasoning as bioio.py's ND2 reader pool: what a held reader
# saves is a warm page table over ``nd2.read_frame``'s mmap view, which is
# only worth as long as the read that built it -- so the TTL is seconds, not
# the long default every reopen-is-cheap format uses.
_READER_TTL = 5.0

_reader_reaper = IdleHandleReaper(_READER_TTL, "nd2-reader-reaper", max_handles=8)


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
    """Reads a Nikon ND2 file through the ``nd2`` package. Single-tensor source."""

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
            dim_labels=source.dim_labels,
        )

    def __init__(
        self,
        url: str,
        source_id: str,
        layout: _Nd2Layout,
        dim_labels: Optional[List[str]] = None,
    ):
        self.source_id = source_id
        self._url = url
        self._layout = layout
        self._source_url = url
        self._source_type = self.SOURCE_TYPE
        self._content_version = content_version_from_path(url)

        native_labels = list(layout.labels)
        if dim_labels and len(dim_labels) != len(native_labels):
            logger.warning(
                "nd2: ignoring %d configured dim_labels for %s -- this file "
                "reads as a %d-axis %s array",
                len(dim_labels),
                url,
                len(native_labels),
                "".join(native_labels),
            )
            dim_labels = None
        self.dim_labels = list(dim_labels or native_labels)
        self._present_loop_axes = tuple(
            axis for axis in _LOOP_AXES if axis in layout.labels
        )

        # One reader per source, held warm the same way CziAdapter holds its
        # libCZI reader -- ``nd2.read_frame`` returns a zero-copy view onto
        # the reader's mmap, so a reopen re-faults every page a crop touches
        # even when the bytes are already resident.
        self._io_lock = threading.Lock()
        self._persistent_reader = None
        self._persistent_last_access = 0.0
        self._active_reads = 0

    # ---- descriptors --------------------------------------------------------

    def get_tensor_descriptor(self) -> TensorDescriptor:
        shape = list(self._layout.shape)
        dtype = self._layout.dtype.str
        return TensorDescriptor(
            array_id=self.array_id,
            dim_labels=self.dim_labels,
            chunk_shape=self._transfer_chunk_shape(shape, dtype),
            shape=shape,
            dtype=dtype,
        )

    def list_tensor_descriptors(self) -> List[TensorDescriptor]:
        return [catalog_entry(self.get_tensor_descriptor())]

    def _native_block(self) -> List[int]:
        """One full C/Y/X[/S] frame: ``read_frame`` addresses P/T/Z, not those."""
        shape = self._layout.shape
        return [
            1 if label.upper() in {"P", "T", "Z"} else int(size)
            for label, size in zip(self.dim_labels, shape, strict=True)
        ]

    def _transfer_chunk_shape(self, shape: List[int], dtype: str) -> List[int]:
        """Never split an ND2's component axes -- they are inside the pixel.

        Mirrors ``NikonAdapter._transfer_chunk_shape`` (biopb/biopb#806): a
        frame is materialised as one interleaved block, ``(Y, X, channel, RGB
        component)``, so C and S sit *below* X and a per-channel chunk would
        fault in every page the other components occupy.
        """
        labels = [str(label).upper() for label in self.dim_labels]
        if len(labels) != len(shape):
            return default_transfer_chunk_shape(shape, dtype, self.dim_labels)
        unit = [
            int(size) if label in {"C", "S"} else 1
            for label, size in zip(labels, shape, strict=True)
        ]
        if math.prod(unit) <= 1:
            return default_transfer_chunk_shape(
                shape, dtype, self.dim_labels, native=self._native_block()
            )
        if estimate_chunk_bytes(tuple(unit), dtype) >= (
            chunk_policy.PREFERRED_ARROW_BATCH_BYTES
        ):
            return unit
        return list(
            compute_transfer_chunk_size(tuple(unit), tuple(shape), dtype, labels)
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
        """The step selects frames on P/T/Z and strides the mmap view inside one."""
        return self._read(bounds, step)

    def _read(self, bounds: ChunkBounds, step: Tuple[int, ...]) -> np.ndarray:
        super().get_data(bounds)  # validate bounds against the descriptor
        labels = [label.upper() for label in self.dim_labels]
        shape = tuple(int(size) for size in self._layout.shape)
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
            axis for axis, label in enumerate(labels) if label in {"P", "T", "Z"}
        ]
        sequence_ranges = [
            range(starts[axis], stops[axis], steps[axis]) for axis in sequence_axes
        ]
        frame_shape = tuple(
            1 if label in {"P", "T", "Z"} else size
            for label, size in zip(labels, shape, strict=True)
        )

        with self._io_lock:
            reader = self._acquire_reader()
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
                    frame_index = self._layout.frame_indices[key]
                    frame = reader.read_frame(frame_index).reshape(frame_shape)

                    source_slices = []
                    output_slices = []
                    for axis, label in enumerate(labels):
                        if label in {"P", "T", "Z"}:
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
                self._persistent_last_access = time.monotonic()
            except Exception:
                self._release_persistent_handle()
                raise
        return output

    def _acquire_reader(self):
        """Open (or reuse) the ND2 reader. Caller holds ``_io_lock``."""
        if self._persistent_reader is not None:
            return self._persistent_reader

        import nd2

        reader = nd2.ND2File(self._url)
        self._persistent_reader = reader
        self._persistent_last_access = time.monotonic()
        _reader_reaper.register(self)
        return reader

    def _release_persistent_handle(self) -> None:
        """Close the reader and permit a later reopen.

        The :class:`~biopb_tensor_server.adapters._handle_reaper.ReapableHandle`
        hook. Caller holds ``_io_lock`` (read path / reaper) or is the GC
        finalizer. Safe to call repeatedly.
        """
        reader, self._persistent_reader = self._persistent_reader, None
        _reader_reaper.discard(self)
        if reader is not None:
            try:
                reader.close()
            except Exception:
                logger.debug("error closing persistent ND2 reader", exc_info=True)

    def close(self) -> None:
        with self._io_lock:
            self._release_persistent_handle()

    def __del__(self):
        try:
            self._release_persistent_handle()
        except Exception:
            pass

    # ---- metadata -------------------------------------------------------------

    def _physical_scale(self) -> Optional[Tuple[List[float], List[str]]]:
        return scale_by_label(self.dim_labels, self._layout.voxel_um, MICRON)

    def get_metadata(self) -> dict:
        return dict(self._layout.ome_summary)


__all__ = ["Nd2Adapter"]
