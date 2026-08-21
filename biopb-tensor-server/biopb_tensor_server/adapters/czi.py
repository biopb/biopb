"""Native pylibCZIrw adapter for Zeiss CZI files.

BioIO reads a CZI through ``bioio-czi``'s Dask array, whose graph carries one
task per plane.  Two costs follow, neither proportional to the bytes asked for:
every read re-optimizes the whole graph (so cost tracks the *file's* plane
count, not the request), and the block is always a whole plane
(``chunk_shape = shape[-2:]``), so a tile request materializes the plane and
discards most of it.  A 256x256 tile out of a 4096x4096 plane costs 31.4 ms
that way against 1.09 ms through ``pylibCZIrw.read(roi=...)``; see
``docs/dask-bypass-benchmarks.md``.

This adapter reads through libCZI directly: ``read(plane=..., scene=...,
roi=...)`` decodes only the subblocks the requested region covers.

**Scope.**  Every local CZI, whatever its acquisition dimensions.  Grayscale
and RGB (``Bgr*``) pixel types both read natively, and an axis outside T/C/Z --
phase, view, illumination, rotation, block -- is carried as its own descriptor
axis under its own name rather than being folded onto a biological one: the
normalization contract only requires that Y/X (and a samples axis) come last,
so any other label is legal wherever it sits.  A remote CZI is declined by
:meth:`CziAdapter.claim`, so it falls to BioIO's ``ZeissAdapter`` through the
registry -- this module imports nothing from BioIO and has no fallback path of
its own.  The one document it refuses is one no reader can represent as a
single tensor: pixel types that differ across channels.

**Handle policy.**  The reader is kept warm between reads and closed by the
shared idle reaper, the same opt-in :mod:`_handle_reaper` describes for
OME-TIFF.  Opening a CZI parses its subblock directory, so the open costs about
0.22 us per subblock on top of a 0.03 ms floor -- never negligible against a
0.1-2 ms ROI read.  Reopening per read (the hdf5/mrc default) measured 1.7x
slower at 40 subblocks and 3.6x at 1 000, so this format does not meet that
default's "the reopen is unmeasurable" precondition.
"""

import logging
import threading
import time
from dataclasses import dataclass
from itertools import product
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
from biopb.tensor.descriptor_pb2 import TensorDescriptor
from biopb.tensor.ticket_pb2 import ChunkBounds

from biopb_tensor_server.adapters._handle_reaper import (
    DEFAULT_HANDLE_REAPER_TTL,
    IdleHandleReaper,
)
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
from biopb_tensor_server.core.errors import TensorNotFound

if TYPE_CHECKING:
    from biopb_tensor_server.core.config import SourceConfig
    from biopb_tensor_server.core.discovery import DiscoveryState

logger = logging.getLogger(__name__)

CZI_EXTENSION = ".czi"

# Descriptor axis order.  libCZI addresses a plane by dimension name and hands
# back Y/X (plus samples for RGB), so the order below is this adapter's own
# convention; T/C/Z/Y/X matches what BioIO reports for the same documents.
_PLANE_DIMS = ("T", "C", "Z")
_SPATIAL_DIMS = ("Y", "X")
_SAMPLES_DIM = "S"

# libCZI's other acquisition dimensions, outermost first (the reverse of its
# own dimension-index order, where Z is innermost).  They are addressed exactly
# like T/C/Z and keep their own names in the descriptor, ahead of the canonical
# axes -- the same shape the native TIFF adapter uses for position/mosaic axes.
_EXTRA_DIMS = ("B", "V", "H", "I", "R")

# libCZI pixel types, mapped to (numpy dtype string, samples per pixel).  A
# Bgr* read comes back as (Y, X, 3) in the file's own channel order -- BioIO
# reports the identical values, so neither reader reorders them.
_PIXEL_TYPES = {
    "Gray8": ("|u1", 1),
    "Gray16": ("<u2", 1),
    "Gray32Float": ("<f4", 1),
    "Bgr24": ("|u1", 3),
    "Bgr48": ("<u2", 3),
    "Bgr96Float": ("<f4", 3),
}

# One pool for CZI readers, separate from the OME-TIFF store pool so each is
# retuned and reported on its own.  See :mod:`_handle_reaper`.
_reader_reaper = IdleHandleReaper(DEFAULT_HANDLE_REAPER_TTL, "czi-reader-reaper")


@dataclass(frozen=True)
class _CziScene:
    """One CZI scene: its index and its bounding rectangle in CZI coordinates."""

    index: int
    x: int
    y: int
    width: int
    height: int


@dataclass(frozen=True)
class _CziLayout:
    """What one probe of a CZI document tells the adapter.

    Built once per source (at registration) and handed to every scene adapter,
    so a read never re-derives it.
    """

    scenes: Tuple[_CziScene, ...]
    #: Non-spatial axes in descriptor order: the document's extra acquisition
    #: dimensions (if any) followed by T, C, Z.
    plane_axes: Tuple[str, ...]
    plane_sizes: Dict[str, int]
    dtype: str
    #: Samples per pixel: 1 for grayscale, 3 for an RGB (``Bgr*``) document.
    samples: int
    scale_um: Dict[str, Optional[float]]
    #: The metadata document's Information subtree only -- the rest is hardware
    #: settings that can run to megabytes and nothing here reads them.
    information: Dict[str, Any]


def _plane_axes(
    bounding_box: Dict[str, Tuple[int, int]],
) -> Tuple[Tuple[str, ...], Dict[str, int]]:
    """Split a bounding box into the non-spatial axes and their sizes.

    ``total_bounding_box`` reports every acquisition dimension the document
    uses, X and Y included.  T, C and Z are always described (size 1 when
    absent) so an ordinary document keeps the T/C/Z/Y/X shape BioIO reports.
    Any other dimension -- phase, view, illumination, rotation, block -- is
    carried under its own name ahead of them, and only when it actually varies:
    libCZI's own default plane coordinates include a dimension only at size > 1,
    and ``read()`` ignores a key for a dimension the document does not vary.

    Scene and mosaic axes never appear in this box: scenes are addressed by the
    ``scene`` argument and mosaic tiles are composed by ``read()``.

    The extents are **counts, not index ranges** -- a document whose T
    subblocks sit at 5..7 reports ``T: (0, 3)``, and its readable indices are
    still 5..7.  So the start conveys nothing, and a non-zero-based document is
    not detectable here; ``read()`` raises "Coordinate for dimension 'T' is
    out-of-range" on the first read.  Finding out earlier would cost an
    O(subblocks) enumeration and change nothing: no reader here can serve it.
    """
    sizes = {
        axis: int(extent[1]) - int(extent[0])
        for axis, extent in bounding_box.items()
        if axis not in _SPATIAL_DIMS
    }
    extra = tuple(axis for axis in _EXTRA_DIMS if sizes.get(axis, 1) > 1)
    axes = extra + _PLANE_DIMS
    return axes, {axis: max(1, int(sizes.get(axis, 1))) for axis in axes}


def _scaling_um(metadata: Dict[str, Any]) -> Dict[str, Optional[float]]:
    """Read X/Y/Z voxel sizes (CZI stores metres) as micrometres."""
    values: Dict[str, Optional[float]] = {"x": None, "y": None, "z": None}
    try:
        items = metadata["ImageDocument"]["Metadata"]["Scaling"]["Items"]["Distance"]
    except (KeyError, TypeError):
        return values
    if isinstance(items, dict):
        items = [items]
    for item in items:
        try:
            axis = str(item.get("@Id", "")).lower()
            size = float(item.get("Value", 0.0)) * 1e6
        except (AttributeError, TypeError, ValueError):
            continue
        if axis in values and size > 0:
            values[axis] = size
    return values


def _image_information(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """The Information subtree, which carries the acquisition summary.

    The full CZI metadata document can be megabytes of hardware settings; the
    catalog wants the part describing the image.
    """
    try:
        information = metadata["ImageDocument"]["Metadata"]["Information"]
    except (KeyError, TypeError):
        return {}
    return dict(information) if isinstance(information, dict) else {}


def read_layout(path: str) -> _CziLayout:
    """Probe a CZI document and describe what it takes to read it.

    Raises for a document that cannot be represented as one tensor at all --
    channels of differing pixel type, or a scene with no pixels.  There is no
    "outside the subset" return: every other local CZI is served here.
    """
    from pylibCZIrw import czi as pyczi

    with pyczi.open_czi(path) as czi:
        pixel_types = set(czi.pixel_types.values())
        if len(pixel_types) != 1:
            raise ValueError(
                f"CZI {path} mixes pixel types across channels "
                f"({sorted(czi.pixel_types.values())}); that has no single "
                "tensor dtype"
            )
        pixel_type = pixel_types.pop()
        if pixel_type not in _PIXEL_TYPES:
            raise ValueError(f"CZI {path} has unsupported pixel type {pixel_type!r}")
        dtype, samples = _PIXEL_TYPES[pixel_type]

        plane_axes, plane_sizes = _plane_axes(czi.total_bounding_box)

        rectangles = czi.scenes_bounding_rectangle
        if rectangles:
            scenes = tuple(
                _CziScene(index, rect.x, rect.y, rect.w, rect.h)
                for index, rect in sorted(rectangles.items())
            )
        else:
            rect = czi.total_bounding_rectangle
            scenes = (_CziScene(0, rect.x, rect.y, rect.w, rect.h),)
        if any(scene.width <= 0 or scene.height <= 0 for scene in scenes):
            # Defensive: pylibCZIrw's writer cannot produce this, so it means a
            # malformed document rather than a layout to describe.
            raise ValueError(f"CZI {path} has a scene with an empty rectangle")

        metadata = czi.metadata

    return _CziLayout(
        scenes=scenes,
        plane_axes=plane_axes,
        plane_sizes=plane_sizes,
        dtype=dtype,
        samples=samples,
        scale_um=_scaling_um(metadata),
        information=_image_information(metadata),
    )


class CziAdapter(TensorAdapter):
    """Reads Zeiss CZI scenes through libCZI, one tensor per scene."""

    SOURCE_TYPE = "czi"

    @classmethod
    def claim(cls, ctx: ClaimContext, state: "DiscoveryState") -> Optional[SourceClaim]:
        """Claim a local CZI by extension alone, without reading its content.

        Definite even under a cloud root and even for a dehydrated placeholder:
        the claim reads nothing, so it cannot trigger a recall, and deferring a
        non-resident file is the source manager's job
        (``_claim_is_unresolved``), not this one's. Guarding on residency here
        would also outlive the placeholder -- the resolve-time re-claim still
        carries ``cloud_root=True``, so a hydrated file would never reach this
        adapter.

        Whether libCZI can serve the document is decided at construction, where
        an unsupported layout falls back to BioIO.
        """
        if not ctx.is_file() or ctx.is_remote:
            return None
        if not ctx.name.lower().endswith(CZI_EXTENSION):
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
    ) -> TensorAdapter:
        """Create a native adapter for a local CZI.

        A remote CZI never reaches here through discovery -- ``claim`` declines
        it, so BioIO's ``ZeissAdapter`` takes it from the registry -- and an
        explicitly configured remote url is refused rather than silently
        rerouted.  A probe that raises propagates: ``bioio-czi`` reads through
        this same pylibCZIrw by default, so a file libCZI cannot open would not
        be readable by way of a fallback either.
        """
        if source.is_remote:
            raise ValueError(f"{cls.__name__} only supports local files")

        # ``file://`` counts as local (see ``is_remote_url``), but libCZI takes a
        # filesystem path.
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
        layout: _CziLayout,
        dim_labels: Optional[List[str]] = None,
        scene_position: Optional[int] = None,
        io_lock: Optional[threading.Lock] = None,
    ):
        self.source_id = source_id
        self._url = url
        self._layout = layout
        self._source_url = url
        self._source_type = self.SOURCE_TYPE
        self._content_version = content_version_from_path(url)
        # Position in ``layout.scenes``, not the scene's own CZI index -- the two
        # coincide for a well-formed document but the read uses ``scene.index``.
        self.scene_position = scene_position
        native_labels = list(layout.plane_axes) + list(_SPATIAL_DIMS)
        if layout.samples > 1:
            native_labels.append(_SAMPLES_DIM)
        if dim_labels and len(dim_labels) != len(native_labels):
            logger.warning(
                "czi: ignoring %d configured dim_labels for %s -- this document "
                "reads as a %d-axis %s array",
                len(dim_labels),
                url,
                len(native_labels),
                "".join(native_labels),
            )
            dim_labels = None
        self.dim_labels = list(dim_labels or native_labels)

        # One lock per source, shared with its scene adapters: libCZI's reader
        # is not documented as thread-safe, and it also fences a reaper close
        # against an in-flight read.
        self._io_lock = io_lock if io_lock is not None else threading.Lock()
        self._tensor_adapters: Dict[str, CziAdapter] = {}
        self._persistent_reader = None
        self._persistent_context = None
        self._persistent_last_access = 0.0
        # Reads hold ``_io_lock`` end to end, so no read is ever in flight when
        # the reaper can take the lock; the counter exists for the protocol.
        self._active_reads = 0

    # ---- descriptors --------------------------------------------------------

    def _scene(self) -> _CziScene:
        """The scene this adapter is bound to (the first, at source level)."""
        position = self.scene_position
        return self._layout.scenes[0 if position is None else position]

    def _descriptor_for(self, position: int) -> TensorDescriptor:
        scene = self._layout.scenes[position]
        layout = self._layout
        trailing = [scene.height, scene.width]
        if layout.samples > 1:
            trailing.append(layout.samples)
        shape = [layout.plane_sizes[axis] for axis in layout.plane_axes] + trailing
        return TensorDescriptor(
            array_id=f"{self.source_id}/Scene:{scene.index}",
            dim_labels=list(self.dim_labels),
            # libCZI decodes per subblock, and a subblock never spans planes,
            # so one plane is the alignment seed; the transfer grid grows whole
            # planes from it (biopb/biopb#809).
            chunk_shape=default_transfer_chunk_shape(
                shape,
                layout.dtype,
                self.dim_labels,
                native=[1] * len(layout.plane_axes) + trailing,
            ),
            shape=shape,
            dtype=layout.dtype,
        )

    def list_tensor_descriptors(self) -> List[TensorDescriptor]:
        # Structural entries only: every scene shares one layout here, so the
        # grid would be right -- but the catalog is not where a grid is
        # published, whoever could compute it (biopb/biopb#812).
        return [
            catalog_entry(self._descriptor_for(position))
            for position in range(len(self._layout.scenes))
        ]

    def get_tensor_descriptor(self) -> TensorDescriptor:
        return self._descriptor_for(
            0 if self.scene_position is None else self.scene_position
        )

    def get_tensor_adapter(self, tensor_id: Optional[str]) -> "CziAdapter":
        """Return the adapter bound to one scene of this source."""
        field = self._within_source_field(tensor_id)
        position = self._position_for_field(field)
        cached = self._tensor_adapters.get(field)
        if cached is not None:
            return cached

        adapter = self.__class__(
            self._url,
            self.source_id,
            self._layout,
            dim_labels=self.dim_labels,
            scene_position=position,
            io_lock=self._io_lock,
        )
        adapter._tensor_name = field
        self._tensor_adapters[field] = adapter
        return adapter

    def _position_for_field(self, field: Optional[str]) -> int:
        """Resolve a within-source field to its position in ``layout.scenes``.

        The position is what indexes the descriptor list; the scene's own CZI
        index (what ``read`` takes) may differ, and lives on the scene record.
        """
        if not field or field == self.source_id:
            return 0
        for position, scene in enumerate(self._layout.scenes):
            if field == f"Scene:{scene.index}":
                return position
        raise TensorNotFound(f"Unknown scene: {field}", reason="unknown_field")

    # ---- reads --------------------------------------------------------------

    def get_data(self, bounds: ChunkBounds) -> np.ndarray:
        """Read the requested region, one libCZI read per plane coordinate."""
        if self.scene_position is None:
            raise ValueError("Cannot get data from source-level adapter")

        super().get_data(bounds)  # validate bounds against the descriptor
        scene = self._scene()
        layout = self._layout
        starts = [int(value) for value in bounds.start]
        stops = [int(value) for value in bounds.stop]

        # Positions, not labels: a configured dim_labels renames the axes but
        # never reorders the array this adapter builds. The plane axes come
        # first, then Y and X, then samples for an RGB document.
        n_plane = len(layout.plane_axes)
        y0, x0 = starts[n_plane], starts[n_plane + 1]
        y1, x1 = stops[n_plane], stops[n_plane + 1]
        roi = (scene.x + x0, scene.y + y0, x1 - x0, y1 - y0)
        if layout.samples > 1:
            sample_slice = slice(starts[n_plane + 2], stops[n_plane + 2])
        else:
            # Grayscale reads come back as (Y, X, 1); drop that trailing axis.
            sample_slice = 0

        output = np.empty(
            tuple(stop - start for start, stop in zip(starts, stops, strict=True)),
            dtype=np.dtype(layout.dtype),
        )
        coordinate_ranges = [
            range(starts[axis], stops[axis]) for axis in range(n_plane)
        ]
        with self._io_lock:
            reader = self._acquire_reader()
            try:
                for coordinates in product(*coordinate_ranges):
                    plane = reader.read(
                        plane=dict(zip(layout.plane_axes, coordinates, strict=True)),
                        scene=scene.index,
                        roi=roi,
                    )
                    destination = tuple(
                        coordinate - starts[axis]
                        for axis, coordinate in enumerate(coordinates)
                    )
                    output[destination] = plane[..., sample_slice]
                self._persistent_last_access = time.monotonic()
            except Exception:
                # A half-open reader is not reusable; drop it so the next read
                # reopens rather than failing on the same handle.
                self._release_persistent_handle()
                raise
        return output

    def _acquire_reader(self):
        """Open (or reuse) a libCZI reader.  Caller holds ``_io_lock``."""
        if self._persistent_reader is not None:
            return self._persistent_reader

        from pylibCZIrw import czi as pyczi

        context = pyczi.open_czi(self._url)
        reader = context.__enter__()
        self._persistent_context = context
        self._persistent_reader = reader
        self._persistent_last_access = time.monotonic()
        _reader_reaper.register(self)
        return reader

    def _release_persistent_handle(self) -> None:
        """Close the reader and permit a later reopen.

        The :class:`~biopb_tensor_server.adapters._handle_reaper.ReapableHandle`
        hook.  Caller holds ``_io_lock`` (read path / reaper) or is the GC
        finalizer.  Safe to call repeatedly.
        """
        context = self._persistent_context
        self._persistent_reader = None
        self._persistent_context = None
        _reader_reaper.discard(self)
        if context is not None:
            try:
                context.__exit__(None, None, None)
            except Exception:
                logger.debug("error closing persistent CZI reader", exc_info=True)

    def close(self) -> None:
        """Release this source's readers, including its scene adapters'."""
        for adapter in list(self._tensor_adapters.values()):
            adapter.close()
        with self._io_lock:
            self._release_persistent_handle()

    def __del__(self):
        try:
            self._release_persistent_handle()
        except Exception:
            pass

    # ---- metadata -----------------------------------------------------------

    def _physical_scale(self) -> Optional[Tuple[List[float], List[str]]]:
        """Voxel size per descriptor axis, from the document's Scaling items."""
        return scale_by_label(self.dim_labels, self._layout.scale_um, MICRON)

    def get_metadata(self) -> dict:
        """The CZI Information subtree: sizes, channels, acquisition summary."""
        return dict(self._layout.information)


__all__ = ["CziAdapter"]
