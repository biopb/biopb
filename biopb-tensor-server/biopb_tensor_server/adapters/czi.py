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

**Supported subset.**  Grayscale pixel types, and no acquisition dimension
outside T/C/Z carrying more than one index.  Anything else -- an RGB (``Bgr*``)
document, a phase/view/illumination axis, a remote URL -- is handed back to
:class:`~biopb_tensor_server.adapters.bioio.ZeissAdapter` by
:meth:`CziAdapter.create_from_config`, so every CZI stays readable and only the
subset this reader can serve faithfully takes the fast path.  Mosaics need no
special case: libCZI composes their tiles behind ``read()``, which is also what
BioIO's ``reconstruct_mosaic`` returns, and the scene's bounding rectangle is
already the composed extent.

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
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
from biopb.tensor.descriptor_pb2 import TensorDescriptor
from biopb.tensor.ticket_pb2 import ChunkBounds

from biopb_tensor_server.adapters._handle_reaper import (
    DEFAULT_HANDLE_REAPER_TTL,
    IdleHandleReaper,
)
from biopb_tensor_server.adapters._scale import MICRON, scale_by_label
from biopb_tensor_server.core.adapter_base import TensorAdapter
from biopb_tensor_server.core.chunk import content_version_from_path
from biopb_tensor_server.core.discovery import ClaimContext, SourceClaim
from biopb_tensor_server.core.errors import TensorNotFound

if TYPE_CHECKING:
    from biopb_tensor_server.core.config import SourceConfig
    from biopb_tensor_server.core.discovery import DiscoveryState

logger = logging.getLogger(__name__)

CZI_EXTENSION = ".czi"

# Descriptor axis order.  libCZI addresses a plane by name and returns it as
# Y/X, so this order is the adapter's own convention rather than the file's;
# it matches what BioIO reported for the same documents.
_PLANE_DIMS = ("T", "C", "Z")
_CANONICAL_DIMS = ("T", "C", "Z", "Y", "X")

# libCZI pixel types this adapter serves, mapped to their numpy dtype string.
# Bgr* types are deliberately absent -- see the module docstring.
_PIXEL_TYPES = {
    "Gray8": "|u1",
    "Gray16": "<u2",
    "Gray32Float": "<f4",
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
    plane_sizes: Dict[str, int]
    dtype: str
    scale_um: Dict[str, Optional[float]]
    #: The metadata document's Information subtree only -- the rest is hardware
    #: settings that can run to megabytes and nothing here reads them.
    information: Dict[str, Any]


def _plane_sizes(bounding_box: Dict[str, Tuple[int, int]]) -> Optional[Dict[str, int]]:
    """Extract T/C/Z sizes, or None when an axis this reader cannot address varies.

    ``total_bounding_box`` reports every acquisition dimension the document
    uses, X and Y included.  Anything outside T/C/Z (phase, view, illumination,
    rotation, block) is left at index 0 by ``read()``, so a document that varies
    one is not fully addressable here and belongs to the BioIO fallback.  Scene
    and mosaic axes never appear in this box -- scenes are addressed by the
    ``scene`` argument and mosaic tiles are composed by ``read()``.
    """
    sizes = {}
    for axis, extent in bounding_box.items():
        start, stop = int(extent[0]), int(extent[1])
        if axis in ("X", "Y"):
            continue
        size = stop - start
        if axis not in _PLANE_DIMS:
            if size > 1:
                return None
            continue
        if start != 0:
            # read() takes an absolute index; a box that does not start at 0
            # would make descriptor position and file index disagree.
            return None
        sizes[axis] = size
    return {axis: int(sizes.get(axis, 1)) for axis in _PLANE_DIMS}


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


def read_layout(path: str) -> Optional[_CziLayout]:
    """Probe a CZI, or return None when it is outside the supported subset."""
    from pylibCZIrw import czi as pyczi

    with pyczi.open_czi(path) as czi:
        pixel_types = set(czi.pixel_types.values())
        if len(pixel_types) != 1:
            # A per-channel pixel type has no single descriptor dtype.
            return None
        dtype = _PIXEL_TYPES.get(pixel_types.pop())
        if dtype is None:
            return None

        plane_sizes = _plane_sizes(czi.total_bounding_box)
        if plane_sizes is None:
            return None

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
            return None

        metadata = czi.metadata

    return _CziLayout(
        scenes=scenes,
        plane_sizes=plane_sizes,
        dtype=dtype,
        scale_um=_scaling_um(metadata),
        information=_image_information(metadata),
    )


def _bioio_fallback(
    source: "SourceConfig",
    credentials_config: Optional[Any],
    reason: str,
) -> TensorAdapter:
    """Serve a CZI this reader declines through BioIO instead."""
    try:
        from biopb_tensor_server.adapters.bioio import ZeissAdapter
    except ImportError as exc:
        raise ValueError(
            f"CZI {source.url} needs the BioIO fallback ({reason}) "
            "but bioio-czi is not installed"
        ) from exc
    logger.info("czi: serving %s through BioIO (%s)", source.url, reason)
    return ZeissAdapter.create_from_config(source, credentials_config)


class CziAdapter(TensorAdapter):
    """Reads Zeiss CZI scenes through libCZI, one tensor per scene."""

    SOURCE_TYPE = "czi"

    @classmethod
    def claim(cls, ctx: ClaimContext, state: "DiscoveryState") -> Optional[SourceClaim]:
        """Claim a resident local CZI without reading its content.

        Whether libCZI can serve the document is decided at construction, where
        an unsupported layout falls back to BioIO -- so the claim stays a pure
        extension check and no scan recalls a cloud placeholder.
        """
        if not ctx.is_file() or ctx.is_remote or ctx.cloud_root:
            return None
        if not ctx.name.lower().endswith(CZI_EXTENSION):
            return None
        if not ctx.is_resident():
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
        """Build a native adapter, or BioIO's when this reader cannot serve the file.

        Returning the fallback from here rather than declining the claim keeps
        the decision on the one path that may read the file: discovery claims by
        extension alone, so the layout is not yet known when the claim is made.
        """
        if source.is_remote:
            return _bioio_fallback(source, credentials_config, "remote source")

        # ``file://`` counts as local (see ``is_remote_url``), but libCZI takes a
        # filesystem path.
        url = str(source.url)
        path = url[len("file://") :] if url.startswith("file://") else url

        try:
            layout = read_layout(path)
        except Exception:
            logger.debug("CZI layout probe failed for %s", url, exc_info=True)
            layout = None
        if layout is None:
            return _bioio_fallback(source, credentials_config, "unsupported layout")

        return cls(path, source.source_id, layout, dim_labels=source.dim_labels)

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
        if dim_labels and len(dim_labels) != len(_CANONICAL_DIMS):
            logger.warning(
                "czi: ignoring %d configured dim_labels for %s -- this reader "
                "always builds a %d-axis %s array",
                len(dim_labels),
                url,
                len(_CANONICAL_DIMS),
                "".join(_CANONICAL_DIMS),
            )
            dim_labels = None
        self.dim_labels = list(dim_labels or _CANONICAL_DIMS)

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
        sizes = self._layout.plane_sizes
        shape = [sizes["T"], sizes["C"], sizes["Z"], scene.height, scene.width]
        return TensorDescriptor(
            array_id=f"{self.source_id}/Scene:{scene.index}",
            dim_labels=list(self.dim_labels),
            # libCZI decodes per subblock, and a subblock never spans planes --
            # one plane is the unit a read can be planned around.
            chunk_shape=[1, 1, 1, scene.height, scene.width],
            shape=shape,
            dtype=self._layout.dtype,
        )

    def list_tensor_descriptors(self) -> List[TensorDescriptor]:
        return [
            self._descriptor_for(position)
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
        """Read the requested region, one libCZI plane read per T/C/Z index."""
        if self.scene_position is None:
            raise ValueError("Cannot get data from source-level adapter")

        super().get_data(bounds)  # validate bounds against the descriptor
        scene = self._scene()
        starts = [int(value) for value in bounds.start]
        stops = [int(value) for value in bounds.stop]
        # Descriptor positions, not labels: a configured dim_labels renames the
        # axes but never reorders the array this adapter builds.
        t0, c0, z0, y0, x0 = starts
        t1, c1, z1, y1, x1 = stops
        roi = (scene.x + x0, scene.y + y0, x1 - x0, y1 - y0)

        output = np.empty(
            (t1 - t0, c1 - c0, z1 - z0, y1 - y0, x1 - x0),
            dtype=np.dtype(self._layout.dtype),
        )
        with self._io_lock:
            reader = self._acquire_reader()
            try:
                for t in range(t0, t1):
                    for c in range(c0, c1):
                        for z in range(z0, z1):
                            plane = reader.read(
                                plane={"T": t, "C": c, "Z": z},
                                scene=scene.index,
                                roi=roi,
                            )
                            # Grayscale reads come back as (Y, X, 1).
                            output[t - t0, c - c0, z - z0] = plane[..., 0]
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
