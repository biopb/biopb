"""Native ``liffile`` adapter for Leica LIF files.

BioIO reads a LIF through ``bioio-lif``, whose Dask array pays the same
graph-rebuild-per-read cost every other bioio-backed format does
(biopb/biopb#799 phase 3). ``liffile`` reads one frame (a Y/X plane) at a time
and has no region-of-interest read, so every read decodes whole planes; the
win over BioIO is only the graph-construction overhead it adds on top.

**Scope.**  Every image in a local LIF container. A LIF file can hold several
images (Leica's project tree), each becoming its own tensor -- the same
dual-role shape :class:`~biopb_tensor_server.adapters.czi.CziAdapter` uses for
CZI scenes. A LIF image carries its own independent dimensions and channel
count, so there is no shared "layout" to factor out: each image is described by
its own :class:`_LifImage`. An image with a dimension this adapter has no axis
for (RGB samples, FLIM histograms, ...) fails the parse, as a file ``liffile``
rejects does -- there is no fallback to BioIO for a local ``.lif``.

The adapter holds no file handle: the header is parsed once into the layout,
and each read reopens the container (one header parse per read, not per plane).

A remote LIF is declined by :meth:`LifAdapter.claim`, so it falls to BioIO's
``LeicaAdapter`` through the registry -- this module imports nothing from
BioIO and has no fallback path of its own.
"""

import json
import logging
from dataclasses import asdict, dataclass
from itertools import product
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
from biopb.tensor.descriptor_pb2 import TensorDescriptor
from biopb.tensor.ticket_pb2 import ChunkBounds

from biopb_tensor_server.adapters._scale import MICRON, scale_by_label
from biopb_tensor_server.core.adapter_base import (
    TensorAdapter,
    TensorEntry,
    catalog_entry,
    strip_source_prefix,
)
from biopb_tensor_server.core.chunk import (
    content_version_from_path,
    default_transfer_chunk_shape,
)
from biopb_tensor_server.core.discovery import ClaimContext, SourceClaim
from biopb_tensor_server.core.errors import TensorNotFound
from biopb_tensor_server.core.normalize import canonical_axes
from biopb_tensor_server.core.registration import (
    RegistrationRecord,
    metadata_record,
)

if TYPE_CHECKING:
    from biopb_tensor_server.core.config import SourceConfig
    from biopb_tensor_server.core.discovery import DiscoveryState

logger = logging.getLogger(__name__)

LIF_EXTENSION = ".lif"

# Descriptor axis order: mosaic tile (if the image has more than one), then
# T/C/Z/Y/X (size 1 where the image has none).
_PLANE_DIMS = ("T", "C", "Z")
_SPATIAL_DIMS = ("Y", "X")
_MOSAIC_DIM = "M"
_KNOWN_DIMS = frozenset(_PLANE_DIMS + _SPATIAL_DIMS + (_MOSAIC_DIM,))
_SETTINGS_PATH = "./Data/Image/Attachment/ATLConfocalSettingDefinition"
_METRES_TO_MICRONS = 1e6


@dataclass(frozen=True)
class _LifImage:
    """One image of a LIF container, as the adapter needs it.

    ``labels`` and ``shape`` are the descriptor's; ``present`` are the
    ``liffile`` dimension codes the file really has, the only ones ``frame``
    accepts as indices.
    """

    name: str
    labels: Tuple[str, ...]
    shape: Tuple[int, ...]
    dtype: str
    present: Tuple[str, ...]
    # Voxel size in micrometres, None where the image has a single element.
    voxel_x: Optional[float]
    voxel_y: Optional[float]
    voxel_z: Optional[float]
    bit_depth: Tuple[int, ...]
    settings: Dict[str, str]


@dataclass(frozen=True)
class _LifLayout:
    """What one probe of a LIF container tells the adapter.

    Built once per source (at registration); every read reopens the file.
    """

    filename: str
    images: Tuple[_LifImage, ...]

    def to_payload(self) -> Dict[str, Any]:
        """The layout as JSON.

        Raises ``TypeError`` for a value JSON cannot carry (the caller then
        stores no payload and the file is parsed on a restart).
        """
        payload = {"images": [asdict(image) for image in self.images]}
        return json.loads(json.dumps(payload))

    @classmethod
    def from_payload(cls, payload: Dict[str, Any], filename: str) -> "_LifLayout":
        """Rebuild a layout from :meth:`to_payload`; *filename* is where the file is now."""
        images = []
        for raw in payload["images"]:
            raw = dict(raw)
            for key in ("labels", "shape", "present", "bit_depth"):
                raw[key] = tuple(raw[key])
            images.append(_LifImage(**raw))
        return cls(filename=filename, images=tuple(images))


def _voxel_um(image: Any, label: str) -> Optional[float]:
    """Spacing of axis *label* in micrometres; ``None`` for a single element."""
    coords = image.coords.get(label)
    if coords is None or len(coords) < 2:
        return None
    step = abs(float(coords[1]) - float(coords[0])) * _METRES_TO_MICRONS
    return step or None


def _describe(image: Any) -> _LifImage:
    sizes = dict(image.sizes)
    extra = {
        label: n for label, n in sizes.items() if label not in _KNOWN_DIMS and n > 1
    }
    if extra:
        raise ValueError(
            f"LIF image {image.path!r} has dimensions with no tensor axis: {extra}"
        )
    if tuple(image.frames.frame_dims) != _SPATIAL_DIMS:
        raise ValueError(
            f"LIF image {image.path!r} stores frames as "
            f"{tuple(image.frames.frame_dims)}, not Y/X planes"
        )
    labels = _PLANE_DIMS + _SPATIAL_DIMS
    if sizes.get(_MOSAIC_DIM, 1) > 1:
        labels = (_MOSAIC_DIM,) + labels
    settings = image.xml_element.find(_SETTINGS_PATH)
    channels = getattr(image, "_channels", ())
    return _LifImage(
        name=image.path,
        labels=labels,
        shape=tuple(int(sizes.get(label, 1)) for label in labels),
        dtype=np.dtype(image.dtype).str,
        present=tuple(label for label in labels if label in sizes),
        voxel_x=_voxel_um(image, "X"),
        voxel_y=_voxel_um(image, "Y"),
        voxel_z=_voxel_um(image, "Z"),
        bit_depth=tuple(int(c.resolution) for c in channels)
        or (np.dtype(image.dtype).itemsize * 8,),
        settings=dict(settings.attrib) if settings is not None else {},
    )


def read_layout(path: str) -> _LifLayout:
    """Parse a LIF container's XML header. Raises for a file ``liffile`` rejects."""
    import liffile

    with liffile.LifFile(path, squeeze=False) as lif:
        return _LifLayout(
            filename=path, images=tuple(_describe(image) for image in lif.images)
        )


@canonical_axes
class LifAdapter(TensorAdapter):
    """Reads Leica LIF images through liffile, one tensor per image."""

    SOURCE_TYPE = "lif"

    @classmethod
    def claim(cls, ctx: ClaimContext, state: "DiscoveryState") -> Optional[SourceClaim]:
        """Claim a local LIF by extension alone, without reading its content.

        Same shape as :class:`~biopb_tensor_server.adapters.czi.CziAdapter`:
        definite even under a cloud root / dehydrated placeholder, remote is
        declined so BioIO's ``LeicaAdapter`` can serve it instead.
        """
        if not ctx.is_file() or ctx.is_remote:
            return None
        if not ctx.name.lower().endswith(LIF_EXTENSION):
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
    ) -> "LifAdapter":
        """Create a native adapter for a local LIF file."""
        if source.is_remote:
            raise ValueError(f"{cls.__name__} only supports local files")

        url = str(source.url)
        path = url[len("file://") :] if url.startswith("file://") else url

        return cls(
            path,
            source.source_id,
            layout=read_layout(path),
        )

    @classmethod
    def create_from_payload(
        cls,
        source: "SourceConfig",
        payload: Dict[str, Any],
        metadata: Dict[str, Any],
        credentials_config: Optional[Any] = None,
    ) -> "LifAdapter":
        """Rebuild from the row's parsed image list: no file is opened."""
        url = str(source.url)
        path = url[len("file://") :] if url.startswith("file://") else url
        return cls(
            path,
            source.source_id,
            layout=_LifLayout.from_payload(payload["layout"], path),
        )

    def catalog_payload(self) -> Optional[Dict[str, Any]]:
        """The parsed images. Source-level only; ``None`` when they hold something
        JSON cannot carry back faithfully."""
        if self.image_position is not None:
            return None
        try:
            return {"layout": self._layout.to_payload()}
        except TypeError:
            logger.debug("LIF layout is not storable as a payload", exc_info=True)
            return None

    def __init__(
        self,
        url: str,
        source_id: str,
        layout: _LifLayout,
        image_position: Optional[int] = None,
    ):
        self.source_id = source_id
        self._url = url
        self._layout = layout
        self._source_url = url
        self._source_type = self.SOURCE_TYPE
        self._content_version = content_version_from_path(url)
        # Position in ``layout.images``, distinct from the image's own
        # (possibly slash-nested) name -- the read uses the position directly.
        self.image_position = image_position

        if image_position is None:
            self.dim_labels = None
        else:
            self.dim_labels = list(self._layout.images[image_position].labels)

        self._tensor_adapters: Dict[str, LifAdapter] = {}

    # ---- descriptors --------------------------------------------------------

    def _descriptor_for(self, position: int) -> TensorDescriptor:
        info = self._layout.images[position]
        labels = list(info.labels)
        shape = list(info.shape)
        dtype = info.dtype
        return TensorDescriptor(
            array_id=f"{self.source_id}/Image:{position}",
            dim_labels=labels,
            # liffile has no ROI: a read always decodes the whole plane
            # (biopb/biopb#799), so the native unit is one full Y/X plane.
            chunk_shape=default_transfer_chunk_shape(
                shape,
                dtype,
                labels,
                native=[1] * (len(labels) - 2) + shape[-2:],
            ),
            shape=shape,
            dtype=dtype,
        )

    def list_tensors(self) -> List[TensorEntry]:
        return self._native_entries()

    def _native_entries(self) -> List[TensorEntry]:
        return [
            catalog_entry(self._descriptor_for(position))
            for position in range(len(self._layout.images))
        ]

    def _native_descriptor(self) -> TensorDescriptor:
        if self.image_position is not None:
            return self._descriptor_for(self.image_position)
        entries = self._native_entries()
        if not entries:
            raise TensorNotFound(
                f"source {self.source_id!r} exposes no images",
                reason="unknown_source",
            )
        return self.get_tensor_adapter(entries[0].array_id)._native_descriptor()

    def get_tensor_adapter(self, tensor_id: Optional[str]) -> "LifAdapter":
        field = strip_source_prefix(self.source_id, tensor_id)
        position = self._position_for_field(field)
        cached = self._tensor_adapters.get(field)
        if cached is not None:
            return cached

        adapter = self.__class__(
            self._url,
            self.source_id,
            self._layout,
            image_position=position,
        )
        adapter._tensor_name = field
        self._tensor_adapters[field] = adapter
        return adapter

    def _position_for_field(self, field: Optional[str]) -> int:
        if not field or field == self.source_id:
            return 0
        for position in range(len(self._layout.images)):
            if field == f"Image:{position}":
                return position
        raise TensorNotFound(f"Unknown image: {field}", reason="unknown_field")

    # ---- reads --------------------------------------------------------------

    @property
    def read_block_shape(self) -> Optional[Tuple[int, ...]]:
        """One whole plane: liffile has no ROI, ``frame`` reads it whole.

        Full rank, matching the ``native=`` seed in :meth:`_descriptor_for` --
        :func:`~.stream_reduce.streaming_unit` zips this positionally against
        the transfer grid, so a Y/X-only tuple would floor the wrong axes.
        """
        if self.image_position is None:
            return None
        shape = self._layout.images[self.image_position].shape
        return tuple([1] * (len(shape) - 2)) + tuple(shape[-2:])

    def get_data(self, bounds: ChunkBounds) -> np.ndarray:
        return self._read(bounds, step=None)

    def get_decimated_data(
        self, bounds: ChunkBounds, step: Tuple[int, ...]
    ) -> Optional[np.ndarray]:
        """Skip whole planes the stride would drop on T/C/Z/M; Y/X still crop
        a fully-decoded plane (liffile has no partial-plane read)."""
        return self._read(bounds, step=step)

    def _read(self, bounds: ChunkBounds, step: Optional[Tuple[int, ...]]) -> np.ndarray:
        if self.image_position is None:
            raise ValueError("Cannot get data from source-level adapter")

        super().get_data(bounds)  # validate bounds against the descriptor
        info = self._layout.images[self.image_position]
        labels = [label.upper() for label in self.dim_labels]
        starts = [int(value) for value in bounds.start]
        stops = [int(value) for value in bounds.stop]
        steps = [1] * len(labels) if step is None else [max(1, int(s)) for s in step]

        n_plane = len(labels) - 2  # every axis but the trailing Y, X
        y0, x0 = starts[n_plane], starts[n_plane + 1]
        y1, x1 = stops[n_plane], stops[n_plane + 1]
        y_step, x_step = steps[n_plane], steps[n_plane + 1]

        plane_ranges = [
            range(starts[axis], stops[axis], steps[axis]) for axis in range(n_plane)
        ]
        out_shape = tuple(
            len(range(starts[axis], stops[axis], steps[axis]))
            for axis in range(len(labels))
        )
        output = np.empty(out_shape, dtype=np.dtype(info.dtype))

        import liffile

        with liffile.LifFile(self._layout.filename, squeeze=False) as lif:
            image = lif.images[self.image_position]
            for coordinates in product(*plane_ranges):
                # An axis the file lacks is a size-1 label here; frame() knows
                # only the codes the file has.
                indices = {
                    label: coordinate
                    for label, coordinate in zip(
                        labels[:n_plane], coordinates, strict=True
                    )
                    if label in info.present
                }
                plane = image.frame(**indices)[y0:y1:y_step, x0:x1:x_step]
                destination = tuple(
                    (coordinate - starts[axis]) // steps[axis]
                    for axis, coordinate in enumerate(coordinates)
                )
                output[destination] = plane
        return output

    # ---- metadata -------------------------------------------------------------

    def _physical_scale(self) -> Optional[Tuple[List[float], List[str]]]:
        """Voxel size in micrometres, from liffile's axis coordinates."""
        if self.image_position is None:
            return None
        info = self._layout.images[self.image_position]
        values = {"x": info.voxel_x, "y": info.voxel_y, "z": info.voxel_z}
        return scale_by_label(self.dim_labels, values, MICRON)

    def registration_record(
        self, tensors, *, import_rois=True, max_rois_per_tensor=None
    ) -> RegistrationRecord:
        """This image's acquisition summary."""
        if self.image_position is None:
            return metadata_record(
                {"format": "lif", "images": len(self._layout.images)}
            )
        info = self._layout.images[self.image_position]
        return metadata_record(
            {
                "format": "lif",
                "name": info.name,
                "channels": info.shape[info.labels.index("C")],
                "bit_depth": list(info.bit_depth),
                "settings": dict(info.settings),
            }
        )


__all__ = ["LifAdapter"]
