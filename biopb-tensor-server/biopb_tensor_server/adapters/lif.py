"""Native ``readlif`` adapter for Leica LIF files.

BioIO reads a LIF through ``bioio-lif``, whose Dask array pays the same
graph-rebuild-per-read cost every other bioio-backed format does (see
``docs/dask-bypass-benchmarks.md``, biopb/biopb#799 phase 3). Unlike CZI or
plain TIFF, ``readlif`` offers no region-of-interest read: ``LifImage.get_frame``
always decodes one whole plane, and it reopens the file for every call rather
than holding a handle (``LifImage._get_item``) -- there is no I/O shape to win
back here, only the graph-construction overhead BioIO's Dask array adds on
top of that same per-plane reopen. That is why this is the lowest-payoff
format in the phase-3 set.

**Scope.**  Every image in a local LIF container. A LIF file can hold several
images (Leica's project tree), each becoming its own tensor -- the same
dual-role shape :class:`~biopb_tensor_server.adapters.czi.CziAdapter` uses for
CZI scenes. Unlike a CZI scene, though, a LIF image carries its own
independent dimensions and channel count, so there is no shared "layout" to
factor out: each image's shape is read from its own ``image_list`` entry.

A remote LIF is declined by :meth:`LifAdapter.claim`, so it falls to BioIO's
``LeicaAdapter`` through the registry -- this module imports nothing from
BioIO and has no fallback path of its own.
"""

import logging
from dataclasses import dataclass
from itertools import product
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
from biopb.tensor.descriptor_pb2 import TensorDescriptor
from biopb.tensor.ticket_pb2 import ChunkBounds

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

LIF_EXTENSION = ".lif"

# Descriptor axis order: mosaic tile (if the image has more than one), then
# the canonical T/C/Z/Y/X readlif itself always reports (size 1 where absent).
_PLANE_DIMS = ("T", "C", "Z")
_SPATIAL_DIMS = ("Y", "X")
_MOSAIC_DIM = "M"


@dataclass(frozen=True)
class _LifLayout:
    """What one probe of a LIF container tells the adapter.

    Built once per source (at registration): the parsed image list plus the
    byte offsets ``readlif.reader.LifFile`` located for each image's pixel
    block. ``LifImage`` construction from these is pure field assignment (no
    I/O), so every read rebuilds it cheaply rather than this adapter holding a
    live ``LifFile``/``LifImage`` handle -- there is nothing stateful to hold:
    ``readlif`` itself reopens the file per plane (see the module docstring).
    """

    filename: str
    image_list: Tuple[Dict[str, Any], ...]
    offsets: Tuple[Tuple[int, int], ...]


def read_layout(path: str) -> _LifLayout:
    """Parse a LIF container's XML header. Raises for a file readlif rejects."""
    from readlif.reader import LifFile

    lif = LifFile(path)
    return _LifLayout(
        filename=path,
        image_list=tuple(lif.image_list),
        offsets=tuple(lif.offsets),
    )


def _lif_image(layout: _LifLayout, position: int):
    """Build the ``readlif.reader.LifImage`` for one entry. No I/O."""
    from readlif.reader import LifImage

    return LifImage(
        layout.image_list[position], layout.offsets[position], layout.filename
    )


def _native_labels(info: Dict[str, Any]) -> Tuple[str, ...]:
    dims = info["dims"]
    labels = _PLANE_DIMS + _SPATIAL_DIMS
    return ((_MOSAIC_DIM,) + labels) if int(dims.m) > 1 else labels


def _native_shape(info: Dict[str, Any]) -> Tuple[int, ...]:
    dims = info["dims"]
    sizes = {
        "T": int(dims.t),
        "C": int(info["channels"]),
        "Z": int(dims.z),
        "Y": int(dims.y),
        "X": int(dims.x),
        _MOSAIC_DIM: int(dims.m),
    }
    return tuple(sizes[axis] for axis in _native_labels(info))


def _dtype_for(info: Dict[str, Any]) -> np.dtype:
    # readlif itself only ever looks at bit_depth[0] to pick the frame's
    # storage type (LifImage._get_item), so this adapter matches that rather
    # than validating every channel agrees.
    bit_depth = info["bit_depth"]
    depth = bit_depth[0] if bit_depth else 8
    return np.dtype(np.uint8 if depth == 8 else np.uint16)


class LifAdapter(TensorAdapter):
    """Reads Leica LIF images through readlif, one tensor per image."""

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
            dim_labels=source.dim_labels,
        )

    def __init__(
        self,
        url: str,
        source_id: str,
        layout: _LifLayout,
        dim_labels: Optional[List[str]] = None,
        image_position: Optional[int] = None,
    ):
        self.source_id = source_id
        self._url = url
        self._layout = layout
        self._source_url = url
        self._source_type = self.SOURCE_TYPE
        self._content_version = content_version_from_path(url)
        # Position in ``layout.image_list``, distinct from the image's own
        # (possibly slash-nested) name -- the read uses the position directly.
        self.image_position = image_position

        if image_position is None:
            self.dim_labels = dim_labels
        else:
            info = self._layout.image_list[image_position]
            native_labels = list(_native_labels(info))
            if dim_labels and len(dim_labels) != len(native_labels):
                logger.warning(
                    "lif: ignoring %d configured dim_labels for %s -- this "
                    "image reads as a %d-axis %s array",
                    len(dim_labels),
                    url,
                    len(native_labels),
                    "".join(native_labels),
                )
                dim_labels = None
            self.dim_labels = list(dim_labels or native_labels)

        self._tensor_adapters: Dict[str, LifAdapter] = {}

    # ---- descriptors --------------------------------------------------------

    def _descriptor_for(self, position: int) -> TensorDescriptor:
        info = self._layout.image_list[position]
        labels = list(_native_labels(info))
        shape = list(_native_shape(info))
        dtype = _dtype_for(info).str
        return TensorDescriptor(
            array_id=f"{self.source_id}/Image:{position}",
            dim_labels=labels,
            # readlif has no ROI: a read always decodes the whole plane
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

    def list_tensor_descriptors(self) -> List[TensorDescriptor]:
        return [
            catalog_entry(self._descriptor_for(position))
            for position in range(len(self._layout.image_list))
        ]

    def get_tensor_descriptor(self) -> TensorDescriptor:
        if self.image_position is not None:
            return self._descriptor_for(self.image_position)
        entries = self.list_tensor_descriptors()
        if not entries:
            raise TensorNotFound(
                f"source {self.source_id!r} exposes no images",
                reason="unknown_source",
            )
        return self.get_tensor_adapter(entries[0].array_id).get_tensor_descriptor()

    def get_tensor_adapter(self, tensor_id: Optional[str]) -> "LifAdapter":
        field = self._within_source_field(tensor_id)
        position = self._position_for_field(field)
        cached = self._tensor_adapters.get(field)
        if cached is not None:
            return cached

        adapter = self.__class__(
            self._url,
            self.source_id,
            self._layout,
            dim_labels=self.dim_labels if self.image_position is None else None,
            image_position=position,
        )
        adapter._tensor_name = field
        self._tensor_adapters[field] = adapter
        return adapter

    def _position_for_field(self, field: Optional[str]) -> int:
        if not field or field == self.source_id:
            return 0
        for position in range(len(self._layout.image_list)):
            if field == f"Image:{position}":
                return position
        raise TensorNotFound(f"Unknown image: {field}", reason="unknown_field")

    # ---- reads --------------------------------------------------------------

    @property
    def read_block_shape(self) -> Optional[Tuple[int, ...]]:
        """One whole plane: readlif has no ROI, ``get_frame`` reads it whole.

        Full rank, matching the ``native=`` seed in :meth:`_descriptor_for` --
        :func:`~.stream_reduce.streaming_unit` zips this positionally against
        the transfer grid, so a Y/X-only tuple would floor the wrong axes.
        """
        if self.image_position is None:
            return None
        shape = _native_shape(self._layout.image_list[self.image_position])
        return tuple([1] * (len(shape) - 2)) + tuple(shape[-2:])

    def get_data(self, bounds: ChunkBounds) -> np.ndarray:
        return self._read(bounds, step=None)

    def get_decimated_data(
        self, bounds: ChunkBounds, step: Tuple[int, ...]
    ) -> Optional[np.ndarray]:
        """Skip whole planes the stride would drop on T/C/Z/M; Y/X still crop
        a fully-decoded plane (readlif has no partial-plane read)."""
        return self._read(bounds, step=step)

    def _read(self, bounds: ChunkBounds, step: Optional[Tuple[int, ...]]) -> np.ndarray:
        if self.image_position is None:
            raise ValueError("Cannot get data from source-level adapter")

        super().get_data(bounds)  # validate bounds against the descriptor
        info = self._layout.image_list[self.image_position]
        labels = [label.upper() for label in self.dim_labels]
        starts = [int(value) for value in bounds.start]
        stops = [int(value) for value in bounds.stop]
        steps = [1] * len(labels) if step is None else [max(1, int(s)) for s in step]

        n_plane = len(labels) - 2  # every axis but the trailing Y, X
        y0, x0 = starts[n_plane], starts[n_plane + 1]
        y1, x1 = stops[n_plane], stops[n_plane + 1]
        y_step, x_step = steps[n_plane], steps[n_plane + 1]

        image = _lif_image(self._layout, self.image_position)
        plane_ranges = [
            range(starts[axis], stops[axis], steps[axis]) for axis in range(n_plane)
        ]
        out_shape = tuple(
            len(range(starts[axis], stops[axis], steps[axis]))
            for axis in range(len(labels))
        )
        output = np.empty(out_shape, dtype=_dtype_for(info))

        for coordinates in product(*plane_ranges):
            by_label = dict(zip(labels[:n_plane], coordinates, strict=True))
            frame = image.get_frame(
                z=by_label.get("Z", 0),
                t=by_label.get("T", 0),
                c=by_label.get("C", 0),
                m=by_label.get(_MOSAIC_DIM, 0),
            )
            plane = np.asarray(frame)[y0:y1:y_step, x0:x1:x_step]
            destination = tuple(
                (coordinate - starts[axis]) // steps[axis]
                for axis, coordinate in enumerate(coordinates)
            )
            output[destination] = plane
        return output

    # ---- metadata -------------------------------------------------------------

    def _physical_scale(self) -> Optional[Tuple[List[float], List[str]]]:
        """Voxel size in micrometres, from readlif's pixels-per-micron scale."""
        if self.image_position is None:
            return None
        scale_x, scale_y, scale_z, _scale_t = self._layout.image_list[
            self.image_position
        ]["scale"]
        values = {
            "x": (1.0 / scale_x) if scale_x else None,
            "y": (1.0 / scale_y) if scale_y else None,
            "z": (1.0 / scale_z) if scale_z else None,
        }
        return scale_by_label(self.dim_labels, values, MICRON)

    def get_metadata(self) -> dict:
        """This image's readlif-parsed acquisition summary."""
        if self.image_position is None:
            return {"format": "lif", "images": len(self._layout.image_list)}
        info = self._layout.image_list[self.image_position]
        return {
            "format": "lif",
            "name": info.get("name"),
            "channels": info.get("channels"),
            "bit_depth": list(info.get("bit_depth") or ()),
            "settings": dict(info.get("settings") or {}),
        }


__all__ = ["LifAdapter"]
