"""Native tifffile adapters for plain TIFF and Zeiss LSM files.

The BioIO tifffile reader builds one Dask task per TIFF page.  That is a poor
fit for local files with many pages: a small read pays the graph construction
and optimization cost before any pixels are decoded.  These adapters keep the
same tifffile ``aszarr`` read path as :class:`OmeTiffAdapter`, but open the
store once and slice it directly.

OME-TIFF remains owned by ``OmeTiffAdapter``.  TIFF sequences and
Micro-Manager datasets remain owned by their directory-level adapters.
"""

import logging
from typing import TYPE_CHECKING, Any, List, Optional, Tuple

import numpy as np
from biopb.tensor.descriptor_pb2 import TensorDescriptor

from biopb_tensor_server.adapters._scale import MICRON, scale_by_label, unit_to_um
from biopb_tensor_server.adapters.ome_tiff import OmeTiffAdapter
from biopb_tensor_server.adapters.tiff import _tiff_pixel_size_um
from biopb_tensor_server.core.chunk import default_transfer_chunk_shape
from biopb_tensor_server.core.discovery import ClaimContext, SourceClaim

if TYPE_CHECKING:
    from biopb_tensor_server.core.discovery import DiscoveryState

logger = logging.getLogger(__name__)

_PERSISTENT_PAGE_THRESHOLD = 16_384
_CANONICAL_DIMS = ("T", "C", "Z", "Y", "X")
_SAMPLES_DIM = "S"
_SUPPORTED_EXTENSIONS = (".tif", ".tiff")


def _mapped_axes(native_axes: str, shape: Tuple[int, ...]) -> Optional[str]:
    """Map tifffile axes onto the canonical descriptor vocabulary.

    Plain TIFF has no required dimension metadata.  tifffile calls otherwise
    unlabeled sequence axes ``Q`` (and some readers use ``I``).  BioIO's
    tifffile reader maps those axes from the right of ``TCZ`` for grayscale
    data, while an RGB samples axis uses the leading ``TC`` slots. Named ``P``
    and ``M`` axes represent position and mosaic layouts; preserve those labels
    rather than relabeling them as biological dimensions or handing the file to
    BioIO. If there are more unknown axes than canonical slots, preserve the
    native labels instead of declining a readable array.

    The return value has one label per native array axis, in native order.  It
    is used only for indexing; no pixel transpose is done here.
    """
    axes = str(native_axes or "").upper()
    if len(axes) != len(shape) or axes.count("Y") != 1 or axes.count("X") != 1:
        return None

    mapped: List[Optional[str]] = [None] * len(axes)
    used = set()
    unknown = []
    for index, axis in enumerate(axes):
        if axis in {"P", "M"}:
            if axis in used:
                return None
            mapped[index] = axis
            used.add(axis)
            continue
        if axis in _CANONICAL_DIMS or axis == _SAMPLES_DIM:
            if axis in used:
                return None
            mapped[index] = axis
            used.add(axis)
        else:
            unknown.append(index)

    if unknown:
        if _SAMPLES_DIM in used:
            # BioIO treats QSYX / QQSYX as C/YX and TC/YX respectively;
            # additional unknown axes, if any, occupy Z after those.
            preferred = ["C"] if len(unknown) == 1 else ["T", "C", "Z"][: len(unknown)]
            targets = [axis for axis in preferred if axis not in used]
            targets.extend(
                axis
                for axis in ("T", "C", "Z")
                if axis not in used and axis not in targets
            )
            targets = targets[: len(unknown)]
        else:
            # QYX / QQYX / QQQYX map to Z / CZ / TCZ.
            targets = [axis for axis in ("T", "C", "Z") if axis not in used]
            targets = targets[-len(unknown) :]
        if len(targets) != len(unknown):
            # The normalization contract permits unknown leading labels. Keep
            # the native order for e.g. QQQQYX rather than treating a readable
            # array as a fallback-only format. The positional read path below
            # handles repeated unknown labels safely.
            return axes
        for index, axis in zip(unknown, targets, strict=True):
            mapped[index] = axis
            used.add(axis)

    if any(axis is None for axis in mapped):
        return None
    return "".join(mapped)  # type: ignore[arg-type]


class _TifffileAdapterBase(OmeTiffAdapter):
    """Shared source/scene adapter for local plain TIFF-like files."""

    _LSM = False

    def __init__(self, *args, dim_labels=None, **kwargs):
        # Empty is not an override: labels that came from an unset protobuf
        # field would otherwise fail the rank check below for every series.
        self._dim_labels_override = bool(dim_labels)
        super().__init__(*args, dim_labels=dim_labels, **kwargs)
        if self.scene_index is None and self._tifffile_descriptor is None:
            message = (
                f"{self.__class__.__name__} cannot read TIFF source "
                f"{self._source_url!r}"
            )
            try:
                descriptors = self._tifffile_descriptors()
            except OSError:
                raise
            except Exception as exc:
                raise ValueError(message) from exc
            if not descriptors:
                raise ValueError(message)
            self._cached_descriptors = descriptors

    @classmethod
    def create_from_config(cls, source, credentials_config=None):
        """Create a native adapter for a local TIFF-like file."""
        if source.is_remote:
            raise ValueError(f"{cls.__name__} only supports local files")
        return cls(str(source.url), source.source_id, dim_labels=source.dim_labels)

    @classmethod
    def claim(cls, ctx: ClaimContext, state: "DiscoveryState") -> Optional[SourceClaim]:
        """Claim a local TIFF or LSM by extension alone, without reading it.

        Claim ownership is determined by registry priority and source family.
        Native construction validates the file and raises for malformed or
        otherwise unreadable resident files; those files cannot be made usable
        by the lower-priority BioIO fallback anyway.

        Definite even under a cloud root and even for a dehydrated placeholder:
        the claim reads nothing, so it cannot trigger a recall, and deferring a
        non-resident file is the source manager's job
        (``_claim_is_unresolved``), not this one's. Guarding on residency here
        would also outlive the placeholder -- the resolve-time re-claim still
        carries ``cloud_root=True``, so a hydrated file would never reach this
        adapter.
        """
        if not ctx.is_file() or ctx.is_remote:
            return None

        name = ctx.name.lower()
        if cls._LSM:
            if not name.endswith(".lsm"):
                return None
        elif not name.endswith(_SUPPORTED_EXTENSIONS):
            return None

        state.try_claim_path(ctx.path_str)
        return SourceClaim(
            source_type=cls.SOURCE_TYPE,
            primary_path=ctx.path_str,
            is_remote=False,
        )

    def _series_indices(self, tiff: Any) -> List[int]:
        """Return the series exposed as tensors by this adapter."""
        # tifffile creates a reduced LSM thumbnail as a second series.  It is a
        # thumbnail, not a pyramid level: its absolute dimensions vary by file
        # and it has no stable scale relationship to the full-resolution image.
        return [0] if self._LSM else list(range(len(tiff.series)))

    def _series_descriptor(
        self, series: Any, series_index: int
    ) -> Optional[TensorDescriptor]:
        native_axes = str(series.axes).upper()
        mapped = _mapped_axes(native_axes, tuple(series.shape))
        if mapped is None:
            return None

        # The native store has one full plane per page.  Keep that native unit
        # in the read plan, including an interleaved RGB(A) samples axis.
        if self._dim_labels_override:
            if len(self.dim_labels) != len(mapped):
                return None
            labels = list(self.dim_labels)
            shape = [int(size) for size in series.shape]
        elif len(set(mapped)) != len(mapped):
            # Unknown axes may repeat (QQQQYX). They are valid descriptor labels,
            # but cannot be resolved through label-to-index dictionaries without
            # collapsing positions. Keep the native rank/order and use the
            # positional store reader below.
            labels = list(mapped)
            shape = [int(size) for size in series.shape]
        else:
            named_axes = [
                axis
                for axis in mapped
                if axis not in _CANONICAL_DIMS and axis != _SAMPLES_DIM
            ]
            labels = named_axes + list(_CANONICAL_DIMS)
            if _SAMPLES_DIM in mapped:
                labels.append(_SAMPLES_DIM)
            by_axis = {
                axis: int(series.shape[index]) for index, axis in enumerate(mapped)
            }
            shape = [by_axis.get(axis, 1) for axis in labels]
        # One whole page (full Y/X and the RGB samples axis) is the read path's
        # native unit; it seeds the transfer grid rather than being it, so a
        # small plane ships several planes per chunk (biopb/biopb#809).
        page = [
            size if str(axis).upper() in {"Y", "X", "S"} else 1
            for axis, size in zip(labels, shape, strict=True)
        ]
        dtype = np.dtype(series.dtype).str
        return TensorDescriptor(
            array_id=f"{self.source_id}/Image:{series_index}",
            dim_labels=labels,
            shape=shape,
            chunk_shape=default_transfer_chunk_shape(shape, dtype, labels, native=page),
            dtype=dtype,
        )

    def _tifffile_descriptors(self) -> Optional[List[TensorDescriptor]]:
        """Build descriptors from tifffile without constructing a Dask graph.

        Configured labels are applied positionally to the native series shape;
        the normalization layer owns any subsequent axis reordering.
        """
        url = self._source_url or ""
        if "://" in url and not url.startswith("file://"):
            return None
        path = url[len("file://") :] if url.startswith("file://") else url
        if not path:
            return None

        import tifffile

        with tifffile.TiffFile(path) as tiff:
            descriptors = []
            for index in self._series_indices(tiff):
                descriptor = self._series_descriptor(tiff.series[index], index)
                if descriptor is None:
                    return None
                descriptors.append(descriptor)
            return descriptors or None

    def get_tensor_adapter(self, tensor_id: str) -> "_TifffileAdapterBase":
        """Create a scene adapter of the same native type."""
        descriptors = self.list_tensor_descriptors()
        field = self._within_source_field(tensor_id)
        scene_index = self._scene_index_for_field(field)
        if field in self._tensor_adapters:
            return self._tensor_adapters[field]

        adapter = self.__class__(
            self._source_url,
            self.source_id,
            scene_index=scene_index,
            tensor_descriptor=descriptors[scene_index],
            dim_labels=self.dim_labels,
            io_lock=self._io_lock,
        )
        adapter._tensor_name = field
        self._tensor_adapters[field] = adapter
        return adapter

    def _open_store(self):
        """Open the selected tifffile series as a persistent page store."""
        import tifffile
        import zarr

        url = self._source_url or ""
        if "://" in url and not url.startswith("file://"):
            return None
        path = url[len("file://") :] if url.startswith("file://") else url
        if not path:
            return None

        series_index = self.scene_index or 0
        tiff = tifffile.TiffFile(path)
        try:
            self._tifffile_page_count = len(tiff.pages)
            series = tiff.series[series_index]
            store = series.aszarr(level=0, chunkmode="page")
            zarr_array = zarr.open(store, mode="r")
            axes = _mapped_axes(str(series.axes), tuple(zarr_array.shape))
            if axes is None:
                raise ValueError("unsupported tifffile axis layout")

            descriptor = self._tifffile_descriptor
            if self._dim_labels_override or len(set(axes)) != len(axes):
                canonical = tuple(int(size) for size in zarr_array.shape)
            else:
                by_axis = {
                    axis: int(zarr_array.shape[index])
                    for index, axis in enumerate(axes)
                }
                canonical = tuple(
                    by_axis.get(axis, 1) for axis in descriptor.dim_labels
                )
            if (
                canonical != tuple(descriptor.shape)
                or zarr_array.dtype.str != descriptor.dtype
            ):
                for obj in (store, tiff):
                    try:
                        obj.close()
                    except Exception:
                        logger.debug(
                            "error closing rejected tifffile store", exc_info=True
                        )
                return None
        except Exception:
            tiff.close()
            raise

        self._persistent_tiff = tiff
        self._persistent_store = store
        return zarr_array, axes

    def _should_persist_store(self) -> bool:
        """Keep a native file handle only when the TIFF has many pages."""
        return getattr(self, "_tifffile_page_count", 0) >= _PERSISTENT_PAGE_THRESHOLD

    def _read_region(self, za, axes, slices):
        """Read configured-label stores in their declared native order."""
        if self._dim_labels_override or len(set(axes)) != len(axes):
            return np.asarray(za[slices])
        return super()._read_region(za, axes, slices)

    def _physical_scale(self):
        """Return TIFF resolution or LSM voxel calibration in micrometres."""
        url = self._source_url or ""
        if "://" in url and not url.startswith("file://"):
            return None
        path = url[len("file://") :] if url.startswith("file://") else url
        if not path:
            return None

        try:
            labels = self.dim_labels or list(self.get_tensor_descriptor().dim_labels)
            import tifffile

            with tifffile.TiffFile(path) as tiff:
                page = tiff.pages[0]
                if self._LSM and tiff.lsm_metadata:
                    metadata = tiff.lsm_metadata
                    values = {
                        "x": float(metadata.get("VoxelSizeX", 0.0)) * 1e6,
                        "y": float(metadata.get("VoxelSizeY", 0.0)) * 1e6,
                        "z": float(metadata.get("VoxelSizeZ", 0.0)) * 1e6,
                    }
                else:
                    imagej = tiff.imagej_metadata or {}
                    imagej_unit_um = unit_to_um(imagej.get("unit"))
                    values = {
                        "x": _tiff_pixel_size_um(page, "XResolution", imagej_unit_um),
                        "y": _tiff_pixel_size_um(page, "YResolution", imagej_unit_um),
                        "z": None,
                    }
                    spacing = imagej.get("spacing")
                    if spacing is not None and imagej_unit_um is not None:
                        values["z"] = float(spacing) * imagej_unit_um
            return scale_by_label(labels, values, MICRON)
        except Exception:
            logger.debug(
                "tifffile physical scale unavailable for %s",
                self._source_url,
                exc_info=True,
            )
            return None

    def get_metadata(self) -> dict:
        """Return lightweight metadata exposed by the native TIFF reader.

        ImageJ and LSM metadata take precedence. Plain TIFFs commonly carry
        tifffile's JSON-shaped metadata (for example, the stored shape) in
        the image description; use that when ImageJ metadata is absent.
        """
        url = self._source_url or ""
        if "://" in url and not url.startswith("file://"):
            return {}
        path = url[len("file://") :] if url.startswith("file://") else url
        if not path:
            return {}
        try:
            import tifffile

            with tifffile.TiffFile(path) as tiff:
                if self._LSM:
                    return dict(tiff.lsm_metadata or {})
                imagej = tiff.imagej_metadata or {}
                if imagej:
                    return dict(imagej)
                for metadata in tiff.shaped_metadata or ():
                    if metadata:
                        return dict(metadata)
                return {}
        except Exception:
            return {}


class TiffAdapter(_TifffileAdapterBase):
    """Native adapter for local non-OME ``.tif`` / ``.tiff`` files."""

    SOURCE_TYPE = "tiff"


class LsmAdapter(_TifffileAdapterBase):
    """Native adapter for Zeiss ``.lsm`` files (full-resolution series only)."""

    SOURCE_TYPE = "lsm"
    _LSM = True


__all__ = ["TiffAdapter", "LsmAdapter"]
