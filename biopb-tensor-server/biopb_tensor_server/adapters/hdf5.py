"""HDF5 adapter for tensor storage.

Relies on OS page cache for raw data caching.

Reopen-per-read (biopb/biopb#71): the adapter snapshots the dataset's shape /
dtype / chunk grid / attrs at construction and keeps NO ``h5py.File`` open, so a
catalogued source never pins its file. Holding the handle for the lifetime of the
catalog entry made an ``.h5`` undeletable on Windows and kept its blocks
allocated after an unlink on POSIX. Reopening costs ~0.09 ms against a ~35 ms
64 MB chunk read (0.26%) and is O(1) in file size, so there is nothing to amortise
-- unlike OME-TIFF, whose open is linear in IFD count and therefore keeps a
persistent handle plus a reaper.
"""

from typing import TYPE_CHECKING, Any, List, Optional, Tuple

import numpy as np
from biopb.tensor.descriptor_pb2 import TensorDescriptor
from biopb.tensor.ticket_pb2 import ChunkBounds

from biopb_tensor_server.core.base import TensorAdapter
from biopb_tensor_server.core.discovery import ClaimContext, SourceClaim

if TYPE_CHECKING:
    from biopb_tensor_server.core.config import SourceConfig
    from biopb_tensor_server.core.discovery import DiscoveryState


class Hdf5Adapter(TensorAdapter):
    """Adapter for HDF5 chunked datasets.

    Chunk ID format:
    - array_id prefix
    - bounds encoding (start, stop coordinates)

    Relies on OS page cache for raw data caching.
    """

    @classmethod
    def claim(cls, ctx: ClaimContext, state: "DiscoveryState") -> Optional[SourceClaim]:
        """Claim HDF5 files (requires explicit dataset path in config).

        HDF5 files are detected but NOT auto-expanded to tensors because
        they require explicit configuration with dataset path. The claim
        signals this via extra_config['needs_dataset'] = True.

        Args:
            ctx: ClaimContext for unified filesystem access
            state: DiscoveryState with try_claim_path() callback

        Returns:
            SourceClaim with needs_dataset flag, None if not HDF5 file
        """
        if not ctx.is_file():
            return None

        name = ctx.name.lower()
        if not name.endswith((".h5", ".hdf5")):
            return None

        state.try_claim_path(ctx.path_str)

        # HDF5 files are claimed but marked as needing explicit dataset config
        # The discovery system will warn about these
        return SourceClaim(
            source_type="hdf5",
            primary_path=ctx.path_str,
            extra_config={"needs_dataset": True},
        )

    @classmethod
    def create_from_config(
        cls, source: "SourceConfig", credentials_config: Optional[Any] = None
    ) -> "Hdf5Adapter":
        """Create adapter instance from SourceConfig.

        Args:
            source: SourceConfig with url, source_id, dim_labels, dataset

        Returns:
            Hdf5Adapter instance

        Raises:
            ValueError: If dataset is not specified
        """
        import h5py

        if source.dataset is None:
            raise ValueError(
                f"HDF5 source '{source.source_id}' requires 'dataset' path in config"
            )

        with h5py.File(str(source.url), "r") as f:
            return cls(f[source.dataset], source.source_id, source.dim_labels)

    def __init__(
        self,
        h5_dataset,
        source_id: str,
        dim_labels: Optional[List[str]] = None,
    ):
        """Initialize HDF5 adapter.

        Everything the adapter needs is copied off *h5_dataset* here; the dataset
        (and the file behind it) is not retained, so the caller may close it
        immediately. Reads reopen the file (see the module docstring).

        Args:
            h5_dataset: h5py Dataset object
            source_id: Unique identifier for this data source
            dim_labels: Optional dimension labels
        """
        self.source_id = source_id
        self.dim_labels = dim_labels or [
            f"dim{i}" for i in range(len(h5_dataset.shape))
        ]

        self._path = h5_dataset.file.filename if hasattr(h5_dataset, "file") else ""
        self._dataset_path = h5_dataset.name
        self._shape = tuple(int(s) for s in h5_dataset.shape)
        self._dtype = np.dtype(h5_dataset.dtype)
        # A contiguous (unchunked) dataset reports chunks=None; the whole array
        # is then one chunk, which the base class splits if oversized.
        self._chunks = (
            tuple(int(c) for c in h5_dataset.chunks) if h5_dataset.chunks else None
        )
        # element_size_um is the only attribute anything reads; snapshot it so
        # _physical_scale needs no open file.
        attrs = getattr(h5_dataset, "attrs", None)
        self._element_size_um = (
            attrs["element_size_um"]
            if attrs is not None and "element_size_um" in attrs
            else None
        )

        # Source-level metadata for DataSourceDescriptor
        self._source_url = self._path
        self._source_type = "hdf5"

    def get_data(self, bounds: ChunkBounds) -> np.ndarray:
        """Read data within bounds from HDF5 dataset.

        Args:
            bounds: Chunk bounds (start, stop coordinates per axis)

        Returns:
            Numpy array with data within the requested bounds

        Raises:
            ValueError: If bounds exceed array shape
        """
        import h5py

        super().get_data(bounds)
        slices = tuple(
            slice(int(s), int(e))
            for s, e in zip(bounds.start, bounds.stop, strict=True)
        )
        # Each read gets its own handle, so concurrent do_get chunk reads need no
        # lock and the file is pinned only for the duration of the read.
        with h5py.File(self._path, "r") as f:
            return f[self._dataset_path][slices]

    def get_tensor_descriptor(self) -> TensorDescriptor:
        return TensorDescriptor(
            array_id=self.array_id,
            dim_labels=self.dim_labels,
            shape=list(self._shape),
            chunk_shape=list(self._chunks or self._shape),
            dtype=self._dtype.str,
        )

    def list_tensor_descriptors(self):
        return [self.get_tensor_descriptor()]

    def _physical_scale(self) -> Optional[Tuple[List[float], List[str]]]:
        """Per-dim voxel size (µm) from the ``element_size_um`` attribute.

        ``element_size_um`` (the ilastik / Imaris convention) is a per-axis
        voxel-size vector in micrometres, aligned 1:1 with the dataset's own
        axes -- i.e. with ``dim_labels``. Projected positionally, with a
        non-positive entry left as ``0.0`` / ``""``. Best-effort: the attribute
        is optional, and a length that does not match the dataset's rank cannot
        be aligned, so both yield ``None`` rather than a guess.
        """
        try:
            if self._element_size_um is None:
                return None
            vals = [float(v) for v in np.atleast_1d(self._element_size_um).tolist()]
            if len(vals) != len(self.dim_labels):
                return None
            scale: List[float] = []
            unit: List[str] = []
            for v in vals:
                if v > 0:
                    scale.append(v)
                    unit.append("µm")
                else:
                    scale.append(0.0)
                    unit.append("")
            if not any(scale):
                return None
            return scale, unit
        except Exception:
            return None

    def get_metadata(self) -> dict:
        """Return HDF5 metadata as dict.

        Returns:
            Dict with dataset info (shape, dtype, chunks)
        """
        return {
            "shape": list(self._shape),
            "dtype": str(self._dtype),
            "chunks": list(self._chunks) if self._chunks else None,
        }
