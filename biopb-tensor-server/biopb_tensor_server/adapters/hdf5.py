"""HDF5 adapter for tensor storage.

Relies on OS page cache for raw data caching.
"""

from typing import TYPE_CHECKING, Any, List, Optional

import numpy as np
from biopb.tensor.descriptor_pb2 import TensorDescriptor
from biopb.tensor.ticket_pb2 import ChunkBounds

from biopb_tensor_server.base import SourceAdapter, TensorAdapter
from biopb_tensor_server.discovery import ClaimContext, SourceClaim

if TYPE_CHECKING:
    from biopb_tensor_server.config import SourceConfig
    from biopb_tensor_server.discovery import DiscoveryState


class Hdf5Adapter(SourceAdapter, TensorAdapter):
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
        if not (name.endswith(".h5") or name.endswith(".hdf5")):
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

        f = h5py.File(str(source.url), "r")
        dataset = f[source.dataset]
        return cls(dataset, source.source_id, source.dim_labels)

    def __init__(
        self,
        h5_dataset,
        source_id: str,
        dim_labels: Optional[List[str]] = None,
    ):
        """Initialize HDF5 adapter.

        Args:
            h5_dataset: h5py Dataset object
            source_id: Unique identifier for this data source
            dim_labels: Optional dimension labels
        """
        self.h5_dataset = h5_dataset
        self.source_id = source_id
        self.dim_labels = dim_labels or [
            f"dim{i}" for i in range(len(h5_dataset.shape))
        ]

        # Source-level metadata for DataSourceDescriptor
        self._source_url = (
            h5_dataset.file.filename if hasattr(h5_dataset, "file") else ""
        )
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
        super().get_data(bounds)
        slices = tuple(slice(int(s), int(e)) for s, e in zip(bounds.start, bounds.stop))
        return self.h5_dataset[slices]

    def get_tensor_descriptor(self) -> TensorDescriptor:
        return TensorDescriptor(
            array_id=self.array_id,
            dim_labels=self.dim_labels,
            shape=list(self.h5_dataset.shape),
            chunk_shape=list(self.h5_dataset.chunks),
            dtype=self.h5_dataset.dtype.str,
        )

    def list_tensor_descriptors(self):
        return [self.get_tensor_descriptor()]

    def get_metadata(self) -> dict:
        """Return HDF5 metadata as dict.

        Returns:
            Dict with dataset info (shape, dtype, chunks)
        """
        return {
            "shape": list(self.h5_dataset.shape),
            "dtype": str(self.h5_dataset.dtype),
            "chunks": list(self.h5_dataset.chunks) if self.h5_dataset.chunks else None,
        }
