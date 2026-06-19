"""Unresolved-source proxy adapter (cloud-storage phase 2).

A cloud/synced-folder source is catalogued as a URL-only, *unresolved* entry:
its shape/dtype/fields are unknown until a user actually opens it, because
reading any byte recalls the whole dehydrated object (slow, fills the disk the
user freed, blocks offline). ``UnresolvedSourceAdapter`` is the placeholder the
server registers for such a source. It is deliberately split into two surfaces:

- A **catalog surface** -- ``list_tensor_descriptors``/``get_source_descriptor``/
  ``get_metadata``/``has_native_pyramid`` -- that NEVER resolves. It reports an
  empty, not-resident source so ListFlights, the metadata-DB sync, and the
  precache worker all stay cheap (precache loops ``list_tensor_descriptors`` and
  skips an empty source before it ever reaches a serving call).

- A **serve surface** -- ``get_tensor_adapter`` -- that IS the consented
  resolution hook. The first ``GetFlightInfo`` routes through it, so on first
  call it hydrates: it re-runs the real claim + ``create_from_config`` on the
  now-resident path (the recorded ``source_type`` was a recall-free guess; the
  authoritative one comes from probing the hydrated content), caches the real
  adapter, fires ``on_resolved`` (the metadata-DB backfill), and delegates. Every
  later call delegates straight to the resolved adapter.

Resolution runs once under a lock (concurrent first opens don't double-hydrate).
A resolution failure (offline / declined / unrecognized) raises
``SourceUnresolvedError``, which the server maps to a clean
``FlightUnavailableError`` at the read boundary.
"""

import logging
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Optional, Tuple

from biopb.tensor.descriptor_pb2 import DataSourceDescriptor

from biopb_tensor_server.base import SourceAdapter, TensorAdapter, to_catalog_url
from biopb_tensor_server.errors import (
    SourceResolveRetriableError,
    SourceUnresolvedError,
)

if TYPE_CHECKING:
    from biopb.tensor.descriptor_pb2 import PyramidLevel, TensorDescriptor

    from biopb_tensor_server.config import SourceConfig
    from biopb_tensor_server.discovery import AdapterRegistry

logger = logging.getLogger(__name__)

# Callback fired once when a source resolves: (source_id, resolved_adapter).
# The source manager wires this to the metadata DB so the catalog row is
# backfilled from NULL shape/dtype to the now-known descriptor.
OnResolved = Any


class UnresolvedSourceAdapter(SourceAdapter):
    """Placeholder adapter that resolves a cloud source lazily on first serve."""

    def __init__(
        self,
        source_config: "SourceConfig",
        registry: "AdapterRegistry",
        credentials_config: Optional[Any] = None,
        on_resolved: Optional[OnResolved] = None,
        cloud_root: bool = False,
    ):
        self._config = source_config
        self.source_id = source_config.source_id
        self._source_url = source_config.url
        # Whether this source came from a ``cloud = true`` root. Carried onto the
        # resolve-time re-claim ``ClaimContext`` so content-membership grouping
        # (multi-file OME-TIFF / DICOM series) stays suppressed at resolve --
        # the file is resident by then, so residency can no longer gate it.
        self._cloud_root = cloud_root
        # Recall-free name/structure guess from claim time; the authoritative
        # type is re-derived from the hydrated content at resolution.
        self._source_type = source_config.type or "unknown"
        self._tensor_name = None
        self._registry = registry
        self._credentials_config = credentials_config
        self._on_resolved = on_resolved
        self._resolved: Optional[SourceAdapter] = None
        self._lock = threading.Lock()

    # --- introspection ------------------------------------------------------

    @property
    def is_resolved(self) -> bool:
        return self._resolved is not None

    # --- catalog surface (never resolves) -----------------------------------

    def list_tensor_descriptors(self) -> List["TensorDescriptor"]:
        if self._resolved is not None:
            return self._resolved.list_tensor_descriptors()
        # Unresolved: an empty tensor list is what keeps ListFlights, the
        # metadata-DB sync, and the precache backlog cheap (and is what the
        # metadata_db `if source_desc.tensors:` guard tolerates -> NULL row).
        return []

    def get_metadata(self) -> dict:
        if self._resolved is not None:
            return self._resolved.get_metadata()
        return {}

    def get_source_descriptor(self) -> DataSourceDescriptor:
        if self._resolved is not None:
            return self._resolved.get_source_descriptor()
        return DataSourceDescriptor(
            source_id=self.source_id,
            source_url=to_catalog_url(self._source_url),
            source_type=self._source_type,
            tensors=[],
            metadata_json="",
            data_resident=False,
        )

    def is_resident(self) -> bool:
        if self._resolved is not None:
            return self._resolved.is_resident()
        return False

    def has_native_pyramid(self) -> bool:
        if self._resolved is not None:
            return self._resolved.has_native_pyramid()
        return False

    def get_native_pyramid_levels(
        self, tensor_id: Optional[str] = None
    ) -> Optional[List["PyramidLevel"]]:
        if self._resolved is not None:
            return self._resolved.get_native_pyramid_levels(tensor_id)
        return None

    def get_physical_scale(
        self, tensor_id: Optional[str] = None
    ) -> Optional[Tuple[List[float], List[str]]]:
        if self._resolved is not None:
            return self._resolved.get_physical_scale(tensor_id)
        return None

    # --- serve surface (NEVER resolves) -------------------------------------

    def get_tensor_adapter(self, tensor_id: Optional[str]) -> TensorAdapter:
        # The serve paths (GetFlightInfo / DoGet) must never trigger a recall:
        # resolution is a single, explicit, consented operation (the server's
        # ``resolve`` action -> ``self.resolve()``), not a side effect of asking
        # to read. Until that has run, refuse with a legible error -- server.py's
        # get_flight_info translates SourceUnresolvedError to a clean
        # FlightUnavailableError ("open to resolve").
        if self._resolved is None:
            raise SourceUnresolvedError(
                f"source {self.source_id!r} is unresolved (cloud / synced-folder); "
                f"resolve it before reading"
            )
        # Already resolved: delegate. An unresolved source advertised no tensors,
        # so a client may still send the source_id / empty default -- map that to
        # the resolved default tensor.
        if not tensor_id or tensor_id == self.source_id:
            descriptors = self._resolved.list_tensor_descriptors()
            if descriptors:
                tensor_id = descriptors[0].array_id
        return self._resolved.get_tensor_adapter(tensor_id)

    # --- resolution (the consented hook) ------------------------------------

    def resolve(self) -> DataSourceDescriptor:
        """Hydrate the source (downloading it if dehydrated) and return its full,
        now-resolved ``DataSourceDescriptor``. The sole resolution trigger; safe
        to call repeatedly (the underlying hydrate is once-only)."""
        self._resolve()
        return self.get_source_descriptor()  # delegates -> full tensor list

    @classmethod
    def create_from_config(
        cls, source: "SourceConfig", credentials_config: Optional[Any] = None
    ) -> "SourceAdapter":
        # Not used: the source manager constructs this proxy directly (it needs
        # the registry + on_resolved callback, which create_from_config can't
        # supply). Resolution itself goes through the *real* adapter's
        # create_from_config, not this one.
        raise NotImplementedError(
            "UnresolvedSourceAdapter is constructed by the source manager, "
            "not via create_from_config"
        )

    # --- resolution ----------------------------------------------------------

    def _resolve(self) -> SourceAdapter:
        """Hydrate and resolve once; subsequent calls return the cached adapter."""
        if self._resolved is not None:
            return self._resolved
        with self._lock:
            if self._resolved is not None:  # lost the race; someone resolved it
                return self._resolved
            adapter = self._build_resolved_adapter()
            self._resolved = adapter
            if self._on_resolved is not None:
                try:
                    self._on_resolved(self.source_id, adapter)
                except Exception:
                    logger.exception(
                        "on_resolved callback failed for source %s", self.source_id
                    )
            logger.info(
                "resolved cloud source %s (%s)", self.source_id, self._source_type
            )
            return adapter

    def _build_resolved_adapter(self) -> SourceAdapter:
        """Re-run the real claim + create_from_config on the now-resident path.

        Probing the hydrated content (the consented hydrate) yields the
        authoritative source_type -- the claim-time type was a recall-free guess
        (e.g. a .zarr provisionally typed ome-zarr may resolve to ome-zarr-hcs).
        A fresh DiscoveryState is used so consumed_paths from the original scan
        do not suppress this source's own member claims.
        """
        from biopb_tensor_server.config import SourceConfig
        from biopb_tensor_server.discovery import ClaimContext, DiscoveryState

        resolved_type = self._source_type
        dim_labels = self._config.dim_labels
        dataset = self._config.dataset

        try:
            ctx = ClaimContext(Path(self._source_url), cloud_root=self._cloud_root)
            claims = self._registry.get_claims_for_path(ctx, DiscoveryState())
        except OSError as e:
            # An IO/recall failure while probing the now-resident path is
            # transient -- do NOT silently degrade to the claim-time guess (that
            # launders a network blip into a wrong type). Surface it as retriable.
            raise SourceResolveRetriableError(
                f"source {self.source_id!r} could not be resolved "
                f"(re-claim recall/IO failed): {e}"
            ) from e
        except Exception as e:  # a non-IO claim error: no claim, keep the guess
            logger.debug("re-claim during resolution failed for %s: %s", self.source_id, e)
            claims = []
        if claims:
            claim = claims[0]
            resolved_type = claim.source_type
            if claim.dim_labels:
                dim_labels = claim.dim_labels
            if claim.extra_config.get("dataset"):
                dataset = claim.extra_config["dataset"]

        adapter_cls = self._registry.get_adapter_for_type(resolved_type)
        if adapter_cls is None:
            raise SourceUnresolvedError(
                f"source {self.source_id!r} could not be resolved: no adapter "
                f"for type {resolved_type!r}"
            )

        config = SourceConfig(
            url=self._source_url,
            type=resolved_type,
            source_id=self.source_id,
            dim_labels=dim_labels,
            dataset=dataset,
            credentials_profile=self._config.credentials_profile,
        )
        try:
            return adapter_cls.create_from_config(config, self._credentials_config)
        except SourceUnresolvedError:
            raise
        except OSError as e:
            # Recall/IO failure opening the (now-resident) content: transient,
            # a retry may succeed once the sync engine delivers the bytes.
            raise SourceResolveRetriableError(
                f"source {self.source_id!r} could not be resolved "
                f"(open/hydrate recall/IO failed): {e}"
            ) from e
        except Exception as e:
            # A parse/format failure on resident bytes is permanent.
            raise SourceUnresolvedError(
                f"source {self.source_id!r} could not be resolved "
                f"(open/hydrate failed): {e}"
            ) from e
