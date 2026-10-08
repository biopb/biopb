"""The registration record of a format whose catalog entry is its metadata dict."""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

from biopb_tensor_server.core.registration import (
    RegistrationRecord,
    strip_mask_bindata,
)


class MetadataRecordMixin:
    """``registration_record`` from the adapter's own ``get_metadata()``.

    ``get_metadata`` is each format's internal: how it reads its file's metadata
    into a dict. Registration calls it once and keeps nothing, and mask bitmaps
    never reach the SQL-queryable column. Formats that also carry annotations
    build their own record (``ome_registration_record``).
    """

    def registration_record(
        self,
        tensors: Sequence[Tuple[str, Sequence[str]]],
        *,
        import_rois: bool = True,
        max_rois_per_tensor: Optional[int] = None,
    ) -> RegistrationRecord:
        return RegistrationRecord(strip_mask_bindata(self.get_metadata() or {}))
