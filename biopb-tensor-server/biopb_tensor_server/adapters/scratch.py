"""The scratch source: one per ``write_dir``, holding nothing of its own.

A producer writes an intermediate result to ``zarr://scratch/@fields/<name>``.
The id is fixed (:data:`SCRATCH_SOURCE_ID`), so there is no container to mint
and nothing to remember between runs.

It owns no directory. Its tensors are uploaded fields under
``<write_dir>/fields/scratch/``, re-attached at boot by
``fields.fields_attacher`` like any other source's, and an empty scratch source
is simply one between uploads -- never reclaimed. What it does own is
:attr:`~ScratchSource.max_upload_ttl`, the cap that keeps a temp store from
accumulating forever.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional

from biopb.tensor.descriptor_pb2 import TensorDescriptor

from biopb_tensor_server.adapters.fields import fields_root, source_fields_dir
from biopb_tensor_server.core.adapter_base import SourceAdapter, TensorAdapter
from biopb_tensor_server.core.errors import TensorNotFound

__all__ = ["DEFAULT_SCRATCH_TTL", "SCRATCH_SOURCE_ID", "ScratchSource"]

#: How long an upload here is kept when nothing says otherwise: a day, long
#: enough for an overnight job to find what it wrote. Overridden by
#: ``ServerConfig.scratch_ttl``.
DEFAULT_SCRATCH_TTL = 86400.0

#: The scratch source's id, on every server with somewhere to write. A plain
#: word so a client
#: can write it down; a discovered id is ``<type>_<hex>``
#: (``discovery.generate_source_id``), so nothing collides with it.
SCRATCH_SOURCE_ID = "scratch"


class ScratchSource(SourceAdapter):
    """An empty source that takes attachments, and caps their lifetime.

    Empty as the catalog already models it -- a resolved source with an empty
    tensor list -- so the base resolves and lists what is attached to it exactly
    as it does for a discovered source.

    Its ``content_version`` is None, the base's word for content this adapter
    does not serve. Every member carries its own token, minted when it is
    uploaded, so a name reclaimed after a discard never inherits the chunk-id
    namespace of what held it before.
    """

    _source_type = "scratch"

    def __init__(self, write_dir: Path, max_upload_ttl: Optional[float] = None) -> None:
        self.source_id = SCRATCH_SOURCE_ID
        #: Where its tensors are, which is what the catalog groups by. There
        #: is no container directory -- this one holds the fields.
        self._source_url = str(
            source_fields_dir(fields_root(Path(write_dir)), SCRATCH_SOURCE_ID)
        )
        #: Ceiling, in seconds, on every upload added here, applied by
        #: ``UploadManager._deadline_for`` including to a request that named no
        #: lifetime. None leaves them undated, as a discovered source does.
        self.max_upload_ttl = max_upload_ttl

    @classmethod
    def create_from_config(cls, config: Any) -> ScratchSource:
        raise NotImplementedError(
            "the scratch source is the server's own, never configured as a data source"
        )

    def list_tensor_descriptors(self) -> List[TensorDescriptor]:
        """None of its own: every tensor here was uploaded, and
        ``catalog_tensors`` appends the published ones after this."""
        return []

    def get_metadata(self) -> dict:
        """Nothing. Metadata describes an acquisition; the tensors here have
        nothing to do with each other and bring their own axes and shape."""
        return {}

    def get_tensor_adapter(self, tensor_id: Optional[str]) -> TensorAdapter:
        """The source's default tensor, or a typed miss; never ``self``.

        Reached only after ``resolve_tensor`` has missed the attachment index,
        so a named field here names nothing, and an unnamed one asks for the
        first published field -- of which an empty scratch source has none.
        """
        field = self._within_source_field(tensor_id)
        if field:
            raise TensorNotFound(
                f"{self.source_id} has no tensor {field!r}.",
                reason="unknown_field",
            )
        default = next(iter(self.attached_fields.values()), None)
        if default is None:
            raise TensorNotFound(
                f"{self.source_id} has no tensors yet: add one with "
                f"add_tensor before reading it.",
                reason="empty_source",
            )
        return default
