"""The server's scratch space: one source, always there, holding nothing.

An upload is a temp store for an intermediate result. Until now getting one
took a round trip -- ``register_source`` minted a container, wrote a directory
under ``write_dir``, recorded an id in it, and every later life had to walk
that subtree to find it again -- and what came back was a permanent source that
happened to have been created by a client. The protocol encouraged the opposite
of what the path is for.

A writable server offers one scratch source instead, at a **fixed id**
(:data:`SCRATCH_SOURCE_ID`). Fixed is the point: a producer writes to
``zarr://scratch/@fields/<name>`` without asking for anything first, so there
is no container to mint, no id to remember and nothing to adopt at boot.

**It holds no bytes and no directory of its own.** Its tensors are uploaded
fields like any other, under ``<write_dir>/fields/scratch/``, so the boot pass
that re-attaches every source's fields (``fields.fields_attacher``) is also the
whole of this source's adoption. Nothing here writes to disk.

**It is not reclaimable.** A registered source was dropped once it stood empty,
because an abandoned one left a directory and a catalog row nothing would ever
reach. Neither exists here: an empty scratch source is a source with an empty
tensor list, which is what it is between uploads.

**What it does own is the lifetime.** :attr:`~ScratchSource.max_upload_ttl`
caps every upload added to it, an unset request included, so a temp store
cannot accumulate forever by omission -- the one thing that makes this a
scratch space rather than a shared permanent one.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional

from biopb.tensor.descriptor_pb2 import TensorDescriptor

from biopb_tensor_server.adapters.fields import fields_root, source_fields_dir
from biopb_tensor_server.core.adapter_base import SourceAdapter, TensorAdapter
from biopb_tensor_server.core.errors import TensorNotFound

__all__ = ["DEFAULT_SCRATCH_TTL", "SCRATCH_SOURCE_ID", "ScratchSource"]

#: How long an upload here is kept when nothing says otherwise: a day.
#: Long enough for an overnight job to still find what it wrote, short
#: enough that a scrap heap nobody tends empties itself. Overridden by
#: ``ServerConfig.scratch_ttl``.
DEFAULT_SCRATCH_TTL = 86400.0

#: The scratch source's id, on every writable server. A plain word rather than
#: a minted one because a client has to be able to write it down: a discovered
#: id is ``<type>_<hex>`` (``discovery.generate_source_id``) and a client that
#: did not create a source cannot guess one, so nothing can collide with this.
SCRATCH_SOURCE_ID = "scratch"


class ScratchSource(SourceAdapter):
    """An empty source that takes attachments, and caps their lifetime.

    Empty in the ordinary sense the catalog already models -- a resolved source
    with an empty tensor list -- so the base resolves, lists and checks what is
    attached to it exactly as it does for a source the server discovered. What
    is here is the methods :class:`SourceAdapter` declares abstract, the cap,
    and the one thing having no format costs it: a resolution miss that cannot
    answer ``self``.

    Its ``content_version`` is None, which is the base's word for content this
    adapter does not serve. Every member carries its own token, minted when it
    is uploaded, so a name reclaimed after a discard never inherits the chunk-id
    namespace of what held it before.
    """

    _source_type = "scratch"

    def __init__(self, write_dir: Path, max_upload_ttl: Optional[float] = None) -> None:
        self.source_id = SCRATCH_SOURCE_ID
        #: Where its tensors actually are, which is what the catalog groups by.
        #: There is no container directory -- this one holds the fields.
        self._source_url = str(
            source_fields_dir(fields_root(Path(write_dir)), SCRATCH_SOURCE_ID)
        )
        #: Ceiling, in seconds, on every upload added here -- read by
        #: ``UploadManager._deadline_for``, which also applies it to a request
        #: that named no lifetime. None leaves uploads here undated, which is
        #: what a source the server discovered gives them.
        self.max_upload_ttl = max_upload_ttl

    @classmethod
    def create_from_config(cls, config: Any) -> ScratchSource:
        raise NotImplementedError(
            "the scratch source is the writable server's own, never configured "
            "as a data source"
        )

    def list_tensor_descriptors(self) -> List[TensorDescriptor]:
        """None of its own: every tensor here was uploaded.

        ``catalog_tensors`` appends the published uploaded fields after this,
        which is the whole of what this source serves.
        """
        return []

    def get_metadata(self) -> dict:
        """Nothing. Metadata describes an acquisition, and this is a scrap heap.

        A tensor uploaded here brings its own axes and shape; there is no
        shared physical scale or channel list for it to inherit, because the
        tensors on it have nothing to do with each other.
        """
        return {}

    def get_tensor_adapter(self, tensor_id: Optional[str]) -> TensorAdapter:
        """The source's default tensor, or a typed miss; never ``self``.

        Reached only after ``resolve_tensor`` has missed the attachment index,
        so a named field that gets here names nothing. An unnamed one asks for
        the default, which is the first published field -- of which an empty
        scratch source has none. Both are resolution misses rather than server
        errors: asking a source with no tensors for its tensor is a caller's
        mistake about what it holds.
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
