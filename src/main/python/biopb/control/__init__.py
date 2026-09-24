"""Find the data plane a biopb control owns — a client, stdlib only.

The control is the process that owns this machine's data plane: it chose the
bind, the port and the scheme, and it wrote the credential. This package asks
it, over its HTTP API, and is the one supported way to do so from outside the
monorepo. ``docs/discovery-contract.md`` is the language-neutral contract it
implements.

It never starts a process and imports nothing beyond the standard library, so
it can be imported where ``biopb.tensor`` (pyarrow) cannot.
"""

from ._client import base_url, data_plane, ensure_data_plane
from ._data_plane import is_local_url

__all__ = ["base_url", "data_plane", "ensure_data_plane", "is_local_url"]
