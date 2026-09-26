"""Find the data plane a biopb control owns — a client, stdlib only.

The control is the process that owns this machine's data plane: it chose the
bind, the port and the scheme, and it wrote the credential. This package asks
it, over its HTTP API, and is the one supported way to do so from outside the
monorepo. It implements a language-neutral contract, so a client in another
language can do the same over plain HTTP. The algorithm plane is asked the
same way: which servers there are, and to start or stop the ones it runs.

It never starts a process and imports nothing beyond the standard library, so
it can be imported where ``biopb.tensor`` (pyarrow) cannot.
"""

from ._algorithms import (
    algorithm_logs,
    algorithms,
    ensure_algorithm,
    refresh_algorithms,
    restart_algorithm,
    stop_algorithm,
)
from ._client import base_url, data_plane, ensure_data_plane
from ._data_plane import is_local_url

__all__ = [
    "algorithm_logs",
    "algorithms",
    "base_url",
    "data_plane",
    "ensure_algorithm",
    "ensure_data_plane",
    "is_local_url",
    "refresh_algorithms",
    "restart_algorithm",
    "stop_algorithm",
]
