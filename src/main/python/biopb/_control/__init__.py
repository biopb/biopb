"""Find the data plane a biopb control owns — a client, stdlib only.

The control is the process that owns this machine's data plane: it chose the
bind, the port and the scheme, and it wrote the credential. This package asks
it, over its HTTP API, and is the one supported way to do so from outside the
monorepo. It implements a language-neutral contract, so a client in another
language can do the same over plain HTTP. The algorithm plane is asked the
same way: which servers there are, and to start or stop the ones it runs.

It never starts a process and imports nothing beyond the standard library, so
it can be imported where ``biopb.tensor`` (pyarrow) cannot.

Private (leading underscore): its names are not meant to be reached at
``biopb._control.x``. The root package re-exports every name below directly
as ``biopb.x``, which is the supported way in -- kept off ``biopb.control``
so that dotted path is never mistaken for ``biopb-control``, the separate
control-plane server distribution this package is a client *of*.
"""

from ._algorithms import (
    algorithm_logs,
    algorithms,
    ensure_algorithm,
    refresh_algorithms,
    restart_algorithm,
    stop_algorithm,
)
from ._client import base_url, ensure_data_plane, find_data_plane, user_base_url
from ._data_plane import (
    ENV_TENSOR_TLS_CA,
    ENV_TENSOR_TLS_FINGERPRINT,
    ENV_TENSOR_TOKEN,
    ENV_TENSOR_URL,
    DataPlaneEndpoint,
    LocalTrustError,
    control_grpc_url,
    data_plane_trust,
    default_data_plane_url,
    is_local_url,
    local_data_plane_fingerprint,
    probe_data_plane_scheme,
    resolve_data_plane,
    resolve_data_plane_token,
)

__all__ = [
    "ENV_TENSOR_TLS_CA",
    "ENV_TENSOR_TLS_FINGERPRINT",
    "ENV_TENSOR_TOKEN",
    "ENV_TENSOR_URL",
    "DataPlaneEndpoint",
    "LocalTrustError",
    "algorithm_logs",
    "algorithms",
    "base_url",
    "control_grpc_url",
    "data_plane_trust",
    "default_data_plane_url",
    "ensure_algorithm",
    "ensure_data_plane",
    "find_data_plane",
    "is_local_url",
    "local_data_plane_fingerprint",
    "probe_data_plane_scheme",
    "refresh_algorithms",
    "resolve_data_plane",
    "resolve_data_plane_token",
    "restart_algorithm",
    "stop_algorithm",
    "user_base_url",
]
