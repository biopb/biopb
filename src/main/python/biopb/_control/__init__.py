"""Find the data plane a biopb control owns — a client, stdlib only.

The control is the process that owns this machine's data plane: it chose the
bind, the port and the scheme, and it wrote the credential. This package asks
it, over its HTTP API. It implements a language-neutral contract, so a client
in another language can do the same over plain HTTP. The algorithm plane is
asked the same way: which servers there are, and to start or stop the ones it
runs.

The client imports nothing beyond the standard library, so it can be imported
where ``biopb.tensor`` (pyarrow) cannot.

Private, for the monorepo's own packages (biopb, biopb-mcp), which import the
names below from here. The public ways in are ``biopb.tensor.Connection`` (find
and dial the data plane), ``biopb.image.connect`` (call an algorithm server)
and the ``biopb`` CLI. Only ``LocalTrustError`` is re-exported, as
``biopb.tensor.LocalTrustError``. The package is not ``biopb.control`` so that
path is never mistaken for ``biopb-control``, the separate control-plane server
distribution this package is a client *of*.
"""

# Not imported here: ``_launch`` (start the control for the shim, the one place
# this package starts a process) and ``_agents`` (register the shim with agent
# clients). The ``biopb`` CLI is their supported interface.
from ._algorithms import (
    algorithm_logs,
    algorithms,
    ensure_algorithm,
    refresh_algorithms,
    restart_algorithm,
    stop_algorithm,
)
from ._client import base_url, ensure_data_plane, find_data_plane
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
from ._endpoints import user_base_url

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
