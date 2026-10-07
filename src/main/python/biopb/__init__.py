"""BioPB Python package: version plus the client for a biopb control plane.

Control-client names are re-exported from the private :mod:`biopb._control`
(not ``biopb.control``, to avoid confusion with the ``biopb-control`` server
distribution). Stdlib only, so a bare ``import biopb`` stays cheap.
"""

from __future__ import annotations

# Prefer the build-time generated file: dist-info METADATA is stamped at install
# time and goes stale in an editable checkout.
try:
    from ._version import version as __version__
except ImportError:
    try:
        import importlib.metadata as _importlib_metadata
    except ImportError:  # pragma: no cover
        import importlib_metadata as _importlib_metadata

    try:
        __version__ = _importlib_metadata.version("biopb")
    except Exception:
        __version__ = "0.0.0"

from ._control import (
    ENV_TENSOR_TLS_CA,
    ENV_TENSOR_TLS_FINGERPRINT,
    ENV_TENSOR_TOKEN,
    ENV_TENSOR_URL,
    DataPlaneEndpoint,
    LocalTrustError,
    algorithm_logs,
    algorithms,
    base_url,
    control_grpc_url,
    default_data_plane_url,
    ensure_algorithm,
    ensure_data_plane,
    find_data_plane,
    is_local_url,
    local_data_plane_fingerprint,
    probe_data_plane_scheme,
    refresh_algorithms,
    resolve_data_plane,
    resolve_data_plane_token,
    restart_algorithm,
    stop_algorithm,
    user_base_url,
)

__all__ = [
    "ENV_TENSOR_TLS_CA",
    "ENV_TENSOR_TLS_FINGERPRINT",
    "ENV_TENSOR_TOKEN",
    "ENV_TENSOR_URL",
    "DataPlaneEndpoint",
    "LocalTrustError",
    "__version__",
    "algorithm_logs",
    "algorithms",
    "base_url",
    "control_grpc_url",
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
