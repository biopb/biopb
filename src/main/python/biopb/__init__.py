"""Top-level BioPB Python package metadata.

Also the client for a biopb control plane (:mod:`biopb._control`, private):
every name below except ``__version__`` is that package's content, re-exported
here rather than at ``biopb.control`` so that dotted path is never mistaken
for ``biopb-control``, the separate control-plane server distribution this is
a client *of*. Stdlib only, so importing bare ``biopb`` stays cheap even
though ``biopb.tensor``/``biopb.image`` (pyarrow, dask) are not imported here.
"""

from __future__ import annotations

# The generated file first, dist-info METADATA only as the fallback -- the order
# the other packages in this repo already use (biopb-control adopted it in
# biopb/biopb#910). METADATA is stamped at install time and the generated file at
# build time, so preferring METADATA reports whenever the SDK was last installed
# rather than what is being imported: in an editable checkout that drifts behind
# its own source. The `v*` tag line this package ships on is separate from the
# product's `release-v*` line, but the drift is the same.
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
)

__all__ = [
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
]
