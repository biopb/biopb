"""The ``tensor-server`` source type, as far as `core.config` defines it.

A ``grpc://`` upstream is the one source type a url alone can name, so
`detect_source_type` routes it and every other scheme -- local paths included --
returns None, leaving format typing to the adapters' `claim()` protocol
(biopb/biopb#277 item B). Alongside it, the `alias` field that namespaces a
proxied `source_id`: slash-free (the id boundary is the first `/`) and carried
through `parse_config`.

Acting on any of this -- expanding an entry into concrete sources -- is
`resolve_test.py`.
"""

import pytest
from biopb_tensor_server.core.config import (
    SourceConfig,
    detect_source_type,
    parse_config,
)
from biopb_tensor_server.core.remote import is_remote_url


class TestTensorServerSourceType:
    """Scheme recognition, url-derived typing, and the alias field."""

    def test_is_remote_url_recognizes_grpc_schemes(self):
        assert is_remote_url("grpc://lab-store:8815") is True
        assert is_remote_url("grpc+tls://lab-store:8815") is True
        assert is_remote_url("grpcs://lab-store:8815") is True
        assert is_remote_url("GRPC://Lab-Store:8815") is True  # case-insensitive
        # unchanged behaviour for local + other remote schemes
        assert is_remote_url("/data/scratch") is False
        assert is_remote_url("s3://bucket/key") is True

    def test_detect_source_type_maps_grpc_to_tensor_server(self):
        assert detect_source_type("grpc://lab:8815") == "tensor-server"
        assert detect_source_type("grpc+tls://lab:8815") == "tensor-server"
        assert detect_source_type("grpcs://lab:8815") == "tensor-server"
        # other remote schemes remain non-auto-detectable
        assert detect_source_type("s3://bucket/key") is None

    def test_detect_source_type_does_not_type_local_paths(self):
        """Filesystem format detection belongs to the adapters (claim()), not here.

        detect_source_type only routes remote schemes now (biopb/biopb#277 item
        B); every local path -- whatever its extension or layout -- returns None
        so the adapters remain the single source of truth for format typing.
        """
        for url in (
            "/data/experiment.zarr",
            "/data/image.ome.tif",
            "/data/plain.tif",
            "/data/scan.czi",
            "/data/acquisition/",
        ):
            assert detect_source_type(url) is None, url

    def test_tensor_server_in_type_literal(self):
        # explicit type still round-trips through SourceConfig
        s = SourceConfig(url="grpc://lab:8815", type="tensor-server")
        assert s.type == "tensor-server"
        assert s.is_remote is True

    def test_alias_must_be_slash_free(self):
        # source_id boundary is the first '/', so an alias prefix cannot contain it
        with pytest.raises(ValueError, match="slash-free"):
            SourceConfig(url="grpc://lab:8815", alias="lab/sub")
        # a slash-free alias is accepted
        assert SourceConfig(url="grpc://lab:8815", alias="lab").alias == "lab"

    def test_alias_parsed_from_config_dict(self):
        cfg = parse_config(
            {
                "sources": [
                    {"url": "grpc://lab:8815", "alias": "lab"},
                    {"url": "grpc://arc:8815", "type": "tensor-server", "alias": "arc"},
                ]
            }
        )
        aliases = {s.url: s.alias for s in cfg.sources}
        assert aliases == {"grpc://lab:8815": "lab", "grpc://arc:8815": "arc"}
