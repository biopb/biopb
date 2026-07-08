"""Tests for catalog source_url normalization (biopb/biopb#131).

``to_catalog_url`` rewrites local filesystem paths to a forward-slash ``file://``
form so a Windows-indexed catalog is consistent with a POSIX one and every
consumer can split on ``/`` alone. It is a pure function (no IO), so the unit
suite runs on every platform; a small stub adapter checks the descriptor build
sites actually apply it.
"""

from biopb_tensor_server.core.base import SourceAdapter, to_catalog_url


class TestToCatalogUrl:
    def test_empty_passthrough(self):
        assert to_catalog_url("") == ""

    def test_windows_absolute_path(self):
        assert (
            to_catalog_url(r"C:\Users\me\Screenshots 1\img.png")
            == "file:///C:/Users/me/Screenshots 1/img.png"
        )

    def test_windows_lowercase_drive(self):
        assert to_catalog_url(r"d:\data\x.tif") == "file:///d:/data/x.tif"

    def test_posix_absolute_path(self):
        assert to_catalog_url("/data/cells/img.tif") == "file:///data/cells/img.tif"

    def test_spaces_and_unicode_left_literal(self):
        # Readable form: separators only, no percent-encoding.
        assert (
            to_catalog_url(r"C:\imágenes\my file.tif")
            == "file:///C:/imágenes/my file.tif"
        )

    def test_windows_and_posix_agree_on_structure(self):
        # The whole point of #131: same path under two roots yields the same
        # forward-slash tail regardless of indexing OS.
        win = to_catalog_url(r"C:\proj\a\b.tif")
        posix = to_catalog_url("/proj/a/b.tif")
        assert win.endswith("/proj/a/b.tif")
        assert posix.endswith("/proj/a/b.tif")

    def test_file_uri_is_idempotent(self):
        u = "file:///C:/Users/me/img.png"
        assert to_catalog_url(u) == u

    def test_remote_urls_untouched(self):
        for u in [
            "s3://bucket/key.zarr",
            "gs://bucket/key.zarr",
            "https://host/data.tif",
            "az://container/blob",
        ]:
            assert to_catalog_url(u) == u

    def test_virtual_scheme_untouched(self):
        assert to_catalog_url("cache://aics_abc123") == "cache://aics_abc123"

    def test_relative_path_best_effort(self):
        assert to_catalog_url("rel/dir/x.tif") == "file:///rel/dir/x.tif"


class _StubAdapter(SourceAdapter):
    """Minimal concrete SourceAdapter to exercise get_source_descriptor()."""

    def __init__(self, source_url: str, source_type: str = "aics"):
        self.source_id = "stub_id"
        self._source_url = source_url
        self._source_type = source_type

    @classmethod
    def create_from_config(cls, source, credentials_config=None):  # pragma: no cover
        raise NotImplementedError

    def list_tensor_descriptors(self):
        return []

    def get_metadata(self):
        return {}

    def is_resident(self):
        # Avoid touching the filesystem for a fabricated path.
        return True


class TestDescriptorNormalization:
    def test_windows_path_becomes_file_uri(self):
        desc = _StubAdapter(
            r"C:\Users\me\OneDrive\Pictures\shot.png"
        ).get_source_descriptor()
        assert desc.source_url == "file:///C:/Users/me/OneDrive/Pictures/shot.png"
        # source_id is independent of the URL string (no re-index needed).
        assert desc.source_id == "stub_id"

    def test_remote_path_unchanged(self):
        desc = _StubAdapter("s3://bucket/x.zarr").get_source_descriptor()
        assert desc.source_url == "s3://bucket/x.zarr"
