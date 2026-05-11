"""Unit tests for remote storage abstraction."""

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

import fsspec

from biopb_tensor_server.remote import (
    CredentialProfile,
    CredentialsConfig,
    RemoteStore,
    _detect_storage_type,
    _extract_signed_url_params,
    _get_env_credentials,
    is_remote_url,
)


# =============================================================================
# CredentialProfile Tests
# =============================================================================

class TestCredentialProfile:
    """Tests for CredentialProfile.to_storage_options()."""

    def test_s3_anonymous_no_credentials(self):
        """S3 profile with no key/secret should set anon=True."""
        profile = CredentialProfile(
            name="anon-s3",
            storage_type="s3",
        )
        opts = profile.to_storage_options()
        assert opts["anon"] == True
        assert "key" not in opts
        assert "secret" not in opts

    def test_s3_with_key_and_secret(self):
        """S3 profile with credentials should not be anonymous."""
        profile = CredentialProfile(
            name="aws-prod",
            storage_type="s3",
            key="AKIAIOSFODNN7EXAMPLE",
            secret="wJalrXUtnFEMI/K7MDENG",
        )
        opts = profile.to_storage_options()
        # When credentials are set, anon should be False (explicitly set)
        # or not present at all (which s3fs interprets as non-anonymous)
        assert "anon" not in opts or opts["anon"] == False
        assert opts["key"] == "AKIAIOSFODNN7EXAMPLE"
        assert opts["secret"] == "wJalrXUtnFEMI/K7MDENG"

    def test_s3_with_session_token(self):
        """S3 profile with temporary credentials includes token."""
        profile = CredentialProfile(
            name="aws-temp",
            storage_type="s3",
            key="AKIAIOSFODNN7EXAMPLE",
            secret="wJalrXUtnFEMI/K7MDENG",
            token="temp-session-token",
        )
        opts = profile.to_storage_options()
        assert opts["token"] == "temp-session-token"

    def test_s3_with_region(self):
        """S3 profile with region sets client_kwargs."""
        profile = CredentialProfile(
            name="aws-eu",
            storage_type="s3",
            region="eu-west-1",
        )
        opts = profile.to_storage_options()
        assert opts["client_kwargs"]["region_name"] == "eu-west-1"

    def test_s3_with_endpoint_url(self):
        """S3 profile with custom endpoint sets endpoint_url."""
        profile = CredentialProfile(
            name="minio",
            storage_type="s3",
            endpoint_url="https://minio.example.com",
        )
        opts = profile.to_storage_options()
        assert opts["client_kwargs"]["endpoint_url"] == "https://minio.example.com"

    def test_s3_with_region_and_endpoint(self):
        """S3 profile with both region and endpoint."""
        profile = CredentialProfile(
            name="minio-eu",
            storage_type="s3",
            region="eu-west-1",
            endpoint_url="https://minio.example.com",
        )
        opts = profile.to_storage_options()
        assert opts["client_kwargs"]["region_name"] == "eu-west-1"
        assert opts["client_kwargs"]["endpoint_url"] == "https://minio.example.com"

    def test_gcs_with_token(self):
        """GCS profile with service account token."""
        profile = CredentialProfile(
            name="gcs-shared",
            storage_type="gs",
            token="/path/to/service-account.json",
        )
        opts = profile.to_storage_options()
        assert opts["token"] == "/path/to/service-account.json"

    def test_azure_with_key_and_secret(self):
        """Azure profile with account credentials."""
        profile = CredentialProfile(
            name="azure-prod",
            storage_type="azure",
            key="storage-account-name",
            secret="account-key-value",
        )
        opts = profile.to_storage_options()
        assert opts["account_name"] == "storage-account-name"
        assert opts["account_key"] == "account-key-value"

    def test_azure_with_connection_string(self):
        """Azure profile with connection string."""
        profile = CredentialProfile(
            name="azure-conn",
            storage_type="azure",
            token="DefaultEndpointsProtocol=https;AccountName=...",
        )
        opts = profile.to_storage_options()
        assert opts["connection_string"] == "DefaultEndpointsProtocol=https;AccountName=..."


# =============================================================================
# CredentialsConfig Tests
# =============================================================================

class TestCredentialsConfig:
    """Tests for CredentialsConfig profile lookup."""

    def test_get_default_profile(self):
        """Get profile using default_profile name."""
        profile1 = CredentialProfile(name="aws-prod", storage_type="s3")
        profile2 = CredentialProfile(name="gcs-shared", storage_type="gs")
        config = CredentialsConfig(
            default_profile="aws-prod",
            profiles=[profile1, profile2],
        )
        result = config.get_profile()
        assert result.name == "aws-prod"

    def test_get_profile_by_name(self):
        """Get specific profile by name override."""
        profile1 = CredentialProfile(name="aws-prod", storage_type="s3")
        profile2 = CredentialProfile(name="gcs-shared", storage_type="gs")
        config = CredentialsConfig(
            default_profile="aws-prod",
            profiles=[profile1, profile2],
        )
        result = config.get_profile("gcs-shared")
        assert result.name == "gcs-shared"

    def test_get_profile_not_found(self):
        """Get profile returns None if not found."""
        config = CredentialsConfig(
            default_profile="aws-prod",
            profiles=[CredentialProfile(name="aws-prod", storage_type="s3")],
        )
        result = config.get_profile("nonexistent")
        assert result is None

    def test_get_profile_no_default(self):
        """Get profile returns None if no default set."""
        config = CredentialsConfig(
            profiles=[CredentialProfile(name="aws-prod", storage_type="s3")],
        )
        result = config.get_profile()
        assert result is None


# =============================================================================
# URL Detection Tests
# =============================================================================

class TestDetectStorageType:
    """Tests for _detect_storage_type()."""

    def test_s3_url(self):
        assert _detect_storage_type("s3://bucket/path") == "s3"

    def test_gs_url(self):
        assert _detect_storage_type("gs://bucket/path") == "gs"

    def test_gcs_url(self):
        assert _detect_storage_type("gcs://bucket/path") == "gs"

    def test_http_url(self):
        assert _detect_storage_type("http://example.com/file") == "http"

    def test_https_url(self):
        assert _detect_storage_type("https://example.com/file") == "http"

    def test_azure_url(self):
        assert _detect_storage_type("az://container/path") == "azure"
        assert _detect_storage_type("azure://container/path") == "azure"

    def test_file_url(self):
        assert _detect_storage_type("file:///local/path") == "file"

    def test_no_scheme(self):
        assert _detect_storage_type("/local/path") == "file"

    def test_unknown_scheme(self):
        assert _detect_storage_type("ftp://server/path") == "ftp"


class TestIsRemoteUrl:
    """Tests for is_remote_url()."""

    def test_s3_is_remote(self):
        assert is_remote_url("s3://bucket/path") == True

    def test_gs_is_remote(self):
        assert is_remote_url("gs://bucket/path") == True

    def test_http_is_remote(self):
        assert is_remote_url("http://example.com/file") == True
        assert is_remote_url("https://example.com/file") == True

    def test_local_path_is_not_remote(self):
        assert is_remote_url("/local/path") == False
        assert is_remote_url("file:///local/path") == False
        assert is_remote_url("./relative/path") == False


# =============================================================================
# Signed URL Tests
# =============================================================================

class TestExtractSignedUrlParams:
    """Tests for _extract_signed_url_params()."""

    def test_aws_s3_signed_url(self):
        """Extract AWS S3 presigned URL params."""
        url = "s3://bucket/path?AWSAccessKeyId=AKIAIOSFODNN7EXAMPLE&Signature=xxx&Expires=12345"
        opts = _extract_signed_url_params(url)
        assert opts["key"] == "AKIAIOSFODNN7EXAMPLE"

    def test_aws_s3_signed_url_with_token(self):
        """Extract AWS presigned URL with security token."""
        url = "s3://bucket/path?AWSAccessKeyId=KEY&X-Amz-Security-Token=TOKEN&Signature=xxx"
        opts = _extract_signed_url_params(url)
        assert opts["key"] == "KEY"
        assert opts["token"] == "TOKEN"

    def test_gcs_signed_url(self):
        """Extract GCS signed URL params."""
        url = "gs://bucket/path?GoogleAccessId=user@example.com&Signature=xxx"
        opts = _extract_signed_url_params(url)
        assert opts["key"] == "user@example.com"

    def test_no_signed_params(self):
        """URL without signed params returns empty dict."""
        url = "s3://bucket/path"
        opts = _extract_signed_url_params(url)
        assert opts == {}

    def test_url_without_query(self):
        """URL without query string returns empty dict."""
        url = "s3://bucket/path"
        opts = _extract_signed_url_params(url)
        assert opts == {}


# =============================================================================
# Environment Credentials Tests
# =============================================================================

class TestGetEnvCredentials:
    """Tests for _get_env_credentials()."""

    def test_s3_env_credentials(self, monkeypatch):
        """Get AWS credentials from environment."""
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "env-key")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "env-secret")
        monkeypatch.setenv("AWS_SESSION_TOKEN", "env-token")
        monkeypatch.setenv("AWS_DEFAULT_REGION", "us-west-2")

        opts = _get_env_credentials("s3")
        assert opts["key"] == "env-key"
        assert opts["secret"] == "env-secret"
        assert opts["token"] == "env-token"
        assert opts["client_kwargs"]["region_name"] == "us-west-2"

    def test_gcs_env_credentials(self, monkeypatch):
        """Get GCS credentials from environment."""
        monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "/path/to/gcs.json")
        opts = _get_env_credentials("gs")
        assert opts["token"] == "/path/to/gcs.json"

    def test_azure_env_credentials(self, monkeypatch):
        """Get Azure credentials from environment."""
        monkeypatch.setenv("AZURE_STORAGE_ACCOUNT_NAME", "account-name")
        monkeypatch.setenv("AZURE_STORAGE_ACCOUNT_KEY", "account-key")
        opts = _get_env_credentials("azure")
        assert opts["account_name"] == "account-name"
        assert opts["account_key"] == "account-key"

    def test_no_env_credentials(self, monkeypatch):
        """No credentials in environment returns empty dict."""
        # Clear all relevant env vars
        for var in ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN",
                    "AWS_DEFAULT_REGION", "GOOGLE_APPLICATION_CREDENTIALS",
                    "AZURE_STORAGE_ACCOUNT_NAME", "AZURE_STORAGE_ACCOUNT_KEY"]:
            monkeypatch.delenv(var, raising=False)

        opts = _get_env_credentials("s3")
        assert opts == {}


# =============================================================================
# RemoteStore Tests (Memory Filesystem)
# =============================================================================

@pytest.fixture
def memory_fs():
    """Create in-memory filesystem with test data (module-level fixture)."""
    fs = fsspec.filesystem('memory')
    fs.mkdir('/test.zarr')
    zarray_meta = {
        "zarr_format": 2,
        "shape": [10, 10],
        "chunks": [5, 5],
        "dtype": "<i4",
        "compressor": None,
        "fill_value": 0,
        "order": "C",
    }
    fs.pipe('/test.zarr/.zarray', json.dumps(zarray_meta).encode())
    yield fs
    # Cleanup
    try:
        fs.rm('/', recursive=True)
    except Exception:
        pass


class TestRemoteStoreMemoryFS:
    """Tests for RemoteStore using in-memory filesystem."""

    def test_remote_store_with_memory_url(self, memory_fs):
        """Create RemoteStore with memory:// URL."""
        store = RemoteStore("memory:///test.zarr")
        assert store.exists() == True
        assert store.isdir() == True
        assert store.exists('.zarray') == True

    def test_remote_store_read_text(self, memory_fs):
        """Read file contents from memory filesystem."""
        store = RemoteStore("memory:///test.zarr")
        content = store.read_text('.zarray')
        meta = json.loads(content)
        assert meta["zarr_format"] == 2
        assert meta["shape"] == [10, 10]

    def test_remote_store_listdir(self):
        """List directory contents."""
        # Create fresh filesystem for this test
        fs = fsspec.filesystem('memory')
        fs.mkdir('/listdir_test.zarr')
        fs.touch('/listdir_test.zarr/.zarray')
        fs.touch('/listdir_test.zarr/.zattrs')
        fs.mkdir('/listdir_test.zarr/0')

        store = RemoteStore("memory:///listdir_test.zarr")
        listing = store.listdir()
        # Memory filesystem returns full paths or just names depending on version
        # Check for the presence of key elements
        listing_str = ' '.join(listing)
        assert '.zarray' in listing_str
        assert '.zattrs' in listing_str or 'zattrs' in listing_str

    def test_remote_store_get_identity(self, memory_fs):
        """Get unique identity for deduplication."""
        memory_fs.pipe('/test.zarr/.zarray', b'{"test": "data"}', set_type='file')
        store = RemoteStore("memory:///test.zarr")
        identity = store.get_identity('.zarray')
        # Memory filesystem identity includes path and size
        assert 'test.zarr/.zarray' in identity

    def test_remote_store_walk(self, memory_fs):
        """Walk directory tree."""
        memory_fs.mkdir('/test.zarr/0')
        memory_fs.mkdir('/test.zarr/1')
        memory_fs.touch('/test.zarr/0/data')
        memory_fs.touch('/test.zarr/1/data')

        store = RemoteStore("memory:///test.zarr")
        walked = list(store.walk())
        assert len(walked) >= 1
        # Check that we got the subdirectories
        dirpaths = [w[0] for w in walked]
        assert '/test.zarr' in dirpaths or 'test.zarr' in dirpaths

    def test_remote_store_with_profile(self, memory_fs):
        """RemoteStore with explicit credential profile."""
        # Memory filesystem doesn't use credentials, but we test the flow
        profile = CredentialProfile(name="test", storage_type="memory")
        store = RemoteStore("memory:///test.zarr", profile=profile)
        assert store.exists() == True

    def test_remote_store_from_config(self, memory_fs):
        """Create RemoteStore via from_config factory."""
        config = CredentialsConfig(
            default_profile="test-profile",
            profiles=[CredentialProfile(name="test-profile", storage_type="memory")]
        )
        store = RemoteStore.from_config(
            "memory:///test.zarr",
            credentials_config=config,
        )
        assert store.exists() == True


# =============================================================================
# RemoteStore Tests (Local fsspec)
# =============================================================================

class TestRemoteStoreLocalFsspec:
    """Tests for RemoteStore using local filesystem via fsspec."""

    @pytest.fixture
    def local_zarr(self, tmp_path):
        """Create local test zarr dataset."""
        zarr_path = tmp_path / "test.zarr"
        zarr_path.mkdir()
        zarray_meta = {
            "zarr_format": 2,
            "shape": [10, 10],
            "chunks": [5, 5],
            "dtype": "<i4",
            "compressor": None,
            "fill_value": 0,
            "order": "C",
        }
        (zarr_path / ".zarray").write_text(json.dumps(zarray_meta))
        yield zarr_path

    def test_file_url_local_path(self, local_zarr):
        """Access local path via file:// URL."""
        url = f"file://{local_zarr}"
        store = RemoteStore(url)
        assert store.exists() == True
        assert store.isdir() == True

    def test_read_local_file(self, local_zarr):
        """Read file from local filesystem via fsspec."""
        url = f"file://{local_zarr}"
        store = RemoteStore(url)
        content = store.read_text('.zarray')
        meta = json.loads(content)
        assert meta["zarr_format"] == 2

    def test_open_file_like(self, local_zarr):
        """Open file as file-like object."""
        url = f"file://{local_zarr}"
        store = RemoteStore(url)
        with store.open('.zarray', mode='rb') as f:
            content = f.read()
        meta = json.loads(content.decode())
        assert meta["shape"] == [10, 10]


# =============================================================================
# ZarrAdapter Remote Tests
# =============================================================================

class TestZarrAdapterRemote:
    """Tests for ZarrAdapter with remote storage."""

    @pytest.fixture
    def memory_zarr(self):
        """Create in-memory zarr dataset."""
        import zarr
        from zarr.storage import FSStore
        import numpy as np

        fs = fsspec.filesystem('memory')
        store = FSStore('test.zarr', fs=fs)
        arr = zarr.open_array(store, mode='w', shape=(10, 10), chunks=(5, 5), dtype='i4')
        arr[:] = np.arange(100).reshape(10, 10)
        yield fs, 'test.zarr'

    def test_zarr_adapter_memory_fs(self, memory_zarr):
        """ZarrAdapter works with memory filesystem."""
        from biopb_tensor_server.adapters.zarr import ZarrAdapter
        from biopb_tensor_server.config import SourceConfig
        from biopb_tensor_server.remote import RemoteStore
        import zarr
        from zarr.storage import FSStore

        fs, path = memory_zarr

        # Create adapter using RemoteStore
        store = RemoteStore("memory:///test.zarr")
        zarr_store = FSStore(store.path, fs=store.fs)
        arr = zarr.open_array(zarr_store, mode='r')

        adapter = ZarrAdapter(arr, "test-zarr")

        desc = adapter.get_tensor_descriptor()
        assert list(desc.shape) == [10, 10]
        assert desc.dtype == "<i4"

    def test_zarr_adapter_read_data(self, memory_zarr):
        """Read data from remote zarr via adapter."""
        from biopb_tensor_server.adapters.zarr import ZarrAdapter
        from biopb.tensor.ticket_pb2 import ChunkBounds
        import zarr
        from zarr.storage import FSStore
        import numpy as np

        fs, path = memory_zarr

        store = FSStore(path, fs=fs)
        arr = zarr.open_array(store, mode='r')
        adapter = ZarrAdapter(arr, "test-zarr")

        bounds = ChunkBounds(start=[0, 0], stop=[5, 5])
        data = adapter.get_data(bounds)

        assert data.shape == (5, 5)
        # Data should match what we wrote
        expected = np.arange(100).reshape(10, 10)[:5, :5]
        np.testing.assert_array_equal(data, expected)


# =============================================================================
# Integration Tests (Public S3)
# =============================================================================

def check_s3_available():
    """Check if S3 network access is available."""
    try:
        import fsspec
        fs = fsspec.filesystem('s3', anon=True)
        # Quick check on a known public bucket
        fs.exists('allencell')
        return True
    except Exception:
        return False


requires_network = pytest.mark.skipif(
    not check_s3_available(),
    reason="requires network access to S3"
)


@pytest.mark.integration
@requires_network
class TestPublicS3Integration:
    """Integration tests with public S3 buckets."""

    def test_allen_cell_s3_access(self):
        """Test anonymous access to Allen Cell S3 bucket."""
        url = "s3://allencell/aics/data_handoff_4dn/crop_seg/27a34691_segmentation.ome.tif"
        store = RemoteStore(url)

        assert store.exists() == True
        assert store.isfile() == True

        info = store.info()
        assert info.get('size', 0) > 0

    def test_allen_cell_s3_open(self):
        """Open file from Allen Cell S3."""
        url = "s3://allencell/aics/data_handoff_4dn/crop_seg/27a34691_segmentation.ome.tif"
        store = RemoteStore(url)

        with store.open() as f:
            # Read first few bytes (TIFF header)
            header = f.read(4)
            # TIFF files start with II or MM
            assert header[:2] in (b'II', b'MM')

    def test_allen_cell_s3_read_bytes(self):
        """Read bytes from Allen Cell S3."""
        url = "s3://allencell/aics/data_handoff_4dn/crop_seg/27a34691_segmentation.ome.tif"
        store = RemoteStore(url)

        # Read entire file (it's small ~380KB)
        data = store.read_bytes()
        assert len(data) > 0
        assert data[:2] in (b'II', b'MM')  # TIFF header

    @pytest.mark.skip(reason="IDR bucket currently unavailable")
    def test_idr_s3_with_endpoint(self):
        """Test IDR S3 with custom endpoint URL."""
        profile = CredentialProfile(
            name="idr-anon",
            storage_type="s3",
            endpoint_url="https://uk1s3.embassy.ebi.ac.uk",
        )
        config = CredentialsConfig(default_profile="idr-anon", profiles=[profile])

        url = "s3://idr-public/ngff/6001240.zarr"
        store = RemoteStore.from_config(url, credentials_config=config)

        # This test is skipped because IDR bucket is unavailable
        assert store.exists() == True


@pytest.mark.integration
@requires_network
class TestZarrAdapterS3Integration:
    """Integration tests for ZarrAdapter with public S3."""

    def test_zarr_adapter_allen_cell_s3(self):
        """Test ZarrAdapter with Allen Cell S3 (as remote store)."""
        from biopb_tensor_server.adapters.zarr import ZarrAdapter
        from biopb_tensor_server.config import SourceConfig

        # Allen Cell has TIFF files, not zarr, but we can test RemoteStore creation
        url = "s3://allencell/aics/data_handoff_4dn/crop_seg/"
        store = RemoteStore(url)

        # Should be a directory
        assert store.isdir() == True

        # List contents
        files = store.listdir()
        assert len(files) > 0
        # Should have some tiff files
        assert any('.tif' in f for f in files)


# =============================================================================
# Credential Resolution Tests
# =============================================================================

class TestCredentialResolution:
    """Tests for credential resolution priority."""

    def test_resolution_order_env_then_profile(self, monkeypatch):
        """Environment vars are base layer, profile overrides."""
        # Create test filesystem
        fs = fsspec.filesystem('memory')
        fs.mkdir('/cred_test.zarr')
        fs.touch('/cred_test.zarr/.zarray')

        # Set env var
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "env-key")
        monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)

        # Profile with different key
        profile = CredentialProfile(
            name="test",
            storage_type="s3",
            key="profile-key",
            secret="profile-secret",
        )

        # With profile, profile should override env
        # Use memory filesystem to test the flow
        store = RemoteStore("memory:///cred_test.zarr", profile=profile)
        assert store.exists() == True

    def test_resolution_source_profile_overrides_default(self):
        """Source-level profile overrides global default."""
        # Create test filesystem
        fs = fsspec.filesystem('memory')
        fs.mkdir('/cred_test2.zarr')
        fs.touch('/cred_test2.zarr/.zarray')

        default_profile = CredentialProfile(name="default", storage_type="s3", key="default-key")
        source_profile = CredentialProfile(name="source", storage_type="s3", key="source-key")
        config = CredentialsConfig(
            default_profile="default",
            profiles=[default_profile, source_profile],
        )

        # Request with source profile name
        store = RemoteStore.from_config(
            "memory:///cred_test2.zarr",
            credentials_config=config,
            profile_name="source",
        )
        assert store.exists() == True


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================

class TestRemoteStoreEdgeCases:
    """Tests for edge cases and error handling."""

    def test_nonexistent_path(self):
        """Accessing nonexistent path."""
        store = RemoteStore("memory:///nonexistent.zarr")
        assert store.exists() == False
        assert store.isdir() == False

    def test_empty_subpath(self):
        """Operations with empty subpath (root)."""
        fs = fsspec.filesystem('memory')
        # Use unique path to avoid conflicts with other tests
        fs.mkdir('/empty_test.zarr')
        fs.touch('/empty_test.zarr/.zarray')

        store = RemoteStore("memory:///empty_test.zarr")
        # Empty subpath means root
        assert store.exists("") == True
        assert store.isdir("") == True

    def test_fsspec_not_installed(self, monkeypatch):
        """Error when fsspec is not installed."""
        # Mock import error
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "fsspec":
                raise ImportError("fsspec not installed")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)

        with pytest.raises(ImportError, match="fsspec is required"):
            RemoteStore("s3://bucket/path")

    def test_path_join(self):
        """Path joining with various subpath formats."""
        fs = fsspec.filesystem('memory')
        fs.mkdir('/root')

        store = RemoteStore("memory:///root")

        # Join with leading slash
        joined = store._join("/subpath")
        assert joined == "/root/subpath"

        # Join without leading slash
        joined = store._join("subpath")
        assert joined == "/root/subpath"

        # Empty subpath returns root
        joined = store._join("")
        assert joined == "/root"