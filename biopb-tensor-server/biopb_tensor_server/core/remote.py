"""Remote storage abstraction using fsspec.

Provides unified access to remote storage (S3, GCS, HTTP, etc.) via fsspec.
Handles credential management, URL parsing, and file operations.

Credential Resolution Order:
1. Signed URL params: If URL contains ?AWSAccessKeyId=..., use those
2. Source-level profile: If source config has credentials_profile="aws-prod"
3. Global default profile: If server config has credentials.default_profile="aws-prod"
4. Environment variables: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, etc.
"""

from __future__ import annotations

import logging
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any, Dict, Iterator, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

logger = logging.getLogger(__name__)


@dataclass
class CredentialProfile:
    """Named credential profile for remote storage.

    Per-field help lives in each field's ``metadata["help"]`` (read by the config
    JSON Schema).
    """

    name: str = field(
        metadata={"help": "Profile name a source references via credentials_profile."}
    )
    storage_type: str = field(
        metadata={"help": "Remote storage backend (s3, gs, http, azure)."}
    )
    key: Optional[str] = field(
        default=None, metadata={"help": "Access key (AWS: access_key_id)."}
    )
    secret: Optional[str] = field(
        default=None, metadata={"help": "Secret key (AWS: secret_access_key)."}
    )
    region: Optional[str] = field(
        default=None, metadata={"help": "Region (e.g. S3 us-east-1)."}
    )
    token: Optional[str] = field(
        default=None,
        metadata={
            "help": "Session token (temporary credentials) or GCS service-account "
            "JSON path."
        },
    )
    endpoint_url: Optional[str] = field(
        default=None,
        metadata={"help": "Custom endpoint for S3-compatible storage (e.g. MinIO)."},
    )

    def to_storage_options(self) -> Dict[str, Any]:
        """Convert profile to fsspec storage_options dict.

        Returns:
            Dictionary of storage options for fsspec filesystem creation
        """
        options: Dict[str, Any] = {}

        if self.storage_type == "s3":
            # S3/S3-compatible storage options
            # Anonymous access if both key and secret are empty
            if not self.key and not self.secret:
                options["anon"] = True
            else:
                if self.key:
                    options["key"] = self.key
                if self.secret:
                    options["secret"] = self.secret
                if self.token:
                    options["token"] = self.token
            if self.region:
                options["client_kwargs"] = {"region_name": self.region}
            if self.endpoint_url:
                options["client_kwargs"] = options.get("client_kwargs", {})
                options["client_kwargs"]["endpoint_url"] = self.endpoint_url

        elif self.storage_type == "gs":
            # GCS storage options
            if self.token:
                # token can be a path to service account JSON or a dict
                if os.path.isfile(self.token):
                    options["token"] = self.token
                else:
                    # Could be inline JSON or None (use default credentials)
                    options["token"] = self.token

        elif self.storage_type == "azure":
            # Azure Blob Storage options
            if self.key:
                options["account_name"] = self.key
            if self.secret:
                options["account_key"] = self.secret
            if self.token:
                options["connection_string"] = self.token

        return options


@dataclass
class CredentialsConfig:
    """Credentials configuration for remote storage.

    Attributes:
        default_profile: Name of the default profile to use
        profiles: List of credential profiles
    """

    default_profile: Optional[str] = None
    profiles: List[CredentialProfile] = field(default_factory=list)

    def get_profile(self, name: Optional[str] = None) -> Optional[CredentialProfile]:
        """Get a credential profile by name.

        Args:
            name: Profile name, or None to use default_profile

        Returns:
            CredentialProfile if found, None otherwise
        """
        profile_name = name or self.default_profile
        if profile_name is None:
            return None

        for profile in self.profiles:
            if profile.name == profile_name:
                return profile

        return None


def _extract_signed_url_params(url: str) -> Dict[str, Any]:
    """Extract signed URL parameters from URL query string.

    Handles AWS S3 presigned URLs and similar signed URL patterns.

    Args:
        url: URL potentially containing signed URL params

    Returns:
        Dictionary of storage options extracted from URL params
    """
    parsed = urlparse(url)
    query_params = parse_qs(parsed.query)

    options: Dict[str, Any] = {}

    # AWS S3 signed URL params. Signature/Expires are carried in the URL itself
    # and handled by fsspec, so only the credential params are extracted here.
    if "AWSAccessKeyId" in query_params:
        options["key"] = query_params["AWSAccessKeyId"][0]
    if "X-Amz-Security-Token" in query_params:
        options["token"] = query_params["X-Amz-Security-Token"][0]

    # Google Cloud Storage signed URL params
    if "GoogleAccessId" in query_params:
        options["key"] = query_params["GoogleAccessId"][0]

    return options


def _get_env_credentials(storage_type: str) -> Dict[str, Any]:
    """Get credentials from environment variables.

    Args:
        storage_type: Storage type ("s3", "gs", "azure")

    Returns:
        Dictionary of storage options from environment
    """
    options: Dict[str, Any] = {}

    if storage_type == "s3":
        # Standard AWS environment variables
        if os.environ.get("AWS_ACCESS_KEY_ID"):
            options["key"] = os.environ["AWS_ACCESS_KEY_ID"]
        if os.environ.get("AWS_SECRET_ACCESS_KEY"):
            options["secret"] = os.environ["AWS_SECRET_ACCESS_KEY"]
        if os.environ.get("AWS_SESSION_TOKEN"):
            options["token"] = os.environ["AWS_SESSION_TOKEN"]
        if os.environ.get("AWS_DEFAULT_REGION"):
            options["client_kwargs"] = {"region_name": os.environ["AWS_DEFAULT_REGION"]}

    elif storage_type == "gs":
        # GCS uses GOOGLE_APPLICATION_CREDENTIALS for service account JSON
        if os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
            options["token"] = os.environ["GOOGLE_APPLICATION_CREDENTIALS"]

    elif storage_type == "azure":
        # Azure environment variables
        if os.environ.get("AZURE_STORAGE_ACCOUNT_NAME"):
            options["account_name"] = os.environ["AZURE_STORAGE_ACCOUNT_NAME"]
        if os.environ.get("AZURE_STORAGE_ACCOUNT_KEY"):
            options["account_key"] = os.environ["AZURE_STORAGE_ACCOUNT_KEY"]
        if os.environ.get("AZURE_STORAGE_CONNECTION_STRING"):
            options["connection_string"] = os.environ["AZURE_STORAGE_CONNECTION_STRING"]

    return options


def _detect_storage_type(url: str) -> str:
    """Detect storage type from URL scheme.

    Args:
        url: URL to analyze

    Returns:
        Storage type string ("s3", "gs", "http", "azure", "file")
    """
    parsed = urlparse(url)
    scheme = parsed.scheme.lower()

    if scheme == "s3":
        return "s3"
    elif scheme == "gs" or scheme == "gcs":
        return "gs"
    elif scheme in ("http", "https"):
        return "http"
    elif scheme == "az" or scheme == "azure":
        return "azure"
    elif scheme == "file" or not scheme:
        return "file"
    else:
        return scheme


class RemoteStore:
    """Abstraction for remote storage via fsspec.

    Wraps fsspec filesystem with credential injection and provides
    unified interface for file operations on remote storage.

    Attributes:
        url: Original URL
        profile: Credential profile (if used)
        fs: fsspec AbstractFileSystem instance
        path: Path component of URL (without scheme)
    """

    def __init__(
        self,
        url: str,
        profile: Optional[CredentialProfile] = None,
        credentials_config: Optional[CredentialsConfig] = None,
        source_profile_name: Optional[str] = None,
    ):
        """Initialize RemoteStore.

        Args:
            url: Remote URL (s3://..., gs://..., etc.)
            profile: Explicit credential profile (overrides config lookup)
            credentials_config: CredentialsConfig for profile lookup
            source_profile_name: Profile name from source config
        """
        self.url = url
        self._profile = profile
        self._credentials_config = credentials_config
        self._source_profile_name = source_profile_name

        self.fs, self.path = self._create_fs()

    def _create_fs(self) -> Tuple[Any, str]:
        """Create fsspec filesystem from URL and credentials.

        Credential resolution order:
        1. Signed URL params (extracted from URL)
        2. Explicit profile (passed to constructor)
        3. Source-level profile name (from source config)
        4. Global default profile (from credentials_config)
        5. Environment variables

        Returns:
            Tuple of (filesystem, path)
        """
        try:
            import fsspec
        except ImportError:
            raise ImportError(
                "fsspec is required for remote storage support. "
                "Install with: pip install fsspec s3fs gcsfs"
            )

        storage_type = _detect_storage_type(self.url)

        # Build storage_options by merging sources (priority order)
        storage_options: Dict[str, Any] = {}

        # 1. Environment variables (base layer)
        env_options = _get_env_credentials(storage_type)
        storage_options.update(env_options)

        # 2. Global default profile
        if self._credentials_config:
            default_profile = self._credentials_config.get_profile()
            if default_profile:
                profile_options = default_profile.to_storage_options()
                storage_options.update(profile_options)

        # 3. Source-level profile (overrides global)
        if self._credentials_config and self._source_profile_name:
            source_profile = self._credentials_config.get_profile(
                self._source_profile_name
            )
            if source_profile:
                profile_options = source_profile.to_storage_options()
                storage_options.update(profile_options)

        # 4. Explicit profile (overrides all)
        if self._profile:
            profile_options = self._profile.to_storage_options()
            storage_options.update(profile_options)

        # 5. Signed URL params (highest priority - embedded in URL)
        signed_options = _extract_signed_url_params(self.url)
        storage_options.update(signed_options)

        # 6. Fallback: For S3, set anon=True if no credentials found
        # This allows public bucket access without explicit configuration
        if storage_type == "s3" and not storage_options:
            storage_options["anon"] = True

        # Create filesystem using fsspec.open()
        # This handles protocol detection and filesystem instantiation
        # storage_options are passed as kwargs to the filesystem
        open_file = fsspec.open(self.url, **storage_options)
        fs = open_file.fs
        path = open_file.path

        return fs, path

    @classmethod
    def from_config(
        cls,
        url: str,
        credentials_config: Optional[CredentialsConfig] = None,
        profile_name: Optional[str] = None,
    ) -> RemoteStore:
        """Create RemoteStore from configuration.

        Args:
            url: Remote URL
            credentials_config: CredentialsConfig with profiles
            profile_name: Optional profile name to use (overrides default)

        Returns:
            RemoteStore instance
        """
        return cls(
            url=url,
            credentials_config=credentials_config,
            source_profile_name=profile_name,
        )

    def walk(
        self,
        maxdepth: Optional[int] = None,
    ) -> Iterator[Tuple[str, List[str], List[str]]]:
        """Recursive directory listing (like os.walk).

        Args:
            maxdepth: Maximum depth to walk (None for unlimited)

        Yields:
            Tuple of (dirpath, dirnames, filenames) for each directory
        """
        return self.fs.walk(self.path, maxdepth=maxdepth)

    def find(
        self,
        pattern: str = "*",
        maxdepth: Optional[int] = None,
        withdirs: bool = False,
    ) -> List[str]:
        """Find files matching pattern.

        Args:
            pattern: Glob pattern to match
            maxdepth: Maximum depth to search
            withdirs: Include directories in results

        Returns:
            List of matching paths
        """
        return self.fs.find(
            self.path, pattern=pattern, maxdepth=maxdepth, withdirs=withdirs
        )

    def exists(self, subpath: str = "") -> bool:
        """Check if path exists.

        Args:
            subpath: Subpath relative to root (empty string for root)

        Returns:
            True if path exists
        """
        full_path = self._join(subpath)
        return self.fs.exists(full_path)

    def isdir(self, subpath: str = "") -> bool:
        """Check if path is directory.

        Args:
            subpath: Subpath relative to root

        Returns:
            True if path is directory
        """
        full_path = self._join(subpath)
        return self.fs.isdir(full_path)

    def isfile(self, subpath: str = "") -> bool:
        """Check if path is file.

        Args:
            subpath: Subpath relative to root

        Returns:
            True if path is file
        """
        full_path = self._join(subpath)
        return self.fs.isfile(full_path)

    def open(self, subpath: str = "", mode: str = "rb") -> IO:
        """Open file for reading.

        Args:
            subpath: Subpath relative to root (empty string for root file)
            mode: File mode (default 'rb' for binary read)

        Returns:
            File-like object
        """
        full_path = self._join(subpath)
        return self.fs.open(full_path, mode=mode)

    def read_bytes(self, subpath: str = "") -> bytes:
        """Read file contents as bytes.

        Args:
            subpath: Subpath relative to root

        Returns:
            File contents as bytes
        """
        full_path = self._join(subpath)
        return self.fs.cat_file(full_path)

    def read_text(self, subpath: str = "", encoding: str = "utf-8") -> str:
        """Read file contents as text.

        Args:
            subpath: Subpath relative to root
            encoding: Text encoding (default utf-8)

        Returns:
            File contents as string
        """
        return self.read_bytes(subpath).decode(encoding)

    def listdir(self, subpath: str = "") -> List[str]:
        """List contents of a directory.

        Args:
            subpath: Subpath relative to root

        Returns:
            List of filenames/directnames in the directory
        """
        full_path = self._join(subpath)
        return self.fs.ls(full_path, detail=False)

    def info(self, subpath: str = "") -> Dict[str, Any]:
        """Get file/directory info.

        Args:
            subpath: Subpath relative to root

        Returns:
            Dictionary with info (size, type, mtime, etc.)
        """
        full_path = self._join(subpath)
        return self.fs.info(full_path)

    def get_identity(self, subpath: str = "") -> str:
        """Get unique identity for deduplication.

        Uses path + size + etag/mtime since remote storage doesn't have inodes.

        Args:
            subpath: Subpath relative to root

        Returns:
            Unique identity string
        """
        full_path = self._join(subpath)
        try:
            info = self.fs.info(full_path)
            size = info.get("size", 0)
            # S3 uses ETag, other storage uses mtime
            etag = info.get("ETag", "") or info.get("etag", "")
            mtime = info.get("mtime", "") or info.get("LastModified", "")
            return f"{full_path}:{size}:{etag or mtime}"
        except Exception:
            # Fallback to path hash
            import hashlib

            return hashlib.sha256(full_path.encode()).hexdigest()[:16]

    def _join(self, subpath: str) -> str:
        """Join root path with subpath.

        Args:
            subpath: Subpath relative to root

        Returns:
            Full path
        """
        if not subpath:
            return self.path
        # Ensure no leading slash in subpath
        subpath = subpath.lstrip("/")
        return f"{self.path.rstrip('/')}/{subpath}"

    def download_to_temp(self, subpath: str = "", suffix: Optional[str] = None) -> Path:
        """Download remote file to a temporary local file.

        Useful for libraries that don't support fsspec directly (e.g., nibabel).

        Args:
            subpath: Subpath relative to root
            suffix: Optional suffix for temp file (e.g., '.nii')

        Returns:
            Path to temporary file (caller responsible for cleanup)
        """
        full_path = self._join(subpath)

        # Create temp file
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp_path = Path(tmp.name)

        # Download file
        self.fs.get_file(full_path, str(tmp_path))

        return tmp_path


def is_remote_url(url: str) -> bool:
    """Whether ``url`` is a remote (non-local) URL.

    The single canonical scheme predicate for the server: it decides whether a
    URL needs remote-storage handling (a :class:`RemoteStore`) or is a plain
    local filesystem path. Recognizes object-store, HTTP(S)/FTP, and the
    ``grpc*`` remote-tensor-cache schemes; everything else — including bare paths
    and ``file://`` — is treated as local.
    """
    remote_prefixes = (
        "s3://",
        "gs://",
        "gcs://",
        "http://",
        "https://",
        "ftp://",
        "az://",
        "azure://",
        "grpc://",
        "grpc+tls://",
        "grpcs://",
    )
    return url.lower().startswith(remote_prefixes)
