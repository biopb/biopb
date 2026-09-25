"""One spelling for a server location (``biopb.tensor._location``).

The canonical form is a cross-SDK contract -- the disk chunk cache names its
per-server directory after it, so a second SDK sharing that tree has to
land on the same string. These are the vectors.
"""

import pytest
from biopb.tensor._location import canonical_location, location_host


@pytest.mark.parametrize(
    ("raw", "expect"),
    [
        # Scheme aliases. grpc:// is the shorthand Python callers write;
        # grpc+tcp:// is what Arrow's own Location factories emit.
        ("grpc://h:8815", "grpc+tcp://h:8815"),
        ("grpc+tcp://h:8815", "grpc+tcp://h:8815"),
        ("grpcs://h:8815", "grpc+tls://h:8815"),
        ("grpc+tls://h:8815", "grpc+tls://h:8815"),
        # A scheme-less authority is the insecure default, matching Arrow.
        ("h:8815", "grpc+tcp://h:8815"),
        # Case and trailing slash are not identity.
        ("GRPC://H:8815", "grpc+tcp://h:8815"),
        ("grpc://h:8815/", "grpc+tcp://h:8815"),
        ("  grpc://h:8815  ", "grpc+tcp://h:8815"),
        # IPv6 keeps its brackets; hex is case-insensitive.
        ("grpc://[::1]:8815", "grpc+tcp://[::1]:8815"),
        ("grpc://[FE80::1]:8815", "grpc+tcp://[fe80::1]:8815"),
        # A unix socket has no authority, and its path IS the address.
        ("grpc+unix:///tmp/s.sock", "grpc+unix:///tmp/s.sock"),
    ],
)
def test_canonical_form(raw, expect):
    assert canonical_location(raw) == expect


def test_it_is_idempotent():
    once = canonical_location("grpc://h:8815")
    assert canonical_location(once) == once


def test_a_missing_port_is_not_invented():
    """Defaulting to 8815 would merge two spellings of a location that cannot be
    connected to either way, and guess at a server nobody named."""
    assert canonical_location("grpc://h") == "grpc+tcp://h"


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("grpc://a:8815", "grpc://b:8815"),  # host
        ("grpc://h:8815", "grpc://h:8816"),  # port
        ("grpc://h:8815", "grpc+tls://h:8815"),  # transport
        # Not resolved: merging these would put a DNS lookup on the chunk-fetch
        # path, for a server the disk cache skips anyway.
        ("grpc://localhost:8815", "grpc://127.0.0.1:8815"),
    ],
)
def test_distinct_locations_stay_distinct(a, b):
    assert canonical_location(a) != canonical_location(b)


def test_an_unparseable_location_falls_back_to_itself():
    """Callers key caches with this on the fetch path, so it must not raise. A
    string that only ever matches itself costs a miss, which is recoverable."""
    bad = "grpc://h:99999"  # port out of range
    assert canonical_location(bad) == bad
    assert canonical_location(bad) == canonical_location(bad)


@pytest.mark.parametrize(
    ("raw", "expect"),
    [
        ("grpc://localhost:8815", "localhost"),
        ("grpc+tcp://LOCALHOST:8815", "localhost"),
        ("localhost:8815", "localhost"),
        ("grpc://[::1]:8815", "::1"),  # unbracketed, for comparison
        ("grpc+unix:///tmp/s.sock", ""),
        ("", ""),
    ],
)
def test_location_host(raw, expect):
    assert location_host(raw) == expect
