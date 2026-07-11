"""Unit tests for the shared, stdlib-only web-auth predicates.

:mod:`biopb._web_auth` is the single source of the token / same-origin / loopback
decisions used by the control, the tensor sidecar, and observe (none can import
another). The predicates take a case-insensitive header getter; these tests drive
them with a plain dict getter. See ``mcp-dedaemonization-migration.md`` §6.1.
"""

from biopb import _web_auth as wa


def _get(**headers):
    """A case-insensitive getter over the given headers (like Starlette's)."""
    low = {k.lower().replace("_", "-"): v for k, v in headers.items()}
    return low.get


class TestExtractBearer:
    def test_authorization_bearer(self):
        assert wa.extract_bearer(_get(authorization="Bearer abc")) == "abc"

    def test_x_biopb_token_fallback(self):
        assert wa.extract_bearer(_get(x_biopb_token="xyz")) == "xyz"

    def test_bearer_wins_over_x_header(self):
        assert (
            wa.extract_bearer(_get(authorization="Bearer a", x_biopb_token="b")) == "a"
        )

    def test_absent_is_empty(self):
        assert wa.extract_bearer(_get()) == ""

    def test_non_bearer_authorization_falls_through(self):
        # A Basic auth header is not a bearer; fall back to X-Biopb-Token.
        assert (
            wa.extract_bearer(_get(authorization="Basic zzz", x_biopb_token="t")) == "t"
        )


class TestTokenValid:
    def test_no_expected_token_is_open(self):
        # No token configured -> not enforced (matches the sidecar dev/no-token path).
        assert wa.token_valid(_get(), None) is True
        assert wa.token_valid(_get(), "") is True

    def test_correct_token(self):
        assert wa.token_valid(_get(authorization="Bearer s3cret"), "s3cret") is True

    def test_wrong_token(self):
        assert wa.token_valid(_get(authorization="Bearer nope"), "s3cret") is False

    def test_missing_token(self):
        assert wa.token_valid(_get(), "s3cret") is False


class TestForgeableCrossSite:
    def test_token_header_is_never_forgeable(self):
        # A cross-origin no-cors fetch can't set Authorization, so its presence
        # means the request is not a CSRF vector -- even if Sec-Fetch-Site says so.
        g = _get(authorization="Bearer t", sec_fetch_site="cross-site")
        assert wa.is_forgeable_cross_site(g) is False

    def test_no_sec_fetch_site_is_not_forgeable(self):
        # Non-browser client (curl/urllib) -> can't be driven by a victim browser.
        assert wa.is_forgeable_cross_site(_get()) is False

    def test_same_origin_and_none_are_safe(self):
        assert wa.is_forgeable_cross_site(_get(sec_fetch_site="same-origin")) is False
        assert wa.is_forgeable_cross_site(_get(sec_fetch_site="none")) is False

    def test_cross_site_without_token_is_forgeable(self):
        assert wa.is_forgeable_cross_site(_get(sec_fetch_site="cross-site")) is True

    def test_cross_origin_without_token_is_forgeable(self):
        assert wa.is_forgeable_cross_site(_get(sec_fetch_site="cross-origin")) is True


class TestHostIsLoopback:
    def test_plain_loopback_ipv4(self):
        assert wa.host_is_loopback("127.0.0.1") is True
        assert wa.host_is_loopback("127.0.0.1:8813") is True

    def test_localhost(self):
        assert wa.host_is_loopback("localhost") is True
        assert wa.host_is_loopback("localhost:8813") is True

    def test_ipv6_loopback_bracketed(self):
        assert wa.host_is_loopback("[::1]") is True
        assert wa.host_is_loopback("[::1]:8813") is True

    def test_external_host_is_not_loopback(self):
        assert wa.host_is_loopback("evil.example") is False
        assert wa.host_is_loopback("evil.example:8813") is False
        assert wa.host_is_loopback("10.0.0.5:8813") is False

    def test_absent_host_is_not_loopback(self):
        assert wa.host_is_loopback(None) is False
        assert wa.host_is_loopback("") is False
