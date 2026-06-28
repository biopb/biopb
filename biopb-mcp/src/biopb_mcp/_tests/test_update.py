"""Tests for the kernel-start auto-updater version check (_update.py, issue #87).

The pure helpers (version parse/compare, release selection, marker read) are
tested directly; check_for_update is tested with its two I/O seams stubbed —
``_http_json`` (the GitHub fetch) and ``biopb_mcp.__version__`` (the running
build) — so no network is touched and the fail-open contract is exercised.
"""

import pytest

from biopb_mcp._config import get_default_config
from biopb_mcp.mcp import _update


def _release(tag, *, prerelease=False, draft=False, url="https://example/r"):
    return {
        "tag_name": tag,
        "prerelease": prerelease,
        "draft": draft,
        "html_url": url,
    }


class TestPureHelpers:
    @pytest.mark.parametrize(
        "version,is_dev",
        [
            ("0.6.6", False),
            ("0.7.0rc1", False),
            ("0.7.3.dev6+g1a2b3c4", True),
            ("0.7.3+g1a2b3c4", True),
            ("0.7.3.dev0", True),
            ("unknown", False),
        ],
    )
    def test_is_dev_version(self, version, is_dev):
        assert _update.is_dev_version(version) is is_dev

    def test_tag_to_version_strips_prefix(self):
        assert _update.tag_to_version("release-v0.6.6") == "0.6.6"

    def test_tag_to_version_leaves_nonmatching(self):
        # A v*/mcp-v* PyPI tag is not on the deployment line — left untouched
        # (and later rejected by the prefix filter in select_latest).
        assert _update.tag_to_version("mcp-v0.7.2") == "mcp-v0.7.2"

    def test_select_latest_picks_highest_clean(self):
        releases = [
            _release("release-v0.6.4"),
            _release("release-v0.6.6"),
            _release("release-v0.6.5"),
            _release("v0.9.9"),  # PyPI library tag, wrong prefix -> ignored
            _release("mcp-v9.9.9"),  # ditto
        ]
        chosen = _update.select_latest(releases)
        assert chosen["tag_name"] == "release-v0.6.6"

    def test_select_latest_skips_prereleases_by_default(self):
        releases = [
            _release("release-v0.6.6"),
            _release("release-v0.7.0rc1", prerelease=True),
        ]
        assert _update.select_latest(releases)["tag_name"] == "release-v0.6.6"

    def test_select_latest_allows_prereleases_when_opted_in(self):
        releases = [
            _release("release-v0.6.6"),
            _release("release-v0.7.0rc1", prerelease=True),
        ]
        chosen = _update.select_latest(releases, allow_prerelease=True)
        assert chosen["tag_name"] == "release-v0.7.0rc1"

    def test_select_latest_skips_rc_by_pep440_even_without_flag(self):
        # GitHub's prerelease flag missing, but the version is a PEP 440 rc.
        releases = [
            _release("release-v0.6.6"),
            _release("release-v0.7.0rc1", prerelease=False),
        ]
        assert _update.select_latest(releases)["tag_name"] == "release-v0.6.6"

    def test_select_latest_skips_drafts(self):
        releases = [
            _release("release-v0.6.6"),
            _release("release-v0.8.0", draft=True),
        ]
        assert _update.select_latest(releases)["tag_name"] == "release-v0.6.6"

    def test_select_latest_tolerates_malformed(self):
        releases = [
            "not-a-dict",
            {"tag_name": None},
            _release("release-vnot.a.version"),
            _release("release-v0.6.6"),
        ]
        assert _update.select_latest(releases)["tag_name"] == "release-v0.6.6"

    def test_select_latest_none_when_empty(self):
        assert _update.select_latest([]) is None
        assert _update.select_latest([_release("v1.0.0")]) is None


class TestReadInstalledVersion:
    def test_reads_and_strips(self, tmp_path):
        p = tmp_path / "release.version"
        p.write_text("0.6.6\n", encoding="utf-8")
        assert _update.read_installed_version(p) == "0.6.6"

    def test_missing_marker_is_none(self, tmp_path):
        assert _update.read_installed_version(tmp_path / "nope") is None

    def test_empty_marker_is_none(self, tmp_path):
        p = tmp_path / "release.version"
        p.write_text("  \n", encoding="utf-8")
        assert _update.read_installed_version(p) is None

    def test_default_path_is_umbrella_config_dir(self):
        # Contract with the installer: ~/.config/biopb/release.version (NOT
        # ~/.config/biopb-mcp). conftest points Path.home() at a tmp dir.
        p = _update.marker_path()
        assert p.parts[-3:] == (".config", "biopb", "release.version")


class TestCheckForUpdate:
    """check_for_update with the network + running-version seams stubbed."""

    @pytest.fixture
    def config(self):
        return get_default_config()

    @pytest.fixture
    def stub_releases(self, monkeypatch):
        """Make _http_json return a fixed releases payload (no network)."""

        def _install(payload):
            monkeypatch.setattr(
                _update, "_http_json", lambda url, timeout: payload
            )

        return _install

    @pytest.fixture(autouse=True)
    def _clean_running_version(self, monkeypatch):
        # Default the running build to a clean release version so the
        # dev-suppression guard doesn't short-circuit tests run from an editable
        # (+gSHA) checkout. Individual tests override as needed.
        monkeypatch.setattr(_update, "_running_library_version", lambda: "0.7.0")

    def test_newer_release_returns_update(self, config, stub_releases):
        stub_releases([_release("release-v0.7.0"), _release("release-v0.6.6")])
        info = _update.check_for_update(config, installed="0.6.6")
        assert info is not None
        assert info.current == "0.6.6"
        assert info.latest == "0.7.0"
        assert info.tag == "release-v0.7.0"
        assert info.html_url == "https://example/r"

    def test_same_version_no_update(self, config, stub_releases):
        stub_releases([_release("release-v0.6.6")])
        assert _update.check_for_update(config, installed="0.6.6") is None

    def test_older_remote_no_update(self, config, stub_releases):
        stub_releases([_release("release-v0.6.5")])
        assert _update.check_for_update(config, installed="0.6.6") is None

    def test_skipped_version_suppressed(self, config, stub_releases):
        config["mcp"]["update"]["skipped_version"] = "0.7.0"
        stub_releases([_release("release-v0.7.0")])
        assert _update.check_for_update(config, installed="0.6.6") is None

    def test_disabled_returns_none(self, config, stub_releases):
        config["mcp"]["update"]["enabled"] = False
        stub_releases([_release("release-v0.7.0")])
        assert _update.check_for_update(config, installed="0.6.6") is None

    def test_dev_build_suppressed(self, config, stub_releases, monkeypatch):
        monkeypatch.setattr(
            _update, "_running_library_version", lambda: "0.7.0.dev3+gdeadbee"
        )
        stub_releases([_release("release-v0.7.0")])
        assert _update.check_for_update(config, installed="0.6.6") is None

    def test_no_marker_suppressed(self, config, stub_releases):
        # installed=None and the (tmp) marker file does not exist.
        stub_releases([_release("release-v0.7.0")])
        assert _update.check_for_update(config) is None

    def test_reads_marker_when_installed_not_passed(
        self, config, stub_releases, tmp_path, monkeypatch
    ):
        marker = tmp_path / "release.version"
        marker.write_text("0.6.6", encoding="utf-8")
        monkeypatch.setattr(_update, "marker_path", lambda: marker)
        stub_releases([_release("release-v0.7.0")])
        info = _update.check_for_update(config)
        assert info is not None and info.latest == "0.7.0"

    def test_network_error_fails_open(self, config, monkeypatch):
        def _boom(url, timeout):
            raise OSError("offline")

        monkeypatch.setattr(_update, "_http_json", _boom)
        assert _update.check_for_update(config, installed="0.6.6") is None

    def test_error_payload_fails_open(self, config, stub_releases):
        # GitHub rate-limit / 404 returns a JSON object, not a list.
        stub_releases({"message": "API rate limit exceeded"})
        assert _update.check_for_update(config, installed="0.6.6") is None

    def test_prerelease_channel_picks_rc(self, config, stub_releases):
        config["mcp"]["update"]["channel"] = "prerelease"
        stub_releases(
            [
                _release("release-v0.6.6"),
                _release("release-v0.7.0rc1", prerelease=True),
            ]
        )
        info = _update.check_for_update(config, installed="0.6.6")
        assert info is not None and info.latest == "0.7.0rc1"
