"""Unit tests for skill ``requires:`` resolution (biopb_mcp.mcp._requires).

The contract under test is asymmetric on purpose: a token is reported unmet only
when it can actually be decided in the MCP server process, and anything
undecidable stays silent -- find_skills is normally called before the kernel
exists, and a wrong "missing" argues the agent out of a skill that would have
worked. So these tests pin the *silence* as hard as the detections, plus the
fail-open behaviour when a check itself blows up.
"""

import pytest

from biopb_mcp.mcp import _requires


@pytest.fixture
def visible(monkeypatch):
    """A session with a viewer, plugins enabled, and no plugins installed."""
    monkeypatch.setattr(_requires, "_is_headless", lambda: False)
    monkeypatch.setattr(_requires, "_namespace_enabled", lambda: True)
    monkeypatch.setattr(_requires, "_available_plugins", lambda: set())


class TestUndecidableStaysSilent:
    @pytest.mark.parametrize(
        "token",
        [
            "tensor",  # connection state, kernel-side
            "dask",  # cluster state, kernel-side
            "ops:segmentation",  # built in the kernel from configured servers
            "ops:restoration",
            "pkg:basicpy",  # third-party: the kernel env may differ
            "pkg:m2stitch",
            "pkg:scikit-image",
            "something-we-invented",  # vocabulary grows in biopb-site, not here
            "",
        ],
    )
    def test_token_is_not_reported(self, visible, token):
        assert _requires.unmet([token]) == []

    def test_a_third_party_package_is_never_reported_even_if_absent(self, visible):
        # The skill's own step-2 import check is authoritative (and carries the
        # degraded path); this process's site-packages is not the kernel's.
        assert _requires.unmet(["pkg:definitely-not-installed-xyz"]) == []

    def test_no_requirements_is_empty(self, visible):
        assert _requires.unmet([]) == []


class TestViewer:
    def test_headless_session_reports_the_viewer_gap(self, monkeypatch, visible):
        monkeypatch.setattr(_requires, "_is_headless", lambda: True)
        (msg,) = _requires.unmet(["viewer"])
        assert msg.startswith("viewer — ")
        assert "headless" in msg
        assert "numeric fallback" in msg  # the actionable part

    def test_visible_session_reports_nothing(self, visible):
        assert _requires.unmet(["viewer"]) == []


class TestPlugin:
    def test_missing_plugin_is_reported_with_the_seed_command(self, visible):
        (msg,) = _requires.unmet(["plugin:segmentation_qc"])
        assert msg.startswith("plugin:segmentation_qc — ")
        assert "biopb-mcp-seed-plugins" in msg
        assert "ask first" in msg  # restarting the kernel needs consent

    def test_present_plugin_is_not_reported(self, monkeypatch, visible):
        monkeypatch.setattr(
            _requires, "_available_plugins", lambda: {"segmentation_qc"}
        )
        assert _requires.unmet(["plugin:segmentation_qc"]) == []

    def test_disabled_namespace_reports_the_switch_not_the_seed_command(
        self, monkeypatch, visible
    ):
        # With plugins switched off, seeding a file would not help — the reason
        # has to name the switch instead.
        monkeypatch.setattr(_requires, "_namespace_enabled", lambda: False)
        monkeypatch.setattr(
            _requires, "_available_plugins", lambda: {"segmentation_qc"}
        )
        (msg,) = _requires.unmet(["plugin:segmentation_qc"])
        assert "services.namespace_enabled" in msg
        assert "biopb-mcp-seed-plugins" not in msg

    def test_plugin_names_resolve_from_the_kernel_dir_and_entry_points(
        self, monkeypatch, tmp_path
    ):
        # The real _available_plugins: a seeded *.py file and an entry point,
        # neither of which is imported to be listed.
        (tmp_path / "my_tool.py").write_text('"""Mine."""\n', encoding="utf-8")
        (tmp_path / "_private.py").write_text('"""Skipped."""\n', encoding="utf-8")
        monkeypatch.setattr("biopb._locations.mcp_plugin_dir", lambda: tmp_path)
        monkeypatch.setattr(
            "biopb._kernel_plugins.entry_point_plugins",
            lambda: [{"name": "lab_pkg", "dist": "lab 1.0"}],
        )
        names = _requires._available_plugins()
        assert {"my_tool", "lab_pkg"} <= names
        assert "_private" not in names  # leading underscore: loader skips it


class TestOwnVersion:
    @pytest.mark.parametrize(
        "installed,spec,ok",
        [
            ("0.11.0", ">=0.11.0", True),
            ("0.11.0", ">=0.12.0", False),
            ("0.12.1", ">=0.12", True),
            ("0.12", ">=0.12.0", True),  # padded, not lexicographic
            ("0.9.0", ">=0.10.0", False),  # numeric, not string, compare
            ("0.12.0", "==0.12", True),
            ("0.13.0", "==0.12", False),
            # A pre-release of the required version counts as meeting it: a
            # strict PEP 440 compare ranks every rc below its own final release
            # and would tell a dev build to upgrade to what it is running.
            ("0.12.0rc8.dev32+g9268773", ">=0.12.0", True),
            ("0.11.0rc8.dev32+g9268773", ">=0.12.0", False),
        ],
    )
    def test_version_comparison(self, monkeypatch, visible, installed, spec, ok):
        monkeypatch.setattr("importlib.metadata.version", lambda dist: installed)
        gaps = _requires.unmet([f"pkg:biopb-mcp{spec}"])
        assert (gaps == []) is ok
        if not ok:
            assert installed in gaps[0] and spec.lstrip(">=") in gaps[0]

    def test_bare_own_package_is_always_met(self, visible):
        # We are running from it, so no version to fail against.
        assert _requires.unmet(["pkg:biopb-mcp"]) == []

    def test_underscore_spelling_resolves_to_the_same_distribution(
        self, monkeypatch, visible
    ):
        monkeypatch.setattr("importlib.metadata.version", lambda dist: "0.1.0")
        assert _requires.unmet(["pkg:biopb_mcp>=99.0"]) != []

    def test_unparseable_version_is_not_reported(self, monkeypatch, visible):
        monkeypatch.setattr("importlib.metadata.version", lambda dist: "not-a-version")
        assert _requires.unmet(["pkg:biopb-mcp>=0.12"]) == []

    def test_unsupported_operator_is_not_reported(self, visible):
        # `<`, `!=`, `~=` are not part of the vocabulary; guessing would be worse
        # than saying nothing.
        assert _requires.unmet(["pkg:biopb-mcp<0.1"]) == []


class TestFailOpen:
    def test_a_raising_check_costs_the_annotation_not_the_skill(
        self, monkeypatch, visible
    ):
        def boom():
            raise RuntimeError("plugin dir unreadable")

        monkeypatch.setattr(_requires, "_available_plugins", boom)
        assert _requires.unmet(["plugin:anything"]) == []

    def test_one_bad_token_does_not_suppress_the_others(self, monkeypatch, visible):
        real = _requires._reason

        def flaky(token):
            if token == "plugin:boom":
                raise RuntimeError("nope")
            return real(token)

        monkeypatch.setattr(_requires, "_reason", flaky)
        monkeypatch.setattr(_requires, "_is_headless", lambda: True)
        assert _requires.unmet(["plugin:boom", "viewer"]) != []

    @pytest.mark.parametrize("junk", [None, "not-a-list", 42, {"viewer": True}])
    def test_malformed_requires_is_tolerated(self, visible, junk):
        assert _requires.unmet(junk) == []

    def test_non_string_tokens_are_skipped(self, visible):
        assert _requires.unmet([None, 3, ["viewer"]]) == []


class TestHeadlessSourceIsNotRestated:
    def test_viewer_check_follows_the_server_flag(self, monkeypatch):
        # _requires must read the launcher's flag rather than keep its own copy;
        # set_headless is the only writer.
        from biopb_mcp.mcp import _server

        monkeypatch.setattr(_requires, "_namespace_enabled", lambda: True)
        original = _server.is_headless()
        try:
            _server.set_headless(True)
            assert _requires.unmet(["viewer"]) != []
            _server.set_headless(False)
            assert _requires.unmet(["viewer"]) == []
        finally:
            _server.set_headless(original)
