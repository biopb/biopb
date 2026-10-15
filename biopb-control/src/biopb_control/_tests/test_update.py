"""The update check: release selection, the baseline marker, and fail-open."""

import sys

import pytest

from biopb_control import _update


def _release(tag, *, prerelease=False, draft=False, url="https://example/r"):
    return {
        "tag_name": tag,
        "prerelease": prerelease,
        "draft": draft,
        "html_url": url,
    }


@pytest.mark.parametrize(
    "version,is_dev",
    [
        ("0.6.6", False),
        ("0.7.0rc1", False),
        ("0.7.3.dev6+g1a2b3c4", True),
        ("0.7.3+g1a2b3c4", True),
        ("0.0.0", False),
    ],
)
def test_is_dev_version(version, is_dev):
    assert _update.is_dev_version(version) is is_dev


class TestSelectLatest:
    def test_picks_the_highest_stable_release(self):
        releases = [
            _release("release-v0.6.6"),
            _release("release-v0.7.0"),
            _release("release-v0.7.0rc1", prerelease=True),
            _release("v9.9.9"),  # an SDK tag, not the product line
            _release("release-v0.8.0", draft=True),
        ]
        assert _update.select_latest(releases)["tag_name"] == "release-v0.7.0"

    def test_prereleases_only_when_allowed(self):
        releases = [_release("release-v0.6.6"), _release("release-v0.7.0rc1")]
        assert _update.select_latest(releases)["tag_name"] == "release-v0.6.6"
        got = _update.select_latest(releases, allow_prerelease=True)
        assert got["tag_name"] == "release-v0.7.0rc1"

    def test_malformed_entries_are_skipped(self):
        releases = ["x", {}, _release("release-vNOPE"), _release("release-v0.6.6")]
        assert _update.select_latest(releases)["tag_name"] == "release-v0.6.6"
        assert _update.select_latest([]) is None


class TestCheck:
    @pytest.fixture
    def releases(self, monkeypatch):
        def _install(payload):
            monkeypatch.setattr(_update, "_fetch_releases", lambda: payload)

        return _install

    def test_a_newer_release_is_reported(self, releases):
        releases([_release("release-v0.7.0"), _release("release-v0.6.6")])
        info = _update.check("0.6.6", installed="0.6.6")
        assert info == _update.UpdateInfo("0.6.6", "0.7.0", "https://example/r")

    @pytest.mark.parametrize("remote", ["release-v0.6.6", "release-v0.6.5"])
    def test_same_or_older_is_not(self, releases, remote):
        releases([_release(remote)])
        assert _update.check("0.6.6", installed="0.6.6") is None

    def test_a_stable_install_ignores_release_candidates(self, releases):
        releases([_release("release-v0.7.0rc1", prerelease=True)])
        assert _update.check("0.6.6", installed="0.6.6") is None

    def test_a_candidate_install_follows_candidates(self, releases):
        releases([_release("release-v0.7.0rc2", prerelease=True)])
        info = _update.check("0.7.0rc1", installed="0.7.0rc1")
        assert info is not None and info.latest == "0.7.0rc2"

    def test_a_dev_build_is_never_checked(self, releases):
        releases([_release("release-v0.7.0")])
        assert _update.check("0.7.0.dev3+gdeadbee", installed="0.6.6") is None

    def test_no_marker_means_no_check(self, releases, tmp_path, monkeypatch):
        monkeypatch.setenv("BIOPB_CONFIG_HOME", str(tmp_path))
        releases([_release("release-v0.7.0")])
        assert _update.check("0.6.6") is None

    def test_the_marker_is_the_baseline(self, releases, tmp_path, monkeypatch):
        monkeypatch.setenv("BIOPB_CONFIG_HOME", str(tmp_path))
        (tmp_path / "biopb").mkdir()
        (tmp_path / "biopb" / "release.version").write_text("0.6.6\n")
        releases([_release("release-v0.7.0")])
        info = _update.check("0.6.6")
        assert info is not None and info.current == "0.6.6"

    def test_a_network_error_fails_open(self, monkeypatch):
        def boom():
            raise OSError("offline")

        monkeypatch.setattr(_update, "_fetch_releases", boom)
        assert _update.check("0.6.6", installed="0.6.6") is None

    def test_an_error_body_fails_open(self, releases):
        releases({"message": "API rate limit exceeded"})  # an object, not a list
        assert _update.check("0.6.6", installed="0.6.6") is None


def test_the_checker_keeps_its_last_answer(monkeypatch):
    info = _update.UpdateInfo("0.6.6", "0.7.0", "u")
    monkeypatch.setattr(_update, "check", lambda running: info)
    checker = _update.UpdateChecker("0.6.6", interval=3600)
    assert checker.info() is None
    checker.start()
    for _ in range(100):
        if checker.info():
            break
        checker._stop.wait(0.05)
    checker.stop()
    assert checker.info() == {"current": "0.6.6", "latest": "0.7.0", "url": "u"}


@pytest.mark.parametrize(
    "value,on", [(None, True), ("1", True), ("0", False), ("off", False)]
)
def test_the_env_switch(monkeypatch, value, on):
    monkeypatch.delenv(_update.ENV_UPDATE_CHECK, raising=False)
    if value is not None:
        monkeypatch.setenv(_update.ENV_UPDATE_CHECK, value)
    assert _update.enabled() is on


class TestStartInstall:
    """The hand-off: a detached installer, unattended, with the result recorded."""

    @pytest.fixture
    def spawned(self, tmp_path, monkeypatch):
        monkeypatch.setenv("BIOPB_STATE_HOME", str(tmp_path))
        (tmp_path / "biopb").mkdir()
        calls = []

        class _Popen:
            def __init__(self, argv, **kwargs):
                calls.append((argv, kwargs))

        monkeypatch.setattr(_update.subprocess, "Popen", _Popen)
        return calls

    def test_runs_the_pinned_installer_unattended(self, spawned, monkeypatch):
        _update.start_install(_update.UpdateInfo("0.6.6", "0.7.0", "u"))
        ((argv, kwargs),) = spawned
        url = (
            _update.INSTALL_PS1_URL
            if sys.platform == "win32"
            else _update.INSTALL_SH_URL
        )
        assert url in " ".join(argv)
        assert kwargs["env"]["BIOPB_NONINTERACTIVE"] == "1"
        assert "BIOPB_INSTALL_RC" not in kwargs["env"]
        assert kwargs["env"][_update.ENV_RESULT] == str(_update.result_file())
        # Detached, so it outlives this control.
        assert kwargs.get("start_new_session") or kwargs.get("creationflags")

    def test_a_candidate_install_asks_for_candidates(self, spawned):
        _update.start_install(_update.UpdateInfo("0.7.0rc1", "0.7.0rc2", "u"))
        assert spawned[0][1]["env"]["BIOPB_INSTALL_RC"] == "1"

    def test_the_exec_token_is_not_handed_down(self, spawned, monkeypatch):
        from biopb._security import credentials

        monkeypatch.setenv(credentials.ENV_CONTROL_EXEC_TOKEN, "s" * 20)
        _update.start_install(_update.UpdateInfo("0.6.6", "0.7.0", "u"))
        assert credentials.ENV_CONTROL_EXEC_TOKEN not in spawned[0][1]["env"]

    def test_records_running_before_it_spawns(self, spawned):
        _update.start_install(_update.UpdateInfo("0.6.6", "0.7.0", "u"))
        assert _update.last_result()["status"] == "running"

    def test_refuses_while_a_run_is_in_progress(self, spawned):
        info = _update.UpdateInfo("0.6.6", "0.7.0", "u")
        _update.start_install(info)
        with pytest.raises(_update.InstallRunning):
            _update.start_install(info)
        assert len(spawned) == 1

    def test_a_finished_or_stale_run_does_not_block(self, spawned):
        info = _update.UpdateInfo("0.6.6", "0.7.0", "u")
        _update.start_install(info)
        _update.result_file().write_text('{"status": "failed"}')
        _update.start_install(info)
        assert len(spawned) == 2
