"""bootstrap.sh: pick a release, fetch its installer, run it.

The network is a stub `curl` in front of PATH: the release listing comes from a
file, an installer download writes a script that reports how it was run, and every
requested URL is logged.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest
from conftest import requires_posix

# bootstrap.sh never runs on Windows; that platform gets bootstrap.ps1.
pytestmark = requires_posix

BOOTSTRAP = Path(__file__).resolve().parent.parent / "bootstrap.sh"

LISTING = """[
  {"tag_name": "release-v0.16.0rc1"},
  {"tag_name": "v0.11.1"},
  {"tag_name": "release-v0.15.2"},
  {"tag_name": "release-v0.15.1"}
]"""

FAKE_CURL = r"""#!/bin/sh
out=""
url=""
while [ $# -gt 0 ]; do
    case "$1" in
        -o) out="$2"; shift ;;
        -*|Accept*) ;;
        http*) url="$1" ;;
    esac
    shift
done
echo "$url" >> "$STUB_DIR/urls"
case "$url" in
    *api.github.com*) cat "$STUB_DIR/listing.json" ;;
    *releases/download/*/install.sh)
        [ -f "$STUB_DIR/installer.sh" ] || exit 22
        cp "$STUB_DIR/installer.sh" "$out" ;;
    *) exit 22 ;;
esac
"""

# Reports its arguments, whether it kept stdin, and one environment variable.
INSTALLER = """#!/bin/bash
echo "ran args=[$*] mark=[${BIOPB_MARK:-}] version=[${BIOPB_INSTALL_VERSION:-}]"
exit ${INSTALLER_STATUS:-0}
"""


@pytest.fixture
def run(tmp_path):
    stubs = tmp_path / "stubs"
    stubs.mkdir()
    curl = stubs / "curl"
    curl.write_text(FAKE_CURL)
    curl.chmod(0o755)
    (stubs / "listing.json").write_text(LISTING)
    (stubs / "installer.sh").write_text(INSTALLER)

    def go(*args, env=None, installer=INSTALLER, listing=LISTING):
        (stubs / "listing.json").write_text(listing)
        if installer is None:
            (stubs / "installer.sh").unlink(missing_ok=True)
        else:
            (stubs / "installer.sh").write_text(installer)
        (stubs / "urls").unlink(missing_ok=True)
        full = {
            "PATH": f"{stubs}:/usr/bin:/bin",
            "STUB_DIR": str(stubs),
            "HOME": str(tmp_path),
            **(env or {}),
        }
        proc = subprocess.run(
            ["bash", str(BOOTSTRAP), *args],
            capture_output=True,
            text=True,
            env=full,
        )
        urls = (stubs / "urls").read_text().split() if (stubs / "urls").exists() else []
        return proc, urls

    return go


def test_runs_the_latest_stable_installer(run):
    proc, urls = run()
    assert proc.returncode == 0, proc.stderr
    assert "ran args=[]" in proc.stdout
    # Not the rc, not the SDK's v* line: the newest clean release-v*.
    assert urls[-1].endswith("/releases/download/release-v0.15.2/install.sh")


def test_rc_channel_takes_the_candidate(run):
    proc, urls = run(env={"BIOPB_INSTALL_RC": "1"})
    assert proc.returncode == 0, proc.stderr
    assert urls[-1].endswith("/release-v0.16.0rc1/install.sh")


def test_rc_zero_is_the_stable_channel(run):
    _, urls = run(env={"BIOPB_INSTALL_RC": "0"})
    assert urls[-1].endswith("/release-v0.15.2/install.sh")


@pytest.mark.parametrize("given", ["0.15.1", "v0.15.1", "release-v0.15.1"])
def test_an_exact_version_skips_the_listing(run, given):
    proc, urls = run(env={"BIOPB_INSTALL_VERSION": given})
    assert proc.returncode == 0, proc.stderr
    assert len(urls) == 1
    assert urls[0].endswith("/release-v0.15.1/install.sh")


def test_arguments_and_environment_reach_the_installer(run):
    proc, _ = run("--uninstall", "--purge", env={"BIOPB_MARK": "x"})
    assert "ran args=[--uninstall --purge] mark=[x]" in proc.stdout


@pytest.mark.parametrize(
    "env, picked",
    [
        ({}, "release-v0.15.2"),
        ({"BIOPB_INSTALL_RC": "1"}, "release-v0.16.0rc1"),
        ({"BIOPB_INSTALL_VERSION": "0.15.1"}, "release-v0.15.1"),
    ],
)
def test_the_installer_is_told_which_release_was_picked(run, env, picked):
    proc, _ = run(env=env)
    assert f"version=[{picked}]" in proc.stdout


def test_the_installers_exit_status_is_ours(run):
    proc, _ = run(env={"INSTALLER_STATUS": "7"})
    assert proc.returncode == 7


def test_stdin_is_not_shared_with_the_installer(run):
    probe = "#!/bin/bash\nread -r line && echo got=[$line] || echo eof\n"
    proc, _ = run(installer=probe)
    assert "eof" in proc.stdout


def test_a_release_without_an_installer_asset_fails_clearly(run):
    proc, _ = run(installer=None)
    assert proc.returncode != 0
    assert "Could not download" in proc.stderr
    assert "ran args" not in proc.stdout


def test_a_truncated_installer_is_not_run(run):
    proc, _ = run(installer="#!/bin/bash\nif true; then\n")
    assert proc.returncode != 0
    assert "not a valid script" in proc.stderr


def test_no_release_found_names_the_override(run):
    proc, _ = run(listing='[{"tag_name": "v0.11.1"}]')
    assert proc.returncode != 0
    assert "BIOPB_INSTALL_VERSION" in proc.stderr


def test_a_tag_that_is_not_a_plain_name_is_refused(run):
    proc, urls = run(env={"BIOPB_INSTALL_VERSION": "0.1/../x?y"})
    assert proc.returncode != 0
    assert urls == []


def test_missing_curl_is_reported(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    for tool in ("bash", "uname"):
        (empty / tool).symlink_to(
            next(p for p in (f"/usr/bin/{tool}", f"/bin/{tool}") if os.path.exists(p))
        )
    proc = subprocess.run(
        [str(empty / "bash"), str(BOOTSTRAP)],
        capture_output=True,
        text=True,
        env={"PATH": str(empty), "HOME": str(tmp_path)},
    )
    assert proc.returncode != 0
    assert "curl is required" in proc.stderr


def test_the_guard_suppresses_main():
    proc = subprocess.run(
        ["bash", "-c", f'BIOPB_INSTALL_LIB=1; . "{BOOTSTRAP}"; type main | head -1'],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "main is a function" in proc.stdout


def test_main_is_still_the_last_line():
    last = BOOTSTRAP.read_text().rstrip().splitlines()[-1]
    assert last == '[ -n "${BIOPB_INSTALL_LIB:-}" ] || main "$@"'
