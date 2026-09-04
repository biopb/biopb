"""Unit tests for install.sh's pure helpers -- the ones that parse or format.

install.sh is ~1900 lines that, until biopb/biopb#653, nothing executed: it was
parsed, linted, and shipped. The functions covered here are the subset with no
side effects on the machine, and they are the ones worth covering first for a
specific reason -- a bug in a parser does not fail, it returns a plausible wrong
answer. #648's extras bug installed the wrong package under the right name and
reported success. shellcheck is green on all of it.

The extras parser has its own module (test_extras_contract.py) because it is
implemented twice and the cases are shared with PowerShell. Everything else lives
here.
"""

from __future__ import annotations

import json
import subprocess

import pytest
from conftest import INSTALL_SH, bash, requires_posix, sh

# install.sh never runs on Windows -- that platform gets install.ps1 and the
# engine, which test_python_probe.py and test_extras_contract.py cover there.
pytestmark = requires_posix

# --- the source guard itself -------------------------------------------------
#
# Everything below depends on it, and it is a change to the file that a truncated
# `curl | bash` reaches, so it gets tested before anything that uses it.


def test_help_runs_without_the_guard():
    """No BIOPB_INSTALL_LIB: the script runs main, exactly as a user's shell does."""
    out = subprocess.run(
        ["bash", str(INSTALL_SH), "--help"], capture_output=True, text=True, timeout=60
    )
    assert out.returncode == 0
    assert "biopb stack installer" in out.stderr


def test_the_guard_suppresses_main():
    """With it, sourcing defines functions and does nothing else.

    --help is the probe because it is main's only side-effect-free path: if main
    ran, its usage text would be on stderr. Silence means the file was read as a
    library, which is the whole contract the tests rest on.
    """
    out = subprocess.run(
        ["bash", str(INSTALL_SH), "--help"],
        capture_output=True,
        text=True,
        timeout=60,
        env={"BIOPB_INSTALL_LIB": "1", "PATH": "/usr/bin:/bin"},
    )
    assert out.returncode == 0
    assert out.stdout == "" and out.stderr == ""


def test_main_is_still_the_last_line():
    """The truncation guard is positional: `main "$@"` last, or it stops working.

    A `curl | bash` cut off mid-transfer must define some functions and then do
    nothing. If a later edit appends anything executable after this call -- or
    moves it up -- a partial download starts running instead, with an arbitrary
    slice of the installer defined. That failure is invisible in review and
    impossible to reproduce locally, so it is asserted here instead.
    """
    lines = [
        ln for ln in INSTALL_SH.read_text(encoding="utf-8").splitlines() if ln.strip()
    ]
    assert lines[-1] == '[ -n "${BIOPB_INSTALL_LIB:-}" ] || main "$@"'


# --- _urldecode --------------------------------------------------------------


@pytest.mark.parametrize(
    ("encoded", "decoded"),
    [
        # The case it exists for: GitHub percent-encodes the '+' of a local version
        # segment in the asset URL, but the file on disk must carry the literal '+'
        # or uv rejects the wheel -- its filename version no longer matches the
        # metadata.
        ("biopb-0.11.0%2Bcuda-py3-none-any.whl", "biopb-0.11.0+cuda-py3-none-any.whl"),
        ("a%20b.whl", "a b.whl"),
        ("%7Etilde", "~tilde"),
        # Nothing to decode passes through untouched -- the common case, since
        # most wheel names have no encoded character at all.
        ("biopb_mcp-0.11.0-py3-none-any.whl", "biopb_mcp-0.11.0-py3-none-any.whl"),
        ("SHA256SUMS", "SHA256SUMS"),
        ("", ""),
    ],
)
def test_urldecode(encoded, decoded):
    out = bash(f"_urldecode {sh(encoded)}")
    assert out.stdout == decoded


def test_urldecode_leaves_percent_encoded_percent_alone():
    """%25 is a literal '%', and decoding it must not start a second round."""
    assert bash("_urldecode '100%25done.whl'").stdout == "100%done.whl"


# --- _release_asset_url ------------------------------------------------------

RELEASE_JSON = json.dumps(
    {
        "tag_name": "release-v0.11.0",
        "assets": [
            {"browser_download_url": "https://example.test/d/SHA256SUMS"},
            {
                "browser_download_url": "https://example.test/d/biopb-0.11.0-py3-none-any.whl"
            },
            {
                "browser_download_url": "https://example.test/d/biopb_mcp-0.11.0-py3-none-any.whl"
            },
            {"browser_download_url": "https://example.test/d/biopb-samples.tar.gz"},
        ],
    }
)


def _asset(pattern, release_json=RELEASE_JSON):
    return bash(
        f"_release_asset_url {sh(pattern)}", env={"RELEASE_JSON": release_json}
    ).stdout.strip()


def test_release_asset_url_finds_an_exact_name():
    assert _asset("SHA256SUMS") == "https://example.test/d/SHA256SUMS"


def test_release_asset_url_accepts_a_regex():
    assert (
        _asset(r"biopb_mcp-.*\.whl")
        == "https://example.test/d/biopb_mcp-0.11.0-py3-none-any.whl"
    )


def test_release_asset_url_anchors_on_a_whole_filename():
    """The match is `/<pattern>$` -- a filename, not a suffix of one.

    Without the leading slash, `samples.tar.gz` would match the URL ending
    `biopb-samples.tar.gz`, and any short pattern would start matching assets it
    was not asked for. The release carries the longer name, so this returns
    nothing; the caller's fallback is what should run, not a wrong download.
    """
    assert _asset(r"samples\.tar\.gz") == ""
    assert (
        _asset(r"biopb-samples\.tar\.gz")
        == "https://example.test/d/biopb-samples.tar.gz"
    )


def test_release_asset_url_is_empty_and_succeeds_when_nothing_matches():
    """No match is a supported outcome, not an error.

    Callers rely on an empty string to fall back or print a friendly message. The
    function runs inside a command substitution under `set -euo pipefail`, where
    grep's exit 1 would otherwise abort the entire installer -- hence its
    `|| true`. This asserts the exit status, which is the part that bites.
    """
    out = bash(
        "_release_asset_url 'no-such-asset'; printf 'rc=%s\\n' \"$?\"",
        env={"RELEASE_JSON": RELEASE_JSON},
    )
    assert out.stdout == "rc=0\n"


def test_release_asset_url_survives_an_unset_release_json():
    """`${RELEASE_JSON:-}` under `set -u` -- an unfetched release must not crash."""
    out = bash("_release_asset_url 'SHA256SUMS'; printf 'rc=%s\\n' \"$?\"")
    assert out.stdout == "rc=0\n"


# --- _agent_launch_cmd / _detect_agents --------------------------------------
#
# Both probe with `command -v` and `[ -d ]`, which are builtins, so these run with
# the stub dir as the ENTIRE PATH. A real `claude` on the developer's machine then
# cannot reach in and change the answer.


def test_agent_launch_cmd_prefers_claude(stub_bin, tmp_path):
    make, path = stub_bin
    make("claude")
    make("opencode")
    assert (
        bash("_agent_launch_cmd", path=path, env={"HOME": str(tmp_path)}).stdout
        == "claude\n"
    )


def test_agent_launch_cmd_falls_back_to_opencode(stub_bin, tmp_path):
    make, path = stub_bin
    make("opencode")
    assert (
        bash("_agent_launch_cmd", path=path, env={"HOME": str(tmp_path)}).stdout
        == "opencode\n"
    )


def test_agent_launch_cmd_falls_back_to_codex_before_cursor(stub_bin, tmp_path):
    """`codex` is a terminal agent; `cursor` is the GUI editor, so codex ranks first."""
    make, path = stub_bin
    make("codex")
    make("cursor")
    assert (
        bash("_agent_launch_cmd", path=path, env={"HOME": str(tmp_path)}).stdout
        == "codex\n"
    )


def test_agent_launch_cmd_accepts_a_config_dir_without_a_binary(stub_bin, tmp_path):
    """opencode installed but not on this shell's PATH still counts."""
    _, path = stub_bin
    (tmp_path / ".config" / "opencode").mkdir(parents=True)
    assert (
        bash("_agent_launch_cmd", path=path, env={"HOME": str(tmp_path)}).stdout
        == "opencode\n"
    )


def test_agent_launch_cmd_prints_nothing_when_no_agent_exists(stub_bin, tmp_path):
    """The "next steps" message drops the line rather than recommending a missing tool."""
    _, path = stub_bin
    out = bash(
        '_agent_launch_cmd; printf "rc=%s\\n" "$?"',
        path=path,
        env={"HOME": str(tmp_path)},
    )
    assert out.stdout == "rc=0\n"


def _detect(path, home, platform="Linux"):
    out = bash(
        "_detect_agents\n"
        'printf "rc=%s\\n" "$?"\n'
        'if [ "${#DETECTED_AGENTS[@]}" -gt 0 ]; then printf "%s\\n" "${DETECTED_AGENTS[@]}"; fi\n',
        path=path,
        env={"HOME": str(home), "PLATFORM": platform},
    )
    lines = out.stdout.splitlines()
    assert lines[0] == "rc=0"
    return lines[1:]


def test_detect_agents_finds_nothing_on_a_bare_machine(stub_bin, tmp_path):
    """An empty result must still return 0.

    This runs under `set -e` inside install_biopb, and "no agents found" is the
    case that triggers the offer to install one -- so returning the last
    command's status here would abort the install on exactly the machines that
    need the offer. Hence the explicit `return 0` the assertion pins.
    """
    _, path = stub_bin
    assert _detect(path, tmp_path) == []


def test_detect_agents_finds_each_kind(stub_bin, tmp_path):
    make, path = stub_bin
    make("claude")
    (tmp_path / ".config" / "Claude").mkdir(parents=True)
    make("codex")
    (tmp_path / ".cursor").mkdir()
    (tmp_path / ".config" / "opencode").mkdir(parents=True)
    assert _detect(path, tmp_path) == [
        "Claude Code",
        "Claude Desktop",
        "Codex CLI",
        "Cursor",
        "opencode",
    ]


def test_detect_agents_ignores_a_leftover_codex_home(stub_bin, tmp_path):
    """~/.codex outlives an uninstall, so it must not count as an agent.

    Counting it would print "AI agent detected: Codex CLI" and skip the offer to
    install one, leaving a machine with no working agent and no prompt.
    """
    _, path = stub_bin
    (tmp_path / ".codex").mkdir()
    assert _detect(path, tmp_path) == []


def test_detect_agents_looks_in_the_platform_specific_place(stub_bin, tmp_path):
    """Claude Desktop's config dir differs by OS; the Linux path must not count on macOS."""
    _, path = stub_bin
    (tmp_path / ".config" / "Claude").mkdir(parents=True)
    assert _detect(path, tmp_path, platform="macOS") == []

    mac = tmp_path / "mac"
    (mac / "Library" / "Application Support" / "Claude").mkdir(parents=True)
    assert _detect(path, mac, platform="macOS") == ["Claude Desktop"]


# --- _pid_is_biopb -----------------------------------------------------------


def _is_biopb(pid):
    out = bash(f"if _pid_is_biopb {pid}; then echo yes; else echo no; fi")
    return out.stdout.strip() == "yes"


def test_pid_is_biopb_matches_a_live_biopb_process():
    # exec -a rewrites argv[0], which is what both branches of the function read
    # (/proc/PID/cmdline on Linux, `ps -o command=` elsewhere).
    proc = subprocess.Popen(["bash", "-c", "exec -a biopb-tensor-server sleep 30"])
    try:
        assert _is_biopb(proc.pid)
    finally:
        proc.kill()
        proc.wait()


def test_pid_is_biopb_rejects_an_unrelated_live_process():
    """The whole point: a recycled PID from a stale pidfile is not force-killed.

    Without the command-line check, uninstall would send SIGKILL to whatever now
    owns a PID the last biopb run recorded.
    """
    proc = subprocess.Popen(["sleep", "30"])
    try:
        assert not _is_biopb(proc.pid)
    finally:
        proc.kill()
        proc.wait()


def test_pid_is_biopb_rejects_a_dead_pid():
    proc = subprocess.Popen(["sleep", "0.01"])
    proc.wait()
    assert not _is_biopb(proc.pid)


# --- _write_server_config ----------------------------------------------------


def _write_config(out_path, data_dir, monitor="true", prior="", alias=""):
    bash(
        f"_write_server_config {sh(out_path)} {sh(data_dir)} "
        f"{sh(monitor)} {sh(prior)} {sh(alias)}"
    )
    return json.loads(out_path.read_text(encoding="utf-8"))


def test_write_server_config_writes_installer_defaults(tmp_path):
    cfg = _write_config(tmp_path / "biopb.json", "/data")
    assert cfg["sources"] == [{"url": "/data", "monitor": True}]
    assert cfg["server"] == {"aggressive_dir_pruning": True}
    assert cfg["cache"]["backend"] == "file"


def test_write_server_config_monitor_is_a_string_comparison(tmp_path):
    """`monitor` arrives as the text "true"/"false"; anything else is false.

    The sample bundle is static and passes "false"; a user data dir passes
    "true". A truthiness bug here silently starts (or stops) a filesystem watcher.
    """
    assert (
        _write_config(tmp_path / "a.json", "/d", monitor="false")["sources"][0][
            "monitor"
        ]
        is False
    )
    assert (
        _write_config(tmp_path / "b.json", "/d", monitor="true")["sources"][0][
            "monitor"
        ]
        is True
    )


def test_write_server_config_sets_an_alias_only_when_given(tmp_path):
    plain = _write_config(tmp_path / "a.json", "/d")
    assert "alias" not in plain["sources"][0]
    aliased = _write_config(tmp_path / "b.json", "/d", alias="samples")
    assert aliased["sources"][0]["alias"] == "samples"


def test_write_server_config_keeps_the_users_tuning(tmp_path):
    """Only `sources` is replaced -- re-running with a new folder is not a reset."""
    prior = tmp_path / "prior.json"
    prior.write_text(
        json.dumps(
            {
                "server": {"aggressive_dir_pruning": False},
                "cache": {"backend": "memory", "file_max_total_gb": 4},
                "something_custom": {"kept": True},
                "sources": [{"url": "/old", "monitor": True}],
            }
        )
    )
    cfg = _write_config(tmp_path / "new.json", "/new", prior=str(prior))
    assert cfg["sources"] == [{"url": "/new", "monitor": True}]
    assert cfg["server"] == {"aggressive_dir_pruning": False}
    assert cfg["cache"] == {"backend": "memory", "file_max_total_gb": 4}
    assert cfg["something_custom"] == {"kept": True}


def test_write_server_config_strips_the_redundant_metadata_db_flag(tmp_path):
    """`enabled = true` is the default and earns a deprecation warning every startup."""
    prior = tmp_path / "prior.json"
    prior.write_text(json.dumps({"metadata_db": {"enabled": True, "path": "/db"}}))
    cfg = _write_config(tmp_path / "new.json", "/d", prior=str(prior))
    assert cfg["metadata_db"] == {"path": "/db"}


def test_write_server_config_preserves_a_deliberate_metadata_db_off(tmp_path):
    """`enabled = false` is a user's choice (read-only mount, disk limits) -- kept."""
    prior = tmp_path / "prior.json"
    prior.write_text(json.dumps({"metadata_db": {"enabled": False}}))
    cfg = _write_config(tmp_path / "new.json", "/d", prior=str(prior))
    assert cfg["metadata_db"] == {"enabled": False}


def test_write_server_config_drops_a_metadata_db_left_empty(tmp_path):
    """Stripping the only key must not leave `metadata_db = {}` behind."""
    prior = tmp_path / "prior.json"
    prior.write_text(json.dumps({"metadata_db": {"enabled": True}}))
    assert "metadata_db" not in _write_config(
        tmp_path / "new.json", "/d", prior=str(prior)
    )


def test_write_server_config_starts_clean_on_an_unreadable_prior(tmp_path):
    """A corrupt config must not abort the install; the new data dir is written anyway."""
    prior = tmp_path / "prior.json"
    prior.write_text("{ this is not json")
    cfg = _write_config(tmp_path / "new.json", "/d", prior=str(prior))
    assert cfg["sources"] == [{"url": "/d", "monitor": True}]
    assert cfg["server"] == {"aggressive_dir_pruning": True}


def test_write_server_config_leaves_no_temp_file(tmp_path):
    """It writes through a temp + os.replace; a leftover means the swap did not happen."""
    out = tmp_path / "biopb.json"
    _write_config(out, "/d")
    assert [p.name for p in tmp_path.iterdir()] == ["biopb.json"]


# --- _mcp_unmerge ------------------------------------------------------------


def _unmerge(path, parent="mcpServers"):
    return bash(f"_mcp_unmerge {sh(path)} {sh(parent)}").stdout.strip()


def test_mcp_unmerge_removes_only_the_biopb_entry(tmp_path):
    cfg = tmp_path / "mcp.json"
    cfg.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "biopb": {"command": "biopb-mcp"},
                    "other": {"command": "other-mcp"},
                },
                "theme": "dark",
            }
        )
    )
    assert _unmerge(cfg) == "removed"
    data = json.loads(cfg.read_text())
    assert data["mcpServers"] == {"other": {"command": "other-mcp"}}
    assert data["theme"] == "dark"


def test_mcp_unmerge_is_silent_when_there_is_nothing_to_remove(tmp_path):
    """Callers report per client off this output, so silence has to mean untouched."""
    cfg = tmp_path / "mcp.json"
    original = json.dumps({"mcpServers": {"other": {}}})
    cfg.write_text(original)
    assert _unmerge(cfg) == ""
    assert cfg.read_text() == original, "a file with no biopb entry is not rewritten"


def test_mcp_unmerge_leaves_a_foreign_file_byte_for_byte(tmp_path):
    """No parent section, a JSON list, or invalid JSON: all no-ops, none fatal.

    This runs during uninstall against config files the installer does not own.
    Rewriting one -- even to identical content -- would reformat a user's file;
    throwing would abort the rest of the teardown.
    """
    for name, content in [
        ("no-parent.json", '{\n  "other": 1\n}\n'),
        ("a-list.json", "[1, 2, 3]"),
        ("broken.json", "{ not json at all"),
    ]:
        cfg = tmp_path / name
        cfg.write_text(content)
        assert _unmerge(cfg) == ""
        assert cfg.read_text() == content


def test_mcp_unmerge_ignores_a_missing_file(tmp_path):
    """Every uninstall hits this: most users have only one of the four clients."""
    missing = tmp_path / "not-there.json"
    assert _unmerge(missing) == ""
    assert not missing.exists()
