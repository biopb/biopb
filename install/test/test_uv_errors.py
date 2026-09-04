"""What the user is told when uv itself is the thing that failed.

Every uv call in the engine runs through Invoke-Uv, which captures uv's streams to
a log and replays them. Assert-UvExit is the other half: it turns a nonzero uv
exit into a throw that QUOTES uv's own reason, because the thrown message is what
both front-ends actually render -- the console prints it, and the Inno wizard has
no console and nothing else to show.

Before this, step 2's `uv python install` was the one uv call made bare, and it is
the call most likely to fail on a lab network: it downloads a ~30MB interpreter
build from python-build-standalone, so a proxy, TLS interception or a blocked
route stops it. What the user got was "Python install failed (exit code 2)", with
uv's actual `error:` line going nowhere a GUI user could reach.

Only the PowerShell installer has this; install.sh reports uv's failure through
the shell's own stderr, which is already in front of the user.
"""

from __future__ import annotations

import pytest
from conftest import ps_literal, pwsh, requires_pwsh


def assert_uv_exit(exit_code, log_lines, tmp_path, *, write_log=True) -> str:
    """Run Assert-UvExit against a fake uv error log; return "OK" or the message.

    LastUvExit is set directly rather than by running uv: the contract under test
    is "given uv failed, what reaches the user", and staging that with a real
    network failure would test the network.
    """
    log = tmp_path / "uv.err.log"
    if write_log:
        log.write_text("".join(f"{line}\n" for line in log_lines), encoding="utf-8")
    out = pwsh(
        f"$script:LastUvExit = {exit_code}\n"
        f"try {{ Assert-UvExit 'Python install' {ps_literal(log)}; 'OK' }}\n"
        "catch { $_.Exception.Message }\n"
    )
    return out.stdout.strip()


@requires_pwsh
def test_success_does_not_throw(tmp_path):
    """Exit 0 is not an error however much uv wrote to stderr on the way."""
    assert assert_uv_exit(0, ["Downloading cpython-3.12..."], tmp_path) == "OK"


@requires_pwsh
def test_uvs_own_error_line_reaches_the_user(tmp_path):
    """The whole point: uv's `error:` line ends up in the thrown message.

    Not merely present in a log the GUI never opens -- in the message the wizard
    puts on screen.
    """
    message = assert_uv_exit(
        2,
        [
            "Downloading cpython-3.12.13-windows-x86_64-none",
            "error: Failed to fetch: `https://github.com/astral-sh/python-build-standalone`",
            "  Caused by: operation timed out",
        ],
        tmp_path,
    )
    assert "exit code 2" in message
    assert "Failed to fetch" in message


@requires_pwsh
def test_first_error_line_wins(tmp_path):
    """uv's first diagnosis, not a later cascade line that is a symptom of it."""
    message = assert_uv_exit(
        2, ["error: the real cause", "error: a downstream complaint"], tmp_path
    )
    assert "the real cause" in message
    assert "a downstream complaint" not in message


@requires_pwsh
def test_falls_back_to_the_last_thing_uv_said(tmp_path):
    """No `error:` line is not a reason to say nothing.

    uv does not prefix every failure, and a bare exit code is what this exists to
    stop the user getting.
    """
    message = assert_uv_exit(
        2, ["warning: something odd", "could not resolve host", ""], tmp_path
    )
    assert "could not resolve host" in message


@requires_pwsh
@pytest.mark.parametrize(
    "log_lines,write_log",
    [
        pytest.param([], True, id="empty-log"),
        pytest.param([], False, id="no-log-at-all"),
    ],
)
def test_still_reports_the_exit_code_with_nothing_to_quote(
    tmp_path, log_lines, write_log
):
    """Degrades to the old message rather than throwing inside the error path.

    A failure to read the log must not replace uv's failure with a different one.
    """
    message = assert_uv_exit(2, log_lines, tmp_path, write_log=write_log)
    assert message == "Python install failed (exit code 2)"
