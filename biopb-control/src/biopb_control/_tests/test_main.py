"""Tests for the ``python -m biopb_control`` argparse entry (`__main__.main`).

The security-relevant behavior:

- the access token is read from the ``BIOPB_TENSOR_TOKEN`` env var, not required
  on the argv, so `biopb control start` never puts the secret on a world-readable
  command line (biopb/biopb#414). The explicit ``--token`` flag is still honored;
- a control listener that is reachable off-box is fail-closed on the *resolved
  bind*: it refuses to run token-less whether it went public via ``--remote``, an
  explicit ``--control-host <public>``, or ``BIOPB_CONTROL_HOST`` — so a public
  control API can never come up unauthenticated. Local mode (loopback) binds
  token-less.
"""

from unittest.mock import patch

import biopb_control.__main__ as m


def _capture(argv, env):
    """Run ``main(argv)`` with a stubbed ``run_control`` and return
    ``(rc, spec, run_kwargs)``, isolating the process env to ``env``."""
    captured = {}

    def _fake_run_control(spec, **kwargs):
        captured["spec"] = spec
        captured["kwargs"] = kwargs
        return 0

    with (
        patch.object(m, "run_control", _fake_run_control),
        patch.dict("os.environ", env, clear=True),
    ):
        rc = m.main(argv)
    return rc, captured.get("spec"), captured.get("kwargs")


_BASE_ARGV = ["run", "--config", "/tmp/biopb.json"]


def test_token_read_from_env_when_flag_absent():
    rc, spec, _ = _capture(_BASE_ARGV, {"BIOPB_TENSOR_TOKEN": "s3cret"})
    assert rc == 0
    assert spec.token == "s3cret"


def test_flag_wins_over_env():
    rc, spec, _ = _capture(
        _BASE_ARGV + ["--token", "from-flag"],
        {"BIOPB_TENSOR_TOKEN": "from-env"},
    )
    assert rc == 0
    assert spec.token == "from-flag"


def test_no_token_anywhere_is_none():
    rc, spec, _ = _capture(_BASE_ARGV, {})
    assert rc == 0
    assert spec.token is None


def test_token_surrounding_whitespace_is_stripped():
    # A token sourced with a trailing newline (BIOPB_TENSOR_TOKEN=$(cat file)) must
    # be normalized at this single resolution point, so the enforced spec.token,
    # the tensor-server env, and the credential file (read back .strip()ed) all
    # carry the same bytes — otherwise a local client's credential-derived token
    # would 401 against the control that wrote it (biopb/biopb#470).
    rc, spec, _ = _capture(_BASE_ARGV, {"BIOPB_TENSOR_TOKEN": "s3cret\n"})
    assert rc == 0
    assert spec.token == "s3cret"


def test_whitespace_only_token_collapses_to_none():
    # A blank/whitespace-only value is not a real credential: it collapses to None
    # (tokenless) rather than becoming a truthy spec.token that would gate on — and
    # write a bogus empty credential for — the empty string.
    rc, spec, _ = _capture(_BASE_ARGV, {"BIOPB_TENSOR_TOKEN": "   \n"})
    assert rc == 0
    assert spec.token is None


def test_local_mode_binds_control_loopback():
    """Default (no --remote): the control listener is not bound publicly."""
    rc, _, kwargs = _capture(_BASE_ARGV, {})
    assert rc == 0
    assert kwargs["control_host"] != "0.0.0.0"


def test_remote_binds_control_public():
    rc, _, kwargs = _capture(
        _BASE_ARGV + ["--remote"], {"BIOPB_TENSOR_TOKEN": "s3cret"}
    )
    assert rc == 0
    assert kwargs["control_host"] == "0.0.0.0"


def test_remote_without_token_is_fail_closed():
    """--remote with no token anywhere must refuse (never serve public + open)."""
    rc, spec, _ = _capture(_BASE_ARGV + ["--remote"], {})
    assert rc == 2
    assert spec is None  # run_control never reached


def test_public_control_host_without_token_is_fail_closed():
    """An explicit public --control-host (no --remote, no token) must also refuse:
    the guard keys on the resolved bind, not on --remote, so a token-less public
    control API -- whose /api/* gate degrades to a spoofable Host check -- can't
    come up."""
    rc, spec, _ = _capture(_BASE_ARGV + ["--control-host", "0.0.0.0"], {})
    assert rc == 2
    assert spec is None


def test_public_control_host_via_env_without_token_is_fail_closed():
    """Same guard catches a public bind smuggled in through BIOPB_CONTROL_HOST."""
    rc, spec, _ = _capture(_BASE_ARGV, {"BIOPB_CONTROL_HOST": "0.0.0.0"})
    assert rc == 2
    assert spec is None


def test_public_control_host_with_token_starts():
    """A public control bind is fine once a token is present."""
    rc, _, kwargs = _capture(
        _BASE_ARGV + ["--control-host", "0.0.0.0"],
        {"BIOPB_TENSOR_TOKEN": "s3cret"},
    )
    assert rc == 0
    assert kwargs["control_host"] == "0.0.0.0"
