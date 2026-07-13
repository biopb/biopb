"""Tests for the ``python -m biopb_control`` argparse entry (`__main__.main`).

The security-relevant behavior:

- the access token is read from the ``BIOPB_TENSOR_TOKEN`` env var, not required
  on the argv, so `biopb control start` never puts the secret on a world-readable
  command line (biopb/biopb#414). The explicit ``--token`` flag is still honored;
- ``--remote`` binds the control's own listener publicly (0.0.0.0) and is
  fail-closed: it refuses to run without a token, so a public control API can
  never come up unauthenticated. Local mode (the default) binds loopback.
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
