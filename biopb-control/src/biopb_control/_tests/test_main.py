"""Tests for the ``python -m biopb_control`` argparse entry (`__main__.main`).

The security-relevant behavior: the access token is read from the
``BIOPB_TENSOR_TOKEN`` env var, not required on the argv, so `biopb control
start` never puts the secret on a world-readable command line (biopb/biopb#414).
The explicit ``--token`` flag is still honored for a direct invocation.
"""

from unittest.mock import patch

import biopb_control.__main__ as m


def _capture_spec(argv, env):
    """Run ``main(argv)`` with a stubbed ``run_control`` and return the spec it
    was handed, isolating the process env to ``env``."""
    captured = {}

    def _fake_run_control(spec, **_kwargs):
        captured["spec"] = spec
        return 0

    with (
        patch.object(m, "run_control", _fake_run_control),
        patch.dict("os.environ", env, clear=True),
    ):
        rc = m.main(argv)
    assert rc == 0
    return captured["spec"]


_BASE_ARGV = ["run", "--config", "/tmp/biopb.json"]


def test_token_read_from_env_when_flag_absent():
    spec = _capture_spec(_BASE_ARGV, {"BIOPB_TENSOR_TOKEN": "s3cret"})
    assert spec.token == "s3cret"


def test_flag_wins_over_env():
    spec = _capture_spec(
        _BASE_ARGV + ["--token", "from-flag"],
        {"BIOPB_TENSOR_TOKEN": "from-env"},
    )
    assert spec.token == "from-flag"


def test_no_token_anywhere_is_none():
    spec = _capture_spec(_BASE_ARGV, {})
    assert spec.token is None


def test_local_bypass_carries_through():
    spec = _capture_spec(_BASE_ARGV + ["--local-bypass"], {})
    assert spec.local_bypass is True
    assert spec.token is None
