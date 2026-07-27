"""The container entrypoint's command shape (biopb/biopb#604 item 3).

The image is Flight-only by default -- `serve`, no HTTP surface -- with the
FastAPI sidecar behind an opt-in env var. That decision lives in a shell script,
so these tests run the real `entrypoint.sh` with a stub `biopb-tensor-server` on
PATH and assert the argv it would have exec'd.
"""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

ENTRYPOINT = Path(__file__).resolve().parents[1] / "entrypoint.sh"

pytestmark = pytest.mark.skipif(
    os.name != "posix" or shutil.which("bash") is None,
    reason="entrypoint.sh is a bash script (POSIX images only)",
)


def _run(tmp_path, env=None, expect_ok=True):
    """Run entrypoint.sh with a stub CLI; return (argv it exec'd, CompletedProcess).

    The stub records ``"$@"`` one arg per line and exits 0, so the entrypoint's
    final `exec` is observable without starting a server.
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    args_file = tmp_path / "argv.txt"
    stub = bin_dir / "biopb-tensor-server"
    stub.write_text('#!/bin/bash\nprintf "%s\\n" "$@" > "$BIOPB_TEST_ARGV"\nexit 0\n')
    stub.chmod(0o755)

    full_env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "BIOPB_TEST_ARGV": str(args_file),
        "BIOPB_TMP": str(tmp_path / "tmp"),
        "DATA_DIR": str(tmp_path),
    }
    # Don't inherit the developer's own settings for anything the script reads.
    for key in (
        "BIOPB_ENABLE_HTTP_SIDECAR",
        "BIOPB_TENSOR_TLS",
        "BIOPB_TLS_CERT",
        "BIOPB_TLS_KEY",
        "BIOPB_TENSOR_TOKEN",
        "BIOPB_TENSOR_ALLOW_NO_TOKEN",
        "BIOPB_CORS_ORIGINS",
        "CONFIG_FILE",
    ):
        full_env.pop(key, None)
    full_env.update(env or {})

    proc = subprocess.run(
        ["bash", str(ENTRYPOINT)],
        env=full_env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if expect_ok:
        assert proc.returncode == 0, proc.stderr
        argv = args_file.read_text().splitlines()
    else:
        argv = args_file.read_text().splitlines() if args_file.exists() else []
    return argv, proc


def test_default_is_flight_only(tmp_path):
    """No env vars -> `serve`, no sidecar flags anywhere."""
    argv, proc = _run(tmp_path)
    assert argv[0] == "serve"
    assert "launch" not in argv
    assert "--web-port" not in argv and "--web-host" not in argv
    assert "8814" not in proc.stdout  # the sidecar port is never announced


def test_default_generates_token_and_config(tmp_path):
    argv, _ = _run(tmp_path)
    assert "--token" in argv
    cfg = Path(argv[argv.index("--config") + 1])
    server = json.loads(cfg.read_text())["server"]
    assert server["host"] == "0.0.0.0"
    assert server["port"] == 8815


def test_sidecar_opt_in_runs_launch(tmp_path):
    argv, _ = _run(tmp_path, {"BIOPB_ENABLE_HTTP_SIDECAR": "1"})
    assert argv[0] == "launch"
    assert argv[argv.index("--web-port") + 1] == "8814"
    assert argv[argv.index("--web-host") + 1] == "0.0.0.0"


def test_sidecar_opt_in_accepts_word_truthies(tmp_path):
    """The shell predicate matches Python's _allow_no_token_from_env spelling."""
    for value in ("true", " YES ", "On"):
        argv, _ = _run(tmp_path, {"BIOPB_ENABLE_HTTP_SIDECAR": value})
        assert argv[0] == "launch", value
    argv, _ = _run(tmp_path, {"BIOPB_ENABLE_HTTP_SIDECAR": "0"})
    assert argv[0] == "serve"


def test_cors_origins_only_apply_to_the_sidecar(tmp_path):
    env = {"BIOPB_ENABLE_HTTP_SIDECAR": "1", "BIOPB_CORS_ORIGINS": "http://a http://b"}
    argv, _ = _run(tmp_path, env)
    assert argv.count("--cors") == 2
    assert "http://a" in argv and "http://b" in argv


def test_tls_opt_in_adds_flag(tmp_path):
    argv, proc = _run(tmp_path, {"BIOPB_TENSOR_TLS": "1"})
    assert argv[0] == "serve"
    assert "--tls" in argv
    assert "grpcs://" in proc.stdout  # the scheme clients must dial


def test_byo_cert_passes_paths(tmp_path):
    cert, key = tmp_path / "c.pem", tmp_path / "k.pem"
    cert.write_text("cert")
    key.write_text("key")
    argv, _ = _run(tmp_path, {"BIOPB_TLS_CERT": str(cert), "BIOPB_TLS_KEY": str(key)})
    assert argv[argv.index("--tls-cert") + 1] == str(cert)
    assert argv[argv.index("--tls-key") + 1] == str(key)
    assert "--tls" not in argv  # BYO wins over the self-signed path


def test_half_a_byo_cert_is_refused(tmp_path):
    _, proc = _run(tmp_path, {"BIOPB_TLS_CERT": "/x.pem"}, expect_ok=False)
    assert proc.returncode == 2
    assert "BIOPB_TLS_KEY" in proc.stderr


def test_tls_with_sidecar_is_refused(tmp_path):
    """The sidecar's internal client can't reach a TLS Flight server yet."""
    env = {"BIOPB_TENSOR_TLS": "1", "BIOPB_ENABLE_HTTP_SIDECAR": "1"}
    _, proc = _run(tmp_path, env, expect_ok=False)
    assert proc.returncode == 2
    assert "TLS" in proc.stderr


def test_allow_no_token_passes_no_token(tmp_path):
    argv, _ = _run(tmp_path, {"BIOPB_TENSOR_ALLOW_NO_TOKEN": "yes"})
    assert "--token" not in argv


def test_explicit_token_wins(tmp_path):
    token = "a" * 32
    argv, _ = _run(
        tmp_path,
        {"BIOPB_TENSOR_TOKEN": token, "BIOPB_TENSOR_ALLOW_NO_TOKEN": "1"},
    )
    assert argv[argv.index("--token") + 1] == token
