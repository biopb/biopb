"""``python -m biopb_control`` — the runnable control entry.

The persistent supervisor is normally spawned by the core CLI's
``biopb control start`` (which owns the pidfile / detach / stop plumbing) as
``python -m biopb_control run ...``; this module is that target. It is plain
argparse (no typer dependency) and does no config resolution of its own -- the
caller passes the already-resolved tensor-server endpoint + launch parameters,
so the control never imports ``biopb_tensor_server`` config (invariant I2).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ._run import run_control
from ._supervisor import DataPlaneSpec


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="biopb-control", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run the control plane (foreground)")
    run.add_argument("--control-host", default=None)
    run.add_argument("--control-port", type=int, default=None)
    run.add_argument("--config", required=True, help="tensor-server config path")
    run.add_argument("--grpc-host", default="127.0.0.1")
    run.add_argument("--grpc-port", type=int, default=8815)
    run.add_argument("--web-host", default="127.0.0.1")
    run.add_argument("--web-port", type=int, default=8814)
    run.add_argument("--static-dir", default=None)
    run.add_argument("--log-level", default="INFO")
    run.add_argument("--server-log", default=None, help="data-plane stdout/stderr log")
    run.add_argument(
        "--token",
        default=None,
        help="tensor-server access token (prefer BIOPB_TENSOR_TOKEN in the env; "
        "a command line is world-readable via `ps`)",
    )
    run.add_argument(
        "--remote",
        action="store_true",
        help="serve on the network behind a token: bind the control listener "
        "publicly (0.0.0.0) and require a token. Without it the control binds "
        "loopback and runs tokenless.",
    )
    run.add_argument(
        "--no-data-plane",
        dest="data_plane",
        action="store_false",
        help="adopt-only: monitor/restart a running server, do not spawn one",
    )
    run.add_argument("--ensure-timeout", type=float, default=60.0)
    run.add_argument("--win-sentinel", default=None, help="Windows stop-sentinel path")
    return parser


def main(argv: list[str] | None = None) -> int:
    import os

    args = _build_parser().parse_args(argv)
    if args.command != "run":  # argparse requires a subcommand; defensive
        return 2

    # The token arrives via the env, not the argv, so it never appears in the
    # process command line (biopb/biopb#414). `biopb control start` exports
    # BIOPB_TENSOR_TOKEN into this child; --token stays honored for a direct
    # `python -m biopb_control run` invocation.
    token = args.token or os.environ.get("BIOPB_TENSOR_TOKEN")

    # Fail-closed: remote mode binds the control's HTTP listener publicly, so it
    # must carry a token. (`biopb control start --remote` always resolves one and
    # exports it here; this guards a direct `python -m biopb_control run --remote`.)
    if args.remote and not token:
        print(
            "biopb-control: --remote requires an access token "
            "(set BIOPB_TENSOR_TOKEN or pass --token).",
            file=sys.stderr,
        )
        return 2

    # Defaults for the control endpoint come from the shared core-SDK module so a
    # bare `python -m biopb_control run` and the CLI agree on 8813. Remote mode
    # binds all interfaces so the browser UI is reachable off-box.
    from biopb._config_control import control_host, control_port

    resolved_control_host = args.control_host or (
        "0.0.0.0" if args.remote else control_host()
    )

    spec = DataPlaneSpec(
        config=Path(args.config),
        grpc_host=args.grpc_host,
        grpc_port=args.grpc_port,
        web_host=args.web_host,
        web_port=args.web_port,
        static_dir=Path(args.static_dir) if args.static_dir else None,
        log_level=args.log_level,
        server_log=Path(args.server_log) if args.server_log else None,
        token=token,
    )
    return run_control(
        spec,
        control_host=resolved_control_host,
        control_port=args.control_port or control_port(),
        data_plane=args.data_plane,
        ensure_timeout=args.ensure_timeout,
        win_sentinel=Path(args.win_sentinel) if args.win_sentinel else None,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    sys.exit(main())
