"""``python -m biopb_admin`` — the runnable admin entry.

The persistent supervisor is normally spawned by the core CLI's
``biopb admin start`` (which owns the pidfile / detach / stop plumbing) as
``python -m biopb_admin run ...``; this module is that target. It is plain
argparse (no typer dependency) and does no config resolution of its own -- the
caller passes the already-resolved tensor-server endpoint + launch parameters,
so the admin never imports ``biopb_tensor_server`` config (invariant I2).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ._admin import run_admin
from ._supervisor import DataPlaneSpec


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="biopb-admin", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run the admin control plane (foreground)")
    run.add_argument("--admin-host", default=None)
    run.add_argument("--admin-port", type=int, default=None)
    run.add_argument("--config", required=True, help="tensor-server config path")
    run.add_argument("--grpc-host", default="127.0.0.1")
    run.add_argument("--grpc-port", type=int, default=8815)
    run.add_argument("--web-host", default="127.0.0.1")
    run.add_argument("--web-port", type=int, default=8814)
    run.add_argument("--static-dir", default=None)
    run.add_argument("--log-level", default="INFO")
    run.add_argument("--server-log", default=None, help="data-plane stdout/stderr log")
    run.add_argument("--token", default=None, help="tensor-server access token")
    run.add_argument(
        "--local-bypass",
        action="store_true",
        help="all-localhost, no token: set BIOPB_WEB_DEV_BYPASS in the child",
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
    args = _build_parser().parse_args(argv)
    if args.command != "run":  # argparse requires a subcommand; defensive
        return 2

    # Defaults for the admin endpoint come from the shared core-SDK module so a
    # bare `python -m biopb_admin run` and the CLI agree on 8813.
    from biopb._config_admin import admin_host, admin_port

    spec = DataPlaneSpec(
        config=Path(args.config),
        grpc_host=args.grpc_host,
        grpc_port=args.grpc_port,
        web_host=args.web_host,
        web_port=args.web_port,
        static_dir=Path(args.static_dir) if args.static_dir else None,
        log_level=args.log_level,
        server_log=Path(args.server_log) if args.server_log else None,
        token=args.token,
        local_bypass=args.local_bypass,
    )
    return run_admin(
        spec,
        admin_host=args.admin_host or admin_host(),
        admin_port=args.admin_port or admin_port(),
        data_plane=args.data_plane,
        ensure_timeout=args.ensure_timeout,
        win_sentinel=Path(args.win_sentinel) if args.win_sentinel else None,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    sys.exit(main())
