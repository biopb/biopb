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
    run.add_argument(
        "--tls",
        action="store_true",
        help="Serve the data plane's flight port over TLS (passed to `launch`).",
    )
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
        help="deprecated and redundant: a public --grpc-host already says the "
        "deployment is public. Kept as an extra token-required trigger; it does "
        "NOT publish this control listener (use --control-host for that).",
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
    #
    # Strip surrounding whitespace at this single resolution point so every
    # consumer carries the *same* bytes: the middleware compares requests against
    # `spec.token` un-touched, the supervisor exports it to the tensor server, and
    # the credential handoff writes it to a file that `read_credential` reads back
    # `.strip()`ed. Without this, a token sourced with a trailing newline
    # (`BIOPB_TENSOR_TOKEN=$(cat tokenfile)`) would be enforced with the newline
    # but read back without it, so a local client's credential-derived token would
    # 401 against the very control that wrote it. A whitespace-only value collapses
    # to `None` (tokenless) rather than a bogus empty credential.
    token = (args.token or os.environ.get("BIOPB_TENSOR_TOKEN") or "").strip() or None

    # Defaults for the control endpoint come from the shared core-SDK module so a
    # bare `python -m biopb_control run` and the CLI agree on 8813.
    #
    # `--remote` does NOT touch this bind (biopb/biopb#614). It used to force
    # 0.0.0.0, which published a plaintext-HTTP admin UI and put the data-plane
    # token on the wire in the clear -- the opposite of what the TLS work under
    # biopb/biopb#604 was for, since the control has no TLS support at all. The
    # flag now means only "the flight plane is public, so require a token"; the
    # browser reaches this listener over an SSH tunnel. Publishing it stays
    # possible, but only as the deliberate, named act of passing a public
    # `--control-host` (or BIOPB_CONTROL_HOST) -- e.g. behind an operator's own
    # TLS proxy.
    from biopb import _web_auth
    from biopb._endpoints import control_host, control_port

    resolved_control_host = args.control_host or control_host()

    # Fail-closed: no listener reachable off-box may run token-less. Two of them
    # can be, and each is checked on its *resolved bind* rather than on a mode
    # flag:
    #
    #   - this control listener. Without a token its /api/* gate falls back to a
    #     loopback-`Host` check, which a non-browser client trivially spoofs
    #     (`Host: 127.0.0.1`) to drive the stop/restart/session verbs. Public only
    #     via an explicit --control-host / BIOPB_CONTROL_HOST.
    #   - the data plane. A public --grpc-host serves pixels and the whole data
    #     API off-box, so it must be authenticated even though *this* listener is
    #     loopback. Deriving it from the address the parent already passes means
    #     the two layers read one fact through one predicate and cannot disagree
    #     about it (biopb/biopb#614).
    #
    # `--remote` is redundant now that --grpc-host carries this, but stays honored
    # so a direct invocation that still passes it is never accepted token-less.
    control_is_public = _web_auth.host_is_public_bind(resolved_control_host)
    plane_is_public = _web_auth.host_is_public_bind(args.grpc_host)
    if not token and (args.remote or control_is_public or plane_is_public):
        exposed = "control API" if control_is_public else "data plane"
        print(
            f"biopb-control: a network-reachable {exposed} bind requires an access "
            "token (set BIOPB_TENSOR_TOKEN or pass --token). Bind --grpc-host and "
            "--control-host to loopback (127.0.0.1) for a tokenless local "
            "deployment.",
            file=sys.stderr,
        )
        return 2

    spec = DataPlaneSpec(
        config=Path(args.config),
        grpc_host=args.grpc_host,
        grpc_port=args.grpc_port,
        tls=args.tls,
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
