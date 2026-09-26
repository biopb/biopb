"""CLI client for algorithm servers (the ``biopb.image`` Ops protocol).

Commands:
    servers     List the algorithm servers the control knows, with their state
    ops         List the operations an algorithm server offers
    process     Run one operation on an image
"""

import json
import sys
import time
from typing import Literal, Optional

import grpc
import imageio
import typer
from google.protobuf import empty_pb2, json_format, struct_pb2
from rich.console import Console
from rich.table import Table

from biopb import _algorithms
from biopb.image import Arg, Call, ImageData, OpList, OpsStub
from biopb.image.utils import (
    deserialize_image_data,
    serialize_from_numpy_to_image_data,
)
from biopb.tensor.serialized_pb2 import SerializedTensor

app = typer.Typer(
    name="image",
    help="Call algorithm servers.",
)
console = Console()
stderr_console = Console(stderr=True)


def _log_timing(start_time: float) -> None:
    """Print elapsed time since start_time to stderr."""
    elapsed = time.time() - start_time
    stderr_console.print(f"[dim]Completed in {elapsed:.2f}s[/dim]")


def _parse_server_address(server: str) -> tuple[str, bool]:
    """Parse server address, strip grpc:// or grpcs:// prefix.

    Returns:
        Tuple of (address, use_tls)
    """
    if server.startswith("grpcs://"):
        return server[8:], True
    if server.startswith("grpc://"):
        return server[7:], False
    return server, False


def _create_grpc_channel(server: str) -> grpc.Channel:
    """Create gRPC channel with user-friendly error handling."""
    addr, use_tls = _parse_server_address(server)
    try:
        if use_tls:
            credentials = grpc.ssl_channel_credentials()
            return grpc.secure_channel(addr, credentials)
        else:
            return grpc.insecure_channel(addr)
    except Exception as exc:
        stderr_console.print(f"[red]Cannot connect to server at {server}:[/red] {exc}")
        raise typer.Exit(1)


def _infer_format(output: str, format: Optional[str]) -> Literal["pb", "pickle"]:
    """Infer output format from filename or explicit format option.

    Args:
        output: Output path or "-" for stdout
        format: Explicit format option (None to infer from filename)

    Returns:
        Format string: "pb" or "pickle"
    """
    if format:
        fmt = format.lower()
        if fmt not in ("pb", "pickle"):
            raise typer.BadParameter(f"Invalid format: {format}. Must be pb or pickle.")
        return fmt

    if output == "-":
        return "pb"  # stdout default is protobuf

    ext = output.lower()
    if ext.endswith((".pkl", ".pickle")):
        return "pickle"
    return "pb"  # default for .pb, no extension, etc.


def _parse_input(input_path: Optional[str]) -> tuple[bool, bytes]:
    """Read input from file or stdin.

    Returns:
        Tuple of (is_file_path, data_or_path)
        - If is_file_path is True: data_or_path is the file path string
        - If is_file_path is False: data_or_path is the raw bytes read from stdin
    """
    if input_path is None or input_path == "-":
        # Read from stdin
        stderr_console.print("[green]Reading input from stdin[/green]")
        data = sys.stdin.buffer.read()
        return (False, data)
    else:
        # Read from file
        stderr_console.print(f"[green]Reading input from file:[/green] {input_path}")
        return (True, input_path)


def _build_arg(is_file: bool, data_or_path) -> Arg:
    """A tensor argument from a file path or raw bytes: an image read with
    imageio, or a SerializedTensor (a reference to a tensor on a plane)."""
    if is_file:
        try:
            np_arr = imageio.imread(data_or_path)
            stderr_console.print(
                f"[green]Loaded image:[/green] shape={np_arr.shape}, dtype={np_arr.dtype}"
            )
            return Arg(eager=serialize_from_numpy_to_image_data(np_arr).eager_data)
        except Exception as img_exc:
            stderr_console.print(
                f"[yellow]imageio failed, trying protobuf parse:[/yellow] {img_exc}"
            )
            with open(data_or_path, "rb") as f:
                raw_bytes = f.read()
            return _parse_bytes_to_arg(raw_bytes)
    return _parse_bytes_to_arg(data_or_path)


def _parse_bytes_to_arg(raw_bytes: bytes) -> Arg:
    """A SerializedTensor if the bytes parse as one, else an image."""
    try:
        serialized = SerializedTensor.FromString(raw_bytes)
        if serialized.location:
            stderr_console.print(
                f"[green]Parsed as SerializedTensor:[/green] location={serialized.location}"
            )
            return Arg(lazy=serialized)
    except Exception:
        pass
    try:
        np_arr = imageio.imread(raw_bytes)
        stderr_console.print(
            f"[green]Parsed as image:[/green] shape={np_arr.shape}, dtype={np_arr.dtype}"
        )
        return Arg(eager=serialize_from_numpy_to_image_data(np_arr).eager_data)
    except Exception as img_exc:
        stderr_console.print(f"[red]Cannot parse input:[/red] {img_exc}")
        raise typer.Exit(1)


def _output_path(output: str, key: str, tensor_count: int) -> str:
    """Where one tensor output goes: *output* itself when it is the only one,
    else *output* with the output's name before the extension."""
    if tensor_count <= 1 or output == "-":
        return output
    stem, dot, ext = output.rpartition(".")
    return f"{stem}-{key}.{ext}" if dot and stem else f"{output}-{key}"


def _write_tensor(arg: Arg, output: str, format: Literal["pb", "pickle"]) -> None:  # noqa: A002 - mirrors the --format option
    """Write one tensor output: pixels as an image file, a reference as a
    SerializedTensor (protobuf or pickle, to a file or stdout)."""
    if arg.WhichOneof("kind") == "eager":
        if output == "-":
            stderr_console.print(
                "[red]Error:[/red] stdout not allowed for eager image data. "
                "Provide output filename."
            )
            raise typer.Exit(1)
        np_arr = deserialize_image_data(ImageData(eager_data=arg.eager))
        stderr_console.print(
            f"[green]Output shape:[/green] {np_arr.shape}, dtype={np_arr.dtype}"
        )
        imageio.imwrite(output, np_arr)
        stderr_console.print(f"[green]Saved to:[/green] {output}")
        return

    serialized = arg.lazy
    stderr_console.print(f"[green]Tensor location:[/green] {serialized.location}")
    if format == "pb":
        pb_bytes = serialized.SerializeToString()
        if output == "-":
            sys.stdout.buffer.write(pb_bytes)
            stderr_console.print(
                f"[green]Protobuf written to stdout[/green] ({len(pb_bytes)} bytes)"
            )
        else:
            with open(output, "wb") as f:
                f.write(pb_bytes)
            stderr_console.print(
                f"[green]Protobuf saved to:[/green] {output} ({len(pb_bytes)} bytes)"
            )
        return

    import pickle

    if output == "-":
        pickle.dump(serialized, sys.stdout.buffer)
        stderr_console.print(
            "[green]Pickled SerializedTensor written to stdout[/green]"
        )
    else:
        with open(output, "wb") as f:
            pickle.dump(serialized, f)
        stderr_console.print(f"[green]Pickled saved to:[/green] {output}")


def _write_outputs(outputs, output: str, format: Literal["pb", "pickle"]) -> None:  # noqa: A002 - mirrors the --format option
    """Tensor outputs to *output*; JSON outputs printed plain, one per line,
    to stdout, or to stderr when a tensor goes to stdout. With more than one
    output, each line is ``<name>: <json>``."""
    tensors = [k for k, v in outputs.items() if v.WhichOneof("kind") != "json"]
    stream = sys.stderr if tensors and output == "-" else sys.stdout
    for key in sorted(outputs):
        arg = outputs[key]
        if arg.WhichOneof("kind") == "json":
            text = json.dumps(json_format.MessageToDict(arg.json))
            print(text if len(outputs) == 1 else f"{key}: {text}", file=stream)
        else:
            _write_tensor(arg, _output_path(output, key, len(tensors)), format)


def _state_style(state: str) -> str:
    """Rich colour for a probe state, so the table reads at a glance."""
    return {
        "up": "green",
        "unreachable": "red",
        "error": "red",
        "invalid": "yellow",
        "unknown": "yellow",
    }.get(state, "white")


@app.command(help="List the configured algorithm servers with a health probe.")
def servers(
    json_output: bool = typer.Option(
        False, "--json", help="Emit machine-readable JSON instead of a table"
    ),
    timeout: float = typer.Option(
        4.0, "--timeout", help="Per-server probe deadline in seconds"
    ),
) -> None:
    """List the algorithm servers, as the control reports them.

    The entries of ~/.config/biopb/algorithms/: a server file the control runs,
    with its state, and a url entry, probed. With no control, the url entries
    are probed from here. Read-only.

    Examples:
        biopb image servers
        biopb image servers --json --timeout 2
    """
    from biopb.control import algorithms

    rows = algorithms(timeout=timeout + 6)
    if rows is None:
        stderr_console.print(
            "[yellow]No control answered:[/yellow] probing url entries only."
        )
        rows = _algorithms.statuses(timeout=timeout)

    if json_output:
        print(json.dumps({"servers": rows}))
        raise typer.Exit(0)

    if not rows:
        stderr_console.print(
            "[yellow]No algorithm servers configured.[/yellow] Add a server file "
            'or a {"url": ...} file to [bold]~/.config/biopb/algorithms/[/bold].'
        )
        raise typer.Exit(0)

    table = Table(title="Algorithm plane servers")
    table.add_column("Name", style="cyan")
    table.add_column("Server", style="cyan")
    table.add_column("Scheme", style="blue")
    table.add_column("State", style="green")
    table.add_column("Ops", style="magenta")

    for r in rows:
        if r["state"] == "up":
            ops_cell = ", ".join(o.get("name", "") for o in r["ops"]) or "-"
        else:
            ops_cell = r.get("error") or "-"
        state = f"[{_state_style(r['state'])}]{r['state']}[/]"
        table.add_row(r["name"], r["target"], r["scheme"], state, ops_cell)

    console.print(table)
    n_up = sum(1 for r in rows if r["state"] == "up")
    stderr_console.print(
        f"\n[green]Servers:[/green] {len(rows)}  [green]up:[/green] {n_up}"
    )


_SERVER_OPTION = typer.Option(
    "grpc://localhost:50051",
    "--server",
    "-s",
    envvar="BIOPB_IMAGE_SERVER",
    help="Algorithm server URI (grpc:// or grpcs://)",
)
_TOKEN_OPTION = typer.Option(
    None,
    "--token",
    "-t",
    envvar="BIOPB_IMAGE_TOKEN",
    help="Bearer token for server authentication",
)


def _describe(stub: OpsStub, metadata) -> OpList:
    return stub.Describe(empty_pb2.Empty(), metadata=metadata, timeout=10)


@app.command(help="List the operations an algorithm server offers.")
def ops(server: str = _SERVER_OPTION, token: Optional[str] = _TOKEN_OPTION) -> None:
    """List the operations an algorithm server offers.

    Example:
        biopb image ops --server grpc://localhost:50051
        biopb image ops -s grpc://myhost:9000 --token mytoken123
    """
    start_time = time.time()
    channel = _create_grpc_channel(server)
    metadata = [("authorization", f"Bearer {token}")] if token else None
    try:
        listing = _describe(OpsStub(channel), metadata)
        if not listing.ops:
            stderr_console.print(f"[yellow]No operations found on {server}[/yellow]")
            _log_timing(start_time)
            return

        table = Table(title="Available Operations")
        table.add_column("Name", style="cyan")
        table.add_column("Description", style="green")
        table.add_column("Labels", style="magenta")
        table.add_column("Tensors", style="blue")
        table.add_column("Arguments", style="yellow")
        for info in listing.ops:
            tensors = ", ".join(
                f"{name}: {t.axes}" + (" (mapped)" if t.mapped else "")
                for name, t in sorted(info.tensors.items())
            )
            kwargs = json.dumps(json_format.MessageToDict(info.kwargs))
            table.add_row(
                info.name,
                info.description or "-",
                ", ".join(info.labels) or "-",
                tensors or "-",
                kwargs if info.kwargs.fields else "-",
            )
        console.print(table)
        stderr_console.print(
            f"\n[green]Server:[/green] {server}  [green]Operations:[/green] {len(listing.ops)}"
        )
        _log_timing(start_time)
    except grpc.RpcError as exc:
        stderr_console.print(f"[red]gRPC error:[/red] {exc.code()} - {exc.details()}")
        raise typer.Exit(1)
    except Exception as exc:
        stderr_console.print(f"[red]Error querying server:[/red] {exc}")
        raise typer.Exit(1)
    finally:
        channel.close()


@app.command(help="Run an operation on an algorithm server.")
def process(
    input: Optional[str] = typer.Argument(  # noqa: A002 - the CLI's positional name
        None,
        help="Input file path or '-' for stdin. If omitted, reads from stdin.",
    ),
    op: Optional[str] = typer.Option(
        None,
        "--op",
        "-o",
        help="Operation name (optional if the server has a single op)",
    ),
    tensor: Optional[str] = typer.Option(
        None,
        "--tensor",
        help="Which tensor argument the input is (optional if the op has one)",
    ),
    kwargs: Optional[str] = typer.Option(
        None,
        "--kwargs",
        "-k",
        help="The op's other arguments as a JSON object, e.g. '{\"sigma\": 2}'",
    ),
    output: str = typer.Option(
        "-",
        "--output",
        "-O",
        help="Output path. Use '-' for stdout. Eager data requires filename.",
    ),
    format: Optional[str] = typer.Option(  # noqa: A002 - public option name
        None,
        "--format",
        "-f",
        help="Output format for lazy data: pb (default) or pickle.",
    ),
    server: str = _SERVER_OPTION,
    token: Optional[str] = _TOKEN_OPTION,
) -> None:
    """Run one operation on an image.

    Input can be:
    - An image file (png, tiff, etc.) read via imageio
    - A protobuf SerializedTensor file (.pb)
    - Stdin containing protobuf or image bytes

    A tensor output is written to --output: pixels as an image file (stdout not
    allowed), a reference as protobuf (.pb) or pickle (.pkl). Several tensor
    outputs get their name before the extension. Other outputs are JSON,
    printed.

    Examples:
        biopb image process input.png --op gaussian --kwargs '{"sigma": 2}' -O out.png
        biopb image process input.pb --op segment -O out.pb
        biopb tensor get my-source -o - | biopb image process --op segment -O -
    """
    start_time = time.time()
    channel = _create_grpc_channel(server)
    fmt = _infer_format(output, format)
    metadata = [("authorization", f"Bearer {token}")] if token else None

    try:
        stub = OpsStub(channel)
        listing = _describe(stub, metadata)
        by_name = {info.name: info for info in listing.ops}
        if op is None:
            if len(by_name) != 1:
                stderr_console.print(
                    f"[red]Error:[/red] --op is required; the server has {sorted(by_name)}"
                )
                raise typer.Exit(1)
            op = next(iter(by_name))
        info = by_name.get(op)
        if info is None:
            stderr_console.print(
                f"[red]Error:[/red] no op {op!r}; the server has {sorted(by_name)}"
            )
            raise typer.Exit(1)
        if tensor is None:
            if len(info.tensors) != 1:
                stderr_console.print(
                    f"[red]Error:[/red] --tensor is required; {op} takes {sorted(info.tensors)}"
                )
                raise typer.Exit(1)
            tensor = next(iter(info.tensors))

        is_file, data_or_path = _parse_input(input)
        args = {tensor: _build_arg(is_file, data_or_path)}
        for name, value in json.loads(kwargs or "{}").items():
            args[name] = Arg(json=json_format.ParseDict(value, struct_pb2.Value()))

        stderr_console.print(f"[green]Sending request to[/green] {server} (op: {op})")
        outputs = None
        for event in stub.Call(Call(op=op, args=args), metadata=metadata):
            if event.progress and not event.outputs:
                stderr_console.print(f"[dim]{event.progress}[/dim]")
            if event.outputs:
                outputs = event.outputs
        if outputs is None:
            stderr_console.print("[yellow]The op returned nothing.[/yellow]")
        else:
            _write_outputs(outputs, output, fmt)
        _log_timing(start_time)

    except typer.Exit:
        raise
    except grpc.RpcError as exc:
        stderr_console.print(f"[red]gRPC error:[/red] {exc.code()} - {exc.details()}")
        raise typer.Exit(1)
    except Exception as exc:
        stderr_console.print(f"[red]Error processing image:[/red] {exc}")
        raise typer.Exit(1)
    finally:
        channel.close()


if __name__ == "__main__":
    app()
