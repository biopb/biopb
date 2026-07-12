"""CLI client for ProcessImage gRPC services.

Commands:
    servers     List the configured algorithm-plane servers with health + ops
    ops         List available operations from a ProcessImage server
    process     Execute an operation on input image data
"""

import json
import sys
import time
from typing import Literal, Optional

import grpc
import imageio
import typer
from google.protobuf import empty_pb2
from rich.console import Console
from rich.table import Table

from biopb import _algorithms
from biopb.image import (
    ImageData,
    OpNames,
    OpSchema,
    ProcessImageStub,
    ProcessRequest,
    ProcessResponse,
)
from biopb.image.utils import (
    deserialize_image_data,
    serialize_from_numpy_to_image_data,
)
from biopb.tensor.serialized_pb2 import SerializedTensor

app = typer.Typer(
    name="image",
    help="ProcessImage client operations",
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
    if ext.endswith(".pkl") or ext.endswith(".pickle"):
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


def _build_image_data(is_file: bool, data_or_path: str) -> "ImageData":
    """Build ImageData from file path or raw bytes.

    Args:
        is_file: True if data_or_path is a file path, False if raw bytes
        data_or_path: File path string or raw bytes

    Returns:
        ImageData protobuf message
    """

    if is_file:
        # Try imageio for image files
        try:
            np_arr = imageio.imread(data_or_path)
            stderr_console.print(
                f"[green]Loaded image:[/green] shape={np_arr.shape}, dtype={np_arr.dtype}"
            )
            return serialize_from_numpy_to_image_data(np_arr)
        except Exception as img_exc:
            stderr_console.print(
                f"[yellow]imageio failed, trying protobuf parse:[/yellow] {img_exc}"
            )
            # Fallback: read file as protobuf
            with open(data_or_path, "rb") as f:
                raw_bytes = f.read()
            return _parse_bytes_to_image_data(raw_bytes)
    else:
        # Raw bytes from stdin - try protobuf first
        return _parse_bytes_to_image_data(data_or_path)


def _parse_bytes_to_image_data(raw_bytes: bytes) -> "ImageData":
    """Parse raw bytes to ImageData.

    Try protobuf SerializedTensor first, fallback to imageio.
    """
    # Try protobuf SerializedTensor
    try:
        serialized = SerializedTensor.FromString(raw_bytes)
        stderr_console.print(
            f"[green]Parsed as SerializedTensor:[/green] location={serialized.location}"
        )
        return ImageData(lazy_data=serialized)
    except Exception:
        pass

    # Try imageio for image data
    try:
        np_arr = imageio.imread(raw_bytes)
        stderr_console.print(
            f"[green]Parsed as image:[/green] shape={np_arr.shape}, dtype={np_arr.dtype}"
        )
        return serialize_from_numpy_to_image_data(np_arr)
    except Exception as img_exc:
        stderr_console.print(f"[red]Cannot parse input:[/red] {img_exc}")
        raise typer.Exit(1)


def _write_output(
    response: ProcessResponse,
    output: str,
    format: Literal["pb", "pickle"],
) -> None:
    """Write response to output file or stdout.

    Args:
        response: ProcessResponse from server
        output: Output path or "-" for stdout
        format: Output format for lazy data ("pb" or "pickle")
    """
    image_data = response.image_data

    # Print annotation if present
    if response.annotation:
        stderr_console.print(f"[green]Server annotation:[/green] {response.annotation}")

    # Check data type
    data_type = image_data.WhichOneof("data")

    if data_type == "eager_data":
        # Eager tensor - must save as image file
        if output == "-":
            stderr_console.print(
                "[red]Error:[/red] stdout not allowed for eager image data. "
                "Provide output filename."
            )
            raise typer.Exit(1)

        stderr_console.print("[green]Server returned eager data[/green]")
        np_arr = deserialize_image_data(image_data)
        stderr_console.print(
            f"[green]Output shape:[/green] {np_arr.shape}, dtype={np_arr.dtype}"
        )
        imageio.imwrite(output, np_arr)
        stderr_console.print(f"[green]Saved to:[/green] {output}")

    elif data_type == "lazy_data":
        # Lazy tensor - protobuf or pickle
        stderr_console.print("[green]Server returned lazy data[/green]")
        serialized = image_data.lazy_data
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
        else:
            # Pickle format
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

    else:
        stderr_console.print(f"[red]Error:[/red] Unknown data type: {data_type}")
        raise typer.Exit(1)


def _state_style(state: str) -> str:
    """Rich colour for a probe state, so the table reads at a glance."""
    return {
        "serving": "green",
        "unreachable": "red",
        "error": "red",
        "invalid": "yellow",
        "unknown": "yellow",
    }.get(state, "white")


@app.command()
def servers(
    json_output: bool = typer.Option(
        False, "--json", help="Emit machine-readable JSON instead of a table"
    ),
    timeout: float = typer.Option(
        4.0, "--timeout", help="Per-server probe deadline in seconds"
    ),
) -> None:
    """List the configured algorithm-plane servers with a health + ops probe.

    Reads the ProcessImage servers wired into the biopb-mcp config (under
    mcp.services.process_image_servers) -- the same set an agent kernel exposes as
    ops -- and probes each for liveness and its advertised operations. This is the
    CLI face of the control dashboard's Algorithm plane section: read-only (no
    lifecycle control), never writes config.

    Examples:
        biopb image servers
        biopb image servers --json --timeout 2
    """
    rows = _algorithms.statuses(timeout=timeout)

    if json_output:
        print(json.dumps({"servers": rows}))
        raise typer.Exit(0)

    if not rows:
        stderr_console.print(
            "[yellow]No algorithm servers configured.[/yellow] Add ProcessImage "
            "server URLs under [bold]mcp.services.process_image_servers[/bold] in "
            "the biopb-mcp config."
        )
        raise typer.Exit(0)

    table = Table(title="Algorithm plane servers")
    table.add_column("Server", style="cyan")
    table.add_column("Scheme", style="blue")
    table.add_column("State", style="green")
    table.add_column("Ops", style="magenta")

    for r in rows:
        if r["state"] == "serving":
            ops_cell = (
                "(single-op)" if r.get("single_op") else ", ".join(r["ops"]) or "-"
            )
        else:
            ops_cell = r.get("error") or "-"
        state = f"[{_state_style(r['state'])}]{r['state']}[/]"
        table.add_row(r["target"], r["scheme"], state, ops_cell)

    console.print(table)
    n_serving = sum(1 for r in rows if r["state"] == "serving")
    stderr_console.print(
        f"\n[green]Servers:[/green] {len(rows)}  [green]serving:[/green] {n_serving}"
    )


@app.command()
def ops(
    server: str = typer.Option(
        "grpc://localhost:50051",
        "--server",
        "-s",
        envvar="BIOPB_IMAGE_SERVER",
        help="ProcessImage server URI",
    ),
    token: Optional[str] = typer.Option(
        None,
        "--token",
        "-t",
        envvar="BIOPB_IMAGE_TOKEN",
        help="Bearer token for server authentication",
    ),
) -> None:
    """List available operations from a ProcessImage server.

    Example:
        biopb image ops --server grpc://localhost:50051
        biopb image ops -s grpc://myhost:9000 --token mytoken123
        BIOPB_IMAGE_TOKEN=mytoken123 biopb image ops
    """
    start_time = time.time()
    channel = _create_grpc_channel(server)
    metadata = [("authorization", f"Bearer {token}")] if token else None
    try:
        stub = ProcessImageStub(channel)
        response: OpNames = stub.GetOpNames(
            empty_pb2.Empty(), metadata=metadata, timeout=10
        )

        if not response.names:
            stderr_console.print(f"[yellow]No operations found on {server}[/yellow]")
            _log_timing(start_time)
            return

        table = Table(title="Available Operations")
        table.add_column("Name", style="cyan")
        table.add_column("Description", style="green")
        table.add_column("Labels", style="magenta")
        table.add_column("Input Hint", style="blue")

        for name in response.names:
            schema: OpSchema = response.op_schemas.get(name)
            if schema:
                labels_str = ", ".join(schema.labels) if schema.labels else "-"
                hint_parts = []
                if schema.input_shape_hint:
                    if schema.input_shape_hint.expected_singletons:
                        hint_parts.append(
                            f"singleton: {','.join(schema.input_shape_hint.expected_singletons)}"
                        )
                    if schema.input_shape_hint.required_multivalue:
                        hint_parts.append(
                            f"multi: {','.join(schema.input_shape_hint.required_multivalue)}"
                        )
                hint_str = "; ".join(hint_parts) if hint_parts else "-"
                table.add_row(name, schema.description or "-", labels_str, hint_str)
            else:
                table.add_row(name, "-", "-", "-")

        console.print(table)
        stderr_console.print(
            f"\n[green]Server:[/green] {server}  [green]Operations:[/green] {len(response.names)}"
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


@app.command()
def process(
    input: Optional[str] = typer.Argument(
        None,
        help="Input file path or '-' for stdin. If omitted, reads from stdin.",
    ),
    op: Optional[str] = typer.Option(
        None,
        "--op",
        "-o",
        help="Operation name (optional if server has single/default op)",
    ),
    output: str = typer.Option(
        "-",
        "--output",
        "-O",
        help="Output path. Use '-' for stdout. Eager data requires filename.",
    ),
    format: Optional[str] = typer.Option(
        None,
        "--format",
        "-f",
        help="Output format for lazy data: pb (default) or pickle.",
    ),
    server: str = typer.Option(
        "grpc://localhost:50051",
        "--server",
        "-s",
        envvar="BIOPB_IMAGE_SERVER",
        help="ProcessImage server URI",
    ),
    token: Optional[str] = typer.Option(
        None,
        "--token",
        "-t",
        envvar="BIOPB_IMAGE_TOKEN",
        help="Bearer token for server authentication",
    ),
) -> None:
    """Execute an image processing operation.

    Input can be:
    - An image file (png, tiff, etc.) read via imageio
    - A protobuf SerializedTensor file (.pb)
    - Stdin containing protobuf or image bytes

    Output depends on server response:
    - Eager data: saved as image file (stdout not allowed)
    - Lazy data: protobuf (.pb) or pickle (.pkl) format

    Examples:
        biopb image process input.png --op mock_echo --output output.png
        biopb image process input.pb --op mock_echo -O output.pb
        biopb tensor get my-source -o - | biopb image process --op segment -O -
        biopb image process input.png --op segment --token mytoken123 -O output.pb
    """
    start_time = time.time()
    channel = _create_grpc_channel(server)
    fmt = _infer_format(output, format)
    metadata = [("authorization", f"Bearer {token}")] if token else None

    try:
        # Parse input
        is_file, data_or_path = _parse_input(input)
        image_data = _build_image_data(is_file, data_or_path)

        # Build request
        request = ProcessRequest(image_data=image_data)
        if op:
            request.op_name = op

        stderr_console.print(
            f"[green]Sending request to[/green] {server}"
            + (f" (op: {op})" if op else "")
        )

        # Call server
        stub = ProcessImageStub(channel)
        response: ProcessResponse = stub.Run(request, metadata=metadata, timeout=60)

        # Write output
        _write_output(response, output, fmt)
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
