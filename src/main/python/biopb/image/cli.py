"""CLI client for algorithm servers (the ``biopb.image`` Ops protocol).

A server is a ``grpc://`` / ``grpcs://`` URL, or the name of a registry entry
the control manages (see ``biopb algorithm list``).

Commands:
    ops         List the operations an algorithm server offers
    call        Run one operation on an image
"""

import json
import sys
import time
from typing import Literal, Optional

import grpc
import imageio
import typer
from rich.console import Console
from rich.table import Table

from biopb.image import (
    Arg,
    OpInfo,
    connect,
    decode_arg,
    encode_arg,
    json_arg,
    json_value,
    op_error,
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
            return encode_arg(np_arr)
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
        return encode_arg(np_arr)
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
        np_arr = decode_arg(arg)
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
            text = json.dumps(json_value(arg.json))
            print(text if len(outputs) == 1 else f"{key}: {text}", file=stream)
        else:
            _write_tensor(arg, _output_path(output, key, len(tensors)), format)


_SERVER_ARG = typer.Argument(
    ..., help="grpc:// or grpcs:// URL, or a registry entry name"
)
_TOKEN_OPTION = typer.Option(
    None,
    "--token",
    "-t",
    envvar="BIOPB_IMAGE_TOKEN",
    help="Bearer token for server authentication (a registry name brings its own)",
)


def _connect(server: str, token: Optional[str]):
    try:
        return connect(server, token=token)
    except (LookupError, RuntimeError, ValueError) as exc:
        stderr_console.print(f"[red]Cannot connect to {server}:[/red] {exc}")
        raise typer.Exit(1) from exc


@app.command(help="List the operations an algorithm server offers.")
def ops(server: str = _SERVER_ARG, token: Optional[str] = _TOKEN_OPTION) -> None:
    """List the operations an algorithm server offers.

    Example:
        biopb image ops grpc://localhost:50051
        biopb image ops cellpose
    """
    start_time = time.time()
    client = _connect(server, token)
    try:
        listing = client.describe()
        if not listing.ops:
            stderr_console.print(f"[yellow]No operations found on {server}[/yellow]")
            _log_timing(start_time)
            return

        table = Table(title="Available Operations")
        table.add_column("Name", style="cyan")
        table.add_column("Description", style="green")
        table.add_column("Labels", style="magenta")
        table.add_column("Mode", style="white")
        table.add_column("Tensors", style="blue")
        table.add_column("Arguments", style="yellow")
        for info in listing.ops:
            tensors = ", ".join(
                f"{name}: {t.axes}" + (" (mapped)" if t.mapped else "")
                for name, t in sorted(info.tensors.items())
            )
            mode = OpInfo.InputMode.Name(info.input).lower()
            if info.streaming:
                mode += ", streaming"
            table.add_row(
                info.name,
                info.description or "-",
                ", ".join(info.labels) or "-",
                mode,
                tensors or "-",
                info.kwargs or "-",
            )
        console.print(table)
        stderr_console.print(
            f"\n[green]Server:[/green] {server}  [green]Operations:[/green] {len(listing.ops)}"
        )
        _log_timing(start_time)
    except grpc.RpcError as exc:
        stderr_console.print(f"[red]gRPC error:[/red] {exc.code()} - {exc.details()}")
        raise typer.Exit(1)
    finally:
        client.close()


@app.command(help="Run an operation on an algorithm server.")
def call(
    server: str = _SERVER_ARG,
    op: Optional[str] = typer.Argument(
        None, help="Operation name (optional if the server has a single op)"
    ),
    input: Optional[str] = typer.Option(  # noqa: A002 - the CLI's option name
        None,
        "--input",
        "-i",
        help="Input file path or '-' for stdin. If omitted, reads from stdin.",
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
        biopb image call cellpose gaussian -i input.png -k '{"sigma": 2}' -O out.png
        biopb image call grpc://host:50051 segment -i input.pb -O out.pb
        biopb tensor get my-source -o - | biopb image call cellpose segment -O -
    """
    start_time = time.time()
    fmt = _infer_format(output, format)
    client = _connect(server, token)

    try:
        by_name = {info.name: info for info in client.describe().ops}
        if op is None:
            if len(by_name) != 1:
                stderr_console.print(
                    f"[red]Error:[/red] OP is required; the server has {sorted(by_name)}"
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
            args[name] = json_arg(value)

        stderr_console.print(f"[green]Sending request to[/green] {server} (op: {op})")
        outputs = None
        for event in client.events(op, args):
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
        stderr_console.print(f"[red]Error:[/red] {op_error(op or '', exc)}")
        raise typer.Exit(1)
    except Exception as exc:
        stderr_console.print(f"[red]Error processing image:[/red] {exc}")
        raise typer.Exit(1)
    finally:
        client.close()


if __name__ == "__main__":
    app()
