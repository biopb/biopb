"""Client-side diagnostic CLI for querying TensorFlight servers.

Commands:
    query       List sources and tensors from a running server
    metadata    Inspect source metadata and tensor descriptors
    get         Download a tensor to file or stdout
    stats       Compute statistics (min, max, mean) for a tensor
"""

from typing import Optional, Tuple
import json
import pickle
import sys
import time

from pathlib import Path
import dask
import numpy as np
import typer
from rich.console import Console
from rich.table import Table

from biopb.tensor.client import TensorFlightClient


app = typer.Typer(
    name="tensor",
    help="TensorFlight client diagnostics",
)
# Main output to stdout; stderr console for logging/timing only
console = Console()
stderr_console = Console(stderr=True)


def _log_timing(start_time: float) -> None:
    """Print elapsed time since start_time to stderr."""
    elapsed = time.time() - start_time
    stderr_console.print(f"[dim]Completed in {elapsed:.2f}s[/dim]")


def _create_flight_client(location: str, cache_bytes: int) -> TensorFlightClient:
    """Create a Flight client, with user-friendly error handling."""
    try:
        return TensorFlightClient(location=location, cache_bytes=cache_bytes)
    except Exception as exc:
        stderr_console.print(f"[red]Cannot connect to server at {location}:[/red] {exc}")
        raise typer.Exit(1)


def _parse_slice_hint(slice_hint: Optional[str]) -> Optional[Tuple[slice, ...]]:
    """Parse a comma-separated slice hint string into a tuple of slices.

    Format: "start:stop,start:stop,..." where start and stop are optional integers.
    Example: "0:100,50:150" → (slice(0, 100), slice(50, 150))
    """
    if not slice_hint:
        return None

    try:
        dims = []
        for part in slice_hint.split(","):
            part = part.strip()
            if not part:
                continue
            if ":" not in part:
                raise ValueError("Slice must be in start:stop format")
            start_str, stop_str = part.split(":", 1)
            start = int(start_str) if start_str else None
            stop = int(stop_str) if stop_str else None
            dims.append(slice(start, stop))
        return tuple(dims) if dims else None
    except (ValueError, IndexError) as e:
        raise typer.BadParameter(f"Invalid slice format: {e}")


def _parse_array_id(array_id: str) -> Tuple[str, Optional[str]]:
    """Parse array_id into source_id and optional tensor_id.

    Format: "source_id[/tensor_id]"
    Examples:
        "my-source" -> ("my-source", None)
        "my-source/pos_0" -> ("my-source", "pos_0")

    Returns:
        Tuple of (source_id, tensor_id)
    """
    if "/" in array_id:
        source_id, tensor_id = array_id.split("/", 1)
        return source_id, tensor_id
    return array_id, None


@app.command()
def query(
    server: str = typer.Option(
        "grpc://localhost:8815",
        "--server",
        "-s",
        help="TensorFlight server URI",
    ),
    cache_bytes: int = typer.Option(
        100_000_000,
        "--cache-bytes",
        help="Maximum bytes for client-side chunk cache",
    ),
):
    """List all data sources and tensors from a running TensorFlight server.

    Example:
        biopb tensor query --server grpc://localhost:8815
        biopb tensor query -s grpc://myhost:9000
    """
    start_time = time.time()
    client = _create_flight_client(server, cache_bytes)
    try:
        sources = client.list_sources()
        if not sources:
            stderr_console.print(f"[yellow]No sources found on {server}[/yellow]")
            _log_timing(start_time)
            return

        table = Table(title="Available Tensor Sources")
        table.add_column("Source ID", style="cyan")
        table.add_column("Tensor ID", style="magenta")
        table.add_column("Shape", style="green")
        table.add_column("Dtype", style="blue")

        for source_id, source_desc in sources.items():
            if not source_desc.tensors:
                table.add_row(source_id, "<no tensors>", "-", "-")
                continue
            for tensor_desc in source_desc.tensors:
                table.add_row(
                    source_id,
                    tensor_desc.array_id,
                    str(list(tensor_desc.shape)),
                    str(tensor_desc.dtype),
                )

        console.print(table)

        cache_info = client.cache_info()
        console.print(
            f"\n[green]Server:[/green] {server}  "
            f"[green]Sources:[/green] {len(sources)}  "
            f"[green]Cache:[/green] {cache_info.get('size_bytes', 0):,} bytes  "
            f"hits={cache_info.get('hits', 0)} misses={cache_info.get('misses', 0)}"
        )
        _log_timing(start_time)
    except typer.Exit:
        raise
    except Exception as exc:
        stderr_console.print(f"[red]Error querying server:[/red] {exc}")
        raise typer.Exit(1)
    finally:
        client.close()


@app.command()
def metadata(
    source_id: str = typer.Argument(..., help="Source identifier to inspect"),
    server: str = typer.Option(
        "grpc://localhost:8815",
        "--server",
        "-s",
        help="TensorFlight server URI",
    ),
    tensor: Optional[str] = typer.Option(
        None,
        "--tensor",
        "-t",
        help="Specific tensor ID to inspect (optional)",
    ),
    cache_bytes: int = typer.Option(
        100_000_000,
        "--cache-bytes",
        help="Maximum bytes for client-side chunk cache",
    ),
):
    """Inspect source metadata and tensor descriptors.

    Example:
        biopb tensor metadata my-source
        biopb tensor metadata my-source --tensor pos_0
        biopb tensor metadata my-source -s grpc://myhost:9000
    """
    start_time = time.time()
    client = _create_flight_client(server, cache_bytes)
    try:
        sources = client.list_sources()
        if source_id not in sources:
            stderr_console.print(f"[red]Source not found:[/red] {source_id}")
            raise typer.Exit(1)

        source_desc = sources[source_id]

        # Show metadata for the entire source
        console.print(f"[bold green]Source:[/bold green] {source_id}")
        console.print(f"[bold green]Tensors:[/bold green] {len(source_desc.tensors)}")

        # List all tensors in the source
        for tensor_desc in source_desc.tensors:
            console.print(
                f"  [cyan]{tensor_desc.array_id}[/cyan] "
                f"shape={list(tensor_desc.shape)} dtype={tensor_desc.dtype}"
            )

        # If --tensor specified, show detailed descriptor info
        if tensor:
            tensor_desc = next(
                (t for t in source_desc.tensors if t.array_id == tensor),
                None,
            )
            if tensor_desc is None:
                stderr_console.print(f"[red]Tensor not found:[/red] {tensor}")
                raise typer.Exit(1)

            console.print(f"\n[bold green]Tensor Descriptor: {tensor}[/bold green]")
            detail_table = Table(show_header=False)
            detail_table.add_row("Array ID", tensor_desc.array_id)
            detail_table.add_row("Shape", str(list(tensor_desc.shape)))
            detail_table.add_row("Dtype", str(tensor_desc.dtype))
            console.print(detail_table)

        # Fetch and display source-level metadata (OME/vendor JSON)
        console.print(f"\n[bold green]Source Metadata:[/bold green]")
        src_metadata = client.get_source_metadata(source_id)
        if src_metadata:
            # Pass JSON string directly to print_json for proper Rich formatting
            console.print_json(json.dumps(src_metadata))
        else:
            stderr_console.print("[yellow]No metadata available[/yellow]")

        _log_timing(start_time)
    except typer.Exit:
        raise
    except Exception as exc:
        console.print(f"[red]Error fetching metadata:[/red] {exc}")
        raise typer.Exit(1)
    finally:
        client.close()


@app.command()
def get(
    array_id: str = typer.Argument(
        ...,
        help="Array identifier: source_id/tensor_id (tensor_id optional for single-tensor sources)",
    ),
    output: str = typer.Option(
        "-",
        "--output",
        "-o",
        help="Output path for tensor data (pickle format). Use '-' for stdout.",
    ),
    server: str = typer.Option(
        "grpc://localhost:8815",
        "--server",
        "-s",
        help="TensorFlight server URI",
    ),
    slice_hint: Optional[str] = typer.Option(
        None,
        "--slice",
        "-S",
        help="Slice specification, e.g. '0:100,0:200'",
    ),
    cache_bytes: int = typer.Option(
        100_000_000,
        "--cache-bytes",
        help="Maximum bytes for client-side chunk cache",
    ),
):
    """Download a tensor to file or stdout (pickle format).

    The slice option restricts the region downloaded. If not specified,
    the entire tensor is downloaded.

    Example:
        biopb tensor get my-source -o output.pkl
        biopb tensor get my-source/pos_0 -o output.pkl
        biopb tensor get my-source/pos_0 -o - > output.pkl
        biopb tensor get my-source/pos_0 --slice 0:100,0:100 -o sliced.pkl
        biopb tensor get my-source/pos_0 -S 0:512 -s grpc://myhost:9000 -o data.pkl
    """
    start_time = time.time()
    client = _create_flight_client(server, cache_bytes)
    try:
        source_id, tensor_id = _parse_array_id(array_id)
        selection = _parse_slice_hint(slice_hint)

        stderr_console.print(
            f"[green]Downloading tensor[/green] {array_id}"
            + (f" (region: {slice_hint})" if slice_hint else "")
        )

        arr = client.get_tensor(source_id, tensor_id, slice_hint=selection)

        # Compute the array (lazy dask array -> actual numpy array)
        result = arr.compute()

        if output == "-":
            # Write to stdout in pickle format
            pickle.dump(result, sys.stdout.buffer)
            stderr_console.print(f"[green]Tensor written to stdout[/green] ({result.nbytes} bytes)")
        else:
            # Write to file
            with open(output, "wb") as f:
                pickle.dump(result, f)
            stderr_console.print(f"[green]Tensor saved to:[/green] {output} ({result.nbytes} bytes)")

        _log_timing(start_time)

    except typer.Exit:
        raise
    except Exception as exc:
        stderr_console.print(f"[red]Failed to download tensor:[/red] {exc}")
        raise typer.Exit(1)
    finally:
        client.close()


@app.command()
def stats(
    array_id: str = typer.Argument(
        ...,
        help="Array identifier: source_id/tensor_id (tensor_id optional for single-tensor sources)",
    ),
    server: str = typer.Option(
        "grpc://localhost:8815",
        "--server",
        "-s",
        help="TensorFlight server URI",
    ),
    slice_hint: Optional[str] = typer.Option(
        None,
        "--slice",
        "-S",
        help="Slice specification, e.g. '0:100,0:200'",
    ),
    cache_bytes: int = typer.Option(
        100_000_000,
        "--cache-bytes",
        help="Maximum bytes for client-side chunk cache",
    ),
):
    """Compute statistics (min, max, mean) for a tensor.

    The slice option restricts the region analyzed. If not specified,
    the entire tensor is analyzed.

    Example:
        biopb tensor stats my-source
        biopb tensor stats my-source/pos_0
        biopb tensor stats my-source/pos_0 --slice 0:100,0:100
        biopb tensor stats my-source/pos_0 -S 0:512 -s grpc://myhost:9000
    """
    start_time = time.time()
    client = _create_flight_client(server, cache_bytes)
    try:
        source_id, tensor_id = _parse_array_id(array_id)
        selection = _parse_slice_hint(slice_hint)

        stderr_console.print(
            f"[green]Computing statistics for[/green] {array_id}"
            + (f" (region: {slice_hint})" if slice_hint else "")
        )

        arr = client.get_tensor(source_id, tensor_id, slice_hint=selection)

        # Compute all statistics in a single graph execution
        min_val, max_val, mean_val = dask.compute(arr.min(), arr.max(), arr.mean())
        stats_dict = {
            "shape": str(list(arr.shape)),
            "dtype": str(arr.dtype),
            "min": float(min_val),
            "max": float(max_val),
            "mean": float(mean_val),
            "count": int(arr.size),
        }

        stats_table = Table(title="Tensor Statistics", show_header=False)
        for key, value in stats_dict.items():
            if key in ("min", "max", "mean"):
                stats_table.add_row(key, f"{value:.6g}")
            else:
                stats_table.add_row(key, str(value))

        console.print(stats_table)
        _log_timing(start_time)

    except typer.Exit:
        raise
    except Exception as exc:
        stderr_console.print(f"[red]Failed to compute statistics:[/red] {exc}")
        raise typer.Exit(1)
    finally:
        client.close()


if __name__ == "__main__":
    app()
