"""Client-side diagnostic CLI for querying TensorFlight servers.

Commands:
    query        List sources and tensors from a running server
    metadata     Inspect source metadata and tensor descriptors
    get          Download a tensor to file or stdout (pickle, zarr, or protobuf format)
    stats        Compute statistics (min, max, mean) for a tensor
    cache-stats  Show the server's cache hit/miss diagnostics

Every command dials the *same* plane through the one resolver in
``biopb._data_plane`` (biopb/biopb#615): ``--server`` -> ``BIOPB_TENSOR_URL`` ->
the control plane's published endpoint -> the default. ``--server`` stays because
a plane launched directly on a custom port is recorded nowhere and so cannot be
discovered; everything else is asked for rather than reconstructed.
"""

import json
import pickle
import sys
import time
from pathlib import Path
from typing import Literal, Optional, Tuple

import dask
import typer
from rich.console import Console
from rich.table import Table

from biopb import _data_plane
from biopb.tensor.client import TensorFlightClient

app = typer.Typer(
    name="tensor",
    help="Query a TensorFlight data plane (sources, tensors, stats, cache).",
)
# Main output to stdout; stderr console for logging/timing only
console = Console()
stderr_console = Console(stderr=True)


# The endpoint override, shared by every command so the five cannot drift into
# five defaults again. ``None`` means "resolve it" -- the default is not a
# constant any more, so spelling one here would reintroduce the reconstruction
# #615 removed.
_OPT_SERVER = typer.Option(
    None,
    "--server",
    "-s",
    help="TensorFlight server URI. Default: $BIOPB_TENSOR_URL, else the endpoint "
    "the control plane publishes, else the local default.",
)
_OPT_TOKEN = typer.Option(
    None,
    "--token",
    "-t",
    help="Bearer token. Default: $BIOPB_TENSOR_TOKEN, else the credential file "
    "the control plane writes.",
)
_OPT_CACHE_BYTES = typer.Option(
    100_000_000, "--cache-bytes", help="Maximum bytes for the client-side chunk cache"
)
_OPT_SLICE = typer.Option(
    None, "--slice", "-S", help="Slice specification, e.g. '0:100,0:200'"
)


def _log_timing(start_time: float) -> None:
    """Print elapsed time since start_time to stderr."""
    elapsed = time.time() - start_time
    stderr_console.print(f"[dim]Completed in {elapsed:.2f}s[/dim]")


def _dial_error(exc: Exception, endpoint: _data_plane.Endpoint) -> str:
    """Why a dial failed, classified by exception *type*.

    Every failure used to render as "server unreachable or cache not initialized"
    (biopb/biopb#615 fault 3), which is false for the two most common ones: the
    server answered and refused the dial. A reader chasing a dead process for what
    is a missing token loses the afternoon.

    Type, not message substring: an unreadable cert stringifies as ``Permission
    denied``, which any auth-marker scan claims and misfiles as "needs a token"
    (biopb/biopb#610). The endpoint's origin is named too — "unreachable" means
    something different for an address the control published than for a guessed
    default.
    """
    import pyarrow.flight as flight

    where = f"{endpoint.url} ({endpoint.origin_note})"
    if isinstance(exc, _data_plane.LocalTrustError):
        return str(exc)
    if isinstance(
        exc, (flight.FlightUnauthenticatedError, flight.FlightUnauthorizedError)
    ):
        if endpoint.token:
            return f"The data plane at {where} rejected the token."
        return (
            f"The data plane at {where} requires an access token. Pass --token, set "
            f"${_data_plane.ENV_TOKEN}, or start it through `biopb control start` "
            "(which writes the credential file local clients read)."
        )
    if isinstance(exc, (flight.FlightUnavailableError, flight.FlightTimedOutError)):
        hint = ""
        if endpoint.origin == "default":
            hint = (
                " Nothing published an endpoint, so this is the default one — a "
                "plane on a different port needs --server."
            )
        return f"Cannot reach the data plane at {where}: {exc}.{hint}"
    return f"{type(exc).__name__} from the data plane at {where}: {exc}"


def _resolve_endpoint(
    server: Optional[str], token: Optional[str]
) -> _data_plane.Endpoint:
    """Resolve the endpoint every command dials, exiting 1 with the reason on failure."""
    try:
        return _data_plane.resolve(server, token)
    except _data_plane.LocalTrustError as exc:
        stderr_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)


def _connect(
    server: Optional[str], token: Optional[str], cache_bytes: int
) -> Tuple[TensorFlightClient, _data_plane.Endpoint]:
    """Resolve the endpoint and open a client to it, or exit 1 saying why not."""
    endpoint = _resolve_endpoint(server, token)
    try:
        # TensorFlightClient.__init__ normalizes the location (grpcs:// ->
        # grpc+tls://) itself, so no pre-normalization is needed here.
        client = TensorFlightClient(
            location=endpoint.url,
            cache_bytes=cache_bytes,
            token=endpoint.token,
            tls_ca_pem=endpoint.tls_ca_pem,
        )
    except Exception as exc:
        stderr_console.print(f"[red]{_dial_error(exc, endpoint)}[/red]")
        raise typer.Exit(1)
    return client, endpoint


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


def _infer_format(
    output: str, format: Optional[str]
) -> Literal["pickle", "zarr", "pb"]:
    """Infer output format from filename or explicit format option.

    Args:
        output: Output path or "-" for stdout
        format: Explicit format option (None to infer from filename)

    Returns:
        Format string: "pickle", "zarr", or "pb"
    """
    if format:
        fmt = format.lower()
        if fmt not in ("pickle", "zarr", "pb"):
            raise typer.BadParameter(
                f"Invalid format: {format}. Must be pickle, zarr, or pb."
            )
        return fmt

    if output == "-":
        return "pb"  # stdout default

    ext = Path(output).suffix.lower()
    if ext in (".zarr", ".zr"):
        return "zarr"
    if ext in (".pb", ".protobuf"):
        return "pb"
    if ext in (".pkl", ".pickle"):
        return "pickle"
    return "pb"  # default


@app.command(help="List the data sources and tensors a server is serving.")
def query(
    server: Optional[str] = _OPT_SERVER,
    token: Optional[str] = _OPT_TOKEN,
    cache_bytes: int = _OPT_CACHE_BYTES,
):
    """List all data sources and tensors from a running TensorFlight server.

    Example:
        biopb tensor query
        biopb tensor query -s grpc://myhost:9000 --token mytoken123
        BIOPB_TENSOR_TOKEN=mytoken123 biopb tensor query
    """
    start_time = time.time()
    client, endpoint = _connect(server, token, cache_bytes)
    try:
        sources = client.list_sources()
        if not sources:
            stderr_console.print(f"[yellow]No sources found on {endpoint.url}[/yellow]")
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
            f"\n[green]Server:[/green] {endpoint.url}  "
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


@app.command(help="Inspect a source's metadata and its tensor descriptors.")
def metadata(
    source_id: str = typer.Argument(..., help="Source identifier to inspect"),
    server: Optional[str] = _OPT_SERVER,
    tensor: Optional[str] = typer.Option(
        None,
        "--tensor",
        help="Specific tensor ID to inspect (optional)",
    ),
    token: Optional[str] = _OPT_TOKEN,
    cache_bytes: int = _OPT_CACHE_BYTES,
):
    """Inspect source metadata and tensor descriptors.

    Example:
        biopb tensor metadata my-source
        biopb tensor metadata my-source --tensor pos_0
        biopb tensor metadata my-source -s grpc://myhost:9000 --token mytoken123
    """
    start_time = time.time()
    client, _ = _connect(server, token, cache_bytes)
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
        console.print("\n[bold green]Source Metadata:[/bold green]")
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


@app.command(help="Download a tensor to a file or stdout (pickle, zarr, or protobuf).")
def get(
    array_id: str = typer.Argument(
        ...,
        help="Array identifier: source_id/tensor_id (tensor_id optional for single-tensor sources)",
    ),
    output: str = typer.Option(
        "-",
        "--output",
        "-o",
        help="Output path. Use '-' for stdout. Format inferred from extension: .pkl (pickle), .zarr (zarr), .pb (protobuf)",
    ),
    format: Optional[str] = typer.Option(
        None,
        "--format",
        "-f",
        help="Output format: pickle (lazy dask), zarr (realized), pb (protobuf). Inferred from filename if not set.",
    ),
    server: Optional[str] = _OPT_SERVER,
    slice_hint: Optional[str] = _OPT_SLICE,
    token: Optional[str] = _OPT_TOKEN,
    cache_bytes: int = _OPT_CACHE_BYTES,
):
    """Download a tensor to file or stdout.

    Supports multiple output formats:
    - pickle: Lazy dask array (task graph, no data transfer)
    - zarr: Realized numpy array written to zarr format
    - pb: SerializedTensor protobuf (lazy, contains chunk tickets)

    Format is inferred from output filename extension, or can be set explicitly with --format.

    Example:
        biopb tensor get my-source -o output.pkl        # pickle (lazy)
        biopb tensor get my-source -o output.zarr       # zarr (realized)
        biopb tensor get my-source -o output.pb         # protobuf (lazy)
        biopb tensor get my-source -o -                 # stdout (pickle)
        biopb tensor get my-source -f zarr -o data      # explicit format
        biopb tensor get my-source --slice 0:100 -o slice.pkl
        biopb tensor get my-source --token mytoken123 -o output.pkl
    """
    start_time = time.time()
    client, _ = _connect(server, token, cache_bytes)
    try:
        selection = _parse_slice_hint(slice_hint)
        fmt = _infer_format(output, format)

        stderr_console.print(
            f"[green]Fetching tensor[/green] {array_id} (format: {fmt})"
            + (f" (region: {slice_hint})" if slice_hint else "")
        )

        if fmt == "pb":
            # Protobuf format: lazy SerializedTensor
            serialized = client.get_tensor_pb(array_id, slice_hint=selection)
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

        elif fmt == "zarr":
            # Zarr format: realized array. Import lazily so that a missing or
            # broken zarr/numcodecs install only affects this output format
            # rather than the whole CLI.
            try:
                import zarr
            except ImportError as exc:
                raise typer.BadParameter(
                    f"zarr output requires the 'zarr' package (install biopb[tensor]): {exc}"
                )

            arr = client.get_tensor(array_id, slice_hint=selection)
            result = arr.compute()

            if output == "-":
                raise typer.BadParameter("zarr format requires file output, not stdout")

            zarr.save_array(output, result)
            stderr_console.print(
                f"[green]Zarr saved to:[/green] {output} ({result.nbytes} bytes)"
            )

        else:
            # Pickle format: lazy dask array (no compute)
            arr = client.get_tensor(array_id, slice_hint=selection)

            if output == "-":
                pickle.dump(arr, sys.stdout.buffer)
                stderr_console.print(
                    f"[green]Dask array written to stdout[/green] (shape={list(arr.shape)})"
                )
            else:
                with open(output, "wb") as f:
                    pickle.dump(arr, f)
                stderr_console.print(
                    f"[green]Dask array saved to:[/green] {output} (shape={list(arr.shape)})"
                )

        _log_timing(start_time)

    except typer.Exit:
        raise
    except Exception as exc:
        stderr_console.print(f"[red]Failed to fetch tensor:[/red] {exc}")
        raise typer.Exit(1)
    finally:
        client.close()


@app.command(help="Compute a tensor's min, max and mean (optionally over a slice).")
def stats(
    array_id: str = typer.Argument(
        ...,
        help="Array identifier: source_id/tensor_id (tensor_id optional for single-tensor sources)",
    ),
    server: Optional[str] = _OPT_SERVER,
    slice_hint: Optional[str] = _OPT_SLICE,
    token: Optional[str] = _OPT_TOKEN,
    cache_bytes: int = _OPT_CACHE_BYTES,
):
    """Compute statistics (min, max, mean) for a tensor.

    The slice option restricts the region analyzed. If not specified,
    the entire tensor is analyzed.

    Example:
        biopb tensor stats my-source
        biopb tensor stats my-source/pos_0
        biopb tensor stats my-source/pos_0 --slice 0:100,0:100
        biopb tensor stats my-source/pos_0 -S 0:512 -s grpc://myhost:9000 --token mytoken123
    """
    start_time = time.time()
    client, _ = _connect(server, token, cache_bytes)
    try:
        selection = _parse_slice_hint(slice_hint)

        stderr_console.print(
            f"[green]Computing statistics for[/green] {array_id}"
            + (f" (region: {slice_hint})" if slice_hint else "")
        )

        arr = client.get_tensor(array_id, slice_hint=selection)

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


# --- cache-stats ---------------------------------------------------------- #
#
# Moved here from `biopb server cache-stats` when that group was retired: it is a
# question you ask a *server over Flight*, which is what every other command in
# this module is, and it needs the same client the others build. Under `server`
# it had its own endpoint resolver, which is how it came to dial a different
# address than `biopb tensor query` against the same plane (biopb/biopb#615).


def _fmt_mb(n_bytes: int) -> str:
    """Format a byte count as MB."""
    return f"{n_bytes / (1024 * 1024):.1f} MB"


def _hit_rate(hits: int, misses: int) -> str:
    """Hit rate as a percentage string (guards divide-by-zero)."""
    total = hits + misses
    return f"{(hits / total * 100):.1f}%" if total else "n/a"


def _render_cache_stats(stats: dict) -> None:
    """Render a CacheStats dict (from TensorFlightClient.cache_stats) as tables."""
    g = stats.get
    table = Table(title="Cache Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green", justify="right")

    hits, misses = g("hits", 0), g("misses", 0)
    table.add_row("Hits", str(hits))
    table.add_row("Misses", str(misses))
    table.add_row("Hit rate", _hit_rate(hits, misses))
    table.add_row("Evictions", str(g("evictions", 0)))
    table.add_row("Pending waits", str(g("pending_waits", 0)))
    table.add_row("Oversized skips", str(g("oversized_skips", 0)))
    table.add_row("Ref-held evictions skipped", str(g("ref_held_evictions_skipped", 0)))
    table.add_row("Entries", str(g("total_entries", 0)))
    table.add_row("Size", _fmt_mb(g("total_bytes", 0)))
    if g("max_entries", 0):
        table.add_row("Max entries", str(g("max_entries")))
    if g("max_bytes", 0):
        table.add_row("Max size", _fmt_mb(g("max_bytes")))
    console.print(table)

    pool_stats = stats.get("pool_stats") or {}
    if pool_stats:
        ptable = Table(title="Per-pool Statistics")
        for col in ("Pool", "Hits", "Misses", "Hit rate", "Segments", "Size"):
            ptable.add_column(
                col,
                style="cyan" if col == "Pool" else "green",
                justify="left" if col == "Pool" else "right",
            )
        for name, p in sorted(pool_stats.items()):
            ptable.add_row(
                name,
                str(p.get("hits", 0)),
                str(p.get("misses", 0)),
                _hit_rate(p.get("hits", 0), p.get("misses", 0)),
                str(p.get("segments", 0)),
                _fmt_mb(p.get("bytes", 0)),
            )
        console.print(ptable)


@app.command("cache-stats", help="Show the server's chunk-cache hit/miss diagnostics.")
def cache_stats(
    server: Optional[str] = _OPT_SERVER,
    token: Optional[str] = _OPT_TOKEN,
    json_output: bool = typer.Option(
        False, "--json", help="Emit machine-readable JSON instead of a table"
    ),
):
    """Show cache hit/miss diagnostics from the running server.

    Liveness is the Flight query itself -- an unreachable server yields no stats,
    so there is no separate PID-file gate; the control plane owns the data-plane
    process and writes no ``tensor-server.pid``.

    ``cache_bytes=0``: this opens a throwaway client to ask a question about the
    *server's* cache, so it must not allocate a client-side one of its own.
    """
    client, endpoint = _connect(server, token, cache_bytes=0)
    try:
        stats = client.cache_stats()
    except Exception as exc:  # noqa: BLE001 - rendered by type, not swallowed
        stderr_console.print(f"[red]{_dial_error(exc, endpoint)}[/red]")
        raise typer.Exit(1)
    finally:
        client.close()

    if not stats:
        # The server answered with no stats at all -- a cache that was never
        # initialized. Distinct from every failure above, which is the whole
        # point: "unreachable or cache not initialized" used to cover both.
        stderr_console.print(
            f"[yellow]The data plane at {endpoint.url} reported no cache "
            "statistics (no cache initialized).[/yellow]"
        )
        raise typer.Exit(1)

    if json_output:
        print(json.dumps(stats))
        raise typer.Exit(0)

    _render_cache_stats(stats)


if __name__ == "__main__":
    app()
