"""CLI for TensorFlight server.

Commands:
    serve      Start the Flight server
    validate   Validate a config file
    list       List tensors in a config file
"""

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from biopb.tensor.config import (
    load_config,
    resolve_all_sources,
    ServerConfig,
)
from biopb.tensor.adapter import ZarrAdapter, Hdf5Adapter, OmeTiffAdapter, MultiFileOmeTiffAdapter, OmeZarrAdapter, configure_compute_backend
from biopb.tensor.server import TensorFlightServer


app = typer.Typer(
    name="tensorflight",
    help="TensorFlight: Arrow Flight server for multi-dimensional arrays",
)
console = Console()


def _create_adapter(source, cache_size: int = 256):
    """Create a backend adapter from a source config."""
    if source.type == "zarr":
        import zarr
        arr = zarr.open_array(str(source.path), mode='r')
        return ZarrAdapter(arr, source.array_id, source.dim_labels, cache_size=cache_size)

    elif source.type == "hdf5":
        import h5py
        f = h5py.File(str(source.path), 'r')
        dataset = f[source.dataset] if source.dataset else list(f.keys())[0]
        return Hdf5Adapter(dataset, source.array_id, source.dim_labels, cache_size=cache_size)

    elif source.type == "ome-tiff":
        import tifffile
        tiff = tifffile.TiffFile(str(source.path))
        return OmeTiffAdapter(tiff, source.array_id, source.dim_labels, cache_size=cache_size)

    elif source.type == "ome-tiff-multifile":
        return MultiFileOmeTiffAdapter(
            str(source.path),
            source.array_id,
            source.dim_labels,
            cache_size=cache_size
        )

    elif source.type == "ome-zarr":
        import zarr
        import json
        # Open OME-Zarr and select resolution level 0
        zarr_path = str(source.path)
        store = zarr.DirectoryStore(zarr_path)

        # Read .zattrs to get multiscales info
        try:
            with open(str(source.path / ".zattrs")) as f:
                zattrs = json.load(f)

            # Get the first resolution level path
            resolution_path = "0"  # Default to first level
            if 'multiscales' in zattrs and zattrs['multiscales']:
                datasets = zattrs['multiscales'][0].get('datasets', [])
                if datasets:
                    resolution_path = datasets[0].get('path', '0')

            # Open the array at the resolution level
            root = zarr.open_group(zarr_path, mode='r')
            if resolution_path in root:
                arr = root[resolution_path]
            else:
                arr = zarr.open_array(store, mode='r')
        except (json.JSONDecodeError, KeyError, FileNotFoundError):
            # Fall back to regular zarr array
            arr = zarr.open_array(zarr_path, mode='r')

        return OmeZarrAdapter(arr, source.array_id, source.dim_labels, cache_size=cache_size)

    else:
        raise ValueError(f"Unknown source type: {source.type}")


@app.command()
def serve(
    config: Path = typer.Option(
        ...,
        "--config", "-c",
        exists=True,
        help="Path to TOML config file",
    ),
    host: Optional[str] = typer.Option(
        None,
        "--host", "-h",
        help="Server host (overrides config)",
    ),
    port: Optional[int] = typer.Option(
        None,
        "--port", "-p",
        help="Server port (overrides config)",
    ),
    cache_size: int = typer.Option(
        256,
        "--cache-size",
        help="LRU cache size per adapter (number of chunks/tiles)",
    ),
    compute_backend: Optional[str] = typer.Option(
        None,
        "--compute-backend",
        help="Compute backend policy: auto, cpu, or gpu",
    ),
    gpu_min_input_mb: Optional[float] = typer.Option(
        None,
        "--gpu-min-input-mb",
        help="Minimum input size in MB before GPU is considered for area-like methods",
    ),
    gpu_min_linear_input_mb: Optional[float] = typer.Option(
        None,
        "--gpu-min-linear-input-mb",
        help="Minimum input size in MB before GPU is considered for linear interpolation",
    ),
    gpu_memory_safety_factor: Optional[int] = typer.Option(
        None,
        "--gpu-memory-safety-factor",
        help="Required free-GPU-memory multiplier over estimated working set",
    ),
    gpu_min_merged_chunks: Optional[int] = typer.Option(
        None,
        "--gpu-min-merged-chunks",
        help="Minimum merged source chunk count before GPU is preferred",
    ):
    """Start the TensorFlight server.

    Example:
        tensorflight serve --config tensorflight.toml
        tensorflight serve -c config.toml --port 9000
    """
    # Load config
    server_config = load_config(config)

    # Apply overrides
    host = host or server_config.host
    port = port or server_config.port

    configure_compute_backend(
        force_backend=compute_backend or server_config.compute_backend,
        gpu_min_input_bytes=int((gpu_min_input_mb if gpu_min_input_mb is not None else server_config.gpu_min_input_mb) * 1024 * 1024),
        gpu_min_linear_input_bytes=int((gpu_min_linear_input_mb if gpu_min_linear_input_mb is not None else server_config.gpu_min_linear_input_mb) * 1024 * 1024),
        gpu_memory_safety_factor=gpu_memory_safety_factor or server_config.gpu_memory_safety_factor,
        gpu_min_merged_chunks=gpu_min_merged_chunks or server_config.gpu_min_merged_chunks,
    )

    # Resolve sources (expand directories)
    sources = resolve_all_sources(server_config)

    if not sources:
        console.print("[yellow]Warning: No tensor sources configured[/yellow]")
        raise typer.Exit(1)

    console.print(f"[green]Loading {len(sources)} tensor source(s)...[/green]")
    console.print(
        "[green]Compute backend policy:[/green] "
        f"backend={compute_backend or server_config.compute_backend}, "
        f"gpu_min_input_mb={gpu_min_input_mb if gpu_min_input_mb is not None else server_config.gpu_min_input_mb}, "
        f"gpu_min_linear_input_mb={gpu_min_linear_input_mb if gpu_min_linear_input_mb is not None else server_config.gpu_min_linear_input_mb}, "
        f"gpu_memory_safety_factor={gpu_memory_safety_factor or server_config.gpu_memory_safety_factor}, "
        f"gpu_min_merged_chunks={gpu_min_merged_chunks or server_config.gpu_min_merged_chunks}"
    )

    # Create adapters
    adapters = {}
    for source in sources:
        try:
            adapter = _create_adapter(source, cache_size=cache_size)
            adapters[source.array_id] = adapter
            desc = adapter.get_tensor_descriptor()
            console.print(f"  [blue]{source.array_id}[/blue]: shape={list(desc.shape)}, dtype={desc.dtype}")
        except Exception as e:
            console.print(f"  [red]Failed to load {source.array_id}: {e}[/red]")

    if not adapters:
        console.print("[red]No tensors loaded successfully[/red]")
        raise typer.Exit(1)

    # Create and start server
    location = f"grpc://{host}:{port}"
    server = TensorFlightServer(location)

    for array_id, adapter in adapters.items():
        server.register_tensor(array_id, adapter)

    console.print(f"\n[green]Starting TensorFlight server at {location}[/green]")
    console.print("Press Ctrl+C to stop\n")

    try:
        server.serve()
    except KeyboardInterrupt:
        console.print("\n[yellow]Shutting down...[/yellow]")
        server.shutdown()


@app.command()
def validate(
    config: Path = typer.Argument(
        ...,
        exists=True,
        help="Path to TOML config file",
    ),
):
    """Validate a config file.

    Example:
        tensorflight validate tensorflight.toml
    """
    try:
        server_config = load_config(config)
        sources = resolve_all_sources(server_config)

        console.print(f"[green]✓ Config valid[/green]")
        console.print(f"  Server: {server_config.host}:{server_config.port}")
        console.print(
            "  Compute: "
            f"backend={server_config.compute_backend}, "
            f"gpu_min_input_mb={server_config.gpu_min_input_mb}, "
            f"gpu_min_linear_input_mb={server_config.gpu_min_linear_input_mb}, "
            f"gpu_memory_safety_factor={server_config.gpu_memory_safety_factor}, "
            f"gpu_min_merged_chunks={server_config.gpu_min_merged_chunks}"
        )
        console.print(f"  Sources: {len(sources)} tensor(s)")

        for source in sources:
            console.print(f"    - {source.array_id} ({source.type}: {source.path})")

    except Exception as e:
        console.print(f"[red]✗ Config invalid: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def list_tensors(
    config: Path = typer.Argument(
        ...,
        exists=True,
        help="Path to TOML config file",
    ),
):
    """List tensors defined in a config file.

    Example:
        tensorflight list tensorflight.toml
    """
    try:
        server_config = load_config(config)
        sources = resolve_all_sources(server_config)

        table = Table(title="Tensor Sources")
        table.add_column("Array ID", style="cyan")
        table.add_column("Type", style="green")
        table.add_column("Path")
        table.add_column("Dim Labels")

        for source in sources:
            labels = ", ".join(source.dim_labels) if source.dim_labels else "-"
            table.add_row(
                source.array_id,
                source.type,
                str(source.path),
                labels,
            )

        console.print(table)

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def version():
    """Show version information."""
    from biopb import __version__
    console.print(f"TensorFlight version: {__version__}")


if __name__ == "__main__":
    app()