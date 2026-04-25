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

from biopb_tensor_server.config import (
    load_config,
    resolve_all_sources,
    ServerConfig,
    CacheConfig,
)
from biopb_tensor_server.adapters.zarr import ZarrAdapter
from biopb_tensor_server.adapters.hdf5 import Hdf5Adapter
from biopb_tensor_server.adapters.tiff import OmeTiffAdapter, MultiFileOmeTiffAdapter
from biopb_tensor_server.adapters.ome_zarr import OmeZarrAdapter
from biopb_tensor_server.base import configure_compute_backend
from biopb_tensor_server.server import TensorFlightServer
from biopb_tensor_server.cache import CacheManager
from biopb_tensor_server.cache.memory_backend import MemoryCacheConfig
from biopb_tensor_server.cache.file_backend import ArrowFileBackend


app = typer.Typer(
    name="biopb-tensor",
    help="BioPB Tensor: Arrow Flight server for multi-dimensional arrays",
)
console = Console()


def _create_adapter(source):
    """Create a backend adapter from a source config.

    Raw data caching relies on OS page cache - no per-adapter LRU cache.
    """
    if source.type == "zarr":
        import zarr
        arr = zarr.open_array(str(source.path), mode='r')
        return ZarrAdapter(arr, source.array_id, source.dim_labels)

    elif source.type == "hdf5":
        import h5py
        f = h5py.File(str(source.path), 'r')
        dataset = f[source.dataset] if source.dataset else list(f.keys())[0]
        return Hdf5Adapter(dataset, source.array_id, source.dim_labels)

    elif source.type == "ome-tiff":
        import tifffile
        tiff = tifffile.TiffFile(str(source.path))
        return OmeTiffAdapter(tiff, source.array_id, source.dim_labels)

    elif source.type == "ome-tiff-multifile":
        return MultiFileOmeTiffAdapter(
            str(source.path),
            source.array_id,
            source.dim_labels,
        )

    elif source.type == "ome-zarr":
        import zarr
        import json
        zarr_path = str(source.path)
        store = zarr.DirectoryStore(zarr_path)

        try:
            with open(str(source.path / ".zattrs")) as f:
                zattrs = json.load(f)

            resolution_path = "0"
            if 'multiscales' in zattrs and zattrs['multiscales']:
                datasets = zattrs['multiscales'][0].get('datasets', [])
                if datasets:
                    resolution_path = datasets[0].get('path', '0')

            root = zarr.open_group(zarr_path, mode='r')
            if resolution_path in root:
                arr = root[resolution_path]
            else:
                arr = zarr.open_array(store, mode='r')
        except (json.JSONDecodeError, KeyError, FileNotFoundError):
            arr = zarr.open_array(zarr_path, mode='r')

        return OmeZarrAdapter(arr, source.array_id, source.dim_labels)

    elif source.type == "aics":
        from aicsimageio import AICSImage
        from biopb_tensor_server.adapters.aicsimageio import AicsImageIoAdapter

        img = AICSImage(str(source.path))
        if source.scene_index is not None:
            img.set_scene(source.scene_index)
        return AicsImageIoAdapter(
            img,
            source.scene_index or 0,
            source.array_id,
            source.dim_labels,
        )

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
    ),
):
    """Start the TensorFlight server.

    Example:
        biopb-tensor serve --config biopb-tensor.toml
        biopb-tensor serve -c config.toml --port 9000
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

    # Initialize cache manager for virtual chunks
    cache_config = server_config.cache
    if cache_config.backend == "memory":
        CacheManager.initialize(cache_config)
        console.print(
            "[green]Virtual chunk cache initialized:[/green] "
            f"backend=memory, "
            f"max_entries={cache_config.memory_max_entries}, "
            f"max_bytes={cache_config.memory_max_bytes // (1024*1024)}MB"
        )
        console.print("[green]Raw chunk cache: OS page cache[/green]")
    elif cache_config.backend == "file":
        manager = CacheManager.initialize(cache_config)
        console.print(
            "[green]Virtual chunk cache initialized:[/green] "
            f"backend=file, "
            f"cache_dir={cache_config.file_cache_dir}, "
            f"max_segment_mb={cache_config.file_max_segment_bytes // (1024*1024)}, "
            f"max_total_gb={cache_config.file_max_total_bytes // (1024*1024*1024)}"
        )
        # Check for recovery status
        if isinstance(manager.backend, ArrowFileBackend):
            recovery_status = manager.backend.get_recovery_status()
            if recovery_status:
                console.print(
                    "[yellow]Cache recovery completed:[/yellow] "
                    f"recovered={recovery_status.recovered_entries} entries "
                    f"({recovery_status.recovered_bytes // (1024*1024)}MB), "
                    f"lost={recovery_status.lost_entries} entries"
                )
                if recovery_status.errors:
                    for err in recovery_status.errors[:3]:
                        console.print(f"[red]  Error: {err}[/red]")
        console.print("[green]Raw chunk cache: OS page cache[/green]")
    else:
        console.print(f"[yellow]Warning: Unknown cache backend '{cache_config.backend}', using memory[/yellow]")
        CacheManager.initialize(CacheConfig())

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
            adapter = _create_adapter(source)
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
        biopb-tensor validate biopb-tensor.toml
    """
    try:
        server_config = load_config(config)
        sources = resolve_all_sources(server_config)

        console.print("[green]✓ Config valid[/green]")
        console.print(f"  Server: {server_config.host}:{server_config.port}")
        console.print(
            "  Compute: "
            f"backend={server_config.compute_backend}, "
            f"gpu_min_input_mb={server_config.gpu_min_input_mb}, "
            f"gpu_min_linear_input_mb={server_config.gpu_min_linear_input_mb}, "
            f"gpu_memory_safety_factor={server_config.gpu_memory_safety_factor}, "
            f"gpu_min_merged_chunks={server_config.gpu_min_merged_chunks}"
        )
        console.print(
            "  Cache: "
            f"backend={server_config.cache.backend}, "
        )
        if server_config.cache.backend == "memory":
            console.print(
                f"    max_entries={server_config.cache.memory_max_entries}, "
                f"max_bytes={server_config.cache.memory_max_bytes // (1024*1024)}MB"
            )
        elif server_config.cache.backend == "file":
            console.print(
                f"    cache_dir={server_config.cache.file_cache_dir}, "
                f"max_segment_mb={server_config.cache.file_max_segment_bytes // (1024*1024)}, "
                f"max_total_gb={server_config.cache.file_max_total_bytes // (1024*1024*1024)}"
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
        biopb-tensor list biopb-tensor.toml
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
    try:
        from biopb import __version__
        console.print(f"TensorFlight server (using biopb {__version__})")
    except ImportError:
        console.print("TensorFlight server")


if __name__ == "__main__":
    app()