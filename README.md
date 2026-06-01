# biopb-mcp

[![License MIT](https://img.shields.io/pypi/l/biopb-mcp.svg?color=green&cacheSeconds=3600)](https://github.com/biopb/biopb-mcp/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/biopb-mcp.svg?color=green&cacheSeconds=3600)](https://pypi.org/project/biopb-mcp)
[![Python Version](https://img.shields.io/pypi/pyversions/biopb-mcp.svg?color=green&cacheSeconds=3600)](https://python.org)
[![tests](https://github.com/biopb/biopb-mcp/workflows/tests/badge.svg)](https://github.com/biopb/biopb-mcp/actions)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/biopb-mcp&cacheSeconds=3600)](https://napari-hub.org/plugins/biopb-mcp)


**biopb-mcp** is the MCP component of the **[BioPB](https://github.com/biopb/biopb)** project


## Installation - recommended

```
curl -fsSL https://biopb.org/install.sh | bash
```

## Installation - alternative methods

These methods only install the mcp component without the rest of **[BioPB](https://github.com/biopb/biopb)**  system.

### Bundled Installers

Pre-built standalone installers are available from the [Releases](https://github.com/biopb/biopb-mcp/releases) page. These include napari with the biopb-mcp plugin pre-installed—no Python setup required.

| Platform | Download |
|----------|----------|
| Windows | `biopb-mcp-windows.zip` |
| macOS (Intel) | `biopb-mcp-macos-intel.tar.gz` |
| macOS (Apple Silicon) | `biopb-mcp-macos-arm.tar.gz` |
| Linux | `biopb-mcp-linux.tar.gz` |

**Usage:**
- **Windows**: Extract the zip and run `biopb-mcp.exe`
- **macOS**: Extract and open `biopb-mcp.app` (right-click → "Open" to bypass Gatekeeper for unsigned apps)
- **Linux**: Extract and run `./biopb-mcp`

### Development Version

To install latest development version:

    pip install git+https://github.com/biopb/biopb-mcp.git


## Troubleshooting

For URL resolution, the auto-start fallback, and how to resolve startup failures (e.g.
a port already in use on a shared node), see [docs/troubleshooting.md](docs/troubleshooting.md).

## Contributing

Contributions are very welcome. Tests can be run with [tox], please ensure
the coverage at least stays the same before you submit a pull request.

## License

Distributed under the terms of the [MIT](https://github.com/biopb/biopb-mcp/raw/main/LICENSE) license,
"biopb-mcp" is free and open source software

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

[file an issue]: https://github.com/biopb/biopb-mcp/issues

[napari]: https://github.com/napari/napari
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/
