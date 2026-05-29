# biopb-mcp

[![License MIT](https://img.shields.io/pypi/l/biopb-mcp.svg?color=green)](https://github.com/biopb/biopb-mcp/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/biopb-mcp.svg?color=green)](https://pypi.org/project/biopb-mcp)
[![Python Version](https://img.shields.io/pypi/pyversions/biopb-mcp.svg?color=green)](https://python.org)
[![tests](https://github.com/biopb/biopb-mcp/workflows/tests/badge.svg)](https://github.com/jiyuuchc/biopb-mcp/actions)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/biopb-mcp)](https://napari-hub.org/plugins/biopb-mcp)


**biopb-mcp** is the MCP component of the **[BioPB](https://github.com/jiyuuchc/biopb)** project


## Installation - recommended

```
curl -fsSL https://biopb.org/install.sh | bash
```

## Alternative methods

These methods only install the mcp component without the rest of **[biopb](https://github.com/jiyuuchc/biopb)**  system.

### Bundled Installers

Pre-built standalone installers are available from the [Releases](https://github.com/jiyuuchc/biopb-mcp/releases) page. These include napari with the biopb-mcp plugin pre-installed—no Python setup required.

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

### From PyPI

For users with existing virtual environement setup for napari: install `biopb-mcp` via [pip]:

    pip install biopb-mcp

### Development Version

To install latest development version:

    pip install git+https://github.com/biopb/biopb-mcp.git


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
