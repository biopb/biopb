# biopb-mcp

[![License MIT](https://img.shields.io/pypi/l/biopb-mcp.svg?color=green&cacheSeconds=3600)](https://github.com/biopb/biopb/raw/main/biopb-mcp/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/biopb-mcp.svg?color=green&cacheSeconds=3600)](https://pypi.org/project/biopb-mcp)
[![Python Version](https://img.shields.io/pypi/pyversions/biopb-mcp.svg?color=green&cacheSeconds=3600)](https://python.org)
[![MCP CI/CD](https://github.com/biopb/biopb/actions/workflows/mcp-ci.yaml/badge.svg)](https://github.com/biopb/biopb/actions/workflows/mcp-ci.yaml)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/biopb-mcp&cacheSeconds=3600)](https://napari-hub.org/plugins/biopb-mcp)


**biopb-mcp** is the MCP component of the **[BioPB](https://github.com/biopb/biopb)** project


## Installation - recommended

```
curl -fsSL https://biopb.org/install.sh | bash
```

## Installation - alternative methods

These methods only install the mcp component without the rest of **[BioPB](https://github.com/biopb/biopb)**  system.

### From PyPI

```
pip install biopb-mcp        # or: uv pip install biopb-mcp
```

> Standalone PyInstaller app bundles (the per-platform `.zip`/`.tar.gz`
> downloads) are temporarily disabled — installation is uv/pip-based for now.
> Use the recommended one-liner above, or the methods here.

### Development Version

To install latest development version:

    pip install "git+https://github.com/biopb/biopb.git#subdirectory=biopb-mcp"


## Troubleshooting

For URL resolution, the auto-start fallback, and how to resolve startup failures (e.g.
a port already in use on a shared node), see [docs/troubleshooting.md](docs/troubleshooting.md).

## Contributing

Contributions are very welcome. Read [development.md](development.md) first to understand the project architecture.

## License

Distributed under the terms of the [MIT](https://github.com/biopb/biopb/raw/main/biopb-mcp/LICENSE) license,
"biopb-mcp" is free and open source software

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

[file an issue]: https://github.com/biopb/biopb/issues

[napari]: https://github.com/napari/napari
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/
