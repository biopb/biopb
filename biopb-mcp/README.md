# biopb-mcp

[![License MIT](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/biopb/biopb/raw/main/biopb-mcp/LICENSE)
[![MCP CI](https://github.com/biopb/biopb/actions/workflows/mcp-ci.yaml/badge.svg)](https://github.com/biopb/biopb/actions/workflows/mcp-ci.yaml)


**biopb-mcp** is the MCP component of the **[BioPB](https://github.com/biopb/biopb)** project


## Installation - recommended

```
curl -fsSL https://biopb.org/install.sh | bash
```

## Installation - alternative methods

These methods only install the mcp component without the rest of **[BioPB](https://github.com/biopb/biopb)**  system.

> **biopb-mcp is no longer published to PyPI.** It ships as part of the
> `release-v*` wheel triple that the recommended installer above
> `file://`-installs; a standalone `pip install biopb-mcp` could never pull the
> tensor server or the full-stack dependency groups it needs anyway. To install
> just the mcp component from source, use the development method below.

### Development / from source

To install the latest source version:

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
