# BioPB - AI-assisted bio-image analysis

[![License MIT](https://img.shields.io/pypi/l/biopb.svg?color=green)](https://github.com/biopb/biopb/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/biopb.svg?color=green)](https://pypi.org/project/biopb)
[![Sonatype Central](https://maven-badges.sml.io/sonatype-central/io.github.jiyuuchc/biopb/badge.svg)](https://maven-badges.sml.io/sonatype-central/io.github.jiyuuchc/biopb/)
[![Python Tests](https://github.com/biopb/biopb/actions/workflows/python-ci.yaml/badge.svg)](https://github.com/biopb/biopb/actions)
[![Java Tests](https://github.com/biopb/biopb/actions/workflows/java-ci.yaml/badge.svg)](https://github.com/biopb/biopb/actions)

The repo provides the core harness framework of the biopb project. The goal is to coordinate a LLM agent with a user in an interactive session to perform complex image ananlysis tasks relavent to scientific researches. 

## Quick Start
```sh
curl -fsSL https://biopb.org/install.sh | bash
```

## biopb-mcp

An MCP server + napari plugin that hands an AI agent a live, shared napari session wired to the data and algorithm servers, so analysis is driven in plain Python instead of fixed GUI buttons. [Read More...](biopb-mcp/README.md)

  - **Shared canvas**: napari viewer is accessible and mutable by both the agent and the user
  - **Persistent kernel**: agent code runs in a real ipython kernel with namespace persistence and full observability
  - **Perceive → act → verify**: the agent runs code, then screenshots/inspects to confirm the result

## biopb-tensor

A blazing-fast imaging data server for sharing your lab's petabyte-scale datasets to LLM agents and human team members alike. [Read More...](biopb-tensor-server/README.md)

  - **Uniform Representation**: all source data mapped to a _multi-resolution_ and _lazy-read_ array for client access
  - **Multi-language**: dask array for Python and ImgLib2.CellImg for Java
  - **Thread-safe & Serializable**: compatible with dask.distribute for distributed computing on larger-than-memory dataset
  - **Metadata Server**: full DuckDB SQL support to query your embedded metadata
  - **On-the-fly Build**: keep your data in original format (.zvi, ndtiff etc). No staging or on-boarding process needed
  - **Built-in Viewer**: browse all your data with any browser (e.g., on an ipad)


## biopb-image-runtime

Deploy complex image processing algorithms (e.g., large deep-learning models) on the network as services. [Read More...](biopb-image-runtime/README.md)

## SDK

Schema, utilities and cli for building your own workflow. Explore your data in jupyter notebook etc.

#### Python

```sh
pip install biopb[tensor]
```

#### Java

```xml
<dependency>
  <groupId>io.github.jiyuuchc</groupId>
  <artifactId>biopb</artifactId>
  <version>CURRENT_VERSION</version>
</dependency>
```

## Related Projects in BioPB

### biopb-servers

Specific implementations of biopb-image-runtimes. [Read More...](https://github.com/biopb/biopb-server)

## Contributing

Contributions are very welcome. Read the [developement document](development.md) first to understand the overall design architecture.

## License

Distributed under the terms of the [MIT](https://github.com/biopb/biopb/raw/main/LICENSE) license,

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

[file an issue]: https://github.com/biopb/biopb/issues


