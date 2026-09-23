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

## biopb-tensor-server

An imaging data server for sharing large collections of imaging datasets, distributed and interfacing with both human users (GUI) and LLMs (API). [Read More...](biopb-tensor-server/README.md)

## biopb-image-runtime

Deploy distributed image processing algorithms (e.g., large deep-learning models) on the network as services. [Read More...](biopb-image-runtime/README.md)

## biopb-mcp

An LLM interface for the full biopb platform, so an AI agent can collaboate with a user on image analysis tasks. Data analysis is python code (ipykernel) with two input streams: (1) Jupyter notebook (human), or (2) chat -> LLM -> agent code. [Read More...](biopb-mcp/README.md)

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
