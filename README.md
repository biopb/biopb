# BioPB - AI-assisted bio-image analysis

[![License MIT](https://img.shields.io/pypi/l/biopb.svg?color=green)](https://github.com/biopb/biopb/raw/main/LICENSE)
[![Release](https://img.shields.io/github/v/release/biopb/biopb?label=release&color=green)](https://github.com/biopb/biopb/releases)
[![PyPI](https://img.shields.io/pypi/v/biopb.svg?color=green&label=SDK)](https://pypi.org/project/biopb)
[![Sonatype Central](https://maven-badges.sml.io/sonatype-central/io.github.jiyuuchc/biopb/badge.svg)](https://maven-badges.sml.io/sonatype-central/io.github.jiyuuchc/biopb/)
[![Python Tests](https://github.com/biopb/biopb/actions/workflows/python-ci.yaml/badge.svg)](https://github.com/biopb/biopb/actions/workflows/python-ci.yaml)
[![Java Tests](https://github.com/biopb/biopb/actions/workflows/java-ci.yaml/badge.svg)](https://github.com/biopb/biopb/actions/workflows/java-ci.yaml)

An experimental biomedical image analysis platform, designed from the ground up around three aspects of the modern analysis pipeline: (1) distributed data serving, (2) easy model customization and deployment, and (3) LLM/agentic workflows that co-evolve with the user's tasks.

## Quick Start
```sh
curl -fsSL https://biopb.org/install.sh | bash
```

## Sub-packages

### biopb-tensor-server

An imaging data server for sharing large collections of imaging datasets, distributed and interfacing with both human users (GUI) and LLMs (API). [Read More...](biopb-tensor-server/README.md)

### biopb-image-runtime

Deploy distributed image processing algorithms (e.g., large deep-learning models) on the network as services. [Read More...](biopb-image-runtime/README.md)

### biopb-mcp

An LLM interface for the full biopb platform, so an AI agent can collaborate with a user on image analysis tasks. Data analysis is python code (ipykernel) with two input streams: (1) Jupyter notebook (human), or (2) chat -> LLM -> agent code. [Read More...](biopb-mcp/README.md)

### SDK

Schema, utilities and cli for building your own workflow. Explore your data in jupyter notebook etc.

## Related Projects

### biopb-servers

Pre-built and ready-to-deploy algorithm servers based on biopb-image-runtimes. [Read More...](https://github.com/biopb/biopb-server)

### biopb-napari-widget

A data browser plugin for napari, tightly integrated with biopb platform so that napari can serve as the direct visualization front-end. [Read More...](https://github.com/biopb/biopb-napari-widget)

### biopb-fiji-plugin

A data browser plugin for Fiji. [Read More...](https://github.com/biopb/biopb-fiji-plugin)

### biopb-ood

Deploying the BioPB stack on HPC/Open OnDemand to take full advantage of its distributed framework. [Read More...](https://github.com/biopb/biopb-ood)

## Contributing

Contributions are very welcome. Read the [development document](development.md) first to understand the overall design architecture.

## License

Distributed under the terms of the [MIT](https://github.com/biopb/biopb/raw/main/LICENSE) license.

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

[file an issue]: https://github.com/biopb/biopb/issues
