# BioPB

_Distributed computing for bio-imaging data._

BioPB provides a standardized way to share multi-dimensional microscopy datasets, analysis algorithms, and computational results in a network-transparent and language-agnostic manner. 

## Biopb-Tensor

A blazing-fast microscopy data server for sharing your lab's petabyte-scale datasets to team members and/or collaborators. [Read More...](biopb-tensor-server/README.md)

  - **Uniform Representation**: all source data mapped to a _multi-resolution_ and _lazy-read_ array for client access
  - **Multi-language**: dask array for Python and ImgLib2.CellImg for Java
  - **Thread-safe & Serializable**: compatible with dask.distribute for distributed computing on larger-than-memory dataset
  - **Metadata Server**: full DuckDB SQL support to query your embedded metadata
  - **On-the-fly Build**: keep your data in original format (.zvi, ndtiff etc). No staging or on-boarding process needed
  - **Built-in Viewer**: browse all your data with any browser (e.g., on an ipad)

### Quick Start
```sh
# Ports: 8814 (browser), 8815 (data)
# Dev mode with localhost-only access (no token required)
docker run --rm \
  --name tensor-server \
  -e BIOPB_WEB_DEV_BYPASS=1 \
  -p 127.0.0.1:8814:8814 \
  -p 127.0.0.1:8815:8815 \
  -v ${DIR_YOUR_DATA}:/data \
  jiyuuchc/biopb-tensor-server:latest
```
Point your browser to http://localhost:8814

## Biopb-Image

Deploy complex image processing algorithms (e.g., large deep-learning models) on the network as services, including pre-built containers for popular algorithms (e.g., segmentation). [Read More...](https://github.com/biopb/biopb-server)

## Napari-Biopb

GUI app for the end-users. The power of biopb in a familiar interface. [Read More...](https://github.com/biopb/napari-biopb)

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
