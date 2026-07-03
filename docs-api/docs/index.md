# biopb Python SDK

API reference for the Python bindings of **biopb**

This reference is generated from the source of the released version and covers
the public SDK surface:

- **[`biopb.image`](reference/biopb/image/index.md)** — the `biopb.image`
  protocol: gRPC service stubs (`ProcessImage`, `ObjectDetection`), request /
  response messages, ROI types, and the image-data serialization helpers in
  `biopb.image.utils`.
- **[`biopb.tensor`](reference/biopb/tensor/index.md)** — the Arrow Flight
  tensor framework: `TensorFlightClient` for lazy, larger-than-memory tensor
  access, plus the associated descriptor / ticket messages.

!!! note "Protocol definitions"
    The `.proto` definitions themselves (field-level documentation for every
    message and RPC) are browsable on
    [buf.build](https://buf.build/jiyuuchc/biopb/docs/main%3Abiopb.image).
    This page documents the **Python API** generated from and built around them.

## Installation

```bash
pip install biopb
# with the lazy tensor / Arrow Flight extras:
pip install "biopb[tensor]"
```
