## An algorithm server

A server file for the biopb.image `Ops` protocol, serving one op backed by the
[cellpose](https://www.cellpose.org/) model. The header is uv's inline script
metadata, so uv installs what it imports.

Run it
```
uv run src/examples/python/biopb-image-runtime/cellpose_server.py
```

It serves on `127.0.0.1:50051`; `--describe` prints its ops and exits. Copied
to `~/.config/biopb/algorithms/cellpose.py`, the control runs it instead and the
kernel's `ops.cellpose` calls it.
