## Server implementation of biopb.image service
This is a minimal server implementation of the biopb.image protocol based on the [cellpose](https://www.cellpose.org/) model.

### Steps to run this example
Install dependencies
```
pip install biopb cellpose
```

Run server
```
python examples/python/biopb-image-server/minimal_cellpose_server.py
```

> **_NOTE:_**  This server runs at `localhost:50051` on HTTP (unencrypted).
