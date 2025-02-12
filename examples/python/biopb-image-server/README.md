## Server implementation of biopb.image service
This is a minimal server implementation of the biopb.image protocol based on the [cellpose](https://www.cellpose.org/) model.

### Steps to run this example
Install dependencies
```
pip install biopb cellpose opencv-python-headless grpcio-tools scikit-image
```

Run server
```
python examples/python/biopb-image-server/minimal_cellpose_server.py
```

Test server: see ../biopb-image-client
  - Change the server address to localhost:50051
  - Use HTTP protocol instead of HTTPS
