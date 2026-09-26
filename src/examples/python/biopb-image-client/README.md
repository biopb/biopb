## An Ops client

A minimal client of the biopb.image `Ops` protocol: it sends a 2D image to one
op of a server and saves the label image it returns. `biopb image process` does
the same with more options.

Install dependencies
```
pip install biopb imageio
```

Run it against a server (such as `../biopb-image-runtime/cellpose_server.py`)
```
python ops_client.py 127.0.0.1:50051 cellpose <input_image_path> <output_label_path>
```
