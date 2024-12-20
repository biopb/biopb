## Server implementation of Cellpose segmentation model
This is a minimal server implementation of the biopb.image protocol based on the [cellpose](https://www.cellpose.org/) model.

### Steps to run this example
Install dependencies
```
pip install cellpose opencv-python-headless grpcio-tools scikit-image
```
Generate the python bindings
```
scripts/gen.sh
export PYTHONPATH=$PYTHONPATH:gen/python
```
Run server
```
python examples/python/server/cellpose.py
```
