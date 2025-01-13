## Example grpc client implementation
This is a minimal client implementation calling a local GRPC server implemeting biopb.image protocol

### Steps to run this example
Install dependencies
```
pip install opencv-python-headless grpcio_tools imageio
```
Generate the python bindings
```
scripts/gen.sh
export PYTHONPATH=$PYTHONPATH:gen/python
```
Run client
```
python examples/python/client/biopb_image_call.py <input_image_path> <output_label_path>
```
