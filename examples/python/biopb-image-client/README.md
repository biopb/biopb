## Example grpc client implementation
This is a minimal client implementation calling a local GRPC server implemeting biopb.image protocol

### Steps to run this example
Install dependencies
```
pip install biopb opencv-python-headless imageio
```

Run client
```
python biopb_image_client.py <input_image_path> <output_label_path>
```
