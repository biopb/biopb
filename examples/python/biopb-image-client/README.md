## Example grpc client implementation
This is a minimal client implementation calling a local GRPC server implemeting biopb.image protocol

### Steps to run this example
Install dependencies
```
pip install biopb imageio
```

Run client
```
python biopb_image_client.py <server> <input_image_path> <output_label_path>
```
