import sys

import grpc
import imageio.v2 as imageio
import biopb.image as proto

from biopb.image.utils import serialize_from_numpy, deserialize_to_numpy

def grpc_call(server, image, use_https=False):
    request = proto.ProcessRequest(
        image_data = proto.ImageData(pixels=serialize_from_numpy(image)),
    )

    if use_https:
        with grpc.secure_channel(target=server, credentials=grpc.ssl_channel_credentials()) as channel:        
            stub = proto.ProcessImageStub(channel)
            response = stub.Run(request)
    else:
        with grpc.insecure_channel(target=server) as channel:        
            stub = proto.ProcessImageStub(channel)
            response = stub.Run(request)

    label = deserialize_to_numpy(response.image_data.pixels)
    
    return label


def main():
    server = sys.argv[1]

    image = imageio.imread(sys.argv[2]).astype("uint8")
    if image.ndim==3 and image.shape[0] < 3:
        image = image.tranpose(1, 2, 0)

    print(f"Loaded input image {sys.argv[2]}")

    label = grpc_call(server, image)

    print(f"Found {label.max()} cells")

    imageio.imwrite(sys.argv[3], label)
    print(f"Label image saved to {sys.argv[1]}")

if __name__ == "__main__":
    main()
