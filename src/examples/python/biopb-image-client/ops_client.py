"""Call one op of an algorithm server on a 2D image file and save its result."""

import sys

import biopb.image as proto
import grpc
import imageio.v2 as imageio
from biopb.image.utils import deserialize_image_data, serialize_from_numpy_to_image_data
from google.protobuf import empty_pb2


def call(server, op, image):
    with grpc.insecure_channel(server) as channel:
        stub = proto.OpsStub(channel)
        info = {o.name: o for o in stub.Describe(empty_pb2.Empty()).ops}
        (tensor,) = info[op].tensors  # the op's one tensor argument
        arg = proto.Arg(eager=serialize_from_numpy_to_image_data(image).eager_data)
        outputs = None
        for event in stub.Call(proto.Call(op=op, args={tensor: arg})):
            if event.progress:
                print(event.progress)
            if event.outputs:
                outputs = event.outputs
    return deserialize_image_data(proto.ImageData(eager_data=outputs["result"].eager))


def main():
    server, op, image_path, output_path = sys.argv[1:5]
    label = call(server, op, imageio.imread(image_path))
    print(f"Found {label.max()} objects")
    imageio.imwrite(output_path, label)


if __name__ == "__main__":
    main()
