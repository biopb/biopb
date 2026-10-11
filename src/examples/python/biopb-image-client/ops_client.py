"""Call one op of an algorithm server on a 2D image file and save its result."""

import sys

import biopb.image as proto
import imageio.v2 as imageio


def call(server, op, image):
    with proto.connect(server) as client:
        (info,) = (o for o in client.describe().ops if o.name == op)
        (tensor,) = info.tensors  # the op's one tensor argument
        return client.call(op, on_progress=print, **{tensor: image})


def main():
    server, op, image_path, output_path = sys.argv[1:5]
    label = call(server, op, imageio.imread(image_path))
    print(f"Found {label.max()} objects")
    imageio.imwrite(output_path, label)


if __name__ == "__main__":
    main()
