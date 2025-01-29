import sys
from pathlib import Path

import cv2
import numpy as np
import grpc
import imageio.v2 as imageio
import biopb.image as proto

SERVER = "lacss.biopb.org"

def grpc_call(image):
    request = proto.DetectionRequest(
        image_data = proto.ImageData(
            pixels = proto.Pixels(
                bindata = proto.BinData(data=image.tobytes()),
                size_x = image.shape[1],
                size_y = image.shape[0],
                size_c = image.shape[2],
                dimension_order = "CXYZT",
                dtype = image.dtype.str,
            )
        ),
        detection_settings = proto.DetectionSettings(),
    )

    # call server
    with grpc.secure_channel(target=SERVER, credentials=grpc.ssl_channel_credentials()) as channel:
        stub = proto.ObjectDetectionStub(channel)
        response = stub.RunDetection(request)

    # generate label
    label = np.zeros(image.shape[:2], dtype="uint8")
    for k, det in enumerate(response.detections):
        polygon = [[p.x, p.y] for p in det.roi.polygon.points]
        polygon = np.round(np.array(polygon)).astype(int)

        cv2.fillPoly(label, [polygon], k + 1)
    
    return label


def main():
    image = imageio.imread(sys.argv[1]).astype("uint8")
    if image.ndim==2:
        image = image[:, :, None]

    print(f"Loaded input image {sys.argv[1]}")

    label = grpc_call(image)
    print(f"Found {label.max()} cells")

    imageio.imwrite(sys.argv[2], label)
    print(f"Label image saved to {sys.argv[1]}")
    

if __name__ == "__main__":
    main()
