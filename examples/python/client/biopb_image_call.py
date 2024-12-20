import sys
from pathlib import Path

import cv2
import numpy as np
import grpc
import imageio
import biopb.image as proto

SERVER = "127.0.0.1:50051"

def grpc_call(image):
    request = proto.DetectionRequest(
        image_data = proto.ImageData(
            pixels = proto.Pixels(
                bindata = proto.BinData(data=image.tobytes()),
                size_x = image.shape[1],
                size_y = image.shape[0],
                size_c = image.shape[2],
                dimension_order = "CXYZT",
                dtype = "u1", # uint8
            )
        ),
        detection_settings = proto.DetectionSettings(
            scaling_hint = 1.0,
        ),
    )

    # call server
    with grpc.secure_channel(target=SERVER) as channel:
        stub = proto.ObjectDetectionStub(channel)
        response = stub.RunObjectDetection(request)

    # generate label
    label = np.zeros(image.shape[:2], dtype="uint8")
    for k, det in enumerate(response.detections):
        polygon = [[p.x, p.y] for p in det.roi.polygon.points]
        polygon = np.round(np.array(polygon)).astype(int)

        cv2.fillPoly(label, [polygon], k + 1)
    
    return label


def main():
    image = imageio.imread(sys.argv[1]).astype("uint8")
    print(f"Loaded input image {sys.argv[1]}")

    label = grpc_call(image)

    imageio.imwrite(sys.argv[2], label)
    print(f"Label image saved to {sys.argv[1]}")
    

if __name__ == "__main__":
    main()
