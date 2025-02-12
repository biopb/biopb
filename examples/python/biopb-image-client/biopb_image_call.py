import sys
from pathlib import Path

import cv2
import numpy as np
import grpc
import imageio.v2 as imageio
import biopb.image as proto
from biopb.image.utils import serialize_from_numpy

SERVER = "lacss.biopb.org"

def grpc_call(image):
    request = proto.DetectionRequest(
        image_data = proto.ImageData(pixels=serialize_from_numpy(image)),
        detection_settings = proto.DetectionSettings(),
    )

    # call server with HTTPS
    with grpc.secure_channel(target=SERVER, credentials=grpc.ssl_channel_credentials()) as channel:        
        stub = proto.ObjectDetectionStub(channel)
        response = stub.RunDetection(request)

    # for HTTP server use insecure_channel
    # with grpc.insecure_channel(target=SERVER) as channel:        
    #     stub = proto.ObjectDetectionStub(channel)
    #     response = stub.RunDetection(request)

    # generate label
    label = np.zeros(image.shape[:2], dtype="uint8")
    for k, det in enumerate(response.detections):
        polygon = [[p.x, p.y] for p in det.roi.polygon.points]
        polygon = np.round(np.array(polygon)).astype(int)

        cv2.fillPoly(label, [polygon], k + 1)
    
    return label


def main():
    image = imageio.imread(sys.argv[1]).astype("uint8")
    if image.ndim==3 and image.shape[0] < 3:
        image = image.tranpose(1, 2, 0)

    print(f"Loaded input image {sys.argv[1]}")

    label = grpc_call(image)
    print(f"Found {label.max()} cells")

    imageio.imwrite(sys.argv[2], label)
    print(f"Label image saved to {sys.argv[1]}")
    

if __name__ == "__main__":
    main()
