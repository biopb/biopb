import logging

import biopb.image as proto
import imageio.v2 as imageio
import grpc
import numpy as np
import typer

from pathlib import Path
from common import decode_image, encode_image, _AUTH_HEADER_KEY

app = typer.Typer(pretty_exceptions_enable=False)

logger = logging.getLogger(__name__)

def construct_request(image: np.ndarray) -> proto.DetectionRequest:
    return proto.DetectionRequest(
        image_data = proto.ImageData(pixleencode_image(image)),
        detection_settings = proto.DetectionSettings(
          scaling_hint = 1.0,
        ),
    )

@app.command()
def main(
    port: int = 50051,
    ip: str = "127.0.0.1",
    token: str = "",
    test_img: Path = Path("test_image.png"),
    debug: bool = False,
):
    logger.setLevel(logging.DEBUG if debug else logging.INFO)

    SERVER = f"{ip}:{port}"
    METADATA = ((_AUTH_HEADER_KEY, "Bearer " + token.strip()),) 
    logger.info(f"Testing server at {SERVER} with token {token}")

    image = imageio.imread(test_img)
    logger.info(f"Loaded image {test_img} with shape {image.shape}")

    # test ObjectDetection service
    logger.info("Testing ObjectDetection service...")
    try:
        with grpc.insecure_channel(SERVER) as channel:
            stub = proto.ObjectDetectionStub(channel)
            response = stub.RunDetection(proto.DetectionRequest(
                image_data=proto.ImageData(pixels=encode_image(image)),
                detection_settings=proto.DetectionSettings(),
            ), metadata=METADATA)
        print(f"ObjectDetection call responsed with {len(response.detections)} detections.")

    except grpc.RpcError as e:
        logger.error(f"ObjectDetection call failed: {e}")
        return
    
    logger.info("Testing ObjectDetection service...")
    try:
        with grpc.insecure_channel(SERVER) as channel:
            stub = proto.ProcessImageStub(channel)
            response = stub.Run(proto.ProcessRequest(
                image_data=proto.ImageData(pixels=encode_image(image)),
            ), metadata=METADATA)

        result = decode_image(response.image_data.pixels)
        print(f"ProcessImage call responsed with {result.max()} detections.")

    except grpc.RpcError as e:
        logger.error(f"ProcessImage call failed: {e}")
        return

if __name__ == "__main__":
    logging.basicConfig()
    app()
