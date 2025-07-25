import logging
import traceback
import threading

import biopb.image as proto
import grpc
import numpy as np
import typer

from typing import Optional
from concurrent import futures
from cellpose import models
from common import decode_image, encode_image, TokenValidationInterceptor, BiopbServicerBase

app = typer.Typer(pretty_exceptions_enable=False)

logger = logging.getLogger(__name__)

_MAX_MSG_SIZE=1024*1024*128
_TARGET_CELL_SIZE=30


def process_input(request: proto.DetectionRequest):
    pixels = request.image_data.pixels
    settings = request.detection_settings

    image = decode_image(pixels)

    physical_size = pixels.physical_size_x or 1

    if settings.HasField("cell_diameter_hint"):
        diameter = settings.cell_diameter_hint / physical_size

    else:
        diameter = _TARGET_CELL_SIZE / (settings.scaling_hint or 1.0)

    if image.shape[0] == 1: # 2D
        image = image.squeeze(0)

    if image.shape[-1] > 1:
        channels = [1, 2]
    else:
        channels = [0, 0]

    kwargs = dict(
        diameter = diameter,
        channels = channels,
    )

    return image, kwargs


def process_result(preds, image):
    import cv2
    from skimage.measure import regionprops

    response = proto.DetectionResponse()

    try:
        masks, flows, styles, _ = preds
    except:
        masks, flows, styles = preds

    for rp in regionprops(masks):
        mask = rp.image.astype("uint8")
        c, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = np.array(c[0], dtype=float).squeeze(1)
        c = c + np.array([rp.bbox[1] , rp.bbox[0]])
        c = c - 0.5

        scored_roi = proto.ScoredROI(
            score = 1.0,
            roi = proto.ROI(
                polygon = proto.Polygon(points = [proto.Point(x=p[0], y=p[1]) for p in c]),
            )
        )

        response.detections.append(scored_roi)
    
    logger.debug(f"Found {len(response.detections)} detections")

    return response


class CellposeServicer(BiopbServicerBase):

    def __init__(self, model):
        self.model = model
        self._lock = threading.RLock()

    def RunDetection(self, request, context):
        with self._server_context(context):
            logger.info(f"Received message of size {request.ByteSize()}")

            image, kwargs = process_input(request)

            if image.ndim == 4:
                raise ValueError("3D input not supported. Use the 'ProcessImage' service instead.")

            logger.info(f"received image {image.shape}")

            preds = self.model.eval(image,  **kwargs,)

            response = process_result(preds, image)

            logger.info(f"Reply with message of size {response.ByteSize()}")

            return response


    def Run(self, request, context):
        with self._server_context(context):
            logger.info(f"Received message of size {request.ByteSize()}")

            pixels = request.image_data.pixels
            image = decode_image(pixels)

            if image.shape[-1] > 1:
                channels = [1, 2]
            else:
                channels = [0, 0]

            logger.info(f"Decoded image {image.shape}")

            if image.shape[0] == 1: # 2D
                image = image.squeeze(0)
                mask = self.model.eval(image, channels = channels)[0]

            else:
                mask = self.model.eval(image, channels = channels, do_3D=True)[0]
                
            response = proto.ProcessResponse(
                image_data = proto.ImageData(pixels = encode_image(mask)),
            )

            logger.info(f"Reply with message of size {response.ByteSize()}")

            return response


@app.command()
def main(
    modeltype: str = "cyto3",
    port: int = 50051,
    workers: int = 10,
    ip: str = "0.0.0.0",
    local: bool = False,
    token: Optional[bool] = None,
    debug: bool = False,
    compression: bool = True,
    gpu: bool = True,
    # max_image_size: int = 1088,
):
    print ("server starting ...")

    logger.setLevel(logging.DEBUG if debug else logging.INFO)

    model = models.Cellpose(model_type = modeltype, gpu=gpu)

    if token is None:
        token = not local
    if token:
        import secrets

        token_str = secrets.token_urlsafe(64)

        print()
        print("COPY THE TOKEN BELOW FOR ACCESS.")
        print("=======================================================================")
        print(f"{token_str}")
        print("=======================================================================")
        print()
    else:
        token_str = None

    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=workers),
        compression=grpc.Compression.Gzip if compression else grpc.Compression.NoCompression,
        interceptors=(TokenValidationInterceptor(token_str),),
        options=(("grpc.max_receive_message_length", _MAX_MSG_SIZE),),
    )

    servicer = CellposeServicer(model)
    proto.add_ObjectDetectionServicer_to_server(
        servicer, server,
    )
    proto.add_ProcessImageServicer_to_server(
        servicer, server,
    )

    if local:
        server.add_secure_port(f"127.0.0.1:{port}", grpc.local_server_credentials())
    else:
        server.add_insecure_port(f"{ip}:{port}")

    logger.info(f"cellpose_server: listening on port {port}")

    print ("server starting ... ready")

    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    logging.basicConfig()
    app()
