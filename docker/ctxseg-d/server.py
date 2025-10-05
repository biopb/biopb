import logging
from concurrent import futures
from pathlib import Path
from typing import Optional

import biopb.image as proto
import grpc
import numpy as np
import typer

from flax import nnx

from common import decode_image, encode_image, TokenValidationInterceptor, BiopbServicerBase

app = typer.Typer(pretty_exceptions_enable=False)

logger = logging.getLogger(__name__)

_MAX_MSG_SIZE=1024*1024*128


def process_input(request: proto.DetectionRequest):
    pixels = request.image_data.pixels
    settings = request.detection_settings

    image = decode_image(pixels).squeeze(0)

    return image


def process_result(masks):
    import cv2
    from skimage.measure import regionprops

    response = proto.DetectionResponse()

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

    return response


class CtxSegServicer(BiopbServicerBase):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def predict(self, image, GS=512):
        from ctxseg.segmentation.utils import pad_channel
        from ctxseg.segmentation.flow import flow_to_mask
        from ctxseg.segmentation.utils import clean_up_mask, remove_small_instances


        @nnx.jit
        def _predict(model, image):
            return model.predict(image)

        image = (image - image.min()) / (image.max() - image.min())
        image = pad_channel(image)
        H, W, _ = image.shape

        flow = np.zeros([H, W, 3])
        cnts = np.zeros([H, W, 1])
        for yc in range(0, H, GS-64):
            for xc in range(0, W, GS-64):
                patch = image[yc:yc+GS, xc:xc+GS]
                ph, pw, _ = patch.shape
                if ph <= 64 or pw <= 64:
                    continue
                patch = np.pad(patch, [[0, GS-ph], [0, GS-pw], [0,0]])
                pred = _predict(self.model, patch[None])[0, :ph, :pw]
                flow[yc:yc+ph, xc:xc+pw] += pred
                cnts[yc:yc+ph, xc:xc+pw] += 1

        flow = flow / cnts
   
        mask = flow_to_mask(flow[..., :2])
        mask = np.where(flow[...,-1] > 0.5, mask, 0).astype(np.uint16)
        mask = clean_up_mask(remove_small_instances(mask, 100))

        return mask


    def RunDetection(self, request, context):
        with self._server_context(context):
            image = process_input(request)

            if image.ndim == 4:
                raise ValueError(f"Model does not support 3D input, got shape {image.shape}")

            logger.info(f"received image {image.shape}")

            preds = self.predict(image)

            response = process_result(preds)

            logger.info(f"Reply with message of size {response.ByteSize()}")

            return response


    def Run(self, request, context):
        with self._server_context(context):
            logger.info(f"Received message of size {request.ByteSize()}")

            pixels = request.image_data.pixels

            image = decode_image(pixels)

            logger.info(f"Decoded image {image.shape}")

            if image.shape[0] == 1: # 2D
                image = image.squeeze(0)
                mask = self.predict(image)

            else:
                raise ValueError(f"Model does not support 3D input, got shape {image.shape}")
                
            response = proto.ProcessResponse(
                image_data = proto.ImageData(pixels = encode_image(mask)),
            )

            logger.info(f"Reply with message of size {response.ByteSize()}")

            return response



def load_model(cp):
    import pickle

    with open(cp, "rb") as f:
        target = pickle.load(f)

    if isinstance(target, nnx.State):
        from ctxseg.modeling.ctxseg import CtxSegD
        rngs = nnx.Rngs(0, dropout=1234)
        model = CtxSegD(ps=2,rngs=rngs)
        nnx.update(model, target)
        target = model
    else:
        assert isinstance(target, nnx.Model), f"Unknown checkpoint format: {type(target)}"
    
    target.eval()

    return target

@app.command()
def main(
    modelpath: Path,
    port: int = 50051,
    workers: int = 10,
    ip: str = "0.0.0.0",
    local: bool = False,
    token: Optional[bool] = None,
    debug: bool = False,
    compression: bool = True,
):
    print ("server starting ...")

    logger.setLevel(logging.DEBUG if debug else logging.INFO)

    model = load_model(modelpath)

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

    servicer = CtxSegServicer(model)
    proto.add_ObjectDetectionServicer_to_server(servicer, server)
    proto.add_ProcessImageServicer_to_server(servicer, server)

    if local:
        server.add_secure_port(f"127.0.0.1:{port}", grpc.local_server_credentials())
    else:
        server.add_insecure_port(f"{ip}:{port}")

    logger.info(f"ctxseg_server: listening on port {port}")

    print ("server starting ... ready")

    server.start()

    server.wait_for_termination()

if __name__ == "__main__":
    logging.basicConfig()
    app()
