import threading
from concurrent import futures

import grpc
import numpy as np
import cv2
import biopb.image as proto
from biopb.image.utils import deserialize_to_numpy
from skimage.measure import regionprops
from cellpose import models

# increase message size limit
_MAX_MSG_SIZE=1024*1024*128

def decode_image(pixels:proto.Pixels) -> np.ndarray:
    np_img = deserialize_to_numpy(pixels)

    return np_img


def process_input(request: proto.DetectionRequest):
    pixels = request.image_data.pixels

    image = decode_image(pixels).astype("float32")

    # use only the first channel
    image = image[..., :1]
    
    assert image.shape[0] == 1 # 2D
    
    image = image.squeeze(0)

    settings = request.detection_settings

    if settings.HasField("cell_diameter_hint"):

        physical_size = pixels.physical_size_x or 1
        diameter = settings.cell_diameter_hint / physical_size

    else:

        diameter = 30 / (settings.scaling_hint or 1.0)   

    kwargs = dict(
        diameter = diameter,
    )

    return image, kwargs


def process_result(preds, image):
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

    return response


class CellposeServicer(proto.ObjectDetectionServicer):

    def __init__(self, model):
        self.model = model
        self._lock = threading.RLock()

    def RunDetection(self, request, context):
        with self._lock:

            try:
                image, kwargs = process_input(request)

                preds = self.model.eval(image,  **kwargs,)

                response = process_result(preds, image)

                return response
            
            except Exception as e:

                context.abort(grpc.StatusCode.UNKNOWN, f"prediction failed with error: {repr(e)}")


def main():
    print ("server starting ...")

    model = models.Cellpose(model_type = "cyto3", gpu=True)

    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=4),
        options=(("grpc.max_receive_message_length", _MAX_MSG_SIZE),),
    )

    proto.add_ObjectDetectionServicer_to_server(
        CellposeServicer(model), 
        server,
    )

    server.add_insecure_port("127.0.0.1:50051")

    server.start()

    print ("server starting ... ready")

    server.wait_for_termination()


if __name__ == "__main__":
    main()

