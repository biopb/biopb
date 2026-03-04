from concurrent import futures

import biopb.image as proto
import grpc

from biopb.image.utils import deserialize_to_numpy, serialize_from_numpy
from cellpose import models


def process_input(request: proto.DetectionRequest):
    pixels = request.image_data.pixels

    image = deserialize_to_numpy(pixels).astype("float32")

    # Use only the first channel
    image = image[0, :, :, :1]

    return image


def process_result(preds):
    try:
        masks, flows, styles, _ = preds
    except:
        masks, flows, styles = preds
    
    pixels = serialize_from_numpy(masks)

    response = proto.ProcessResponse(
        image_data = proto.ImageData(pixels = pixels),
    )

    return response


class CellposeServicer(proto.ProcessImageServicer):
    def __init__(self, model):
        self.model = model

    def Run(self, request, context):
        try:
            image = process_input(request)

            preds = self.model.eval(image)

            response = process_result(preds)

            return response
        
        except Exception as e:

            context.abort(grpc.StatusCode.UNKNOWN, f"prediction failed with error: {repr(e)}")


def main():
    print ("server starting ...")

    model = models.Cellpose(model_type = "cyto3", gpu=True)

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))

    proto.add_ProcessImageServicer_to_server(CellposeServicer(model), server)

    server.add_insecure_port("127.0.0.1:50051")

    server.start()

    print ("server starting ... ready")

    server.wait_for_termination()


if __name__ == "__main__":
    main()

