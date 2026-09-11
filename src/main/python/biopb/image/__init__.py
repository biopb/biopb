# One resolution for the SDK, in the parent package: `biopb.image` is not a
# distribution of its own, and looking the version up again here left
# `__version__` *undefined* whenever the lookup failed -- importing from a
# source tree that was never installed made `biopb.image.__version__` raise
# AttributeError instead of reporting anything.
from biopb import __version__
from biopb.image.annotation_pb2 import (
    RoiAnnotation,
    RoiConflict,
    RoiDeleteRequest,
    RoiDeleteResult,
    RoiListRequest,
    RoiListResult,
    RoiPutRequest,
    RoiPutResult,
)
from biopb.image.bindata_pb2 import BinData
from biopb.image.detection_request_pb2 import DetectionRequest
from biopb.image.detection_response_pb2 import DetectionResponse, ScoredROI
from biopb.image.detection_settings_pb2 import DetectionSettings
from biopb.image.image_data_pb2 import ImageAnnotation, ImageData, Pixels, Tensor
from biopb.image.op_schema_pb2 import InputShapeHint, OpNames, OpSchema
from biopb.image.roi_pb2 import (
    ROI,
    Ellipse,
    Mask,
    Mesh,
    Point,
    Polygon,
    Polyline,
    Rectangle,
)
from biopb.image.rpc_object_detection_pb2_grpc import (
    ObjectDetection,
    ObjectDetectionServicer,
    ObjectDetectionStub,
    add_ObjectDetectionServicer_to_server,
)
from biopb.image.rpc_process_image_pb2 import ProcessRequest, ProcessResponse
from biopb.image.rpc_process_image_pb2_grpc import (
    ProcessImage,
    ProcessImageServicer,
    ProcessImageStub,
    add_ProcessImageServicer_to_server,
)

# Utility functions for image data serialization/deserialization
from biopb.image.utils import (
    deserialize_image_data,
    deserialize_to_numpy,  # deprecated, use deserialize_image_data_to_numpy instead
    get_image_data_dim_labels,
    get_image_data_shape,
    normalize_array_dims,
    serialize_from_numpy,  # deprecated, use serialize_from_numpy_to_image_data instead
    serialize_from_numpy_to_image_data,
)
