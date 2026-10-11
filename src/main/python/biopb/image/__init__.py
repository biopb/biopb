# No `__version__` here: one distribution, one version, published as
# `biopb.__version__` (biopb/biopb#998). `biopb.tensor` has never had one.
from biopb.image._arg import (
    NDIM_LABELS,
    decode_arg,
    encode_arg,
    json_arg,
    json_value,
    jsonable,
)
from biopb.image._client import OpsClient, connect, make_channel, op_error

# Utility functions for image data serialization/deserialization
from biopb.image._utils import (
    deserialize_image_data,
    get_image_data_dim_labels,
    get_image_data_shape,
    mask_to_roi,
    normalize_array_dims,
    roi_to_mask,
    serialize_from_numpy_to_image_data,
)
from biopb.image.annotation_pb2 import (
    RoiAnnotation,
    RoiConflict,
    RoiDeleteResult,
    RoiListResult,
    RoiPruneRequest,
    RoiPruneResult,
    RoiPutResult,
    RoiSetInfo,
    RoiUnseen,
)
from biopb.image.bindata_pb2 import BinData
from biopb.image.image_data_pb2 import ImageAnnotation, ImageData, Pixels, Tensor
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
from biopb.image.rpc_ops_pb2 import Arg, Call, Event, OpInfo, OpList, TensorArg
from biopb.image.rpc_ops_pb2_grpc import (
    Ops,
    OpsServicer,
    OpsStub,
    add_OpsServicer_to_server,
)

# mkdocstrings only documents a module's re-exports when __all__ names them
# explicitly (see biopb/__init__.py, biopb/tensor/__init__.py) -- without it
# this page rendered as an empty stub despite every name below being the
# actual public surface.
__all__ = [
    "Arg",
    "BinData",
    "Call",
    "Ellipse",
    "Event",
    "ImageAnnotation",
    "ImageData",
    "Mask",
    "Mesh",
    "NDIM_LABELS",
    "OpInfo",
    "OpList",
    "Ops",
    "OpsClient",
    "OpsServicer",
    "OpsStub",
    "Pixels",
    "Point",
    "Polygon",
    "Polyline",
    "ROI",
    "Rectangle",
    "RoiAnnotation",
    "RoiConflict",
    "RoiDeleteResult",
    "RoiListResult",
    "RoiPruneRequest",
    "RoiPruneResult",
    "RoiPutResult",
    "RoiSetInfo",
    "RoiUnseen",
    "Tensor",
    "TensorArg",
    "add_OpsServicer_to_server",
    "connect",
    "decode_arg",
    "deserialize_image_data",
    "encode_arg",
    "get_image_data_dim_labels",
    "get_image_data_shape",
    "json_arg",
    "json_value",
    "jsonable",
    "make_channel",
    "mask_to_roi",
    "normalize_array_dims",
    "op_error",
    "roi_to_mask",
    "serialize_from_numpy_to_image_data",
]
