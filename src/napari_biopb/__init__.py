import logging as _logging

try:
    from ._version import version as __version__
except ImportError:
    import importlib.metadata

    __version__ = importlib.metadata.version("napari-biopb")
except Exception:
    __version__ = "unknown"

from .image_processing import ImageProcessingWidget, ObjectDetectionWidget
from .tensor_browser import TensorBrowserWidget

__all__ = (
    "ObjectDetectionWidget",
    "ImageProcessingWidget",
    "TensorBrowserWidget",
)

_logger = _logging.getLogger(__name__)
