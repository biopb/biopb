try:
    from ._version import version as __version__
except ImportError:
    import importlib.metadata

    __version__ = importlib.metadata.version("napari-biopb")
except Exception:
    __version__ = "unknown"

from ._object_detection import ObjectDetectionWidget
from ._image_processing import ImageProcessingWidget
from ._tensor_browser import TensorBrowserWidget

__all__ = (
    "ObjectDetectionWidget",
    "ImageProcessingWidget",
    "TensorBrowserWidget",
)
