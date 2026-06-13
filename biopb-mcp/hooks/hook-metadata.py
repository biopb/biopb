"""PyInstaller hook for biopb-mcp dependencies that need metadata.

Collects package metadata for importlib.metadata.version() calls.
"""

from PyInstaller.utils.hooks import copy_metadata

# Collect metadata for key packages
datas = copy_metadata("biopb-mcp")
datas += copy_metadata("biopb")
datas += copy_metadata("napari")
datas += copy_metadata("magicgui")
datas += copy_metadata("scikit-image")
