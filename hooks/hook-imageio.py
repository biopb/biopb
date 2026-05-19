"""PyInstaller hook for imageio.

Collects package metadata required for importlib.metadata.version().
"""

from PyInstaller.utils.hooks import copy_metadata

datas = copy_metadata("imageio")
