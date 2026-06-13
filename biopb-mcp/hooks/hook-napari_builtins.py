"""PyInstaller hook for napari_builtins.

Collects the builtins.yaml manifest for npe2 plugin discovery.
"""

from PyInstaller.utils.hooks import collect_data_files

datas = collect_data_files("napari_builtins")
