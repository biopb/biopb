"""PyInstaller hook for napari-biopb plugin.

Collects the npe2 manifest and all submodules for the frozen bundle.
"""

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# Collect napari-biopb's npe2 manifest and data files
datas = collect_data_files("napari_biopb")

# Include all napari-biopb submodules
hiddenimports = collect_submodules("napari_biopb")
