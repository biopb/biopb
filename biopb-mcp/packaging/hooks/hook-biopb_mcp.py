"""PyInstaller hook for biopb-mcp plugin.

Collects the npe2 manifest and all submodules for the frozen bundle.
"""

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# Collect biopb-mcp's npe2 manifest and data files
datas = collect_data_files("biopb_mcp")

# Include all biopb-mcp submodules
hiddenimports = collect_submodules("biopb_mcp")
