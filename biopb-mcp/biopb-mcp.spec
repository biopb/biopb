# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec file for biopb-mcp bundled application."""

import sys
from PyInstaller.building.build_main import Analysis, PYZ, EXE, COLLECT, BUNDLE
from PyInstaller.utils.hooks import copy_metadata

import napari

sys.modules["FixTk"] = None

NAME = "biopb-mcp"
WINDOWED = True
DEBUG = False
UPX = False
BLOCK_CIPHER = None

# Collect package metadata for importlib.metadata.version() calls
METADATA_DATAS = []
for pkg in [
    "imageio",
    "napari",
    "biopb-mcp",
    "biopb",
    "magicgui",
    "scikit-image",
    "numpy",
    "pandas",
    "scipy",
    "zarr",
    "dask",
]:
    try:
        METADATA_DATAS += copy_metadata(pkg)
    except Exception:
        pass  # Package not installed or no metadata


def get_icon():
    """Return platform-appropriate icon file."""
    if sys.platform.startswith("win"):
        return "logo.ico"
    elif sys.platform == "darwin":
        return "logo.icns"
    return None


def get_version():
    """Return Windows version info for exe."""
    if sys.platform != "win32":
        return None

    from PyInstaller.utils.win32 import versioninfo as vi

    ver_str = napari.__version__
    version = ver_str.replace("+", ".").split(".")
    version = [int(x) for x in version if x.isnumeric()]
    version += [0] * (4 - len(version))
    version = tuple(version)[:4]

    return vi.VSVersionInfo(
        ffi=vi.FixedFileInfo(filevers=version, prodvers=version),
        kids=[
            vi.StringFileInfo(
                [
                    vi.StringTable(
                        "000004b0",
                        [
                            vi.StringStruct("CompanyName", NAME),
                            vi.StringStruct("FileDescription", NAME),
                            vi.StringStruct("FileVersion", ver_str),
                            vi.StringStruct("LegalCopyright", "MIT License"),
                            vi.StringStruct("OriginalFileName", NAME + ".exe"),
                            vi.StringStruct("ProductName", NAME),
                            vi.StringStruct("ProductVersion", ver_str),
                        ],
                    )
                ]
            ),
            vi.VarFileInfo([vi.VarStruct("Translation", [0, 1200])]),
        ],
    )


a = Analysis(
    ["main.py"],
    hookspath=["hooks"],
    excludes=[
        "FixTk",
        "tcl",
        "tk",
        "_tkinter",
        "tkinter",
        "Tkinter",
        "matplotlib",
        "PyQt5",
        "PyQt5.QtCore",
        "PyQt5.QtGui",
        "PyQt5.QtWidgets",
        "PySide2",
    ],
    datas=METADATA_DATAS,
    cipher=BLOCK_CIPHER,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=BLOCK_CIPHER)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=NAME,
    debug=DEBUG,
    upx=UPX,
    console=(not WINDOWED),
    icon=get_icon(),
    version=get_version(),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    upx=UPX,
    name=NAME,
)

if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name=NAME + ".app",
        icon=get_icon(),
        bundle_identifier=f"com.{NAME}.{NAME}",
        info_plist={
            "CFBundleIdentifier": f"com.{NAME}.{NAME}",
            "CFBundleShortVersionString": napari.__version__,
            "NSHighResolutionCapable": "True",
        },
    )
