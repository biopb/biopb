from setuptools.command.build_py import build_py as _build_py
from pathlib import Path


class read_version(_build_py):
    def run(self):
        version_file = Path(__file__).parent / "VERSION"
        version = version_file.read_text().strip()
        version_py = Path(__file__).parent / "biopb_tensor_server" / "_version.py"
        version_py.write_text(f'__version__ = "{version}"\n')
        super().run()