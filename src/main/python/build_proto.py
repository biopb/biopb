"""Custom setuptools command to generate protobuf bindings before build.

This module provides a custom build command that automatically generates
Python protobuf/gRPC bindings from .proto files using buf before the 
standard build process runs.

Usage:
    pip install .              # Normal install, triggers build
    pip install -e .           # Editable install, triggers build
    python -m pip build .      # Build wheel/sdist, triggers build

Requirements:
    buf CLI must be installed. See https://buf.build/docs/installation
"""

import shutil
import subprocess
import sys
from pathlib import Path

from setuptools.command.build_py import build_py as _build_py


class build_proto(_build_py):
    """Custom build_py command that generates protobuf bindings first."""

    def run(self):
        """Run proto generation then the standard build."""
        # Generate protobuf bindings using buf
        self.generate_proto()
        # Continue with standard build
        super().run()

    def generate_proto(self):
        """Generate Python bindings from .proto files using buf."""
        # Find project root (parent of src/main/python)
        project_root = Path(__file__).resolve().parent.parent.parent.parent

        print(f"\n{'='*60}")
        print("Generating protobuf bindings with buf...")
        print(f"Project root: {project_root}")

        # Check if buf is installed
        buf_path = shutil.which("buf")
        if buf_path is None:
            print(
                "ERROR: buf CLI not found. Please install buf:\n"
                "  https://buf.build/docs/installation\n"
                "\nOr with Homebrew: brew install buf\n"
                "Or with curl: curl -sSL https://buf.build/installation | bash",
                file=sys.stderr
            )
            sys.exit(1)

        print(f"Using buf: {buf_path}")

        # Run buf generate
        result = subprocess.run(
            ["buf", "generate"],
            cwd=project_root,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            print(f"ERROR: buf generate failed:", file=sys.stderr)
            print(result.stderr, file=sys.stderr)
            sys.exit(1)

        print("buf generate completed successfully")
        print(f"{'='*60}\n")
