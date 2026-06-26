"""Entry point for running biopb_tensor_server as a module.

Usage:
    python -m biopb_tensor_server serve --config biopb-tensor.toml
    python -m biopb_tensor_server validate config.toml
    python -m biopb_tensor_server list config.toml
"""

from biopb_tensor_server.cli import app

if __name__ == "__main__":
    app()
