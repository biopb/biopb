"""Entry point for running biopb.tensor as a module.

Usage:
    python -m biopb.tensor serve --config tensorflight.toml
    python -m biopb.tensor validate config.toml
    python -m biopb.tensor list config.toml
"""

from biopb.tensor.cli import app

if __name__ == "__main__":
    app()