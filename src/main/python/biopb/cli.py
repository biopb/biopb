"""Top-level CLI for BioPB."""

import typer
from biopb.tensor.cli import app as tensor_app

app = typer.Typer(
    name="biopb",
    help="BioPB: open protobuf/gRPC protocols for biomedical image processing",
)
app.add_typer(tensor_app, name="tensor", help="TensorFlight client diagnostics")

if __name__ == "__main__":
    app()