"""Top-level CLI for BioPB."""

import typer
from biopb.tensor.cli import app as tensor_app
from biopb.image.cli import app as image_app

app = typer.Typer(
    name="biopb",
    help="BioPB: open protobuf/gRPC protocols for biomedical image processing",
)
app.add_typer(tensor_app, name="tensor", help="TensorFlight client diagnostics")
app.add_typer(image_app, name="image", help="ProcessImage client operations")

if __name__ == "__main__":
    app()