"""An Ops server that echoes its input: docker-compose.yaml's test service."""

from biopb_image_base import Tensor, op, serve


@op(description="Echo the image back", labels=["test"])
def echo(image: Tensor("YX")):
    return image


@op(description="Echo the image back, by reference", input="lazy")
def echo_lazy(image: Tensor("YX")):
    return image


if __name__ == "__main__":
    serve()
