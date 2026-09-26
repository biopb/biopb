# /// script
# requires-python = ">=3.10"
# dependencies = ["biopb-image-base", "cellpose"]
# ///
"""A cellpose segmentation op, served over the biopb.image Ops protocol."""

from functools import lru_cache

from biopb_image_base import Tensor, op, serve


@lru_cache(maxsize=1)
def _model():
    from cellpose import models

    return models.CellposeModel(gpu=True)


@op(description="Cell segmentation with cellpose", labels=["segmentation"])
def cellpose(image: Tensor("YX"), diameter: float = 0.0):
    """A label image of the cells in a 2D image."""
    masks = _model().eval(image, diameter=diameter or None)[0]
    return masks.astype("uint32")


if __name__ == "__main__":
    serve()
