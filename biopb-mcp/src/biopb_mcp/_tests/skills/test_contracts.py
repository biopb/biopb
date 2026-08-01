"""Do the third-party APIs the skill bodies quote still look like that?

A skill body is an un-versioned assertion about someone else's package, read by
an agent months later. The failures this layer exists for have all been of one
kind -- the prose stayed still while the library moved:

  * `m2stitch.stitch_images(row_col_transpose=...)` defaults to True, i.e. it
    swaps rows and cols for you. A body passing them positionally produced
    exactly the "diagonal staircase" its own failure table described.
  * `BaSiC.fit()` taking a numpy stack, and exposing `.flatfield` / `.darkfield`
    as attributes rather than a returned tuple.

These are `importorskip`ed: the packages are a *skill's* dependency, not this
package's, so an ordinary checkout runs the rest of the suite unaffected.
Arming them costs several GB (basicpy is torch-backed), which is why CI does not
-- see ``test_satisfiability.py`` for the half that does run there, and
``skills/README.md`` for how to arm this one.
"""

from __future__ import annotations

import inspect
import re

import pytest

from .conftest import SKILLS_DIR

pytestmark = pytest.mark.contract

SKILL = SKILLS_DIR / "flatfield-and-stitch-tiles.md"
# Fences sit inside numbered steps, so they are indented -- do not anchor at
# column 0.
FENCE = re.compile(
    r"^[ \t]*```(?:python)?[ \t]*\n(.*?)^[ \t]*```", re.DOTALL | re.MULTILINE
)


@pytest.fixture(scope="module")
def code() -> str:
    """Only the fenced code, not the prose around it. The failure-modes table
    quotes these same call signatures in English, so matching the whole body
    would pass even after the call itself lost the argument."""
    body = SKILL.read_text(encoding="utf-8")
    fences = FENCE.findall(body)
    assert fences, f"{SKILL.name} has no code fences"
    return "\n".join(fences)


# --- m2stitch -------------------------------------------------------------


def test_stitch_images_still_takes_row_col_transpose(code):
    m2stitch = pytest.importorskip("m2stitch")
    sig = inspect.signature(m2stitch.stitch_images)
    assert "row_col_transpose" in sig.parameters
    # The body passes it explicitly *and* tells the reader what the default is.
    # If upstream flips the default, that sentence becomes a lie.
    default = sig.parameters["row_col_transpose"].default
    assert default is True, (
        "m2stitch changed the row_col_transpose default; the skill body and its "
        "failure-modes row both state it is True"
    )
    assert "row_col_transpose=False" in code


def test_stitch_images_still_takes_ncc_threshold(code):
    m2stitch = pytest.importorskip("m2stitch")
    sig = inspect.signature(m2stitch.stitch_images)
    assert "ncc_threshold" in sig.parameters
    assert "ncc_threshold=NCC_THRESHOLD" in code


def test_the_first_two_positional_args_are_rows_then_cols():
    """The body passes `corrected, rows, cols` positionally after the image."""
    m2stitch = pytest.importorskip("m2stitch")
    names = list(inspect.signature(m2stitch.stitch_images).parameters)
    assert names[:3] == ["images", "rows", "cols"], names[:3]


# --- basicpy --------------------------------------------------------------


def test_basic_exposes_fit_and_the_two_fields(code):
    basicpy = pytest.importorskip("basicpy")
    basic = basicpy.BaSiC(get_darkfield=False)
    assert callable(basic.fit)
    for attr in ("flatfield", "darkfield"):
        assert hasattr(basic, attr), f"BaSiC lost .{attr}"
    assert "basic.flatfield, basic.darkfield" in code


def test_basic_still_takes_get_darkfield_at_construction(code):
    basicpy = pytest.importorskip("basicpy")
    assert "get_darkfield" in basicpy.BaSiC.model_fields
    assert "BaSiC(get_darkfield=GET_DARKFIELD)" in code


def test_basic_fit_takes_a_numpy_stack_and_returns_nothing():
    """The body states BaSiC is numpy-only and in-memory -- which is what the
    step-2 memory gate is sized against -- and reads the fields off the object
    afterwards rather than from a return value. Both are in the signature, so
    assert them there: actually fitting would run the solver on whatever
    accelerator is present, which is an outcome test, not a contract one."""
    basicpy = pytest.importorskip("basicpy")
    np = pytest.importorskip("numpy")
    sig = inspect.signature(basicpy.BaSiC.fit)
    images = sig.parameters["images"]
    assert images.annotation is np.ndarray, (
        f"BaSiC.fit no longer declares a numpy stack: {images.annotation}"
    )
    assert sig.return_annotation is None, (
        "BaSiC.fit returns something now; the body reads .flatfield/.darkfield "
        "off the object instead"
    )
