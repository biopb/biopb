"""What a cache miss costs, decided from the chunk_id.

The class is written into a cache segment, so it must be a property of the chunk
and not of whoever stored it first: the precache and a client can ask for the
same coarse chunk_id, and whichever arrives first would otherwise fix the class
for the life of that segment.

Only a *computed* scaled chunk can be cheap, and only when its scale is off the
advertised ladder. Full resolution is normal, and so is a native pyramid level:
``_plan_from_precomputed`` mints those chunk_ids against the level's own store,
so they arrive unscaled and need no special case.

The ladder lives here rather than on the adapter so the pyramid config does not
have to be threaded down the read path. Adapters are built without the server's
config, and a per-chunk call cannot afford to carry one through every override
of ``resolve_chunk_data``; the server sets it once at construction instead.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, FrozenSet, Optional, Sequence, Tuple

from biopb_tensor_server.core.chunk import (
    compute_pyramid_scale_hints,
    decode_scale_info,
    is_scaled_chunk,
)
from biopb_tensor_server.core.config import PyramidConfig

if TYPE_CHECKING:
    from biopb_tensor_server.cache import RetentionClass

# The server's ladder knobs, set once at construction (TensorFlightServer) and
# read per chunk. A default instance keeps a direct caller -- a test, a script
# driving an adapter with no server -- on the shipped ladder rather than on none.
_active_pyramid_config = PyramidConfig()


def set_active_pyramid_config(config: Optional[PyramidConfig]) -> None:
    """Install the ladder every retention decision is judged against.

    One process serves one config, so this is process state rather than an
    argument on the read path. A second server in the same process (tests) wins
    the global, and an adapter that already memoized its ladder keeps the old
    one -- neither matters while the config is read once at server start.
    """
    global _active_pyramid_config
    _active_pyramid_config = config or PyramidConfig()


def active_pyramid_config() -> PyramidConfig:
    """The ladder knobs currently in force."""
    return _active_pyramid_config


def computed_ladder(
    shape: Sequence[int], dim_labels: Optional[Sequence[str]]
) -> FrozenSet[Tuple[int, ...]]:
    """The computed pyramid's scales for one tensor, under the active config.

    ``PyramidConfig`` is the single source of truth for the levels the server
    advertises *and* the ones the precache warms, so a decision made from this
    set cannot drift from either.
    """
    cfg = _active_pyramid_config
    return frozenset(
        tuple(int(s) for s in level)
        for level in compute_pyramid_scale_hints(
            list(shape),
            list(dim_labels) if dim_labels else None,
            threshold=cfg.threshold,
            downscale_factor=cfg.downscale_factor,
            pixel_budget_cubic_root=cfg.pixel_budget_cubic_root,
            plane_max_pixels=cfg.plane_max_pixels,
        )
    )


def retention_for_chunk(
    chunk_id: bytes, ladder: FrozenSet[Tuple[int, ...]]
) -> RetentionClass:
    """Classify one chunk against a tensor's ``computed_ladder``.

    A rung of the ladder is what a client returns to on every open, and what the
    precache warms. Any other scale is a one-off, asked for in passing and
    regenerable from the full-resolution chunks under it.
    """
    if not is_scaled_chunk(chunk_id):
        return "normal"
    scale = tuple(int(s) for s in decode_scale_info(chunk_id))
    return "normal" if scale in ladder else "cheap"
