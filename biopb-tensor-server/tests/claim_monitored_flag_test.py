"""``ClaimContext.monitored``: set by the walk, so only a monitored root's claims
keep their content probes."""

import biopb_tensor_server.adapters.ome_tiff as ome_tiff_mod
import numpy as np
import tifffile
from biopb_tensor_server.adapters.ome_tiff import OmeTiffAdapter
from biopb_tensor_server.core.discovery import (
    AdapterRegistry,
    ClaimContext,
    DiscoveryState,
    discover_sources,
)


def test_a_context_is_unmonitored_unless_told_otherwise(tmp_path):
    assert ClaimContext(tmp_path).monitored is False
    assert ClaimContext(tmp_path, monitored=True).monitored is True


def test_the_walk_sets_it_on_every_probe(tmp_path):
    seen = {}

    class _Recorder:
        @classmethod
        def claim(cls, ctx, state):
            seen[ctx.path_str] = ctx.monitored
            return None

    registry = AdapterRegistry()
    registry.register(_Recorder, "recorder")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "a.bin").write_text("x")

    discover_sources(tmp_path, registry, monitored=True)
    assert len(seen) >= 3 and all(seen.values())  # the root, sub and a.bin
    seen.clear()
    discover_sources(tmp_path, registry)
    assert seen and not any(seen.values())


def test_an_ome_claim_memoizes_only_when_monitored(tmp_path):
    ome_tiff_mod._OME_PROBE_MEMO.clear()
    p = tmp_path / "one.ome.tif"
    tifffile.imwrite(
        p,
        np.zeros((4, 4), np.uint8),
        description="<OME><Image/></OME>",
        metadata=None,
    )
    OmeTiffAdapter.claim(ClaimContext(p), DiscoveryState())
    assert len(ome_tiff_mod._OME_PROBE_MEMO) == 0
    OmeTiffAdapter.claim(ClaimContext(p, monitored=True), DiscoveryState())
    assert len(ome_tiff_mod._OME_PROBE_MEMO) == 1
    ome_tiff_mod._OME_PROBE_MEMO.clear()
