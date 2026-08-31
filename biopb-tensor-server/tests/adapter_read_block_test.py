"""Every adapter classifies its read granularity, or the suite says so.

``read_block_shape`` is what the streamed scaled read floors its tile at
(biopb/biopb#640). Getting it wrong is silent -- every pixel stays
bit-identical, the read merely costs more -- so the failure mode this guards is
a *new* adapter inheriting the base ``None`` without anyone deciding it should.
Each class below is listed once, with the reason, and adding an adapter without
touching this file fails.
"""

import inspect

import pytest
from biopb_tensor_server.core.adapter_base import TensorAdapter

# Reads are quantized to a block the backend cannot read below.
QUANTIZED = {
    "ZarrAdapter": "the store's chunk",
    "OmeZarrAdapter": "inherits ZarrAdapter",
    "_HcsFieldAdapter": "inherits ZarrAdapter",
    "_QptiffLevelAdapter": "inherits ZarrAdapter (the level's tile grid)",
    "Hdf5Adapter": "the dataset's chunk, or None where contiguous",
    "OmeTiffAdapter": "one page: aszarr(chunkmode='page') decodes it whole",
    "_TifffileAdapterBase": "inherits OmeTiffAdapter",
    "TiffAdapter": "inherits OmeTiffAdapter",
    "LsmAdapter": "inherits OmeTiffAdapter",
    "TiffSequenceAdapter": "one strile, reported as tile or page",
    "MicroManagerLegacyAdapter": "one strile, reported as tile or page",
    "DicomAdapter": "one frame: pixel_array decodes at least that",
    "DicomSeriesAdapter": "one slice file",
    "NdTiffAdapter": "the dask block behind the plane",
    "EmdAdapter": "the dask block",
    "_BioioAdapterBase": "the dask block a slice materialises",
    "ZeissAdapter": "inherits _BioioAdapterBase",
    "LeicaAdapter": "inherits _BioioAdapterBase",
    "DvAdapter": "inherits _BioioAdapterBase",
    "OlympusAdapter": "inherits _BioioAdapterBase",
    "BioformatsAdapter": "inherits _BioioAdapterBase",
    "AicsImageIoAdapter": "inherits _BioioAdapterBase",
}

# No part of a read is wasted: a crop costs its own pages and nothing more.
UNQUANTIZED = {
    "MrcAdapter": "one np.memmap; indexing computes byte offsets",
    "NiftiAdapter": "nibabel dataobj slicing",
    "NikonAdapter": "nd2.read_frame returns an mmap view, then crops",
    "CziAdapter": "a libCZI ROI composes only the subblocks it touches",
    "RemoteTensorAdapter": "forwards arbitrary bounds upstream",
    "CachedSourceAdapter": "get_data raises; served by chunk_id only",
    "QptiffAdapter": "source-level; serves levels through _QptiffLevelAdapter",
    "TensorAdapter": "the base class itself",
}


def _adapters():
    import biopb_tensor_server.adapters as pkg

    seen = {}
    for name in dir(pkg):
        obj = getattr(pkg, name)
        if inspect.isclass(obj) and issubclass(obj, TensorAdapter):
            seen[obj.__name__] = obj
    for module in pkg.__dict__.values():
        if inspect.ismodule(module):
            for obj in vars(module).values():
                if inspect.isclass(obj) and issubclass(obj, TensorAdapter):
                    seen[obj.__name__] = obj
    return seen


def test_every_adapter_is_classified():
    """A new adapter must be listed, not silently inherit ``None``."""
    classified = set(QUANTIZED) | set(UNQUANTIZED)
    found = set(_adapters())
    assert found - classified == set(), (
        f"unclassified adapters: {sorted(found - classified)} -- declare "
        "read_block_shape and add them to QUANTIZED, or add them to UNQUANTIZED "
        "with the reason their reads waste nothing"
    )


def test_the_lists_are_disjoint_and_real():
    assert not (set(QUANTIZED) & set(UNQUANTIZED))
    stale = (set(QUANTIZED) | set(UNQUANTIZED)) - set(_adapters()) - {"TensorAdapter"}
    assert not stale, f"listed but no longer present: {sorted(stale)}"


@pytest.mark.parametrize("name", sorted(UNQUANTIZED))
def test_unquantized_adapters_report_none(name):
    """Inherited or overridden, the answer must be None -- not merely absent."""
    cls = _adapters().get(name)
    if cls is None:
        pytest.skip(f"{name} not importable in this environment")
    prop = getattr(cls, "read_block_shape", None)
    assert isinstance(prop, property)
    source = inspect.getsource(prop.fget)
    assert "return None" in source, f"{name} must report no quantization"


@pytest.mark.parametrize("name", sorted(QUANTIZED))
def test_quantized_adapters_declare_a_block(name):
    """Declared on the class or inherited from one that declares it."""
    cls = _adapters().get(name)
    if cls is None:
        pytest.skip(f"{name} not importable in this environment")
    prop = getattr(cls, "read_block_shape", None)
    assert isinstance(prop, property), f"{name} has no read_block_shape"
    owner = next(
        (k for k in cls.__mro__ if "read_block_shape" in vars(k)),
        None,
    )
    assert owner is not None and owner is not TensorAdapter, (
        f"{name} inherits the base None but is listed as quantized"
    )
