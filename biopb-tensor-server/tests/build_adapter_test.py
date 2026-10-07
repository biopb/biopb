"""A fresh registration serves the adapter rebuilt from its own catalog record."""

import numpy as np
import pytest
import tifffile
from biopb_tensor_server.adapters import OmeTiffAdapter
from biopb_tensor_server.core.adapter_base import build_adapter
from biopb_tensor_server.core.config import SourceConfig


class _Parsed:
    def __init__(self, resolved=True, payload=None):
        self._resolved = resolved
        self._payload = payload

    def is_resolved(self):
        return self._resolved

    def catalog_payload(self):
        return self._payload

    def get_metadata(self):
        return {"k": np.int16(3)}


def _stub(parsed, rebuild=None):
    class Stub:
        @classmethod
        def create_from_config(cls, source, credentials_config=None):
            return parsed

    if rebuild is not None:
        Stub.create_from_payload = classmethod(rebuild)
    return Stub


SOURCE = SourceConfig(url="/x.dat", source_id="s")


def test_an_adapter_without_a_payload_path_is_served_as_parsed():
    parsed = _Parsed(payload={"a": 1})
    assert build_adapter(_stub(parsed), SOURCE) is parsed


def test_an_unresolved_source_is_served_as_parsed():
    parsed = _Parsed(resolved=False, payload={"a": 1})
    rebuilt = _stub(parsed, lambda cls, *a: pytest.fail("rebuilt"))
    assert build_adapter(rebuilt, SOURCE) is parsed


def test_no_payload_is_served_as_parsed():
    parsed = _Parsed(payload=None)
    rebuilt = _stub(parsed, lambda cls, *a: pytest.fail("rebuilt"))
    assert build_adapter(rebuilt, SOURCE) is parsed


def test_a_declined_rebuild_is_served_as_parsed():
    parsed = _Parsed(payload={"a": 1})
    assert build_adapter(_stub(parsed, lambda cls, *a: None), SOURCE) is parsed


def test_the_rebuild_is_fed_the_json_normalized_record():
    seen = []

    def rebuild(cls, source, payload, metadata, credentials_config):
        seen.append((payload, metadata))
        return "rebuilt"

    parsed = _Parsed(payload={"shape": (2, 3)})
    assert build_adapter(_stub(parsed, rebuild), SOURCE) == "rebuilt"
    # tuples and numpy scalars as the row stores them: plain JSON values
    assert seen == [({"shape": [2, 3]}, {"k": 3})]
    assert type(seen[0][1]["k"]) is int


def test_an_incomplete_payload_fails_the_first_registration():
    def rebuild(cls, source, payload, metadata, credentials_config):
        return payload["missing"]

    with pytest.raises(KeyError):
        build_adapter(_stub(_Parsed(payload={"a": 1}), rebuild), SOURCE)


def test_an_ome_tiff_is_served_from_its_payload_with_the_parsed_answers(tmp_path):
    path = tmp_path / "a.ome.tif"
    data = np.arange(3 * 16 * 16, dtype=np.uint16).reshape(3, 16, 16)
    tifffile.imwrite(path, data, ome=True, metadata={"axes": "ZYX"})
    source = SourceConfig(url=str(path), source_id="s0")

    parsed = OmeTiffAdapter.create_from_config(source)
    built = build_adapter(OmeTiffAdapter, source)

    assert built._hydrated_payload is not None  # not the parsed adapter
    assert [d.SerializeToString() for d in built.list_tensor_descriptors()] == [
        d.SerializeToString() for d in parsed.list_tensor_descriptors()
    ]
    assert built.get_metadata() == parsed.get_metadata()
    built.close()
