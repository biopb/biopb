"""nd2 and czi rebuilt from their payload serve what the parsed adapter serves."""

import pytest
from biopb_tensor_server.core.config import SourceConfig

from tests.payload_equivalence import assert_hydrates_equivalently


class TestNd2:
    def test_a_rebuilt_adapter_is_the_parsed_one_without_opening_the_file(
        self, tmp_path, monkeypatch
    ):
        pytest.importorskip("nd2")
        import nd2
        from biopb_tensor_server.adapters import Nd2Adapter

        from tests.nd2_adapter_test import _install_fake

        _install_fake(monkeypatch)
        path = tmp_path / "img.nd2"
        path.write_bytes(b"\x00")
        source = SourceConfig(url=str(path), type="nd2", source_id="src")
        parsed = Nd2Adapter.create_from_config(source)

        assert_hydrates_equivalently(
            parsed, source, monkeypatch=monkeypatch, opens=[(nd2, "ND2File")]
        )


class TestCzi:
    def test_a_rebuilt_adapter_is_the_parsed_one(self, tmp_path):
        pytest.importorskip("pylibCZIrw")
        from biopb_tensor_server.adapters import CziAdapter
        from biopb_tensor_server.fixtures import create_zeiss_czi

        path, _ = create_zeiss_czi(
            str(tmp_path), n_t=2, n_c=2, n_z=3, image_shape=(24, 32)
        )
        source = SourceConfig(url=str(path), type="czi", source_id="src")
        parsed = CziAdapter.create_from_config(source)

        assert_hydrates_equivalently(parsed, source)
