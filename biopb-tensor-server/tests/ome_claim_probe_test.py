"""The OME-TIFF claim's content probe: file names out of the OME-XML, without
building a tree of it, and the first-IFD description read that feeds it."""

import numpy as np
import pytest
import tifffile
from biopb_tensor_server.adapters.ome_tiff import (
    OmeTiffAdapter,
    _existing_files,
    _files_from_ome_xml,
    _files_from_ome_xml_parsed,
    _probe_ome_files,
)
from biopb_tensor_server.core.discovery import ClaimContext, DiscoveryState


def _xml(*refs, prefix="", quote='"'):
    tiff_data = "".join(
        f"<{prefix}TiffData IFD='{i}'><{prefix}UUID FileName={quote}{name}{quote}>"
        f"urn:uuid:0</{prefix}UUID></{prefix}TiffData>"
        for i, name in enumerate(refs)
    )
    return f"<OME><Image><Pixels>{tiff_data}</Pixels></Image></OME>".encode()


@pytest.mark.parametrize(
    "xml, expected",
    [
        pytest.param(
            _xml("a.ome.tif", "b.ome.tif", "a.ome.tif"),
            ("a.ome.tif", "b.ome.tif"),
            id="distinct",
        ),
        pytest.param(_xml("z.tif", "a.tif"), ("z.tif", "a.tif"), id="document-order"),
        pytest.param(_xml("x.tif", prefix="ome:"), ("x.tif",), id="namespaced"),
        pytest.param(_xml("x.tif", quote="'"), ("x.tif",), id="single-quoted"),
        pytest.param(_xml("a&amp;b.tif"), ("a&b.tif",), id="entity"),
        pytest.param(
            b'<OME><TiffData><UUID FileName = "b.tif">u</UUID></TiffData></OME>',
            ("b.tif",),
            id="spaced",
        ),
        pytest.param(
            b"<OME><TiffData><UUID FileName ='b.tif'>u</UUID></TiffData></OME>",
            ("b.tif",),
            id="spaced-single-quoted",
        ),
        pytest.param(
            b'<OME><TiffData><UUID FileName="a.tif">u</UUID></TiffData>'
            b"<Key>FileName</Key></OME>",
            ("a.tif",),
            id="token-elsewhere",
        ),
        pytest.param(b"<OME><Image/></OME>", (), id="no-references"),
        pytest.param(b"<OME><broken", (), id="malformed"),
    ],
)
def test_files_from_ome_xml(xml, expected):
    assert _files_from_ome_xml(xml) == expected


def test_scan_agrees_with_the_parser():
    xml = _xml(*(f"s_{i % 7}.ome.tif" for i in range(500)))
    assert _files_from_ome_xml(xml) == _files_from_ome_xml_parsed(xml)


def _write(path, description, **kw):
    tifffile.imwrite(
        path, np.zeros((4, 4), np.uint8), description=description, metadata=None, **kw
    )


@pytest.mark.parametrize("kw", [{}, {"bigtiff": True}, {"byteorder": ">"}])
def test_probe_reads_ome_xml_in_every_tiff_flavor(tmp_path, kw):
    p = tmp_path / "s.ome.tif"
    _write(p, _xml("s.ome.tif", "s_1.ome.tif").decode(), **kw)
    assert _probe_ome_files(p) == ("s.ome.tif", "s_1.ome.tif")


def test_probe_single_file_ome_has_no_references(tmp_path):
    p = tmp_path / "one.ome.tif"
    _write(p, "<OME><Image/></OME>")
    assert _probe_ome_files(p) == ()


# Short ids: pytest puts the id in an environment variable, which Windows caps.
@pytest.mark.parametrize(
    "description",
    [
        pytest.param(None, id="none"),
        pytest.param("plain text", id="text"),
        pytest.param("x" * 100_000, id="long"),
    ],
)
def test_probe_none_without_ome_xml(tmp_path, description):
    p = tmp_path / "plain.tif"
    _write(p, description)
    assert _probe_ome_files(p) is None


def test_probe_imagej_tiff_is_not_ome(tmp_path):
    p = tmp_path / "ij.tif"
    tifffile.imwrite(p, np.zeros((2, 4, 4), np.uint8), imagej=True)
    assert _probe_ome_files(p) is None


def test_probe_tolerates_junk(tmp_path):
    junk = tmp_path / "junk.tif"
    junk.write_bytes(b"not a tiff at all")
    assert _probe_ome_files(junk) is None
    assert _probe_ome_files(tmp_path / "missing.tif") is None
    cut = tmp_path / "cut.tif"
    _write(cut, _xml("a.tif").decode())
    cut.write_bytes(cut.read_bytes()[:20])
    assert _probe_ome_files(cut) is None


def test_existing_files_keeps_order_and_drops_missing(tmp_path):
    (tmp_path / "b.tif").write_bytes(b"")
    (tmp_path / "a.tif").write_bytes(b"")
    got = _existing_files(("b.tif", "gone.tif", "a.tif"), tmp_path)
    assert got == [tmp_path / "b.tif", tmp_path / "a.tif"]


def test_claim_groups_a_multi_file_set_under_its_first_file(tmp_path):
    names = ("s_MMStack.ome.tif", "s_MMStack_1.ome.tif")
    for n in names:
        _write(tmp_path / n, _xml(*names).decode())
    state = DiscoveryState()
    claim = OmeTiffAdapter.claim(ClaimContext(tmp_path / names[1]), state)
    assert claim.primary_path == str(tmp_path / names[0])
    assert state.is_path_claimed(str(tmp_path / names[1]))


def test_claim_single_file_claims_itself(tmp_path):
    p = tmp_path / "one.ome.tif"
    _write(p, "<OME><Image/></OME>")
    claim = OmeTiffAdapter.claim(ClaimContext(p), DiscoveryState())
    assert claim.primary_path == str(p)


def test_claim_declines_a_plain_tiff(tmp_path):
    p = tmp_path / "plain.tif"
    _write(p, None)
    assert OmeTiffAdapter.claim(ClaimContext(p), DiscoveryState()) is None
