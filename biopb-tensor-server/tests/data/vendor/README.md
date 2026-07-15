# Vendor-format read-test fixtures

Real sample files for the fixture-gated vendor read tests in
`tests/adapter_integration_test.py::TestBioioReadPath`.

CZI, ND2 and LIF have no Python library that writes a file their `bioio-*`
plugin reads back faithfully, so their read paths can only be exercised against
a genuine sample (unlike DeltaVision `.dv`, which `mrc.imwrite` synthesizes and
the DV test covers self-contained). Drop a **tiny** real file here and the
matching test claims it, reads it through its adapter, and asserts
descriptor/data consistency; with no file present the test self-skips.

| Extension | Adapter        | Plugin       |
|-----------|----------------|--------------|
| `.czi`    | `ZeissAdapter` | `bioio-czi`  |
| `.nd2`    | `NikonAdapter` | `bioio-nd2`  |
| `.lif`    | `LeicaAdapter` | `bioio-lif`  |

The first matching file (alphabetical) per extension is used. Point
`BIOPB_TEST_VENDOR_DIR` at another directory to override this location (e.g. a
CI step that fetches samples out-of-tree instead of committing binaries).
