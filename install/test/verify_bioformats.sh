#!/bin/bash
# Verify the Bio-Formats extra installed by install.sh works end-to-end.
set -uo pipefail

# Locate the python of the biopb-tensor-server uv tool environment, where the
# [bioformats] extra (aicsimageio[bioformats] + scyjava + cjdk) was installed.
export PATH="$HOME/.local/bin:$PATH"
# Find it by what it CONTAINS, not by name: install.sh puts everything in one
# uv tool env (`biopb`, with biopb-tensor-server as a --with dependency and
# --with-executables-from), so there is no tool named biopb-tensor-server. This
# script looked for one and could never find it -- it had never run in CI to say
# so. Probing each env keeps it correct if that layout changes again.
TOOLS_DIR="$(uv tool dir 2>/dev/null || echo "$HOME/.local/share/uv/tools")"
TOOL_PY=""
for candidate in "$TOOLS_DIR"/*/bin/python; do
    [ -x "$candidate" ] || continue
    if "$candidate" -c 'import biopb_tensor_server' 2>/dev/null; then
        TOOL_PY="$candidate"
        break
    fi
done
if [ -z "$TOOL_PY" ]; then
    echo "ERROR: no uv tool environment provides biopb_tensor_server."
    echo "Searched: $TOOLS_DIR/*/bin/python"
    echo "Run 'BIOPB_INSTALL_BIOFORMATS=1 bash /install.sh' first."
    exit 2
fi
echo "Using interpreter: $TOOL_PY"

cat > /tmp/verify_bioformats.py << 'PYEOF'
import glob
import os
import sys

failures = []

# 1. The new adapter is importable and registered for .zvi.
try:
    from biopb_tensor_server.adapters import (
        BioformatsAdapter,
        get_default_registry,
    )
    assert BioformatsAdapter is not None, "BioformatsAdapter is None (aicsimageio missing?)"
    assert ".zvi" in BioformatsAdapter.BIOFORMATS_ONLY_EXTENSIONS
    get_default_registry()
    print("[PASS] BioformatsAdapter registered; exts =",
          BioformatsAdapter.BIOFORMATS_ONLY_EXTENSIONS)
except Exception as e:
    failures.append("adapter registration: %r" % (e,))
    print("[FAIL] adapter registration:", repr(e))

# 2. The reader plugin shipped with the bioformats extra. This is
#    bioio_bioformats, NOT the aicsimageio-era bioformats_jar package the bioio
#    migration removed -- the jar itself is fetched lazily by scyjava/cjdk on
#    first read (see adapters/bioio.py), so there is no jar package to import.
try:
    import bioio_bioformats  # noqa: F401
    print("[PASS] bioio_bioformats importable")
except Exception as e:
    failures.append("bioio_bioformats import: %r" % (e,))
    print("[FAIL] bioio_bioformats import:", repr(e))
    print("       -> the 'bioformats' component was not installed; rerun "
          "install.sh and tick it.")

# 3. The JVM starts. No system Java is in this image, so a PASS here proves
#    scyjava/cjdk auto-fetched a JDK into the user cache -- the linchpin.
try:
    import scyjava
    scyjava.start_jvm()
    System = scyjava.jimport("java.lang.System")
    print("[PASS] JVM started (scyjava/cjdk auto-fetch OK); java.version =",
          System.getProperty("java.version"))
except Exception as e:
    failures.append("JVM start: %r" % (e,))
    print("[FAIL] JVM start:", repr(e))
    print("       -> if this says no JVM was found, scyjava did not auto-fetch "
          "a JDK; the install may need cjdk fetch enabled.")

# 3b. Which Bio-Formats the JVM loaded is the one the server pins. Unpinned, the
#    plugin asks Maven for RELEASE, which has been a release candidate. The
#    server sets BIOFORMATS_VERSION when its adapters are imported (check 1).
#    A release built before the pin has no such constant; that is reported, not
#    failed, since this scenario installs whatever release it is pointed at.
try:
    from biopb_tensor_server.adapters import bioio as _bioio
    pinned = getattr(_bioio, "BIOFORMATS_VERSION", None)
    loaded = str(scyjava.jimport("loci.formats.FormatTools").VERSION)
    if pinned is None:
        print("[SKIP] this release does not pin Bio-Formats; loaded", loaded)
    elif loaded == pinned:
        print("[PASS] Bio-Formats %s loaded, as pinned" % loaded)
    else:
        failures.append("Bio-Formats %s loaded, %s pinned" % (loaded, pinned))
        print("[FAIL] Bio-Formats %s loaded, %s pinned" % (loaded, pinned))
except Exception as e:
    failures.append("Bio-Formats version: %r" % (e,))
    print("[FAIL] Bio-Formats version:", repr(e))

# 4. Read a real file through Bio-Formats, under /data. A ZVI is what this
#    extra exists for, but any format Bio-Formats reads exercises the same path
#    -- the resolved jar, the JVM, and the reader -- and a small CZI is what CI
#    can fetch (there is no public ZVI). The reader is named explicitly: bioio
#    would otherwise pick its own CZI reader and never touch Java.
#
#    BIOPB_REQUIRE_SAMPLE=1 (CI) turns "no file" into a failure, so a fetch that
#    quietly did not happen cannot read as a pass.
samples = sorted(
    p
    for ext in ("zvi", "czi")
    for p in glob.glob("/data/**/*." + ext, recursive=True)
)
if samples:
    path = samples[0]
    try:
        import numpy as np
        from bioio import BioImage

        import bioio_bioformats

        img = BioImage(path, reader=bioio_bioformats.Reader)
        plane = np.asarray(img.get_image_dask_data("YX", T=0, C=0, Z=0).compute())
        assert plane.size > 0 and plane.any(), "the plane read back empty"
        print("[PASS] read %s through Bio-Formats: shape=%s dtype=%s dims=%s"
              % (path, img.shape, img.dtype, img.dims.order))
    except Exception as e:
        failures.append("Bio-Formats read (%s): %r" % (path, e))
        print("[FAIL] Bio-Formats read (%s):" % path, repr(e))
elif os.environ.get("BIOPB_REQUIRE_SAMPLE"):
    failures.append("no .zvi/.czi under /data, and BIOPB_REQUIRE_SAMPLE is set")
    print("[FAIL] no .zvi/.czi under /data, and BIOPB_REQUIRE_SAMPLE is set")
else:
    print("[SKIP] no .zvi/.czi under /data -- mount one "
          "(BIOPB_TEST_DATA=/dir ./run.sh bioformats) to test the read; the bioformats image bakes one in.")

print()
if failures:
    print("VERIFY FAILED (%d issue(s))" % len(failures))
    sys.exit(1)
print("VERIFY OK")
PYEOF

"$TOOL_PY" /tmp/verify_bioformats.py
