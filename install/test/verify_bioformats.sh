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

# 2. The Bio-Formats jar shipped with the bioformats extra.
try:
    import bioformats_jar  # noqa: F401
    print("[PASS] bioformats_jar importable")
except Exception as e:
    failures.append("bioformats_jar import: %r" % (e,))
    print("[FAIL] bioformats_jar import:", repr(e))
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

# 4. Read a real ZVI if one is present under /data (true end-to-end).
zvis = sorted(glob.glob("/data/**/*.zvi", recursive=True))
if zvis:
    path = zvis[0]
    try:
        from aicsimageio import AICSImage
        from aicsimageio.readers.bioformats_reader import BioformatsReader
        img = AICSImage(path, reader=BioformatsReader)
        arr = img.dask_data
        print("[PASS] read %s: shape=%s dtype=%s dims=%s"
              % (path, arr.shape, arr.dtype, img.dims.order))
    except Exception as e:
        failures.append("ZVI read (%s): %r" % (path, e))
        print("[FAIL] ZVI read (%s):" % path, repr(e))
else:
    print("[SKIP] no .zvi under /data -- mount one "
          "(BIOPB_TEST_DATA=/dir ./run.sh bioformats) to test the read.")

print()
if failures:
    print("VERIFY FAILED (%d issue(s))" % len(failures))
    sys.exit(1)
print("VERIFY OK")
PYEOF

"$TOOL_PY" /tmp/verify_bioformats.py
