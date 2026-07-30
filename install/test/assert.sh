#!/bin/bash
#
# Post-condition checks for one install.sh scenario, run INSIDE the container.
# `run.sh --assert <scenario>` mounts this at /assert.sh and runs it instead of
# dropping you at the interactive shell.
#
# This is the only layer that runs the installer end to end. The cheaper layers
# (`bash -n`, shellcheck, the pytest unit tests) answer "did this diff break a
# parser"; this answers "does the whole thing work", including the one behaviour
# no unit test can reach: `uv tool install --force` rebuilds the environment from
# the arguments it is handed, so a package the user added stays only because the
# installer replays it. That is a property of uv, not of our code, and the only
# way to check it is to install twice and look.
#
# Minutes per scenario, so this is nightly / pre-release, not per-PR. See #653.
#
# Not `set -e`: a failed check must be reported alongside the others, not turn
# the run into a single mystery exit code. Exit status is the failure count.

set -uo pipefail

SCENARIO="${1:-clean}"
export PATH="$HOME/.local/bin:$PATH"

# A package the biopb deployment does not depend on, so finding it in the
# environment afterwards can only mean the extras replay put it there. Tiny, pure
# Python, no dependencies of its own -- the check should time out on nothing.
MARKER_PKG="cowsay"
# A requirement that cannot resolve at any index, for the fail-soft path.
BOGUS_PKG="biopb-not-a-real-package==99.99.99"

CONFIG_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/biopb"
EXTRAS_FILE="$CONFIG_DIR/extra-packages.txt"

FAILURES=0
_pass() { printf '  \033[0;32mok\033[0m    %s\n' "$1"; }
_fail() { printf '  \033[0;31mFAIL\033[0m  %s\n' "$1"; FAILURES=$((FAILURES + 1)); }
_phase() { printf '\n\033[1m== %s\033[0m\n' "$1"; }

# check <description> <command...>
check() {
    local what="$1"; shift
    if "$@" >/dev/null 2>&1; then _pass "$what"; else _fail "$what"; fi
}

# The interpreter of the shared uv tool environment -- the one the napari kernel
# runs, and the one a replayed extra package has to land in. Asked of uv rather
# than hardcoded, since UV_TOOL_DIR is configurable.
tool_python() {
    local dir
    dir=$(uv tool dir 2>/dev/null) || return 1
    printf '%s/biopb/bin/python' "$dir"
}

# check_marker <description> -- is the seeded extra package importable in that
# environment? Reports through check() rather than returning a status, so a
# missing tool dir reads as a failed check instead of a silent skip.
check_marker() {
    local py
    py=$(tool_python) || { _fail "$1 (no uv tool dir)"; return; }
    check "$1" "$py" -c "import $MARKER_PKG"
}

run_install() {
    # BIOPB_NONINTERACTIVE: no prompts, there is no tty here.
    # BIOPB_NO_SERVER_START: the control plane is not what this is checking, and a
    #   daemon left running holds the tool dir open against the next --force.
    # BIOPB_INSTALL_SAMPLES=0: the sample bundle is a release asset download worth
    #   minutes, and every scenario below points the config somewhere explicit.
    env BIOPB_NONINTERACTIVE=1 BIOPB_NO_SERVER_START=1 BIOPB_INSTALL_SAMPLES=0 \
        "$@" bash /install.sh 2>&1
}

# Per-scenario install environment. An array, not a string, so "no override"
# stays an empty word list rather than something the shell has to be told not to
# split.
#
# BIOPB_DATA_DIR=/root for the three images that seed TIFFs there and bake no
# config -- it keeps the install off the release-asset download path. The other
# two bake a config, which always wins over BIOPB_DATA_DIR, so setting it there
# would only be misleading. bioformats instead needs the extra that its whole
# scenario is about; without it /verify_bioformats.sh has nothing to verify.
INSTALL_ENV=()
case "$SCENARIO" in
    clean|uv-preinstalled|old-python) INSTALL_ENV=(BIOPB_DATA_DIR=/root) ;;
    bioformats)                       INSTALL_ENV=(BIOPB_INSTALL_BIOFORMATS=1) ;;
esac

# ---------------------------------------------------------------------------

_phase "1/5  Install, carrying a user extra package"

mkdir -p "$CONFIG_DIR"
cat > "$EXTRAS_FILE" <<EOF
# seeded by install/test/assert.sh
$MARKER_PKG
EOF

if run_install "${INSTALL_ENV[@]}" > /tmp/install-1.log; then
    _pass "first install exits 0"
else
    _fail "first install exits 0 (see /tmp/install-1.log)"
    tail -30 /tmp/install-1.log
fi

_phase "2/5  Post-conditions"

for shim in biopb biopb-mcp biopb-tensor-server biopb-control; do
    # The shim file, not `command -v`: whether a FUTURE shell finds it depends on
    # uv's rc-file editing, which correctly does nothing in the uv-preinstalled
    # scenario (the dir is already on PATH there). What must hold everywhere is
    # that the executable exists and runs.
    check "shim installed: $shim" test -x "$HOME/.local/bin/$shim"
done
check "biopb-tensor-server runs" biopb-tensor-server version

CONFIG_JSON="$CONFIG_DIR/biopb.json"
check "config written: biopb.json" test -f "$CONFIG_JSON"
check "config is valid JSON with a source" python3 -c "
import json, sys
cfg = json.load(open('$CONFIG_JSON'))
assert cfg['sources'], 'sources is empty'
assert cfg['sources'][0]['url'], 'source has no url'
"
check_marker "extra package reached the environment"

_phase "3/5  Rerun -- the --force rebuild"

# The whole reason extra-packages.txt exists. uv rebuilds the tool env from the
# arguments of THIS invocation, so anything not replayed is gone; before #648 the
# installer did not replay it and the loss surfaced later as an import that used
# to work.
if run_install "${INSTALL_ENV[@]}" > /tmp/install-2.log; then
    _pass "rerun exits 0 (idempotent)"
else
    _fail "rerun exits 0 (see /tmp/install-2.log)"
    tail -30 /tmp/install-2.log
fi
check_marker "extra package SURVIVED the rebuild"
check "config still present after rerun" test -f "$CONFIG_JSON"

_phase "4/5  Scenario-specific"

case "$SCENARIO" in
    old-python)
        # The image ships Python 3.7 as python3. The point of the scenario is that
        # the installer rejects it and uses a uv-managed interpreter instead.
        py=$(tool_python)
        check "tool env is on a modern Python, not the system 3.7" \
            "$py" -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)"
        ;;
    rerun)
        # The image stages a pre-#34 legacy TOML. An upgrade keeps the user's
        # settings but converts the file, so the server stops warning about the
        # deprecated format on every startup.
        check "legacy TOML migrated to JSON" test -f "$CONFIG_JSON"
        check "legacy TOML backed up, not deleted" test -f "$CONFIG_DIR/biopb.toml.bak"
        ;;
    bioformats)
        check "Bio-Formats / ZVI support works" bash /verify_bioformats.sh
        ;;
    *)
        printf '  (none for %s)\n' "$SCENARIO"
        ;;
esac

_phase "5/5  Fail-soft: a bad extra must not block the upgrade"

# A user requirement joins the same resolve as the release's own pins, so one
# unresolvable line would otherwise block an upgrade over a package the
# deployment does not need. The installer retries without the extras, and only
# then names them -- an install that failed for an unrelated reason (no network,
# full disk) fails the retry too and must NOT be blamed on the extras.
printf '%s\n' "$BOGUS_PKG" >> "$EXTRAS_FILE"
if run_install "${INSTALL_ENV[@]}" > /tmp/install-3.log; then
    _pass "install with an unresolvable extra still exits 0"
else
    _fail "install with an unresolvable extra still exits 0 (see /tmp/install-3.log)"
    tail -30 /tmp/install-3.log
fi
check "the deployment landed anyway" biopb-tensor-server version
check "the user is told which file to fix" \
    grep -q "Could not resolve your extra packages" /tmp/install-3.log
check "the bad line is named" grep -qF "$BOGUS_PKG" /tmp/install-3.log

# ---------------------------------------------------------------------------

printf '\n'
if [ "$FAILURES" -eq 0 ]; then
    printf '\033[0;32mPASS\033[0m  %s: all checks passed\n' "$SCENARIO"
else
    printf '\033[0;31mFAIL\033[0m  %s: %s check(s) failed\n' "$SCENARIO" "$FAILURES"
fi
exit "$FAILURES"
