#!/bin/bash
#
# biopb bootstrap: what https://biopb.org/install.sh serves.
#   curl -fsSL https://biopb.org/install.sh | bash
#
# It installs nothing itself. It checks the machine can run an installer, picks
# a release, downloads that release's own install.sh asset and runs it, so the
# installer that runs is always the one written for the release it installs. It
# carries no release logic of its own and so does not change from release to
# release.
#
# Arguments go to the installer (`... | bash -s -- --uninstall`), and so does
# the environment. The release is chosen the way the installer chooses it:
#   BIOPB_INSTALL_VERSION=X.Y.Z  that exact release (release-vX.Y.Z, vX.Y.Z also work)
#   BIOPB_INSTALL_RC=1           the latest release candidate
#   otherwise                    the latest stable release
#
# Requirements: bash, curl, tar

_err() { printf 'ERROR: %s\n' "$*" >&2; }

RELEASE_REPO="biopb/biopb"
TAG_PREFIX="release-v"

_check_requirements() {
    case "$(uname -s)" in
        Linux|Darwin) ;;
        MINGW*|MSYS*|CYGWIN*)
            _err "This is the Linux/macOS installer. On Windows use PowerShell:"
            echo "  irm https://biopb.org/install.ps1 | iex" >&2
            return 1 ;;
        *)
            _err "Unsupported platform: $(uname -s)"
            return 1 ;;
    esac
    local tool
    for tool in curl tar; do
        command -v "$tool" >/dev/null 2>&1 || { _err "$tool is required but not installed."; return 1; }
    done
}

# Print the tag of the release to install.
_resolve_tag() {
    local version="${BIOPB_INSTALL_VERSION:-}"
    if [ -n "$version" ]; then
        case "$version" in
            "$TAG_PREFIX"*) printf '%s\n' "$version" ;;
            v*)             printf '%s\n' "${TAG_PREFIX}${version#v}" ;;
            *)              printf '%s\n' "${TAG_PREFIX}${version}" ;;
        esac
        return 0
    fi
    # The repo hosts several release lines, so /releases/latest is not ours:
    # take the newest release-v* tag, a clean X.Y.Z unless candidates are wanted.
    local re="^${TAG_PREFIX}[0-9]+\.[0-9]+\.[0-9]+$"
    case "${BIOPB_INSTALL_RC:-0}" in
        0|"") ;;
        *)    re="^${TAG_PREFIX}[0-9]+\.[0-9]+\.[0-9]+((a|b|rc)[0-9]+)?$" ;;
    esac
    local releases tag
    releases=$(curl -fsSL -H "Accept: application/vnd.github+json" \
        "https://api.github.com/repos/$RELEASE_REPO/releases?per_page=100") || return 1
    # `|| true`: no match makes grep exit 1, and the caller reports the empty tag.
    tag=$(printf '%s' "$releases" \
        | grep '"tag_name"' \
        | sed -E 's/.*"tag_name"[[:space:]]*:[[:space:]]*"([^"]+)".*/\1/' \
        | grep -E "$re" | head -1) || true
    [ -n "$tag" ] || return 1
    printf '%s\n' "$tag"
}

main() {
    _check_requirements || return 1

    local tag
    if ! tag=$(_resolve_tag); then
        _err "Could not find a biopb release to install (network, or GitHub rate limit)."
        echo "  Name one with BIOPB_INSTALL_VERSION=X.Y.Z" >&2
        return 1
    fi
    # The tag goes into a URL: refuse anything that is not a plain tag name.
    if ! printf '%s' "$tag" | grep -qE '^[A-Za-z0-9._+-]+$'; then
        _err "Unexpected release tag: $tag"
        return 1
    fi

    local dir url
    dir=$(mktemp -d) || return 1
    url="https://github.com/$RELEASE_REPO/releases/download/$tag/install.sh"
    printf 'Fetching the %s installer...\n' "$tag"
    if ! curl -fsSL "$url" -o "$dir/install.sh"; then
        _err "Could not download $url"
        echo "  The release may not exist, or may predate its own installer." >&2
        rm -rf "$dir"
        return 1
    fi
    if ! bash -n "$dir/install.sh" 2>/dev/null; then
        _err "The downloaded installer is not a valid script."
        rm -rf "$dir"
        return 1
    fi

    # stdin is null, as it is after a `curl | bash`: the installer prompts on /dev/tty.
    local status=0
    bash "$dir/install.sh" "$@" </dev/null || status=$?
    rm -rf "$dir"
    return "$status"
}

# Last, so a download cut off mid-transfer defines functions and does nothing.
# BIOPB_INSTALL_LIB=1 lets the tests source this file for its helpers alone.
[ -n "${BIOPB_INSTALL_LIB:-}" ] || main "$@"
