#!/bin/bash
#
# Biopb Tensor Server Installer
# Usage: curl -fsSL https://biopb.org/install.sh | bash
#
# Idempotent: rerun to upgrade to latest version
#
# By default this installs prebuilt wheels from the latest GitHub release.
# Set BIOPB_INSTALL_FROM_SOURCE=1 to instead build HEAD from a git checkout
# (the fast path for development); that mode additionally needs git + a compiler.
#
# Requirements: curl, tar (+ git for BIOPB_INSTALL_FROM_SOURCE=1)
#

# ANSI colors — suppressed when stdout is not a terminal
if [ -t 1 ]; then
    RED=$'\033[0;31m'; YELLOW=$'\033[0;33m'; GREEN=$'\033[0;32m'
    CYAN=$'\033[0;36m'; BOLD=$'\033[1m'; DIM=$'\033[2m'; RESET=$'\033[0m'
else
    RED=''; YELLOW=''; GREEN=''; CYAN=''; BOLD=''; DIM=''; RESET=''
fi

_step() { printf "\n${BOLD}%s${RESET}\n" "$*"; }
_ok()   { printf "  ${GREEN}%s${RESET}\n" "$*"; }
_info() { printf "  %s\n" "$*"; }
_warn() { printf "  ${YELLOW}WARNING:${RESET} %s\n" "$*"; }
_err()  { printf "${RED}ERROR:${RESET} %s\n" "$*" >&2; }
_note() { printf "  ${DIM}NOTE: %s${RESET}\n" "$*"; }
_cmd()  { printf "  ${CYAN}%s${RESET}\n" "$*"; }

# Yes/No prompt (default Yes). Usage: if _confirm "Question?"; then ...; fi
# Reads from /dev/tty so it works when the script is piped in from curl.
_confirm() {
    local reply
    printf "  ${BOLD}%s${RESET} [Y/n]: " "$1" >/dev/tty
    read -r reply </dev/tty || reply=""
    reply="${reply%$'\r'}"
    
    [[ -z "$reply" ]] && reply="y"
    [[ "$reply" =~ ^[Yy]([Ee][Ss])?$ ]]
}

# Interactive checkbox menu. Redraws in place; all output goes to /dev/tty.
# Usage: _checkbox "Label one" "Label two" ...
# Prints space-separated 1/0 values (one per label) to stdout.
# Items default to checked; set the global CHECKBOX_DEFAULTS array (one 1/0 per
# label) before calling to override individual initial states.
_checkbox() {
    local labels=("$@")
    local n=${#labels[@]}
    local sel=()
    for ((i=0; i<n; i++)); do
        sel+=("${CHECKBOX_DEFAULTS[$i]:-1}")
    done
    local first=1

    while true; do
        # On subsequent iterations move cursor up and clear to redraw in place.
        # Lines printed: 1 blank + 1 header + 1 blank + n items + 1 blank + 1 prompt + 1 (Enter newline) = n+5
        [ "$first" = "0" ] && printf "\033[%dA\033[J" "$((n + 5))" >/dev/tty
        first=0

        printf "\n  %sOptional components:%s\n\n" "$BOLD" "$RESET" >/dev/tty
        for ((i=0; i<n; i++)); do
            local mark
            if [ "${sel[$i]}" = "1" ]; then
                mark="${GREEN}[x]${RESET}"
            else
                mark="${DIM}[ ]${RESET}"
            fi
            printf "    %d. %b  %s\n" "$((i+1))" "$mark" "${labels[$i]}" >/dev/tty
        done
        printf "\n  ${DIM}Toggle [1-%d] or Enter to confirm:${RESET} " "$n" >/dev/tty

        local choice; read -r choice </dev/tty
        if [ -z "$choice" ]; then
            break
        elif [[ "$choice" =~ ^[0-9]+$ ]] && [ "$choice" -ge 1 ] && [ "$choice" -le "$n" ]; then
            local idx=$((choice - 1))
            [ "${sel[idx]}" = "1" ] && sel[idx]=0 || sel[idx]=1
        fi
    done

    printf '%s' "${sel[*]}"
}

# Prompt the user to choose a data directory.
# Usage: _pick_data_dir <varname>  — writes result into caller's variable (no subshell).
# All prompts go to /dev/tty. Requires PLATFORM to be set before calling.
_pick_data_dir() {
    # Caller passes the name of a variable to receive the result. We assign into
    # it with `printf -v` rather than a `local -n` nameref, because namerefs need
    # bash >= 4.3 and macOS ships bash 3.2.
    local _retvar_name=$1
    local candidates=() seen=()

    for dir in \
        "$HOME" \
        "$HOME/data" "$HOME/Data" \
        "$HOME/microscopy" "$HOME/Microscopy" \
        "$HOME/Documents" \
        /mnt/data /data; do
        [ -d "$dir" ] || continue
        local real; real=$(realpath "$dir" 2>/dev/null || echo "$dir")
        local dup=0
        for s in "${seen[@]+"${seen[@]}"}"; do [ "$s" = "$real" ] && dup=1 && break; done
        [ "$dup" = "0" ] && candidates+=("$dir") && seen+=("$real")
    done

    if [ "$PLATFORM" = "WSL" ]; then
        local win_user; win_user=$(cmd.exe /c "echo %USERNAME%" 2>/dev/null | tr -d '\r')
        local wsl_home="/mnt/c/Users/$win_user"
        if [ -n "$win_user" ] && [ -d "$wsl_home" ]; then
            local real; real=$(realpath "$wsl_home" 2>/dev/null || echo "$wsl_home")
            local dup=0
            for s in "${seen[@]+"${seen[@]}"}"; do [ "$s" = "$real" ] && dup=1 && break; done
            [ "$dup" = "0" ] && candidates+=("$wsl_home")
        fi
    fi

    local n=${#candidates[@]}
    local manual_opt=$((n + 1))
    local default_dir="${candidates[0]:-$HOME}"

    printf "\n  %sSelect your microscopy data directory:%s\n\n" "$BOLD" "$RESET" >/dev/tty
    local i=1
    for dir in "${candidates[@]+"${candidates[@]}"}"; do
        printf "    ${CYAN}%d)${RESET} %s\n" "$i" "$dir" >/dev/tty
        i=$((i + 1))
    done
    printf "    ${CYAN}%d)${RESET} Enter path manually\n\n" "$manual_opt" >/dev/tty
    printf "  %sChoice [1]:%s " "$DIM" "$RESET" >/dev/tty
    local choice; read -r choice </dev/tty
    choice="${choice:-1}"

    if [ "$choice" = "$manual_opt" ]; then
        local manual
        printf "  Path [%s]: " "$default_dir" >/dev/tty
        read -r manual </dev/tty
        manual="${manual%$'\r'}"
        printf -v "$_retvar_name" '%s' "${manual:-$default_dir}"
    elif [[ "$choice" =~ ^[0-9]+$ ]] && [ "$choice" -ge 1 ] && [ "$choice" -le "$n" ]; then
        printf -v "$_retvar_name" '%s' "${candidates[$((choice - 1))]}"
    else
        printf "  Invalid choice, using default\n" >/dev/tty
        printf -v "$_retvar_name" '%s' "$default_dir"
    fi
}

# Run a short inline Python script (reads the program from stdin, forwards args).
# Python is guaranteed at this point — the installer is a Python toolchain bootstrapped
# by uv — so we prefer a Python interpreter on PATH and fall back to uv's managed one
# (`--no-project` so a stray pyproject.toml in the cwd doesn't trigger a project sync).
_py() {
    if command -v python3 &>/dev/null; then
        python3 "$@"
    elif command -v python &>/dev/null; then
        python "$@"
    else
        uv run --no-project python "$@"
    fi
}

# Merge a biopb stdio MCP entry into JSON <file> under top-level key <parent>
# (e.g. "mcpServers" or "mcp"). <style> picks the entry shape: "stdio" for the
# standard {command,args} form — no "type" (Claude Desktop, Cursor, the canonical
# mcp.json) — and "opencode" for opencode's {type:"local", command:[...]} form.
# <command> and the trailing args are the `biopb-mcp` invocation the client
# spawns. Preserves all other content, creates the file (and parents) if absent,
# and writes atomically. JSON-escaping happens in Python, so a command path with
# spaces is safe. Uses Python (always present) instead of jq, which labs may not
# have. Returns non-zero and leaves the file untouched on any error.
_mcp_merge() {
    local file="$1" parent="$2" style="$3" command="$4"; shift 4
    mkdir -p "$(dirname "$file")"
    _py - "$file" "$parent" "$style" "$command" "$@" 2>/dev/null <<'PY'
import json, os, sys
path, parent, style, command, *cmd_args = sys.argv[1:]
try:
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
except FileNotFoundError:
    data = {}
if not isinstance(data, dict):
    sys.exit("%s: top-level JSON is not an object" % path)
if style == "opencode":
    entry = {"type": "local", "command": [command, *cmd_args], "enabled": True}
else:
    # Standard mcpServers stdio form (Claude Desktop, Cursor, the canonical
    # mcp.json): bare command+args, no "type" — that is the form every
    # mcpServers client accepts (a stray "type" trips stricter validators).
    entry = {"command": command, "args": cmd_args}
section = data.get(parent)
if not isinstance(section, dict):
    section = data[parent] = {}
section["biopb"] = entry
tmp = path + ".biopb.tmp"
with open(tmp, "w", encoding="utf-8") as fh:
    json.dump(data, fh, indent=2)
    fh.write("\n")
os.replace(tmp, path)
PY
}

# Write the opencode zen API key into opencode's credential store
# (~/.local/share/opencode/auth.json) under provider id "opencode" (the id zen
# uses). Merges into any existing auth.json so other providers survive; writes
# atomically with 0600 perms. Returns non-zero on any error.
_opencode_write_auth() {
    local key="$1"
    local auth_file="$HOME/.local/share/opencode/auth.json"
    mkdir -p "$(dirname "$auth_file")"
    _py - "$auth_file" "$key" 2>/dev/null <<'PY'
import json, os, sys
path, key = sys.argv[1], sys.argv[2]
try:
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
except FileNotFoundError:
    data = {}
if not isinstance(data, dict):
    data = {}
data["opencode"] = {"type": "api", "key": key}
tmp = path + ".biopb.tmp"
fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
with os.fdopen(fd, "w", encoding="utf-8") as fh:
    json.dump(data, fh, indent=2)
    fh.write("\n")
os.replace(tmp, path)
PY
}

# Merge the biopb server into a standard `mcpServers` JSON config (Claude Desktop,
# Cursor, …). biopb-mcp speaks stdio, so the client spawns the command itself.
# Usage: _mcp_json_merge <config-file> <command> <label> [args...]
_mcp_json_merge() {
    local file="$1" command="$2" label="$3"; shift 3
    if _mcp_merge "$file" "mcpServers" "stdio" "$command" "$@"; then
        _ok "$label: registered biopb ($file)"
    else
        _warn "$label: could not update $file — add biopb manually (see $CONFIG_DIR/mcp.json)"
    fi
}

# Detect AI agents (MCP clients) already on the system that biopb-mcp can plug into.
# Populates the DETECTED_AGENTS array with human-readable names; if it comes back empty,
# the MCP-configuration step offers to install one (opencode) before registering biopb.
_detect_agents() {
    DETECTED_AGENTS=()
    command -v claude &>/dev/null && DETECTED_AGENTS+=("Claude Code")
    [ -d "$HOME/.hermes" ] && DETECTED_AGENTS+=("Hermes")
    case "$PLATFORM" in
        macOS)     [ -d "$HOME/Library/Application Support/Claude" ] && DETECTED_AGENTS+=("Claude Desktop") ;;
        Linux|WSL) [ -d "$HOME/.config/Claude" ] && DETECTED_AGENTS+=("Claude Desktop") ;;
    esac
    [ -d "$HOME/.cursor" ] && DETECTED_AGENTS+=("Cursor")
    { command -v opencode &>/dev/null || [ -d "$HOME/.config/opencode" ]; } && DETECTED_AGENTS+=("opencode")
    return 0
}

# Print the terminal command the user should run to launch their AI agent, or
# nothing if no CLI agent is available. Used by the final "next steps" message so
# we recommend whatever the user actually has rather than hardcoding one agent.
_agent_launch_cmd() {
    if command -v claude &>/dev/null; then
        echo "claude"
    elif command -v opencode &>/dev/null || [ -d "$HOME/.config/opencode" ]; then
        echo "opencode"
    elif command -v cursor &>/dev/null || [ -d "$HOME/.cursor" ]; then
        echo "cursor"
    fi
}

# Install the opencode agent (https://opencode.ai) and walk the user through
# authenticating it against opencode zen. Called when no agent is detected and the
# user opts in. Best-effort: a failed install/login warns but never aborts the run.
_install_opencode() {
    _info "Installing opencode..."
    if curl -fsSL https://opencode.ai/install | bash; then
        _ok "opencode installed"
    else
        _warn "opencode install failed — install it manually from https://opencode.ai, then rerun this script"
        return 0
    fi

    # opencode's installer drops the binary in ~/.opencode/bin and edits shell rc files
    # (not this process), so expose it here for the auth + registration steps that follow.
    export PATH="$HOME/.opencode/bin:$PATH"

    # opencode's installer only appends to ~/.bashrc, so on macOS (zsh) and other
    # non-bash shells ~/.opencode/bin never lands on PATH and `opencode` is missing
    # from new terminals. Symlink it into ~/.local/bin, which uv keeps on PATH for
    # every shell (see _ensure_local_bin_on_path), so the command is always found.
    if [ -x "$HOME/.opencode/bin/opencode" ]; then
        mkdir -p "$HOME/.local/bin"
        ln -sf "$HOME/.opencode/bin/opencode" "$HOME/.local/bin/opencode"
    fi

    # We prompt for the key and write opencode's credential file ourselves rather
    # than running `opencode auth login`: that command is a full-screen TUI which
    # does not receive keyboard input when driven from a piped (curl|bash)
    # installer, so the user could never finish it.
    echo ""
    _info "opencode needs an API key to talk to an LLM (free with opencode zen):"
    _info "  1. Open ${BOLD}https://opencode.ai/auth${RESET}, sign in, create a new API key, and copy it."
    printf "  ${DIM}Paste the API key (or press Enter to skip): ${RESET}" >/dev/tty
    local _key=""
    read -rs _key </dev/tty || _key=""
    _key="${_key%$'\r'}"
    printf "\n" >/dev/tty   # read -s ate the user's newline

    if [ -n "$_key" ]; then
        if _opencode_write_auth "$_key"; then
            _ok "opencode authenticated (opencode zen)"
        else
            _warn "Couldn't write opencode credentials — run later: opencode auth login"
        fi
    else
        _info "No key entered — authenticate later with: opencode auth login"
    fi
}

# Detect installed agent systems and register the biopb MCP server with each.
# Always drops a canonical, client-agnostic definition at $CONFIG_DIR/mcp.json.
# If nothing is detected, prints guidance so the user can wire it up themselves.
_setup_mcp() {
    local mcp_cmd
    mcp_cmd=$(command -v biopb-mcp 2>/dev/null || echo "biopb-mcp")

    # biopb-mcp 0.6.0+ speaks MCP over stdio: the AI agent spawns it as a child
    # process (`biopb-mcp --transport stdio`) rather than connecting to a
    # long-running HTTP server. Each client therefore needs the *command* to run,
    # not a URL — and we register the resolved absolute path so GUI agents (e.g.
    # Claude Desktop), which don't inherit the shell PATH, can still find it.
    local mcp_args=(--transport stdio)

    # Minimal biopb-mcp config, mainly to ship preconfigured biopb.image servicers.
    # Preserved if it already exists so the user's tweaks survive a rerun.
    local mcp_config_dir="$HOME/.config/biopb-mcp"
    local mcp_config="$mcp_config_dir/config.json"
    mkdir -p "$mcp_config_dir"
    if [ -f "$mcp_config" ]; then
        _ok "biopb-mcp config exists at $mcp_config (preserved)"
    else
        # The tensor server's localhost fast path is now the file-cache mmap
        # handoff (biopb/biopb#9), which beats the gRPC socket and is enabled by
        # default, so no shm opt-out is seeded here anymore.
        cat > "$mcp_config" << 'EOF'
{
  "mcp": {
    "process_image_servers": [
      "grpcs://cellpose.biopb.org:443"
    ]
  }
}
EOF
        _ok "Created biopb-mcp config: $mcp_config"
    fi

    # Canonical standalone definition (standard mcpServers JSON; most clients accept it).
    if _mcp_merge "$CONFIG_DIR/mcp.json" "mcpServers" "stdio" "$mcp_cmd" "${mcp_args[@]}"; then
        _ok "MCP definition written: $CONFIG_DIR/mcp.json"
    else
        _warn "Could not write $CONFIG_DIR/mcp.json"
    fi

    local detected=0

    # --- Claude Code (managed through the `claude` CLI) ---
    if command -v claude &>/dev/null; then
        detected=1
        if claude mcp get biopb &>/dev/null; then
            _ok "Claude Code: biopb already registered"
        elif claude mcp add --scope user biopb -- "$mcp_cmd" "${mcp_args[@]}" &>/dev/null; then
            _ok "Claude Code: registered biopb (user scope)"
        else
            _warn "Claude Code detected but registration failed — add it manually:"
            _cmd "claude mcp add --scope user biopb -- $mcp_cmd ${mcp_args[*]}"
        fi
    fi

    # --- Hermes (NousResearch) — YAML config at ~/.hermes/config.yaml ---
    if [ -d "$HOME/.hermes" ]; then
        detected=1
        if [ -f "$HOME/.hermes/config.yaml" ] && grep -qE '^\s*biopb:' "$HOME/.hermes/config.yaml" 2>/dev/null; then
            _ok "Hermes: biopb already present in config.yaml"
        else
            _ok "Hermes detected"
            _info "Add the following under 'mcp_servers:' in $HOME/.hermes/config.yaml:"
            printf "    %sbiopb:\n      command: \"%s\"\n      args: [\"--transport\", \"stdio\"]%s\n" "$DIM" "$mcp_cmd" "$RESET"
        fi
    fi

    # --- Claude Desktop ---
    local cd_cfg=""
    case "$PLATFORM" in
        macOS)     cd_cfg="$HOME/Library/Application Support/Claude/claude_desktop_config.json" ;;
        Linux|WSL) cd_cfg="$HOME/.config/Claude/claude_desktop_config.json" ;;
    esac
    if [ -n "$cd_cfg" ] && [ -d "$(dirname "$cd_cfg")" ]; then
        detected=1
        _mcp_json_merge "$cd_cfg" "$mcp_cmd" "Claude Desktop" "${mcp_args[@]}"
    fi

    # --- Cursor ---
    if [ -d "$HOME/.cursor" ]; then
        detected=1
        _mcp_json_merge "$HOME/.cursor/mcp.json" "$mcp_cmd" "Cursor" "${mcp_args[@]}"
    fi

    # --- opencode ---
    local opencode_cfg_dir="$HOME/.config/opencode"
    if command -v opencode &>/dev/null || [ -d "$opencode_cfg_dir" ]; then
        detected=1
        local opencode_cfg="$opencode_cfg_dir/opencode.json"
        if _mcp_merge "$opencode_cfg" "mcp" "opencode" "$mcp_cmd" "${mcp_args[@]}"; then
            _ok "opencode: registered biopb ($opencode_cfg)"
        else
            _warn "opencode: could not update $opencode_cfg — add biopb manually"
            _info "Add under 'mcp' in $opencode_cfg:"
            printf "    %s\"biopb\": {\"type\": \"local\", \"command\": [\"%s\", \"--transport\", \"stdio\"], \"enabled\": true}%s\n" "$DIM" "$mcp_cmd" "$RESET"
        fi
    fi

    if [ "$detected" = "0" ]; then
        _info "No supported agent system detected (Claude Code, Claude Desktop, Cursor, Hermes, opencode)."
        _info "To use biopb, register this stdio command with your MCP client:"
        _cmd "$mcp_cmd ${mcp_args[*]}"
        _info "A ready-to-use definition is at: $CONFIG_DIR/mcp.json"
    fi

    # There is no separate server to start: the agent spawns biopb-mcp over stdio
    # on demand, which opens the napari window and brings up the data plane.
    _info "Your AI agent launches biopb-mcp automatically; a napari window opens"
    _info "when it does — no need to start it by hand."
}

# Ensure ~/.local/bin (uv's tool bin dir) is on the user's PATH.
# Persists it for FUTURE shells via uv's own mechanism, plus prints explicit
# guidance for the CURRENT session (uv tool update-shell can't affect the
# running shell, so we never rely on it alone).
#
# Critical: judge against the user's ORIGINAL PATH ($1), not our process PATH —
# install.sh prepends ~/.local/bin for its own use, which would otherwise make
# both this check and `uv tool update-shell` believe the dir is already set up
# (uv prints "already in PATH" and edits no rc file).
_ensure_local_bin_on_path() {
    local bin_dir="$HOME/.local/bin"
    local original_path="${1:-$PATH}"

    # Already persistently on the user's PATH? Nothing to do.
    case ":$original_path:" in
        *":$bin_dir:"*) return 0 ;;
    esac

    # Persist for future shells. Run uv by absolute path with the original PATH
    # so it sees the dir as missing and updates the shell rc (idempotent; tolerate
    # old uv lacking the subcommand).
    local persisted=0
    local uv_bin; uv_bin=$(command -v uv 2>/dev/null || true)
    if [ -n "$uv_bin" ] && PATH="$original_path" "$uv_bin" tool update-shell &>/dev/null; then
        persisted=1
    fi

    local future_msg
    if [ "$persisted" = "1" ]; then
        future_msg="New shells should work automatically. But for THIS terminal, you need:"
    else
        future_msg="Add it to your shell profile, and run it now for THIS terminal:"
    fi

    # Full-width rules (no right border) so the banner is prominent but never
    # misaligns, regardless of how long $HOME / the path is.
    # local rule="──────────────────────────────────────────────────────────────────"
    # printf "\n${YELLOW}${BOLD}  %s${RESET}\n" "$rule"
    # printf   "${YELLOW}${BOLD}  ⚠  ACTION REQUIRED — PATH not configured${RESET}\n"
    # printf   "${YELLOW}${BOLD}  %s${RESET}\n" "$rule"
    _warn   "  %s is not on your PATH; biopb commands live there.\n" "$bin_dir"
    _warn   "  %s\n" "$future_msg"
    _info   "      ${CYAN}export PATH=\"\$HOME/.local/bin:\$PATH\"${RESET}\n"
    echo ""
    # printf   "${YELLOW}${BOLD}  %s${RESET}\n\n" "$rule"
}

# --- Release-based install helpers -------------------------------------------
# The default install path pulls prebuilt wheels (and the data browser) from the
# most recent GitHub release rather than building HEAD from git. That drops the
# git/buf/proto-generation step and keeps the self-contained server wheel paired
# with the exact biopb wheel it was built against (no PyPI version-coupling).

# Fetch the latest release metadata once and cache it in RELEASE_JSON / RELEASE_TAG.
# One API call serves both the wheels and the data browser, keeping us under the
# unauthenticated GitHub rate limit. Returns non-zero if it can't be fetched.
_fetch_latest_release() {
    [ -n "${RELEASE_JSON:-}" ] && return 0
    RELEASE_JSON=$(curl -fsSL -H "Accept: application/vnd.github+json" \
        "https://api.github.com/repos/$RELEASE_REPO/releases/latest" 2>/dev/null) || return 1
    # `|| true`: a missing/error API response (no "tag_name") makes grep exit 1,
    # which under `set -euo pipefail` would otherwise abort the whole installer
    # from inside this command substitution. We want to return 1 and let the
    # caller print a friendly message, so the empty-tag check below handles it.
    RELEASE_TAG=$(printf '%s' "$RELEASE_JSON" \
        | grep '"tag_name"' | head -1 \
        | sed -E 's/.*"tag_name"[[:space:]]*:[[:space:]]*"([^"]+)".*/\1/') || true
    [ -n "${RELEASE_TAG:-}" ] || return 1
    if ! printf '%s' "$RELEASE_TAG" | grep -qE '^[A-Za-z0-9._+/-]+$'; then
        _warn "Unexpected release tag format: $RELEASE_TAG"
        RELEASE_TAG=""
        return 1
    fi
    return 0
}

# Percent-decode a URL path component (e.g. %2B -> +). GitHub encodes the '+' in
# a wheel's local version segment as %2B in the asset URL, but the on-disk file
# must carry the literal '+' or pip/uv reject the wheel (its filename version no
# longer matches the metadata). Used to recover the real wheel name from the URL.
_urldecode() { printf '%b' "${1//%/\\x}"; }

# Print the download URL of the first release asset whose filename matches the
# given extended-regex (anchored at the end of the URL). Empty if none matches.
# Requires _fetch_latest_release to have populated RELEASE_JSON first.
# `|| true`: no match makes grep exit 1, which under `set -euo pipefail` would
# abort the whole installer from inside the caller's command substitution.
# Callers rely on an empty string to fall back / show a friendly error, so we
# stay tolerant and return 0 with no output instead.
_release_asset_url() {
    printf '%s' "${RELEASE_JSON:-}" \
        | grep -oE '"browser_download_url"[[:space:]]*:[[:space:]]*"[^"]+"' \
        | sed -E 's/.*"(https[^"]+)".*/\1/' \
        | grep -E "/$1\$" | head -1 || true
}

install_biopb() {
    set -euo pipefail

    REPO_URL="https://github.com/biopb/biopb"
    REPO="git+$REPO_URL"
    RELEASE_REPO="biopb/biopb"   # owner/name for the GitHub Releases API
    WEBAPP_DIR="$HOME/.local/share/biopb/webapp"
    CONFIG_DIR="$HOME/.config/biopb"

    # Install path: default is prebuilt wheels from the latest GitHub release
    # (no git checkout, no buf/proto build). BIOPB_INSTALL_FROM_SOURCE=1 switches
    # to building HEAD from git — the fast path for development/testing.
    if [ -n "${BIOPB_INSTALL_FROM_SOURCE:-}" ] && [ "${BIOPB_INSTALL_FROM_SOURCE}" != "0" ]; then
        INSTALL_FROM_SOURCE=1
    else
        INSTALL_FROM_SOURCE=0
    fi

    printf "\n%s" "${CYAN}"
    echo "    ____  _       ____  ____  "
    echo "   / __ )(_)___  / __ \\/ __ ) "
    echo "  / __  / / __ \\/ /_/ / __  |"
    echo " / /_/ / / /_/ / ____/ /_/ / "
    echo "/_____/_/\\____/_/   /_____/  "
    printf "%s\n" "${RESET}"
    echo ""
    echo "      Tensor Server Installer"
    echo ""

    # ===== 0. System Check =====
    _step "[0/6] Checking system..."

    OS=$(uname -s)
    ARCH=$(uname -m)

    # Detect platform
    case "$OS" in
        Linux)
            if grep -qi "microsoft\|wsl" /proc/version 2>/dev/null; then
                PLATFORM="WSL"
            else
                PLATFORM="Linux"
            fi
            ;;
        Darwin)
            PLATFORM="macOS"
            ;;
        *)
            _err "Unsupported OS: $OS"
            _info "Supported: Linux, macOS, WSL"
            exit 1
            ;;
    esac

    # Check architecture
    case "$ARCH" in
        x86_64|amd64|arm64|aarch64) ;;
        *)
            _err "Unsupported architecture: $ARCH"
            _info "Supported: x86_64, arm64"
            exit 1
            ;;
    esac

    _ok "Platform: $PLATFORM ($ARCH)"

    # On a fresh Mac /usr/bin/git is only a stub that triggers (or fails to trigger)
    # the Command Line Tools install; it passes `command -v git` below but real
    # git/cc are absent, so the later `uv tool install … @ git+…` dies with a
    # cryptic `xcrun: error: invalid active developer path`. Check for the actual
    # toolchain up front and give an actionable message instead.
    if [ "$PLATFORM" = "macOS" ] && [ "$INSTALL_FROM_SOURCE" = "1" ] && ! xcode-select -p &>/dev/null; then
        _err "Xcode Command Line Tools not found."
        _info "biopb needs them for git and the C compiler/headers used to build packages."
        _info "Install them, then rerun this script:"
        _cmd "xcode-select --install"
        _info "A system dialog will appear — click Install, wait for it to finish, then rerun."
        exit 1
    fi

    # Check required tools. git is only needed to build from a source checkout;
    # the default release install just downloads prebuilt wheels (curl + tar).
    required_tools=(curl tar)
    [ "$INSTALL_FROM_SOURCE" = "1" ] && required_tools+=(git)
    for tool in "${required_tools[@]}"; do
        if ! command -v "$tool" &>/dev/null; then
            _err "$tool not found"
            case "$PLATFORM" in
                Linux|WSL) _info "Install: sudo apt install $tool" ;;
                macOS)     _info "Install: brew install $tool (or Xcode CLI)" ;;
            esac
            exit 1
        fi
        _ok "$tool: found"
    done

    _ok "System check passed"

    # ===== Optional components =====
    # biopb-mcp is always installed (it is the primary interface), so it is not
    # offered here. Bio-Formats defaults to off: it pulls in a heavyweight Java
    # toolchain that most labs don't need (only legacy/proprietary formats need it).
    CHECKBOX_DEFAULTS=(1 0)
    read -r INSTALL_WEBAPP INSTALL_BIOFORMATS <<< "$(_checkbox \
        "Built-in data viewer: see all your images in a browser (Chrome, Safari and others)" \
        "Bio-Formats (more image formats; needs Java and extra setup during first run)")"
    unset CHECKBOX_DEFAULTS
    echo ""

    # ===== 1. Install uv + buf (if needed) =====
    _step "[1/6] Ensuring build tools..."

    # Remember the user's real shell PATH before we prepend ~/.local/bin for our
    # own process — _ensure_local_bin_on_path needs it to tell whether the dir is
    # *persistently* on PATH (our transient export must not mask that).
    ORIGINAL_PATH="$PATH"
    export PATH="$HOME/.local/bin:$PATH"
    if ! command -v uv &>/dev/null; then
        _info "Installing uv..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        _ok "uv installed"
    else
        _ok "uv already installed ($(uv --version))"
    fi

    # buf generates the protobuf/Flight stubs at build time, so it is only needed
    # when building from a source checkout. Release wheels ship the stubs prebuilt.
    if [ "$INSTALL_FROM_SOURCE" = "1" ]; then
        BUF_VERSION="1.70.0"
        # Install into the user's ~/.local/bin (already prepended to PATH above), not
        # /usr/local/bin: the latter is root-owned on a normal account, so curl -o
        # there fails with "failed to write to destination". Keeps buf user-local and
        # consistent with uv's tool bin dir and the Windows installer.
        BUF_BIN="$HOME/.local/bin"
        if ! command -v buf &>/dev/null || [ "$(buf --version 2>/dev/null)" != "$BUF_VERSION" ]; then
            _info "Installing buf $BUF_VERSION..."
            mkdir -p "$BUF_BIN"
            rm -f "$BUF_BIN/buf"
            curl -sSL \
                "https://github.com/bufbuild/buf/releases/download/v${BUF_VERSION}/buf-$(uname -s)-$(uname -m)" \
                -o "$BUF_BIN/buf"
            chmod +x "$BUF_BIN/buf"
            _ok "buf $BUF_VERSION installed"
        else
            _ok "buf already installed ($(buf --version))"
        fi
        unset BUF_VERSION BUF_BIN
    fi

    # ===== 2. Python =====
    _step "[2/6] Ensuring Python..."

    # biopb-mcp (always installed) requires Python >= 3.10.
    MIN_MINOR=10

    # Upper bound: the default `aics` extra pulls aicsimageio, which hard-pins
    # `lxml<5`. No lxml 4.x ships a wheel for CPython >= 3.13, so on a 3.13+
    # interpreter lxml is built from source — which fails on a fresh machine
    # without libxml2/libxslt dev headers (the common WSL install failure).
    # Cap at 3.12, the newest Python with a prebuilt lxml 4.x wheel; if the
    # system Python is newer we fall back to a uv-managed 3.12 below.
    MAX_MINOR=12

    # PYTHON_SPEC is the interpreter we hand to `uv tool install` below via --python.
    # We MUST pin it: without --python, uv auto-discovers an interpreter and may pick
    # an old system python3 (e.g. macOS 3.9) that satisfies a loose lower bound,
    # then fails the build — even though we just installed 3.11 via uv.
    PYTHON_SPEC=""
    PYTHON_VERSION=""
    if command -v python3 &>/dev/null; then
        PYTHON_VERSION=$(python3 -c "import sys; print(sys.version_info[:2])" 2>/dev/null || echo "")
        if [ -n "$PYTHON_VERSION" ]; then
            MAJOR=$(echo "$PYTHON_VERSION" | tr -d '(),' | cut -d' ' -f1)
            MINOR=$(echo "$PYTHON_VERSION" | tr -d '(),' | cut -d' ' -f2)
            if [ "$MAJOR" -eq 3 ] && [ "$MINOR" -ge "$MIN_MINOR" ] && [ "$MINOR" -le "$MAX_MINOR" ]; then
                _ok "Using system Python: $(python3 --version)"
                PYTHON_SPEC=$(command -v python3)
            elif [ "$MAJOR" -gt 3 ] || { [ "$MAJOR" -eq 3 ] && [ "$MINOR" -gt "$MAX_MINOR" ]; }; then
                _warn "System Python too new ($(python3 --version)); using a managed 3.$MAX_MINOR (aicsimageio's lxml<5 has no wheel for 3.13+)"
                PYTHON_VERSION=""
            else
                _warn "System Python too old ($(python3 --version)), need >= 3.$MIN_MINOR"
                PYTHON_VERSION=""
            fi
        fi
    fi

    if [ -z "$PYTHON_VERSION" ]; then
        _info "Installing Python 3.$MAX_MINOR via uv..."
        uv python install "3.$MAX_MINOR"
        _ok "Python 3.$MAX_MINOR ready"
        PYTHON_SPEC="3.$MAX_MINOR"
    fi

    # ===== 3. Install biopb packages =====
    _step "[3/6] Installing biopb packages..."

    TENSOR_EXTRAS="web,ome-zarr,aics,medical,ndtiff"
    if [ "$INSTALL_BIOFORMATS" = "1" ]; then
        TENSOR_EXTRAS="$TENSOR_EXTRAS,bioformats"
        _info "  including Bio-Formats (Java fetched on first use, not now)"
    fi

    # Resolve where biopb + biopb-tensor-server come from. They must be installed
    # as a matched pair from a single build: the tensor server is self-contained
    # and may use proto fields newer than any biopb published on PyPI, so biopb is
    # always pinned to the sibling artifact (a git ref in source mode, a local
    # wheel in release mode) and the resolver is never allowed to pull it from PyPI.
    local biopb_req tensor_req
    if [ "$INSTALL_FROM_SOURCE" = "1" ]; then
        _info "Building from source (HEAD of $REPO_URL)"
        biopb_req="biopb[tensor] @ $REPO"
        tensor_req="biopb-tensor-server[$TENSOR_EXTRAS] @ $REPO#subdirectory=biopb-tensor-server"
    else
        if ! _fetch_latest_release; then
            _err "Could not fetch the latest biopb release from $RELEASE_REPO."
            _info "Check your network, or build from source instead:"
            _cmd "BIOPB_INSTALL_FROM_SOURCE=1 curl -fsSL https://biopb.org/install.sh | bash"
            exit 1
        fi
        local sdk_url tensor_url
        sdk_url=$(_release_asset_url 'biopb-[^/]+\.whl')
        tensor_url=$(_release_asset_url 'biopb_tensor_server-[^/]+\.whl')
        if [ -z "$sdk_url" ] || [ -z "$tensor_url" ]; then
            _err "Release $RELEASE_TAG has no biopb wheels attached."
            _info "Build from source instead:"
            _cmd "BIOPB_INSTALL_FROM_SOURCE=1 curl -fsSL https://biopb.org/install.sh | bash"
            exit 1
        fi
        _info "Installing from release $RELEASE_TAG"
        WHEELS_DIR=$(mktemp -d)
        # Remove the wheel download dir on any exit (success, error, or set -e).
        trap 'rm -rf "${WHEELS_DIR:-}"' EXIT
        local sdk_whl="$WHEELS_DIR/$(_urldecode "$(basename "$sdk_url")")"
        local tensor_whl="$WHEELS_DIR/$(_urldecode "$(basename "$tensor_url")")"
        curl -fsSL "$sdk_url" -o "$sdk_whl"
        curl -fsSL "$tensor_url" -o "$tensor_whl"
        # Direct file:// references pin biopb to this exact wheel, so uv resolves
        # the server's `biopb` dependency to it rather than to PyPI.
        biopb_req="biopb[tensor] @ file://$sdk_whl"
        tensor_req="biopb-tensor-server[$TENSOR_EXTRAS] @ file://$tensor_whl"
    fi

    # Install everything into ONE uv tool environment so the components can import
    # and drive each other at runtime:
    #   - `biopb server start` runs `sys.executable -m biopb_tensor_server.cli`,
    #     so the server must be importable from biopb's interpreter (this also
    #     restores `from biopb_tensor_server.config import load_config`);
    #   - biopb-mcp is a napari plugin + MCP server that talks to the tensor
    #     server and runs a napari viewer in this same env.
    # biopb is the primary tool (exposes the `biopb` command); --with adds the
    # siblings to the same env and --with-executables-from also links their
    # console scripts onto PATH (plain --with does not expose executables).
    #
    # biopb-mcp requires the [mcp] extra (mcp, uvicorn, jupyter_client, ipykernel,
    # psutil) — without it `import mcp` fails. We require >=0.6.0: that release makes
    # stdio the default transport (matching the MCP client config this installer
    # writes) and also drops biopb-mcp's stray, unpinned grpcio-tools dependency,
    # which otherwise collapses the shared solve to an unbuildable grpcio-tools==1.30.0.
    # It comes from PyPI in both modes (no biopb-mcp wheel ships in the release).
    local install_args=(
        --upgrade
        --force
        --python "$PYTHON_SPEC"
        "$biopb_req"
        --with "$tensor_req"
        --with-executables-from biopb-tensor-server
    )
    _info "  including biopb-mcp + napari"
    install_args+=(
        --with "biopb-mcp[mcp]>=0.6.0"
        --with "napari[all]"
        --with-executables-from biopb-mcp
    )

    _info "Installing biopb into one shared environment..."
    uv tool install "${install_args[@]}"
    # The wheel download dir (if any) is removed by the EXIT trap set above.

    VERSION_OUTPUT=$(biopb-tensor-server version 2>/dev/null || echo "installed")
    _ok "$VERSION_OUTPUT"

    # ===== 4. Webapp =====
    _step "[4/6] Installing data browser..."

    if [ "$INSTALL_WEBAPP" = "1" ]; then
        mkdir -p "$WEBAPP_DIR"

        # Reuses the release metadata already fetched for the wheels (cached);
        # in source mode this is the first and only call.
        if _fetch_latest_release; then
            INSTALLED_TAG=""
            [ -f "$WEBAPP_DIR/.version" ] && INSTALLED_TAG=$(cat "$WEBAPP_DIR/.version")
            if [ "$INSTALLED_TAG" = "$RELEASE_TAG" ]; then
                _ok "Data browser already up to date ($RELEASE_TAG)"
            else
                _info "Downloading $RELEASE_TAG..."
                rm -rf "${WEBAPP_DIR:?}"
                mkdir -p "$WEBAPP_DIR"
                local webapp_url
                webapp_url=$(_release_asset_url 'webapp\.tar\.gz')
                webapp_url="${webapp_url:-$REPO_URL/releases/download/$RELEASE_TAG/webapp.tar.gz}"
                curl -fsSL "$webapp_url" \
                    | tar -xzf - -C "$WEBAPP_DIR" --strip-components=1
                printf '%s' "$RELEASE_TAG" > "$WEBAPP_DIR/.version"
                _ok "Data browser installed to: $WEBAPP_DIR"
            fi
        else
            _warn "Could not fetch latest release, data browser not installed"
            _info "Server will run in API-only mode"
        fi
    else
        _info "Skipped"
    fi

    # ===== 5. Config =====
    _step "[5/6] Config..."

    mkdir -p "$CONFIG_DIR"
    CONFIG_FILE="$CONFIG_DIR/biopb.toml"

    if [ -f "$CONFIG_FILE" ]; then
        _ok "Config exists at $CONFIG_FILE (preserved)"
    else
        local DATA_DIR
        if [ -n "${BIOPB_DATA_DIR:-}" ]; then
            DATA_DIR="$BIOPB_DATA_DIR"
            _ok "Using BIOPB_DATA_DIR: $DATA_DIR"
        else
            _pick_data_dir DATA_DIR
            echo ""
            _ok "Data directory: $DATA_DIR"
        fi

        if [[ "$DATA_DIR" == *$'\n'* ]]; then
            _err "DATA_DIR path cannot contain newlines: $DATA_DIR"
            exit 1
        fi
        TOML_DATA_DIR="${DATA_DIR//\\/\\\\}"
        TOML_DATA_DIR="${TOML_DATA_DIR//\"/\\\"}"
        cat > "$CONFIG_FILE" << EOF
[server]
host = "127.0.0.1"
port = 8815
aggressive_dir_pruning = true

[cache]
backend = "file"
file_max_segment_mb = 256
file_max_total_gb = 128

[metadata_db]
enabled = true

[[sources]]
url = "$TOML_DATA_DIR"
monitor = true
EOF
        _ok "Created: $CONFIG_FILE"
    fi

    # ===== 6. Wire biopb-mcp into the user's agent system =====
    _step "[6/6] Configuring MCP client..."

    # An MCP client (AI agent) is what actually drives biopb-mcp. Detect known
    # agents; if none is present, offer to install opencode so the user ends up
    # with a working setup instead of a server with nothing to talk to it.
    _detect_agents
    if [ "${#DETECTED_AGENTS[@]}" -gt 0 ]; then
        _ok "AI agent detected: ${DETECTED_AGENTS[*]}"
    else
        _info ""
        _info "BioPB needs an AI agent to work, but it seems you don't have one installed."
        _info "We can install and setup one (opencode) for you. This allows you to start"
        _info "playing with the system using a free AI agent - no cost."
        _info ""
        if _confirm "Install the opencode agent now?"; then
            _install_opencode
        else
            _info "No agent will be installed; set one up later and rerun to wire it in."
        fi
    fi
    _setup_mcp

    # ===== Summary =====
    printf "\n%s%s%s\n" "${BOLD}" "${YELLOW}" "=== Installation Complete ===${YELLOW}"

    # Make sure the user can actually invoke the freshly installed commands.
    _ensure_local_bin_on_path "$ORIGINAL_PATH"

    printf "%s%s%s\n" "${BOLD}" "${GREEN}" "To launch the data server only without other components:${RESET}"
    _cmd "biopb server start"
    echo ""

    if [ "$INSTALL_WEBAPP" = "0" ] || [ "$INSTALL_BIOFORMATS" = "0" ]; then
        printf "%s%s%s\n" "${BOLD}" "${GREEN}" "Optional components:${RESET}"
    fi
    if [ "$INSTALL_WEBAPP" = "0" ]; then
        _note "Data browser not installed — rerun this script to install"
    else
        _ok "Data browser available at http://localhost:8815"
    fi
    if [ "$INSTALL_BIOFORMATS" = "0" ]; then
        _note "Bio-Formats not installed — ZVI/OIB/OIF and similar legacy formats unsupported"
        _note "to add later, rerun this script and enable Bio-Formats"
    fi
    if [ "$INSTALL_WEBAPP" = "0" ] || [ "$INSTALL_BIOFORMATS" = "0" ]; then
        echo ""
    fi

    printf "%s%s%s\n" "${BOLD}" "${GREEN}" "biopb-mcp configuration file at:${RESET}"
    _cmd "         $HOME/.config/biopb-mcp/config.json"
    echo ""

    printf "%s%s%s\n" "${BOLD}" "${GREEN}" "Data server configuration file at:${RESET}"
    _cmd "         $CONFIG_FILE"
    echo ""

    printf "%s%s%s\n" "${BOLD}" "${GREEN}" "To upgrade: rerun this script${RESET}"
    echo ""
    echo ""

    local agent_cmd
    agent_cmd=$(_agent_launch_cmd)

    local rule="──────────────────────────────────────────────────────────────────"
    printf "%s%s%s%s\n" "${BOLD}" "${GREEN}" "$rule" "${RESET}"
    printf "%s%sCongratulations! You are ready to use BioPB. Next steps:%s\n\n" "${BOLD}" "${GREEN}" "${RESET}"

    printf "  %s1. Start a new shell session, then launch your AI agent.%s Run:\n" "${BOLD}" "${RESET}"
    if [ -n "$agent_cmd" ]; then
        _cmd "$agent_cmd"
    else
        _info "your AI agent (e.g. Claude Code, opencode, Cursor)"
    fi
    _info "The agent launches biopb-mcp for you and a napari window opens."
    _info "Keep the agent session running while you work."
    echo ""

    printf "  %s2. Check your data.%s Confirm it appears in the napari Tensor Browser panel.\n" "${BOLD}" "${RESET}"
    echo ""

    printf "  %s3. Prompt away!%s Ask the agent to load, segment, or measure your images.\n" "${BOLD}" "${RESET}"
    echo ""

    printf "%s%s%s%s\n" "${BOLD}" "${GREEN}" "$rule" "${RESET}"
}

# Only run if script was fully downloaded (function defined completely)
install_biopb
