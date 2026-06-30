#!/bin/bash
#
# biopb stack installer (biopb-mcp + biopb + biopb-tensor-server)
# Usage: curl -fsSL https://biopb.org/install.sh | bash
#
# Idempotent: rerun to upgrade to latest version
#
# Unattended upgrades: set BIOPB_NONINTERACTIVE=1 to suppress every prompt (keeps
# an existing config; leaves the remote algorithm plugins off unless
# BIOPB_REMOTE_PLUGINS=1). It is an upgrade feature — a FRESH unattended install
# must also pass BIOPB_DATA_DIR or it errors out. Example:
#   curl -fsSL https://biopb.org/install.sh | BIOPB_NONINTERACTIVE=1 bash
#
# This installs prebuilt wheels from the latest biopb GitHub release-v*
# deployment (the single release that carries all three mutually-paired wheels).
# By default it tracks the latest STABLE release (clean X.Y.Z). Set
# BIOPB_INSTALL_RC=1 to instead track the latest release candidate (a/b/rc
# prerelease, typically cut off the dev branch) — the fast path for testing an
# upcoming release before it lands on main.
#
# Requirements: curl, tar
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

# Prompt the user to choose a data directory.
# Usage: _pick_data_dir <varname> [keep]  — writes result into caller's variable
# (no subshell). All prompts go to /dev/tty. Requires PLATFORM to be set first.
#
# When the second arg is non-empty ("keep" mode), an extra "0) Keep my current
# config file" option is shown as the *default*; choosing it (or Enter, or any
# invalid input) returns the empty string as a sentinel meaning "don't touch the
# existing config." Callers pass this when a config (biopb.json or legacy
# biopb.toml) already exists.
_pick_data_dir() {
    # Caller passes the name of a variable to receive the result. We assign into
    # it with `printf -v` rather than a `local -n` nameref, because namerefs need
    # bash >= 4.3 and macOS ships bash 3.2.
    local _retvar_name=$1
    local keep_mode="${2:-}"
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
        # On WSL, offer dedicated data subfolders under the Windows profile, but
        # NEVER the profile root itself. Recursively scanning the profile walks
        # AppData and, fatally, OneDrive "Files On-Demand" placeholders, which
        # hydrate-on-read through drvfs and hang discovery before the server can
        # bind. Data/Microscopy folders aren't OneDrive-redirected
        # by default the way Documents/Desktop/Pictures are; users who keep data
        # elsewhere can still type a /mnt/c/... path manually.
        local win_user; win_user=$(cmd.exe /c "echo %USERNAME%" 2>/dev/null | tr -d '\r')
        if [ -n "$win_user" ]; then
            for dir in \
                "/mnt/c/Users/$win_user/Microscopy" \
                "/mnt/c/Users/$win_user/Data" \
                "/mnt/c/Users/$win_user/data"; do
                [ -d "$dir" ] || continue
                local real; real=$(realpath "$dir" 2>/dev/null || echo "$dir")
                local dup=0
                for s in "${seen[@]+"${seen[@]}"}"; do [ "$s" = "$real" ] && dup=1 && break; done
                [ "$dup" = "0" ] && candidates+=("$dir") && seen+=("$real")
            done
        fi
    fi

    local n=${#candidates[@]}
    local manual_opt=$((n + 1))
    local default_dir="${candidates[0]:-$HOME}"

    printf "\n  %sSelect your microscopy data directory:%s\n\n" "$BOLD" "$RESET" >/dev/tty
    if [ -n "$keep_mode" ]; then
        printf "    ${CYAN}0)${RESET} Keep my current config file ${DIM}(default)${RESET}\n" >/dev/tty
    fi
    local i=1
    for dir in "${candidates[@]+"${candidates[@]}"}"; do
        printf "    ${CYAN}%d)${RESET} %s\n" "$i" "$dir" >/dev/tty
        i=$((i + 1))
    done
    printf "    ${CYAN}%d)${RESET} Enter path manually\n\n" "$manual_opt" >/dev/tty

    # Default differs by mode: keep current config (0) vs first candidate (1).
    local default_choice=1
    [ -n "$keep_mode" ] && default_choice=0
    printf "  %sChoice [%s]:%s " "$DIM" "$default_choice" "$RESET" >/dev/tty
    local choice; read -r choice </dev/tty
    choice="${choice:-$default_choice}"

    if [ -n "$keep_mode" ] && [ "$choice" = "0" ]; then
        printf -v "$_retvar_name" '%s' ""   # sentinel: keep the existing config
    elif [ "$choice" = "$manual_opt" ]; then
        local manual
        printf "  Path [%s]: " "$default_dir" >/dev/tty
        read -r manual </dev/tty
        manual="${manual%$'\r'}"
        printf -v "$_retvar_name" '%s' "${manual:-$default_dir}"
    elif [[ "$choice" =~ ^[0-9]+$ ]] && [ "$choice" -ge 1 ] && [ "$choice" -le "$n" ]; then
        printf -v "$_retvar_name" '%s' "${candidates[$((choice - 1))]}"
    elif [ -n "$keep_mode" ]; then
        # In keep mode, an unrecognized choice is non-destructive: keep the config.
        printf "  Invalid choice, keeping current config\n" >/dev/tty
        printf -v "$_retvar_name" '%s' ""
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

# Write the tensor-server config as JSON (biopb.json) -- the canonical format
# (biopb/biopb#34). JSON generation is stdlib on both ends (json.dump), so the
# old hand-rolled TOML escaping is gone.
#
# Usage: _write_server_config <out-json> <data-dir> [prior-config]
# When <prior-config> exists, its settings (server/cache/...) are loaded and
# *preserved*; only the `sources` list is replaced with the chosen data dir, so
# re-running with a new folder no longer discards the user's tuning. A prior
# JSON is read with the stdlib; a legacy TOML is read for migration when this
# Python has a TOML parser (3.11+ stdlib `tomllib`, else `tomli`) and otherwise
# falls back to fresh defaults. The caller retires the legacy TOML. Writes
# atomically; returns non-zero on any error.
_write_server_config() {
    local out="$1" data_dir="$2" prior="${3:-}"
    mkdir -p "$(dirname "$out")"
    _py - "$out" "$data_dir" "$prior" <<'PY'
import json, os, sys

out, data_dir, prior = sys.argv[1], sys.argv[2], sys.argv[3]

data = {}
if prior and os.path.exists(prior):
    try:
        if prior.endswith(".toml"):
            try:
                import tomllib
            except ModuleNotFoundError:
                import tomli as tomllib  # 3.10 fallback
            with open(prior, "rb") as fh:
                data = tomllib.load(fh)
        else:
            with open(prior, encoding="utf-8") as fh:
                data = json.load(fh)
    except Exception:
        # Unreadable/unparseable prior config: start clean rather than abort
        # the install. The new data dir is written either way.
        data = {}
if not isinstance(data, dict):
    data = {}

# A fresh config ships the installer defaults; an existing one keeps whatever
# server/cache/... values it already had. `metadata_db.enabled` is intentionally
# omitted -- the DB is on by default and the flag is deprecated (biopb/biopb#225).
# When migrating a prior config that still carries the flag, strip the noisy
# `enabled = true` form (it's the default) so the server doesn't warn on every
# startup. `enabled = false` is a deliberate user choice (read-only mount, disk
# constraints, etc.) -- preserve it; the deprecation warning on startup is the
# intended informational signal, and Phase 4 is the single hard cutover.
data.setdefault("server", {"host": "127.0.0.1", "port": 8815,
                           "aggressive_dir_pruning": True})
data.setdefault("cache", {"backend": "file", "file_max_segment_mb": 256,
                          "file_max_total_gb": 128})
md = data.pop("metadata_db", None)
if isinstance(md, dict):
    if md.get("enabled", True):
        md.pop("enabled", None)
    if md:
        data["metadata_db"] = md
# Point the server at exactly one watched folder, replacing any prior sources.
data["sources"] = [{"url": data_dir, "monitor": True}]

tmp = out + ".biopb.tmp"
with open(tmp, "w", encoding="utf-8") as fh:
    json.dump(data, fh, indent=2)
    fh.write("\n")
os.replace(tmp, out)
PY
}

# Merge the biopb server into a standard `mcpServers` JSON config (Claude Desktop,
# Cursor, …). biopb-mcp speaks stdio, so the client spawns the command itself.
# Usage: _mcp_json_merge <config-file> <command> <label> [args...]
# Returns 0 only if biopb was actually written into the client's config.
_mcp_json_merge() {
    local file="$1" command="$2" label="$3"; shift 3
    if _mcp_merge "$file" "mcpServers" "stdio" "$command" "$@"; then
        _ok "$label: registered biopb ($file)"
        return 0
    fi
    _warn "$label: could not update $file — add biopb manually (see $CONFIG_DIR/mcp.json)"
    return 1
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
    # Read silently but echo a "*" per character so the user gets visual
    # confirmation their paste registered (plain `read -s` shows nothing, which
    # makes it hard to tell whether a paste worked).
    local _key="" _ch=""
    while IFS= read -rs -n1 _ch </dev/tty; do
        # Empty _ch on a newline-terminated read => Enter pressed; we're done.
        [ -z "$_ch" ] && break
        case "$_ch" in
            $'\r') break ;;                              # carriage return => done
            $'\x7f'|$'\b')                               # backspace / delete
                if [ -n "$_key" ]; then
                    _key="${_key%?}"
                    printf '\b \b' >/dev/tty
                fi
                ;;
            *)
                _key+="$_ch"
                printf '*' >/dev/tty
                ;;
        esac
    done
    printf "\n" >/dev/tty   # the silent read ate the user's newline

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

    # biopb-mcp speaks MCP over stdio: the AI agent spawns it as a child process
    # (`biopb-mcp --transport stdio`). Recent biopb-mcp makes that child a thin
    # bridge that brings up — and shares across clients — a local http daemon on
    # demand (the daemon outlives any one client). The client contract is
    # unchanged, so each client still needs the *command* to run, not a URL —
    # and we register the resolved absolute path so GUI agents (e.g. Claude
    # Desktop), which don't inherit the shell PATH, can still find it.
    local mcp_args=(--transport stdio)

    # Minimal biopb-mcp config, mainly to ship preconfigured biopb.image servicers.
    # Preserved if it already exists so the user's tweaks survive a rerun.
    local mcp_config_dir="$HOME/.config/biopb-mcp"
    local mcp_config="$mcp_config_dir/config.json"
    mkdir -p "$mcp_config_dir"
    if [ -f "$mcp_config" ]; then
        _ok "biopb-mcp config exists at $mcp_config (preserved)"
    else
        # The default algorithm plugins point at remote, off-site servers (cell
        # segmentation, etc.) hosted at UConn Health. Those servers log client
        # IPs, so we ask for consent before enabling them by default rather than
        # quietly shipping a third-party network dependency. Declining just
        # leaves process_image_servers empty; the user can add servers later by
        # editing the config. _confirm defaults to Yes (Enter = enable).
        _info "BioPB ships with algorithm plugins that use remote servers for"
        _info "certain computations, e.g. cell segmentation. The servers are"
        _info "hosted at UConn Health and log client IP addresses."
        _info ""
        local process_image_servers='        "grpcs://cellpose.biopb.org:443"'
        if [ "${NONINTERACTIVE:-0}" = "1" ]; then
            # Consent can't be asked unattended: enable only on explicit opt-in,
            # otherwise leave the IP-logging servers off.
            if [ "${BIOPB_REMOTE_PLUGINS:-0}" = "1" ]; then
                _ok "Remote algorithm plugins enabled (BIOPB_REMOTE_PLUGINS=1)"
            else
                process_image_servers=''
                _ok "Remote algorithm plugins disabled (non-interactive; set BIOPB_REMOTE_PLUGINS=1 to enable)"
            fi
        elif _confirm "Enable the remote algorithm plugins?"; then
            _ok "Remote algorithm plugins enabled"
        else
            process_image_servers=''
            _ok "Remote algorithm plugins disabled (add servers later in $mcp_config)"
        fi

        # The tensor server's localhost fast path is now the file-cache mmap
        # handoff (biopb/biopb#9), which beats the gRPC socket and is enabled by
        # default, so no shm opt-out is seeded here anymore.
        cat > "$mcp_config" << EOF
{
  "mcp": {
    "services": {
      "process_image_servers": [
$process_image_servers
      ]
    }
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

    # Assume the user has no working wiring until a branch below actually writes
    # biopb into a client's config; cleared (0) only on a real registration, so a
    # detected-but-unregistered client (failed `claude mcp add`, unwritable file,
    # or Hermes' manual-only path) still gets the canonical fallback at the end.
    local need_to_show_mcp_config=1

    # --- Claude Code (managed through the `claude` CLI) ---
    if command -v claude &>/dev/null; then
        if claude mcp get biopb &>/dev/null; then
            _ok "Claude Code: biopb already registered"
            need_to_show_mcp_config=0
        elif claude mcp add --scope user biopb -- "$mcp_cmd" "${mcp_args[@]}" &>/dev/null; then
            _ok "Claude Code: registered biopb (user scope)"
            need_to_show_mcp_config=0
        else
            _warn "Claude Code detected but registration failed — add it manually:"
            _cmd "claude mcp add --scope user biopb -- $mcp_cmd ${mcp_args[*]}"
        fi
    fi

    # --- Hermes (NousResearch) — YAML config at ~/.hermes/config.yaml ---
    if [ -d "$HOME/.hermes" ]; then
        if [ -f "$HOME/.hermes/config.yaml" ] && grep -qE '^\s*biopb:' "$HOME/.hermes/config.yaml" 2>/dev/null; then
            _ok "Hermes: biopb already present in config.yaml"
            need_to_show_mcp_config=0
        else
            # We can't safely edit Hermes' YAML, so we only print the snippet — that
            # is not an actual setup, so the flag stays set.
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
        _mcp_json_merge "$cd_cfg" "$mcp_cmd" "Claude Desktop" "${mcp_args[@]}" && need_to_show_mcp_config=0
    fi

    # --- Cursor ---
    if [ -d "$HOME/.cursor" ]; then
        _mcp_json_merge "$HOME/.cursor/mcp.json" "$mcp_cmd" "Cursor" "${mcp_args[@]}" && need_to_show_mcp_config=0
    fi

    # --- opencode ---
    local opencode_cfg_dir="$HOME/.config/opencode"
    if command -v opencode &>/dev/null || [ -d "$opencode_cfg_dir" ]; then
        local opencode_cfg="$opencode_cfg_dir/opencode.json"
        if _mcp_merge "$opencode_cfg" "mcp" "opencode" "$mcp_cmd" "${mcp_args[@]}"; then
            _ok "opencode: registered biopb ($opencode_cfg)"
            need_to_show_mcp_config=0
        else
            _warn "opencode: could not update $opencode_cfg — add biopb manually"
            _info "Add under 'mcp' in $opencode_cfg:"
            printf "    %s\"biopb\": {\"type\": \"local\", \"command\": [\"%s\", \"--transport\", \"stdio\"], \"enabled\": true}%s\n" "$DIM" "$mcp_cmd" "$RESET"
        fi
    fi

    # Nothing auto-wired biopb into a client. Defer the notice to the final
    # summary (via a global) so it groups with the other warnings there.
    MCP_NEEDS_MANUAL="$need_to_show_mcp_config"
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
# Make ~/.local/bin reachable from future shells. Prints nothing; instead it sets
# two globals the final summary reads, so the PATH warning groups with the others:
#   NEED_NEW_TERMINAL  — 1 if the user must open a new shell before biopb is found.
#   PATH_EXPORT_HINT   — the export line to run by hand, only if we couldn't persist.
_ensure_local_bin_on_path() {
    local bin_dir="$HOME/.local/bin"
    local original_path="${1:-$PATH}"

    NEED_NEW_TERMINAL=0
    PATH_EXPORT_HINT=""

    # Already persistently on the user's PATH? Nothing to do.
    case ":$original_path:" in
        *":$bin_dir:"*) return 0 ;;
    esac

    # Persist for future shells. Run uv by absolute path with the original PATH
    # so it sees the dir as missing and updates the shell rc (idempotent; tolerate
    # old uv lacking the subcommand). If that fails, hand the user the export line.
    local uv_bin; uv_bin=$(command -v uv 2>/dev/null || true)
    if ! { [ -n "$uv_bin" ] && PATH="$original_path" "$uv_bin" tool update-shell &>/dev/null; }; then
        PATH_EXPORT_HINT="export PATH=\"\$HOME/.local/bin:\$PATH\""
    fi

    NEED_NEW_TERMINAL=1
}

# --- Release-based install helpers -------------------------------------------
# The default install path pulls prebuilt wheels (and the web interface) from the
# most recent GitHub release rather than building HEAD from git. That drops the
# git/buf/proto-generation step and keeps the self-contained server wheel paired
# with the exact biopb wheel it was built against (no PyPI version-coupling).

# Fetch the latest release metadata once and cache it in RELEASE_JSON / RELEASE_TAG.
# One API call serves both the wheels and the web interface, keeping us under the
# unauthenticated GitHub rate limit. Returns non-zero if it can't be fetched.
_fetch_latest_release() {
    [ -n "${RELEASE_JSON:-}" ] && return 0
    # The monorepo hosts several release lines, so /releases/latest is NOT
    # component-specific. List releases (date-desc) and take the newest whose
    # tag matches the deployment line, then fetch that release by tag. By default
    # the match requires a CLEAN version (release-vX.Y.Z) so prerelease tags
    # (release-v…a/b/rc — release candidates cut off dev) are skipped and never
    # become the default download. With BIOPB_INSTALL_RC=1 the regex also admits
    # a PEP 440 prerelease suffix, so the newest candidate wins.
    local _releases
    _releases=$(curl -fsSL -H "Accept: application/vnd.github+json" \
        "https://api.github.com/repos/$RELEASE_REPO/releases?per_page=100" 2>/dev/null) || return 1
    local _tag_re="^${RELEASE_TAG_PREFIX:-release-v}[0-9]+\.[0-9]+\.[0-9]+$"
    [ "${ALLOW_RC:-0}" = "1" ] && \
        _tag_re="^${RELEASE_TAG_PREFIX:-release-v}[0-9]+\.[0-9]+\.[0-9]+((a|b|rc)[0-9]+)?$"
    # `|| true`: an error/empty response makes grep exit 1, which under
    # `set -euo pipefail` would abort the installer from this command
    # substitution. We want to return 1 and let the caller print a friendly
    # message, so the empty-tag check below handles it.
    RELEASE_TAG=$(printf '%s' "$_releases" \
        | grep '"tag_name"' \
        | sed -E 's/.*"tag_name"[[:space:]]*:[[:space:]]*"([^"]+)".*/\1/' \
        | grep -E "$_tag_re" | head -1) || true
    [ -n "${RELEASE_TAG:-}" ] || return 1
    RELEASE_JSON=$(curl -fsSL -H "Accept: application/vnd.github+json" \
        "https://api.github.com/repos/$RELEASE_REPO/releases/tags/$RELEASE_TAG" 2>/dev/null) || return 1
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

# Print the SHA-256 hex digest of file $1, or nothing if no tool is available
# (Linux ships GNU `sha256sum`; macOS ships `shasum`). The empty result lets the
# caller skip the integrity check rather than abort on a toolless host.
_sha256() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" 2>/dev/null | awk '{print $1}'
    elif command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "$1" 2>/dev/null | awk '{print $1}'
    fi
}

# Verify each wheel path in "$@" against the release's SHA256SUMS asset before it
# is file://-installed. Hard-fails (exits) on a checksum mismatch or a wheel with
# no entry in a SHA256SUMS that exists. Fails OPEN (warns, returns 0) when the
# release predates checksums or no sha256 tool is present, so installs of older
# releases — and toolless hosts — still work. Requires _fetch_latest_release.
_verify_wheels() {
    local sums_url sums
    sums_url=$(_release_asset_url 'SHA256SUMS')
    if [ -z "$sums_url" ]; then
        _warn "Release $RELEASE_TAG has no SHA256SUMS; skipping wheel integrity check"
        return 0
    fi
    sums=$(curl -fsSL "$sums_url" 2>/dev/null) || {
        _warn "Could not fetch SHA256SUMS; skipping wheel integrity check"
        return 0
    }

    local f base expected actual
    for f in "$@"; do
        base=$(basename "$f")
        # SHA256SUMS lines are "<hex>  <filename>"; a leading '*' on the name
        # marks binary mode (Git Bash / Cygwin emit it, GNU/Linux does not), so
        # strip it before matching the basename.
        expected=$(printf '%s' "$sums" \
            | awk -v b="$base" '{f=$2; sub(/^\*/, "", f)} f == b {print $1; exit}')
        if [ -z "$expected" ]; then
            _err "No checksum for $base in the release SHA256SUMS"
            exit 1
        fi
        actual=$(_sha256 "$f")
        if [ -z "$actual" ]; then
            _warn "No sha256 tool found; skipping integrity check for $base"
            continue
        fi
        if [ "$actual" != "$expected" ]; then
            _err "Checksum mismatch for $base — refusing to install"
            _info "  expected $expected"
            _info "  actual   $actual"
            exit 1
        fi
    done
    _ok "Wheel checksums verified"
}

# Print the tail of the server log, indented, for diagnosing a bad startup.
_tail_log() {
    local log="$1"
    [ -f "$log" ] || return 0
    _info "recent server log ($log):"
    tail -n 15 "$log" 2>/dev/null | while IFS= read -r line; do
        printf "      ${DIM}%s${RESET}\n" "$line"
    done
}

# Start (or restart) the background data server, then report its health.
# Best-effort: never aborts the install. Skip with BIOPB_NO_SERVER_START=1.
# Starting now lets the pre-cache warm overviews before the user opens anything,
# and a restart makes an already-running (stale) server pick up the new code.
_start_data_server() {
    local log_file="$HOME/.local/share/biopb/logs/tensor-server.log"

    if [ "${BIOPB_NO_SERVER_START:-0}" = "1" ]; then
        _info "Skipping server start (BIOPB_NO_SERVER_START=1)"
        _info "  start it later with: ${CYAN}biopb server start${RESET}"
        return 0
    fi
    if ! command -v biopb >/dev/null 2>&1; then
        _warn "biopb not found on PATH; skipping server start"
        _info "  start it later with: ${CYAN}biopb server start${RESET}"
        return 0
    fi

    # 'restart' loads the just-installed code if a server is already running,
    # and is a plain start otherwise.
    biopb server restart >/dev/null 2>&1 || true

    # Ask the daemon for its health, polling until it reaches SERVING (or 60s).
    # stderr carries live progress ("data server starting - N found so far...")
    # and is intentionally NOT swallowed so the user sees the wait; stdout is the
    # JSON verdict we parse below.
    local out health count
    out=$(biopb server status --json --wait 60) || out=""

    # Tolerate an older biopb that predates --json/--wait: fall back to a plain
    # liveness check so the installer still works during a version transition.
    if [ -z "$out" ]; then
        if biopb server status 2>/dev/null | grep -q "Running"; then
            _ok "Data server started"
        else
            _warn "Data server may not have started"
            _tail_log "$log_file"
            _info "  full log: ${CYAN}$log_file${RESET}"
        fi
        return 0
    fi

    health=$(printf '%s' "$out" | sed -n 's/.*"health"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p')
    count=$(printf '%s' "$out" | sed -n 's/.*"source_count"[[:space:]]*:[[:space:]]*\([0-9][0-9]*\).*/\1/p')

    if [ "$health" != "SERVING" ]; then
        _warn "Data server did not come up cleanly"
        _info "  it may still be scanning a large folder, or failed to start:"
        _tail_log "$log_file"
        _info "  full log: ${CYAN}$log_file${RESET}"
        return 0
    fi

    if [ -z "$count" ] || [ "$count" = "0" ]; then
        _warn "Data server is running but found no data sources"
        _info "  check that your data folder holds supported images (see config):"
        _cmd "  ${ACTIVE_CONFIG:-$CONFIG_FILE}"
        _tail_log "$log_file"
        return 0
    fi

    _ok "Data server ready — $count data source(s) found; pre-caching overviews"
}

# Stop a running biopb-mcp daemon (best-effort) so the just-installed code takes
# effect. UNLIKE the data server we deliberately do NOT restart it: the MCP
# daemon owns a *visible* napari viewer and is brought up on demand by each AI
# client's stdio bridge, so a plain stop is enough — the next agent reconnect
# spawns a fresh (new-code) daemon via ensure_daemon, without the installer
# popping a viewer window the user never asked for. Stopping does close any open
# viewer and drops live agent sessions, so we announce it, and only act when a
# daemon is actually running (a first install / not-running case stays silent).
_stop_mcp_server() {
    command -v biopb >/dev/null 2>&1 || return 0

    # `biopb mcp status --json` -> {"running": true|false, ...}. Only proceed on
    # a live daemon so nothing is printed (or torn down) when none is up.
    if ! biopb mcp status --json 2>/dev/null \
        | grep -q '"running"[[:space:]]*:[[:space:]]*true'; then
        return 0
    fi

    _info "Stopping the biopb MCP server so the update takes effect"
    _info "  (this closes any open napari viewer; it restarts on demand)"
    biopb mcp stop >/dev/null 2>&1 || true
}

install_biopb() {
    set -euo pipefail

    # All three wheels (+ webapp) are pulled from ONE biopb release-v*
    # deployment — a mutually-paired set built from the tagged commit. All three
    # packages live in the biopb monorepo (biopb-mcp and biopb-tensor-server are
    # subdirectories of biopb/biopb).
    BIOPB_REPO_URL="https://github.com/biopb/biopb"
    REPO_URL="$BIOPB_REPO_URL"        # webapp release-asset fallback URL
    RELEASE_REPO="biopb/biopb"        # owner/name for the GitHub Releases API
    # The monorepo hosts multiple release lines (release-v*, v*, mcp-v*,
    # server-v*). The all-in-one deployment the installer wants is the
    # release-v* one (see docs/release-model.md), so the release fetch filters
    # by this prefix instead of using /releases/latest (which is repo-wide).
    RELEASE_TAG_PREFIX="release-v"
    WEBAPP_DIR="$HOME/.local/share/biopb/webapp"
    CONFIG_DIR="$HOME/.config/biopb"

    # Release channel: default tracks the latest STABLE release (clean X.Y.Z).
    # BIOPB_INSTALL_RC=1 also admits the latest release candidate (a/b/rc
    # prerelease, typically cut off dev) — the fast path for testing an upcoming
    # release before it lands on main. Both channels install prebuilt wheels;
    # neither builds from git.
    if [ -n "${BIOPB_INSTALL_RC:-}" ] && [ "${BIOPB_INSTALL_RC}" != "0" ]; then
        ALLOW_RC=1
    else
        ALLOW_RC=0
    fi

    printf "\n%s" "${CYAN}"
    echo "    ____  _       ____  ____  "
    echo "   / __ )(_)___  / __ \\/ __ ) "
    echo "  / __  / / __ \\/ /_/ / __  |"
    echo " / /_/ / / /_/ / ____/ /_/ / "
    echo "/_____/_/\\____/_/   /_____/  "
    printf "%s\n" "${RESET}"
    echo ""
    echo "      biopb stack installer"
    echo ""

    # ===== 0. System Check =====
    _step "[0/7] Checking system..."

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

    # The install downloads prebuilt wheels (release-v* deployment), so only
    # curl + tar are needed — no git, buf, or compiler.
    required_tools=(curl tar)
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
    # No longer offered interactively (biopb/biopb#237). The web interface now
    # carries the server admin page (config / status / restart) on top of the
    # image viewer, so it is installed by default rather than being optional.
    # Bio-Formats stays off by default: the Python adapters now cover the formats
    # most labs use, and it pulls in a heavyweight Java toolchain. Both remain
    # overridable for scripted installs via env vars:
    #   BIOPB_INSTALL_WEBAPP=0      skip the web interface (API-only server)
    #   BIOPB_INSTALL_BIOFORMATS=1  add Bio-Formats (Java fetched on first use)
    if [ "${BIOPB_INSTALL_WEBAPP:-1}" != "0" ]; then INSTALL_WEBAPP=1; else INSTALL_WEBAPP=0; fi
    if [ "${BIOPB_INSTALL_BIOFORMATS:-0}" = "1" ]; then INSTALL_BIOFORMATS=1; else INSTALL_BIOFORMATS=0; fi

    # ===== Non-interactive / unmanned mode =====
    # BIOPB_NONINTERACTIVE=1 suppresses every prompt so the installer can run
    # unattended (cron upgrades, CI, image bakes). It is primarily an UPGRADE
    # feature: with an existing config, that config is kept untouched and nothing
    # is asked. A fresh unattended install must pass BIOPB_DATA_DIR (we will not
    # guess a data directory) — without it the run errors out (see step 5) rather
    # than indexing a default folder. Either way the remote algorithm plugins stay
    # DISABLED unless BIOPB_REMOTE_PLUGINS=1 — consent can't be asked unattended,
    # so we never silently enable the off-site IP-logging servers.
    if [ -n "${BIOPB_NONINTERACTIVE:-}" ] && [ "${BIOPB_NONINTERACTIVE}" != "0" ]; then
        NONINTERACTIVE=1
        _info "Non-interactive mode (BIOPB_NONINTERACTIVE=1): prompts suppressed"
    else
        NONINTERACTIVE=0
    fi

    # ===== 1. Install uv (if needed) =====
    _step "[1/7] Ensuring build tools..."

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

    # ===== 2. Python =====
    _step "[2/7] Ensuring Python..."

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
    _step "[3/7] Installing biopb packages..."

    TENSOR_EXTRAS="web,aics,medical,ndtiff,hdf5"
    if [ "$INSTALL_BIOFORMATS" = "1" ]; then
        TENSOR_EXTRAS="$TENSOR_EXTRAS,bioformats"
        _info "  including Bio-Formats (Java fetched on first use, not now)"
    fi

    # Resolve where the three packages come from. They must be installed as a
    # matched set from a single build: the tensor server is self-contained and
    # may use proto fields newer than any biopb on PyPI, and biopb-mcp is tightly
    # coupled to both — so all three are pinned to the sibling wheels from one
    # release-v* deployment (release CI builds the mutually-paired triple from
    # the tagged commit) and the resolver is never allowed to pull biopb /
    # biopb-tensor-server / biopb-mcp from PyPI. One download is one consistent
    # set — no PyPI-vs-release version skew.
    local biopb_req tensor_req mcp_req
    # napari is the one runtime dep resolved from PyPI. We pin it to the exact
    # version this release was built/tested against (carried in its versions.json
    # attribute, read below) so the deployed object graph matches the graph-walk
    # thread-safety test — and so the napari[all] Qt binding is the tested one.
    local napari_req="napari[all]"
    if ! _fetch_latest_release; then
        _err "Could not fetch the latest biopb release-v* deployment from $RELEASE_REPO."
        if [ "$ALLOW_RC" = "1" ]; then
            _info "No release candidate found. Check your network, or install the stable release:"
            _cmd "curl -fsSL https://biopb.org/install.sh | bash"
        else
            _info "Check your network and rerun. To try the latest release candidate:"
            _cmd "curl -fsSL https://biopb.org/install.sh | BIOPB_INSTALL_RC=1 bash"
        fi
        exit 1
    fi
    local mcp_url sdk_url tensor_url
    mcp_url=$(_release_asset_url 'biopb_mcp-[^/]+\.whl')
    sdk_url=$(_release_asset_url 'biopb-[^/]+\.whl')
    tensor_url=$(_release_asset_url 'biopb_tensor_server-[^/]+\.whl')
    if [ -z "$mcp_url" ] || [ -z "$sdk_url" ] || [ -z "$tensor_url" ]; then
        _err "Release $RELEASE_TAG is missing one of the biopb wheels."
        _info "Try again later, or report this against $RELEASE_REPO."
        exit 1
    fi
    # Pin napari from the release's versions.json attribute so the installed
    # napari is identical to the one this release was built/tested against
    # (closes the last dev/deploy version-skew — and the napari[all] Qt
    # binding, which is napari-version-dependent). The same manifest carries the
    # deployment `release` version, which we record post-install as the
    # auto-updater's baseline (issue #87). Tolerant: an older release without the
    # manifest falls back to the unversioned napari spec and a tag-derived
    # version. RELEASE_VERSION is read here but written only after a clean install.
    local versions_url versions_json napari_pin
    versions_url=$(_release_asset_url 'versions\.json')
    if [ -n "$versions_url" ]; then
        versions_json=$(curl -fsSL "$versions_url" 2>/dev/null) || versions_json=""
        napari_pin=$(printf '%s' "$versions_json" \
            | sed -n 's/.*"napari"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p')
        [ -n "$napari_pin" ] && napari_req="napari[all]==$napari_pin"
        RELEASE_VERSION=$(printf '%s' "$versions_json" \
            | sed -n 's/.*"release"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p')
    fi
    # Fall back to the tag (release-vX.Y.Z -> X.Y.Z) when the manifest is absent
    # or lacks `release`, so the recorded baseline is always a clean PEP 440
    # version the update check can compare with packaging.version.
    RELEASE_VERSION="${RELEASE_VERSION:-${RELEASE_TAG#"${RELEASE_TAG_PREFIX:-release-v}"}}"
    _info "Installing from release $RELEASE_TAG"
    WHEELS_DIR=$(mktemp -d)
    # Remove the wheel download dir on any exit (success, error, or set -e).
    trap 'rm -rf "${WHEELS_DIR:-}"' EXIT
    local mcp_whl="$WHEELS_DIR/$(_urldecode "$(basename "$mcp_url")")"
    local sdk_whl="$WHEELS_DIR/$(_urldecode "$(basename "$sdk_url")")"
    local tensor_whl="$WHEELS_DIR/$(_urldecode "$(basename "$tensor_url")")"
    curl -fsSL "$mcp_url" -o "$mcp_whl"
    curl -fsSL "$sdk_url" -o "$sdk_whl"
    curl -fsSL "$tensor_url" -o "$tensor_whl"
    # Verify the downloaded wheels against the release's SHA256SUMS before they
    # are file://-installed (aborts on a mismatch; fails open on an older release
    # without the manifest). See the auto-updater trust item in issue #87.
    _verify_wheels "$mcp_whl" "$sdk_whl" "$tensor_whl"
    # Direct file:// references pin each package to this exact wheel, so uv
    # resolves their inter-dependencies (the server's `biopb`, biopb-mcp's
    # `biopb[tensor]`) to the downloaded set rather than to PyPI.
    mcp_req="biopb-mcp[mcp] @ file://$mcp_whl"
    biopb_req="biopb[tensor] @ file://$sdk_whl"
    tensor_req="biopb-tensor-server[$TENSOR_EXTRAS] @ file://$tensor_whl"

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
    # psutil) — without it `import mcp` fails; the extra is applied to the pinned
    # wheel/ref ($mcp_req) just like the others. It now ships in the biopb-mcp
    # release alongside biopb + tensor-server (one matched triple), so unlike the
    # old layout it is no longer pulled from PyPI. napari[all] is the one runtime
    # dep still resolved from PyPI, but pinned to the release's versions.json
    # version ($napari_req, set above) so it matches the tested build.
    local install_args=(
        --upgrade
        --force
        --python "$PYTHON_SPEC"
        "$biopb_req"
        --with "$tensor_req"
        --with-executables-from biopb-tensor-server
    )
    _info "  including biopb-mcp + $napari_req"
    install_args+=(
        --with "$mcp_req"
        --with "$napari_req"
        --with-executables-from biopb-mcp
    )

    # Retire any running old-code MCP daemon before the new wheels land, so the
    # next agent reconnect brings up the just-installed code (the daemon is
    # spawned on demand, so there is nothing to restart here).
    _stop_mcp_server

    _info "Installing biopb into one shared environment..."
    uv tool install "${install_args[@]}"
    # The wheel download dir (if any) is removed by the EXIT trap set above.

    VERSION_OUTPUT=$(biopb-tensor-server version 2>/dev/null || echo "installed")
    _ok "$VERSION_OUTPUT"

    # Record the installed deployment version as the kernel-start auto-updater's
    # baseline (issue #87): the check compares the latest release-v* deployment's
    # versions.json `release` against this marker. Written only now, after a clean
    # install, so a half-finished run never advertises a version it isn't on.
    # Best-effort — a write failure only re-prompts a future update, never the
    # install. (biopb_mcp.__version__ is a decoupled library version and is
    # deliberately NOT used for this comparison.)
    if mkdir -p "$CONFIG_DIR" 2>/dev/null \
        && printf '%s' "$RELEASE_VERSION" > "$CONFIG_DIR/release.version" 2>/dev/null; then
        _info "  recorded installed release $RELEASE_VERSION"
    else
        _warn "Could not record installed release version (update checks may re-prompt)"
    fi

    # ===== 4. Webapp =====
    _step "[4/7] Installing web interface..."

    if [ "$INSTALL_WEBAPP" = "1" ]; then
        mkdir -p "$WEBAPP_DIR"

        # Reuses the release metadata already fetched for the wheels (cached).
        if _fetch_latest_release; then
            INSTALLED_TAG=""
            [ -f "$WEBAPP_DIR/.version" ] && INSTALLED_TAG=$(cat "$WEBAPP_DIR/.version")
            if [ "$INSTALLED_TAG" = "$RELEASE_TAG" ]; then
                _ok "Web interface already up to date ($RELEASE_TAG)"
            else
                _info "Downloading $RELEASE_TAG..."
                rm -rf "${WEBAPP_DIR:?}"
                mkdir -p "$WEBAPP_DIR"
                local webapp_url
                webapp_url=$(_release_asset_url 'webapp\.tar\.gz')
                webapp_url="${webapp_url:-$REPO_URL/releases/download/$RELEASE_TAG/webapp.tar.gz}"
                local tmp
                tmp=$(mktemp)
                if curl -fsSL "$webapp_url" -o "$tmp" 2>/dev/null; then
                    tar -xzf "$tmp" -C "$WEBAPP_DIR" --strip-components=1
                    printf '%s' "$RELEASE_TAG" > "$WEBAPP_DIR/.version"
                    _ok "Web interface installed to: $WEBAPP_DIR"
                else
                    _warn "No webapp.tar.gz in release $RELEASE_TAG; server will run in API-only mode"
                fi
                rm -f "$tmp"
            fi
        else
            _warn "Could not fetch latest release, web interface not installed"
            _info "Server will run in API-only mode"
        fi
    else
        _info "Skipped"
    fi

    # ===== 5. Config =====
    _step "[5/7] Config..."

    mkdir -p "$CONFIG_DIR"
    CONFIG_FILE="$CONFIG_DIR/biopb.json"        # canonical format (biopb/biopb#34)
    LEGACY_CONFIG="$CONFIG_DIR/biopb.toml"      # pre-#34 installs

    # An existing config in either format counts for the keep-vs-rewrite decision.
    # biopb.json wins when both are present (matches the server's find_config).
    local EXISTING_CONFIG=""
    if [ -f "$CONFIG_FILE" ]; then
        EXISTING_CONFIG="$CONFIG_FILE"
    elif [ -f "$LEGACY_CONFIG" ]; then
        EXISTING_CONFIG="$LEGACY_CONFIG"
    fi

    # The data-directory prompt is always offered; when a config already exists it
    # gains a default "0) Keep my current config file" option. Choosing a fresh
    # data dir no longer rewrites the whole file: we load the existing config,
    # preserve its settings, and replace only the `sources` list (biopb/biopb#34).
    # An empty DATA_DIR is the "keep" sentinel.
    local DATA_DIR
    if [ -n "$EXISTING_CONFIG" ] && [ -z "${BIOPB_DATA_DIR:-}" ]; then
        # Existing config + no override: keep it. Non-interactive skips the prompt
        # (the unmanned-upgrade fast path); interactive offers "0) keep" as default.
        if [ "$NONINTERACTIVE" = "1" ]; then
            DATA_DIR=""
            _note "Non-interactive: keeping existing config ($EXISTING_CONFIG)."
        else
            _pick_data_dir DATA_DIR keep
            echo ""
        fi
    elif [ -n "$EXISTING_CONFIG" ]; then
        # BIOPB_DATA_DIR is a non-interactive override; it only applies to a fresh
        # install. With a config already present we keep it (its data dir wins).
        _note "BIOPB_DATA_DIR is set but a config already exists; keeping it (remove $EXISTING_CONFIG to apply BIOPB_DATA_DIR)."
        DATA_DIR=""
    elif [ -n "${BIOPB_DATA_DIR:-}" ]; then
        DATA_DIR="$BIOPB_DATA_DIR"
        _ok "Using BIOPB_DATA_DIR: $DATA_DIR"
    elif [ "$NONINTERACTIVE" = "1" ]; then
        # Non-interactive is an UPGRADE feature (no existing config = fresh install).
        # We won't guess a data directory unattended, so require BIOPB_DATA_DIR and
        # fail clearly rather than silently indexing some default folder.
        _err "Non-interactive mode needs an existing install or an explicit data directory."
        _info "  No config found at $CONFIG_FILE — this looks like a fresh install."
        _info "  Set BIOPB_DATA_DIR=/path/to/microscopy to provision unattended, or"
        _info "  rerun without BIOPB_NONINTERACTIVE to choose interactively."
        exit 1
    else
        _pick_data_dir DATA_DIR
        echo ""
        _ok "Data directory: $DATA_DIR"
    fi

    # ACTIVE_CONFIG is the file the running server will read -- the JSON we write,
    # or the untouched existing file when the user keeps it (shown in the summary).
    local ACTIVE_CONFIG="$EXISTING_CONFIG"
    if [ -z "$DATA_DIR" ]; then
        _ok "Keeping current config: $EXISTING_CONFIG"
    else
        if [[ "$DATA_DIR" == *$'\n'* ]]; then
            _err "DATA_DIR path cannot contain newlines: $DATA_DIR"
            exit 1
        fi
        if ! _write_server_config "$CONFIG_FILE" "$DATA_DIR" "$EXISTING_CONFIG"; then
            _err "Failed to write config: $CONFIG_FILE"
            exit 1
        fi
        ACTIVE_CONFIG="$CONFIG_FILE"
        # Retire a legacy TOML we just superseded so the server does not warn
        # about both files shadowing (find_config prefers biopb.json). Its
        # settings were carried into the new JSON above.
        if [ "$EXISTING_CONFIG" = "$LEGACY_CONFIG" ] && [ -f "$LEGACY_CONFIG" ]; then
            mv "$LEGACY_CONFIG" "$LEGACY_CONFIG.bak.$(date +%Y%m%d%H%M%S)"
            _info "Migrated legacy TOML config to JSON (old file backed up)"
        fi
        if [ -n "$EXISTING_CONFIG" ]; then
            _ok "Updated: $CONFIG_FILE"
        else
            _ok "Created: $CONFIG_FILE"
        fi
    fi

    # ===== 6. Start the data server =====
    # Before MCP wiring so a typo in the data dir (step 5) surfaces right after
    # the choice, while pre-cache gets the earliest possible head start.
    _step "[6/7] Starting data server..."
    _start_data_server

    # ===== 7. Wire biopb-mcp into the user's agent system =====
    _step "[7/7] Configuring MCP client..."

    # An MCP client (AI agent) is what actually drives biopb-mcp. Detect known
    # agents; if none is present, offer to install opencode so the user ends up
    # with a working setup instead of a server with nothing to talk to it.
    _detect_agents
    if [ "${#DETECTED_AGENTS[@]}" -gt 0 ]; then
        _ok "AI agent detected: ${DETECTED_AGENTS[*]}"
    elif [ "$NONINTERACTIVE" = "1" ]; then
        _note "Non-interactive: no AI agent detected and none installed; set one up later and rerun to wire it in."
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
    # Two groups, in order: all informational blocks, then all warnings. Every
    # block is one headline line, optional indented detail lines, then one blank
    # line. _ensure_local_bin_on_path and _setup_mcp only set flags, so their
    # warnings land in the warning group below rather than printing inline.
    printf "\n%s%s%s\n\n" "${BOLD}" "${YELLOW}" "=== Installation Complete ===${RESET}"

    _ensure_local_bin_on_path "$ORIGINAL_PATH"

    # Headlines via _info (indent 2, matching _ok/_warn), detail lines indent 4.
    # --- informational blocks ---
    if [ "$INSTALL_WEBAPP" = "1" ]; then
        _info "Web interface available at http://localhost:8814"
        echo ""
    fi

    _info "biopb-mcp configuration file:"
    _cmd "  $HOME/.config/biopb-mcp/config.json"
    echo ""

    _info "Data server configuration file:"
    _cmd "  $ACTIVE_CONFIG"
    echo ""

    _info "To upgrade: rerun this script"
    echo ""

    # --- warnings ---
    if [ "$NEED_NEW_TERMINAL" = "1" ]; then
        _warn "you need to start a new terminal in order to use biopb"
        [ -n "$PATH_EXPORT_HINT" ] && _info "  or run now: ${CYAN}$PATH_EXPORT_HINT${RESET}"
        echo ""
    fi

    if [ "$INSTALL_WEBAPP" = "0" ]; then
        _warn "Web interface not installed (BIOPB_INSTALL_WEBAPP=0)"
        _info "  rerun without that env var to install it"
        echo ""
    fi

    if [ "${MCP_NEEDS_MANUAL:-0}" = "1" ]; then
        _warn "biopb is not registered with any MCP client"
        _info "  register it manually using $CONFIG_DIR/mcp.json"
        echo ""
    fi

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

# ===== Uninstall path =========================================================
# Mirrors the Windows engine's -Uninstall / -Purge teardown (biopb-engine.ps1,
# Invoke-BiopbUninstall): stop the daemons, unregister biopb from every agent
# this installer wires up, remove the one shared uv tool environment, and — only
# with --purge — delete config and cached/state data. The user's image data is
# NEVER touched. Best-effort throughout (no `set -e`): a missing component or a
# stop that does nothing must not abort the rest of the teardown.

# Remove the biopb stdio entry from a JSON MCP config: delete the "biopb" key
# under top-level <parent> in <file>. No-op when the file, the parent section,
# or the entry is absent (the file is left byte-for-byte untouched). Preserves
# all other content and writes atomically — the exact inverse of _mcp_merge.
# Prints "removed" iff it deleted an entry, so callers can report per client.
_mcp_unmerge() {
    local file="$1" parent="$2"
    [ -f "$file" ] || return 0
    _py - "$file" "$parent" 2>/dev/null <<'PY'
import json, os, sys
path, parent = sys.argv[1], sys.argv[2]
try:
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
except (FileNotFoundError, ValueError):
    sys.exit(0)
if not isinstance(data, dict):
    sys.exit(0)
section = data.get(parent)
if not isinstance(section, dict) or "biopb" not in section:
    sys.exit(0)
del section["biopb"]
tmp = path + ".biopb.tmp"
with open(tmp, "w", encoding="utf-8") as fh:
    json.dump(data, fh, indent=2)
    fh.write("\n")
os.replace(tmp, path)
print("removed")
PY
}

# Unregister biopb from every MCP client the installer can wire it into: Claude
# Code via its CLI, and the JSON-config clients (Claude Desktop, Cursor,
# opencode) via _mcp_unmerge. Requires PLATFORM to be set (Claude Desktop's
# config path is OS-specific).
_unregister_agents() {
    local removed_any=0

    # Claude Code — managed through the `claude` CLI. The installer adds it at
    # user scope; try that first, then the default scope, to cover older wirings.
    if command -v claude &>/dev/null; then
        if claude mcp remove biopb -s user &>/dev/null \
            || claude mcp remove biopb &>/dev/null; then
            _ok "Claude Code: unregistered biopb"
            removed_any=1
        fi
    fi

    # JSON-config clients. Each row is "label|file|parent-key".
    local cd_cfg=""
    case "$PLATFORM" in
        macOS)     cd_cfg="$HOME/Library/Application Support/Claude/claude_desktop_config.json" ;;
        Linux|WSL) cd_cfg="$HOME/.config/Claude/claude_desktop_config.json" ;;
    esac
    local row label file parent
    for row in \
        "Claude Desktop|$cd_cfg|mcpServers" \
        "Cursor|$HOME/.cursor/mcp.json|mcpServers" \
        "opencode|$HOME/.config/opencode/opencode.json|mcp"; do
        IFS='|' read -r label file parent <<< "$row"
        [ -n "$file" ] || continue
        if [ "$(_mcp_unmerge "$file" "$parent")" = "removed" ]; then
            _ok "$label: unregistered biopb ($file)"
            removed_any=1
        fi
    done

    # Hermes' YAML is edited by hand on install, so we can't safely edit it back.
    if [ -f "$HOME/.hermes/config.yaml" ] \
        && grep -qE '^\s*biopb:' "$HOME/.hermes/config.yaml" 2>/dev/null; then
        _info "Hermes: remove the 'biopb:' entry from $HOME/.hermes/config.yaml by hand"
    fi

    [ "$removed_any" = "0" ] && _info "No MCP client registrations found to remove"
    return 0
}

# Print usage for the flag-driven entry point to stderr (help is diagnostic, and
# stdout may be the curl|bash pipe).
_usage() {
    cat >&2 <<EOF
biopb stack installer

Usage:
  curl -fsSL https://biopb.org/install.sh | bash                          # install / upgrade
  curl -fsSL https://biopb.org/install.sh | bash -s -- --uninstall [--purge]

Options:
  --uninstall   Remove the biopb stack: stop the data/MCP servers, unregister
                biopb from detected AI agents, and remove the package
                environment. Keeps config and cached data unless --purge.
  --purge       With --uninstall, also delete config and cached/state data
                (~/.config/biopb, ~/.config/biopb-mcp, ~/.local/share/biopb).
                Implies --uninstall. Never touches your image data.
  -h, --help    Show this help.
EOF
}

uninstall_biopb() {
    local purge="${1:-0}"

    # Minimal platform detection — _unregister_agents needs it for the
    # OS-specific Claude Desktop config path. (The install path detects this in
    # its own step-0 system check, which we deliberately skip here.)
    case "$(uname -s)" in
        Linux)  if grep -qi "microsoft\|wsl" /proc/version 2>/dev/null; then PLATFORM="WSL"; else PLATFORM="Linux"; fi ;;
        Darwin) PLATFORM="macOS" ;;
        *)      PLATFORM="Linux" ;;
    esac

    printf "\n%s%sUninstalling the biopb stack%s\n" "${BOLD}" "${CYAN}" "${RESET}"

    # 1. Stop the daemons first. On some platforms a live process keeps its files
    #    open and `uv tool uninstall` then can't remove the tool dir (the Windows
    #    engine hits exactly this — os error 5), so stopping precedes removal.
    _step "[1/3] Stopping biopb services..."
    if command -v biopb &>/dev/null; then
        biopb server stop &>/dev/null || true
        biopb mcp stop &>/dev/null || true
        _ok "Data server and MCP server stopped (if they were running)"
    else
        _info "biopb command not on PATH; nothing to stop"
    fi

    # 2. Unregister from agents BEFORE removing the package, while `claude` and
    #    the config paths are still meaningful.
    _step "[2/3] Unregistering from AI agents..."
    _unregister_agents

    # 3. Remove the one shared uv tool environment (holds all three packages and
    #    their console scripts: biopb, biopb-tensor-server, biopb-mcp).
    _step "[3/3] Removing biopb packages..."
    if command -v uv &>/dev/null; then
        if uv tool uninstall biopb &>/dev/null; then
            _ok "Removed the biopb tool environment (biopb, biopb-tensor-server, biopb-mcp)"
        else
            _info "biopb tool environment not present (already removed?)"
        fi
    else
        _warn "uv not found; cannot remove the biopb tool environment"
        _info "  install uv and run: ${CYAN}uv tool uninstall biopb${RESET}"
    fi

    # Optional purge of config + cached/state data. Never the user's images:
    # only biopb's own dotfile dirs are removed, never any configured data dir.
    if [ "$purge" = "1" ]; then
        _step "Purging config and cached data..."
        local d
        for d in \
            "$HOME/.config/biopb" \
            "$HOME/.config/biopb-mcp" \
            "$HOME/.local/share/biopb"; do
            if [ -e "$d" ]; then
                rm -rf "$d"
                _ok "Removed $d"
            fi
        done
        _info "Your image data was not touched."
    else
        _info "Config and cached data were kept. Remove them with --purge, or:"
        _cmd "  rm -rf ~/.config/biopb ~/.config/biopb-mcp ~/.local/share/biopb"
    fi

    printf "\n%s%s=== biopb uninstalled ===%s\n\n" "${BOLD}" "${GREEN}" "${RESET}"
    _info "uv and any AI agent (e.g. opencode) were left installed."
    echo ""
}

# --- entry point -------------------------------------------------------------
# Parse a tiny flag set so `... | bash -s -- --uninstall [--purge]` works while a
# bare `... | bash` still installs (no args => install, the unchanged default).
main() {
    local action="install" purge=0
    while [ $# -gt 0 ]; do
        case "$1" in
            --uninstall|--remove) action="uninstall" ;;
            --purge)              action="uninstall"; purge=1 ;;
            -h|--help)            _usage; return 0 ;;
            *) _err "Unknown option: $1"; _usage; return 2 ;;
        esac
        shift
    done

    if [ "$action" = "uninstall" ]; then
        uninstall_biopb "$purge"
    else
        install_biopb
    fi
}

# Only run if the script was fully downloaded (every function defined completely).
main "$@"
