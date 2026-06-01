#!/bin/bash
#
# Biopb Tensor Server Installer
# Usage: curl -fsSL https://biopb.org/install.sh | bash
#
# Idempotent: rerun to upgrade to latest version
#
# Requirements: curl, git, tar
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

# Interactive checkbox menu. Redraws in place; all output goes to /dev/tty.
# Usage: _checkbox "Label one" "Label two" ...
# Prints space-separated 1/0 values (one per label) to stdout.
_checkbox() {
    local labels=("$@")
    local n=${#labels[@]}
    local sel=(); for ((i=0; i<n; i++)); do sel+=(1); done
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
    local -n _retvar=$1
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
        _retvar="${manual:-$default_dir}"
    elif [[ "$choice" =~ ^[0-9]+$ ]] && [ "$choice" -ge 1 ] && [ "$choice" -le "$n" ]; then
        _retvar="${candidates[$((choice - 1))]}"
    else
        printf "  Invalid choice, using default\n" >/dev/tty
        _retvar="$default_dir"
    fi
}

# Merge the biopb server into a standard `mcpServers` JSON config (Claude Desktop,
# Cursor, …). Uses jq to merge when present; otherwise creates the file if absent,
# or leaves an existing one untouched and tells the user to add biopb by hand.
# Usage: _mcp_json_merge <config-file> <command> <label>
_mcp_json_merge() {
    local file="$1" cmd="$2" label="$3"
    mkdir -p "$(dirname "$file")"

    if [ -f "$file" ]; then
        if command -v jq &>/dev/null; then
            local tmp="$file.biopb.tmp"
            if jq --arg c "$cmd" '.mcpServers.biopb = {command: $c, args: []}' "$file" > "$tmp" 2>/dev/null; then
                mv "$tmp" "$file"
                _ok "$label: registered biopb (merged into $file)"
            else
                rm -f "$tmp"
                _warn "$label: could not merge $file — add biopb manually (see $CONFIG_DIR/mcp.json)"
            fi
        else
            _warn "$label: $file already exists and jq is not installed"
            _info "Add the biopb entry manually (see $CONFIG_DIR/mcp.json)"
        fi
        return
    fi

    cat > "$file" << EOF
{
  "mcpServers": {
    "biopb": {
      "command": "$cmd",
      "args": []
    }
  }
}
EOF
    _ok "$label: created $file"
}

# Detect installed agent systems and register the biopb MCP server with each.
# Always drops a canonical, client-agnostic definition at $CONFIG_DIR/mcp.json.
# If nothing is detected, prints guidance so the user can wire it up themselves.
_setup_mcp() {
    local mcp_cmd
    mcp_cmd=$(command -v biopb-mcp 2>/dev/null || echo "biopb-mcp")

    # Minimal biopb-mcp config, mainly to ship preconfigured biopb.image servicers.
    # Preserved if it already exists so the user's tweaks survive a rerun.
    local mcp_config_dir="$HOME/.config/biopb-mcp"
    local mcp_config="$mcp_config_dir/config.json"
    mkdir -p "$mcp_config_dir"
    if [ -f "$mcp_config" ]; then
        _ok "biopb-mcp config exists at $mcp_config (preserved)"
    else
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
    cat > "$CONFIG_DIR/mcp.json" << EOF
{
  "mcpServers": {
    "biopb": {
      "command": "$mcp_cmd",
      "args": []
    }
  }
}
EOF
    _ok "MCP definition written: $CONFIG_DIR/mcp.json"

    local detected=0

    # --- Claude Code (managed through the `claude` CLI) ---
    if command -v claude &>/dev/null; then
        detected=1
        if claude mcp get biopb &>/dev/null; then
            _ok "Claude Code: biopb already registered"
        elif claude mcp add --scope user biopb -- "$mcp_cmd" &>/dev/null; then
            _ok "Claude Code: registered biopb (user scope)"
        else
            _warn "Claude Code detected but registration failed — add it manually:"
            _cmd "claude mcp add --scope user biopb -- $mcp_cmd"
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
            printf "    %sbiopb:\n      command: \"%s\"\n      args: []%s\n" "$DIM" "$mcp_cmd" "$RESET"
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
        _mcp_json_merge "$cd_cfg" "$mcp_cmd" "Claude Desktop"
    fi

    # --- Cursor ---
    if [ -d "$HOME/.cursor" ]; then
        detected=1
        _mcp_json_merge "$HOME/.cursor/mcp.json" "$mcp_cmd" "Cursor"
    fi

    if [ "$detected" = "0" ]; then
        _info "No supported agent system detected (Claude Code, Claude Desktop, Cursor, Hermes)."
        _info "To use biopb, point your MCP client at this command:"
        _cmd "$mcp_cmd"
        _info "A ready-to-use definition is at: $CONFIG_DIR/mcp.json"
    fi
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
        future_msg="New shells are set up automatically. For THIS terminal, run:"
    else
        future_msg="Add it to your shell profile, and run it now for THIS terminal:"
    fi

    # Full-width rules (no right border) so the banner is prominent but never
    # misaligns, regardless of how long $HOME / the path is.
    local rule="──────────────────────────────────────────────────────────────────"
    printf "\n${YELLOW}${BOLD}  %s${RESET}\n" "$rule"
    printf   "${YELLOW}${BOLD}  ⚠  ACTION REQUIRED — PATH not configured${RESET}\n"
    printf   "${YELLOW}${BOLD}  %s${RESET}\n" "$rule"
    printf   "  %s is not on your PATH; biopb commands live there.\n" "$bin_dir"
    printf   "  %s\n" "$future_msg"
    printf   "      ${CYAN}export PATH=\"\$HOME/.local/bin:\$PATH\"${RESET}\n"
    printf   "${YELLOW}${BOLD}  %s${RESET}\n\n" "$rule"
}

install_biopb() {
    set -euo pipefail

    REPO_URL="https://github.com/biopb/biopb"
    REPO="git+$REPO_URL"
    WEBAPP_DIR="$HOME/.local/share/biopb/webapp"
    CONFIG_DIR="$HOME/.config/biopb"

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

    # Check required tools
    for tool in curl git tar; do
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
    read -r INSTALL_WEBAPP INSTALL_MCP INSTALL_BIOFORMATS <<< "$(_checkbox \
        "Built-in data browser" \
        "biopb-mcp (MCP server)" \
        "Bio-Formats support (ZVI, OIB, OIF, ...; auto-downloads Java on first use)")"
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

    # ===== 2. Python =====
    _step "[2/6] Ensuring Python..."

    # biopb-mcp requires Python >= 3.10; otherwise 3.8 is sufficient.
    MIN_MINOR=8
    if [ "$INSTALL_MCP" = "1" ]; then
        MIN_MINOR=10
    fi

    PYTHON_VERSION=""
    if command -v python3 &>/dev/null; then
        PYTHON_VERSION=$(python3 -c "import sys; print(sys.version_info[:2])" 2>/dev/null || echo "")
        if [ -n "$PYTHON_VERSION" ]; then
            MAJOR=$(echo "$PYTHON_VERSION" | tr -d '(),' | cut -d' ' -f1)
            MINOR=$(echo "$PYTHON_VERSION" | tr -d '(),' | cut -d' ' -f2)
            if [ "$MAJOR" -gt 3 ] || { [ "$MAJOR" -eq 3 ] && [ "$MINOR" -ge "$MIN_MINOR" ]; }; then
                _ok "Using system Python: $(python3 --version)"
            else
                _warn "System Python too old ($(python3 --version)), need >= 3.$MIN_MINOR"
                PYTHON_VERSION=""
            fi
        fi
    fi

    if [ -z "$PYTHON_VERSION" ]; then
        _info "Installing Python 3.11 via uv..."
        uv python install 3.11
        _ok "Python 3.11 ready"
    fi

    # ===== 3. Install biopb packages =====
    _step "[3/6] Installing biopb packages..."

    TENSOR_EXTRAS="web,ome-zarr,aics,medical,ndtiff"
    if [ "$INSTALL_BIOFORMATS" = "1" ]; then
        TENSOR_EXTRAS="$TENSOR_EXTRAS,bioformats"
        _info "  including Bio-Formats (Java fetched on first use, not now)"
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
    # psutil) — without it `import mcp` fails. We require >=0.5.4: that release
    # drops biopb-mcp's stray, unpinned grpcio-tools dependency, which otherwise
    # collapses the shared solve to an unbuildable grpcio-tools==1.30.0.
    local install_args=(
        --upgrade
        --force
        "biopb[tensor] @ $REPO"
        --with "biopb-tensor-server[$TENSOR_EXTRAS] @ $REPO#subdirectory=biopb-tensor-server"
        --with-executables-from biopb-tensor-server
    )
    if [ "$INSTALL_MCP" = "1" ]; then
        _info "  including biopb-mcp + napari"
        install_args+=(
            --with "biopb-mcp[mcp]>=0.5.4"
            --with "napari[all]"
            --with-executables-from biopb-mcp
        )
    fi

    _info "Installing biopb into one shared environment..."
    uv tool install "${install_args[@]}"

    VERSION_OUTPUT=$(biopb-tensor-server version 2>/dev/null || echo "installed")
    _ok "$VERSION_OUTPUT"

    # ===== 4. Webapp =====
    _step "[4/6] Installing data browser..."

    if [ "$INSTALL_WEBAPP" = "1" ]; then
        mkdir -p "$WEBAPP_DIR"

        LATEST_TAG=$(curl -fsSL "https://api.github.com/repos/jiyuuchc/biopb/releases/latest" \
            | grep '"tag_name"' \
            | sed -E 's/.*"([^"]+)".*/\1/' \
            || echo "")

        if [ -n "$LATEST_TAG" ] && ! printf '%s' "$LATEST_TAG" | grep -qE '^[A-Za-z0-9._+/-]+$'; then
            _warn "Unexpected tag format, skipping data browser install"
            LATEST_TAG=""
        fi

        if [ -n "$LATEST_TAG" ]; then
            INSTALLED_TAG=""
            [ -f "$WEBAPP_DIR/.version" ] && INSTALLED_TAG=$(cat "$WEBAPP_DIR/.version")
            if [ "$INSTALLED_TAG" = "$LATEST_TAG" ]; then
                _ok "Data browser already up to date ($LATEST_TAG)"
            else
                _info "Downloading $LATEST_TAG..."
                rm -rf "${WEBAPP_DIR:?}"
                mkdir -p "$WEBAPP_DIR"
                curl -fsSL "$REPO_URL/releases/download/$LATEST_TAG/webapp.tar.gz" \
                    | tar -xzf - -C "$WEBAPP_DIR" --strip-components=1
                printf '%s' "$LATEST_TAG" > "$WEBAPP_DIR/.version"
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

    if [ "$INSTALL_MCP" = "1" ]; then
        _setup_mcp
    else
        _info "Skipped (biopb-mcp not installed)"
    fi

    # ===== Summary =====
    printf "\n%s%s%s\n" "${BOLD}" "${YELLOW}" "=== Installation Complete ===${YELLOW}"

    # Make sure the user can actually invoke the freshly installed commands.
    _ensure_local_bin_on_path "$ORIGINAL_PATH"

    if [ "$INSTALL_MCP" = "1" ]; then
        printf "%s%s%s\n" "${BOLD}" "${GREEN}" "To launch biopb-mcp as an MCP server directly:${RESET}"
        _cmd "biopb-mcp"
        echo ""
    fi

    printf "%s%s%s\n" "${BOLD}" "${GREEN}" "To launch the data server only without other components:${RESET}"
    _cmd "biopb server start"
    echo ""

    if [ "$INSTALL_WEBAPP" = "0" ] || [ "$INSTALL_MCP" = "0" ] || [ "$INSTALL_BIOFORMATS" = "0" ]; then
        printf "%s%s%s\n" "${BOLD}" "${GREEN}" "Optional components:${RESET}"
    fi
    if [ "$INSTALL_WEBAPP" = "0" ]; then
        _note "Data browser not installed — rerun this script to install"
    else
        _ok "Data browser available at http://localhost:8815"
    fi
    if [ "$INSTALL_MCP" = "0" ]; then
        _note "biopb-mcp not installed"
        _note "to add it into the shared environment, rerun this script and enable it"
    fi
    if [ "$INSTALL_BIOFORMATS" = "0" ]; then
        _note "Bio-Formats not installed — ZVI/OIB/OIF and similar legacy formats unsupported"
        _note "to add later, rerun this script and enable Bio-Formats, or:"
        _cmd "         pip install \"biopb-tensor-server[bioformats]\""
    fi
    if [ "$INSTALL_WEBAPP" = "0" ] || [ "$INSTALL_MCP" = "0" ] || [ "$INSTALL_BIOFORMATS" = "0" ]; then
        echo ""
    fi

    if [ "$INSTALL_MCP" = "1" ]; then
        printf "%s%s%s\n" "${BOLD}" "${GREEN}" "biopb-mcp configuration file at:${RESET}"
        _cmd "         $HOME/.config/biopb-mcp/config.json"
        echo ""
    fi

    printf "%s%s%s\n" "${BOLD}" "${GREEN}" "Data server configuration file at:${RESET}"
    _cmd "         $CONFIG_FILE"
    echo ""

    printf "%s%s%s\n" "${BOLD}" "${GREEN}" "To upgrade: rerun this script${RESET}"
    echo ""
    echo ""
}

# Only run if script was fully downloaded (function defined completely)
install_biopb
