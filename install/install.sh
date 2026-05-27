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
    _step "[0/5] Checking system..."

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
    read -r INSTALL_WEBAPP INSTALL_NAPARI <<< "$(_checkbox "Built-in data browser" "napari-biopb")"
    echo ""

    # ===== 1. Install uv + buf (if needed) =====
    _step "[1/5] Ensuring build tools..."

    export PATH="$HOME/.local/bin:$PATH"
    if ! command -v uv &>/dev/null; then
        _info "Installing uv..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        _ok "uv installed"
    else
        _ok "uv already installed ($(uv --version))"
    fi

    BUF_VERSION="1.70.0"
    BUF_BIN="/usr/local/bin"
    if ! command -v buf &>/dev/null || [ "$(buf --version 2>/dev/null)" != "$BUF_VERSION" ]; then
        _info "Installing buf $BUF_VERSION..."
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
    _step "[2/5] Ensuring Python..."

    PYTHON_VERSION=""
    if command -v python3 &>/dev/null; then
        PYTHON_VERSION=$(python3 -c "import sys; print(sys.version_info[:2])" 2>/dev/null || echo "")
        if [ -n "$PYTHON_VERSION" ]; then
            MAJOR=$(echo "$PYTHON_VERSION" | tr -d '(),' | cut -d' ' -f1)
            MINOR=$(echo "$PYTHON_VERSION" | tr -d '(),' | cut -d' ' -f2)
            if [ "$MAJOR" -gt 3 ] || { [ "$MAJOR" -eq 3 ] && [ "$MINOR" -ge 8 ]; }; then
                _ok "Using system Python: $(python3 --version)"
            else
                _warn "System Python too old ($(python3 --version)), need >= 3.8"
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
    _step "[3/5] Installing biopb packages..."

    _info "Installing biopb SDK..."
    uv tool install --upgrade \
        "biopb[tensor] @ $REPO"

    _info "Installing biopb-tensor-server..."
    uv tool install --upgrade \
        "biopb-tensor-server[web,ome-zarr,aics,medical,ndtiff] @ $REPO#subdirectory=biopb-tensor-server"

    if [ "$INSTALL_NAPARI" = "1" ]; then
        _info "Installing napari-biopb..."
        if command -v napari &>/dev/null; then
            uv tool install --upgrade napari --with napari-biopb
        else
            uv tool install "napari[all]" --with napari-biopb
        fi
        _ok "napari-biopb installed"
    fi

    VERSION_OUTPUT=$(biopb-tensor-server version 2>/dev/null || echo "installed")
    _ok "$VERSION_OUTPUT"

    # ===== 4. Webapp =====
    _step "[4/5] Installing data browser..."

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
    _step "[5/5] Config..."

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

[cache]
backend = "file"
file_max_segment_mb = 256
file_max_total_gb = 128

[[sources]]
url = "$TOML_DATA_DIR"
monitor = true
EOF
        _ok "Created: $CONFIG_FILE"
    fi

    # ===== Summary =====
    printf "\n%s%s%s\n" "${BOLD}" "${YELLOW}" "=== Installation Complete ===${YELLOW}"

    printf "%s%s%s\n" "${BOLD}" "${GREEN}" "To launch:${RESET}"
    _cmd "biopb server start"
    echo ""

    if [ "$INSTALL_NAPARI" = "1" ]; then
        printf "%s%s%s\n" "${BOLD}" "${GREEN}" "To launch napari-biopb:${RESET}"
        _cmd "napari -w napari-biopb"
        echo ""
    fi

    if [ "$INSTALL_WEBAPP" = "0" ] || [ "$INSTALL_NAPARI" = "0" ]; then
        printf "%s%s%s\n" "${BOLD}" "${GREEN}" "Optional components:${RESET}"
        if [ "$INSTALL_WEBAPP" = "0" ]; then
            _note "Data browser not installed — rerun this script to install"
        fi
        if [ "$INSTALL_NAPARI" = "0" ]; then
            _note "napari-biopb not installed"
            _note "to install separately:"
            _cmd "         uv tool install \"napari[all]\" --with napari-biopb"
            _note "or into an existing napari env:"
            _cmd "         pip install napari-biopb"
        fi
        echo ""
    fi

    printf "%s%s%s\n" "${BOLD}" "${GREEN}" "Configuration file available at:${RESET}"
    _cmd "         $CONFIG_FILE"
    echo ""

    printf "%s%s%s\n" "${BOLD}" "${GREEN}" "To upgrade: rerun this script${RESET}"
    echo ""
    echo ""
}

# Only run if script was fully downloaded (function defined completely)
install_biopb
