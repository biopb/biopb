<#
.SYNOPSIS
    biopb stack installer (Windows / PowerShell) -- interactive console front-end

.DESCRIPTION
    Usage: irm https://biopb.org/install.ps1 | iex

    Idempotent: rerun to upgrade to latest version.

    This is the INTERACTIVE CONSOLE front-end. It collects the user's choices
    (data directory, remote-plugin consent) with friendly prompts, then hands them
    to the headless install engine (biopb-engine.ps1), which does the real work.
    Component selection is no longer prompted -- the web interface installs by
    default and Bio-Formats is opt-in via $env:BIOPB_INSTALL_BIOFORMATS.
    The Inno Setup GUI wizard is a second front-end over the same engine -- one
    install brain, two skins, so the console and GUI can never drift.

    The engine is loaded from a sibling biopb-engine.ps1 when present (local
    checkout / unpacked installer), otherwise downloaded from biopb.org (the
    `irm | iex` path, where there is no script file on disk).

    This installs prebuilt wheels from the latest biopb release-v* GitHub
    deployment. By default it tracks the latest STABLE release; set
    $env:BIOPB_INSTALL_RC = "1" to track the latest release candidate.

    Unattended upgrades: set $env:BIOPB_NONINTERACTIVE = "1" to suppress every
    prompt (keeps an existing config; leaves remote plugins off unless
    $env:BIOPB_REMOTE_PLUGINS = "1"). It is an upgrade feature -- a FRESH unattended
    install must also set $env:BIOPB_DATA_DIR or it errors out.

    Requirements: PowerShell 5.1+, tar (bundled on Windows 10 1803+).
#>

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

# Where to fetch the engine from when running via `irm | iex` (no script on disk).
# Served from biopb.org alongside install.ps1 (see docs/release-model.md).
$EngineUrl = "https://biopb.org/biopb-engine.ps1"

$ISSUE_URL = "https://github.com/biopb/biopb-mcp/issues/new"

# Non-interactive / unmanned mode: $env:BIOPB_NONINTERACTIVE = "1" suppresses every
# prompt so the installer can upgrade unattended (Task Scheduler, CI, image bakes).
# The upgrade path is the common case: an existing config is kept untouched. A fresh
# unattended install uses $env:BIOPB_DATA_DIR (else a default) and leaves the remote
# algorithm plugins OFF unless $env:BIOPB_REMOTE_PLUGINS = "1" -- consent can't be
# asked unattended, so the off-site IP-logging servers are never silently enabled.
$script:NonInteractive = [bool]$env:BIOPB_NONINTERACTIVE -and ($env:BIOPB_NONINTERACTIVE -ne '0')

# ----- Output helpers used by the front-end's own prompts/summary -------------
function Write-Step { param([string]$Msg) Write-Host ""; Write-Host $Msg -ForegroundColor White }
function Write-Ok   { param([string]$Msg) Write-Host "  $Msg" -ForegroundColor Green }
function Write-Inf  { param([string]$Msg) Write-Host "  $Msg" }
function Write-Warn2{ param([string]$Msg) Write-Host "  WARNING: " -ForegroundColor Yellow -NoNewline; Write-Host $Msg }
function Write-Err2 { param([string]$Msg) Write-Host "ERROR: $Msg" -ForegroundColor Red }
function Write-Note { param([string]$Msg) Write-Host "  NOTE: $Msg" -ForegroundColor DarkGray }
function Write-Cmd  { param([string]$Msg) Write-Host "  $Msg" -ForegroundColor Cyan }

# Pause before the script ends so the final message stays readable. Skipped when
# input is non-interactive (CI, piped stdin) so automation does not hang.
function Wait-ForExit {
    if ($script:NonInteractive) { return }
    if ([Environment]::UserInteractive -and -not [Console]::IsInputRedirected) {
        Write-Host ""
        Read-Host "  Press Enter to exit" | Out-Null
    }
}

# Locate (or download) the headless engine and return its SOURCE TEXT (not a
# path). The caller dot-sources it as an in-memory scriptblock at SCRIPT scope --
# dot-sourcing inside this function would define the engine's functions
# (Invoke-BiopbInstall, the Report-* renderers) in this function's child scope,
# where they vanish on return and are unavailable to Main. Returning the text and
# dot-sourcing in Main keeps them alive for the whole session. Dot-sourcing (not
# running) also skips the engine's auto-run block; we drive it via Invoke-BiopbInstall.
#
# We return TEXT rather than a .ps1 path on purpose: a factory-default Windows
# client ships ExecutionPolicy=Restricted, which blocks dot-sourcing a script
# *file* -- so writing the downloaded engine to %TEMP% and dot-sourcing it fails
# mid-install with a cryptic PSSecurityException, even though the `irm | iex`
# entry ran fine (in-memory expressions are never policy-gated). An in-memory
# scriptblock is the same in-memory path: it runs under Restricted, needs no
# temp file, and keeps $MyInvocation.InvocationName '.' so the auto-run guard
# still skips. (AllSigned / GPO-locked machines still need a signed engine --
# nothing short of signing helps there.)
function Resolve-EngineSource {
    # $PSScriptRoot is empty under `irm | iex` (no file on disk) -> download.
    $local = if ($PSScriptRoot) { Join-Path $PSScriptRoot 'biopb-engine.ps1' } else { $null }
    if ($local -and (Test-Path -LiteralPath $local)) {
        return (Get-Content -Raw -LiteralPath $local)
    }
    Write-Inf "Fetching install engine..."
    return (Invoke-RestMethod -Uri $EngineUrl)
}

function Show-Banner {
    Write-Host ""
    Write-Host "    ____  _       ____  ____  " -ForegroundColor Cyan
    Write-Host "   / __ )(_)___  / __ \/ __ ) " -ForegroundColor Cyan
    Write-Host "  / __  / / __ \/ /_/ / __  |" -ForegroundColor Cyan
    Write-Host " / /_/ / / /_/ / ____/ /_/ / " -ForegroundColor Cyan
    Write-Host "/_____/_/\____/_/   /_____/  " -ForegroundColor Cyan
    Write-Host ""
    Write-Host "      biopb stack installer"
    Write-Host ""
}

# A light-hearted pre-flight confirmation on interactive runs: offer to file a
# bug report if the install goes sideways. Returns $true if the user opted in
# (default yes); unattended runs skip it and return $false.
function Invoke-Preflight {
    # Unattended runs skip the pre-flight banter; don't nag with a bug-report
    # offer no one is watching for.
    if ($script:NonInteractive) { return $false }
    Write-Host ""
    Write-Host "  --- Before we begin ---" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "  If this script faceplants, a 30-second bug report helps us teach"
    Write-Host "  it to land on its feet. Nothing is sent anywhere automatically --"
    Write-Host "  you would just paste the error into a GitHub issue yourself."
    $rep = Read-Host "  Willing to file a quick report if it breaks? [Y/n]"
    $reportOnFailure = ($rep -notmatch '^(n|no)$')   # default: yes
    if ($reportOnFailure) {
        Write-Host "  You're a hero. Onward!" -ForegroundColor Green
    } else {
        Write-Host "  No worries -- onward anyway!" -ForegroundColor Green
    }
    return $reportOnFailure
}

# Render the human-facing summary from the engine's result object. Wording is a
# front-end concern, so it lives here rather than in the engine.
function Show-Summary {
    param($Result)
    $BiopbHome = $Result.BiopbHome
    $configFile = $Result.ConfigFile
    $ConfigDir = $Result.ConfigDir

    Write-Host ""
    Write-Host "=== Installation Complete ===" -ForegroundColor Yellow
    Write-Host ""

    Write-Inf "Your AI agent launches biopb-mcp over stdio - just start your agent"
    Write-Inf "(Claude Code/Desktop, Cursor, opencode); a napari window opens with it."
    Write-Host ""

    if ($Result.WebappInstalled) {
        Write-Inf "Web interface available at http://localhost:8813"
        Write-Inf "  open it anytime with: biopb dashboard (or the Desktop shortcut)"
        Write-Host ""
    }

    Write-Inf "biopb-mcp configuration file:"
    Write-Cmd "  $ConfigDir\mcp-config.json"
    Write-Host ""

    Write-Inf "Data server configuration file:"
    Write-Cmd "  $configFile"
    Write-Host ""

    Write-Inf "To upgrade: rerun this script"
    Write-Host ""

    # Discoverability for the opt-in Defender exclusion (issue #384). Bytecode is
    # already precompiled at install (admin-free); this is the extra, privileged
    # win that needs one UAC prompt, so we only point at it rather than doing it.
    Write-Inf "Faster startup (optional): exclude the biopb install from Windows"
    Write-Inf "Defender real-time scanning (needs admin - one UAC prompt):"
    Write-Cmd "  biopb quick-start --enable"
    Write-Host ""

    if (-not $Result.WebappInstalled) {
        Write-Warn2 "Web interface not installed (download failed)"
        Write-Inf "  the dashboard won't work until you rerun this script to fetch it"
        Write-Host ""
    }

    if ($Result.McpNeedsManual) {
        Write-Warn2 "biopb is not registered with any MCP client"
        Write-Inf "  register it manually using $ConfigDir\mcp.json"
        Write-Host ""
    }
}

# ============================================================================
# Main
# ============================================================================
Show-Banner
$reportOnFailure = Invoke-Preflight

try {
    # Dot-source at SCRIPT scope (not inside a helper) so the engine's functions --
    # Invoke-BiopbInstall, Report-* -- persist through Main. An in-memory
    # scriptblock (not a .ps1 file) so it runs under a Restricted ExecutionPolicy.
    . ([scriptblock]::Create((Resolve-EngineSource)))

    $BiopbHome  = $env:USERPROFILE
    # Canonical config is biopb.json (biopb/biopb#34); a legacy biopb.toml from a
    # pre-#34 install still counts as "a config exists" for the keep prompt.
    $configDir  = Get-BiopbTree "XDG_CONFIG_HOME" ".config"
    $configFile = Join-Path $configDir "biopb.json"
    $legacyConfig = Join-Path $configDir "biopb.toml"

    # ----- Resolve component choices (no longer prompted -- biopb/biopb#237) -----

    # Components are no longer offered interactively. The web interface is mandatory
    # (the dashboard is the SPA; a legacy $env:BIOPB_INSTALL_WEBAPP=0 is ignored).
    # Bio-Formats stays off by default: the Python adapters cover the formats most
    # labs use, and it pulls a heavyweight Java toolchain. Overridable for scripted
    # installs via env var:
    #   $env:BIOPB_INSTALL_BIOFORMATS = "1"  add Bio-Formats (Java fetched on first use)
    $installBioformats = ($env:BIOPB_INSTALL_BIOFORMATS -eq '1')

    # Data directory / keep-config decision. No prompt: a fresh install lets the
    # engine seed the sample-image bundle and point the config at it (empty
    # $dataDir + $keepConfig=$false is the engine's "seed samples" signal), so a
    # non-CLI user lands on real data with zero questions. An existing config is
    # kept UNTOUCHED. BIOPB_DATA_DIR still overrides on a fresh install (skips the
    # samples). Set $env:BIOPB_INSTALL_SAMPLES=0 to seed nothing.
    $dataDir = ""
    $keepConfig = $false
    $configExists = (Test-Path -LiteralPath $configFile) -or (Test-Path -LiteralPath $legacyConfig)
    if ($configExists -and (-not $env:BIOPB_DATA_DIR)) {
        # Existing config, no override: keep it exactly as-is (upgrade fast path).
        $keepConfig = $true
        Write-Note "Existing config found; keeping it as-is."
    }
    elseif ($configExists) {
        # BIOPB_DATA_DIR is a fresh-install override only; an existing config wins.
        $existing = if (Test-Path -LiteralPath $configFile) { $configFile } else { $legacyConfig }
        Write-Note "BIOPB_DATA_DIR is set but a config already exists; keeping it (remove $existing to apply it)."
        $keepConfig = $true
    }
    elseif ($env:BIOPB_DATA_DIR) {
        $dataDir = $env:BIOPB_DATA_DIR
        Write-Ok "Using BIOPB_DATA_DIR: $dataDir"
    }
    else {
        # Fresh install, no override: the engine seeds samples and points there
        # (unless BIOPB_INSTALL_SAMPLES=0 opts out, in which case it points the
        # config at an empty folder for drag-drop). Don't claim seeding when the
        # user opted out -- the engine prints the actual outcome either way.
        if ($env:BIOPB_INSTALL_SAMPLES -eq '0') {
            Write-Note "Fresh install: sample seeding disabled (BIOPB_INSTALL_SAMPLES=0); starting with an empty data folder."
        } else {
            Write-Ok "Fresh install: seeding sample images."
        }
    }

    # Remote algorithm plugins consent. The default plugins point at off-site
    # servers (cell segmentation, etc.) hosted at UConn Health that log client
    # IPs, so ask before enabling them rather than quietly shipping a third-party
    # network dependency. Only fires when no biopb-mcp config exists yet, so a
    # prior choice survives a rerun. Default is Yes (Enter = enable).
    $noRemotePlugins = $false
    $mcpConfig = Join-Path (Get-BiopbTree "XDG_CONFIG_HOME" ".config") "mcp-config.json"
    if (-not (Test-Path -LiteralPath $mcpConfig)) {
        if ($script:NonInteractive) {
            # Consent can't be asked unattended: enable only on explicit opt-in.
            if ($env:BIOPB_REMOTE_PLUGINS -eq '1') {
                Write-Ok "Remote algorithm plugins enabled (BIOPB_REMOTE_PLUGINS=1)"
            } else {
                $noRemotePlugins = $true
                Write-Ok "Remote algorithm plugins disabled (non-interactive; set BIOPB_REMOTE_PLUGINS=1 to enable)"
            }
        } else {
            Write-Host ""
            Write-Inf "BioPB ships with algorithm plugins that use remote servers for"
            Write-Inf "certain computations, e.g. cell segmentation. The servers are"
            Write-Inf "hosted at UConn Health and log client IP addresses."
            Write-Host ""
            $plug = Read-Host "  Enable the remote algorithm plugins? [Y/n]"
            if ($plug -match '^(n|no)$') { $noRemotePlugins = $true }
        }
    }

    # ----- Drive the engine in-process, rendering its progress in color (-Mode console) -----
    $result = Invoke-BiopbInstall `
        -DataDir $dataDir `
        -Bioformats:$installBioformats `
        -KeepConfig:$keepConfig `
        -NoRemotePlugins:$noRemotePlugins `
        -Mode console

    Show-Summary -Result $result
    Wait-ForExit
} catch {
    Write-Host ""
    Write-Err2 "Well, that didn't go to plan: $($_.Exception.Message)"
    Write-Host ""
    if ($reportOnFailure) {
        Write-Host "  As promised -- please toss the faceplant our way so we can fix it:" -ForegroundColor Yellow
        Write-Cmd $ISSUE_URL
        Write-Host "  Paste the error above plus your Windows version (run ``winver``). Thank you!"
    } else {
        Write-Host "  No report, no problem. If you change your mind, the door's here:" -ForegroundColor Yellow
        Write-Cmd $ISSUE_URL
    }
    Write-Host ""
    Wait-ForExit
    exit 1
}
