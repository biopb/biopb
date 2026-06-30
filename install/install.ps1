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
    prompt (keeps an existing config; a fresh install uses $env:BIOPB_DATA_DIR or
    a default, and leaves remote plugins off unless $env:BIOPB_REMOTE_PLUGINS = "1").

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

# Locate (or download) the headless engine and return the path to dot-source.
# The caller must dot-source the returned path at SCRIPT scope -- dot-sourcing
# inside this function would define the engine's functions (Invoke-BiopbInstall,
# Get-DataDirCandidates, the Report-* renderers) in this function's child scope,
# where they vanish on return and are unavailable to Main. Returning the path and
# dot-sourcing in Main keeps them alive for the whole session. Dot-sourcing (not
# running) also skips the engine's auto-run block; we drive it via Invoke-BiopbInstall.
function Resolve-EnginePath {
    # $PSScriptRoot is empty under `irm | iex` (no file on disk) -> download.
    $local = if ($PSScriptRoot) { Join-Path $PSScriptRoot 'biopb-engine.ps1' } else { $null }
    if ($local -and (Test-Path -LiteralPath $local)) {
        return $local
    }
    Write-Inf "Fetching install engine..."
    $engineSrc = Invoke-RestMethod -Uri $EngineUrl
    $tmp = Join-Path $env:TEMP "biopb-engine.ps1"
    Set-Content -LiteralPath $tmp -Value $engineSrc -Encoding UTF8
    return $tmp
}

# Prompt for a data directory; returns the chosen path string. With -KeepOption an
# extra "0) Keep my current config file" choice is the default; selecting it (or
# Enter / invalid input) returns $null, the "leave existing config untouched"
# sentinel. Uses the engine's Get-DataDirCandidates for the candidate list.
function Select-DataDir {
    param([string]$BiopbHome, [switch]$KeepOption)

    $candidates = @(Get-DataDirCandidates -BiopbHome $BiopbHome)
    $n = $candidates.Count
    $manualOpt = $n + 1
    # Fall back to a dedicated data subfolder, never the profile root.
    $defaultDir = if ($n -gt 0) { $candidates[0] } else { (Join-Path $BiopbHome 'Microscopy') }

    Write-Host ""
    Write-Host "  Select your microscopy data directory:" -ForegroundColor White
    Write-Host ""
    if ($KeepOption) {
        Write-Host "    0) " -ForegroundColor Cyan -NoNewline
        Write-Host "Keep my current config file (default)"
    }
    for ($i = 0; $i -lt $n; $i++) {
        Write-Host ("    {0}) " -f ($i + 1)) -ForegroundColor Cyan -NoNewline
        Write-Host $candidates[$i]
    }
    Write-Host ("    {0}) " -f $manualOpt) -ForegroundColor Cyan -NoNewline
    Write-Host "Enter path manually"
    Write-Host ""
    $defaultChoice = if ($KeepOption) { "0" } else { "1" }
    $choice = Read-Host "  Choice [$defaultChoice]"
    if ([string]::IsNullOrWhiteSpace($choice)) { $choice = $defaultChoice }

    if ($KeepOption -and $choice -eq "0") { return $null }   # sentinel: keep config

    if ($choice -eq "$manualOpt") {
        $manual = Read-Host "  Path [$defaultDir]"
        if ([string]::IsNullOrWhiteSpace($manual)) { return $defaultDir }
        return $manual
    }
    elseif ($choice -match '^\d+$' -and [int]$choice -ge 1 -and [int]$choice -le $n) {
        return $candidates[[int]$choice - 1]
    }
    elseif ($KeepOption) {
        Write-Host "  Invalid choice, keeping current config"
        return $null
    }
    else {
        Write-Host "  Invalid choice, using default"
        return $defaultDir
    }
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

# Two light-hearted pre-flight confirmations. Returns $true if the user opted in
# to filing a bug report should things go sideways. Exits if they bail.
function Invoke-Preflight {
    # Unattended runs skip the pre-flight banter; don't nag with a bug-report
    # offer no one is watching for.
    if ($script:NonInteractive) { return $false }
    Write-Host ""
    Write-Host "  --- Before we begin ---" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "  The Windows installer is shiny, new, and proudly experimental."
    Write-Host "  It has been road-tested about as much as a chocolate teapot, so"
    Write-Host "  expect the occasional wobble while we knock off the rough edges."
    $go = Read-Host "  Feeling brave enough to continue? [y/N]"
    if ($go -notmatch '^(y|yes)$') {
        Write-Host ""
        Write-Host "  Wise. We'll be right here once it's more house-trained. Cheers!" -ForegroundColor Cyan
        Wait-ForExit
        exit 0
    }

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
        Write-Inf "Web interface available at http://localhost:8814"
        Write-Host ""
    }

    Write-Inf "biopb-mcp configuration file:"
    Write-Cmd "  $BiopbHome\.config\biopb-mcp\config.json"
    Write-Host ""

    Write-Inf "Data server configuration file:"
    Write-Cmd "  $configFile"
    Write-Host ""

    Write-Inf "To upgrade: rerun this script"
    Write-Host ""

    if (-not $Result.WebappInstalled) {
        Write-Warn2 "Web interface not installed (`$env:BIOPB_INSTALL_WEBAPP = `"0`")"
        Write-Inf "  rerun without that env var to install it"
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
    # Invoke-BiopbInstall, Get-DataDirCandidates, Report-* -- persist through Main.
    . (Resolve-EnginePath)

    $BiopbHome  = $env:USERPROFILE
    # Canonical config is biopb.json (biopb/biopb#34); a legacy biopb.toml from a
    # pre-#34 install still counts as "a config exists" for the keep prompt.
    $configDir  = Join-Path $BiopbHome ".config\biopb"
    $configFile = Join-Path $configDir "biopb.json"
    $legacyConfig = Join-Path $configDir "biopb.toml"

    # ----- Resolve component choices (no longer prompted -- biopb/biopb#237) -----

    # Components are no longer offered interactively. The web interface now carries
    # the server admin page (config / status / restart) on top of the image viewer,
    # so it installs by default. Bio-Formats stays off by default: the Python
    # adapters cover the formats most labs use, and it pulls a heavyweight Java
    # toolchain. Both remain overridable for scripted installs via env vars:
    #   $env:BIOPB_INSTALL_WEBAPP = "0"      skip the web interface (API-only server)
    #   $env:BIOPB_INSTALL_BIOFORMATS = "1"  add Bio-Formats (Java fetched on first use)
    $installWebapp     = ($env:BIOPB_INSTALL_WEBAPP -ne '0')
    $installBioformats = ($env:BIOPB_INSTALL_BIOFORMATS -eq '1')

    # Data directory / keep-config decision, mirroring the original flow.
    $dataDir = ""
    $keepConfig = $false
    $configExists = (Test-Path -LiteralPath $configFile) -or (Test-Path -LiteralPath $legacyConfig)
    if ($configExists -and (-not $env:BIOPB_DATA_DIR)) {
        # Existing config + no override: keep it. Non-interactive skips the prompt
        # (the unmanned-upgrade fast path); interactive offers "0) keep" as default.
        if ($script:NonInteractive) {
            $keepConfig = $true
            Write-Note "Non-interactive: keeping existing config."
        } else {
            $picked = Select-DataDir -BiopbHome $BiopbHome -KeepOption
            if ($null -eq $picked) { $keepConfig = $true } else { $dataDir = $picked }
        }
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
    elseif ($script:NonInteractive) {
        # Fresh unattended install with no data dir given: fall back to a dedicated
        # subfolder (never the profile root, which OneDrive placeholders can hang).
        $dataDir = Join-Path $BiopbHome 'Microscopy'
        Write-Note "Non-interactive: no BIOPB_DATA_DIR set; defaulting to $dataDir"
    }
    else {
        $dataDir = Select-DataDir -BiopbHome $BiopbHome
        Write-Ok "Data directory: $dataDir"
    }

    # Remote algorithm plugins consent. The default plugins point at off-site
    # servers (cell segmentation, etc.) hosted at UConn Health that log client
    # IPs, so ask before enabling them rather than quietly shipping a third-party
    # network dependency. Only fires when no biopb-mcp config exists yet, so a
    # prior choice survives a rerun. Default is Yes (Enter = enable).
    $noRemotePlugins = $false
    $mcpConfig = Join-Path $BiopbHome ".config\biopb-mcp\config.json"
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
        -Webapp:$installWebapp `
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
