<#
.SYNOPSIS
    biopb stack installer (Windows / PowerShell) -- interactive console front-end

.DESCRIPTION
    Usage: irm https://biopb.org/install.ps1 | iex

    Idempotent: rerun to upgrade to latest version.

    This is the INTERACTIVE CONSOLE front-end. It collects the user's choices
    (component toggles, data directory) with friendly prompts, then hands them to
    the headless install engine (biopb-engine.ps1), which does the real work.
    The Inno Setup GUI wizard is a second front-end over the same engine -- one
    install brain, two skins, so the console and GUI can never drift.

    The engine is loaded from a sibling biopb-engine.ps1 when present (local
    checkout / unpacked installer), otherwise downloaded from biopb.org (the
    `irm | iex` path, where there is no script file on disk).

    This installs prebuilt wheels from the latest biopb release-v* GitHub
    deployment. By default it tracks the latest STABLE release; set
    $env:BIOPB_INSTALL_RC = "1" to track the latest release candidate.

    Requirements: PowerShell 5.1+, tar (bundled on Windows 10 1803+).
#>

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

# Where to fetch the engine from when running via `irm | iex` (no script on disk).
# Served from biopb.org alongside install.ps1 (see docs/release-model.md).
$EngineUrl = "https://biopb.org/biopb-engine.ps1"

$ISSUE_URL = "https://github.com/biopb/biopb-mcp/issues/new"

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
    if ([Environment]::UserInteractive -and -not [Console]::IsInputRedirected) {
        Write-Host ""
        Read-Host "  Press Enter to exit" | Out-Null
    }
}

# Locate (or download) the headless engine and dot-source it so its functions --
# Invoke-BiopbInstall, Get-DataDirCandidates, the Report-* renderers -- are
# available in this session. Dot-sourcing (not running) means the engine's
# auto-run block is skipped; we drive it explicitly via Invoke-BiopbInstall.
function Import-Engine {
    # $PSScriptRoot is empty under `irm | iex` (no file on disk) -> download.
    $local = if ($PSScriptRoot) { Join-Path $PSScriptRoot 'biopb-engine.ps1' } else { $null }
    if ($local -and (Test-Path -LiteralPath $local)) {
        . $local
        return
    }
    Write-Inf "Fetching install engine..."
    $engineSrc = Invoke-RestMethod -Uri $EngineUrl
    # Dot-source the downloaded text in THIS scope so its functions persist.
    # (`iex` in a function would define them in the function's child scope.)
    $tmp = Join-Path $env:TEMP "biopb-engine.ps1"
    Set-Content -LiteralPath $tmp -Value $engineSrc -Encoding UTF8
    . $tmp
}

# Interactive component selector (replaces the engine's parameters with prompts).
# Components default ON unless -Defaults supplies a per-label initial state; user
# types a number to toggle, Enter to confirm. Returns a bool[] aligned with $Labels.
function Select-Components {
    param([string[]]$Labels, [bool[]]$Defaults)
    $n = $Labels.Count
    $sel = New-Object bool[] $n
    for ($i = 0; $i -lt $n; $i++) {
        $sel[$i] = if ($Defaults -and $i -lt $Defaults.Count) { $Defaults[$i] } else { $true }
    }

    while ($true) {
        Write-Host ""
        Write-Host "  Optional components:" -ForegroundColor White
        Write-Host ""
        for ($i = 0; $i -lt $n; $i++) {
            $mark  = if ($sel[$i]) { "[x]" } else { "[ ]" }
            $color = if ($sel[$i]) { "Green" } else { "DarkGray" }
            Write-Host ("    {0}. " -f ($i + 1)) -NoNewline
            Write-Host $mark -ForegroundColor $color -NoNewline
            Write-Host ("  {0}" -f $Labels[$i])
        }
        Write-Host ""
        $choice = Read-Host "  Toggle [1-$n] or Enter to confirm"
        if ([string]::IsNullOrWhiteSpace($choice)) { break }
        if ($choice -match '^\d+$') {
            $idx = [int]$choice - 1
            if ($idx -ge 0 -and $idx -lt $n) { $sel[$idx] = -not $sel[$idx] }
        }
    }
    return $sel
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
        Write-Inf "Data browser available at http://localhost:8815"
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
        Write-Warn2 "Data browser not installed"
        Write-Inf "  rerun this script to install it"
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
    Import-Engine   # defines Invoke-BiopbInstall, Get-DataDirCandidates, Report-*

    $BiopbHome  = $env:USERPROFILE
    $configFile = Join-Path $BiopbHome ".config\biopb\biopb.toml"

    # ----- Collect choices interactively (the GUI collects these on wizard pages) -----

    # biopb-mcp is always installed (primary interface), so it is not offered.
    # Bio-Formats defaults off (heavyweight Java toolchain most labs don't need).
    $sel = Select-Components -Labels @(
        "Built-in data viewer: see all your images in a browser (Chrome, Safari and others)",
        "Bio-Formats (more image formats; needs Java and extra setup during first run)"
    ) -Defaults @($true, $false)
    $installWebapp     = $sel[0]
    $installBioformats = $sel[1]

    # Data directory / keep-config decision, mirroring the original flow.
    $dataDir = ""
    $keepConfig = $false
    $configExists = Test-Path -LiteralPath $configFile
    if ($configExists -and (-not $env:BIOPB_DATA_DIR)) {
        $picked = Select-DataDir -BiopbHome $BiopbHome -KeepOption
        if ($null -eq $picked) { $keepConfig = $true } else { $dataDir = $picked }
    }
    elseif ($configExists) {
        # BIOPB_DATA_DIR is a fresh-install override only; an existing config wins.
        Write-Note "BIOPB_DATA_DIR is set but a config already exists; keeping it (remove $configFile to apply it)."
        $keepConfig = $true
    }
    elseif ($env:BIOPB_DATA_DIR) {
        $dataDir = $env:BIOPB_DATA_DIR
        Write-Ok "Using BIOPB_DATA_DIR: $dataDir"
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
        Write-Host ""
        Write-Inf "BioPB ships with algorithm plugins that use remote servers for"
        Write-Inf "certain computations, e.g. cell segmentation. The servers are"
        Write-Inf "hosted at UConn Health and log client IP addresses."
        Write-Host ""
        $plug = Read-Host "  Enable the remote algorithm plugins? [Y/n]"
        if ($plug -match '^(n|no)$') { $noRemotePlugins = $true }
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
