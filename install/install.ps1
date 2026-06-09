<#
.SYNOPSIS
    Biopb Tensor Server Installer (Windows / PowerShell)

.DESCRIPTION
    Windows port of install/install.sh.
    Usage: irm https://biopb.org/install.ps1 | iex

    Idempotent: rerun to upgrade to latest version.

    By default this installs prebuilt wheels from the latest GitHub release.
    Set $env:BIOPB_INSTALL_FROM_SOURCE = "1" to instead build HEAD from a git
    checkout (the development fast path); that mode additionally needs git.

    Requirements: PowerShell 5.1+, tar (bundled on Windows 10 1803+);
                  git is needed only for BIOPB_INSTALL_FROM_SOURCE=1.

    Paths follow the same home-relative XDG layout the Python packages read
    (config under ~/.config, data under ~/.local/share); on Windows these
    resolve under %USERPROFILE%, matching Python's Path.home().
#>

# Stop on any error; mirror `set -euo pipefail`.
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'  # speeds up Invoke-WebRequest

# ----- Output helpers (color via Write-Host, robust across terminals) ---------
function Write-Step { param([string]$Msg) Write-Host ""; Write-Host $Msg -ForegroundColor White }
function Write-Ok   { param([string]$Msg) Write-Host "  $Msg" -ForegroundColor Green }
function Write-Inf  { param([string]$Msg) Write-Host "  $Msg" }
function Write-Warn2{ param([string]$Msg) Write-Host "  WARNING: " -ForegroundColor Yellow -NoNewline; Write-Host $Msg }
function Write-Err2 { param([string]$Msg) Write-Host "ERROR: $Msg" -ForegroundColor Red }
function Write-Note { param([string]$Msg) Write-Host "  NOTE: $Msg" -ForegroundColor DarkGray }
function Write-Cmd  { param([string]$Msg) Write-Host "  $Msg" -ForegroundColor Cyan }

# Write UTF-8 without a BOM. Windows PowerShell 5.1's `Set-Content -Encoding utf8`
# emits a BOM, which breaks Python tomllib and 5.1's own ConvertFrom-Json. Paths
# passed here are absolute, so .NET's cwd does not matter.
function Set-FileUtf8NoBom {
    param([string]$Path, [string]$Content)
    [System.IO.File]::WriteAllText($Path, $Content, (New-Object System.Text.UTF8Encoding($false)))
}

# Abort if the most recent native command failed. PowerShell does not honor
# $ErrorActionPreference='Stop' for external executables, so mirror `set -e`
# explicitly around the critical install steps.
function Assert-LastExit {
    param([string]$What)
    if ($LASTEXITCODE -ne 0) { throw "$What failed (exit code $LASTEXITCODE)" }
}

# Pause before the script ends so the final message stays readable. On Windows
# the host often closes the instant the script returns (double-click, or a window
# spawned just to run the installer), taking any goodbye or error text with it.
# Skipped when input is non-interactive (CI, piped stdin) so automation does not
# hang waiting on a keypress.
function Wait-ForExit {
    if ([Environment]::UserInteractive -and -not [Console]::IsInputRedirected) {
        Write-Host ""
        Read-Host "  Press Enter to exit" | Out-Null
    }
}

# Simplified component selector (replaces the bash in-place ANSI checkbox).
# Components default ON unless -Defaults supplies a per-label initial state;
# user types a number to toggle, Enter to confirm. Returns a bool[] aligned
# with $Labels.
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

# Prompt the user to choose a data directory; returns the chosen path string.
# With -KeepOption an extra "0) Keep my current config file" choice is shown as
# the default; selecting it (or Enter, or any invalid input) returns $null as a
# sentinel meaning "leave the existing config untouched." Callers pass this when
# a biopb.toml already exists, since hand-editing TOML is error-prone.
function Select-DataDir {
    param([string]$BiopbHome, [switch]$KeepOption)

    $candidates = New-Object System.Collections.Generic.List[string]
    $seen = New-Object System.Collections.Generic.HashSet[string]
    foreach ($d in @(
        $BiopbHome,
        (Join-Path $BiopbHome 'Documents'),
        (Join-Path $BiopbHome 'Data'),
        (Join-Path $BiopbHome 'data'),
        (Join-Path $BiopbHome 'Microscopy'),
        (Join-Path $BiopbHome 'OneDrive')
    )) {
        if ((Test-Path -LiteralPath $d) -and $seen.Add($d.ToLowerInvariant())) {
            $candidates.Add($d) | Out-Null
        }
    }
    # Offer non-system fixed drives (data drives) as roots.
    foreach ($drv in Get-PSDrive -PSProvider FileSystem -ErrorAction SilentlyContinue) {
        if ($drv.Name -ne 'C' -and $drv.Root -and (Test-Path -LiteralPath $drv.Root)) {
            if ($seen.Add($drv.Root.ToLowerInvariant())) { $candidates.Add($drv.Root) | Out-Null }
        }
    }

    $n = $candidates.Count
    $manualOpt = $n + 1
    $defaultDir = if ($n -gt 0) { $candidates[0] } else { $BiopbHome }

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
    # Default differs by mode: keep current config (0) vs first candidate (1).
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
        # In keep mode, an unrecognized choice is non-destructive: keep the config.
        Write-Host "  Invalid choice, keeping current config"
        return $null
    }
    else {
        Write-Host "  Invalid choice, using default"
        return $defaultDir
    }
}

# Ensure a directory is on the user PATH (persisted) and the current session.
function Add-ToUserPath {
    param([string]$Dir)
    $userPath = [Environment]::GetEnvironmentVariable('Path', 'User')
    if (($userPath -split ';') -notcontains $Dir) {
        $newPath = if ([string]::IsNullOrEmpty($userPath)) { $Dir } else { "$userPath;$Dir" }
        [Environment]::SetEnvironmentVariable('Path', $newPath, 'User')
    }
    if (($env:Path -split ';') -notcontains $Dir) { $env:Path = "$env:Path;$Dir" }
}

# Merge the biopb server into a standard mcpServers JSON config (Claude Desktop,
# Cursor, ...). Uses native JSON (ConvertFrom/ConvertTo-Json) in place of jq.
# biopb-mcp speaks stdio, so the client spawns the command itself — a bare
# command+args entry (no "type") is the form every mcpServers client accepts.
# Returns $true only if biopb was actually written into the client's config.
function Merge-McpJson {
    param([string]$File, [string]$Command, [string[]]$CmdArgs, [string]$Label, [string]$ConfigDir)
    $dir = Split-Path -Parent $File
    if (-not (Test-Path -LiteralPath $dir)) { New-Item -ItemType Directory -Force -Path $dir | Out-Null }

    $entry = [pscustomobject]@{ command = $Command; args = @($CmdArgs) }

    if (Test-Path -LiteralPath $File) {
        try {
            $json = Get-Content -Raw -LiteralPath $File | ConvertFrom-Json
            if ($null -eq $json.mcpServers) {
                $json | Add-Member -NotePropertyName mcpServers -NotePropertyValue ([pscustomobject]@{}) -Force
            }
            if ($json.mcpServers.PSObject.Properties.Name -contains 'biopb') {
                $json.mcpServers.biopb = $entry
            } else {
                $json.mcpServers | Add-Member -NotePropertyName biopb -NotePropertyValue $entry -Force
            }
            Set-FileUtf8NoBom -Path $File -Content ($json | ConvertTo-Json -Depth 20)
            Write-Ok "${Label}: registered biopb (merged into $File)"
        } catch {
            Write-Warn2 "${Label}: could not merge $File - add biopb manually (see $ConfigDir\mcp.json)"
            return $false
        }
        return $true
    }

    $obj = [pscustomobject]@{ mcpServers = [pscustomobject]@{ biopb = $entry } }
    Set-FileUtf8NoBom -Path $File -Content ($obj | ConvertTo-Json -Depth 20)
    Write-Ok "${Label}: created $File"
    return $true
}

# Detect installed agent systems and register the biopb MCP server with each.
function Set-McpClients {
    param([string]$BiopbHome, [string]$ConfigDir)

    $mcpCmd = (Get-Command biopb-mcp -ErrorAction SilentlyContinue).Source
    if (-not $mcpCmd) { $mcpCmd = "biopb-mcp" }

    # biopb-mcp 0.6.0+ speaks MCP over stdio: the AI agent spawns it as a child
    # process (`biopb-mcp --transport stdio`) rather than connecting to a
    # long-running HTTP server. Each client needs the *command* to run, not a URL;
    # we register the resolved absolute path so GUI agents (e.g. Claude Desktop),
    # which don't inherit the shell PATH, can still find it.
    $mcpArgs = @("--transport", "stdio")

    # Minimal biopb-mcp config (preconfigured biopb.image servicers). Preserved
    # if it already exists so the user's tweaks survive a rerun.
    $mcpConfigDir = Join-Path $BiopbHome ".config\biopb-mcp"
    $mcpConfig = Join-Path $mcpConfigDir "config.json"
    if (-not (Test-Path -LiteralPath $mcpConfigDir)) { New-Item -ItemType Directory -Force -Path $mcpConfigDir | Out-Null }
    if (Test-Path -LiteralPath $mcpConfig) {
        Write-Ok "biopb-mcp config exists at $mcpConfig (preserved)"
    } else {
        $mcpConfigContent = @'
{
  "mcp": {
    "services": {
      "process_image_servers": [
        "grpcs://cellpose.biopb.org:443"
      ]
    }
  }
}
'@
        Set-FileUtf8NoBom -Path $mcpConfig -Content $mcpConfigContent
        Write-Ok "Created biopb-mcp config: $mcpConfig"
    }

    # Canonical standalone definition (standard mcpServers JSON; most clients accept it).
    $canonical = [pscustomobject]@{
        mcpServers = [pscustomobject]@{ biopb = [pscustomobject]@{ command = $mcpCmd; args = $mcpArgs } }
    }
    Set-FileUtf8NoBom -Path (Join-Path $ConfigDir "mcp.json") -Content ($canonical | ConvertTo-Json -Depth 20)
    Write-Ok "MCP definition written: $ConfigDir\mcp.json"

    # Assume the user has no working wiring until a branch below actually writes
    # biopb into a client's config; cleared ($false) only on a real registration,
    # so a detected-but-unregistered client (failed `claude mcp add` or an
    # unwritable config) still gets the canonical fallback at the end.
    $needToShowMcpConfig = $true

    # --- Claude Code (managed through the `claude` CLI) ---
    if (Get-Command claude -ErrorAction SilentlyContinue) {
        & claude mcp get biopb *> $null
        if ($LASTEXITCODE -eq 0) {
            Write-Ok "Claude Code: biopb already registered"
            $needToShowMcpConfig = $false
        } else {
            & claude mcp add --scope user biopb -- $mcpCmd @mcpArgs *> $null
            if ($LASTEXITCODE -eq 0) {
                Write-Ok "Claude Code: registered biopb (user scope)"
                $needToShowMcpConfig = $false
            } else {
                Write-Warn2 "Claude Code detected but registration failed - add it manually:"
                Write-Cmd "claude mcp add --scope user biopb -- $mcpCmd $($mcpArgs -join ' ')"
            }
        }
    }

    # --- Claude Desktop (Windows: %APPDATA%\Claude) ---
    $cdCfg = Join-Path $env:APPDATA "Claude\claude_desktop_config.json"
    if (Test-Path -LiteralPath (Split-Path -Parent $cdCfg)) {
        if (Merge-McpJson -File $cdCfg -Command $mcpCmd -CmdArgs $mcpArgs -Label "Claude Desktop" -ConfigDir $ConfigDir) {
            $needToShowMcpConfig = $false
        }
    }

    # --- Cursor ---
    $cursorDir = Join-Path $BiopbHome ".cursor"
    if (Test-Path -LiteralPath $cursorDir) {
        if (Merge-McpJson -File (Join-Path $cursorDir "mcp.json") -Command $mcpCmd -CmdArgs $mcpArgs -Label "Cursor" -ConfigDir $ConfigDir) {
            $needToShowMcpConfig = $false
        }
    }

    # --- opencode ---
    $opencodeCfgDir = Join-Path $BiopbHome ".config\opencode"
    if ((Get-Command opencode -ErrorAction SilentlyContinue) -or (Test-Path -LiteralPath $opencodeCfgDir)) {
        $opencodeCfg = Join-Path $opencodeCfgDir "opencode.json"
        if (-not (Test-Path -LiteralPath $opencodeCfgDir)) { New-Item -ItemType Directory -Force -Path $opencodeCfgDir | Out-Null }

        $opencodeEntry = [pscustomobject]@{
            type = "local"
            command = @($mcpCmd) + $mcpArgs
            enabled = $true
        }

        if (Test-Path -LiteralPath $opencodeCfg) {
            try {
                $json = Get-Content -Raw -LiteralPath $opencodeCfg | ConvertFrom-Json
                if ($null -eq $json.mcp) {
                    $json | Add-Member -NotePropertyName mcp -NotePropertyValue ([pscustomobject]@{}) -Force
                }
                if ($json.mcp.PSObject.Properties.Name -contains 'biopb') {
                    $json.mcp.biopb = $opencodeEntry
                } else {
                    $json.mcp | Add-Member -NotePropertyName biopb -NotePropertyValue $opencodeEntry -Force
                }
                Set-FileUtf8NoBom -Path $opencodeCfg -Content ($json | ConvertTo-Json -Depth 20)
                Write-Ok "opencode: registered biopb (merged into $opencodeCfg)"
                $needToShowMcpConfig = $false
            } catch {
                Write-Warn2 "opencode: could not merge $opencodeCfg - add biopb manually"
                Write-Inf "Add under 'mcp' in $opencodeCfg :"
                Write-Host ("    `"biopb`": { `"type`": `"local`", `"command`": [`"$mcpCmd`", `"--transport`", `"stdio`"], `"enabled`": true }") -ForegroundColor DarkGray
            }
        } else {
            $opencodeObj = [pscustomobject]@{ mcp = [pscustomobject]@{ biopb = $opencodeEntry } }
            Set-FileUtf8NoBom -Path $opencodeCfg -Content ($opencodeObj | ConvertTo-Json -Depth 20)
            Write-Ok "opencode: created $opencodeCfg"
            $needToShowMcpConfig = $false
        }
    }

    # Nothing auto-wired biopb into a client. Defer the notice to the final
    # summary (via a script-scoped flag) so it groups with the other warnings.
    $script:McpNeedsManual = $needToShowMcpConfig
}

$ISSUE_URL = "https://github.com/biopb/biopb/issues/new"

function Show-Banner {
    Write-Host ""
    Write-Host "    ____  _       ____  ____  " -ForegroundColor Cyan
    Write-Host "   / __ )(_)___  / __ \/ __ ) " -ForegroundColor Cyan
    Write-Host "  / __  / / __ \/ /_/ / __  |" -ForegroundColor Cyan
    Write-Host " / /_/ / / /_/ / ____/ /_/ / " -ForegroundColor Cyan
    Write-Host "/_____/_/\____/_/   /_____/  " -ForegroundColor Cyan
    Write-Host ""
    Write-Host "      Tensor Server Installer"
    Write-Host ""
}

# Two light-hearted pre-flight confirmations. Returns $true if the user opted
# in to filing a bug report should things go sideways. Exits if they bail.
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

# Fetch the latest GitHub release metadata (a parsed object with .tag_name and
# .assets[].name / .browser_download_url). One call serves both the wheels and
# the data browser. Throws on network/HTTP error; callers catch and fall back.
function Get-LatestRelease {
    param([string]$Repo)
    return Invoke-RestMethod -Uri "https://api.github.com/repos/$Repo/releases/latest" `
        -Headers @{ "User-Agent" = "biopb-installer" }
}

# Print the tail of the server log, indented, for diagnosing a bad startup.
function Show-LogTail {
    param([string]$LogFile)
    if (-not (Test-Path -LiteralPath $LogFile)) { return }
    Write-Inf "recent server log ($LogFile):"
    Get-Content -LiteralPath $LogFile -Tail 15 -ErrorAction SilentlyContinue | ForEach-Object {
        Write-Host "      $_" -ForegroundColor DarkGray
    }
}

# Start (or restart) the background data server, then report its health.
# Best-effort: never aborts the install. Skip with $env:BIOPB_NO_SERVER_START=1.
# Starting now lets the pre-cache warm overviews before the user opens anything,
# and a restart makes an already-running (stale) server pick up the new code.
function Start-DataServer {
    param([string]$BiopbHome, [string]$ConfigFile)

    $logFile = Join-Path $BiopbHome ".local\share\biopb\logs\tensor-server.log"

    if ($env:BIOPB_NO_SERVER_START -eq "1") {
        Write-Inf "Skipping server start (BIOPB_NO_SERVER_START=1)"
        Write-Inf "  start it later with: biopb server start"
        return
    }
    if (-not (Get-Command biopb -ErrorAction SilentlyContinue)) {
        Write-Warn2 "biopb not found on PATH; skipping server start"
        Write-Inf "  start it later with: biopb server start"
        return
    }

    # 'restart' loads the just-installed code if a server is already running,
    # and is a plain start otherwise.
    try { & biopb server restart *> $null } catch { }

    # Ask the daemon for its health, polling until SERVING (or 60s). Merge stderr
    # (live progress: "data server starting - N found so far...") into the stream
    # and surface it via Write-Inf as it arrives; the JSON verdict (the line
    # starting with '{') on stdout is captured for parsing below.
    $result = @{ json = $null }
    try {
        & biopb server status --json --wait 60 2>&1 | ForEach-Object {
            $s = "$_"
            if ($s.TrimStart().StartsWith("{")) { $result.json = $s.Trim() }
            elseif ($s.Trim()) { Write-Inf $s.Trim() }
        }
    } catch { }
    $out = $result.json
    if (-not $out) { $out = "" }

    # Tolerate an older biopb that predates --json/--wait: fall back to a plain
    # liveness check so the installer still works during a version transition.
    if (-not $out) {
        $plain = ""
        try { $plain = (& biopb server status 2>$null | Out-String) } catch { $plain = "" }
        if ($plain -match "Running") {
            Write-Ok "Data server started"
        } else {
            Write-Warn2 "Data server may not have started"
            Show-LogTail -LogFile $logFile
            Write-Inf "  full log: $logFile"
        }
        return
    }

    $health = $null; $count = $null
    try {
        $obj = $out | ConvertFrom-Json
        $health = $obj.health
        $count = $obj.source_count
    } catch { }

    if ($health -ne "SERVING") {
        Write-Warn2 "Data server did not come up cleanly"
        Write-Inf "  it may still be scanning a large folder, or failed to start:"
        Show-LogTail -LogFile $logFile
        Write-Inf "  full log: $logFile"
        return
    }

    if ((-not $count) -or ($count -eq 0)) {
        Write-Warn2 "Data server is running but found no data sources"
        Write-Inf "  check that your data folder holds supported images (see config):"
        Write-Cmd "  $ConfigFile"
        Show-LogTail -LogFile $logFile
        return
    }

    Write-Ok "Data server ready - $count data source(s) found; pre-caching overviews"
}

function Install-Biopb {
    $RepoUrl     = "https://github.com/biopb/biopb"
    $Repo        = "git+$RepoUrl"
    $ReleaseRepo = "biopb/biopb"             # owner/name for the GitHub Releases API
    $BiopbHome   = $env:USERPROFILE          # matches Python Path.home() on Windows
    $WebappDir   = Join-Path $BiopbHome ".local\share\biopb\webapp"
    $ConfigDir   = Join-Path $BiopbHome ".config\biopb"
    $LocalBin    = Join-Path $BiopbHome ".local\bin"

    # Default: install prebuilt wheels from the latest GitHub release (no git/buf
    # build). Set $env:BIOPB_INSTALL_FROM_SOURCE = "1" to build HEAD from git.
    $InstallFromSource = ($env:BIOPB_INSTALL_FROM_SOURCE) -and ($env:BIOPB_INSTALL_FROM_SOURCE -ne '0')
    $release = $null   # cached release metadata, fetched on demand below

    # ===== 0. System Check =====
    Write-Step "[0/7] Checking system..."

    $arch = $env:PROCESSOR_ARCHITECTURE
    switch ($arch) {
        "AMD64" { $bufArch = "x86_64" }
        "ARM64" { $bufArch = "arm64" }
        default {
            Write-Err2 "Unsupported architecture: $arch"
            Write-Inf "Supported: x86_64 (AMD64), arm64 (ARM64)"
            Wait-ForExit
            exit 1
        }
    }
    Write-Ok "Platform: Windows ($arch)"

    # Required tools. curl/Invoke-WebRequest is built in; we always need tar, and
    # git only when building from a source checkout.
    $requiredTools = @("tar")
    if ($InstallFromSource) { $requiredTools += "git" }
    foreach ($tool in $requiredTools) {
        if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) {
            Write-Err2 "$tool not found"
            switch ($tool) {
                "git"  { Write-Inf "Install: winget install Git.Git  (then reopen PowerShell)" }
                "tar"  { Write-Inf "tar ships with Windows 10 1803+; please update Windows" }
            }
            Wait-ForExit
            exit 1
        }
        Write-Ok "${tool}: found"
    }
    Write-Ok "System check passed"

    # ===== Optional components =====
    # biopb-mcp is always installed (it is the primary interface), so it is not
    # offered here. Bio-Formats defaults to off: it pulls in a heavyweight Java
    # toolchain that most labs don't need (only legacy/proprietary formats need it).
    $sel = Select-Components -Labels @(
        "Built-in data viewer: see all your images in a browser (Chrome, Safari and others)",
        "Bio-Formats (more image formats; needs Java and extra setup during first run)"
    ) -Defaults @($true, $false)
    $InstallWebapp     = $sel[0]
    $InstallBioformats = $sel[1]
    Write-Host ""

    # ===== 1. Install uv + buf (if needed) =====
    Write-Step "[1/7] Ensuring build tools..."

    # uv's installer (piped in below) voluntarily aborts unless the effective
    # execution policy is one of Unrestricted/RemoteSigned/Bypass -- even though
    # code piped through `iex` is otherwise exempt from policy enforcement. Set
    # the Process scope so that self-check passes: it is session-only (gone when
    # this window closes), needs no admin, and -- importantly -- cannot override
    # a GPO-enforced policy, so a genuinely locked-down machine still wins, which
    # is the correct outcome. We only touch it when the current policy would
    # block uv, and never fail the install if even that is disallowed.
    $allowedPolicy = @('Unrestricted', 'RemoteSigned', 'Bypass')
    if ((Get-ExecutionPolicy).ToString() -notin $allowedPolicy) {
        Write-Note "Temporarily allowing scripts for this session (execution policy -> RemoteSigned, Process scope) so the uv installer can run."
        Set-ExecutionPolicy RemoteSigned -Scope Process -Force -ErrorAction SilentlyContinue
    }

    Add-ToUserPath $LocalBin
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
        Write-Inf "Installing uv..."
        Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
        # uv installs to %USERPROFILE%\.local\bin; make it usable this session.
        Add-ToUserPath $LocalBin
        if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
            Write-Err2 "uv installation did not land on PATH - reopen PowerShell and rerun"
            Wait-ForExit
            exit 1
        }
        Write-Ok "uv installed"
    } else {
        Write-Ok "uv already installed ($(uv --version))"
    }

    # buf generates the protobuf/Flight stubs at build time, so it is only needed
    # when building from a source checkout. Release wheels ship the stubs prebuilt.
    if ($InstallFromSource) {
        $bufVersion = "1.70.0"
        $bufPath = Join-Path $LocalBin "buf.exe"
        $bufCurrent = ""
        if (Get-Command buf -ErrorAction SilentlyContinue) { $bufCurrent = (buf --version 2>$null) }
        if ($bufCurrent -ne $bufVersion) {
            Write-Inf "Installing buf $bufVersion..."
            if (-not (Test-Path -LiteralPath $LocalBin)) { New-Item -ItemType Directory -Force -Path $LocalBin | Out-Null }
            if (Test-Path -LiteralPath $bufPath) { Remove-Item -LiteralPath $bufPath -Force }
            $bufUrl = "https://github.com/bufbuild/buf/releases/download/v$bufVersion/buf-Windows-$bufArch.exe"
            Invoke-WebRequest -Uri $bufUrl -OutFile $bufPath
            Write-Ok "buf $bufVersion installed"
        } else {
            Write-Ok "buf already installed ($bufCurrent)"
        }
    }

    # ===== 2. Python =====
    Write-Step "[2/7] Ensuring Python..."

    # biopb-mcp (always installed) requires Python >= 3.10.
    $minMinor = 10

    # $pythonSpec is the interpreter we pin `uv tool install` to below via --python.
    # Without it, uv auto-discovers an interpreter and may pick an old system
    # python (e.g. 3.9) that then fails biopb-mcp's Requires-Python>=3.10.
    $pythonOk = $false
    $pythonSpec = ""
    $pyExe = (Get-Command python -ErrorAction SilentlyContinue).Source
    if ($pyExe) {
        $verStr = & $pyExe -c "import sys; print(sys.version_info[0], sys.version_info[1])" 2>$null
        if ($LASTEXITCODE -eq 0 -and $verStr) {
            $parts = $verStr.Trim() -split '\s+'
            $maj = [int]$parts[0]; $min = [int]$parts[1]
            if ($maj -gt 3 -or ($maj -eq 3 -and $min -ge $minMinor)) {
                Write-Ok "Using system Python: $(& $pyExe --version)"
                $pythonOk = $true
                $pythonSpec = $pyExe
            } else {
                Write-Warn2 "System Python too old ($(& $pyExe --version)), need >= 3.$minMinor"
            }
        }
    }
    if (-not $pythonOk) {
        Write-Inf "Installing Python 3.11 via uv..."
        uv python install 3.11
        Assert-LastExit "Python install"
        Write-Ok "Python 3.11 ready"
        $pythonSpec = "3.11"
    }

    # ===== 3. Install biopb packages =====
    Write-Step "[3/7] Installing biopb packages..."

    $tensorExtras = "web,aics,medical,ndtiff,hdf5"
    if ($InstallBioformats) {
        $tensorExtras = "$tensorExtras,bioformats"
        Write-Inf "  including Bio-Formats (Java fetched on first use, not now)"
    }

    # Resolve where biopb + biopb-tensor-server come from. They must be installed
    # as a matched pair from a single build: the tensor server is self-contained
    # and may use proto fields newer than any biopb on PyPI, so biopb is pinned to
    # the sibling artifact (a git ref in source mode, a local wheel in release
    # mode) and the resolver is never allowed to pull it from PyPI.
    if ($InstallFromSource) {
        Write-Inf "Building from source (HEAD of $RepoUrl)"
        $biopbReq  = "biopb[tensor] @ $Repo"
        $tensorReq = "biopb-tensor-server[$tensorExtras] @ $Repo#subdirectory=biopb-tensor-server"
    } else {
        try { $release = Get-LatestRelease -Repo $ReleaseRepo } catch { $release = $null }
        if (-not $release) {
            Write-Err2 "Could not fetch the latest biopb release from $ReleaseRepo."
            Write-Inf "Check your network, or build from source by setting:"
            Write-Cmd '$env:BIOPB_INSTALL_FROM_SOURCE = "1"'
            throw "release fetch failed"
        }
        $sdkAsset    = $release.assets | Where-Object { $_.name -match '^biopb-.*\.whl$' } | Select-Object -First 1
        $tensorAsset = $release.assets | Where-Object { $_.name -match '^biopb_tensor_server-.*\.whl$' } | Select-Object -First 1
        if (-not $sdkAsset -or -not $tensorAsset) {
            Write-Err2 "Release $($release.tag_name) has no biopb wheels attached."
            Write-Inf "Build from source by setting:"
            Write-Cmd '$env:BIOPB_INSTALL_FROM_SOURCE = "1"'
            throw "release wheels missing"
        }
        Write-Inf "Installing from release $($release.tag_name)"
        $wheelsDir = Join-Path $env:TEMP "biopb-wheels"
        if (Test-Path -LiteralPath $wheelsDir) { Remove-Item -LiteralPath $wheelsDir -Recurse -Force }
        New-Item -ItemType Directory -Force -Path $wheelsDir | Out-Null
        $sdkWhl    = Join-Path $wheelsDir $sdkAsset.name
        $tensorWhl = Join-Path $wheelsDir $tensorAsset.name
        Invoke-WebRequest -Uri $sdkAsset.browser_download_url -OutFile $sdkWhl
        Invoke-WebRequest -Uri $tensorAsset.browser_download_url -OutFile $tensorWhl
        # Direct file:// references pin biopb to this exact wheel, so uv resolves
        # the server's biopb dependency to it rather than to PyPI.
        $biopbReq  = "biopb[tensor] @ $(([System.Uri]$sdkWhl).AbsoluteUri)"
        $tensorReq = "biopb-tensor-server[$tensorExtras] @ $(([System.Uri]$tensorWhl).AbsoluteUri)"
    }

    # Install everything into ONE uv tool environment so the components can
    # import and drive each other at runtime:
    #   - `biopb server start` runs `sys.executable -m biopb_tensor_server.cli`,
    #     so the server must be importable from biopb's interpreter;
    #   - biopb-mcp is a napari plugin + MCP server that talks to the tensor
    #     server and runs a napari viewer in this same env.
    # biopb is the primary tool (exposes the `biopb` command); --with adds the
    # siblings to the same env and --with-executables-from also links their
    # console scripts onto PATH (plain --with does not expose executables).
    #
    # biopb-mcp requires the [mcp] extra (mcp, uvicorn, jupyter_client,
    # ipykernel, psutil) - without it `import mcp` fails. We require >=0.6.0:
    # that release makes stdio the default transport (matching the MCP client
    # config this installer writes) and also drops biopb-mcp's stray, unpinned
    # grpcio-tools dependency, which otherwise collapses the shared solve to an
    # unbuildable grpcio-tools==1.30.0. It comes from PyPI in both modes.
    $installArgs = @(
        "tool", "install", "--upgrade", "--force",
        "--python", $pythonSpec,
        $biopbReq,
        "--with", $tensorReq,
        "--with-executables-from", "biopb-tensor-server"
    )
    Write-Inf "  including biopb-mcp + napari"
    $installArgs += @(
        "--with", "biopb-mcp[mcp]>=0.6.0",
        "--with", "napari[all]",
        "--with-executables-from", "biopb-mcp"
    )

    Write-Inf "Installing biopb into one shared environment..."
    try {
        uv @installArgs
        Assert-LastExit "biopb install"
    } finally {
        # Remove the temporary wheel download dir on success or failure.
        # $wheelsDir is only set in release mode; $null (source mode) is a no-op.
        if ($wheelsDir -and (Test-Path -LiteralPath $wheelsDir)) {
            Remove-Item -LiteralPath $wheelsDir -Recurse -Force -ErrorAction SilentlyContinue
        }
    }

    # Refresh PATH so freshly installed tool shims are visible this session.
    Add-ToUserPath $LocalBin
    $versionOutput = (biopb-tensor-server version 2>$null)
    if (-not $versionOutput) { $versionOutput = "installed" }
    Write-Ok "$versionOutput"

    # ===== 4. Webapp =====
    Write-Step "[4/7] Installing data browser..."

    if ($InstallWebapp) {
        if (-not (Test-Path -LiteralPath $WebappDir)) { New-Item -ItemType Directory -Force -Path $WebappDir | Out-Null }

        # Reuse the release metadata already fetched for the wheels; in source
        # mode this is the first (and only) fetch.
        if (-not $release) { try { $release = Get-LatestRelease -Repo $ReleaseRepo } catch { $release = $null } }
        $latestTag = if ($release) { $release.tag_name } else { "" }

        if ($latestTag -and ($latestTag -notmatch '^[A-Za-z0-9._+/-]+$')) {
            Write-Warn2 "Unexpected tag format, skipping data browser install"
            $latestTag = ""
        }

        if ($latestTag) {
            $versionFile = Join-Path $WebappDir ".version"
            $installedTag = if (Test-Path -LiteralPath $versionFile) { (Get-Content -Raw -LiteralPath $versionFile).Trim() } else { "" }
            if ($installedTag -eq $latestTag) {
                Write-Ok "Data browser already up to date ($latestTag)"
            } else {
                Write-Inf "Downloading $latestTag..."
                Remove-Item -LiteralPath $WebappDir -Recurse -Force -ErrorAction SilentlyContinue
                New-Item -ItemType Directory -Force -Path $WebappDir | Out-Null
                $webAsset = $release.assets | Where-Object { $_.name -eq 'webapp.tar.gz' } | Select-Object -First 1
                $webUrl = if ($webAsset) { $webAsset.browser_download_url } else { "$RepoUrl/releases/download/$latestTag/webapp.tar.gz" }
                $tarball = Join-Path $env:TEMP "biopb-webapp.tar.gz"
                Invoke-WebRequest -Uri $webUrl -OutFile $tarball
                tar -xzf $tarball -C $WebappDir --strip-components=1
                Remove-Item -LiteralPath $tarball -Force -ErrorAction SilentlyContinue
                Set-FileUtf8NoBom -Path $versionFile -Content $latestTag
                Write-Ok "Data browser installed to: $WebappDir"
            }
        } else {
            Write-Warn2 "Could not fetch latest release, data browser not installed"
            Write-Inf "Server will run in API-only mode"
        }
    } else {
        Write-Inf "Skipped"
    }

    # ===== 5. Config =====
    Write-Step "[5/7] Config..."

    if (-not (Test-Path -LiteralPath $ConfigDir)) { New-Item -ItemType Directory -Force -Path $ConfigDir | Out-Null }
    $configFile = Join-Path $ConfigDir "biopb.toml"

    # We never edit an existing biopb.toml in place — hand-editing TOML is
    # error-prone and most users can't do it. Instead the data-directory prompt is
    # always offered; when a config already exists it gains a default "0) Keep my
    # current config file" option, and choosing a fresh data dir backs up the old
    # config and writes a brand-new one. $null $dataDir is the "keep" sentinel.
    $dataDir = $null
    $keepConfig = $false
    if ((Test-Path -LiteralPath $configFile) -and (-not $env:BIOPB_DATA_DIR)) {
        $dataDir = Select-DataDir -BiopbHome $BiopbHome -KeepOption
        Write-Host ""
        if ($null -eq $dataDir) { $keepConfig = $true }
    }
    elseif (Test-Path -LiteralPath $configFile) {
        # BIOPB_DATA_DIR is a non-interactive override; it only applies to a fresh
        # install. With a config already present we keep it (its data dir wins).
        Write-Note "BIOPB_DATA_DIR is set but a config already exists; keeping it (remove $configFile to apply BIOPB_DATA_DIR)."
        $keepConfig = $true
    }
    elseif ($env:BIOPB_DATA_DIR) {
        $dataDir = $env:BIOPB_DATA_DIR
        Write-Ok "Using BIOPB_DATA_DIR: $dataDir"
    }
    else {
        $dataDir = Select-DataDir -BiopbHome $BiopbHome
        Write-Host ""
        Write-Ok "Data directory: $dataDir"
    }

    if ($keepConfig) {
        Write-Ok "Keeping current config: $configFile"
    } else {
        # Preserve the previous config (a chosen new data dir must never silently
        # discard the user's old settings) by renaming it to a timestamped backup.
        if (Test-Path -LiteralPath $configFile) {
            $backup = "$configFile.bak." + (Get-Date -Format "yyyyMMddHHmmss")
            Move-Item -LiteralPath $configFile -Destination $backup -Force
            Write-Inf "Backed up previous config to $backup"
        }

        # Escape for a TOML basic string: backslashes first, then quotes.
        $tomlDataDir = $dataDir -replace '\\', '\\' -replace '"', '\"'
        $tomlContent = @"
[server]
host = "127.0.0.1"
port = 8815
aggressive_dir_pruning = true

# Cache decoded chunks on disk as Arrow IPC segments so repeat reads skip
# re-decoding the raw format. The file backend is now Windows-safe -- it copies
# batches off the segment mmap so eviction can unlink the file (copy-on-read,
# biopb/biopb#5). Matches the POSIX installer.
[cache]
backend = "file"
file_max_segment_mb = 256
file_max_total_gb = 128

[metadata_db]
enabled = true

[[sources]]
url = "$tomlDataDir"
monitor = true
"@
        Set-FileUtf8NoBom -Path $configFile -Content $tomlContent
        Write-Ok "Created: $configFile"
    }

    # ===== 6. Start the data server =====
    # Before MCP wiring so a typo in the data dir (step 5) surfaces right after
    # the choice, while pre-cache gets the earliest possible head start.
    Write-Step "[6/7] Starting data server..."
    Start-DataServer -BiopbHome $BiopbHome -ConfigFile $configFile

    # ===== 7. Wire biopb-mcp into the user's agent system =====
    Write-Step "[7/7] Configuring MCP client..."

    Set-McpClients -BiopbHome $BiopbHome -ConfigDir $ConfigDir

    # ===== Summary =====
    # Two groups, in order: all informational blocks, then all warnings. Every
    # block is one headline line, optional indented detail lines, then one blank
    # line. Set-McpClients only sets a flag, so its warning lands in the warning
    # group below rather than printing inline.
    Write-Host ""
    Write-Host "=== Installation Complete ===" -ForegroundColor Yellow
    Write-Host ""

    # Headlines via Write-Inf (indent 2, matching Write-Ok/Write-Warn2), detail
    # lines indent 4.
    # --- informational blocks ---
    Write-Inf "Your AI agent launches biopb-mcp over stdio - just start your agent"
    Write-Inf "(Claude Code/Desktop, Cursor, opencode); a napari window opens with it."
    Write-Host ""

    if ($InstallWebapp) {
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

    # --- warnings ---
    if (-not $InstallWebapp) {
        Write-Warn2 "Data browser not installed"
        Write-Inf "  rerun this script to install it"
        Write-Host ""
    }

    if ($script:McpNeedsManual) {
        Write-Warn2 "biopb is not registered with any MCP client"
        Write-Inf "  register it manually using $ConfigDir\mcp.json"
        Write-Host ""
    }
}

Show-Banner
$reportOnFailure = Invoke-Preflight

try {
    Install-Biopb
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
