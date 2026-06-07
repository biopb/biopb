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
function Select-DataDir {
    param([string]$BiopbHome)

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
    for ($i = 0; $i -lt $n; $i++) {
        Write-Host ("    {0}) " -f ($i + 1)) -ForegroundColor Cyan -NoNewline
        Write-Host $candidates[$i]
    }
    Write-Host ("    {0}) " -f $manualOpt) -ForegroundColor Cyan -NoNewline
    Write-Host "Enter path manually"
    Write-Host ""
    $choice = Read-Host "  Choice [1]"
    if ([string]::IsNullOrWhiteSpace($choice)) { $choice = "1" }

    if ($choice -eq "$manualOpt") {
        $manual = Read-Host "  Path [$defaultDir]"
        if ([string]::IsNullOrWhiteSpace($manual)) { return $defaultDir }
        return $manual
    }
    elseif ($choice -match '^\d+$' -and [int]$choice -ge 1 -and [int]$choice -le $n) {
        return $candidates[[int]$choice - 1]
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
        }
        return
    }

    $obj = [pscustomobject]@{ mcpServers = [pscustomobject]@{ biopb = $entry } }
    Set-FileUtf8NoBom -Path $File -Content ($obj | ConvertTo-Json -Depth 20)
    Write-Ok "${Label}: created $File"
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

    $detected = $false

    # --- Claude Code (managed through the `claude` CLI) ---
    if (Get-Command claude -ErrorAction SilentlyContinue) {
        $detected = $true
        & claude mcp get biopb *> $null
        if ($LASTEXITCODE -eq 0) {
            Write-Ok "Claude Code: biopb already registered"
        } else {
            & claude mcp add --scope user biopb -- $mcpCmd @mcpArgs *> $null
            if ($LASTEXITCODE -eq 0) {
                Write-Ok "Claude Code: registered biopb (user scope)"
            } else {
                Write-Warn2 "Claude Code detected but registration failed - add it manually:"
                Write-Cmd "claude mcp add --scope user biopb -- $mcpCmd $($mcpArgs -join ' ')"
            }
        }
    }

    # --- Claude Desktop (Windows: %APPDATA%\Claude) ---
    $cdCfg = Join-Path $env:APPDATA "Claude\claude_desktop_config.json"
    if (Test-Path -LiteralPath (Split-Path -Parent $cdCfg)) {
        $detected = $true
        Merge-McpJson -File $cdCfg -Command $mcpCmd -CmdArgs $mcpArgs -Label "Claude Desktop" -ConfigDir $ConfigDir
    }

    # --- Cursor ---
    $cursorDir = Join-Path $BiopbHome ".cursor"
    if (Test-Path -LiteralPath $cursorDir) {
        $detected = $true
        Merge-McpJson -File (Join-Path $cursorDir "mcp.json") -Command $mcpCmd -CmdArgs $mcpArgs -Label "Cursor" -ConfigDir $ConfigDir
    }

    # --- opencode ---
    $opencodeCfgDir = Join-Path $BiopbHome ".config\opencode"
    if ((Get-Command opencode -ErrorAction SilentlyContinue) -or (Test-Path -LiteralPath $opencodeCfgDir)) {
        $detected = $true
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
            } catch {
                Write-Warn2 "opencode: could not merge $opencodeCfg - add biopb manually"
                Write-Inf "Add under 'mcp' in $opencodeCfg :"
                Write-Host ("    `"biopb`": { `"type`": `"local`", `"command`": [`"$mcpCmd`", `"--transport`", `"stdio`"], `"enabled`": true }") -ForegroundColor DarkGray
            }
        } else {
            $opencodeObj = [pscustomobject]@{ mcp = [pscustomobject]@{ biopb = $opencodeEntry } }
            Set-FileUtf8NoBom -Path $opencodeCfg -Content ($opencodeObj | ConvertTo-Json -Depth 20)
            Write-Ok "opencode: created $opencodeCfg"
        }
    }

    if (-not $detected) {
        Write-Inf "No supported agent system detected (Claude Code, Claude Desktop, Cursor, opencode)."
        Write-Inf "To use biopb, register this stdio command with your MCP client:"
        Write-Cmd "$mcpCmd $($mcpArgs -join ' ')"
        Write-Inf "A ready-to-use definition is at: $ConfigDir\mcp.json"
    }

    # There is no separate server to start: the agent spawns biopb-mcp over stdio
    # on demand, which opens the napari window and brings up the data plane.
    Write-Inf "Your AI agent launches biopb-mcp automatically; a napari window opens when it does."
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
    Write-Step "[0/6] Checking system..."

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
    # Bio-Formats defaults to off: it pulls in a heavyweight Java toolchain that
    # most labs don't need (only legacy/proprietary formats require it).
    $sel = Select-Components -Labels @(
        "Built-in data browser",
        "biopb-mcp (MCP server)",
        "Bio-Formats support (ZVI, OIB, OIF, ...; auto-downloads Java on first use)"
    ) -Defaults @($true, $true, $false)
    $InstallWebapp     = $sel[0]
    $InstallMcp        = $sel[1]
    $InstallBioformats = $sel[2]
    Write-Host ""

    # ===== 1. Install uv + buf (if needed) =====
    Write-Step "[1/6] Ensuring build tools..."

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
    Write-Step "[2/6] Ensuring Python..."

    # biopb-mcp requires Python >= 3.10; otherwise 3.8 is sufficient.
    $minMinor = if ($InstallMcp) { 10 } else { 8 }

    $pythonOk = $false
    $pyExe = (Get-Command python -ErrorAction SilentlyContinue).Source
    if ($pyExe) {
        $verStr = & $pyExe -c "import sys; print(sys.version_info[0], sys.version_info[1])" 2>$null
        if ($LASTEXITCODE -eq 0 -and $verStr) {
            $parts = $verStr.Trim() -split '\s+'
            $maj = [int]$parts[0]; $min = [int]$parts[1]
            if ($maj -gt 3 -or ($maj -eq 3 -and $min -ge $minMinor)) {
                Write-Ok "Using system Python: $(& $pyExe --version)"
                $pythonOk = $true
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
    }

    # ===== 3. Install biopb packages =====
    Write-Step "[3/6] Installing biopb packages..."

    $tensorExtras = "web,ome-zarr,aics,medical,ndtiff"
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
        $biopbReq,
        "--with", $tensorReq,
        "--with-executables-from", "biopb-tensor-server"
    )
    if ($InstallMcp) {
        Write-Inf "  including biopb-mcp + napari"
        $installArgs += @(
            "--with", "biopb-mcp[mcp]>=0.6.0",
            "--with", "napari[all]",
            "--with-executables-from", "biopb-mcp"
        )
    }

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
    Write-Step "[4/6] Installing data browser..."

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
    Write-Step "[5/6] Config..."

    if (-not (Test-Path -LiteralPath $ConfigDir)) { New-Item -ItemType Directory -Force -Path $ConfigDir | Out-Null }
    $configFile = Join-Path $ConfigDir "biopb.toml"

    if (Test-Path -LiteralPath $configFile) {
        Write-Ok "Config exists at $configFile (preserved)"
    } else {
        if ($env:BIOPB_DATA_DIR) {
            $dataDir = $env:BIOPB_DATA_DIR
            Write-Ok "Using BIOPB_DATA_DIR: $dataDir"
        } else {
            $dataDir = Select-DataDir -BiopbHome $BiopbHome
            Write-Host ""
            Write-Ok "Data directory: $dataDir"
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

    # ===== 6. Wire biopb-mcp into the user's agent system =====
    Write-Step "[6/6] Configuring MCP client..."

    if ($InstallMcp) {
        Set-McpClients -BiopbHome $BiopbHome -ConfigDir $ConfigDir
    } else {
        Write-Inf "Skipped (biopb-mcp not installed)"
    }

    # ===== Summary =====
    Write-Host ""
    Write-Host "=== Installation Complete ===" -ForegroundColor Yellow

    if ($InstallMcp) {
        Write-Host "Your AI agent launches biopb-mcp over stdio - just start your agent" -ForegroundColor Green
        Write-Host "(Claude Code/Desktop, Cursor, opencode); a napari window opens with it." -ForegroundColor Green
        Write-Host ""
    }

    Write-Host "To launch the data server only without other components:" -ForegroundColor Green
    Write-Cmd "biopb server start"
    Write-Host ""

    if (-not $InstallWebapp -or -not $InstallMcp -or -not $InstallBioformats) {
        Write-Host "Optional components:" -ForegroundColor Green
    }
    if (-not $InstallWebapp) {
        Write-Note "Data browser not installed - rerun this script to install"
    } else {
        Write-Ok "Data browser available at http://localhost:8815"
    }
    if (-not $InstallMcp) {
        Write-Note "biopb-mcp not installed"
        Write-Note "to add it into the shared environment, rerun this script and enable it"
    }
    if (-not $InstallBioformats) {
        Write-Note "Bio-Formats not installed - ZVI/OIB/OIF and similar legacy formats unsupported"
        Write-Note "to add later, rerun this script and enable Bio-Formats, or:"
        Write-Cmd "         pip install `"biopb-tensor-server[bioformats]`""
    }
    if (-not $InstallWebapp -or -not $InstallMcp -or -not $InstallBioformats) { Write-Host "" }

    if ($InstallMcp) {
        Write-Host "biopb-mcp configuration file at:" -ForegroundColor Green
        Write-Cmd "         $BiopbHome\.config\biopb-mcp\config.json"
        Write-Host ""
    }

    Write-Host "Data server configuration file at:" -ForegroundColor Green
    Write-Cmd "         $configFile"
    Write-Host ""

    Write-Host "To upgrade: rerun this script" -ForegroundColor Green
    Write-Host ""
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
