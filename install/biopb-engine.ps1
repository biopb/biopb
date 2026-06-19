<#
.SYNOPSIS
    biopb stack install engine (headless, Windows / PowerShell)

.DESCRIPTION
    The non-interactive core extracted from install/install.ps1. It performs the
    actual install orchestration (uv -> Python -> wheel triple -> webapp -> config
    -> server -> MCP wiring) and takes EVERY user choice as a parameter -- no
    Read-Host, no banner, no exit/pause. It is the single "install brain" driven
    by two front-ends:

      * install/install.ps1        - the interactive console front-end (irm | iex)
      * the Inno Setup GUI wizard  - runs this file as a child process and parses
                                     its tagged progress stream (see -Mode gui)

    DUAL USE
      1. Dot-sourced  (`. .\biopb-engine.ps1`)  -> only defines functions; the
         console front-end then calls Invoke-BiopbInstall with -Mode console for
         pretty colored output, exactly like the original installer.
      2. Run directly (`powershell -File biopb-engine.ps1 -DataDir ... -Webapp`)
         -> runs the install in -Mode gui, emitting machine-parseable records:
              ::biopb::STEP|<n>|<total>|<message>
              ::biopb::OK|<message>
              ::biopb::INFO|<message>
              ::biopb::DETAIL|<message>
              ::biopb::WARN|<message>
              ::biopb::NOTE|<message>
              ::biopb::CMD|<message>
              ::biopb::ERROR|<message>
              ::biopb::DONE|<exitcode>
         Any line WITHOUT the ::biopb:: prefix is raw sub-command log output (uv,
         the data server, ...) and the GUI shows it as log detail. The STEP record
         drives the progress bar (n/total); the trailing DONE record carries the
         final exit code.

    Idempotent: rerun to upgrade. Paths follow the home-relative XDG layout the
    Python packages read; on Windows these resolve under %USERPROFILE%, matching
    Python's Path.home().
#>

[CmdletBinding()]
param(
    # Target microscopy data directory. When empty AND -KeepConfig is not set, the
    # engine keeps an existing biopb.toml if present, else falls back to a
    # dedicated data subfolder (never the profile root).
    [string]$DataDir = "",

    # Install the browser data viewer (webapp.tar.gz).
    [switch]$Webapp,

    # Add the Bio-Formats extra (pulls a Java toolchain on first use).
    [switch]$Bioformats,

    # Track the latest release CANDIDATE (a/b/rc prerelease) instead of stable.
    # Overrides the BIOPB_INSTALL_RC env var when supplied.
    [switch]$Rc,

    # Skip starting the data server at the end (BIOPB_NO_SERVER_START=1 equivalent).
    [switch]$NoServerStart,

    # Explicitly keep an existing biopb.toml untouched (do not rewrite it).
    [switch]$KeepConfig,

    # Walk all steps and emit the full progress stream WITHOUT doing any real
    # work (no downloads, no install, no config/server changes). For testing the
    # console output and the GUI wizard safely. Honors -Webapp for the result.
    [switch]$DryRun,

    # Do NOT seed the off-site cellpose algorithm server into a fresh biopb-mcp
    # config (those servers log client IPs). Absent = enabled (the default Yes).
    [switch]$NoRemotePlugins,

    # Uninstall mode: remove the biopb stack instead of installing it. With
    # -Purge, also delete config and cached data (never the user's images).
    [switch]$Uninstall,
    [switch]$Purge,

    # In gui mode, ALSO append each tagged record to this file (no BOM, UTF-8).
    # This is the transport the Inno wizard polls -- a parent process cannot
    # stream a child's stdout, but it can tail a file. A full diagnostic
    # transcript is written alongside as "<LogFile>.full.log".
    [string]$LogFile = "",

    # Output mode. 'console' = colored Write-Host (front-end uses this in-process);
    # 'gui' = tagged ::biopb:: records on stdout (and -LogFile) for a parent to parse.
    [ValidateSet('console', 'gui')]
    [string]$Mode = 'gui'
)

# Stop on any error; mirror `set -euo pipefail`.
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'  # speeds up Invoke-WebRequest

# ============================================================================
# Reporter -- the integration seam between the engine and either front-end.
# Both modes go through the SAME calls; only the rendering differs, so the
# console and GUI can never drift on WHAT gets reported.
# ============================================================================
$script:ReportMode = $Mode
$script:TotalSteps = 7
$script:McpNeedsManual = $false
$script:LogFilePath = $null   # structured -LogFile path (gui mode); appended per record

# Emit one tagged record in gui mode: to stdout (for parents that can read it)
# AND, when a -LogFile is active, APPENDED to that file. We append-and-close per
# record (no persistent handle) rather than holding a StreamWriter open, because
# the Inno wizard polls the file with a deny-write share that a held-open write
# handle blocks -- which froze the progress bar until the install finished. With
# per-record appends the file is unlocked between writes, so reads succeed live.
function Emit-Gui {
    param([string]$Record)
    Write-Output $Record
    if ($script:LogFilePath) {
        try { [System.IO.File]::AppendAllText($script:LogFilePath, $Record + "`r`n", (New-Object System.Text.UTF8Encoding($false))) } catch { }
    }
}

function Report-Step {
    param([int]$N, [string]$Msg)
    if ($script:ReportMode -eq 'gui') {
        Emit-Gui "::biopb::STEP|$N|$($script:TotalSteps)|$Msg"
    } else {
        Write-Host ""
        Write-Host "[$N/$($script:TotalSteps)] $Msg" -ForegroundColor White
    }
}
function Report-Ok {
    param([string]$Msg)
    if ($script:ReportMode -eq 'gui') { Emit-Gui "::biopb::OK|$Msg" }
    else { Write-Host "  $Msg" -ForegroundColor Green }
}
function Report-Info {
    param([string]$Msg)
    if ($script:ReportMode -eq 'gui') { Emit-Gui "::biopb::INFO|$Msg" }
    else { Write-Host "  $Msg" }
}
function Report-Detail {
    param([string]$Msg)
    if ($script:ReportMode -eq 'gui') { Emit-Gui "::biopb::DETAIL|$Msg" }
    else { Write-Host "    $Msg" -ForegroundColor DarkGray }
}
function Report-Warn {
    param([string]$Msg)
    if ($script:ReportMode -eq 'gui') { Emit-Gui "::biopb::WARN|$Msg" }
    else { Write-Host "  WARNING: " -ForegroundColor Yellow -NoNewline; Write-Host $Msg }
}
function Report-Note {
    param([string]$Msg)
    if ($script:ReportMode -eq 'gui') { Emit-Gui "::biopb::NOTE|$Msg" }
    else { Write-Host "  NOTE: $Msg" -ForegroundColor DarkGray }
}
function Report-Cmd {
    param([string]$Msg)
    if ($script:ReportMode -eq 'gui') { Emit-Gui "::biopb::CMD|$Msg" }
    else { Write-Host "  $Msg" -ForegroundColor Cyan }
}
function Report-Error {
    param([string]$Msg)
    if ($script:ReportMode -eq 'gui') { Emit-Gui "::biopb::ERROR|$Msg" }
    else { Write-Host "ERROR: $Msg" -ForegroundColor Red }
}

# ============================================================================
# Helpers (ported verbatim from install.ps1 -- no interactive bits here).
# ============================================================================

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

# Compute candidate microscopy data directories WITHOUT prompting. Front-ends use
# this to populate their data-directory pickers (console menu, GUI dir page), so
# the candidate logic lives in one place. Returns a string[] of existing dirs.
# Offers dedicated data subfolders only -- never the profile root, Documents, or
# OneDrive -- plus non-system fixed-drive roots; OneDrive "Files On-Demand"
# placeholders hydrate-on-read and hang recursive discovery before the server can
# bind.
function Get-DataDirCandidates {
    param([string]$BiopbHome)
    $candidates = New-Object System.Collections.Generic.List[string]
    $seen = New-Object System.Collections.Generic.HashSet[string]
    foreach ($d in @(
        (Join-Path $BiopbHome 'Data'),
        (Join-Path $BiopbHome 'data'),
        (Join-Path $BiopbHome 'Microscopy')
    )) {
        if ((Test-Path -LiteralPath $d) -and $seen.Add($d.ToLowerInvariant())) {
            $candidates.Add($d) | Out-Null
        }
    }
    foreach ($drv in Get-PSDrive -PSProvider FileSystem -ErrorAction SilentlyContinue) {
        if ($drv.Name -ne 'C' -and $drv.Root -and (Test-Path -LiteralPath $drv.Root)) {
            if ($seen.Add($drv.Root.ToLowerInvariant())) { $candidates.Add($drv.Root) | Out-Null }
        }
    }
    return $candidates.ToArray()
}

# Merge the biopb server into a standard mcpServers JSON config (Claude Desktop,
# Cursor, ...). biopb-mcp speaks stdio, so the client spawns the command itself --
# a bare command+args entry (no "type") is the form every mcpServers client
# accepts. Returns $true only if biopb was actually written into the client config.
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
            Report-Ok "${Label}: registered biopb (merged into $File)"
        } catch {
            Report-Warn "${Label}: could not merge $File - add biopb manually (see $ConfigDir\mcp.json)"
            return $false
        }
        return $true
    }

    $obj = [pscustomobject]@{ mcpServers = [pscustomobject]@{ biopb = $entry } }
    Set-FileUtf8NoBom -Path $File -Content ($obj | ConvertTo-Json -Depth 20)
    Report-Ok "${Label}: created $File"
    return $true
}

# Detect installed agent systems and register the biopb MCP server with each.
# -NoRemotePlugins leaves process_image_servers empty when creating a fresh
# config (the default off-site cellpose server logs client IPs, so enabling it is
# a consent decision the front-end collects). Existing configs are preserved
# regardless, so a prior choice survives a rerun.
function Set-McpClients {
    param([string]$BiopbHome, [string]$ConfigDir, [switch]$NoRemotePlugins)

    # Best-effort agent wiring must never abort the install. Under the script's
    # ErrorActionPreference='Stop', a native CLI that writes to stderr -- e.g.
    # `claude mcp get biopb` when biopb is not registered yet -- raises a
    # TERMINATING NativeCommandError in Windows PowerShell 5.1, even with *>$null.
    # Soften it for this function only (function-scoped) so that probe can't kill
    # the install; we gate on $LASTEXITCODE explicitly below.
    $ErrorActionPreference = 'SilentlyContinue'

    $mcpCmd = (Get-Command biopb-mcp -ErrorAction SilentlyContinue).Source
    if (-not $mcpCmd) { $mcpCmd = "biopb-mcp" }

    # biopb-mcp speaks MCP over stdio: the AI agent spawns it as a child process
    # (`biopb-mcp --transport stdio`). We register the resolved absolute path so
    # GUI agents (e.g. Claude Desktop), which don't inherit the shell PATH, can
    # still find it.
    $mcpArgs = @("--transport", "stdio")

    # Minimal biopb-mcp config (preconfigured biopb.image servicers). Preserved if
    # it already exists so the user's tweaks survive a rerun.
    $mcpConfigDir = Join-Path $BiopbHome ".config\biopb-mcp"
    $mcpConfig = Join-Path $mcpConfigDir "config.json"
    if (-not (Test-Path -LiteralPath $mcpConfigDir)) { New-Item -ItemType Directory -Force -Path $mcpConfigDir | Out-Null }
    if (Test-Path -LiteralPath $mcpConfig) {
        Report-Ok "biopb-mcp config exists at $mcpConfig (preserved)"
    } else {
        # Seed the off-site cellpose server only with consent (-NoRemotePlugins
        # absent). Declining leaves process_image_servers empty; the user can add
        # servers later by editing the config.
        if ($NoRemotePlugins) {
            $processImageServers = ''
            Report-Ok "Remote algorithm plugins disabled (add servers later in $mcpConfig)"
        } else {
            $processImageServers = '        "grpcs://cellpose.biopb.org:443"'
            Report-Ok "Remote algorithm plugins enabled"
        }
        $mcpConfigContent = @"
{
  "mcp": {
    "services": {
      "process_image_servers": [
$processImageServers
      ]
    }
  }
}
"@
        Set-FileUtf8NoBom -Path $mcpConfig -Content $mcpConfigContent
        Report-Ok "Created biopb-mcp config: $mcpConfig"
    }

    # Canonical standalone definition (standard mcpServers JSON; most clients accept it).
    $canonical = [pscustomobject]@{
        mcpServers = [pscustomobject]@{ biopb = [pscustomobject]@{ command = $mcpCmd; args = $mcpArgs } }
    }
    Set-FileUtf8NoBom -Path (Join-Path $ConfigDir "mcp.json") -Content ($canonical | ConvertTo-Json -Depth 20)
    Report-Ok "MCP definition written: $ConfigDir\mcp.json"

    # Assume no working wiring until a branch below writes biopb into a client's
    # config; cleared ($false) only on a real registration.
    $needToShowMcpConfig = $true

    # --- Claude Code (managed through the `claude` CLI) ---
    if (Get-Command claude -ErrorAction SilentlyContinue) {
        # Register idempotently with config-only commands. Do NOT probe with
        # `claude mcp get`: it runs a live CONNECTION test, which makes the claude
        # CLI spawn `biopb-mcp`. claude is a Node app and launches that console
        # child WITHOUT hiding its window, so under the GUI installer (engine runs
        # hidden) a stray console window pops up and lingers after the install.
        # `remove` (a harmless no-op when absent) followed by `add` only edits
        # claude's config -- neither connects -- so it is idempotent and windowless.
        & claude mcp remove biopb -s user *> $null
        & claude mcp add --scope user biopb -- $mcpCmd @mcpArgs *> $null
        if ($LASTEXITCODE -eq 0) {
            Report-Ok "Claude Code: registered biopb (user scope)"
            $needToShowMcpConfig = $false
        } else {
            Report-Warn "Claude Code detected but registration failed - add it manually:"
            Report-Cmd "claude mcp add --scope user biopb -- $mcpCmd $($mcpArgs -join ' ')"
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
                Report-Ok "opencode: registered biopb (merged into $opencodeCfg)"
                $needToShowMcpConfig = $false
            } catch {
                Report-Warn "opencode: could not merge $opencodeCfg - add biopb manually"
            }
        } else {
            $opencodeObj = [pscustomobject]@{ mcp = [pscustomobject]@{ biopb = $opencodeEntry } }
            Set-FileUtf8NoBom -Path $opencodeCfg -Content ($opencodeObj | ConvertTo-Json -Depth 20)
            Report-Ok "opencode: created $opencodeCfg"
            $needToShowMcpConfig = $false
        }
    }

    # Defer the "register manually" notice to the front-end's summary.
    $script:McpNeedsManual = $needToShowMcpConfig
}

# Deregister the biopb MCP server from every client Set-McpClients touches.
# Best-effort and idempotent: a client that was never registered is a no-op.
function Remove-McpClients {
    param([string]$BiopbHome)

    # Best-effort (see Set-McpClients): soften EAP for this function so a native
    # CLI writing to stderr (e.g. removing a server that isn't registered) can't
    # raise a terminating error under the script's ErrorActionPreference='Stop'.
    $ErrorActionPreference = 'SilentlyContinue'

    # Claude Code (via the CLI).
    if (Get-Command claude -ErrorAction SilentlyContinue) {
        & claude mcp remove biopb *> $null
        if ($LASTEXITCODE -eq 0) { Report-Ok "Claude Code: removed biopb" }
    }

    # JSON-config clients: delete the biopb entry under its container property.
    $targets = @(
        @{ File = (Join-Path $env:APPDATA "Claude\claude_desktop_config.json"); Prop = 'mcpServers'; Label = 'Claude Desktop' },
        @{ File = (Join-Path $BiopbHome ".cursor\mcp.json");                     Prop = 'mcpServers'; Label = 'Cursor' },
        @{ File = (Join-Path $BiopbHome ".config\opencode\opencode.json");       Prop = 'mcp';        Label = 'opencode' }
    )
    foreach ($t in $targets) {
        if (-not (Test-Path -LiteralPath $t.File)) { continue }
        try {
            $json = Get-Content -Raw -LiteralPath $t.File | ConvertFrom-Json
            $container = $json.$($t.Prop)
            if ($container -and ($container.PSObject.Properties.Name -contains 'biopb')) {
                $container.PSObject.Properties.Remove('biopb')
                Set-FileUtf8NoBom -Path $t.File -Content ($json | ConvertTo-Json -Depth 20)
                Report-Ok "$($t.Label): removed biopb"
            }
        } catch {
            Report-Warn "$($t.Label): could not edit $($t.File) - remove biopb manually"
        }
    }
}

# Fetch the latest GitHub release metadata for a given tag prefix. The monorepo
# hosts several release lines, so /releases/latest is NOT component-specific: list
# releases (date-desc) and take the newest whose tag is a CLEAN $TagPrefix+X.Y.Z.
# With -AllowRc the regex also admits a PEP 440 prerelease suffix.
function Get-LatestRelease {
    param([string]$Repo, [string]$TagPrefix = "", [bool]$AllowRc = $false)
    $headers = @{ "User-Agent" = "biopb-installer" }
    $releases = Invoke-RestMethod -Uri "https://api.github.com/repos/$Repo/releases?per_page=100" `
        -Headers $headers
    $rx = if ($AllowRc) {
        "^" + [regex]::Escape($TagPrefix) + "\d+\.\d+\.\d+((a|b|rc)\d+)?$"
    } else {
        "^" + [regex]::Escape($TagPrefix) + "\d+\.\d+\.\d+$"
    }
    $match = $releases | Where-Object { $_.tag_name -match $rx } | Select-Object -First 1
    if (-not $match) { throw "No release matching '$TagPrefix' X.Y.Z in $Repo" }
    return $match
}

# Print the tail of the server log for diagnosing a bad startup.
function Show-LogTail {
    param([string]$LogFile)
    if (-not (Test-Path -LiteralPath $LogFile)) { return }
    Report-Info "recent server log ($LogFile):"
    Get-Content -LiteralPath $LogFile -Tail 15 -ErrorAction SilentlyContinue | ForEach-Object {
        Report-Detail $_
    }
}

# Start (or restart) the background data server, then report its health.
# Best-effort: never aborts the install.
function Start-DataServer {
    param([string]$BiopbHome, [string]$ConfigFile, [bool]$NoStart)

    $logFile = Join-Path $BiopbHome ".local\share\biopb\logs\tensor-server.log"

    if ($NoStart) {
        Report-Info "Skipping server start"
        Report-Detail "start it later with: biopb server start"
        return
    }
    if (-not (Get-Command biopb -ErrorAction SilentlyContinue)) {
        Report-Warn "biopb not found on PATH; skipping server start"
        Report-Detail "start it later with: biopb server start"
        return
    }

    # 'restart' loads the just-installed code if a server is already running,
    # and is a plain start otherwise.
    try { & biopb server restart *> $null } catch { }

    # Ask the daemon for its health, polling until SERVING (or 60s). Merge stderr
    # (live progress) into the stream and surface it as it arrives; the JSON
    # verdict (line starting with '{') on stdout is captured for parsing.
    $result = @{ json = $null }
    try {
        & biopb server status --json --wait 60 2>&1 | ForEach-Object {
            $s = "$_"
            if ($s.TrimStart().StartsWith("{")) { $result.json = $s.Trim() }
            elseif ($s.Trim()) { Report-Info $s.Trim() }
        }
    } catch { }
    $out = $result.json
    if (-not $out) { $out = "" }

    # Tolerate an older biopb that predates --json/--wait.
    if (-not $out) {
        $plain = ""
        try { $plain = (& biopb server status 2>$null | Out-String) } catch { $plain = "" }
        if ($plain -match "Running") {
            Report-Ok "Data server started"
        } else {
            Report-Warn "Data server may not have started"
            Show-LogTail -LogFile $logFile
            Report-Detail "full log: $logFile"
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
        Report-Warn "Data server did not come up cleanly"
        Report-Detail "it may still be scanning a large folder, or failed to start:"
        Show-LogTail -LogFile $logFile
        Report-Detail "full log: $logFile"
        return
    }

    if ((-not $count) -or ($count -eq 0)) {
        Report-Warn "Data server is running but found no data sources"
        Report-Detail "check that your data folder holds supported images (see config):"
        Report-Cmd "$ConfigFile"
        Show-LogTail -LogFile $logFile
        return
    }

    Report-Ok "Data server ready - $count data source(s) found; pre-caching overviews"
}

# ============================================================================
# Invoke-BiopbInstall -- the headless install body (was Install-Biopb). Returns a
# result object the front-end uses to render its summary. Throws on fatal errors
# (the front-end catches); never calls exit/Wait-ForExit.
# ============================================================================
function Invoke-BiopbInstall {
    [CmdletBinding()]
    param(
        [string]$DataDir = "",
        [switch]$Webapp,
        [switch]$Bioformats,
        [switch]$Rc,
        [switch]$NoServerStart,
        [switch]$KeepConfig,
        [switch]$DryRun,
        [switch]$NoRemotePlugins,
        [string]$LogFile = "",
        [ValidateSet('console', 'gui')][string]$Mode = 'gui'
    )

    $script:ReportMode = $Mode
    $script:McpNeedsManual = $false

    # Open the structured log + full transcript (gui mode only). The wizard tails
    # the structured file; the transcript captures everything (incl. uv output)
    # for diagnosing a failure. Both best-effort -- never block the install.
    $transcriptOn = $false
    if ($Mode -eq 'gui' -and $LogFile) {
        # Truncate/create the structured log fresh, then append per record (see
        # Emit-Gui) so the wizard can read it live without a share conflict.
        try {
            [System.IO.File]::WriteAllText($LogFile, '', (New-Object System.Text.UTF8Encoding($false)))
            $script:LogFilePath = $LogFile
        } catch { $script:LogFilePath = $null }
        try { Start-Transcript -Path "$LogFile.full.log" -Force | Out-Null; $transcriptOn = $true } catch { }
    }

    try {

    # ----- Dry run: emit the full progress stream, change nothing. -----
    if ($DryRun) {
        $BiopbHome  = $env:USERPROFILE
        $ConfigDir  = Join-Path $BiopbHome ".config\biopb"
        $configFile = Join-Path $ConfigDir "biopb.toml"
        $InstallWebapp = [bool]$Webapp
        $stepMsgs = @(
            "Checking system...",
            "Ensuring build tools...",
            "Ensuring Python...",
            "Installing biopb packages...",
            "Installing data browser...",
            "Config...",
            "Starting data server...",
            "Configuring MCP client..."
        )
        for ($i = 0; $i -lt $stepMsgs.Count; $i++) {
            Report-Step $i $stepMsgs[$i]
            Start-Sleep -Milliseconds 400   # let the gauge visibly advance
            Report-Ok "(dry run) $($stepMsgs[$i])"
        }
        Report-Note "DRY RUN - no changes were made to your system"
        $result = [pscustomobject]@{
            BiopbHome = $BiopbHome; ConfigFile = $configFile; ConfigDir = $ConfigDir
            WebappInstalled = $InstallWebapp; McpNeedsManual = $false
        }
        if ($Mode -eq 'gui') {
            if ($result.WebappInstalled) { Emit-Gui "::biopb::RESULT|webapp|1" }
            Emit-Gui "::biopb::RESULT|config|$($result.ConfigFile)"
            Emit-Gui "::biopb::DONE|0"
        }
        return $result
    }

    # All three wheels (+ webapp) are pulled from ONE biopb release-v* deployment.
    $BiopbRepoUrl = "https://github.com/biopb/biopb"
    $RepoUrl      = $BiopbRepoUrl
    $ReleaseRepo  = "biopb/biopb"
    $ReleaseTagPrefix = "release-v"
    $BiopbHome   = $env:USERPROFILE          # matches Python Path.home() on Windows
    $WebappDir   = Join-Path $BiopbHome ".local\share\biopb\webapp"
    $ConfigDir   = Join-Path $BiopbHome ".config\biopb"
    $LocalBin    = Join-Path $BiopbHome ".local\bin"

    # Release channel: -Rc (or BIOPB_INSTALL_RC env) admits the latest candidate.
    $AllowRc = [bool]$Rc -or (($env:BIOPB_INSTALL_RC) -and ($env:BIOPB_INSTALL_RC -ne '0'))
    $release = $null

    # ===== 0. System Check =====
    Report-Step 0 "Checking system..."

    # A 32-bit (WOW64) PowerShell -- which a 32-bit installer launches -- reports
    # PROCESSOR_ARCHITECTURE = "x86" even on a 64-bit OS; the real arch is in
    # PROCESSOR_ARCHITEW6432. Prefer the latter so we detect the true machine.
    $arch = $env:PROCESSOR_ARCHITEW6432
    if (-not $arch) { $arch = $env:PROCESSOR_ARCHITECTURE }
    switch ($arch) {
        "AMD64" { }
        "ARM64" {
            throw "Windows on ARM64 is not supported yet: key dependencies (pyarrow, the napari Qt stack) do not ship native Windows ARM64 wheels. Use an x64 (Intel/AMD) Windows machine."
        }
        default { throw "Unsupported architecture: $arch (supported: AMD64 / x64)" }
    }
    Report-Ok "Platform: Windows ($arch)"

    # Only tar is needed (no git, buf, or compiler -- release wheels ship prebuilt).
    foreach ($tool in @("tar")) {
        if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) {
            throw "$tool not found (tar ships with Windows 10 1803+; please update Windows)"
        }
        Report-Ok "${tool}: found"
    }
    Report-Ok "System check passed"

    $InstallWebapp     = [bool]$Webapp
    $InstallBioformats = [bool]$Bioformats

    # ===== 1. Install uv (if needed) =====
    Report-Step 1 "Ensuring build tools..."

    # uv's installer voluntarily aborts unless the effective execution policy is
    # Unrestricted/RemoteSigned/Bypass. Set the Process scope (session-only, no
    # admin, cannot override a GPO-enforced policy) so its self-check passes.
    $allowedPolicy = @('Unrestricted', 'RemoteSigned', 'Bypass')
    if ((Get-ExecutionPolicy).ToString() -notin $allowedPolicy) {
        Report-Note "Temporarily allowing scripts for this session (execution policy -> RemoteSigned, Process scope) so the uv installer can run."
        Set-ExecutionPolicy RemoteSigned -Scope Process -Force -ErrorAction SilentlyContinue
    }

    Add-ToUserPath $LocalBin
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
        Report-Info "Installing uv..."
        Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
        Add-ToUserPath $LocalBin
        if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
            throw "uv installation did not land on PATH - reopen PowerShell and rerun"
        }
        Report-Ok "uv installed"
    } else {
        Report-Ok "uv already installed ($(uv --version))"
    }

    # ===== 2. Python =====
    Report-Step 2 "Ensuring Python..."

    $minMinor = 10
    $pythonOk = $false
    $pythonSpec = ""
    $pyExe = (Get-Command python -ErrorAction SilentlyContinue).Source
    if ($pyExe) {
        $verStr = & $pyExe -c "import sys; print(sys.version_info[0], sys.version_info[1])" 2>$null
        if ($LASTEXITCODE -eq 0 -and $verStr) {
            $parts = $verStr.Trim() -split '\s+'
            $maj = [int]$parts[0]; $min = [int]$parts[1]
            if ($maj -gt 3 -or ($maj -eq 3 -and $min -ge $minMinor)) {
                Report-Ok "Using system Python: $(& $pyExe --version)"
                $pythonOk = $true
                $pythonSpec = $pyExe
            } else {
                Report-Warn "System Python too old ($(& $pyExe --version)), need >= 3.$minMinor"
            }
        }
    }
    if (-not $pythonOk) {
        Report-Info "Installing Python 3.11 via uv..."
        uv python install 3.11
        Assert-LastExit "Python install"
        Report-Ok "Python 3.11 ready"
        $pythonSpec = "3.11"
    }

    # ===== 3. Install biopb packages =====
    Report-Step 3 "Installing biopb packages..."

    # On Windows a running biopb process keeps its executables under the uv tool
    # dir open, so `uv tool install --force` cannot delete that dir to reinstall
    # and aborts with "Access is denied" (os error 5) -> uv exit code 2. Stop any
    # previously installed biopb daemons -- the data server AND the biopb-mcp
    # server (which also owns a detached napari kernel) -- so the upgrade can
    # replace the locked binaries. Best-effort (try/catch swallows the benign
    # "nothing running" stderr that would otherwise raise a terminating
    # NativeCommandError) and a no-op on a clean machine. Done before the downloads
    # so the OS has time to release the handles.
    if (Get-Command biopb -ErrorAction SilentlyContinue) {
        try { & biopb server stop *> $null } catch { }
        try { & biopb mcp stop    *> $null } catch { }
        Report-Detail "stopped any running biopb daemons (data + mcp) so their files can be replaced"
    }

    $tensorExtras = "web,aics,medical,ndtiff,hdf5"
    if ($InstallBioformats) {
        $tensorExtras = "$tensorExtras,bioformats"
        Report-Info "including Bio-Formats (Java fetched on first use, not now)"
    }

    # Resolve the wheel triple from a single release-v* build (a matched set);
    # never let the resolver pull biopb/tensor-server/mcp from PyPI.
    try { $release = Get-LatestRelease -Repo $ReleaseRepo -TagPrefix $ReleaseTagPrefix -AllowRc $AllowRc } catch { $release = $null }
    if (-not $release) {
        if ($AllowRc) {
            throw "Could not fetch the latest biopb release candidate from $ReleaseRepo (check network, or unset the RC channel for stable)."
        } else {
            throw "Could not fetch the latest biopb release-v* deployment from $ReleaseRepo (check network and rerun)."
        }
    }
    $mcpAsset    = $release.assets | Where-Object { $_.name -match '^biopb_mcp-.*\.whl$' } | Select-Object -First 1
    $sdkAsset    = $release.assets | Where-Object { $_.name -match '^biopb-.*\.whl$' } | Select-Object -First 1
    $tensorAsset = $release.assets | Where-Object { $_.name -match '^biopb_tensor_server-.*\.whl$' } | Select-Object -First 1
    if (-not $mcpAsset -or -not $sdkAsset -or -not $tensorAsset) {
        throw "Release $($release.tag_name) is missing one of the biopb wheels."
    }
    Report-Info "Installing from release $($release.tag_name)"
    $wheelsDir = Join-Path $env:TEMP "biopb-wheels"
    if (Test-Path -LiteralPath $wheelsDir) { Remove-Item -LiteralPath $wheelsDir -Recurse -Force }
    New-Item -ItemType Directory -Force -Path $wheelsDir | Out-Null
    $mcpWhl    = Join-Path $wheelsDir $mcpAsset.name
    $sdkWhl    = Join-Path $wheelsDir $sdkAsset.name
    $tensorWhl = Join-Path $wheelsDir $tensorAsset.name
    Invoke-WebRequest -Uri $mcpAsset.browser_download_url -OutFile $mcpWhl
    Invoke-WebRequest -Uri $sdkAsset.browser_download_url -OutFile $sdkWhl
    Invoke-WebRequest -Uri $tensorAsset.browser_download_url -OutFile $tensorWhl
    # Direct file:// references pin each package to this exact wheel.
    $mcpReq    = "biopb-mcp[mcp] @ $(([System.Uri]$mcpWhl).AbsoluteUri)"
    $biopbReq  = "biopb[tensor] @ $(([System.Uri]$sdkWhl).AbsoluteUri)"
    $tensorReq = "biopb-tensor-server[$tensorExtras] @ $(([System.Uri]$tensorWhl).AbsoluteUri)"

    # Install everything into ONE uv tool environment so the components can import
    # and drive each other at runtime. biopb is the primary tool; --with adds the
    # siblings and --with-executables-from links their console scripts onto PATH.
    $installArgs = @(
        "tool", "install", "--upgrade", "--force",
        "--python", $pythonSpec,
        $biopbReq,
        "--with", $tensorReq,
        "--with-executables-from", "biopb-tensor-server"
    )
    Report-Info "including biopb-mcp + napari"
    $installArgs += @(
        "--with", $mcpReq,
        "--with", "napari[all]",
        "--with-executables-from", "biopb-mcp"
    )

    Report-Info "Installing biopb into one shared environment..."
    try {
        # Capture uv's own stdout+stderr so a failure is diagnosable. A native
        # child run UNPIPED writes straight to the console handle, which
        # Start-Transcript does not intercept -- so previously uv's real error
        # (e.g. the os-error-5 lock above) was lost and only the generic "exit
        # code 2" survived. Piping through Write-Host routes every line through the
        # PowerShell host, which the transcript ($LogFile.full.log) DOES capture --
        # and, deliberately, NOT through Emit-Gui, so the verbose uv/pip output
        # stays OUT of the structured ::biopb:: stream the wizard polls. (Teeing
        # the hundreds of napari[all] lines into that stream swamped the GUI's
        # per-line memo updates and froze the progress gauge at the step boundary.)
        # In console mode Write-Host simply prints, matching the original installer.
        # 2>&1 under EAP='Stop' can turn a benign uv stderr line into a terminating
        # NativeCommandError, so soften EAP for the call and gate on the real exit
        # code explicitly via Assert-LastExit.
        $prevEAP = $ErrorActionPreference
        $ErrorActionPreference = 'Continue'
        uv @installArgs 2>&1 | ForEach-Object { Write-Host "$_" }
        $ErrorActionPreference = $prevEAP
        Assert-LastExit "biopb install"
    } finally {
        if ($wheelsDir -and (Test-Path -LiteralPath $wheelsDir)) {
            Remove-Item -LiteralPath $wheelsDir -Recurse -Force -ErrorAction SilentlyContinue
        }
    }

    # Refresh PATH so freshly installed tool shims are visible this session.
    Add-ToUserPath $LocalBin
    $versionOutput = (biopb-tensor-server version 2>$null)
    if (-not $versionOutput) { $versionOutput = "installed" }
    Report-Ok "$versionOutput"

    # ===== 4. Webapp =====
    Report-Step 4 "Installing data browser..."

    if ($InstallWebapp) {
        if (-not (Test-Path -LiteralPath $WebappDir)) { New-Item -ItemType Directory -Force -Path $WebappDir | Out-Null }

        if (-not $release) { try { $release = Get-LatestRelease -Repo $ReleaseRepo -TagPrefix $ReleaseTagPrefix -AllowRc $AllowRc } catch { $release = $null } }
        $latestTag = if ($release) { $release.tag_name } else { "" }

        if ($latestTag -and ($latestTag -notmatch '^[A-Za-z0-9._+/-]+$')) {
            Report-Warn "Unexpected tag format, skipping data browser install"
            $latestTag = ""
        }

        if ($latestTag) {
            $versionFile = Join-Path $WebappDir ".version"
            $installedTag = if (Test-Path -LiteralPath $versionFile) { (Get-Content -Raw -LiteralPath $versionFile).Trim() } else { "" }
            if ($installedTag -eq $latestTag) {
                Report-Ok "Data browser already up to date ($latestTag)"
            } else {
                Report-Info "Downloading $latestTag..."
                Remove-Item -LiteralPath $WebappDir -Recurse -Force -ErrorAction SilentlyContinue
                New-Item -ItemType Directory -Force -Path $WebappDir | Out-Null
                $webAsset = $release.assets | Where-Object { $_.name -eq 'webapp.tar.gz' } | Select-Object -First 1
                $webUrl = if ($webAsset) { $webAsset.browser_download_url } else { "$RepoUrl/releases/download/$latestTag/webapp.tar.gz" }
                $tarball = Join-Path $env:TEMP "biopb-webapp.tar.gz"
                $webOk = $true
                try { Invoke-WebRequest -Uri $webUrl -OutFile $tarball }
                catch { $webOk = $false }
                if ($webOk) {
                    tar -xzf $tarball -C $WebappDir --strip-components=1
                    Remove-Item -LiteralPath $tarball -Force -ErrorAction SilentlyContinue
                    Set-FileUtf8NoBom -Path $versionFile -Content $latestTag
                    Report-Ok "Data browser installed to: $WebappDir"
                } else {
                    Report-Warn "No webapp.tar.gz in release $latestTag; server will run in API-only mode"
                }
            }
        } else {
            Report-Warn "Could not fetch latest release, data browser not installed"
            Report-Detail "Server will run in API-only mode"
        }
    } else {
        Report-Info "Skipped"
    }

    # ===== 5. Config =====
    Report-Step 5 "Config..."

    if (-not (Test-Path -LiteralPath $ConfigDir)) { New-Item -ItemType Directory -Force -Path $ConfigDir | Out-Null }
    $configFile = Join-Path $ConfigDir "biopb.toml"
    $configExists = Test-Path -LiteralPath $configFile

    # Decide keep-vs-write. The interactive prompt now lives in the front-end; the
    # engine just honors the resolved choice:
    #   -KeepConfig                  -> keep an existing config untouched
    #   -DataDir <path>              -> (re)write config pointing at that dir
    #   neither, config exists       -> keep it (safe default)
    #   neither, no config           -> default to a dedicated data subfolder
    $effectiveKeep = $KeepConfig -or ((-not $DataDir) -and $configExists)
    $effectiveDataDir = $DataDir
    if (-not $effectiveKeep -and -not $effectiveDataDir) {
        $effectiveDataDir = Join-Path $BiopbHome 'Microscopy'
    }

    if ($effectiveKeep) {
        Report-Ok "Keeping current config: $configFile"
    } else {
        # Preserve any previous config by renaming it to a timestamped backup.
        if ($configExists) {
            $backup = "$configFile.bak." + (Get-Date -Format "yyyyMMddHHmmss")
            Move-Item -LiteralPath $configFile -Destination $backup -Force
            Report-Info "Backed up previous config to $backup"
        }

        # Escape for a TOML basic string: backslashes first, then quotes.
        $tomlDataDir = $effectiveDataDir -replace '\\', '\\' -replace '"', '\"'
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
        Report-Ok "Created: $configFile (data dir: $effectiveDataDir)"
    }

    # ===== 6. Start the data server =====
    Report-Step 6 "Starting data server..."
    Start-DataServer -BiopbHome $BiopbHome -ConfigFile $configFile -NoStart ([bool]$NoServerStart)

    # ===== 7. Wire biopb-mcp into the user's agent system =====
    Report-Step 7 "Configuring MCP client..."
    Set-McpClients -BiopbHome $BiopbHome -ConfigDir $ConfigDir -NoRemotePlugins:$NoRemotePlugins

    # Result object: the front-end renders the human-facing summary from this, so
    # the summary wording is a front-end concern, not the engine's.
    $result = [pscustomobject]@{
        BiopbHome      = $BiopbHome
        ConfigFile     = $configFile
        ConfigDir      = $ConfigDir
        WebappInstalled = $InstallWebapp
        McpNeedsManual = $script:McpNeedsManual
    }

    # The gui protocol (RESULT + the terminal DONE record) is emitted HERE, while
    # the log writer is still open, so the wizard always finds DONE in the file.
    # On failure the catch below emits DONE|1 before the finally closes the file.
    if ($Mode -eq 'gui') {
        if ($result.WebappInstalled) { Emit-Gui "::biopb::RESULT|webapp|1" }
        if ($result.McpNeedsManual)  { Emit-Gui "::biopb::RESULT|mcp_manual|1" }
        Emit-Gui "::biopb::RESULT|config|$($result.ConfigFile)"
        Emit-Gui "::biopb::DONE|0"
    }
    return $result

    } catch {
        # Report the failure through the same reporter (console: red text;
        # gui: an ERROR record), then mark the stream done with a nonzero code.
        Report-Error $_.Exception.Message
        if ($Mode -eq 'gui') { Emit-Gui "::biopb::DONE|1" }
        throw
    } finally {
        # Nothing to close -- records are appended per line (no held handle), and
        # DONE was emitted above so the file is already complete for the wizard.
        $script:LogFilePath = $null
        if ($transcriptOn) { try { Stop-Transcript | Out-Null } catch { } }
    }
}

# ============================================================================
# Invoke-BiopbUninstall -- tear the stack down through the same reporter. The
# real install lives under %USERPROFILE% (.local\bin shims, the uv tool env,
# .config, .local\share), NOT under the GUI's {app} dir, so removing {app} alone
# would orphan everything -- this is what the GUI uninstaller drives. -Purge also
# deletes config + cached data; without it those are kept. Never touches images.
# ============================================================================
function Invoke-BiopbUninstall {
    param(
        [switch]$Purge,
        [string]$LogFile = "",
        [ValidateSet('console', 'gui')][string]$Mode = 'console'
    )
    $script:ReportMode = $Mode
    $script:TotalSteps = 4
    $BiopbHome = $env:USERPROFILE
    # config + cached data (NOT the user's microscopy images).
    $dataDirs = @(
        (Join-Path $BiopbHome ".config\biopb"),
        (Join-Path $BiopbHome ".config\biopb-mcp"),
        (Join-Path $BiopbHome ".local\share\biopb")
    )

    if ($Mode -eq 'gui' -and $LogFile) {
        try {
            [System.IO.File]::WriteAllText($LogFile, '', (New-Object System.Text.UTF8Encoding($false)))
            $script:LogFilePath = $LogFile
        } catch { $script:LogFilePath = $null }
    }
    try {
        Report-Step 1 "Stopping data server..."
        # Stop the data server AND the biopb-mcp server (+ its napari kernel):
        # both lock executables under the uv tool dir, so a still-running daemon
        # would make `uv tool uninstall` fail to remove the dir on Windows.
        if (Get-Command biopb -ErrorAction SilentlyContinue) {
            try { & biopb server stop *> $null } catch { }
            try { & biopb mcp stop    *> $null } catch { }
        }
        Report-Ok "Data server and MCP server stopped (if they were running)"

        Report-Step 2 "Removing biopb packages..."
        if (Get-Command uv -ErrorAction SilentlyContinue) {
            try { uv tool uninstall biopb *> $null } catch { }
            Report-Ok "Removed the biopb uv tool environment (and its console shims)"
        } else {
            Report-Warn "uv not found; skipped package removal"
        }

        Report-Step 3 "Deregistering MCP clients..."
        Remove-McpClients -BiopbHome $BiopbHome

        Report-Step 4 "Cleaning up..."
        if ($Purge) {
            foreach ($d in $dataDirs) {
                if (Test-Path -LiteralPath $d) {
                    Remove-Item -LiteralPath $d -Recurse -Force -ErrorAction SilentlyContinue
                    Report-Ok "Removed $d"
                }
            }
            Report-Note "biopb configuration and cached data removed (your images were not touched)"
        } else {
            Report-Info "Kept your configuration and cached data:"
            foreach ($d in $dataDirs) { if (Test-Path -LiteralPath $d) { Report-Detail $d } }
        }
        # The .local\bin PATH entry is shared with uv and other uv tools, so it is
        # deliberately left on PATH rather than risk breaking them.
        Report-Note "The .local\bin PATH entry is shared with uv and was left untouched"

        if ($Mode -eq 'gui') { Emit-Gui "::biopb::DONE|0" }
    } catch {
        Report-Error $_.Exception.Message
        if ($Mode -eq 'gui') { Emit-Gui "::biopb::DONE|1" }
        throw
    } finally {
        # Nothing to close -- records are appended per line (no held handle).
        $script:LogFilePath = $null
    }
}

# ============================================================================
# Auto-run: only when invoked directly as a script (GUI / standalone). When
# dot-sourced ($MyInvocation.InvocationName -eq '.') we define functions only,
# so the console front-end can reuse them and render in-process.
# ============================================================================
if ($MyInvocation.InvocationName -ne '.') {
    try {
        if ($Uninstall) {
            Invoke-BiopbUninstall -Purge:$Purge -LogFile $LogFile -Mode $Mode | Out-Null
            exit 0
        }
        # Out-Null: the returned object must not leak onto stdout and pollute the
        # tagged stream. Invoke-BiopbInstall emits RESULT/DONE itself.
        Invoke-BiopbInstall `
            -DataDir $DataDir `
            -Webapp:$Webapp `
            -Bioformats:$Bioformats `
            -Rc:$Rc `
            -NoServerStart:$NoServerStart `
            -KeepConfig:$KeepConfig `
            -DryRun:$DryRun `
            -NoRemotePlugins:$NoRemotePlugins `
            -LogFile $LogFile `
            -Mode $Mode | Out-Null
        exit 0
    } catch {
        # Invoke-BiopbInstall already reported the error and (gui) emitted DONE|1.
        exit 1
    }
}
