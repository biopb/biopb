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
    # engine keeps an existing config (biopb.json or legacy biopb.toml) if present,
    # else falls back to a dedicated data subfolder (never the profile root).
    [string]$DataDir = "",

    # Install the web interface (webapp.tar.gz) -- the image viewer plus the
    # server admin page (config / status / restart). Default ON when neither this
    # switch nor $env:BIOPB_INSTALL_WEBAPP is given; set the env var to "0" to skip.
    [switch]$Webapp,

    # Add the Bio-Formats extra (pulls a Java toolchain on first use). Default OFF;
    # an explicit switch wins, else honors $env:BIOPB_INSTALL_BIOFORMATS = "1".
    [switch]$Bioformats,

    # Track the latest release CANDIDATE (a/b/rc prerelease) instead of stable.
    # Overrides the BIOPB_INSTALL_RC env var when supplied.
    [switch]$Rc,

    # Skip starting the data server at the end (BIOPB_NO_SERVER_START=1 equivalent).
    [switch]$NoServerStart,

    # Explicitly keep an existing config untouched (do not rewrite it).
    [switch]$KeepConfig,

    # Mark the written source as cloud/synced storage (OneDrive, Dropbox, iCloud):
    # the server admits dehydrated "Files On-Demand" placeholders as *unresolved*
    # sources (resolved lazily on first read) instead of skipping them. Absent =
    # auto-detected from the data dir (any path under a known cloud root; see
    # Test-IsCloudPath), so this switch is only a manual override for cloud roots
    # the env-var/registry probes miss.
    [switch]$Cloud,

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

# Install the uv tool environment under %LOCALAPPDATA%, not uv's Roaming default
# (%APPDATA%\uv\tools). The biopb tool env holds native binaries and a long-lived
# napari/Qt process -- machine-specific state that has no business in a roaming
# profile (which on managed machines is synced/redirected, inviting locks and slow
# logons). This matches the rest of the installer, which is deliberately Local (the
# Inno GUI installs under {localappdata}\biopb; see docs/windows-installer.md). Set
# at script scope so it is inherited by every `uv` invocation in install AND
# uninstall. An explicit user UV_TOOL_DIR wins. Runs on dot-source too, so the
# console front-end (which dot-sources this engine) gets it before it drives uv.
if ((-not $env:UV_TOOL_DIR) -and $env:LOCALAPPDATA) {
    $env:UV_TOOL_DIR = Join-Path $env:LOCALAPPDATA "uv\tools"
}

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

# Write the tensor-server config as JSON (biopb.json) -- the canonical format
# (biopb/biopb#34). JSON generation is stdlib here (ConvertTo-Json), so the old
# hand-rolled TOML escaping (`-replace '\\','\\' -replace '"','\"'`) is gone.
#
# When -Prior points at an existing biopb.json its settings (server/cache/...)
# are loaded and *preserved*; only the `sources` list is replaced with the chosen
# data dir, so re-running with a new folder no longer discards tuning. PowerShell
# has no TOML parser, so migrating from a legacy biopb.toml starts from the
# installer defaults instead (the caller retires the .toml). `metadata_db.enabled`
# is intentionally omitted -- the DB is on by default and the flag is deprecated
# (biopb/biopb#225).
function Write-ServerConfig {
    param(
        [string]$Path,         # biopb.json to write
        [string]$DataDir,
        [bool]$Cloud,
        [string]$Prior = ""    # existing config to preserve (.json) or migrate (.toml)
    )

    $data = $null
    if ($Prior -and $Prior.EndsWith(".json") -and (Test-Path -LiteralPath $Prior)) {
        try { $data = Get-Content -Raw -LiteralPath $Prior | ConvertFrom-Json } catch { $data = $null }
    }
    if ($null -eq $data) {
        $data = [pscustomobject]@{
            server = [pscustomobject]@{
                host                   = "127.0.0.1"
                port                   = 8815
                aggressive_dir_pruning = $true
            }
            cache = [pscustomobject]@{
                backend             = "file"
                file_max_segment_mb = 256
                file_max_total_gb   = 128
            }
        }
    }

    # One watched folder, replacing any prior sources. A cloud/synced root admits
    # Files-On-Demand placeholders as unresolved sources (cloud = true).
    $src = [ordered]@{ url = $DataDir; monitor = $true }
    if ($Cloud) { $src["cloud"] = $true }
    $sources = @([pscustomobject]$src)
    if ($data.PSObject.Properties.Name -contains 'sources') {
        $data.sources = $sources
    } else {
        $data | Add-Member -NotePropertyName sources -NotePropertyValue $sources -Force
    }

    # Strip the noisy `metadata_db.enabled = true` (the default) if a prior
    # config carried it, mirroring the fresh-template skip -- the DB is on by
    # default and the flag warns on every startup. `enabled = false` is a
    # deliberate user choice (read-only mount, disk constraints, etc.) -- preserve
    # it; the deprecation warning on startup is the intended signal, and Phase 4
    # is the single hard cutover.
    if ($data.PSObject.Properties.Name -contains 'metadata_db') {
        $md = $data.metadata_db
        if ($null -ne $md -and $md.PSObject.Properties.Name -contains 'enabled') {
            if ($md.enabled -ne $false) {
                $remaining = @($md.PSObject.Properties.Name)
                if ($remaining.Count -le 1) {
                    $data.PSObject.Properties.Remove('metadata_db')
                } else {
                    $md.PSObject.Properties.Remove('enabled')
                }
            }
        }
    }

    Set-FileUtf8NoBom -Path $Path -Content ($data | ConvertTo-Json -Depth 20)
}

# Abort if the most recent native command failed. PowerShell does not honor
# $ErrorActionPreference='Stop' for external executables, so mirror `set -e`
# explicitly around the critical install steps.
function Assert-LastExit {
    param([string]$What)
    if ($LASTEXITCODE -ne 0) { throw "$What failed (exit code $LASTEXITCODE)" }
}

# Force-terminate any process running from a biopb uv tool environment so its
# locked binaries under <tooldir>\biopb\Scripts can be deleted. The graceful
# `biopb server stop` / `biopb mcp stop` only reach daemons THIS install's pidfiles
# track; a data server launched ad-hoc from a shell, a detached napari kernel, or
# an agent-spawned stdio biopb-mcp keep the *_pb2/python.exe binaries open and make
# `uv tool install --force` (and `uv tool uninstall`) fail with "Access is denied
# (os error 5)". We match by executable path, covering the current UV_TOOL_DIR AND
# the legacy %APPDATA% (Roaming) default, so an upgrade from an older Roaming
# install -- whose orphaned server may still be running -- also unlocks cleanly.
# Returns the number of processes stopped.
function Stop-BiopbToolProcesses {
    $dirs = New-Object System.Collections.Generic.List[string]
    if ($env:UV_TOOL_DIR)  { $dirs.Add((Join-Path $env:UV_TOOL_DIR 'biopb'))           | Out-Null }
    if ($env:LOCALAPPDATA) { $dirs.Add((Join-Path $env:LOCALAPPDATA 'uv\tools\biopb')) | Out-Null }
    if ($env:APPDATA)      { $dirs.Add((Join-Path $env:APPDATA 'uv\tools\biopb'))      | Out-Null }
    $prefixes = @($dirs | ForEach-Object { $_.TrimEnd('\').ToLowerInvariant() + '\' } | Select-Object -Unique)

    $killed = 0
    foreach ($proc in Get-Process -ErrorAction SilentlyContinue) {
        $path = $null
        try { $path = $proc.Path } catch { $path = $null }   # access-denied on some PIDs
        if (-not $path) { continue }
        $lp = $path.ToLowerInvariant()
        foreach ($pre in $prefixes) {
            if ($lp.StartsWith($pre)) {
                try { Stop-Process -Id $proc.Id -Force -ErrorAction Stop; $killed++ } catch { }
                break
            }
        }
    }
    if ($killed -gt 0) {
        Report-Detail "force-stopped $killed leftover biopb process(es) holding the tool dir open"
        # Let Windows release the file handles before uv deletes the directory.
        Start-Sleep -Milliseconds 800
    }
    return $killed
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

# Known cloud / synced-folder roots on this machine (OneDrive, iCloud, Dropbox).
# Returns existing absolute dir paths (no trailing slash), de-duplicated. These
# are folders whose contents are "Files On-Demand" placeholders that hydrate on
# read; biopb handles them with a source-level `cloud = true` flag (admit
# placeholders as *unresolved* sources, resolve lazily) rather than skipping them
# -- see the tensor server's cloud-storage phase 2. Used both to offer cloud
# folders in the pickers and to auto-mark a chosen cloud dir (Test-IsCloudPath).
function Get-CloudRoots {
    # OneDrive exports these when signed in (personal and/or business); iCloud's
    # Windows app mounts under the profile; Dropbox records its folder(s) in a
    # JSON sidecar. Probe all, then keep the ones that actually exist.
    $raw = @($env:OneDrive, $env:OneDriveConsumer, $env:OneDriveCommercial)
    # Env vars only cover ONE business account (the most recently active one);
    # the registry lists every signed-in account (personal + each business) via
    # its UserFolder value, so a user with two business OneDrives sees both.
    try {
        $acctKey = 'HKCU:\Software\Microsoft\OneDrive\Accounts'
        if (Test-Path -LiteralPath $acctKey) {
            foreach ($sub in Get-ChildItem -LiteralPath $acctKey -ErrorAction SilentlyContinue) {
                $uf = @((Get-ItemProperty -LiteralPath $sub.PSPath -ErrorAction SilentlyContinue).UserFolder)[0]
                if ($uf) { $raw += [string]$uf }
            }
        }
    } catch { }   # missing/inaccessible key: env vars still cover the common case
    if ($env:USERPROFILE) { $raw += (Join-Path $env:USERPROFILE 'iCloudDrive') }
    foreach ($base in @($env:LOCALAPPDATA, $env:APPDATA)) {
        if (-not $base) { continue }
        $info = Join-Path $base 'Dropbox\info.json'
        if (Test-Path -LiteralPath $info) {
            try {
                $j = Get-Content -Raw -LiteralPath $info | ConvertFrom-Json
                foreach ($acct in @('personal', 'business')) {
                    if ($j.$acct -and $j.$acct.path) { $raw += [string]$j.$acct.path }
                }
            } catch { }   # malformed sidecar: just skip Dropbox detection
        }
    }
    $roots = New-Object System.Collections.Generic.List[string]
    $seen = New-Object System.Collections.Generic.HashSet[string]
    foreach ($p in $raw) {
        if ($p -and (Test-Path -LiteralPath $p)) {
            $full = ([System.IO.Path]::GetFullPath($p)).TrimEnd('\', '/')
            if ($seen.Add($full.ToLowerInvariant())) { $roots.Add($full) | Out-Null }
        }
    }
    return $roots.ToArray()
}

# True when $Path is at or under any known cloud/synced root. Drives the config
# writer's auto-detection so that no matter HOW the data dir was chosen (console
# menu, GUI browse, manual entry, BIOPB_DATA_DIR), a cloud folder is written with
# `cloud = true` -- placeholders are then admitted as unresolved sources instead
# of hanging recursive discovery before the server can bind.
function Test-IsCloudPath {
    param([string]$Path)
    if (-not $Path) { return $false }
    try { $full = ([System.IO.Path]::GetFullPath($Path)).TrimEnd('\', '/') } catch { return $false }
    $fullLc = $full.ToLowerInvariant()
    foreach ($root in Get-CloudRoots) {
        $rootLc = $root.ToLowerInvariant()
        # Equal root, or a child under either separator (Windows uses '\'; tolerate
        # '/' so an env var that carries forward slashes still matches).
        if ($fullLc -eq $rootLc -or $fullLc.StartsWith($rootLc + '\') -or $fullLc.StartsWith($rootLc + '/')) {
            return $true
        }
    }
    return $false
}

# Compute candidate microscopy data directories WITHOUT prompting. Front-ends use
# this to populate their data-directory pickers (console menu, GUI dir page), so
# the candidate logic lives in one place. Returns a string[] of existing dirs:
# dedicated data subfolders (never the profile root or Documents), non-system
# fixed-drive roots, then cloud/synced folders. Cloud roots were once excluded
# here because Files-On-Demand placeholders hung discovery; they are now offered
# because the engine marks any dir under a cloud root `cloud = true`
# (Test-IsCloudPath), which admits placeholders as unresolved sources instead. A
# Microscopy subfolder under a cloud root is preferred over the whole synced root.
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
# With -AllowRc the regex also admits a PEP 440 prerelease suffix. PEP 440 lets the
# prerelease marker be glued to the version (1.0rc1) OR dot-separated (1.0.rc1);
# the tag convention here uses the dot form (e.g. release-v0.10.0.rc5), so the
# regex tolerates an optional '.' before a/b/rc (matches both spellings).
function Get-LatestRelease {
    param([string]$Repo, [string]$TagPrefix = "", [bool]$AllowRc = $false)
    $headers = @{ "User-Agent" = "biopb-installer" }
    $releases = Invoke-RestMethod -Uri "https://api.github.com/repos/$Repo/releases?per_page=100" `
        -Headers $headers
    $rx = if ($AllowRc) {
        "^" + [regex]::Escape($TagPrefix) + "\d+\.\d+\.\d+(\.?(a|b|rc)\d+)?$"
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
        # Progressive discovery (biopb/biopb#212) decoupled SERVING from the data
        # folder scan: the server reaches SERVING as soon as it binds and scans in
        # the background, so a big folder no longer holds it out of SERVING. Not
        # SERVING after 60s therefore points to a real startup failure (crash,
        # port already in use, or a wedged bind), not a slow scan.
        Report-Warn "Data server did not reach SERVING within 60s"
        Report-Detail "it likely failed to start or is wedged (a slow folder scan no"
        Report-Detail "longer blocks SERVING, so this is not just still scanning):"
        Show-LogTail -LogFile $logFile
        Report-Detail "full log: $logFile"
        Report-Detail "recheck once with: biopb server status --wait 30"
        return
    }

    if ((-not $count) -or ($count -eq 0)) {
        # SERVING no longer implies a complete catalog: the folder scan runs in
        # the background and registers sources as it walks. status --wait returns
        # at the first SERVING, which is normally *before* the scan has indexed
        # anything -- so 0 sources here usually just means "scan not finished
        # yet," not "empty folder." The count climbs on its own shortly after.
        Report-Info "Data server is up; catalog is still building in the background"
        Report-Detail "no sources indexed yet - normal right after a (re)start"
        Report-Detail "recheck in a moment: biopb server status"
        Report-Detail "if it stays at 0, confirm the data folder holds supported images:"
        Report-Cmd "$ConfigFile"
        return
    }

    Report-Ok "Data server ready - $count data source(s) so far; still scanning + pre-caching in the background"
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
        $configFile = Join-Path $ConfigDir "biopb.json"
        $InstallWebapp = if ($PSBoundParameters.ContainsKey('Webapp')) { [bool]$Webapp } else { $env:BIOPB_INSTALL_WEBAPP -ne '0' }
        $stepMsgs = @(
            "Checking system...",
            "Ensuring build tools...",
            "Ensuring Python...",
            "Installing biopb packages...",
            "Installing web interface...",
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
    $SamplesDir  = Join-Path $BiopbHome ".local\share\biopb\samples"
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

    # Component selection is no longer prompted (biopb/biopb#237). An explicit
    # -Webapp/-Bioformats from a front-end wins; absent that, fall back to env vars
    # with the web interface (now carrying the admin page) defaulting ON and
    # Bio-Formats OFF -- so a direct engine invocation behaves the same way.
    $InstallWebapp     = if ($PSBoundParameters.ContainsKey('Webapp'))     { [bool]$Webapp }     else { $env:BIOPB_INSTALL_WEBAPP -ne '0' }
    $InstallBioformats = if ($PSBoundParameters.ContainsKey('Bioformats')) { [bool]$Bioformats } else { $env:BIOPB_INSTALL_BIOFORMATS -eq '1' }

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

    # biopb-mcp (always installed) requires Python >= 3.10.
    $minMinor = 10

    # Upper bound: the default `aics` extra pulls aicsimageio, which hard-pins
    # `lxml<5`. No lxml 4.x ships a wheel for CPython >= 3.13, so on a 3.13+
    # interpreter uv builds lxml from source -- which fails on a fresh Windows
    # box without libxml2/libxslt headers and an MSVC compiler. Cap at 3.12, the
    # newest Python with a prebuilt lxml 4.x wheel; if the system Python is newer
    # we fall back to a uv-managed 3.12 below. Mirrors install.sh (MAX_MINOR).
    $maxMinor = 12

    $pythonOk = $false
    $pythonSpec = ""
    $pyExe = (Get-Command python -ErrorAction SilentlyContinue).Source
    if ($pyExe) {
        $verStr = & $pyExe -c "import sys; print(sys.version_info[0], sys.version_info[1])" 2>$null
        if ($LASTEXITCODE -eq 0 -and $verStr) {
            $parts = $verStr.Trim() -split '\s+'
            $maj = [int]$parts[0]; $min = [int]$parts[1]
            if ($maj -eq 3 -and $min -ge $minMinor -and $min -le $maxMinor) {
                Report-Ok "Using system Python: $(& $pyExe --version)"
                $pythonOk = $true
                $pythonSpec = $pyExe
            } elseif ($maj -gt 3 -or ($maj -eq 3 -and $min -gt $maxMinor)) {
                Report-Warn "System Python too new ($(& $pyExe --version)); using a managed 3.$maxMinor (aicsimageio's lxml<5 has no wheel for 3.13+)"
            } else {
                Report-Warn "System Python too old ($(& $pyExe --version)), need >= 3.$minMinor"
            }
        }
    }
    if (-not $pythonOk) {
        Report-Info "Installing Python 3.$maxMinor via uv..."
        uv python install "3.$maxMinor"
        Assert-LastExit "Python install"
        Report-Ok "Python 3.$maxMinor ready"
        $pythonSpec = "3.$maxMinor"
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
    # Belt-and-suspenders: the graceful stops above miss servers launched ad-hoc
    # from a shell, detached napari kernels, and agent-spawned stdio biopb-mcp --
    # any of which keeps the tool dir locked and triggers os error 5. Force-stop
    # anything still running out of a biopb tool env (runs even if the biopb shim
    # is absent/broken, since the lock can outlive it).
    Stop-BiopbToolProcesses | Out-Null

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

    # Verify the wheels against the release's SHA256SUMS before installing them
    # (issue #87 trust item). Hard-fail on a mismatch or a wheel missing from a
    # SHA256SUMS that exists; fail open (warn) when the release predates it.
    $sumsAsset = $release.assets | Where-Object { $_.name -eq 'SHA256SUMS' } | Select-Object -First 1
    if ($sumsAsset) {
        $sums = @{}
        # Download to a temp file and read it back rather than reading .Content
        # directly: GitHub serves SHA256SUMS as application/octet-stream, and
        # PowerShell 5.1's Invoke-WebRequest returns .Content as a byte[] for
        # non-text content types -- splitting a byte[] on "`n" yields individual
        # bytes, so the regex matches nothing and every wheel fails with
        # "No checksum for ...". -OutFile + Get-Content -Raw sidesteps the
        # encoding trap and matches the wheel-download pattern above.
        $sumsFile = Join-Path $wheelsDir "SHA256SUMS"
        Invoke-WebRequest -Uri $sumsAsset.browser_download_url -OutFile $sumsFile -UseBasicParsing
        foreach ($line in ((Get-Content -Raw -LiteralPath $sumsFile) -split "`n")) {
            # "<64-hex>  <filename>" (a leading '*' marks binary mode — strip it).
            $m = [regex]::Match($line.Trim(), '^([0-9a-fA-F]{64})\s+\*?(.+)$')
            if ($m.Success) { $sums[$m.Groups[2].Value] = $m.Groups[1].Value.ToLower() }
        }
        foreach ($w in @($mcpWhl, $sdkWhl, $tensorWhl)) {
            $base = Split-Path -Leaf $w
            $expected = $sums[$base]
            if (-not $expected) { throw "No checksum for $base in the release SHA256SUMS" }
            $actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $w).Hash.ToLower()
            if ($actual -ne $expected) { throw "Checksum mismatch for $base - refusing to install (expected $expected, got $actual)" }
        }
        Report-Ok "Wheel checksums verified"
    } else {
        Report-Warn "Release $($release.tag_name) has no SHA256SUMS; skipping wheel integrity check"
    }

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
        # Stream uv's own stdout+stderr through the reporter so the live install
        # detail is BOTH visible (the GUI shows it in its scrolling log; the
        # console prints it) and captured in the diagnostic transcript. A native
        # child run UNPIPED writes straight to the console handle, which
        # Start-Transcript does not intercept -- so the real error (e.g. the
        # os-error-5 lock above) would otherwise be lost and only the generic
        # "exit code 2" survive. The wizard now uses this streaming detail as its
        # primary progress feedback (it no longer drives a determinate gauge, so
        # the volume of uv/pip lines is no longer a problem). 2>&1 under EAP='Stop'
        # can turn a benign uv stderr line into a terminating NativeCommandError,
        # so soften EAP for the call and gate on the real exit code via Assert-LastExit.
        $prevEAP = $ErrorActionPreference
        $ErrorActionPreference = 'Continue'
        uv @installArgs 2>&1 | ForEach-Object { Report-Detail "$_" }
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

    # Record the installed deployment version as the kernel-start auto-updater's
    # baseline (issue #87): the check compares the latest release-v* deployment's
    # versions.json `release` against this marker. Read `release` from the same
    # manifest; fall back to the tag (release-vX.Y.Z -> X.Y.Z). Best-effort — a
    # write failure only re-prompts a future update, never the install.
    $releaseVersion = ""
    $verAsset = $release.assets | Where-Object { $_.name -eq 'versions.json' } | Select-Object -First 1
    if ($verAsset) {
        try { $releaseVersion = ((Invoke-WebRequest -Uri $verAsset.browser_download_url -UseBasicParsing).Content | ConvertFrom-Json).release } catch { $releaseVersion = "" }
    }
    if (-not $releaseVersion) { $releaseVersion = ($release.tag_name -replace "^$([regex]::Escape($ReleaseTagPrefix))", "") }
    try {
        if (-not (Test-Path -LiteralPath $ConfigDir)) { New-Item -ItemType Directory -Force -Path $ConfigDir | Out-Null }
        Set-FileUtf8NoBom -Path (Join-Path $ConfigDir "release.version") -Content $releaseVersion
        Report-Info "recorded installed release $releaseVersion"
    } catch {
        Report-Warn "Could not record installed release version (update checks may re-prompt)"
    }

    # ===== 4. Webapp =====
    Report-Step 4 "Installing web interface..."

    if ($InstallWebapp) {
        if (-not (Test-Path -LiteralPath $WebappDir)) { New-Item -ItemType Directory -Force -Path $WebappDir | Out-Null }

        if (-not $release) { try { $release = Get-LatestRelease -Repo $ReleaseRepo -TagPrefix $ReleaseTagPrefix -AllowRc $AllowRc } catch { $release = $null } }
        $latestTag = if ($release) { $release.tag_name } else { "" }

        if ($latestTag -and ($latestTag -notmatch '^[A-Za-z0-9._+/-]+$')) {
            Report-Warn "Unexpected tag format, skipping web interface install"
            $latestTag = ""
        }

        if ($latestTag) {
            $versionFile = Join-Path $WebappDir ".version"
            $installedTag = if (Test-Path -LiteralPath $versionFile) { (Get-Content -Raw -LiteralPath $versionFile).Trim() } else { "" }
            if ($installedTag -eq $latestTag) {
                Report-Ok "Web interface already up to date ($latestTag)"
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
                    Report-Ok "Web interface installed to: $WebappDir"
                } else {
                    Report-Warn "No webapp.tar.gz in release $latestTag; server will run in API-only mode"
                }
            }
        } else {
            Report-Warn "Could not fetch latest release, web interface not installed"
            Report-Detail "Server will run in API-only mode"
        }
    } else {
        Report-Info "Skipped"
    }

    # ===== 5. Config =====
    Report-Step 5 "Config..."

    if (-not (Test-Path -LiteralPath $ConfigDir)) { New-Item -ItemType Directory -Force -Path $ConfigDir | Out-Null }
    $configFile   = Join-Path $ConfigDir "biopb.json"   # canonical (biopb/biopb#34)
    $legacyConfig = Join-Path $ConfigDir "biopb.toml"   # pre-#34 installs

    # An existing config in either format counts; biopb.json wins when both exist
    # (matches the server's find_config).
    $existingConfig = ""
    if (Test-Path -LiteralPath $configFile)        { $existingConfig = $configFile }
    elseif (Test-Path -LiteralPath $legacyConfig)  { $existingConfig = $legacyConfig }
    $configExists = [bool]$existingConfig

    # Decide keep-vs-write. The interactive prompt now lives in the front-end; the
    # engine just honors the resolved choice:
    #   -KeepConfig                  -> keep an existing config untouched
    #   -DataDir <path>              -> (re)write config pointing at that dir
    #   neither, config exists       -> keep it (safe default)
    #   neither, no config           -> seed the sample bundle and point at it, so
    #                                   a non-CLI user lands on real data with no
    #                                   prompt (they add their own via GUI drag-drop
    #                                   / the admin page). BIOPB_DATA_DIR or a GUI
    #                                   data-dir page still overrides by passing -DataDir.
    $effectiveKeep = $KeepConfig -or ((-not $DataDir) -and $configExists)
    $effectiveDataDir = $DataDir
    $seedSamples = $false
    if (-not $effectiveKeep -and -not $effectiveDataDir) {
        $effectiveDataDir = $SamplesDir
        $seedSamples = $true
    }

    # $activeConfig is the file the running server will read -- the JSON we write,
    # or the untouched existing file when the user keeps it.
    $activeConfig = $existingConfig
    if ($effectiveKeep) {
        Report-Ok "Keeping current config: $existingConfig"
    } else {
        # Cloud/synced root? Auto-detect from the path so any front-end (GUI,
        # console menu, manual entry, BIOPB_DATA_DIR) gets it right; -Cloud forces
        # it on for roots the probes miss. A cloud source admits dehydrated
        # Files-On-Demand placeholders as unresolved sources instead of hanging
        # discovery -- see the tensor server's cloud-storage phase 2.
        # The seeded samples live on the LOCAL profile drive (.local\share, never a
        # OneDrive/Dropbox Known-Folder), so force cloud off for that path -- the
        # files are resident and must not be treated as Files-On-Demand placeholders.
        $isCloud = if ($seedSamples) { $false } else { [bool]$Cloud -or (Test-IsCloudPath -Path $effectiveDataDir) }

        # Fresh install with no chosen dir: populate the sample bundle first so the
        # folder we point at is non-empty. Fails soft -- a missing asset / download
        # error / checksum mismatch just leaves an empty folder (the user can then
        # drag-drop their own data). Mirrors the webapp fetch above; honors
        # BIOPB_INSTALL_SAMPLES=0 to skip.
        if ($seedSamples -and ($env:BIOPB_INSTALL_SAMPLES -ne '0')) {
            if (-not $release) { try { $release = Get-LatestRelease -Repo $ReleaseRepo -TagPrefix $ReleaseTagPrefix -AllowRc $AllowRc } catch { $release = $null } }
            $sTag = if ($release) { $release.tag_name } else { "" }
            if ($sTag) {
                $sVersionFile = Join-Path $SamplesDir ".version"
                $sInstalled = if (Test-Path -LiteralPath $sVersionFile) { (Get-Content -Raw -LiteralPath $sVersionFile).Trim() } else { "" }
                if ($sInstalled -eq $sTag) {
                    Report-Ok "Sample images already up to date ($sTag)"
                } else {
                    Report-Info "Downloading sample images ($sTag)..."
                    $sAsset = $release.assets | Where-Object { $_.name -eq 'biopb-samples.tar.gz' } | Select-Object -First 1
                    $sUrl = if ($sAsset) { $sAsset.browser_download_url } else { "$RepoUrl/releases/download/$sTag/biopb-samples.tar.gz" }
                    $sTarball = Join-Path $env:TEMP "biopb-samples.tar.gz"
                    $sOk = $true
                    try { Invoke-WebRequest -Uri $sUrl -OutFile $sTarball } catch { $sOk = $false }
                    # Soft checksum check: never seed corrupt/tampered data, but never
                    # abort the install over it. $expectedSum stays $null on any lookup
                    # miss (older release, fetch error) -> treated as "not verifiable".
                    if ($sOk) {
                        $expectedSum = $null
                        try {
                            $sumsAsset = $release.assets | Where-Object { $_.name -eq 'SHA256SUMS' } | Select-Object -First 1
                            if ($sumsAsset) {
                                # Download to a temp file and read it back rather than
                                # reading .Content directly: on PowerShell 5.1 (the GUI
                                # installer's shell) GitHub serves SHA256SUMS as
                                # application/octet-stream, so .Content is a byte[] and
                                # splitting it on "`n" yields per-byte tokens -- the
                                # lookup never matches and the check silently no-ops.
                                # -OutFile + Get-Content -Raw matches the wheel path above.
                                $sSumsFile = Join-Path $env:TEMP "biopb-samples-SHA256SUMS"
                                Invoke-WebRequest -Uri $sumsAsset.browser_download_url -OutFile $sSumsFile -UseBasicParsing
                                foreach ($line in ((Get-Content -Raw -LiteralPath $sSumsFile) -split "`n")) {
                                    $parts = $line.Trim() -split '\s+', 2
                                    if ($parts.Count -eq 2 -and ($parts[1] -replace '^\*','') -eq 'biopb-samples.tar.gz') { $expectedSum = $parts[0].ToLower(); break }
                                }
                                Remove-Item -LiteralPath $sSumsFile -Force -ErrorAction SilentlyContinue
                            }
                        } catch { $expectedSum = $null }
                        if ($expectedSum) {
                            $actualSum = (Get-FileHash -LiteralPath $sTarball -Algorithm SHA256).Hash.ToLower()
                            if ($actualSum -ne $expectedSum) {
                                Report-Warn "Sample bundle checksum mismatch; skipping sample images"
                                $sOk = $false
                            }
                        }
                    }
                    if ($sOk) {
                        # Sync the bundle in without a blunt Remove-Item -Recurse on
                        # the user-facing monitored data dir. Extract to a staging
                        # dir, delete only the previous bundle's files (tracked in
                        # .bundle-manifest) so a shrunk bundle leaves no orphans,
                        # then copy the new set in -- drag-dropped user files in the
                        # samples dir survive a re-seed.
                        $sStage = Join-Path $env:TEMP "biopb-samples-stage"
                        Remove-Item -LiteralPath $sStage -Recurse -Force -ErrorAction SilentlyContinue
                        New-Item -ItemType Directory -Force -Path $sStage | Out-Null
                        tar -xzf $sTarball -C $sStage
                        New-Item -ItemType Directory -Force -Path $SamplesDir | Out-Null
                        $sManifest = Join-Path $SamplesDir ".bundle-manifest"
                        if (Test-Path -LiteralPath $sManifest) {
                            foreach ($rel in (Get-Content -LiteralPath $sManifest)) {
                                if ($rel) { Remove-Item -LiteralPath (Join-Path $SamplesDir $rel) -Force -ErrorAction SilentlyContinue }
                            }
                        }
                        $sFiles = Get-ChildItem -LiteralPath $sStage -Recurse -File | ForEach-Object { $_.FullName.Substring($sStage.Length + 1) }
                        Set-Content -LiteralPath $sManifest -Value $sFiles
                        Copy-Item -Path (Join-Path $sStage '*') -Destination $SamplesDir -Recurse -Force
                        Remove-Item -LiteralPath $sStage -Recurse -Force -ErrorAction SilentlyContinue
                        Remove-Item -LiteralPath $sTarball -Force -ErrorAction SilentlyContinue
                        Set-FileUtf8NoBom -Path $sVersionFile -Content $sTag
                        Report-Ok "Sample images installed to: $SamplesDir"
                    } else {
                        Report-Warn "Sample images not installed; starting with an empty data folder"
                    }
                }
            } else {
                Report-Warn "Could not fetch release; sample images not installed"
            }
        }
        # Whenever the config points at the samples dir (fresh install), ensure it
        # exists -- even when seeding was skipped via BIOPB_INSTALL_SAMPLES=0 or
        # failed soft -- so the server never starts pointed at a missing dir. Emit
        # the skip note so an opted-out user isn't told nothing (the front-end's
        # "seeding" line is now conditional too).
        if ($seedSamples) {
            if ($env:BIOPB_INSTALL_SAMPLES -eq '0') {
                Report-Info "Sample images skipped (BIOPB_INSTALL_SAMPLES=0); starting with an empty data folder"
            }
            if (-not (Test-Path -LiteralPath $SamplesDir)) { New-Item -ItemType Directory -Force -Path $SamplesDir | Out-Null }
        }

        # Load existing settings (json) / migrate from defaults (toml) and replace
        # only the sources block -- a new data dir no longer discards tuning (#34).
        Write-ServerConfig -Path $configFile -DataDir $effectiveDataDir -Cloud $isCloud -Prior $existingConfig
        $activeConfig = $configFile

        # Retire a legacy TOML we just superseded so the server does not warn about
        # both files shadowing (find_config prefers biopb.json).
        if ($existingConfig -eq $legacyConfig -and (Test-Path -LiteralPath $legacyConfig)) {
            $backup = "$legacyConfig.bak." + (Get-Date -Format "yyyyMMddHHmmss")
            Move-Item -LiteralPath $legacyConfig -Destination $backup -Force
            Report-Info "Migrated legacy TOML config to JSON (old file backed up)"
        }

        $verb = if ($existingConfig) { "Updated" } else { "Created" }
        if ($isCloud) {
            Report-Ok "${verb}: $configFile (cloud data dir: $effectiveDataDir)"
            Report-Info "Cloud-synced folder: images are indexed without downloading (cloud = true)."
        } else {
            Report-Ok "${verb}: $configFile (data dir: $effectiveDataDir)"
        }
    }

    # ===== 6. Start the data server =====
    Report-Step 6 "Starting data server..."
    Start-DataServer -BiopbHome $BiopbHome -ConfigFile $activeConfig -NoStart ([bool]$NoServerStart)

    # ===== 7. Wire biopb-mcp into the user's agent system =====
    Report-Step 7 "Configuring MCP client..."
    Set-McpClients -BiopbHome $BiopbHome -ConfigDir $ConfigDir -NoRemotePlugins:$NoRemotePlugins

    # Result object: the front-end renders the human-facing summary from this, so
    # the summary wording is a front-end concern, not the engine's.
    $result = [pscustomobject]@{
        BiopbHome      = $BiopbHome
        ConfigFile     = $activeConfig
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
        # Force-stop any leftover process holding the tool dir open, else the
        # `uv tool uninstall` below fails to delete it with os error 5 (same as
        # the install path -- see Stop-BiopbToolProcesses).
        Stop-BiopbToolProcesses | Out-Null
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
        $invokeArgs = @{
            DataDir         = $DataDir
            Rc              = $Rc
            NoServerStart   = $NoServerStart
            KeepConfig      = $KeepConfig
            DryRun          = $DryRun
            NoRemotePlugins = $NoRemotePlugins
            LogFile         = $LogFile
            Mode            = $Mode
        }
        # Forward -Webapp/-Bioformats only when explicitly passed to the script, so
        # an unset switch falls through to Invoke-BiopbInstall's env-var default
        # (web interface ON, Bio-Formats OFF) rather than being pinned to $false.
        if ($PSBoundParameters.ContainsKey('Webapp'))     { $invokeArgs.Webapp     = $Webapp }
        if ($PSBoundParameters.ContainsKey('Bioformats')) { $invokeArgs.Bioformats = $Bioformats }
        Invoke-BiopbInstall @invokeArgs | Out-Null
        exit 0
    } catch {
        # Invoke-BiopbInstall already reported the error and (gui) emitted DONE|1.
        exit 1
    }
}
