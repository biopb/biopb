<#
.SYNOPSIS
    biopb stack install engine (headless, Windows / PowerShell)

.DESCRIPTION
    The non-interactive core extracted from install/install.ps1. It performs the
    actual install orchestration (uv -> Python -> wheel set -> webapp -> config
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

    # DEPRECATED / accepted-but-ignored: the web interface is always installed now
    # (the dashboard IS the SPA). Kept so existing front-ends / direct invocations
    # that still pass -Webapp don't error; it has no effect.
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

    # Reset an existing config to the sample bundle: re-wire the sources block to
    # the curated samples (exactly as a fresh install would) while preserving the
    # user's other settings, instead of keeping the config. No-op when no config
    # exists (a fresh install already seeds samples) and overridden by -DataDir.
    # This is the GUI's "don't keep my configuration" choice -- it re-points the
    # server at the samples without ever prompting for a data folder.
    [switch]$Reset,

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

# Release pin -- stamped at publish, empty in the committed source (the bash
# installer's BIOPB_PINNED_RELEASE twin). release.yaml rewrites this line to the
# exact `release-vX.Y.Z` it ships alongside -- both when publishing the engine to
# biopb.org (the "Pin installers to this release" step) AND when the
# windows-installer job bakes it into the Inno .exe -- so a served/bundled engine
# installs THAT release, not whatever is newest at run time. Empty here so a raw
# fetch of the committed engine still tracks the latest stable release. Override at
# run time with $env:BIOPB_INSTALL_VERSION. Keep the `$script:BiopbPinnedRelease =`
# LHS verbatim -- the stamp anchors on it.
$script:BiopbPinnedRelease = ''

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

# The `biopb` subdir of an XDG base dir, mirroring biopb._locations: config
# in the config tree, portable assets (webapp/samples) in the data tree, logs /
# pid / sentinels in the STATE tree. Honors the XDG env var (as Python does on
# every platform), defaulting to the conventional home-relative dir, so writer
# (installer) and reader (code) resolve the same path.
function Get-BiopbTree {
    param([string]$EnvVar, [string]$DefaultRel)
    $base = [Environment]::GetEnvironmentVariable($EnvVar)
    if (-not $base) { $base = Join-Path $env:USERPROFILE $DefaultRel }
    return Join-Path $base "biopb"
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
        [bool]$Monitor = $true, # watch the source (false for the static sample bundle)
        [string]$Prior = "",   # existing config to preserve (.json) or migrate (.toml)
        [string]$Alias = ""    # catalog tree-root label ("samples" for the sample bundle)
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
                file_max_total_gb   = 32
            }
        }
    }

    # One folder, replacing any prior sources. A user data dir is watched; the
    # static sample bundle is not (monitor = false). A cloud/synced root admits
    # Files-On-Demand placeholders as unresolved sources (cloud = true). An alias
    # (set for the sample bundle) makes the source its own catalog tree root in
    # the browser rather than nesting under the absolute install path.
    $src = [ordered]@{ url = $DataDir; monitor = $Monitor }
    if ($Cloud) { $src["cloud"] = $true }
    if ($Alias) { $src["alias"] = $Alias }
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
# `biopb control stop` only reaches the control-owned data plane; a data server
# launched ad-hoc from a shell, a detached napari kernel, or an agent-spawned
# stdio biopb-mcp keep the *_pb2/python.exe binaries open and make
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
    # it already exists so the user's tweaks survive a rerun. Co-located with the
    # tensor config in ~/.config/biopb (distinct from the client-definition
    # mcp.json below); the schema is flat sections.
    $mcpConfig = Join-Path $ConfigDir "mcp-config.json"
    if (-not (Test-Path -LiteralPath $ConfigDir)) { New-Item -ItemType Directory -Force -Path $ConfigDir | Out-Null }
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
  "services": {
    "process_image_servers": [
$processImageServers
    ]
  }
}
"@
        Set-FileUtf8NoBom -Path $mcpConfig -Content $mcpConfigContent
        Report-Ok "Created biopb-mcp config: $mcpConfig"
    }

    # Seed the built-in example kernel plugin(s) into ~/.config/biopb/kernel/ so
    # they load into the agent kernel namespace at startup and are visible as a
    # "bring your own tool" example (biopb/biopb-mcp#92). Delivered as a file
    # there (not only an installed module) so it is user-visible/editable and
    # loads via the robust startup-file path. Idempotent (never clobbers a
    # user-edited file); best-effort so a failure never aborts the install.
    $seedCmd = (Get-Command biopb-mcp-seed-plugins -ErrorAction SilentlyContinue).Source
    if ($seedCmd) {
        try {
            & $seedCmd | Out-Null
            Report-Ok "Seeded example kernel plugins: $ConfigDir\kernel"
        } catch {
            Report-Info "Skipped seeding example kernel plugins (add later: biopb-mcp-seed-plugins)"
        }
    }

    # Canonical standalone definition (standard mcpServers JSON; most clients accept it).
    $canonical = [pscustomobject]@{
        mcpServers = [pscustomobject]@{ biopb = [pscustomobject]@{ command = $mcpCmd; args = $mcpArgs } }
    }
    Set-FileUtf8NoBom -Path (Join-Path $ConfigDir "mcp.json") -Content ($canonical | ConvertTo-Json -Depth 20)
    Report-Ok "MCP definition written: $ConfigDir\mcp.json"

    # Register with every detected client through the single source of truth:
    # `biopb agents` (core biopb._agents) -- the same catalog + write logic the
    # control-plane dashboard uses. It resolves the absolute biopb-mcp path and
    # writes each client's own config (Claude Code via its CLI -- windowless, so no
    # stray console pops under the hidden GUI engine; the rest via an atomic JSON
    # merge that preserves the user's other servers), so this engine no longer
    # carries a second per-client copy. Output is captured; the summary flag below
    # reads registration state back from the same source of truth.
    & biopb agents register --all *> $null
    Report-Ok "Registered biopb with detected agent clients (biopb agents)"

    # The "register manually" notice fires only if nothing ended up registered.
    # Ask `biopb agents list --json` (clean JSON on stdout) rather than tracking it
    # here -- one source of truth for the verdict too.
    $script:McpNeedsManual = $true
    try {
        $agents = (((& biopb agents list --json 2>$null) | Out-String) | ConvertFrom-Json).agents
        if ($agents | Where-Object { $_.state -eq 'registered' }) {
            $script:McpNeedsManual = $false
        }
    } catch { }
}

# Deregister the biopb MCP server from every client Set-McpClients touches.
# Best-effort and idempotent: a client that was never registered is a no-op.
#
# Unlike registration (which delegates to `biopb agents register`), this stays
# hand-rolled ON PURPOSE: uninstall removes the biopb uv tool -- and its `biopb`
# shim -- in the step BEFORE this one (Invoke-BiopbUninstall), so `biopb agents
# unregister` would no longer exist here. The per-client removal below needs only
# the `claude` CLI and direct file edits, so it works after biopb is gone.
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

# Fetch GitHub release metadata for the deployment line. With -PinTag set (the
# served/bundled-engine default, or $env:BIOPB_INSTALL_VERSION), fetch THAT exact
# release by tag -- one API call, no listing, so the installer/release pairing
# can't skew. Otherwise resolve live: the monorepo hosts several release lines, so
# /releases/latest is NOT component-specific -- list releases (date-desc) and take
# the newest whose tag is a CLEAN $TagPrefix+X.Y.Z. With -AllowRc the regex also
# admits a PEP 440 prerelease suffix. PEP 440 lets the prerelease marker be glued
# to the version (1.0rc1) OR dot-separated (1.0.rc1); the tag convention here uses
# the dot form (e.g. release-v0.10.0.rc5), so the regex tolerates an optional '.'
# before a/b/rc (matches both spellings).
function Get-LatestRelease {
    param([string]$Repo, [string]$TagPrefix = "", [bool]$AllowRc = $false, [string]$PinTag = "")
    $headers = @{ "User-Agent" = "biopb-installer" }
    if ($PinTag) {
        # Validate before the tag goes into the URL -- it comes from the
        # environment / a stamped line, so never trust it into a request raw.
        if ($PinTag -notmatch '^[A-Za-z0-9._+/-]+$') { throw "Unexpected release tag format: $PinTag" }
        return Invoke-RestMethod -Uri "https://api.github.com/repos/$Repo/releases/tags/$PinTag" -Headers $headers
    }
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
function Start-ControlPlane {
    param([string]$BiopbHome, [string]$ConfigFile, [bool]$NoStart)

    $logsDir     = Join-Path (Get-BiopbTree "XDG_STATE_HOME" ".local\state") "logs"
    $controlLog  = Join-Path $logsDir "control.log"
    $serverLog   = Join-Path $logsDir "tensor-server.log"

    if ($NoStart) {
        Report-Info "Skipping control-plane start"
        Report-Detail "start it later with: biopb control start"
        return
    }
    if (-not (Get-Command biopb -ErrorAction SilentlyContinue)) {
        Report-Warn "biopb not found on PATH; skipping control-plane start"
        Report-Detail "start it later with: biopb control start"
        return
    }

    # Native calls below merge stderr via 2>&1; under the script's EAP='Stop',
    # PS 5.1 turns a stderr line into a terminating NativeCommandError, so soften
    # EAP around them and gate on real exit codes.
    $prevEAP = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'

    # Retire a prior control plane (+ the data plane it owns) so the new control
    # plane can bind a fresh plane it owns -- it refuses an in-use gRPC port.
    # Best-effort; a no-op on a clean machine.
    try { & biopb control stop  *> $null } catch { }

    # Start the control plane; it brings up the data plane by default. Don't
    # swallow a failure (biopb/biopb#324): e.g. a gRPC port held by an untracked
    # process makes the control plane refuse, and the CLI prints the real cause.
    $startOut = @()
    try { $startOut = @(& biopb control start 2>&1 | ForEach-Object { "$_" }) } catch { $startOut += "$_" }
    if ($LASTEXITCODE -ne 0) {
        $ErrorActionPreference = $prevEAP
        Report-Warn "Control plane failed to start"
        foreach ($line in $startOut) {
            $t = "$line".Trim()
            if ($t) { Report-Detail $t }
        }
        Show-LogTail -LogFile $controlLog
        Report-Detail "full log: $controlLog"
        Report-Detail "after fixing the cause, run: biopb control start"
        return
    }

    # `control start` returns once its control API is listening but before the data
    # plane finishes booting, so poll `control status` until the plane reports serving.
    # Progressive discovery (biopb/biopb#212) reaches SERVING as soon as the
    # server binds and scans in the background, so not-serving after 60s points to
    # a real startup failure. `control status --json` carries only the plane's state
    # (the lean control plane does no Flight query, so no source_count -- it climbs later).
    $serving = $false; $conflict = $false; $controlUp = $false
    for ($i = 0; $i -lt 60; $i++) {
        $out = ""
        try { $out = (& biopb control status --json 2>$null | Out-String) } catch { $out = "" }
        try {
            $status = $out | ConvertFrom-Json
            $controlUp = [bool]$status.control_api
            $state = $status.data_plane.state
            if ($state -eq "serving")  { $serving = $true;  break }
            if ($state -eq "conflict") { $conflict = $true; break }
        } catch { $controlUp = $false }
        Start-Sleep -Seconds 1
    }
    $ErrorActionPreference = $prevEAP

    if ($serving) {
        Report-Ok "Control plane started - data plane serving; catalog + pre-cache building in the background"
    } elseif ($conflict) {
        Report-Warn "Data-plane gRPC port is held by another process; the control plane will not adopt it"
        Report-Detail "stop that server, then: biopb control start"
    } elseif ($controlUp) {
        # Control plane up but its tensor server never reached serving: the fault
        # is in the data plane, so surface the tensor-server log.
        Report-Warn "Data plane did not reach serving within 60s"
        Report-Detail "the control plane is up but its tensor server failed to start or is wedged:"
        Show-LogTail -LogFile $serverLog
        Report-Detail "full log: $serverLog"
        Report-Detail "recheck with: biopb control status"
    } else {
        # `control start` returned (its API was listening) but the control API is
        # now unreachable: the control process itself crashed -- its log, not the
        # tensor server's, has the cause.
        Report-Warn "Control plane stopped responding within 60s"
        Report-Detail "it started but its control API is now unreachable (the control process likely crashed):"
        Show-LogTail -LogFile $controlLog
        Report-Detail "full log: $controlLog"
        Report-Detail "recheck with: biopb control status"
    }
}

# Precompile the tool env's Python bytecode (.py -> .pyc) so the first MCP
# session and first start_kernel don't pay the one-time compile cost
# (biopb/biopb#384). This is the biggest EASY win on Windows: Defender scans each
# freshly written .pyc and .pyd on first import, so paying the compile once at
# install time (admin-free) removes a large slice of the first-launch wait. It
# covers both trees regardless of which process imports them -- the server stack
# an MCP session child loads, and the heavy napari/Qt tree the child kernel
# loads on first start_kernel (where the user waits longest). Idempotent (a
# rerun after an upgrade compiles only new files) and best-effort: any failure
# just means the first run recompiles as before. (The privileged half of #384 --
# a Windows Defender path exclusion -- is deliberately left to an opt-in CLI
# command, since it needs elevation; precompile needs none, so it runs always.)
function Invoke-Precompile {
    try {
        # Soften EAP for this whole function (function-scoped, auto-restored on
        # return): the engine runs under EAP='Stop', where a native command's
        # `2>&1` turns any benign stderr line into a terminating
        # NativeCommandError (the same trap the uv-install call documents). Under
        # 'Stop' the compileall below would abort on the first stderr line a
        # vendored py2-only module prints -- leaving a PARTIAL cache on exactly
        # the platform (Windows) precompile matters most. 'Continue' lets it run
        # to completion; the try/catch remains the outer safety net.
        $ErrorActionPreference = 'Continue'

        $toolDir = (uv tool dir 2>$null)
        if (-not $toolDir) { return }
        # Windows tool-venv layout puts the interpreter under Scripts\ (bin/ on POSIX).
        $py = Join-Path $toolDir 'biopb\Scripts\python.exe'
        if (-not (Test-Path -LiteralPath $py)) { return }
        # Compile exactly the env's own site-packages (whatever was just installed).
        # The dict key uses '' (escaped single quotes), NOT "purelib": Windows
        # PowerShell 5.1 strips embedded double quotes when it builds a native
        # command line, so "purelib" reaches python.exe as the bare name purelib
        # -> NameError -> empty $site -> this function silently returns and NOTHING
        # is precompiled (biopb/biopb#388). Single quotes survive that quoting.
        $site = (& $py -c 'import sysconfig; print(sysconfig.get_paths()[''purelib''])' 2>$null)
        if (-not $site -or -not (Test-Path -LiteralPath $site)) { return }
        Report-Info "Precompiling Python bytecode (removes the first-launch compile lag)..."
        # -j 0 = all cores. Deliberately NO -q: without it compileall prints a
        # "Compiling '<file>'..." line per file (it does so even under -j 0), which
        # we count to surface throttled progress instead of a silent 30s+ pause --
        # long on Windows because Defender scans every .pyc as it is written (#384).
        # We stream rather than discard to Out-Null, reporting one line per 1000
        # files so the console/GUI log stays bounded. No -f: a fresh install
        # compiles every file anyway, and a rerun (upgrade) then recompiles only
        # the new files -- keeping this idempotent. Non-"Compiling" lines (Listing,
        # or a vendored py2-only module's error on stderr, folded in by 2>&1) are
        # ignored, and compileall's nonzero exit on such a failure stays ignored.
        # (Under -j 0 the per-file lines come from worker processes; on POSIX they
        # reach this pipe, and on Windows the spawned workers inherit stdout so they
        # should too -- worst case the count updates in bursts, never wrongly.)
        $compiled = 0
        & $py -m compileall -j 0 "$site" 2>&1 | ForEach-Object {
            if ($_ -like 'Compiling *') {
                $compiled++
                if ($compiled % 1000 -eq 0) { Report-Detail "  ...$compiled files" }
            }
        }
        Report-Ok "Bytecode precompiled ($compiled files; first viewer launch will be faster)"
    } catch {
        # Non-fatal: the first run just recompiles, exactly as before this ran.
    }
}

# Drop a "biopb Dashboard" shortcut (.lnk) on the user's Desktop that runs
# `biopb dashboard` -- start the control plane if needed, then open the browser.
# Best-effort: a failure only means no icon, never aborts the install. Skip with
# BIOPB_INSTALL_SHORTCUT=0. GUI launchers don't inherit the shell PATH, so the
# shortcut targets biopb.exe by its absolute path.
function Install-DesktopShortcut {
    param([string]$BiopbHome)

    if ($env:BIOPB_INSTALL_SHORTCUT -eq '0') {
        Report-Info "Desktop shortcut skipped (BIOPB_INSTALL_SHORTCUT=0)"
        return
    }

    $cmd = Get-Command biopb -ErrorAction SilentlyContinue
    if (-not $cmd) {
        Report-Info "biopb not on PATH; skipping Desktop shortcut"
        return
    }
    $biopbExe = $cmd.Source

    try {
        $desktop = [Environment]::GetFolderPath('Desktop')
        if (-not $desktop) { $desktop = Join-Path $BiopbHome 'Desktop' }
        if (-not (Test-Path -LiteralPath $desktop)) { New-Item -ItemType Directory -Force -Path $desktop | Out-Null }
        $lnkPath = Join-Path $desktop 'biopb Dashboard.lnk'
        $shell = New-Object -ComObject WScript.Shell
        $sc = $shell.CreateShortcut($lnkPath)
        $sc.TargetPath = $biopbExe
        $sc.Arguments = 'dashboard'
        $sc.Description = 'Start the biopb control plane and open the dashboard'
        $sc.WorkingDirectory = Split-Path $biopbExe -Parent
        # Brand the shortcut with the webapp's icon (shipped in the webapp bundle);
        # only if present, else leave the default (biopb.exe's) icon.
        $icon = Join-Path (Get-BiopbTree "XDG_DATA_HOME" ".local\share") "webapp\favicon.ico"
        if (Test-Path -LiteralPath $icon) { $sc.IconLocation = "$icon,0" }
        $sc.Save()
        Report-Ok "Desktop shortcut created: $lnkPath"
    } catch {
        Report-Warn "Could not create Desktop shortcut: $_"
    }
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
        [switch]$Reset,
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
        $ConfigDir  = Get-BiopbTree "XDG_CONFIG_HOME" ".config"
        $configFile = Join-Path $ConfigDir "biopb.json"
        $stepMsgs = @(
            "Checking system...",
            "Ensuring build tools...",
            "Ensuring Python...",
            "Installing biopb packages...",
            "Installing web interface...",
            "Config...",
            "Starting control plane...",
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
            WebappInstalled = $true; McpNeedsManual = $false
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
    # Data TREE root (webapp/ and samples/ live under it) -- deliberately NOT named
    # $DataDir: that is the -DataDir parameter carrying the user's chosen source
    # folder (BIOPB_DATA_DIR). #484 reused the $DataDir name here, silently
    # clobbering the parameter -- which forced $effectiveDataDir to the data root,
    # so $seedSamples was always false (samples never seeded) and BIOPB_DATA_DIR was
    # ignored. Keep the two distinct.
    $DataRoot    = Get-BiopbTree "XDG_DATA_HOME" ".local\share"
    $WebappDir   = Join-Path $DataRoot "webapp"
    $SamplesDir  = Join-Path $DataRoot "samples"
    $ConfigDir   = Get-BiopbTree "XDG_CONFIG_HOME" ".config"
    $LocalBin    = Join-Path $BiopbHome ".local\bin"

    # Release channel: -Rc (or BIOPB_INSTALL_RC env) admits the latest candidate.
    $AllowRc = [bool]$Rc -or (($env:BIOPB_INSTALL_RC) -and ($env:BIOPB_INSTALL_RC -ne '0'))

    # Effective release to install ($PinTag). Precedence (twin of install.sh):
    #   BIOPB_INSTALL_VERSION  explicit override -- install/downgrade to an exact
    #                          release (accepts release-vX.Y.Z, vX.Y.Z, or X.Y.Z)
    #   > RC channel           query the latest CANDIDATE live (ignores any pin;
    #                          rc builds are not published to biopb.org)
    #   > $script:BiopbPinnedRelease  the tag stamped into this engine at publish
    #   > "" (empty)           raw engine: track the latest stable release
    $PinTag = ""
    if ($env:BIOPB_INSTALL_VERSION) {
        $PinTag = $env:BIOPB_INSTALL_VERSION
        if     ($PinTag.StartsWith($ReleaseTagPrefix)) { }                               # release-v0.11.0
        elseif ($PinTag.StartsWith("v"))               { $PinTag = "$ReleaseTagPrefix$($PinTag.Substring(1))" }  # v0.11.0
        else                                           { $PinTag = "$ReleaseTagPrefix$PinTag" }                  # 0.11.0
    } elseif ($AllowRc) {
        $PinTag = ""
    } elseif ($script:BiopbPinnedRelease) {
        $PinTag = $script:BiopbPinnedRelease
    }
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

    # Component selection is no longer prompted (biopb/biopb#237). The web interface
    # is mandatory (the dashboard is the SPA; -Webapp / BIOPB_INSTALL_WEBAPP are
    # ignored). Bio-Formats stays opt-in: an explicit -Bioformats from a front-end
    # wins, else the env var, default OFF (it pulls a Java toolchain on first use).
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

    # Upper bound: two things cap Python at 3.12. (1) The biopb packages declare
    # requires-python ">=3.10,<3.13", so 3.13+ is refused at resolution. (2) The
    # installer's `czi` extra pulls the CZI reader (pylibczirw / aicspylibczi), which
    # ships no cp313 wheel yet -- on 3.13+ uv would build it from source (cmake +
    # libCZI + an MSVC compiler), which fails on a fresh Windows box. If the
    # system Python is newer we fall back to a uv-managed 3.12 below. Mirrors
    # install.sh (MAX_MINOR).
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
                Report-Warn "System Python too new ($(& $pyExe --version)); using a managed 3.$maxMinor (biopb requires Python <3.13; the CZI reader has no 3.13 wheel yet)"
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
    # previously installed biopb daemons -- the control plane (which owns and
    # holds open a supervised tensor-server child), the standalone data server,
    # AND the biopb-mcp server (which owns a detached napari kernel) -- so the
    # upgrade can replace the locked binaries. The control plane goes FIRST: it owns the
    # data plane, so stopping it tears that child down and stops it respawning
    # anything the next two commands / the force-kill would race. Best-effort
    # (try/catch swallows the benign "nothing running" stderr) and a no-op on a
    # clean machine. Done before the downloads so the OS releases the handles.
    if (Get-Command biopb -ErrorAction SilentlyContinue) {
        try { & biopb control stop  *> $null } catch { }
        Report-Detail "stopped the running biopb control + data plane so their files can be replaced"
    }
    # Belt-and-suspenders: the graceful stops above miss servers launched ad-hoc
    # from a shell, detached napari kernels, and agent-spawned stdio biopb-mcp --
    # any of which keeps the tool dir locked and triggers os error 5. Force-stop
    # anything still running out of a biopb tool env (runs even if the biopb shim
    # is absent/broken, since the lock can outlive it).
    Stop-BiopbToolProcesses | Out-Null

    # Windows has a pylibczirw wheel, so the CZI reader ([czi]) is always included
    # here -- only Intel macOS lacks the wheel (handled in install.sh).
    # HDF5 ([hdf5] -> h5py) is opt-in, not bundled by default (see install.sh).
    $tensorExtras = "web,aics,czi,medical,ndtiff"
    if ($InstallBioformats) {
        $tensorExtras = "$tensorExtras,bioformats"
        Report-Info "including Bio-Formats (Java fetched on first use, not now)"
    }

    # Resolve the wheel set from a single release-v* build (a matched set);
    # never let the resolver pull biopb/tensor-server/mcp from PyPI.
    try { $release = Get-LatestRelease -Repo $ReleaseRepo -TagPrefix $ReleaseTagPrefix -AllowRc $AllowRc -PinTag $PinTag } catch { $release = $null }
    if (-not $release) {
        if ($PinTag) {
            throw "Could not fetch biopb release $PinTag from $ReleaseRepo (check the version exists and your network; omit BIOPB_INSTALL_VERSION to install the latest stable release)."
        } elseif ($AllowRc) {
            throw "Could not fetch the latest biopb release candidate from $ReleaseRepo (check network, or unset the RC channel for stable)."
        } else {
            throw "Could not fetch the latest biopb release-v* deployment from $ReleaseRepo (check network and rerun)."
        }
    }
    $mcpAsset    = $release.assets | Where-Object { $_.name -match '^biopb_mcp-.*\.whl$' } | Select-Object -First 1
    $sdkAsset    = $release.assets | Where-Object { $_.name -match '^biopb-.*\.whl$' } | Select-Object -First 1
    $tensorAsset = $release.assets | Where-Object { $_.name -match '^biopb_tensor_server-.*\.whl$' } | Select-Object -First 1
    # biopb-control (control plane). Its underscore filename (biopb_control-…) is not
    # matched by the sdk pattern '^biopb-.*' above, so the two stay distinct.
    $controlAsset  = $release.assets | Where-Object { $_.name -match '^biopb_control-.*\.whl$' } | Select-Object -First 1
    if (-not $mcpAsset -or -not $sdkAsset -or -not $tensorAsset -or -not $controlAsset) {
        throw "Release $($release.tag_name) is missing one of the biopb wheels."
    }
    Report-Info "Installing from release $($release.tag_name)"
    $wheelsDir = Join-Path $env:TEMP "biopb-wheels"
    if (Test-Path -LiteralPath $wheelsDir) { Remove-Item -LiteralPath $wheelsDir -Recurse -Force }
    New-Item -ItemType Directory -Force -Path $wheelsDir | Out-Null
    $mcpWhl    = Join-Path $wheelsDir $mcpAsset.name
    $sdkWhl    = Join-Path $wheelsDir $sdkAsset.name
    $tensorWhl = Join-Path $wheelsDir $tensorAsset.name
    $controlWhl  = Join-Path $wheelsDir $controlAsset.name
    Invoke-WebRequest -Uri $mcpAsset.browser_download_url -OutFile $mcpWhl
    Invoke-WebRequest -Uri $sdkAsset.browser_download_url -OutFile $sdkWhl
    Invoke-WebRequest -Uri $tensorAsset.browser_download_url -OutFile $tensorWhl
    Invoke-WebRequest -Uri $controlAsset.browser_download_url -OutFile $controlWhl

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
        foreach ($w in @($mcpWhl, $sdkWhl, $tensorWhl, $controlWhl)) {
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
    $controlReq  = "biopb-control @ $(([System.Uri]$controlWhl).AbsoluteUri)"

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
    # biopb-control (control plane): `biopb control …` runs through the core CLI (which
    # spawns `python -m biopb_control`), so it just needs to be importable here;
    # --with-executables-from also links the standalone `biopb-control` script.
    Report-Info "including biopb-control"
    $installArgs += @(
        "--with", $controlReq,
        "--with-executables-from", "biopb-control"
    )

    Report-Info "Installing biopb into one shared environment (first run can take several minutes; on Windows, antivirus scans every file uv writes)..."
    $uvOutLog = Join-Path $env:TEMP "biopb-uv-install.out.log"
    $uvErrLog = Join-Path $env:TEMP "biopb-uv-install.err.log"
    try {
        # uv only animates its progress bar on a TTY, and it goes SILENT for
        # minutes during the prepare/link phase (no per-line output there) -- which
        # reads as a frozen console, worst on Windows where Defender real-time-scans
        # every file uv unpacks/links (see biopb/biopb#384). --verbose does NOT help:
        # it front-loads a burst during the ~seconds-long resolve, then is just as
        # silent through the ~minute of prepare+install.
        #
        # So run uv detached with its output redirected to a log, emit our OWN
        # heartbeat while it works, then replay the log through the reporter so uv's
        # full detail (the "Installed N packages" summary, or the real error on
        # failure) still lands in the console/GUI/diagnostic transcript. Start-Process
        # is given a single pre-quoted argument line, not an -ArgumentList array:
        # the requirement specs are PEP 508 direct refs containing spaces
        # ("biopb[tensor] @ file:///..."), which an array would split under Windows
        # PowerShell 5.1. Only spaces/quotes need quoting here (no embedded quotes).
        $uvExe = (Get-Command uv -ErrorAction Stop).Source
        $argLine = ($installArgs | ForEach-Object {
            if ($_ -match '[\s"]') { '"' + ($_ -replace '"', '\"') + '"' } else { $_ }
        }) -join ' '

        $proc = Start-Process -FilePath $uvExe -ArgumentList $argLine -NoNewWindow -PassThru `
            -RedirectStandardOutput $uvOutLog -RedirectStandardError $uvErrLog
        # Touch .Handle so the Process object retains the native handle; without
        # this, Start-Process -PassThru discards it on exit and $proc.ExitCode
        # comes back $null -- which Assert-LastExit would misread as a failure.
        $null = $proc.Handle
        # Wait in 1s slices so we notice a fast (warm-cache) exit promptly, but
        # only emit a heartbeat every ~10s so a slow install shows steady life
        # without spamming. WaitForExit(ms) returns true the instant uv exits.
        $sw = [System.Diagnostics.Stopwatch]::StartNew()
        $lastTick = 0
        while (-not $proc.WaitForExit(1000)) {
            $elapsed = [int]$sw.Elapsed.TotalSeconds
            if (($elapsed - $lastTick) -ge 10) {
                $lastTick = $elapsed
                Report-Detail "  ...still installing (${elapsed}s elapsed)"
            }
        }
        # The timed WaitForExit above returns the instant uv signals exit, which can
        # be before .NET has cached the process's exit code; the parameterless
        # overload blocks until the process is fully reaped so $proc.ExitCode is
        # reliably populated below (belt-and-suspenders with the .Handle touch above).
        $proc.WaitForExit()

        # Replay captured output (after exit, so the streams are flushed/closed).
        # -Encoding UTF8: uv writes UTF-8, but Windows PowerShell 5.1 Get-Content
        # defaults to ANSI and would mangle any non-ASCII in the replayed summary /
        # error lines (mojibake in the console + diagnostic transcript).
        foreach ($f in @($uvOutLog, $uvErrLog)) {
            if (Test-Path -LiteralPath $f) {
                Get-Content -LiteralPath $f -Encoding UTF8 | ForEach-Object {
                    if ($_ -ne '') { Report-Detail "$_" }
                }
            }
        }
        # Start-Process leaves $LASTEXITCODE untouched, so set it from the process
        # object for the shared Assert-LastExit gate. Guard a null ExitCode -- it
        # would make `$LASTEXITCODE -ne 0` true and false-fail a good install -- as an
        # explicit, legible error rather than a phantom "failed (exit code )".
        if ($null -eq $proc.ExitCode) { throw "biopb install: uv exit code unavailable" }
        $global:LASTEXITCODE = $proc.ExitCode
        Assert-LastExit "biopb install"
    } finally {
        Remove-Item -LiteralPath $uvOutLog, $uvErrLog -Force -ErrorAction SilentlyContinue
        if ($wheelsDir -and (Test-Path -LiteralPath $wheelsDir)) {
            Remove-Item -LiteralPath $wheelsDir -Recurse -Force -ErrorAction SilentlyContinue
        }
    }

    # Refresh PATH so freshly installed tool shims are visible this session.
    Add-ToUserPath $LocalBin
    $versionOutput = (biopb-tensor-server version 2>$null)
    if (-not $versionOutput) { $versionOutput = "installed" }
    Report-Ok "$versionOutput"

    # Warm the bytecode cache now (admin-free) so the first viewer launch is fast.
    Invoke-Precompile

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
    # Mandatory (the dashboard is the SPA). $WebappOk records whether it is actually
    # on disk after this step, so the summary flags a failed fetch -- the only way
    # it can be missing now is a network/asset error, not a user opt-out.
    Report-Step 4 "Installing web interface..."

    $WebappOk = $false
    if (-not (Test-Path -LiteralPath $WebappDir)) { New-Item -ItemType Directory -Force -Path $WebappDir | Out-Null }

    if (-not $release) { try { $release = Get-LatestRelease -Repo $ReleaseRepo -TagPrefix $ReleaseTagPrefix -AllowRc $AllowRc -PinTag $PinTag } catch { $release = $null } }
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
            $WebappOk = $true
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
                $WebappOk = $true
            } else {
                Report-Warn "Could not download the web interface (webapp.tar.gz) from $latestTag"
                Report-Detail "The dashboard needs it -- rerun to retry."
            }
        }
    } else {
        Report-Warn "Could not fetch the latest release; web interface not installed"
        Report-Detail "The dashboard needs it -- rerun to retry."
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
    #   -Reset                       -> re-wire an existing config's sources to the
    #                                   sample bundle (other settings preserved),
    #                                   the same end state as a fresh install; the
    #                                   GUI's "don't keep my config" choice
    #   -DataDir <path>              -> (re)write config pointing at that dir
    #   neither, config exists       -> keep it (safe default)
    #   neither, no config           -> seed the sample bundle and point at it, so
    #                                   a non-CLI user lands on real data with no
    #                                   prompt (they add their own via GUI drag-drop
    #                                   / the admin page). BIOPB_DATA_DIR still
    #                                   overrides by passing -DataDir.
    # -Reset forces the fresh sample-seed path below even when a config exists (an
    # explicit -DataDir still wins); it is a no-op without a config.
    $effectiveKeep = (-not $Reset) -and ($KeepConfig -or ((-not $DataDir) -and $configExists))
    $effectiveDataDir = $DataDir
    $seedSamples = $false
    if (-not $effectiveKeep -and -not $effectiveDataDir) {
        $effectiveDataDir = $SamplesDir
        $seedSamples = $true
    }

    # Always download the sample bundle into $SamplesDir (idempotent -- skips when
    # already at this release; honors BIOPB_INSTALL_SAMPLES=0). Runs on every install
    # regardless of the keep-vs-seed decision below: whether the server is *pointed*
    # at the samples stays a fresh-install choice ($seedSamples), but the bytes are
    # fetched on every run. Fails soft -- a missing asset / download error / checksum
    # mismatch just warns and leaves the folder as-is.
    if ($env:BIOPB_INSTALL_SAMPLES -ne '0') {
        if (-not $release) { try { $release = Get-LatestRelease -Repo $ReleaseRepo -TagPrefix $ReleaseTagPrefix -AllowRc $AllowRc -PinTag $PinTag } catch { $release = $null } }
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

    # $activeConfig is the file the running server will read -- the JSON we write,
    # or the untouched existing file when the user keeps it.
    $activeConfig = $existingConfig
    if ($effectiveKeep) {
        # Keeping the user's existing config. If it is a pre-#34 legacy TOML,
        # convert it in place to the canonical JSON via `biopb server
        # migrate-config` (settings preserved verbatim, old file backed up to
        # biopb.toml.bak) so an upgraded install stops warning about the
        # deprecated format. A JSON config is already canonical -- nothing to do.
        if ($existingConfig -eq $legacyConfig -and (Get-Command biopb -ErrorAction SilentlyContinue)) {
            Report-Info "Migrating legacy TOML config to canonical JSON..."
            $prevEAP2 = $ErrorActionPreference
            $ErrorActionPreference = 'Continue'
            try { & biopb server migrate-config *> $null } catch { }
            $ErrorActionPreference = $prevEAP2
            if ($LASTEXITCODE -eq 0 -and (Test-Path -LiteralPath $configFile)) {
                $activeConfig = $configFile
                Report-Ok "Migrated config: $legacyConfig -> $configFile (old file backed up)"
            } else {
                Report-Warn "Could not migrate legacy config; keeping $existingConfig"
            }
        } else {
            Report-Ok "Keeping current config: $existingConfig"
        }
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
        # The curated sample bundle is static -- don't watch it. A user data dir is
        # monitored so newly added files auto-register.
        $isMonitored = -not $seedSamples

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
        $sourceAlias = if ($seedSamples) { "samples" } else { "" }
        Write-ServerConfig -Path $configFile -DataDir $effectiveDataDir -Cloud $isCloud -Monitor $isMonitored -Prior $existingConfig -Alias $sourceAlias
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

    # ===== 6. Wire biopb-mcp into the user's agent system =====
    # Before starting the control plane: agent wiring is quick and self-contained,
    # so the client is configured even if the (slower) control-plane start is
    # interrupted -- and the control plane comes up on demand anyway when the agent
    # first launches biopb-mcp.
    Report-Step 6 "Configuring MCP client..."
    Set-McpClients -BiopbHome $BiopbHome -ConfigDir $ConfigDir -NoRemotePlugins:$NoRemotePlugins

    # ===== 7. Start the control plane (which owns the data plane) =====
    Report-Step 7 "Starting control plane..."
    Start-ControlPlane -BiopbHome $BiopbHome -ConfigFile $activeConfig -NoStart ([bool]$NoServerStart)

    # Drop a "biopb Dashboard" launcher on the Desktop (runs `biopb dashboard`).
    Install-DesktopShortcut -BiopbHome $BiopbHome

    # Result object: the front-end renders the human-facing summary from this, so
    # the summary wording is a front-end concern, not the engine's.
    $result = [pscustomobject]@{
        BiopbHome      = $BiopbHome
        ConfigFile     = $activeConfig
        ConfigDir      = $ConfigDir
        WebappInstalled = $WebappOk
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
        (Get-BiopbTree "XDG_CONFIG_HOME" ".config"),
        (Join-Path $BiopbHome ".config\biopb-mcp"),
        (Get-BiopbTree "XDG_STATE_HOME" ".local\state"),
        (Get-BiopbTree "XDG_DATA_HOME" ".local\share"),
        (Join-Path $BiopbHome ".local\share\biopb-mcp")
    )

    if ($Mode -eq 'gui' -and $LogFile) {
        try {
            [System.IO.File]::WriteAllText($LogFile, '', (New-Object System.Text.UTF8Encoding($false)))
            $script:LogFilePath = $LogFile
        } catch { $script:LogFilePath = $null }
    }
    try {
        Report-Step 1 "Stopping biopb services..."
        # Stop the control plane, the data server, AND the biopb-mcp server (+ its
        # napari kernel): each locks executables under the uv tool dir, so a
        # still-running daemon would make `uv tool uninstall` fail to remove the
        # dir on Windows. The control plane goes first -- it owns the data plane, so
        # stopping it is a complete teardown of that pair and stops it respawning.
        if (Get-Command biopb -ErrorAction SilentlyContinue) {
            try { & biopb control stop  *> $null } catch { }
        }
        # Force-stop any leftover process holding the tool dir open, else the
        # `uv tool uninstall` below fails to delete it with os error 5 (same as
        # the install path -- see Stop-BiopbToolProcesses).
        Stop-BiopbToolProcesses | Out-Null
        Report-Ok "biopb services stopped (if they were running)"

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
            # The file-backend cache lives in the system temp dir (the tensor
            # server's _default_file_cache_dir), NOT under .local\share, so the
            # $dataDirs sweep misses it. Remove it best-effort: the per-user
            # default location, plus any custom cache.file_cache_dir read from the
            # config before it is deleted. The server was stopped in step 1.
            $cacheDirs = @((Join-Path $env:TEMP "biopb-cache-$env:USERNAME"))
            $cfgFile = Join-Path (Get-BiopbTree "XDG_CONFIG_HOME" ".config") "biopb.json"
            if (Test-Path -LiteralPath $cfgFile) {
                try {
                    $custom = (Get-Content -Raw -LiteralPath $cfgFile | ConvertFrom-Json).cache.file_cache_dir
                    if ($custom) { $cacheDirs += $custom }
                } catch { }
            }
            foreach ($c in $cacheDirs) {
                if (Test-Path -LiteralPath $c) {
                    Remove-Item -LiteralPath $c -Recurse -Force -ErrorAction SilentlyContinue
                    if (Test-Path -LiteralPath $c) { Report-Info "Could not remove cache $c (left in place)" }
                    else { Report-Ok "Removed cache $c" }
                }
            }
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
            Reset           = $Reset
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
