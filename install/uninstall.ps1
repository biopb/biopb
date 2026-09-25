<#
.SYNOPSIS
    biopb stack uninstaller (Windows / PowerShell)
.DESCRIPTION
    Usage: irm https://biopb.org/uninstall.ps1 | iex
    Removes what install.ps1 installed, through the same engine's teardown (the
    GUI install's Add/Remove Programs entry runs the same one). Asks whether to
    also delete config and cached data; your images are never touched.
    Unattended: & ([scriptblock]::Create((irm https://biopb.org/uninstall.ps1))) -Purge
    (or -KeepData) answers that question up front.
#>
param(
    [switch]$Purge,
    [switch]$KeepData
)
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

# A sibling engine (a checkout, the unpacked GUI installer) wins; otherwise the
# biopb.org one, whose teardown removes any release's install.
$local = if ($PSScriptRoot) { Join-Path $PSScriptRoot 'biopb-engine.ps1' } else { $null }
if ($local -and (Test-Path -LiteralPath $local)) {
    $engine = Get-Content -Raw -LiteralPath $local
} else {
    Write-Host "  Fetching the biopb engine..."
    $engine = Invoke-RestMethod -Uri "https://biopb.org/biopb-engine.ps1"
}

if (-not $Purge -and -not $KeepData) {
    $canAsk = [Environment]::UserInteractive -and -not [Console]::IsInputRedirected -and
        -not ([Environment]::GetCommandLineArgs() -match '^-NonI')
    if ($canAsk) {
        Write-Host ""
        Write-Host "  Also remove biopb configuration and cached data?"
        Write-Host "  (config files, caches, logs and the sample images; your own images are NOT affected)"
        $answer = Read-Host "  Remove them? [y/N]"
        $Purge = [bool]($answer -match '^(y|yes)$')
    }
}
$doPurge = [bool]$Purge

# Dot-sourced at script scope so its functions persist; an in-memory scriptblock
# so it runs under a Restricted ExecutionPolicy. Dot-sourcing rebinds the
# engine's own -Purge parameter here, hence $doPurge above.
. ([scriptblock]::Create($engine))
Invoke-BiopbUninstall -Purge:$doPurge -Mode console
