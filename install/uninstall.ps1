<#
.SYNOPSIS
    biopb stack uninstaller (Windows / PowerShell)
.DESCRIPTION
    Saved with the install, beside that release's engine, by biopb-engine.ps1:
    run %USERPROFILE%\.local\share\biopb\uninstall\uninstall.cmd. Removes the
    install through the engine's teardown (the GUI install's Add/Remove Programs
    entry runs the same one). Asks whether to also delete config and cached
    data; your images are never touched. -Purge or -KeepData answers up front.
#>
param(
    [switch]$Purge,
    [switch]$KeepData
)
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

# The engine of the release that installed this, saved beside it.
$enginePath = if ($PSScriptRoot) { Join-Path $PSScriptRoot 'biopb-engine.ps1' } else { '' }
if (-not $enginePath -or -not (Test-Path -LiteralPath $enginePath)) {
    throw "biopb-engine.ps1 not found beside uninstall.ps1; run the uninstall.cmd saved with your install"
}
$engine = Get-Content -Raw -LiteralPath $enginePath

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
