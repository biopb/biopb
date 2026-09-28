<#
.SYNOPSIS
    biopb bootstrap: what https://biopb.org/install.ps1 serves.

.DESCRIPTION
    Usage: irm https://biopb.org/install.ps1 | iex

    It installs nothing itself. It checks the machine can run an installer, picks
    a release, downloads that release's own install.ps1 asset and runs it, so the
    installer that runs is always the one written for the release it installs. It
    carries no release logic of its own and so does not change from release to
    release.

    The release is chosen the way the installer chooses it:
    $env:BIOPB_INSTALL_VERSION = "X.Y.Z"  that exact release (release-vX.Y.Z, vX.Y.Z also work)
    $env:BIOPB_INSTALL_RC = "1"           the latest release candidate
    otherwise                             the latest stable release

    The environment reaches the installer as it is.

    Requirements: PowerShell 5.1+, tar (bundled on Windows 10 1803+).
#>

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

$BootstrapRepo = 'biopb/biopb'
$BootstrapPrefix = 'release-v'

# The tag of the release to install.
function Resolve-BiopbTag {
    $version = $env:BIOPB_INSTALL_VERSION
    if ($version) {
        if     ($version.StartsWith($BootstrapPrefix)) { return $version }
        elseif ($version.StartsWith('v'))              { return "$BootstrapPrefix$($version.Substring(1))" }
        else                                           { return "$BootstrapPrefix$version" }
    }
    # The repo hosts several release lines, so /releases/latest is not ours:
    # take the newest release-v* tag, a clean X.Y.Z unless candidates are wanted.
    $re = "^$([regex]::Escape($BootstrapPrefix))\d+\.\d+\.\d+$"
    if ($env:BIOPB_INSTALL_RC -and $env:BIOPB_INSTALL_RC -ne '0') {
        $re = "^$([regex]::Escape($BootstrapPrefix))\d+\.\d+\.\d+((a|b|rc)\d+)?$"
    }
    $releases = Invoke-RestMethod -Uri "https://api.github.com/repos/$BootstrapRepo/releases?per_page=100"
    $tag = @($releases | ForEach-Object { $_.tag_name } | Where-Object { $_ -match $re }) | Select-Object -First 1
    if (-not $tag) { throw "Could not find a biopb release to install (network, or GitHub rate limit). Name one with `$env:BIOPB_INSTALL_VERSION = 'X.Y.Z'." }
    return $tag
}

function Invoke-BiopbBootstrap {
    if (-not (Get-Command tar -ErrorAction SilentlyContinue)) {
        throw "tar is required but not found (it ships with Windows 10 1803 and later)."
    }
    $tag = Resolve-BiopbTag
    # The tag goes into a URL: refuse anything that is not a plain tag name.
    if ($tag -notmatch '^[A-Za-z0-9._+-]+$') { throw "Unexpected release tag: $tag" }

    $url = "https://github.com/$BootstrapRepo/releases/download/$tag/install.ps1"
    Write-Host "Fetching the $tag installer..."
    try {
        $response = Invoke-WebRequest -UseBasicParsing -Uri $url
    } catch {
        throw "Could not download $url -- the release may not exist, or may predate its own installer."
    }
    $text = $response.Content
    if ($text -is [byte[]]) { $text = [System.Text.Encoding]::UTF8.GetString($text) }
    $text = $text.TrimStart([char]0xFEFF)

    # In memory, not a file: a factory-default ExecutionPolicy blocks a script
    # file but not this (the same reason install.ps1 loads its engine this way).
    try {
        $installer = [scriptblock]::Create($text)
    } catch {
        throw "The downloaded installer is not a valid script."
    }
    & $installer
}

# Last, and not run when BIOPB_INSTALL_LIB is set, so the tests can load the
# helpers alone (the twin of install.sh's guard).
if (-not $env:BIOPB_INSTALL_LIB) {
    try {
        Invoke-BiopbBootstrap
    } catch {
        Write-Host ""
        Write-Host "ERROR: $($_.Exception.Message)" -ForegroundColor Red
        exit 1
    }
}
