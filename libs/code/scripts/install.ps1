# Install the released deepagents-code package as an isolated uv tool.

[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$UvInstallationDocs = "https://docs.astral.sh/uv/getting-started/installation/"
$ValidPrereleaseStrategies = @(
    "disallow",
    "allow",
    "if-necessary",
    "explicit",
    "if-necessary-or-explicit"
)

function Add-CurrentPathEntry {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Directory
    )

    if ([string]::IsNullOrWhiteSpace($Directory)) {
        return
    }

    $entries = @($env:Path -split ";" | Where-Object { $_ })
    $remainingEntries = @(
        $entries | Where-Object {
            -not [string]::Equals(
                $_,
                $Directory,
                [StringComparison]::OrdinalIgnoreCase
            )
        }
    )
    $env:Path = (@($Directory) + $remainingEntries) -join ";"
}

function Get-CanonicalPath {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path
    )

    $resolved = Resolve-Path -LiteralPath $Path -ErrorAction Stop
    return [System.IO.Path]::GetFullPath($resolved.ProviderPath)
}

function Get-AbsolutePath {
    param([string]$Path)

    if ([string]::IsNullOrWhiteSpace($Path)) {
        return $null
    }

    try {
        $expandedPath = [Environment]::ExpandEnvironmentVariables($Path.Trim())
        if ($expandedPath.StartsWith("~")) {
            $expandedPath = Join-Path $HOME $expandedPath.Substring(1).TrimStart(
                '\',
                '/'
            )
        }
        return [System.IO.Path]::GetFullPath($expandedPath)
    } catch {
        return $null
    }
}

function Get-ValidDirectoryPath {
    param([string]$Path)

    $absolutePath = Get-AbsolutePath -Path $Path
    if (
        [string]::IsNullOrWhiteSpace($absolutePath) -or
        -not (Test-Path -LiteralPath $absolutePath -PathType Container)
    ) {
        return $null
    }

    try {
        return Get-CanonicalPath -Path $absolutePath
    } catch {
        return $null
    }
}

function Refresh-CurrentPath {
    foreach ($scope in @("User", "Machine")) {
        $persistedPath = [Environment]::GetEnvironmentVariable("Path", $scope)
        if (-not [string]::IsNullOrWhiteSpace($persistedPath)) {
            foreach ($entry in $persistedPath -split ";") {
                if (-not [string]::IsNullOrWhiteSpace($entry)) {
                    Add-CurrentPathEntry -Directory $entry
                }
            }
        }
    }

    Add-CurrentPathEntry -Directory (Join-Path $HOME ".local\bin")
}

function Find-Uv {
    $command = Get-Command "uv" `
        -CommandType Application `
        -ErrorAction SilentlyContinue |
        Select-Object -First 1
    if ($null -ne $command) {
        return $command.Source
    }

    foreach ($candidate in @(
        (Join-Path $HOME ".local\bin\uv.exe"),
        (Join-Path $env:LOCALAPPDATA "uv\bin\uv.exe")
    )) {
        if (Test-Path -LiteralPath $candidate -PathType Leaf) {
            return $candidate
        }
    }

    return $null
}

function Stop-MissingUv {
    $message = @(
        "uv is required but was not found."
        "This installer does not download or execute remote PowerShell scripts."
        ""
        "Preferred installation on supported Windows:"
        "  winget install --id astral-sh.uv -e"
        ""
        "Then open a new PowerShell window or refresh the current PATH and rerun this script."
        "Other trusted installation options: $UvInstallationDocs"
    ) -join [Environment]::NewLine

    [Console]::Error.WriteLine($message)
    exit 1
}

function Get-Extras {
    $rawExtras = [Environment]::GetEnvironmentVariable("DEEPAGENTS_CODE_EXTRAS")
    if ([string]::IsNullOrWhiteSpace($rawExtras)) {
        return @()
    }

    $extras = @()
    foreach ($rawExtra in $rawExtras.Split(",")) {
        $extra = $rawExtra.Trim()
        if ($extra -notmatch "^[A-Za-z0-9][A-Za-z0-9-]*$") {
            throw (
                "DEEPAGENTS_CODE_EXTRAS must contain comma-separated extra " +
                "names such as 'anthropic,ollama'."
            )
        }
        $extras += $extra
    }

    return $extras
}

function Get-PrereleaseStrategy {
    $strategy = [Environment]::GetEnvironmentVariable(
        "DEEPAGENTS_CODE_PRERELEASE"
    )
    if ([string]::IsNullOrWhiteSpace($strategy)) {
        return "allow"
    }

    $strategy = $strategy.Trim().ToLowerInvariant()
    if ($ValidPrereleaseStrategies -notcontains $strategy) {
        throw (
            "DEEPAGENTS_CODE_PRERELEASE must be one of: " +
            ($ValidPrereleaseStrategies -join ", ")
        )
    }

    return $strategy
}

function Get-ToolBinDirectory {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Uv
    )

    $toolBin = $null
    $toolBinOutput = @()
    $toolBinExitCode = -1
    $previousErrorActionPreference = $ErrorActionPreference
    $nativeErrorPreference = Get-Variable `
        -Name "PSNativeCommandUseErrorActionPreference" `
        -ErrorAction SilentlyContinue
    $hasNativeErrorPreference = $null -ne $nativeErrorPreference
    $previousNativeErrorPreference = $false
    if ($hasNativeErrorPreference) {
        $previousNativeErrorPreference = [bool]$nativeErrorPreference.Value
    }

    try {
        # Windows PowerShell 5.1 turns redirected native stderr into error
        # records. Probe older uv without letting those records terminate the
        # installer before the documented legacy fallbacks can run.
        $ErrorActionPreference = "Continue"
        if ($hasNativeErrorPreference) {
            Set-Variable `
                -Name "PSNativeCommandUseErrorActionPreference" `
                -Value $false `
                -Scope Local
        }
        $toolBinOutput = @(& $Uv tool dir --bin 2>&1)
        $toolBinExitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $previousErrorActionPreference
        if ($hasNativeErrorPreference) {
            Set-Variable `
                -Name "PSNativeCommandUseErrorActionPreference" `
                -Value $previousNativeErrorPreference `
                -Scope Local
        }
    }

    if ($toolBinExitCode -eq 0) {
        $toolBin = $toolBinOutput |
            Where-Object { -not [string]::IsNullOrWhiteSpace([string]$_) } |
            Select-Object -First 1
    }

    $absoluteToolBin = Get-ValidDirectoryPath -Path ([string]$toolBin)
    if (-not [string]::IsNullOrWhiteSpace($absoluteToolBin)) {
        return $absoluteToolBin
    }

    $xdgDataHome = [Environment]::GetEnvironmentVariable("XDG_DATA_HOME")
    $xdgDataBin = $null
    if (-not [string]::IsNullOrWhiteSpace($xdgDataHome)) {
        $xdgDataBin = Join-Path $xdgDataHome "..\bin"
    }

    foreach ($candidate in @(
        [Environment]::GetEnvironmentVariable("UV_TOOL_BIN_DIR"),
        [Environment]::GetEnvironmentVariable("XDG_BIN_HOME"),
        $xdgDataBin,
        (Join-Path $HOME ".local\bin")
    )) {
        $absoluteToolBin = Get-ValidDirectoryPath -Path ([string]$candidate)
        if (-not [string]::IsNullOrWhiteSpace($absoluteToolBin)) {
            return $absoluteToolBin
        }
    }

    throw "uv installed deepagents-code, but its tool bin directory could not be located."
}

$uv = Find-Uv
if ([string]::IsNullOrWhiteSpace($uv)) {
    Stop-MissingUv
}

$extras = @(Get-Extras)
$requirement = "deepagents-code"
if ($extras.Count -gt 0) {
    $requirement += "[$($extras -join ',')]"
}
$prerelease = Get-PrereleaseStrategy

Write-Host "Installing $requirement..."
& $uv tool install -U --prerelease $prerelease $requirement
if ($LASTEXITCODE -ne 0) {
    throw "uv tool install failed with exit code $LASTEXITCODE."
}

$toolBin = Get-ToolBinDirectory -Uv $uv

& $uv tool update-shell | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Warning "uv could not update the persistent shell PATH automatically."
}
Refresh-CurrentPath
Add-CurrentPathEntry -Directory $toolBin

$dcode = $null
foreach ($name in @("dcode.exe", "dcode.cmd", "dcode")) {
    $candidate = Join-Path $toolBin $name
    if (Test-Path -LiteralPath $candidate -PathType Leaf) {
        $dcode = $candidate
        break
    }
}
if ([string]::IsNullOrWhiteSpace($dcode)) {
    throw "deepagents-code installed, but dcode was not found in $toolBin."
}

$dcodeCommand = Get-Command "dcode" `
    -CommandType Application `
    -ErrorAction SilentlyContinue |
    Select-Object -First 1
if ($null -eq $dcodeCommand) {
    throw (
        "deepagents-code installed dcode at '$dcode', but " +
        "'Get-Command dcode -CommandType Application' could not resolve it " +
        "after '$toolBin' was prepended to PATH."
    )
}
$installedDcode = Get-CanonicalPath -Path $dcode
$resolvedDcode = Get-CanonicalPath -Path ([string]$dcodeCommand.Source)
if (-not [string]::Equals(
    $installedDcode,
    $resolvedDcode,
    [StringComparison]::OrdinalIgnoreCase
)) {
    throw (
        "deepagents-code installed dcode at '$installedDcode', but " +
        "'Get-Command dcode -CommandType Application' resolves to " +
        "'$resolvedDcode'. The installed shim is shadowed on PATH."
    )
}

$versionOutput = @(& $dcode -v 2>&1)
if ($LASTEXITCODE -ne 0) {
    throw "deepagents-code installed, but 'dcode -v' failed."
}

Write-Host ($versionOutput -join [Environment]::NewLine)
Write-Host "Installation complete. Run: dcode"
Write-Host "If dcode is not found in an existing terminal, open a new PowerShell or cmd window."
