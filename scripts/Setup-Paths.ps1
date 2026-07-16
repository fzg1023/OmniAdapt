[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$DataDir,

    [string]$SaveDir
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
if (-not $SaveDir) {
    $SaveDir = Join-Path $root "output"
}

Push-Location $root
try {
    python tracking/create_default_local_file.py `
        --workspace_dir $root `
        --data_dir $DataDir `
        --save_dir $SaveDir
}
finally {
    Pop-Location
}
