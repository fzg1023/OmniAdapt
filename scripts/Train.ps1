[CmdletBinding()]
param(
    [string]$Config = "std_full",

    [ValidateSet("single", "multiple", "multi_node")]
    [string]$Mode = "single",

    [int]$NumGpus = 1,

    [string]$SaveDir
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
if (-not $SaveDir) {
    $SaveDir = Join-Path $root "output"
}

Push-Location $root
try {
    python tracking/train.py `
        --script omniadapt `
        --config $Config `
        --mode $Mode `
        --nproc_per_node $NumGpus `
        --save_dir $SaveDir
}
finally {
    Pop-Location
}
