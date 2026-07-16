[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$Checkpoint,

    [Parameter(Mandatory = $true)]
    [ValidateSet("lasher", "lasher_val", "rgbt234", "rgbt210", "gtot")]
    [string]$Dataset,

    [Parameter(Mandatory = $true)]
    [string]$DataRoot,

    [int]$Epoch = 12,

    [int]$Workers = 1,

    [string]$Config = "std_full"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot

Push-Location $root
try {
    python tracking/test.py `
        --yaml_name $Config `
        --checkpoint $Checkpoint `
        --dataset $Dataset `
        --data_root $DataRoot `
        --epoch $Epoch `
        --workers $Workers
}
finally {
    Pop-Location
}
