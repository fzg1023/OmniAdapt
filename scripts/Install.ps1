[CmdletBinding()]
param(
    [string]$PythonBin = "python"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot

Push-Location $root
try {
    & $PythonBin -m pip install --upgrade pip
    & $PythonBin -m pip install -r requirements.txt
}
finally {
    Pop-Location
}

Write-Host "Dependencies installed. Install a PyTorch/torchvision build matching your CUDA version if it is not already available."
