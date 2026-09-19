<#
Bootstrap environment on Windows (PowerShell).
Creates a venv and installs requirements.txt.
#>
param(
    [string]$python = "python"
)

$venvPath = "venv"
if (-not (Test-Path $venvPath)) {
    & $python -m venv $venvPath
}

.\venv\Scripts\Activate.ps1
pip install -U pip
if (Test-Path "requirements.txt") {
    pip install -r requirements.txt
} else {
    Write-Host "No requirements.txt found in project root."
}

Write-Host "Environment ready. Activate with: .\venv\Scripts\Activate.ps1"
