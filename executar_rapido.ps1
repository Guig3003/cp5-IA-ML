$ErrorActionPreference = "Stop"
$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $python)) {
    throw "Ambiente virtual ausente. Consulte o README.md para a instalacao."
}

& $python (Join-Path $PSScriptRoot "mnist_mlp.py") --mode quick
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

