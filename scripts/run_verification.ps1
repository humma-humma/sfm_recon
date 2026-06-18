param(
    [string]$Python = $env:SFM_PYTHON
)

$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")

if ([string]::IsNullOrWhiteSpace($Python)) {
    $LocalOpen3D = "C:\Users\mopu01\AppData\Local\anaconda3\envs\mardm_open3d\python.exe"
    if (Test-Path -LiteralPath $LocalOpen3D) {
        $Python = $LocalOpen3D
    } else {
        $Python = "python"
    }
}

Push-Location -LiteralPath $RepoRoot
try {
    $env:PYTHONPATH = "src"
    $env:PYTHONDONTWRITEBYTECODE = "1"

    & $Python -m pytest -q -p no:cacheprovider
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }

    & $Python "scripts\verify_regression.py" --root "."
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
} finally {
    Pop-Location
}
