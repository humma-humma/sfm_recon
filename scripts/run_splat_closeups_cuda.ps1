$ErrorActionPreference = 'Stop'
$RepoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
if (!(Test-Path -LiteralPath (Join-Path $RepoRoot 'outputs/splat_closeups/scene/camera_cases.json'))) {
    throw 'Run scripts/test_splat_closeups.py locally to prepare the matching cameras first.'
}
docker version --format '{{.Server.Version}}'
if ($LASTEXITCODE -ne 0) { throw 'This PowerShell session cannot access Docker.' }
docker run --rm --gpus all `
    -v "${RepoRoot}:/workspace" `
    -v "${RepoRoot}/outputs/nerfstudio_cache:/root/.cache" `
    -w /workspace `
    ghcr.io/nerfstudio-project/nerfstudio:1.1.5 `
    python scripts/render_splat_closeups_cuda.py
if ($LASTEXITCODE -ne 0) { throw 'Native close-up rendering failed. See outputs/splat_closeups/native_report.json.' }
Write-Host 'Open outputs/splat_closeups/native.html to compare all four datasets.'
