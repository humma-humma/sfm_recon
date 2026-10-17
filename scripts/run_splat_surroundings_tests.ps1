$ErrorActionPreference = 'Stop'
$RepoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
docker version --format '{{.Server.Version}}'
if ($LASTEXITCODE -ne 0) { throw 'This PowerShell session cannot access Docker.' }
docker run --rm --gpus all `
    -v "${RepoRoot}:/workspace" `
    -v "${RepoRoot}/outputs/nerfstudio_cache:/root/.cache" `
    -w /workspace -e PYTHONPATH=/workspace/src `
    ghcr.io/nerfstudio-project/nerfstudio@sha256:b59b8e1012d7a43679d3234b3de9c8416a4b8435fcbf21b9d8c4494b8563f19e `
    python scripts/test_splat_surroundings.py
if ($LASTEXITCODE -ne 0) { throw 'Experiment failed. See outputs/splat_surroundings_tests/logs and workflow.json; rerun resumes completed cases.' }
Write-Host 'Open outputs/splat_surroundings_tests/index.html for rendered comparisons.'
