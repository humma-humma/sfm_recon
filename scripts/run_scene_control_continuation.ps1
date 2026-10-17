$ErrorActionPreference = 'Stop'
$RepoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
docker version --format '{{.Server.Version}}'
if ($LASTEXITCODE -ne 0) { throw 'This PowerShell session cannot access Docker.' }
docker run --rm --gpus all `
    -v "${RepoRoot}:/workspace" `
    -v "${RepoRoot}/outputs/nerfstudio_cache:/root/.cache" `
    -w /workspace -e PYTHONPATH=/workspace/src `
    ghcr.io/nerfstudio-project/nerfstudio@sha256:b59b8e1012d7a43679d3234b3de9c8416a4b8435fcbf21b9d8c4494b8563f19e `
    python scripts/continue_scene_control.py
if ($LASTEXITCODE -ne 0) { throw 'Continuation failed. See outputs/scene_control_continuation/logs.' }
Write-Host 'Open outputs/scene_control_continuation/index.html to compare source / fresh 5k / 10k / 15k.'
