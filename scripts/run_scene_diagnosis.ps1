$ErrorActionPreference = 'Stop'
$RepoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$RawRoot = (Resolve-Path -LiteralPath (Join-Path $RepoRoot '../Experiments/Stage_3_Data/stage3')).Path
docker version --format '{{.Server.Version}}'
if ($LASTEXITCODE -ne 0) { throw 'This PowerShell session cannot access Docker.' }
docker run --rm --gpus all `
    -v "${RepoRoot}:/workspace" `
    -v "${RawRoot}:/workspace/raw_scene:ro" `
    -v "${RepoRoot}/outputs/nerfstudio_cache:/root/.cache" `
    -w /workspace -e PYTHONPATH=/workspace/src `
    ghcr.io/nerfstudio-project/nerfstudio@sha256:b59b8e1012d7a43679d3234b3de9c8416a4b8435fcbf21b9d8c4494b8563f19e `
    python scripts/render_scene_diagnosis.py
if ($LASTEXITCODE -ne 0) { throw 'Scene rendering diagnosis failed.' }
Write-Host 'Complete: outputs/scene_diagnosis/render_complete.json'
