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
    ghcr.io/nerfstudio-project/nerfstudio:1.1.5 `
    python scripts/render_scene_diagnosis.py
if ($LASTEXITCODE -ne 0) { throw 'Scene rendering diagnosis failed.' }
Write-Host 'Complete: outputs/scene_diagnosis/render_complete.json'
