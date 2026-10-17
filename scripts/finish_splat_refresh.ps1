param()
$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$Dataset = (Resolve-Path -LiteralPath (Join-Path $RepoRoot "..\Experiments\Stage_3_Data\stage3")).Path
$Manifest = Join-Path $RepoRoot "outputs\splat_refresh_v1\manifest.json"
if (!(Test-Path -LiteralPath $Manifest)) { throw "Prepared inputs are missing: $Manifest" }
docker version --format '{{.Server.Version}}'
if ($LASTEXITCODE -ne 0) { throw "This PowerShell session cannot access Docker." }
docker run --rm --gpus all `
    -v "${RepoRoot}:/workspace" `
    -v "${RepoRoot}/outputs/nerfstudio_cache:/root/.cache" `
    -v "${Dataset}:/datasets/stage3:ro" `
    -w /workspace -e PYTHONPATH=/workspace/src `
    ghcr.io/nerfstudio-project/nerfstudio@sha256:b59b8e1012d7a43679d3234b3de9c8416a4b8435fcbf21b9d8c4494b8563f19e `
    python scripts/finish_splat_refresh.py
if ($LASTEXITCODE -ne 0) { throw "GPU workflow failed; see outputs/splat_refresh_v1/logs and gpu_workflow.json." }
Write-Host "GPU work finished. Reports and render pairs: outputs/splat_refresh_v1/evaluation"
