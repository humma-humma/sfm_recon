param([string]$Python = 'python')
$ErrorActionPreference='Stop'
$RepoRoot=(Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$RawRoot=(Resolve-Path -LiteralPath (Join-Path $RepoRoot '../Experiments/Stage_3_Data/stage3')).Path
Push-Location -LiteralPath $RepoRoot
try {
    $env:PYTHONPATH='src'
    $Checkpoint='outputs/depth_research/train/scene_depth/splatfacto/run_10000/nerfstudio_models/step-000009999.ckpt'
    if (-not (Test-Path -LiteralPath $Checkpoint)) {
        $DriveName=[System.IO.Path]::GetPathRoot($RepoRoot).Substring(0,1)
        $Available=(Get-PSDrive -Name $DriveName).Free
        if ($null -ne $Available -and $Available -lt 1GB) { throw 'Free at least 1 GB on the project drive before starting a new experiment.' }
    }
    if (-not (Test-Path -LiteralPath 'outputs/depth_research/plan.json')) {
        & $Python scripts/prepare_depth_research.py
        if ($LASTEXITCODE -ne 0) { throw 'Depth preparation failed' }
    }
    if (-not (Test-Path -LiteralPath 'outputs/depth_research/tsdf_metrics.json')) {
        & $Python scripts/prepare_depth_research.py --mesh-only
        if ($LASTEXITCODE -ne 0) { throw 'TSDF baseline failed' }
    }
    if (-not (Test-Path -LiteralPath $Checkpoint)) {
        docker run --rm --pull never --gpus all -v "${RepoRoot}:/workspace" -v "${RepoRoot}/outputs/nerfstudio_cache:/root/.cache" -w /workspace -e PYTHONPATH=/workspace/src ghcr.io/nerfstudio-project/nerfstudio@sha256:b59b8e1012d7a43679d3234b3de9c8416a4b8435fcbf21b9d8c4494b8563f19e python scripts/train_depth_research.py
        if ($LASTEXITCODE -ne 0) { throw 'Depth training failed' }
    }
    if (-not (Test-Path -LiteralPath 'outputs/depth_research/evaluation/metrics.json')) {
        docker run --rm --pull never --gpus all -v "${RepoRoot}:/workspace" -v "${RawRoot}:/workspace/raw_scene:ro" -v "${RepoRoot}/outputs/nerfstudio_cache:/root/.cache" -w /workspace -e PYTHONPATH=/workspace/src ghcr.io/nerfstudio-project/nerfstudio@sha256:b59b8e1012d7a43679d3234b3de9c8416a4b8435fcbf21b9d8c4494b8563f19e python scripts/evaluate_splat_checkpoint.py --config /workspace/outputs/splat_surroundings_tests/train/scene_control/splatfacto/run_5000/config.yml --checkpoint "/workspace/$Checkpoint" --output /workspace/outputs/depth_research/evaluation --depth-root /workspace/raw_scene/depth
        if ($LASTEXITCODE -ne 0) { throw 'Depth evaluation failed' }
    }
    & $Python scripts/report_depth_research.py
    if ($LASTEXITCODE -ne 0) { throw 'Depth report failed' }
    Write-Host 'Complete: web_demo/research.html'
} finally { Pop-Location }
