param(
    [string]$Python = 'C:\Users\mopu01\AppData\Local\anaconda3\envs\mardm_open3d\python.exe',
    [string]$Scratch = (Join-Path $env:TEMP 'sfm_sharpness_trials_v1')
)
$ErrorActionPreference = 'Stop'
$RepoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
New-Item -ItemType Directory -Force -Path $Scratch | Out-Null
Push-Location -LiteralPath $RepoRoot
try {
    $env:PYTHONPATH = 'src'
    $Weights = Join-Path $Scratch 'RealESRGAN_x2plus.pth'
    if (-not (Test-Path -LiteralPath $Weights)) {
        Invoke-WebRequest -Uri 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth' -OutFile $Weights
    }
    $Expected = '49FAFD45F8FD7AA8D31AB2A22D14D91B536C34494A5CFE31EB5D89C2FA266ABB'
    if ((Get-FileHash -Algorithm SHA256 -LiteralPath $Weights).Hash -ne $Expected) { throw 'Unexpected pretrained weight contents.' }
    docker run --rm --pull never --gpus all -v "${RepoRoot}:/workspace" -v "${Scratch}:/scratch" -v "${RepoRoot}/outputs/nerfstudio_cache:/root/.cache" -w /workspace -e PYTHONPATH=/workspace/src ghcr.io/nerfstudio-project/nerfstudio:1.1.5 python scripts/render_sharpness_trials.py --scratch /scratch
    if ($LASTEXITCODE -ne 0) { throw 'Fixed-checkpoint rendering failed.' }
    & $Python scripts/evaluate_sharpness_trials.py --scratch $Scratch
    if ($LASTEXITCODE -ne 0) { throw 'Sharpness evaluation failed.' }
    docker run --rm --pull never --gpus all -v "${RepoRoot}:/workspace" -v "${RepoRoot}/outputs/nerfstudio_cache:/root/.cache" -w /workspace ghcr.io/nerfstudio-project/nerfstudio:1.1.5 python scripts/score_sharpness_perceptual.py
    if ($LASTEXITCODE -ne 0) { throw 'Perceptual scoring failed.' }
    & $Python -c "import sys,json; from pathlib import Path; sys.path.insert(0,'scripts'); from evaluate_sharpness_trials import build_page; build_page(json.loads(Path('outputs/sharpness_trials/results.json').read_text()))"
    if ($LASTEXITCODE -ne 0) { throw 'Comparison page generation failed.' }
    Write-Host 'Complete: web_demo/sharpness.html'
} finally { Pop-Location }
