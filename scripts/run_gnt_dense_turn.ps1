param(
    [string]$TrialRoot=(Join-Path $env:TEMP 'sfm_lightweight_methods_v1'),
    [string]$Original=(Join-Path $env:USERPROFILE 'sfm_reconstruction_artifacts/gnt_traversal_control6000_960'),
    [string]$Destination=(Join-Path $env:USERPROFILE 'sfm_reconstruction_artifacts/gnt_finalization'),
    [string]$Python='C:\Users\mopu01\AppData\Local\anaconda3\envs\mardm_open3d\python.exe',
    [string]$FFmpeg='C:\ffmpeg\bin\ffmpeg.exe'
)
$ErrorActionPreference='Stop'
$ProjectRoot=Split-Path $PSScriptRoot -Parent
$Output=Join-Path $Destination 'assets/dense_turn'
$Adaptive=Join-Path $Destination 'assets/adaptive_scene'
New-Item -ItemType Directory -Force -Path $Output | Out-Null
& $Python (Join-Path $PSScriptRoot 'publish_gnt_dense_turn.py') --output $Output --ffmpeg $FFmpeg
if($LASTEXITCODE -ne 0){throw 'Dense clip index failed'}
& docker run --rm --pull never --gpus all --name sfm-gnt-dense-turn -v "${ProjectRoot}:/workspace:ro" -v "${TrialRoot}:/trial" -v "${Original}:/original:ro" -v "${Adaptive}:/adaptive:ro" -v "${Output}:/dense" -w /workspace -e PYTHONPATH=/workspace/src:/trial/python_deps ghcr.io/nerfstudio-project/nerfstudio:1.1.5 python scripts/render_gnt_dense_turn.py --trial /trial --original /original --adaptive /adaptive --output /dense
if($LASTEXITCODE -ne 0){throw 'Dense rendering stopped; rerun to resume saved frames.'}
& $Python (Join-Path $PSScriptRoot 'publish_gnt_dense_turn.py') --output $Output --ffmpeg $FFmpeg
if($LASTEXITCODE -ne 0){throw 'Dense clip export or verification failed'}
Write-Host 'Complete: http://localhost:8088/gnt-dense-turn.html'
