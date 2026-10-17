param(
    [int]$MaxAdded=48,
    [int]$MaxProbes=96,
    [string]$TrialRoot=(Join-Path $env:TEMP 'sfm_lightweight_methods_v1'),
    [string]$Original=(Join-Path $env:USERPROFILE 'sfm_reconstruction_artifacts/gnt_traversal_control6000_960'),
    [string]$Destination=(Join-Path $env:USERPROFILE 'sfm_reconstruction_artifacts/gnt_finalization'),
    [string]$Python='C:\Users\mopu01\AppData\Local\anaconda3\envs\mardm_open3d\python.exe',
    [string]$FFmpeg='C:\ffmpeg\bin\ffmpeg.exe'
)
$ErrorActionPreference='Stop'
$ProjectRoot=Split-Path $PSScriptRoot -Parent
$Output=Join-Path $Destination 'assets/adaptive_scene'
New-Item -ItemType Directory -Force -Path $Output | Out-Null
& $Python (Join-Path $PSScriptRoot 'publish_gnt_adaptive.py') --original $Original --output $Output --ffmpeg $FFmpeg
if($LASTEXITCODE -ne 0){throw 'Adaptive index publication failed'}
& docker run --rm --pull never --gpus all --name sfm-gnt-adaptive -v "${ProjectRoot}:/workspace:ro" -v "${TrialRoot}:/trial" -v "${Original}:/original:ro" -v "${Output}:/adaptive" -w /workspace -e PYTHONPATH=/workspace/src:/trial/python_deps ghcr.io/nerfstudio-project/nerfstudio:1.1.5 python scripts/render_gnt_adaptive.py --trial /trial --original /original --output /adaptive --max-added $MaxAdded --max-probes $MaxProbes
if($LASTEXITCODE -ne 0){throw 'Adaptive rendering stopped. Rerun with the same budget to resume.'}
& $Python (Join-Path $PSScriptRoot 'publish_gnt_adaptive.py') --original $Original --output $Output --ffmpeg $FFmpeg
if($LASTEXITCODE -ne 0){throw 'Adaptive export or verification failed'}
Write-Host 'Complete: http://localhost:8088/gnt-adaptive.html'
