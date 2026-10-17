param(
    [string]$TrialRoot=(Join-Path $env:TEMP 'sfm_lightweight_methods_v1'),
    [string]$Original=(Join-Path $env:USERPROFILE 'sfm_reconstruction_artifacts/gnt_traversal_control6000_960'),
    [string]$Destination=(Join-Path $env:USERPROFILE 'sfm_reconstruction_artifacts/gnt_finalization'),
    [string]$Python='python',
    [string]$FFmpeg='ffmpeg'
)
$ErrorActionPreference='Stop'
$ProjectRoot=Split-Path $PSScriptRoot -Parent
$Output=Join-Path $Destination 'assets/dense_turn'
$Adaptive=Join-Path $Destination 'assets/adaptive_scene'
New-Item -ItemType Directory -Force -Path $Output | Out-Null
& $Python (Join-Path $PSScriptRoot 'publish_gnt_dense_turn.py') --output $Output --ffmpeg $FFmpeg
if($LASTEXITCODE -ne 0){throw 'Dense clip index failed'}
& docker run --rm --pull never --gpus all --name sfm-gnt-dense-turn -v "${ProjectRoot}:/workspace:ro" -v "${TrialRoot}:/trial" -v "${Original}:/original:ro" -v "${Adaptive}:/adaptive:ro" -v "${Output}:/dense" -w /workspace -e PYTHONPATH=/workspace/src:/trial/python_deps ghcr.io/nerfstudio-project/nerfstudio@sha256:b59b8e1012d7a43679d3234b3de9c8416a4b8435fcbf21b9d8c4494b8563f19e python scripts/render_gnt_dense_turn.py --trial /trial --original /original --adaptive /adaptive --output /dense
if($LASTEXITCODE -ne 0){throw 'Dense rendering stopped; rerun to resume saved frames.'}
& $Python (Join-Path $PSScriptRoot 'publish_gnt_dense_turn.py') --output $Output --ffmpeg $FFmpeg
if($LASTEXITCODE -ne 0){throw 'Dense clip export or verification failed'}
Write-Host 'Complete: http://localhost:8088/gnt-dense-turn.html'
