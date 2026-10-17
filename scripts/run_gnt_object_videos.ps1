param(
    [string]$TrialRoot=(Join-Path $env:TEMP 'sfm_lightweight_methods_v1'),
    [string]$Destination=(Join-Path $env:USERPROFILE 'sfm_reconstruction_artifacts/gnt_finalization'),
    [string]$Python='C:\Users\mopu01\AppData\Local\anaconda3\envs\mardm_open3d\python.exe',
    [string]$FFmpeg='C:\ffmpeg\bin\ffmpeg.exe'
)
$ErrorActionPreference='Stop'
$ProjectRoot=Split-Path $PSScriptRoot -Parent
$Assets=Join-Path $Destination 'assets'
$VideoRoot=Join-Path $Assets 'videos'
New-Item -ItemType Directory -Force -Path $VideoRoot | Out-Null
$env:PYTHONPATH=Join-Path $ProjectRoot 'src'
& $Python (Join-Path $PSScriptRoot 'publish_gnt_videos.py') --destination $Destination --ffmpeg $FFmpeg
if($LASTEXITCODE -ne 0){throw 'Initial video index failed'}
foreach($Dataset in @('box','boot','milk')){
    & docker run --rm --pull never --gpus all --name sfm-gnt-videos -v "${ProjectRoot}:/workspace:ro" -v "${TrialRoot}:/trial" -v "${VideoRoot}:/videos" -w /workspace -e PYTHONPATH=/workspace/src:/trial/python_deps ghcr.io/nerfstudio-project/nerfstudio:1.1.5 python scripts/render_gnt_object_video.py --trial /trial --output /videos --dataset $Dataset
    if($LASTEXITCODE -ne 0){throw "Video rendering failed: $Dataset. Rerun to resume completed frames."}
    & $Python (Join-Path $PSScriptRoot 'publish_gnt_videos.py') --destination $Destination --ffmpeg $FFmpeg
    if($LASTEXITCODE -ne 0){throw 'Video encoding or verification failed'}
}
Write-Host 'Complete: http://localhost:8088/gnt-videos.html'
