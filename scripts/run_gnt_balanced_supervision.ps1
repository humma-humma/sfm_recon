param(
    [string]$Destination=(Join-Path $env:USERPROFILE 'sfm_reconstruction_artifacts/gnt_finalization'),
    [string]$Python='python'
)
$ProjectRoot=Split-Path $PSScriptRoot -Parent
$env:PYTHONPATH=Join-Path $ProjectRoot 'src'
$Log=Join-Path $Destination 'gnt_splat_scene/logs/balanced_supervision.log'
$ErrorActionPreference='Continue'
& docker run --rm --pull never --gpus all --name sfm-gnt-balanced-supervision -v "${ProjectRoot}:/workspace:ro" -v "${Destination}:/finalization" -w /workspace -e PYTHONPATH=/workspace/src -e TORCH_HOME=/workspace/outputs/nerfstudio_cache/torch ghcr.io/nerfstudio-project/nerfstudio@sha256:b59b8e1012d7a43679d3234b3de9c8416a4b8435fcbf21b9d8c4494b8563f19e python scripts/test_gnt_balanced_supervision.py --root /finalization *> $Log
$Code=$LASTEXITCODE
$ErrorActionPreference='Stop'
if($Code -ne 0){throw "Balanced supervision failed with exit $Code. Check $Log"}
& $Python (Join-Path $PSScriptRoot 'publish_gnt_dense_supervision.py') --root $Destination --balanced
if($LASTEXITCODE -ne 0){throw 'Publication failed'}
& $Python (Join-Path $PSScriptRoot 'verify_gnt_dense_supervision_browser.py') --balanced
if($LASTEXITCODE -ne 0){throw 'Browser validation failed'}
Write-Host 'Complete: http://localhost:8088/gnt-balanced-supervision.html'
