param(
    [string]$Destination=(Join-Path $env:USERPROFILE 'sfm_reconstruction_artifacts/gnt_finalization'),
    [string]$Python='C:\Users\mopu01\AppData\Local\anaconda3\envs\mardm_open3d\python.exe'
)
$ErrorActionPreference='Stop'
$ProjectRoot=Split-Path $PSScriptRoot -Parent
$Log=Join-Path $Destination 'gnt_splat_scene/logs/diagnose_transfer.log'
# Native warnings on stderr are not failures; inspect the actual process exit.
$ErrorActionPreference='Continue'
& docker run --rm --pull never --gpus all --name sfm-gnt-transfer-diagnosis -v "${ProjectRoot}:/workspace:ro" -v "${Destination}:/finalization" -w /workspace -e PYTHONPATH=/workspace/src -e TORCH_HOME=/workspace/outputs/nerfstudio_cache/torch ghcr.io/nerfstudio-project/nerfstudio:1.1.5 python scripts/diagnose_gnt_transfer.py --root /finalization *> $Log
$Code=$LASTEXITCODE
$ErrorActionPreference='Stop'
if($Code -ne 0){throw "Diagnosis failed with exit $Code. Check $Log"}
& $Python (Join-Path $PSScriptRoot 'publish_gnt_transfer_diagnosis.py') --root $Destination
if($LASTEXITCODE -ne 0){throw 'Diagnosis publication failed'}
& $Python (Join-Path $PSScriptRoot 'verify_gnt_transfer_browser.py')
if($LASTEXITCODE -ne 0){throw 'Diagnosis browser validation failed'}
Write-Host 'Complete: http://localhost:8088/gnt-transfer-diagnosis.html'
