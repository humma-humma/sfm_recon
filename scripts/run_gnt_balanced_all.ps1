param(
    [string]$Destination=(Join-Path $env:USERPROFILE 'sfm_reconstruction_artifacts/gnt_finalization'),
    [string]$TrialRoot=(Join-Path $env:TEMP 'sfm_lightweight_methods_v1'),
    [string]$Python='C:\Users\mopu01\AppData\Local\anaconda3\envs\mardm_open3d\python.exe'
)
$ErrorActionPreference='Stop'
$ProjectRoot=Split-Path $PSScriptRoot -Parent
$env:PYTHONPATH=Join-Path $ProjectRoot 'src'
$Base=Join-Path $Destination 'balanced_rollout'
$Logs=Join-Path $Base 'logs'
New-Item -ItemType Directory -Force -Path $Logs | Out-Null
$Status=Join-Path $Base 'job.json'
$State=@{status='running';pid=$PID;started_utc=[DateTime]::UtcNow.ToString('o');dataset='';phase='preflight';datasets=@{scene='pending';box='pending';boot='pending';milk='pending'}}
function Save-State {
    $State.updated_utc=[DateTime]::UtcNow.ToString('o')
    [IO.File]::WriteAllText("${Status}.tmp",($State | ConvertTo-Json -Depth 8),[Text.UTF8Encoding]::new($false))
    Move-Item -LiteralPath "${Status}.tmp" -Destination $Status -Force
}
function Invoke-Gpu([string]$Dataset,[string]$Phase,[string[]]$PythonArgs){
    $State.dataset=$Dataset;$State.phase=$Phase;$State.datasets[$Dataset]=$Phase;Save-State
    $Log=Join-Path $Logs "${Dataset}_${Phase}.log"
    Write-Host "Starting $Dataset / $Phase; log: $Log"
    $ErrorActionPreference='Continue'
    & docker run --rm --pull never --gpus all --name "sfm-balanced-all-${Dataset}-${Phase}" -v "${ProjectRoot}:/workspace:ro" -v "${Destination}:/finalization" -v "${TrialRoot}:/trial" -w /workspace -e PYTHONPATH=/workspace/src:/trial/python_deps -e TORCH_HOME=/workspace/outputs/nerfstudio_cache/torch ghcr.io/nerfstudio-project/nerfstudio:1.1.5 python @PythonArgs *> $Log
    $Code=$LASTEXITCODE
    $ErrorActionPreference='Stop'
    if($Code -ne 0){throw "$Dataset / $Phase exited $Code. See $Log"}
}
function Publish-Dataset([string]$Dataset){
    $State.dataset=$Dataset;$State.phase='encoding';$State.datasets[$Dataset]='encoding';Save-State
    & $Python (Join-Path $PSScriptRoot 'publish_gnt_balanced_rollout.py') --root $Destination --dataset $Dataset
    if($LASTEXITCODE -ne 0){throw "Publication failed for $Dataset"}
    $State.datasets[$Dataset]='complete';Save-State
}
try {
    Save-State
    & $Python (Join-Path $PSScriptRoot 'publish_gnt_balanced_rollout.py') --root $Destination
    if($LASTEXITCODE -ne 0){throw 'Initial publication failed'}
    foreach($Dataset in @('box','boot','milk')){
        Invoke-Gpu $Dataset 'preflight' @('scripts/render_gnt_rollout_teachers.py','--root','/finalization','--trial','/trial','--dataset',$Dataset,'--preflight')
    }
    Invoke-Gpu 'scene' 'export' @('scripts/export_gnt_balanced_rollout.py','--root','/finalization','--dataset','scene')
    Publish-Dataset 'scene'
    foreach($Dataset in @('box','boot','milk')){
        Invoke-Gpu $Dataset 'sparse' @('scripts/render_gnt_object_video.py','--trial','/trial','--output','/finalization/assets/videos','--dataset',$Dataset)
        Invoke-Gpu $Dataset 'dense' @('scripts/render_gnt_rollout_teachers.py','--root','/finalization','--trial','/trial','--dataset',$Dataset)
        Invoke-Gpu $Dataset 'fit' @('scripts/fit_gnt_rollout_object.py','--root','/finalization','--dataset',$Dataset)
        Invoke-Gpu $Dataset 'export' @('scripts/export_gnt_balanced_rollout.py','--root','/finalization','--dataset',$Dataset)
        Publish-Dataset $Dataset
    }
    & $Python (Join-Path $PSScriptRoot 'verify_gnt_balanced_all_browser.py')
    if($LASTEXITCODE -ne 0){throw 'Final browser validation failed'}
    $State.status='complete';$State.phase='complete';Save-State
    Write-Host 'Complete: http://localhost:8088/gnt-balanced-all.html'
} catch {
    $State.status='failed';$State.error=$_.Exception.Message;Save-State
    throw
}
