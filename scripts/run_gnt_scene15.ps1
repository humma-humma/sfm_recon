param(
    [string]$Destination=(Join-Path $env:USERPROFILE 'sfm_reconstruction_artifacts/gnt_finalization'),
    [string]$TrialRoot=(Join-Path $env:TEMP 'sfm_lightweight_methods_v1'),
    [string]$Original=(Join-Path $env:USERPROFILE 'sfm_reconstruction_artifacts/gnt_traversal_control6000_960'),
    [string]$Python='C:\Users\mopu01\AppData\Local\anaconda3\envs\mardm_open3d\python.exe'
)
$ErrorActionPreference='Stop'
$ProjectRoot=Split-Path $PSScriptRoot -Parent
$env:PYTHONPATH=Join-Path $ProjectRoot 'src'
$Output=Join-Path $Destination 'assets/scene15'
New-Item -ItemType Directory -Force -Path $Output | Out-Null
$Job=Join-Path $Output 'job.json'
$State=@{status='running';phase='starting';pid=$PID;started_utc=[DateTime]::UtcNow.ToString('o')}
function Save-State {
    $State.updated_utc=[DateTime]::UtcNow.ToString('o')
    [IO.File]::WriteAllText("${Job}.tmp",($State | ConvertTo-Json),[Text.UTF8Encoding]::new($false))
    Move-Item -LiteralPath "${Job}.tmp" -Destination $Job -Force
}
try {
    Save-State
    & $Python (Join-Path $PSScriptRoot 'publish_gnt_scene15.py') --root $Destination
    if($LASTEXITCODE -ne 0){throw 'Initial publication failed'}
    $State.phase='rendering';Save-State
    $ErrorActionPreference='Continue'
    & docker run --rm --pull never --gpus all --name sfm-gnt-scene15 -v "${ProjectRoot}:/workspace:ro" -v "${Destination}:/finalization" -v "${TrialRoot}:/trial" -v "${Original}:/original:ro" -w /workspace -e PYTHONPATH=/workspace/src:/trial/python_deps -e TORCH_HOME=/workspace/outputs/nerfstudio_cache/torch ghcr.io/nerfstudio-project/nerfstudio:1.1.5 python scripts/render_gnt_scene15.py --root /finalization --trial /trial --original /original *> (Join-Path $Output 'render.log')
    $Code=$LASTEXITCODE
    $ErrorActionPreference='Stop'
    if($Code -ne 0){throw "Rendering exited $Code; see render.log. Rerun to resume verified frames."}
    $State.phase='encoding';Save-State
    & $Python (Join-Path $PSScriptRoot 'publish_gnt_scene15.py') --root $Destination
    if($LASTEXITCODE -ne 0){throw 'Video export/verification failed'}
    $State.phase='browser_verification';Save-State
    & $Python (Join-Path $PSScriptRoot 'verify_gnt_scene15_browser.py')
    if($LASTEXITCODE -ne 0){throw 'Browser verification failed'}
    $State.phase='complete';$State.status='complete';Save-State
    Write-Host 'Complete: http://localhost:8088/gnt-scene-15fps.html'
} catch {
    $State.status='failed';$State.error=$_.Exception.Message;Save-State
    throw
}
