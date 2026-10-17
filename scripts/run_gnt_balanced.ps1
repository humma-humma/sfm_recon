param(
    [ValidateSet('All','Train','Render','Report')][string]$Stage='All',
    [string]$TrialRoot=(Join-Path $env:TEMP 'sfm_lightweight_methods_v1'),
    [string]$RenderScratch=(Join-Path $env:TEMP 'sfm_sharpness_trials_v1'),
    [string]$Python='C:\Users\mopu01\AppData\Local\anaconda3\envs\mardm_open3d\python.exe'
)
$ErrorActionPreference='Stop'
$ProjectRoot=Split-Path $PSScriptRoot -Parent
foreach($Phase in @('Train','Render')){
    if($Stage -notin @('All',$Phase)){continue}
    if($Phase -eq 'Train'){
        if(Test-Path (Join-Path $TrialRoot 'gnt_tuning/balanced_003200.pth')){throw 'Balanced checkpoint already exists. Use -Stage Render or Report; training will not overwrite it.'}
        $CommandPhase='balanced'
    }else{$CommandPhase='balanced-final'}
    & docker run --rm --pull never --gpus all --name sfm-gnt-balanced -v "${ProjectRoot}:/workspace:ro" -v "${TrialRoot}:/trial" -v "${RenderScratch}:/render_scratch:ro" -w /workspace -e PYTHONPATH=/workspace/src:/trial/python_deps ghcr.io/nerfstudio-project/nerfstudio:1.1.5 python scripts/tune_gnt.py --trial /trial --render-scratch /render_scratch --phase $CommandPhase --steps 5000
    if($LASTEXITCODE -ne 0){throw "GNT $Phase failed (exit $LASTEXITCODE)"}
}
if($Stage -in @('All','Report')){
    $env:PYTHONPATH=Join-Path $ProjectRoot 'src'
    & $Python (Join-Path $PSScriptRoot 'publish_gnt_balanced.py') --trial $TrialRoot
    if($LASTEXITCODE -ne 0){throw 'Balanced GNT report failed'}
    Write-Host "Complete: $ProjectRoot\web_demo\gnt-balanced.html"
}
