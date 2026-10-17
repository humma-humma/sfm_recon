param(
    [ValidateSet('All','Sweep','Adapt','Refine','Final','Report')][string]$Stage='All',
    [int]$Steps=600,
    [int]$RefineSteps=1800,
    [string]$TrialRoot=(Join-Path $env:TEMP 'sfm_lightweight_methods_v1'),
    [string]$RenderScratch=(Join-Path $env:TEMP 'sfm_sharpness_trials_v1'),
    [string]$Python='python'
)
$ErrorActionPreference='Stop'
$ProjectRoot=Split-Path $PSScriptRoot -Parent
foreach($Phase in @('Sweep','Adapt','Refine','Final')){
    if($Stage -notin @('All',$Phase)){continue}
    $Extra=@()
    $PhaseSteps=$Steps
    $CommandPhase=$Phase.ToLower()
    if($Phase -eq 'Refine'){$CommandPhase='adapt';$PhaseSteps=$RefineSteps;$Extra=@('--resume-step',"$Steps",'--train-width','960')}
    & docker run --rm --pull never --gpus all --name sfm-gnt-tuning -v "${ProjectRoot}:/workspace:ro" -v "${TrialRoot}:/trial" -v "${RenderScratch}:/render_scratch:ro" -w /workspace -e PYTHONPATH=/workspace/src:/trial/python_deps ghcr.io/nerfstudio-project/nerfstudio@sha256:b59b8e1012d7a43679d3234b3de9c8416a4b8435fcbf21b9d8c4494b8563f19e python scripts/tune_gnt.py --trial /trial --render-scratch /render_scratch --phase $CommandPhase --steps $PhaseSteps @Extra
    if($LASTEXITCODE -ne 0){throw "GNT $Phase failed (exit $LASTEXITCODE)"}
}
if($Stage -in @('All','Report')){
    & $Python (Join-Path $PSScriptRoot 'publish_gnt_tuning.py') --trial $TrialRoot
    if($LASTEXITCODE -ne 0){throw 'GNT report failed'}
    Write-Host "Complete: $ProjectRoot\web_demo\gnt-tuning.html"
}
