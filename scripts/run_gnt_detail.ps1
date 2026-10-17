param(
    [ValidateSet('All','Control','Patch','Render','Report')][string]$Stage='All',
    [string]$TrialRoot=(Join-Path $env:TEMP 'sfm_lightweight_methods_v1'),
    [string]$RenderScratch=(Join-Path $env:TEMP 'sfm_sharpness_trials_v1'),
    [string]$Python='python'
)
$ErrorActionPreference='Stop'
$ProjectRoot=Split-Path $PSScriptRoot -Parent
foreach($Phase in @('Control','Patch','Render')){
    if($Stage -notin @('All',$Phase)){continue}
    $Extra=@()
    if($Phase -ne 'Render'){
        $Arm=$Phase.ToLowerInvariant()
        if(Test-Path (Join-Path $TrialRoot "gnt_tuning/detail_${Arm}_006000.pth")){
            $Report=Get-Content -Raw (Join-Path $TrialRoot 'gnt_tuning/results.json') | ConvertFrom-Json
            $Rows=@($Report.runs | Where-Object {$_.label -eq "detail_${Arm}6000" -and $_.kind -eq 'validation' -and $_.width -eq 480})
            if($Rows.Count -ne 6){throw "Existing $Arm weights have incomplete validation; preserve them and recover validation before continuing."}
            Write-Host "Preserving completed $Arm arm"
            continue
        }
        $CommandPhase='detail'
        $Extra=@('--detail-arm',$Arm)
    }else{$CommandPhase='detail-final'}
    & docker run --rm --pull never --gpus all --name sfm-gnt-detail -v "${ProjectRoot}:/workspace:ro" -v "${TrialRoot}:/trial" -v "${RenderScratch}:/render_scratch:ro" -w /workspace -e PYTHONPATH=/workspace/src:/trial/python_deps ghcr.io/nerfstudio-project/nerfstudio@sha256:b59b8e1012d7a43679d3234b3de9c8416a4b8435fcbf21b9d8c4494b8563f19e python scripts/tune_gnt.py --trial /trial --render-scratch /render_scratch --phase $CommandPhase @Extra
    if($LASTEXITCODE -ne 0){throw "GNT $Phase failed (exit $LASTEXITCODE)"}
}
if($Stage -in @('All','Report')){
    $env:PYTHONPATH=Join-Path $ProjectRoot 'src'
    & $Python (Join-Path $PSScriptRoot 'publish_gnt_balanced.py') --trial $TrialRoot --experiment detail
    if($LASTEXITCODE -ne 0){throw 'GNT detail report failed'}
    Write-Host "Complete: $ProjectRoot\web_demo\gnt-detail.html"
}
