param(
    [ValidateSet('All','Render','Export')][string]$Stage='All',
    [ValidateSet(480,960)][int]$Width=960,
    [string]$TrialRoot=(Join-Path $env:TEMP 'sfm_lightweight_methods_v1'),
    [string]$RenderScratch=(Join-Path $env:TEMP 'sfm_sharpness_trials_v1'),
    [string]$Python='C:\Users\mopu01\AppData\Local\anaconda3\envs\mardm_open3d\python.exe',
    [string]$FFmpeg='C:\ffmpeg\bin\ffmpeg.exe'
)
$ErrorActionPreference='Stop'
$ProjectRoot=Split-Path $PSScriptRoot -Parent
$Destination=Join-Path $env:USERPROFILE "sfm_reconstruction_artifacts/gnt_traversal_control6000_$Width"
if($Stage -in @('All','Render')){
    $Report=Get-Content -Raw (Join-Path $TrialRoot 'gnt_tuning/results.json') | ConvertFrom-Json
    if(-not $Report.traversal_benchmark){throw 'Run the traversal batch benchmark first.'}
    & docker run --rm --pull never --gpus all --name sfm-gnt-traversal -v "${ProjectRoot}:/workspace:ro" -v "${TrialRoot}:/trial" -v "${RenderScratch}:/render_scratch:ro" -w /workspace -e PYTHONPATH=/workspace/src:/trial/python_deps ghcr.io/nerfstudio-project/nerfstudio:1.1.5 python scripts/tune_gnt.py --trial /trial --render-scratch /render_scratch --phase traversal --traversal-width $Width
    if($LASTEXITCODE -ne 0){throw "Traversal rendering failed (exit $LASTEXITCODE). Completed frames are preserved; rerun to resume."}
}
if($Stage -in @('All','Export')){
    & $Python (Join-Path $PSScriptRoot 'publish_gnt_traversal.py') --trial $TrialRoot --destination $Destination --ffmpeg $FFmpeg
    if($LASTEXITCODE -ne 0){throw 'Traversal export or verification failed'}
    $Link=Join-Path $ProjectRoot "web_demo/assets/gnt_traversal_$Width"
    if(Test-Path -LiteralPath $Link){
        $Existing=Get-Item -LiteralPath $Link
        if($Existing.LinkType -ne 'Junction' -or [IO.Path]::GetFullPath($Existing.Target[0]) -ne [IO.Path]::GetFullPath($Destination)){throw 'Existing traversal asset path does not match this export; preserving it.'}
    }else{New-Item -ItemType Junction -Path $Link -Target $Destination | Out-Null}
    $Report=Get-Content -Raw (Join-Path $TrialRoot 'gnt_tuning/results.json') | ConvertFrom-Json
    $Progress=Join-Path $TrialRoot "gnt_tuning/renders/scene/$($Report.traversal.label)/progress.json"
    if(Test-Path -LiteralPath $Progress){
        $State=Get-Content -Raw $Progress | ConvertFrom-Json
        $State.published=$true
        $ProgressJson=$State | ConvertTo-Json
        for($Attempt=0;$Attempt -lt 20;$Attempt++){
            try{
                [IO.File]::WriteAllText($Progress,$ProgressJson,[Text.UTF8Encoding]::new($false))
                break
            }catch [IO.IOException]{
                if($Attempt -eq 19){throw}
                Start-Sleep -Milliseconds 250
            }
        }
    }
    Write-Host "Complete: $ProjectRoot\web_demo\gnt-traversal.html"
    Write-Host "Video: $Destination\scene_control6000.mp4"
}
