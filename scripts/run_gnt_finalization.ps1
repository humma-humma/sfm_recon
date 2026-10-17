param(
    [ValidateSet('All','Runtime','Compile','Objects','Blend','Report')][string]$Stage='All',
    [switch]$FullFlow,
    [string]$TrialRoot=(Join-Path $env:TEMP 'sfm_lightweight_methods_v1'),
    [string]$RenderScratch=(Join-Path $env:TEMP 'sfm_sharpness_trials_v1'),
    [string]$Python='C:\Users\mopu01\AppData\Local\anaconda3\envs\mardm_open3d\python.exe',
    [string]$FFmpeg='C:\ffmpeg\bin\ffmpeg.exe'
)
$ErrorActionPreference='Stop'
$ProjectRoot=Split-Path $PSScriptRoot -Parent
$Destination=Join-Path $env:USERPROFILE 'sfm_reconstruction_artifacts/gnt_finalization'
$Assets=Join-Path $Destination 'assets'
New-Item -ItemType Directory -Force -Path $Assets | Out-Null
$Link=Join-Path $ProjectRoot 'web_demo/assets/gnt_finalization'
if(Test-Path -LiteralPath $Link){
    $Existing=Get-Item -LiteralPath $Link
    if($Existing.LinkType -ne 'Junction' -or [IO.Path]::GetFullPath($Existing.Target[0]) -ne [IO.Path]::GetFullPath($Assets)){throw 'Existing finalization asset path differs; preserving it.'}
}else{New-Item -ItemType Junction -Path $Link -Target $Assets | Out-Null}
$RuntimePath=Join-Path $TrialRoot 'gnt_finalization/runtime.json'
if($Stage -in @('All','Runtime')){
    $Completed=$false
    if(Test-Path -LiteralPath $RuntimePath){$Runtime=Get-Content -Raw $RuntimePath | ConvertFrom-Json;$Completed=$Runtime.status -eq 'complete'}
    if(-not $Completed -or $Stage -eq 'Runtime'){
        & docker run --rm --pull never --gpus all --name sfm-gnt-runtime -v "${ProjectRoot}:/workspace:ro" -v "${TrialRoot}:/trial" -w /workspace -e PYTHONPATH=/workspace/src:/trial/python_deps ghcr.io/nerfstudio-project/nerfstudio:1.1.5 python scripts/benchmark_gnt_runtime.py --trial /trial
        if($LASTEXITCODE -ne 0){throw 'GNT runtime benchmark failed'}
    }
}
if($Stage -in @('All','Compile')){
    $CompilePath=Join-Path $TrialRoot 'gnt_finalization/compile.json'
    if($Stage -eq 'Compile' -or -not (Test-Path -LiteralPath $CompilePath)){
        & docker run --rm --pull never --gpus all --name sfm-gnt-compile -v "${ProjectRoot}:/workspace:ro" -v "${TrialRoot}:/trial" -w /workspace -e PYTHONPATH=/workspace/src:/trial/python_deps ghcr.io/nerfstudio-project/nerfstudio:1.1.5 python scripts/benchmark_gnt_compile.py --trial /trial
        if($LASTEXITCODE -ne 0){throw 'Compiler benchmark failed before recording a result'}
    }
}
if($Stage -in @('All','Objects')){
    $Runtime=Get-Content -Raw $RuntimePath | ConvertFrom-Json
    if($Runtime.status -ne 'complete'){throw 'Finish the runtime benchmark before starting object GPU work.'}
    New-Item -ItemType Directory -Force -Path (Join-Path $ProjectRoot 'outputs/gnt_finalization') | Out-Null
    Copy-Item -LiteralPath $RuntimePath -Destination (Join-Path $ProjectRoot 'outputs/gnt_finalization/runtime.json')
    if(Test-Path (Join-Path $TrialRoot 'gnt_finalization/compile.json')){Copy-Item -LiteralPath (Join-Path $TrialRoot 'gnt_finalization/compile.json') -Destination (Join-Path $ProjectRoot 'outputs/gnt_finalization/compile.json')}
    foreach($Dataset in @('box','boot','milk')){
        & docker run --rm --pull never --gpus all --name sfm-gnt-objects -v "${ProjectRoot}:/workspace:ro" -v "${TrialRoot}:/trial" -v "${RenderScratch}:/render_scratch:ro" -w /workspace -e PYTHONPATH=/workspace/src:/trial/python_deps ghcr.io/nerfstudio-project/nerfstudio:1.1.5 python scripts/adapt_gnt_objects.py --trial /trial --render-scratch /render_scratch --dataset $Dataset
        if($LASTEXITCODE -ne 0){throw "GNT object phase failed: $Dataset. Saved checkpoints and frames remain available for resuming."}
        $env:PYTHONPATH=Join-Path $ProjectRoot 'src'
        & $Python (Join-Path $PSScriptRoot 'publish_gnt_objects.py') --trial $TrialRoot --destination $Destination
        if($LASTEXITCODE -ne 0){throw 'Object gallery publication failed'}
    }
    & $Python (Join-Path $PSScriptRoot 'verify_gnt_objects.py') --destination $Destination
    if($LASTEXITCODE -ne 0){throw 'Object archive or gallery verification failed'}
    & $Python (Join-Path $PSScriptRoot 'verify_gnt_objects_browser.py')
    if($LASTEXITCODE -ne 0){throw 'Object browser verification failed'}
}
if($Stage -in @('All','Blend')){
    $BlendExtra=@();if($FullFlow){$BlendExtra=@('--full-flow')}
    & $Python (Join-Path $PSScriptRoot 'test_gnt_blending.py') --destination $Destination --ffmpeg $FFmpeg @BlendExtra
    if($LASTEXITCODE -ne 0){throw 'GNT post-render blending tests failed'}
}
if($Stage -eq 'Report'){
    $env:PYTHONPATH=Join-Path $ProjectRoot 'src'
    & $Python (Join-Path $PSScriptRoot 'publish_gnt_objects.py') --trial $TrialRoot --destination $Destination
    if($LASTEXITCODE -ne 0){throw 'Object gallery publication failed'}
}
Write-Host "Stage complete: $Stage"
Write-Host 'Object review: http://localhost:8088/gnt-objects.html'
Write-Host 'Motion review: http://localhost:8088/gnt-motion.html'
