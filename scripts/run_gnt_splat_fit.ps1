param(
    [string]$Destination=(Join-Path $env:USERPROFILE 'sfm_reconstruction_artifacts/gnt_finalization'),
    [string]$Original=(Join-Path $env:USERPROFILE 'sfm_reconstruction_artifacts/gnt_traversal_control6000_960'),
    [string]$Python='python',
    [string]$FFmpeg='ffmpeg'
)
$ErrorActionPreference='Stop'
# Windows PowerShell treats native stderr warnings as errors under Stop.
# Native process failures are checked explicitly with LASTEXITCODE below.
function Invoke-DockerLog {
    param([string[]]$Arguments,[string]$Log)
    $ErrorActionPreference='Continue'
    & docker @Arguments *> $Log
    if($LASTEXITCODE -ne 0){throw "Docker exited $LASTEXITCODE. Check $Log"}
}
$ProjectRoot=Split-Path $PSScriptRoot -Parent
$env:PYTHONPATH=Join-Path $ProjectRoot 'src'
if(-not (Test-Path (Join-Path $Destination 'gnt_splat_scene/data_manifest.json'))){
    & $Python (Join-Path $PSScriptRoot 'prepare_gnt_splat.py') --destination $Destination --original $Original
    if($LASTEXITCODE -ne 0){throw 'Teacher dataset preparation failed'}
}
$Running=& docker ps --filter 'name=^sfm-gnt-splat-fit$' --format '{{.Names}}'
if($Running){
    $Code=& docker wait sfm-gnt-splat-fit
    if($LASTEXITCODE -ne 0 -or $Code -ne '0'){throw 'The initial mixed fit did not complete successfully; inspect its log before resuming.'}
}
$DockerArgs=@('run','--rm','--pull','never','--gpus','all','--name','sfm-gnt-splat-fit','-v',"${ProjectRoot}:/workspace:ro",'-v',"${Destination}:/finalization",'-v',"${Original}:/original:ro",'-w','/workspace','-e','PYTHONPATH=/workspace/src','-e','TORCH_HOME=/workspace/outputs/nerfstudio_cache/torch','ghcr.io/nerfstudio-project/nerfstudio@sha256:b59b8e1012d7a43679d3234b3de9c8416a4b8435fcbf21b9d8c4494b8563f19e')
$Logs=Join-Path $Destination 'gnt_splat_scene/logs';New-Item -ItemType Directory -Force -Path $Logs | Out-Null
foreach($Arm in @('mixed','photos')){
    Invoke-DockerLog -Arguments ($DockerArgs+@('python','scripts/train_gnt_splat.py','--root','/finalization','--arm',$Arm)) -Log (Join-Path $Logs "train_$Arm.log")
    if($LASTEXITCODE -ne 0){throw "Splat fitting failed: $Arm. Check $Logs"}
    $Labels=@("${Arm}_10000");if($Arm -eq 'mixed'){$Labels=@('baseline','mixed_5000','mixed_10000')}
    foreach($Label in $Labels){
        $Extra=@();if($Label -eq 'mixed_10000'){$Extra=@('--full-route')}
        Invoke-DockerLog -Arguments ($DockerArgs+@('python','scripts/evaluate_gnt_splat.py','--root','/finalization','--dense','/finalization/assets/dense_turn','--original','/original','--label',$Label)+$Extra) -Log (Join-Path $Logs "evaluate_$Label.log")
        if($LASTEXITCODE -ne 0){throw "Splat evaluation failed: $Label. Check $Logs"}
        & $Python (Join-Path $PSScriptRoot 'publish_gnt_splat.py') --root $Destination --ffmpeg $FFmpeg
        if($LASTEXITCODE -ne 0){throw 'Splat comparison publication failed'}
    }
}
Invoke-DockerLog -Arguments ($DockerArgs+@('ns-export','gaussian-splat','--load-config','/finalization/gnt_splat_scene/models/mixed/splatfacto/run_10000/config.yml','--output-dir','/finalization/assets/gnt_splat_scene/ply')) -Log (Join-Path $Logs 'export_ply.log')
if($LASTEXITCODE -ne 0){throw 'Splat PLY export failed'}
Write-Host 'Complete: http://localhost:8088/gnt-splat-fit.html'
