param(
    [string]$Destination=(Join-Path $env:USERPROFILE 'sfm_reconstruction_artifacts/gnt_finalization'),
    [string]$Original=(Join-Path $env:USERPROFILE 'sfm_reconstruction_artifacts/gnt_traversal_control6000_960'),
    [string]$Python='python',
    [string]$FFmpeg='ffmpeg'
)
$ErrorActionPreference='Stop'
$ProjectRoot=Split-Path $PSScriptRoot -Parent
$Logs=Join-Path $Destination 'gnt_splat_scene/logs'
function Invoke-NoiseDocker {
    param([string[]]$TaskArgs,[string]$Log)
    $ErrorActionPreference='Continue'
    & docker @DockerArgs @TaskArgs *> $Log
    if($LASTEXITCODE -ne 0){throw "Docker exited $LASTEXITCODE. Check $Log"}
}
$DockerArgs=@('run','--rm','--pull','never','--gpus','all','--name','sfm-gnt-splat-noise','-v',"${ProjectRoot}:/workspace:ro",'-v',"${Destination}:/finalization",'-v',"${Original}:/original:ro",'-w','/workspace','-e','PYTHONPATH=/workspace/src','-e','TORCH_HOME=/workspace/outputs/nerfstudio_cache/torch','ghcr.io/nerfstudio-project/nerfstudio@sha256:b59b8e1012d7a43679d3234b3de9c8416a4b8435fcbf21b9d8c4494b8563f19e')
$Running=& docker ps --filter 'name=^sfm-gnt-splat-fit$' --format '{{.Names}}'
if($Running){
    $Code=& docker wait sfm-gnt-splat-fit
    if($LASTEXITCODE -ne 0 -or $Code -ne '0'){throw 'Existing comparison export failed; inspect its log'}
}
if(-not (Test-Path (Join-Path $Destination 'gnt_splat_scene/data/teacher/train_list.txt'))){throw 'Prepare the teacher dataset before running'}
Invoke-NoiseDocker -TaskArgs @('python','scripts/train_gnt_splat.py','--root','/finalization','--arm','teacher') -Log (Join-Path $Logs 'train_teacher.log')
function Evaluate-Noise {
    param([string]$Label)
    Invoke-NoiseDocker -TaskArgs @('python','scripts/evaluate_gnt_splat.py','--root','/finalization','--dense','/finalization/assets/dense_turn','--original','/original','--label',$Label) -Log (Join-Path $Logs "evaluate_$Label.log")
    & $Python (Join-Path $PSScriptRoot 'publish_gnt_splat.py') --root $Destination --ffmpeg $FFmpeg
    if($LASTEXITCODE -ne 0){throw 'Comparison publication failed'}
}
Evaluate-Noise 'teacher_10000'
Invoke-NoiseDocker -TaskArgs @('python','scripts/refine_gnt_splat_geometry.py','--root','/finalization') -Log (Join-Path $Logs 'refine_geometry.log')
Evaluate-Noise 'geometry_12000'
Evaluate-Noise 'pruned_12000'
Write-Host 'Noise trials complete: http://localhost:8088/gnt-splat-fit.html'
