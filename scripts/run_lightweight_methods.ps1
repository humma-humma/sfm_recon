param(
    [ValidateSet('Setup','GNT','2DGS','Report','All')][string]$Stage='All',
    [string[]]$Datasets=@('box','boot','milk','scene'),
    [string]$TrialRoot=(Join-Path $env:TEMP 'sfm_lightweight_methods_v1'),
    [string]$RenderScratch=(Join-Path $env:TEMP 'sfm_sharpness_trials_v1'),
    [string]$Python='C:\Users\mopu01\AppData\Local\anaconda3\envs\mardm_open3d\python.exe'
)
$ErrorActionPreference='Stop'
$ProjectRoot=Split-Path $PSScriptRoot -Parent
$Image='ghcr.io/nerfstudio-project/nerfstudio:1.1.5'
foreach($Dataset in $Datasets){if($Dataset -notin @('box','boot','milk','scene')){throw "Unknown dataset: $Dataset"}}
New-Item -ItemType Directory -Force -Path $TrialRoot | Out-Null
function Check-Exit([string]$Task){if($LASTEXITCODE -ne 0){throw "$Task failed (exit $LASTEXITCODE)"}}
function Invoke-Trial([string[]]$Command){
    & docker run --rm --pull never --gpus all -v "${ProjectRoot}:/workspace:ro" -v "${TrialRoot}:/trial" -v "${RenderScratch}:/render_scratch:ro" -w /workspace -e PYTHONPATH=/workspace/src:/trial/python_deps $Image @Command
    Check-Exit 'Research trial'
}
if($Stage -in @('Setup','All')){
    $Repos=@(
        @{Folder='GNT';Url='https://github.com/VITA-Group/GNT';Commit='53e0fc3d39b4244503d6c4de1e34bbf3199dd2a8'},
        @{Folder='2dgs';Url='https://github.com/hbb1/2d-gaussian-splatting';Commit='f3e3b9fa67bbd1c75e05167ff37391d8dab2a678'}
    )
    foreach($Repo in $Repos){
        $Destination=Join-Path $TrialRoot $Repo.Folder
        if(!(Test-Path $Destination)){
            & git clone --depth 1 --no-checkout $Repo.Url $Destination
            Check-Exit 'Clone official repository'
            & git -C $Destination fetch --depth 1 origin $Repo.Commit
            Check-Exit 'Fetch pinned revision'
            & git -C $Destination checkout $Repo.Commit
            Check-Exit 'Pin official repository'
            & git -C $Destination submodule update --init --recursive --depth 1
            Check-Exit 'Pin submodules'
        }
        $Actual=(& git -C $Destination rev-parse HEAD).Trim()
        if($Actual -ne $Repo.Commit){throw "Unexpected commit in $Destination; preserve it and select a fresh TrialRoot"}
    }
    $Weights=Join-Path $TrialRoot 'GNT/generalization_200000.pth'
    if(!(Test-Path $Weights)){
        Invoke-WebRequest 'https://drive.usercontent.google.com/download?id=1YvOJXa5eGpKgoMYcxC2ma7prB1n5UwRn&export=download&confirm=t' -OutFile $Weights
    }
    if((Get-Item $Weights).Length -ne 117498781){throw 'Official GNT download has an unexpected size'}
    if((Get-FileHash $Weights -Algorithm SHA256).Hash -ne '3B06CCAD304B0FD70457D9601917E94BF950FEAFC3B7BEEDA230DB4BC579F2EE'){throw 'Official GNT checkpoint SHA256 mismatch'}
    $Packages=Join-Path $TrialRoot 'cuda_packages'
    New-Item -ItemType Directory -Force -Path $Packages | Out-Null
    foreach($Package in @('cuda-nvcc','cuda-cudart-dev','cuda-cccl','cuda-cudart')){
        $File="$Package-11.8.89-0.tar.bz2"
        $Destination=Join-Path $Packages $File
        if(!(Test-Path $Destination)){Invoke-WebRequest "https://api.anaconda.org/download/nvidia/$Package/11.8.89/linux-64/$File" -OutFile $Destination}
    }
    # Compiler exists only inside this disposable container. Built extensions stay on C:.
    $Build=@'
set -eu
mkdir -p /tmp/cuda
for package in /trial/cuda_packages/*.tar.bz2; do tar -xjf "$package" -C /tmp/cuda; done
export CUDA_HOME=/tmp/cuda
export PATH=/tmp/cuda/bin:$PATH
export PYTHONPATH=/trial/python_deps
export TORCH_CUDA_ARCH_LIST=7.5
export MAX_JOBS=2
python -m pip install --target /trial/python_deps wheel==0.48.0 ninja==1.13.2
python -m pip install --no-build-isolation --no-deps --target /trial/python_deps /trial/2dgs/submodules/diff-surfel-rasterization /trial/2dgs/submodules/simple-knn
'@
    Invoke-Trial -Command @('bash','-c',$Build)
}
if($Stage -in @('GNT','2DGS','All') -and !(Test-Path (Join-Path $RenderScratch 'render_manifest.json'))){throw 'Prior fixed-camera render manifest is required'}
if($Stage -in @('GNT','All')){
    foreach($Dataset in $Datasets){
        Invoke-Trial -Command @('python','scripts/run_gnt_trial.py','--trial','/trial','--render-scratch','/render_scratch','--dataset',$Dataset,'--case-group','diagnostic')
        Invoke-Trial -Command @('python','scripts/run_gnt_trial.py','--trial','/trial','--render-scratch','/render_scratch','--dataset',$Dataset,'--case-group','motion','--width','480')
    }
}
if($Stage -in @('2DGS','All')){
    foreach($Dataset in $Datasets){
        Invoke-Trial -Command @('python','scripts/run_2dgs_trial.py','--trial','/trial','--render-scratch','/render_scratch','--dataset',$Dataset)
        if($Dataset -in @('boot','milk')){
            Invoke-Trial -Command @('python','scripts/run_2dgs_trial.py','--trial','/trial','--render-scratch','/render_scratch','--dataset',$Dataset,'--lambda-dist','0')
        }
    }
}
if($Stage -in @('Report','All')){
    $env:TORCH_HOME=Join-Path $ProjectRoot 'outputs/nerfstudio_cache/torch'
    & $Python (Join-Path $PSScriptRoot 'evaluate_lightweight_trials.py') --trial $TrialRoot --render-scratch $RenderScratch
    Check-Exit 'Comparison report'
    Write-Host "Complete: $ProjectRoot\web_demo\lightweight.html"
}
