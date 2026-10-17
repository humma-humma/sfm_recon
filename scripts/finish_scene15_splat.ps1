param(
    [string]$Destination=(Join-Path $env:USERPROFILE 'sfm_reconstruction_artifacts/gnt_finalization'),
    [string]$Python='C:\Users\mopu01\AppData\Local\anaconda3\envs\mardm_open3d\python.exe'
)
$ErrorActionPreference='Stop'
$Output=Join-Path $Destination 'assets/scene15_splat'
$Job=Join-Path $Output 'job.json'
$State=@{status='running';phase='waiting_for_splat_render';pid=$PID;started_utc=[DateTime]::UtcNow.ToString('o')}
function Save-State {
    $State.updated_utc=[DateTime]::UtcNow.ToString('o')
    [IO.File]::WriteAllText("${Job}.tmp",($State | ConvertTo-Json),[Text.UTF8Encoding]::new($false))
    Move-Item -LiteralPath "${Job}.tmp" -Destination $Job -Force
}
try {
    Save-State
    $Deadline=[DateTime]::UtcNow.AddMinutes(30)
    do {
        $ProgressPath=Join-Path $Output 'render_progress.json'
        $Ready=$false
        if(Test-Path -LiteralPath $ProgressPath){$Progress=Get-Content -LiteralPath $ProgressPath -Raw | ConvertFrom-Json;$Ready=$Progress.status -eq 'complete'}
        if(-not $Ready){if([DateTime]::UtcNow -gt $Deadline){throw 'Timed out waiting for splat renderer'};Start-Sleep -Seconds 5}
    } while(-not $Ready)
    $State.phase='correction_encoding_verification';Save-State
    & $Python (Join-Path $PSScriptRoot 'publish_scene15_splat.py') --root $Destination
    if($LASTEXITCODE -ne 0){throw 'Full-scene correction/export failed'}
    $State.phase='browser_verification';Save-State
    & $Python (Join-Path $PSScriptRoot 'verify_gnt_splat_correction_browser.py') --scene
    if($LASTEXITCODE -ne 0){throw 'Full-scene comparison browser verification failed'}
    $State.status='complete';$State.phase='complete';Save-State
} catch {
    $State.status='failed';$State.error=$_.Exception.Message;Save-State
    throw
}
