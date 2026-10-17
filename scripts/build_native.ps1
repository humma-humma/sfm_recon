param(
    [string]$Python = "python",
    [string]$CMake = "cmake",
    [string]$Generator = "Visual Studio 17 2022",
    [string]$CeresPrefix = ""
)
$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$BuildDir = Join-Path $RepoRoot "outputs\native_build"
$InstallDir = Join-Path $RepoRoot "src\sfm_reconstruction"
$PythonPath = (Get-Command $Python -ErrorAction Stop).Source
$CeresArgs = @("-DSFM_WITH_CERES=OFF")
if ($CeresPrefix) {
    $ResolvedCeres = (Resolve-Path -LiteralPath $CeresPrefix).Path
    $CeresArgs = @("-DSFM_WITH_CERES=ON", "-DCMAKE_PREFIX_PATH=$ResolvedCeres/Library")
}
& $CMake -S (Join-Path $RepoRoot "native") -B $BuildDir -G $Generator -A x64 "-DPython_EXECUTABLE=$PythonPath" @CeresArgs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
# File tracking can fail in restricted Windows sessions; compilation does not need it.
& $CMake --build $BuildDir --config Release -- /p:TrackFileAccess=false
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $CMake --install $BuildDir --config Release --prefix $InstallDir
exit $LASTEXITCODE
