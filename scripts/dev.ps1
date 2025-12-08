$ErrorActionPreference = "Stop"

Set-Location (Join-Path $PSScriptRoot "..")

$buildDir = Join-Path (Get-Location) "build"
if (-not (Test-Path $buildDir)) {
    New-Item -ItemType Directory -Path $buildDir | Out-Null
}
Set-Location $buildDir

Write-Host "Configuring CMake..."
cmake -G "Ninja" -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ ..

Write-Host "Building..."
cmake --build .

$exe = Join-Path $buildDir "bin\voxel_engine.exe"
if (Test-Path $exe) {
    Write-Host "Running voxel_engine.exe..."
    & $exe
} else {
    Write-Host "Build completed, but executable not found at $exe"
}


