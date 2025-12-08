$ErrorActionPreference = "Stop"

Set-Location (Join-Path $PSScriptRoot "..")

$buildDir = Join-Path (Get-Location) "build"
if (-not (Test-Path $buildDir)) {
    New-Item -ItemType Directory -Path $buildDir | Out-Null
}
Set-Location $buildDir

$cCompiler = "C:/Program Files/LLVM/bin/clang.exe"
$cxxCompiler = "C:/Program Files/LLVM/bin/clang++.exe"
$rcCompiler = "C:/Program Files/LLVM/bin/llvm-rc.exe"

if (-not (Test-Path $cCompiler) -or -not (Test-Path $cxxCompiler) -or -not (Test-Path $rcCompiler)) {
    Write-Host "clang/clang++/llvm-rc not found under C:/Program Files/LLVM/bin. Adjust scripts/dev.ps1 to your LLVM install path."
    exit 1
}

Write-Host "Configuring CMake..."
cmake -G "Ninja" -DCMAKE_C_COMPILER="$cCompiler" -DCMAKE_CXX_COMPILER="$cxxCompiler" -DCMAKE_RC_COMPILER="$rcCompiler" ..

Write-Host "Building..."
cmake --build .

$exe = Join-Path $buildDir "bin\voxel_engine.exe"
if (Test-Path $exe) {
    Write-Host "Running voxel_engine.exe..."
    Push-Location (Split-Path $exe)
    .\voxel_engine.exe
    Pop-Location
} else {
    Write-Host "Build completed, but executable not found at $exe"
}


