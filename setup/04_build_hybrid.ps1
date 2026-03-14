# ============================================================
# Step 4: Configure and build VSLAM with ENABLE_DEEP_FRONTEND=ON
# Run from PowerShell after steps 1-3 complete:
#   cd C:\Users\nengj\OneDrive\Desktop\VSLAM
#   .\setup\04_build_hybrid.ps1
# ============================================================

$ErrorActionPreference = "Stop"
$root     = "C:\Users\nengj\OneDrive\Desktop\VSLAM"
$buildDir = "$root\build_hybrid"
$torchDir = "C:\Users\nengj\AppData\Local\Programs\Python\Python310\lib\site-packages\torch\share\cmake\Torch"

# Verify prerequisites
if (-not (Test-Path $torchDir)) {
    Write-Host "ERROR: libtorch not found at $torchDir" -ForegroundColor Red
    Write-Host "Python torch cmake dir not found." -ForegroundColor Red
    exit 1
}

Write-Host "=== Configuring CMake (hybrid mode) ===" -ForegroundColor Cyan
cmake -B $buildDir `
      -DENABLE_DEEP_FRONTEND=ON `
      "-DTorch_DIR=$torchDir" `
      -DCMAKE_BUILD_TYPE=Release `
      $root

if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host ""
Write-Host "=== Building (Release, all cores) ===" -ForegroundColor Cyan
cmake --build $buildDir --config Release --parallel

if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host ""
Write-Host "=== Build complete ===" -ForegroundColor Green
$exe = "$buildDir\Release\vslam.exe"
Write-Host "  Executable: $exe"
Write-Host ""
Write-Host "Run with:"
Write-Host "  cd $root"
Write-Host "  $exe --sequence data/dataset/sequences/00 \"
Write-Host "       --hybrid \"
Write-Host "       --xfeat models/xfeat.pt \"
Write-Host "       --lg    models/lighterglue.pt"
