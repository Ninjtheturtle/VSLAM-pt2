# ============================================================
# Master installer — run this ONE script to do everything.
# Open PowerShell as Administrator, then:
#
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
#   cd C:\Users\nengj\OneDrive\Desktop\VSLAM
#   .\setup\00_run_all.ps1
#
# What it does:
#   1. Installs TensorRT 10.3 via pip → extracts C++ SDK to C:\TensorRT
#   2. Downloads libtorch CUDA 12.4 (~2.4 GB) → C:\libtorch
#   3. Exports XFeat + LightGlue to ONNX then TRT FP16 engines
#   4. Configures + builds vslam.exe with ENABLE_DEEP_FRONTEND=ON
#
# Total download: ~2.6 GB  |  Disk usage: ~5 GB
# ============================================================

$ErrorActionPreference = "Stop"
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$root      = Split-Path -Parent $scriptDir

Set-Location $root
Write-Host "Working directory: $root" -ForegroundColor Cyan

# ── Step 1: TensorRT ─────────────────────────────────────────────────────────
Write-Host ""
Write-Host "━━━ STEP 1/4: TensorRT ━━━" -ForegroundColor Yellow
& "$scriptDir\01_install_tensorrt.ps1"

# ── Step 2: libtorch ─────────────────────────────────────────────────────────
Write-Host ""
Write-Host "━━━ STEP 2/4: libtorch ━━━" -ForegroundColor Yellow
& "$scriptDir\02_download_libtorch.ps1"

# ── Step 3: Export models ─────────────────────────────────────────────────────
Write-Host ""
Write-Host "━━━ STEP 3/4: Export XFeat + LightGlue engines ━━━" -ForegroundColor Yellow
python "$scriptDir\03_export_models.py"
if ($LASTEXITCODE -ne 0) {
    Write-Host "Model export failed — check error above." -ForegroundColor Red
    exit 1
}

# ── Step 4: Build ─────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "━━━ STEP 4/4: Build vslam.exe (hybrid) ━━━" -ForegroundColor Yellow
& "$scriptDir\04_build_hybrid.ps1"

Write-Host ""
Write-Host "╔══════════════════════════════════════════════════╗" -ForegroundColor Green
Write-Host "║  ALL DONE — ready to run!                        ║" -ForegroundColor Green
Write-Host "╠══════════════════════════════════════════════════╣" -ForegroundColor Green
Write-Host "║  cd $root" -ForegroundColor Green
Write-Host "║  build_hybrid\Release\vslam.exe \`" -ForegroundColor Green
Write-Host "║    --sequence data/dataset/sequences/00 \`" -ForegroundColor Green
Write-Host "║    --hybrid \`" -ForegroundColor Green
Write-Host "║    --xfeat models/xfeat_fp16.engine \`" -ForegroundColor Green
Write-Host "║    --lg    models/lighterglue_fp16.engine" -ForegroundColor Green
Write-Host "╚══════════════════════════════════════════════════╝" -ForegroundColor Green
