# ============================================================
# Step 1: Install TensorRT via pip + extract C++ SDK to C:\TensorRT
# Run from an Administrator PowerShell:
#   cd C:\Users\nengj\OneDrive\Desktop\VSLAM\setup
#   .\01_install_tensorrt.ps1
# ============================================================

$ErrorActionPreference = "Stop"
$trtRoot = "C:\TensorRT"

Write-Host "=== Installing TensorRT Python package ===" -ForegroundColor Cyan
pip install tensorrt==10.3.0 --extra-index-url https://pypi.nvidia.com

if ($LASTEXITCODE -ne 0) {
    Write-Host "Retrying with plain pip index..." -ForegroundColor Yellow
    pip install tensorrt==10.3.0
}

# Locate the installed tensorrt package directory
$trtPyDir = python -c "import tensorrt, os; print(os.path.dirname(tensorrt.__file__))"
Write-Host "TensorRT Python package at: $trtPyDir" -ForegroundColor Green

# Extract C++ headers and libs into C:\TensorRT
Write-Host "=== Extracting C++ SDK to $trtRoot ===" -ForegroundColor Cyan
New-Item -ItemType Directory -Force -Path "$trtRoot\include" | Out-Null
New-Item -ItemType Directory -Force -Path "$trtRoot\lib"     | Out-Null
New-Item -ItemType Directory -Force -Path "$trtRoot\bin"     | Out-Null

# Copy headers
$headerSrc = "$trtPyDir\include"
if (Test-Path $headerSrc) {
    Copy-Item "$headerSrc\*" "$trtRoot\include\" -Recurse -Force
    Write-Host "  Copied headers from $headerSrc" -ForegroundColor Green
} else {
    # Headers may be inside a nested 'tensorrt' subdirectory
    $altSrc = Split-Path $trtPyDir -Parent
    Get-ChildItem "$altSrc" -Recurse -Filter "NvInfer.h" | ForEach-Object {
        $hDir = $_.DirectoryName
        Copy-Item "$hDir\*.h" "$trtRoot\include\" -Force
        Write-Host "  Copied headers from $hDir" -ForegroundColor Green
    }
}

# Copy libs (.lib files for linking)
Get-ChildItem "$trtPyDir" -Recurse -Filter "*.lib" | ForEach-Object {
    Copy-Item $_.FullName "$trtRoot\lib\" -Force
}
Get-ChildItem "$trtPyDir" -Recurse -Filter "nvinfer*.dll" | ForEach-Object {
    Copy-Item $_.FullName "$trtRoot\lib\" -Force
    Copy-Item $_.FullName "$trtRoot\bin\" -Force
}

# Copy trtexec if present
$trtexec = Get-ChildItem "$trtPyDir" -Recurse -Filter "trtexec.exe" | Select-Object -First 1
if ($trtexec) {
    Copy-Item $trtexec.FullName "$trtRoot\bin\trtexec.exe" -Force
    Write-Host "  trtexec.exe found and copied" -ForegroundColor Green
} else {
    Write-Host "  trtexec.exe not in pip package — will use Python trt API for engine export" -ForegroundColor Yellow
}

# Add TRT bin to PATH for this session
$env:PATH = "$trtRoot\bin;$env:PATH"

Write-Host ""
Write-Host "=== TensorRT setup complete ===" -ForegroundColor Green
Write-Host "  Include: $trtRoot\include"
Write-Host "  Lib:     $trtRoot\lib"
Write-Host "  Bin:     $trtRoot\bin"
Write-Host ""
Write-Host "Verify with: python -c `"import tensorrt; print(tensorrt.__version__)`""
python -c "import tensorrt; print('TensorRT version:', tensorrt.__version__)"
