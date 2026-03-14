# ============================================================
# Step 2: Download libtorch C++ (CUDA 12.4 Release) to C:\libtorch
# Run from PowerShell (no admin required):
#   .\02_download_libtorch.ps1
# ============================================================

$ErrorActionPreference = "Stop"
$dest      = "C:\libtorch"
$zipPath   = "$env:TEMP\libtorch_cu124.zip"
# PyTorch 2.5.1 with CUDA 12.4 — Windows shared-with-deps (includes all DLLs)
$url = "https://download.pytorch.org/libtorch/cu124/libtorch-win-shared-with-deps-2.5.1%2Bcu124.zip"

if (Test-Path "$dest\share\cmake\Torch\TorchConfig.cmake") {
    Write-Host "libtorch already installed at $dest — skipping download." -ForegroundColor Green
    exit 0
}

Write-Host "=== Downloading libtorch (~2.4 GB) ===" -ForegroundColor Cyan
Write-Host "URL: $url"
Write-Host "This will take a few minutes..."

# Use BITS for reliable large file download with progress
try {
    Import-Module BitsTransfer
    Start-BitsTransfer -Source $url -Destination $zipPath -DisplayName "libtorch download"
} catch {
    Write-Host "BITS failed, falling back to WebClient..." -ForegroundColor Yellow
    $wc = New-Object System.Net.WebClient
    $wc.DownloadFile($url, $zipPath)
}

Write-Host "Download complete. Extracting to $dest ..." -ForegroundColor Cyan
New-Item -ItemType Directory -Force -Path $dest | Out-Null

# Extract (Expand-Archive is slow for 2GB; use .NET directly)
Add-Type -AssemblyName System.IO.Compression.FileSystem
[System.IO.Compression.ZipFile]::ExtractToDirectory($zipPath, "C:\")
# The zip extracts to C:\libtorch\ directly

# Verify
if (Test-Path "$dest\share\cmake\Torch\TorchConfig.cmake") {
    Write-Host ""
    Write-Host "=== libtorch installed successfully ===" -ForegroundColor Green
    Write-Host "  Path:      $dest"
    Write-Host "  Torch_DIR: $dest\share\cmake\Torch"
} else {
    Write-Host "ERROR: TorchConfig.cmake not found after extraction!" -ForegroundColor Red
    Write-Host "Check contents of $dest"
    exit 1
}

# Cleanup zip
Remove-Item $zipPath -Force -ErrorAction SilentlyContinue
Write-Host "Temp zip removed."
