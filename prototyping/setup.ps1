# RL-LLM-AutoTrainer Setup Script (Windows PowerShell)
# Run with: .\setup.ps1

$ErrorActionPreference = "Stop"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host " RL-LLM-AutoTrainer Setup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Get script directory
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

# Check Python
Write-Host "[1/4] Checking Python installation..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1
    Write-Host "  Found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "  ERROR: Python not found. Please install Python 3.9+" -ForegroundColor Red
    exit 1
}

# Check Node.js
Write-Host "[2/4] Checking Node.js installation..." -ForegroundColor Yellow
try {
    $nodeVersion = node --version 2>&1
    Write-Host "  Found: Node.js $nodeVersion" -ForegroundColor Green
} catch {
    Write-Host "  ERROR: Node.js not found. Please install Node.js LTS" -ForegroundColor Red
    exit 1
}

# Install Python dependencies
Write-Host "[3/4] Installing Python dependencies..." -ForegroundColor Yellow
Write-Host "  This may take several minutes for PyTorch..." -ForegroundColor Gray
pip install -r "$ScriptDir\requirements.txt"
if ($LASTEXITCODE -ne 0) {
    Write-Host "  ERROR: Failed to install Python dependencies" -ForegroundColor Red
    exit 1
}
Write-Host "  Python dependencies installed successfully" -ForegroundColor Green

# Install Node.js dependencies and build CLI
Write-Host "[4/4] Installing CLI wrapper dependencies..." -ForegroundColor Yellow
Push-Location "$ScriptDir\cli-wrapper"
try {
    npm install
    if ($LASTEXITCODE -ne 0) { throw "npm install failed" }

    npm run build
    if ($LASTEXITCODE -ne 0) { throw "npm build failed" }

    Write-Host "  CLI wrapper built successfully" -ForegroundColor Green
} catch {
    Write-Host "  ERROR: $_" -ForegroundColor Red
    Pop-Location
    exit 1
}
Pop-Location

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host " Setup Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Usage:" -ForegroundColor Cyan
Write-Host "  cd cli-wrapper && npm start" -ForegroundColor White
Write-Host ""
Write-Host "Or run directly:" -ForegroundColor Cyan
Write-Host "  node prototyping/cli-wrapper/dist/index.js" -ForegroundColor White
Write-Host ""
