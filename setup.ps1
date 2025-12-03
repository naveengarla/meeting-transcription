# Setup script for Meeting Transcription App
# Run this script to set up the development environment

Write-Host "Setting up Meeting Transcription App..." -ForegroundColor Green

# Check Python version
Write-Host "`nChecking Python version..." -ForegroundColor Cyan
$pythonVersion = python --version 2>&1
if ($pythonVersion -match "Python 3\.([0-9]+)") {
    $minorVersion = [int]$matches[1]
    if ($minorVersion -ge 10) {
        Write-Host "Success: $pythonVersion detected" -ForegroundColor Green
    } else {
        Write-Host "Error: Python 3.10+ required. Found: $pythonVersion" -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "Error: Python not found or version check failed" -ForegroundColor Red
    exit 1
}

# Create virtual environment
Write-Host "`nCreating virtual environment..." -ForegroundColor Cyan
if (Test-Path "venv") {
    Write-Host "Virtual environment already exists" -ForegroundColor Yellow
} else {
    python -m venv venv
    Write-Host "Success: Virtual environment created" -ForegroundColor Green
}

# Activate virtual environment
Write-Host "`nActivating virtual environment..." -ForegroundColor Cyan
& .\venv\Scripts\Activate.ps1

# Upgrade pip
Write-Host "`nUpgrading pip..." -ForegroundColor Cyan
python -m pip install --upgrade pip

# Install dependencies
Write-Host "`nInstalling dependencies..." -ForegroundColor Cyan
pip install -r requirements.txt

# Copy .env.example to .env if not exists
if (-not (Test-Path ".env")) {
    Write-Host "`nCreating .env file..." -ForegroundColor Cyan
    Copy-Item ".env.example" ".env"
    Write-Host "Success: .env file created. Please update with your settings." -ForegroundColor Green
}

Write-Host "`nSetup complete!" -ForegroundColor Green
Write-Host "`nTo run the application:" -ForegroundColor Cyan
Write-Host "  1. Activate the virtual environment: .\venv\Scripts\Activate.ps1" -ForegroundColor White
Write-Host "  2. Run the app: python run.py" -ForegroundColor White
Write-Host "`nNote: For cloud transcription, update .env with Azure credentials." -ForegroundColor Yellow
