# Setup script for Meeting Transcription App
# Production-ready setup with comprehensive system checks

Write-Host "=" * 60 -ForegroundColor Green
Write-Host "Meeting Transcription App - Setup" -ForegroundColor Green
Write-Host "=" * 60 -ForegroundColor Green

# 1. Check Windows Version
Write-Host "`n[1/9] Checking Windows version..." -ForegroundColor Cyan
$winVersion = [System.Environment]::OSVersion.Version
$winBuild = $winVersion.Build

if ($winBuild -lt 17763) {
    Write-Host "Error: Windows 10 Build 17763 (Version 1809) or later required" -ForegroundColor Red
    Write-Host "Found: Build $winBuild" -ForegroundColor Red
    Write-Host "Please update Windows before continuing" -ForegroundColor Yellow
    exit 1
} else {
    $winName = (Get-WmiObject -Class Win32_OperatingSystem).Caption
    Write-Host "Success: $winName (Build $winBuild)" -ForegroundColor Green
}

# 2. Check if running as Administrator
Write-Host "`n[2/9] Checking administrator privileges..." -ForegroundColor Cyan
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    Write-Host "Warning: Not running as administrator" -ForegroundColor Yellow
    Write-Host "Some audio features may require admin privileges" -ForegroundColor Yellow
} else {
    Write-Host "Success: Running with administrator privileges" -ForegroundColor Green
}

# 3. Check Python version
Write-Host "`n[3/9] Checking Python version..." -ForegroundColor Cyan
try {
    $pythonVersion = python --version 2>&1
    if ($pythonVersion -match "Python 3\.([0-9]+)\.([0-9]+)") {
        $minorVersion = [int]$matches[1]
        $patchVersion = [int]$matches[2]
        
        if ($minorVersion -ge 10) {
            Write-Host "Success: Python 3.$minorVersion.$patchVersion detected" -ForegroundColor Green
        } else {
            Write-Host "Error: Python 3.10+ required. Found: $pythonVersion" -ForegroundColor Red
            Write-Host "Download from: https://www.python.org/downloads/" -ForegroundColor Yellow
            exit 1
        }
    } else {
        throw "Python version check failed"
    }
} catch {
    Write-Host "Error: Python not found or version check failed" -ForegroundColor Red
    Write-Host "Install Python 3.10+ from: https://www.python.org/downloads/" -ForegroundColor Yellow
    exit 1
}

# 4. Check available disk space
Write-Host "`n[4/9] Checking disk space..." -ForegroundColor Cyan
$drive = Get-PSDrive -Name C
$freeSpaceGB = [math]::Round($drive.Free / 1GB, 2)

if ($freeSpaceGB -lt 3) {
    Write-Host "Error: Minimum 3GB disk space required (models are ~1-3GB)" -ForegroundColor Red
    Write-Host "Found: $freeSpaceGB GB free" -ForegroundColor Red
    exit 1
} elseif ($freeSpaceGB -lt 5) {
    Write-Host "Warning: Low disk space ($freeSpaceGB GB free)" -ForegroundColor Yellow
    Write-Host "5GB+ recommended for best experience" -ForegroundColor Yellow
} else {
    Write-Host "Success: $freeSpaceGB GB free" -ForegroundColor Green
}

# 5. Check Visual C++ Redistributable
Write-Host "`n[5/9] Checking Visual C++ Redistributable..." -ForegroundColor Cyan
$vcRedistPaths = @(
    "HKLM:\SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64",
    "HKLM:\SOFTWARE\WOW6432Node\Microsoft\VisualStudio\14.0\VC\Runtimes\x64"
)

$vcInstalled = $false
foreach ($path in $vcRedistPaths) {
    if (Test-Path $path) {
        $vcInstalled = $true
        break
    }
}

if ($vcInstalled) {
    Write-Host "Success: Visual C++ Redistributable found" -ForegroundColor Green
} else {
    Write-Host "Warning: Visual C++ Redistributable not found" -ForegroundColor Yellow
    Write-Host "May be needed for audio libraries" -ForegroundColor Yellow
    Write-Host "Download from: https://aka.ms/vs/17/release/vc_redist.x64.exe" -ForegroundColor Yellow
}

# 6. Create virtual environment
Write-Host "`n[6/9] Creating virtual environment..." -ForegroundColor Cyan
if (Test-Path "venv") {
    Write-Host "Virtual environment already exists" -ForegroundColor Yellow
} else {
    python -m venv venv
    Write-Host "Success: Virtual environment created" -ForegroundColor Green
}

# 7. Activate virtual environment
Write-Host "`n[7/9] Activating virtual environment..." -ForegroundColor Cyan
& .\venv\Scripts\Activate.ps1

# Upgrade pip
Write-Host "`nUpgrading pip..." -ForegroundColor Cyan
python -m pip install --upgrade pip

# 8. Install dependencies
Write-Host "`n[8/9] Installing dependencies..." -ForegroundColor Cyan
pip install -r requirements.txt

# 9. Copy .env.example to .env if not exists
Write-Host "`n[9/9] Creating configuration file..." -ForegroundColor Cyan
if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "Success: .env file created. Please update with your settings." -ForegroundColor Green
} else {
    Write-Host ".env file already exists" -ForegroundColor Yellow
}

# Run system check
Write-Host "`n" + "=" * 60 -ForegroundColor Green
Write-Host "Running comprehensive system check..." -ForegroundColor Green
Write-Host "=" * 60 -ForegroundColor Green
python check_system.py

Write-Host "`n" + "=" * 60 -ForegroundColor Green
Write-Host "Setup complete!" -ForegroundColor Green
Write-Host "=" * 60 -ForegroundColor Green
Write-Host "`nTo run the application:" -ForegroundColor Cyan
Write-Host "  1. Activate the virtual environment: .\venv\Scripts\Activate.ps1" -ForegroundColor White
Write-Host "  2. Run the app: python run.py" -ForegroundColor White
Write-Host "`nNote: For cloud transcription, update .env with Azure credentials." -ForegroundColor Yellow
