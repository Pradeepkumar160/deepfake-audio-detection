# Deepfake Audio Detector - PowerShell Setup Script
# Run this once from the project directory:
#   cd DeepfakeAudioDetection
#   .\setup.ps1

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Deepfake Audio Detection System - Setup" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# Check Python
Write-Host "[1/5] Checking Python installation..." -ForegroundColor Yellow
try {
    $pyver = python --version 2>&1
    Write-Host "  Found: $pyver" -ForegroundColor Green
} catch {
    Write-Host "  ERROR: Python not found. Install Python 3.10+ from https://python.org" -ForegroundColor Red
    exit 1
}

# Create virtual environment
Write-Host "[2/5] Creating virtual environment..." -ForegroundColor Yellow
if (Test-Path "venv") {
    Write-Host "  venv already exists, skipping." -ForegroundColor Gray
} else {
    python -m venv venv
    Write-Host "  Created venv/" -ForegroundColor Green
}

# Activate venv
Write-Host "[3/5] Activating virtual environment..." -ForegroundColor Yellow
& ".\venv\Scripts\Activate.ps1"
Write-Host "  Activated." -ForegroundColor Green

# Upgrade pip
Write-Host "[4/5] Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip --quiet

# Install dependencies
Write-Host "[5/5] Installing dependencies (this may take 3-5 minutes)..." -ForegroundColor Yellow
pip install -r requirements.txt

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "  Dependency install failed. Trying fallback (no version pins)..." -ForegroundColor Yellow
    pip install fastapi "uvicorn[standard]" python-multipart websockets numpy torch torchaudio librosa soundfile scikit-learn pydantic
}

Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host "  Setup complete!" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1. Activate venv:  .\venv\Scripts\Activate.ps1"
Write-Host "  2. Generate data:  python generate_samples.py"
Write-Host "  3. Start server:   python deepfake_audio_detector.py"
Write-Host "  4. Open browser:   http://localhost:8000"
Write-Host "  5. Click 'Train Model', then upload audio files to analyze."
Write-Host ""
