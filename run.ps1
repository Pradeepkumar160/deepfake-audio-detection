# Deepfake Audio Detector - Quick Run Script
# Usage (from project directory):
#   .\run.ps1

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Deepfake Audio Detection System" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# Activate virtual environment if it exists
if (Test-Path ".\venv\Scripts\Activate.ps1") {
    Write-Host "Activating virtual environment..." -ForegroundColor Yellow
    & ".\venv\Scripts\Activate.ps1"
} else {
    Write-Host "No venv found — using global Python. Run setup.ps1 first if packages are missing." -ForegroundColor Yellow
}

# Generate samples if dataset is empty
$realCount = (Get-ChildItem -Path "dataset\real" -Filter "*.wav" -ErrorAction SilentlyContinue).Count
if ($realCount -lt 1) {
    Write-Host "No training data found. Generating synthetic samples..." -ForegroundColor Yellow
    python generate_samples.py
}

Write-Host ""
Write-Host "Starting server..." -ForegroundColor Green
Write-Host "Open browser at: http://localhost:8000" -ForegroundColor Cyan
Write-Host "Press Ctrl+C to stop." -ForegroundColor Gray
Write-Host ""

python deepfake_audio_detector.py
