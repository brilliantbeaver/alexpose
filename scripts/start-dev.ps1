# AlexPose Development Server Startup Script (Windows PowerShell)
# This script starts both the backend FastAPI server and frontend Next.js server

Write-Host "🚀 Starting AlexPose Development Servers..." -ForegroundColor Cyan
Write-Host ""

# Check if we're in the project root
if (-not (Test-Path "pyproject.toml")) {
    Write-Host "❌ Error: Please run this script from the project root directory" -ForegroundColor Red
    exit 1
}

# Check if uv is installed
try {
    $uvVersion = uv --version 2>&1
    Write-Host "✓ UV found: $uvVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Error: UV is not installed. Please install it first:" -ForegroundColor Red
    Write-Host "   https://docs.astral.sh/uv/getting-started/installation/" -ForegroundColor Yellow
    exit 1
}

# Check if Node.js is installed
try {
    $nodeVersion = node --version 2>&1
    Write-Host "✓ Node.js found: $nodeVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Error: Node.js is not installed. Please install it first:" -ForegroundColor Red
    Write-Host "   https://nodejs.org/" -ForegroundColor Yellow
    exit 1
}

Write-Host ""
Write-Host "📦 Checking dependencies..." -ForegroundColor Cyan

# Check if frontend dependencies are installed
if (-not (Test-Path "frontend/node_modules")) {
    Write-Host "⚠️  Frontend dependencies not found. Installing..." -ForegroundColor Yellow
    Push-Location frontend
    npm install
    Pop-Location
}

Write-Host ""
Write-Host "🔧 Starting Backend Server (FastAPI)..." -ForegroundColor Cyan
Write-Host "   URL: http://localhost:8000" -ForegroundColor Gray
Write-Host "   Docs: http://localhost:8000/docs" -ForegroundColor Gray

# Start backend in a new PowerShell window
$backendScript = @"
Write-Host '🐍 Backend Server Starting...' -ForegroundColor Green
Write-Host 'Press Ctrl+C to stop the server' -ForegroundColor Yellow
Write-Host ''
uv run uvicorn server.main:app --reload --host 127.0.0.1 --port 8000
"@

Start-Process powershell -ArgumentList "-NoExit", "-Command", $backendScript

# Wait a bit for backend to start
Write-Host "⏳ Waiting for backend to start..." -ForegroundColor Yellow
Start-Sleep -Seconds 3

# Check if backend is running
try {
    $response = Invoke-WebRequest -Uri "http://localhost:8000/health" -UseBasicParsing -TimeoutSec 5
    Write-Host "✓ Backend server is running!" -ForegroundColor Green
} catch {
    Write-Host "⚠️  Backend server may still be starting..." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "🎨 Starting Frontend Server (Next.js)..." -ForegroundColor Cyan
Write-Host "   URL: http://localhost:3000" -ForegroundColor Gray

# Start frontend in a new PowerShell window
$frontendScript = @"
Write-Host '⚛️  Frontend Server Starting...' -ForegroundColor Green
Write-Host 'Press Ctrl+C to stop the server' -ForegroundColor Yellow
Write-Host ''
Set-Location frontend
npm run dev
"@

Start-Process powershell -ArgumentList "-NoExit", "-Command", $frontendScript

Write-Host ""
Write-Host "✅ Both servers are starting!" -ForegroundColor Green
Write-Host ""
Write-Host "📍 Access Points:" -ForegroundColor Cyan
Write-Host "   Frontend:  http://localhost:3000" -ForegroundColor White
Write-Host "   Backend:   http://localhost:8000" -ForegroundColor White
Write-Host "   API Docs:  http://localhost:8000/docs" -ForegroundColor White
Write-Host "   GAVD Upload: http://localhost:3000/training/gavd" -ForegroundColor White
Write-Host ""
Write-Host "💡 Tip: Close the PowerShell windows to stop the servers" -ForegroundColor Yellow
Write-Host ""
