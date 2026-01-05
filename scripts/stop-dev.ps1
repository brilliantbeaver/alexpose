# AlexPose Development Server Stop Script (Windows PowerShell)

Write-Host "🛑 Stopping AlexPose Development Servers..." -ForegroundColor Cyan
Write-Host ""

# Stop processes on port 8000 (Backend)
$backend = Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue
if ($backend) {
    $backendPid = $backend.OwningProcess
    Write-Host "Stopping backend server (PID: $backendPid)..." -ForegroundColor Yellow
    Stop-Process -Id $backendPid -Force -ErrorAction SilentlyContinue
    Write-Host "✓ Backend server stopped" -ForegroundColor Green
} else {
    Write-Host "⚠️  No backend server found on port 8000" -ForegroundColor Yellow
}

# Stop processes on port 3000 (Frontend)
$frontend = Get-NetTCPConnection -LocalPort 3000 -ErrorAction SilentlyContinue
if ($frontend) {
    $frontendPid = $frontend.OwningProcess
    Write-Host "Stopping frontend server (PID: $frontendPid)..." -ForegroundColor Yellow
    Stop-Process -Id $frontendPid -Force -ErrorAction SilentlyContinue
    Write-Host "✓ Frontend server stopped" -ForegroundColor Green
} else {
    Write-Host "⚠️  No frontend server found on port 3000" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "✅ All servers stopped" -ForegroundColor Green
