#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Start the AlexPose server with clean state (kills any existing instances)

.DESCRIPTION
    This script ensures a clean server startup by:
    1. Checking for existing processes on port 8000
    2. Killing any stale server processes
    3. Starting a fresh server instance
    4. Verifying successful startup

.EXAMPLE
    .\scripts\start-server-clean.ps1
#>

param(
    [int]$Port = 8000,
    [string]$Host = "127.0.0.1",
    [switch]$NoReload
)

Write-Host "🚀 AlexPose Server Clean Startup" -ForegroundColor Cyan
Write-Host "=" * 60 -ForegroundColor Cyan

# Step 1: Check for existing processes on port
Write-Host "`n📡 Checking port $Port..." -ForegroundColor Yellow

$existingProcess = netstat -ano | Select-String ":$Port.*LISTENING"

if ($existingProcess) {
    Write-Host "⚠️  Found existing process on port $Port" -ForegroundColor Yellow
    
    # Extract PID from netstat output
    $pidMatch = $existingProcess -match '\s+(\d+)\s*$'
    if ($pidMatch) {
        $pid = $Matches[1]
        Write-Host "   PID: $pid" -ForegroundColor Gray
        
        # Kill the process
        Write-Host "🔪 Killing stale process..." -ForegroundColor Yellow
        try {
            Stop-Process -Id $pid -Force -ErrorAction Stop
            Write-Host "✅ Stale process killed successfully" -ForegroundColor Green
            Start-Sleep -Seconds 1  # Give OS time to release port
        }
        catch {
            Write-Host "❌ Failed to kill process: $_" -ForegroundColor Red
            Write-Host "   Try manually: taskkill /F /PID $pid" -ForegroundColor Yellow
            exit 1
        }
    }
}
else {
    Write-Host "✅ Port $Port is available" -ForegroundColor Green
}

# Step 2: Verify configuration
Write-Host "`n📋 Verifying configuration..." -ForegroundColor Yellow

try {
    $configCmd = "from ambient.core.config import ConfigurationManager; cm = ConfigurationManager('config', 'development'); result = cm.validate_configuration(); print('VALID' if result else 'INVALID')"
    $configTest = python -c $configCmd 2>&1 | Out-String
    
    if ($configTest -match "VALID") {
        Write-Host "✅ Configuration is valid" -ForegroundColor Green
    }
    else {
        Write-Host "⚠️  Configuration has warnings (continuing anyway)" -ForegroundColor Yellow
    }
}
catch {
    Write-Host "⚠️  Could not verify configuration (continuing anyway)" -ForegroundColor Yellow
}

# Step 3: Start the server
Write-Host "`n🚀 Starting server..." -ForegroundColor Yellow
Write-Host "   Host: $Host" -ForegroundColor Gray
Write-Host "   Port: $Port" -ForegroundColor Gray

if ($NoReload) {
    Write-Host "   Reload: Disabled" -ForegroundColor Gray
    Write-Host ""
    uvicorn server.main:app --host $Host --port $Port
}
else {
    Write-Host "   Reload: Enabled" -ForegroundColor Gray
    Write-Host ""
    uvicorn server.main:app --reload --host $Host --port $Port
}
