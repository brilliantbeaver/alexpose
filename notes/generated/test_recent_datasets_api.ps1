# Test script to diagnose Recent Datasets loading issue

Write-Host "Testing Recent Datasets API..." -ForegroundColor Cyan
Write-Host ""

# Test 1: Check if backend is running
Write-Host "Test 1: Checking if backend server is running..." -ForegroundColor Yellow
try {
    $healthResponse = Invoke-WebRequest -Uri "http://localhost:8000/health" -UseBasicParsing -TimeoutSec 5
    Write-Host "Backend server is running (Status: $($healthResponse.StatusCode))" -ForegroundColor Green
} catch {
    Write-Host "Backend server is NOT running!" -ForegroundColor Red
    Write-Host "Please start the server with: ./scripts/start-dev.ps1" -ForegroundColor Yellow
    exit 1
}

Write-Host ""

# Test 2: Check if frontend is running
Write-Host "Test 2: Checking if frontend server is running..." -ForegroundColor Yellow
try {
    $frontendResponse = Invoke-WebRequest -Uri "http://localhost:3000" -UseBasicParsing -TimeoutSec 5
    Write-Host "Frontend server is running (Status: $($frontendResponse.StatusCode))" -ForegroundColor Green
} catch {
    Write-Host "Frontend server is NOT running!" -ForegroundColor Red
    Write-Host "Please start the server with: ./scripts/start-dev.ps1" -ForegroundColor Yellow
    exit 1
}

Write-Host ""

# Test 3: Test the GAVD list API endpoint
Write-Host "Test 3: Testing GAVD list API endpoint..." -ForegroundColor Yellow
try {
    $apiResponse = Invoke-WebRequest -Uri "http://localhost:8000/api/v1/gavd/list?limit=5" -UseBasicParsing -TimeoutSec 10
    Write-Host "API endpoint is responding (Status: $($apiResponse.StatusCode))" -ForegroundColor Green
    
    $jsonData = $apiResponse.Content | ConvertFrom-Json
    Write-Host "  - Success: $($jsonData.success)" -ForegroundColor Gray
    Write-Host "  - Count: $($jsonData.count)" -ForegroundColor Gray
    Write-Host "  - Datasets: $($jsonData.datasets.Count)" -ForegroundColor Gray
    
    if ($jsonData.datasets.Count -gt 0) {
        Write-Host "  - First dataset ID: $($jsonData.datasets[0].dataset_id)" -ForegroundColor Gray
        Write-Host "  - First dataset status: $($jsonData.datasets[0].status)" -ForegroundColor Gray
    }
} catch {
    Write-Host "API endpoint failed!" -ForegroundColor Red
    Write-Host "  Error: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}

Write-Host ""

# Test 4: Check metadata directory
Write-Host "Test 4: Checking metadata directory..." -ForegroundColor Yellow
$metadataDir = "data/training/gavd/metadata"
if (Test-Path $metadataDir) {
    $metadataFiles = Get-ChildItem -Path $metadataDir -Filter "*.json"
    Write-Host "Metadata directory exists" -ForegroundColor Green
    Write-Host "  - Files found: $($metadataFiles.Count)" -ForegroundColor Gray
    
    if ($metadataFiles.Count -gt 0) {
        Write-Host "  - Latest file: $($metadataFiles[0].Name)" -ForegroundColor Gray
    }
} else {
    Write-Host "Metadata directory does not exist" -ForegroundColor Yellow
    Write-Host "  This is normal if no datasets have been uploaded yet" -ForegroundColor Gray
}

Write-Host ""
Write-Host "All tests completed!" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1. Open browser to: http://localhost:3000/training/gavd" -ForegroundColor White
Write-Host "  2. Click on Recent Datasets tab" -ForegroundColor White
Write-Host "  3. Open browser console (F12) to see logs" -ForegroundColor White
Write-Host "  4. Check for any errors in console" -ForegroundColor White
Write-Host ""
