$ErrorActionPreference = "Stop"

Write-Host "=== 1. Running tests ===" -ForegroundColor Cyan

python -m pytest -v

if ($LASTEXITCODE -ne 0) {
    throw "Tests failed. Deployment cancelled."
}

Write-Host "=== 2. Building Docker image ===" -ForegroundColor Cyan

docker compose build

if ($LASTEXITCODE -ne 0) {
    throw "Docker build failed."
}

Write-Host "=== 3. Updating containers ===" -ForegroundColor Cyan

docker compose up -d

if ($LASTEXITCODE -ne 0) {
    throw "Container update failed."
}

Write-Host "=== 4. Checking API ===" -ForegroundColor Cyan

Start-Sleep -Seconds 5

try {
    $response = Invoke-RestMethod `
        -Uri "http://localhost:8000/health" `
        -Method Get

    if ($response.status -ne "ok") {
        throw "API health check failed."
    }
}
catch {
    docker compose logs --tail=50 api
    throw "Deployment completed, but API health check failed."
}

Write-Host "=== Deployment successful ===" -ForegroundColor Green