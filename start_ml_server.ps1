# Start the ML inference server for DeClickify
Write-Host "Starting DeClickify ML Server..." -ForegroundColor Green
Write-Host ""
Write-Host "The server will run on http://localhost:5000" -ForegroundColor Cyan
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Yellow
Write-Host ""

python ml_server.py
