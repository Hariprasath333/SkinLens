# SkinLens: start the app and share it on a free public HTTPS link (Cloudflare quick tunnel).
# Run from the clinical_system folder:   powershell -ExecutionPolicy Bypass -File .\start_public.ps1
# Stop everything with Ctrl+C in this window (the server window closes too).

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Get-Command cloudflared -ErrorAction SilentlyContinue)) {
    Write-Host "cloudflared is not installed. Install it once with:" -ForegroundColor Yellow
    Write-Host "    winget install --id Cloudflare.cloudflared" -ForegroundColor Yellow
    Write-Host "then close and reopen PowerShell and run this script again."
    exit 1
}

# Public-safe settings: no self-registration, cookies only over HTTPS, server reachable only through the tunnel.
$env:ALLOW_REGISTRATION = "0"
$env:COOKIE_SECURE = "1"

Write-Host "Starting the SkinLens server (model loading takes about a minute)..." -ForegroundColor Cyan
$server = Start-Process -FilePath ".\.venv\Scripts\python.exe" `
    -ArgumentList "-m", "uvicorn", "webapp.main:app", "--host", "127.0.0.1", "--port", "8080", "--proxy-headers" `
    -PassThru

# wait until /health answers
$ok = $false
for ($i = 0; $i -lt 120; $i++) {
    Start-Sleep -Seconds 2
    try { Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8080/health -TimeoutSec 3 | Out-Null; $ok = $true; break } catch {}
}
if (-not $ok) { Write-Host "The server did not start. Check its window for the error." -ForegroundColor Red; exit 1 }

Write-Host ""
Write-Host "Server is up. Opening the public tunnel..." -ForegroundColor Green
Write-Host "Look below for a line with  https://....trycloudflare.com  - that is your public link." -ForegroundColor Green
Write-Host "Anyone with the link can open the app while this window is open. Use demo data only." -ForegroundColor Yellow
Write-Host ""
try {
    cloudflared tunnel --url http://127.0.0.1:8080
} finally {
    if ($server -and -not $server.HasExited) { Stop-Process -Id $server.Id -Force }
    Write-Host "Tunnel and server stopped."
}
