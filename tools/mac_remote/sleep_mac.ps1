<#
.SYNOPSIS
    Puts the remote Mac Mini to sleep via SSH command.
.PARAMETER TargetIP
    IP address of the remote Mac Mini.
.PARAMETER User
    macOS username.
#>
param (
    [string]$TargetIP = "10.20.103.34",
    [string]$User = ""
)

Write-Host "=================================================" -ForegroundColor Yellow
Write-Host " Putting Remote Mac Mini to Sleep ($TargetIP)" -ForegroundColor Yellow
Write-Host "=================================================" -ForegroundColor Yellow

if ([string]::IsNullOrWhiteSpace($User)) {
    $User = Read-Host "Enter your macOS username on the Mac Mini"
}

if ([string]::IsNullOrWhiteSpace($User)) {
    Write-Host "[-] Username cannot be empty." -ForegroundColor Red
    exit 1
}

Write-Host "Connecting via SSH to issue sleep command..." -ForegroundColor Cyan
# Uses osascript which puts Mac to sleep without needing sudo password prompt
ssh -o ConnectTimeout=5 "$User@$TargetIP" "osascript -e 'tell application \"System Events\" to sleep'"

if ($LASTEXITCODE -eq 0) {
    Write-Host "[SUCCESS] Sleep command sent to Mac Mini." -ForegroundColor Green
} else {
    Write-Host "[-] Failed to connect or send command. Make sure Remote Login (SSH) is enabled on the Mac." -ForegroundColor Red
}
