<#
.SYNOPSIS
    Checks the power and network status of the remote Mac Mini.
#>
param (
    [string]$TargetIP = "10.20.103.34"
)

Write-Host "Pinging Mac Mini at $TargetIP..." -ForegroundColor Cyan
$alive = Test-Connection -ComputerName $TargetIP -Count 3 -Quiet

if ($alive) {
    Write-Host "[STATUS] Mac Mini is ONLINE / AWAKE." -ForegroundColor Green
} else {
    Write-Host "[STATUS] Mac Mini is OFFLINE / ASLEEP." -ForegroundColor Red
}
