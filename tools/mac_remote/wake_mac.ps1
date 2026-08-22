<#
.SYNOPSIS
    Sends a Wake-on-LAN (WOL) magic packet to wake the remote Mac Mini.
.PARAMETER MacAddress
    Target MAC Address. Default is set to the detected Mac Mini address.
.PARAMETER IPAddress
    Target broadcast IP or Mac IP.
#>
param (
    [string]$MacAddress = "FA:4F:12:32:BB:81",
    [string]$TargetIP = "10.20.103.34"
)

Write-Host "=================================================" -ForegroundColor Cyan
Write-Host " Waking Remote Mac Mini ($TargetIP)" -ForegroundColor Cyan
Write-Host " MAC: $MacAddress" -ForegroundColor Cyan
Write-Host "=================================================" -ForegroundColor Cyan

try {
    $cleanMac = $MacAddress -replace "[:-]", ""
    if ($cleanMac.Length -ne 12) {
        throw "Invalid MAC Address format: $MacAddress"
    }

    # Construct standard 102-byte WOL Magic Packet (6x 0xFF + 16x MAC)
    $packet = [byte[]](@(0xFF) * 6 + (0..15 | ForEach-Object {
        0..5 | ForEach-Object { [byte]::Parse($cleanMac.Substring($_ * 2, 2), [System.Globalization.NumberStyles]::HexNumber) }
    }))

    # Send UDP broadcast over port 9 and port 7
    $udpClient = New-Object System.Net.Sockets.UdpClient
    $udpClient.Connect([System.Net.IPAddress]::Broadcast, 9)
    $udpClient.Send($packet, $packet.Length) | Out-Null
    $udpClient.Close()

    Write-Host "[+] Magic packet broadcasted successfully." -ForegroundColor Green
    Write-Host "Waiting 5 seconds for the Mac to power on..." -ForegroundColor Yellow
    Start-Sleep -Seconds 5

    # Check ping
    $ping = Test-Connection -ComputerName $TargetIP -Count 2 -Quiet
    if ($ping) {
        Write-Host "[SUCCESS] Mac Mini is AWAKE and responding to ping at $TargetIP!" -ForegroundColor Green
    } else {
        Write-Host "[INFO] Packet sent. If the Mac was sleeping, it may take 10-15 seconds to connect to the network." -ForegroundColor Gray
    }
}
catch {
    Write-Host "[-] Error waking Mac Mini: $_" -ForegroundColor Red
}
