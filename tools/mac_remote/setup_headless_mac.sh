#!/usr/bin/env bash
# ==============================================================================
# One-time Headless Server Configuration for Mac Mini
# Run this directly on the Mac Mini (or via SSH) with sudo privileges.
# ==============================================================================

echo ">>> Configuring Mac Mini for 24/7 Headless Server Operation..."

# 1. Prevent system / CPU sleep entirely
sudo pmset -a sleep 0

# 2. Prevent disk sleep
sudo pmset -a disksleep 0

# 3. Disable display sleep timer
sudo pmset -a displaysleep 0

# 4. Enable Wake-on-LAN (womp = Wake On Magic Packet)
sudo pmset -a womp 1

# 5. Automatically power back on after a power outage
sudo pmset -a autorestart 1

# 6. Prevent automatic sleep when idle
sudo pmset -a halfdim 0

echo ">>> Configuration Complete! Current Power Management Settings:"
pmset -g
