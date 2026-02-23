#!/usr/bin/env bash
set -euo pipefail

NAME="bird_ai"

if docker ps --format '{{.Names}}' | grep -qx "$NAME"; then
  echo "[+] Stopping $NAME..."
  docker stop "$NAME" >/dev/null
  echo "[+] Stopped."
else
  echo "[=] $NAME is not running."
fi
