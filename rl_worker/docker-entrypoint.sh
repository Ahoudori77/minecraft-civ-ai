#!/usr/bin/env bash
set -e

# Xvfb をバックグラウンドで起動
if ! pgrep -f "Xvfb :99" >/dev/null; then
  Xvfb :99 -screen 0 1280x720x24 -ac &
fi

# ユーザーコマンド（docker-compose.yml で上書き可）
exec "$@"
