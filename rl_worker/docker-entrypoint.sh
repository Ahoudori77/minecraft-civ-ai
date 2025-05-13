#!/usr/bin/env bash
set -e

# Xvfb をバックグラウンドで起動
Xvfb :99 -screen 0 1280x720x24 -ac &

# ユーザーコマンド（docker-compose.yml で上書き可）
exec "$@"
