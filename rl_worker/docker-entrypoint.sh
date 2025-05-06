#!/usr/bin/env bash
set -e
# Xvfbをバックグラウンドで起動
Xvfb :99 -screen 0 1280x720x24 -ac &
# そのままコマンド実行
exec "$@"
