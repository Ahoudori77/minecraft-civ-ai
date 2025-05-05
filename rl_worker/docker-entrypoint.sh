#!/usr/bin/env bash
set -e
# Xvfb をバックグラウンドで起動
Xvfb :99 -screen 0 1280x720x24 -ac &
# 指定されたコマンド（デフォルトは python train.py）を実行
exec "$@"
