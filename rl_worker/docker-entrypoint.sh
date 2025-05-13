#!/usr/bin/env bash
set -e

# Xvfb をバックグラウンド起動（GPU なしのダミー X サーバ）
Xvfb :99 -screen 0 1280x720x24 -ac &

# 利用者コマンドをそのまま実行
exec "$@"
