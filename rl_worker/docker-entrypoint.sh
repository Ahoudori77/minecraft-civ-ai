#!/usr/bin/env bash
set -e

# 画面サイズは環境変数で調整可（例： "1920x1080x24"）
XVFB_WHD=${XVFB_WHD:-1400x900x24}

# Mesa のソフトウェア GL を常に使う
export LIBGL_ALWAYS_SOFTWARE=1

# Xvfb をバックグラウンドで起動
Xvfb :99 -screen 0 ${XVFB_WHD} +extension GLX +extension RENDER -ac -noreset &
XVFB_PID=$!

# docker stop で綺麗に終了させる
trap "kill -TERM ${XVFB_PID}" TERM INT

# 少し待ってからメインプロセスへ
sleep 2
exec "$@"
