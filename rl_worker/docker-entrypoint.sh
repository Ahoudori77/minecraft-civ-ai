#!/usr/bin/env bash
set -euo pipefail

# ── 1. 環境変数 ──────────────────────────────
export DISPLAY=:99
XVFB_WHD=${XVFB_WHD:-1280x720x24}   # 例: 1600x900x24

# ── 2. 前回クラッシュ時のゴミ掃除 ───────────
rm -f /tmp/.X99-lock /tmp/.X11-unix/X99 || true

# ── 3. Xvfb ごと任意コマンドを実行 ─────────
exec xvfb-run -a \                                 # 空いている :99 を自動で探す
  --error-file=/proc/1/fd/1 \                      # ← ここが肝！  標準出力/エラーを
                                                  #    PID1 の stdout (= docker log) に送る
  -s "-screen 0 1024x768x24 +extension GLX" \      # 画面サイズ/GLX 有効
  "$@"     