#!/usr/bin/env bash
set -euo pipefail

# ── 1. 環境変数 ──────────────────────────────
export DISPLAY=:99
XVFB_WHD="${XVFB_WHD:-1280x720x24}"   # 例: 1600x900x24

# ── 2. 前回クラッシュ時のゴミ掃除 ───────────
rm -f /tmp/.X99-lock /tmp/.X11-unix/X99 || true

# ── 3. Xvfb ごと任意コマンドを実行 ─────────
exec xvfb-run -a \
  --error-file=/proc/1/fd/1 \                 # ← Xvfb の stdout/stderr → docker logs
  -s "-screen 0 ${XVFB_WHD} +extension GLX" \ # 仮想画面サイズ
  -- "$@"                                     # -- 以降が Python コマンド
