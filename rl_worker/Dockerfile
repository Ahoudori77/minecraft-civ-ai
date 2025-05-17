# ------------------------------------------------------------
# rl_worker/Dockerfile        (CPU only • Python 3.11)
# ------------------------------------------------------------
FROM python:3.11-slim

# ─────────────────────────────────────────────────────────────
# 1. OS-level runtime deps
# ─────────────────────────────────────────────────────────────
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        openjdk-17-jdk-headless \
        xvfb x11-utils xauth \
        libgl1-mesa-glx ffmpeg \
        build-essential python3-dev \
        curl wget unzip git && \
    rm -rf /var/lib/apt/lists/*

ENV DISPLAY=:99
ENV LIBGL_ALWAYS_SOFTWARE=1

WORKDIR /app

# ─────────────────────────────────────────────────────────────
# 2. MineRL patched wheel  (※ 拡張子を保持して保存する)
# ─────────────────────────────────────────────────────────────
ENV MINERL_WHL=minerl_mirror-0.4.4-cp311-cp311-linux_x86_64.whl
RUN wget -q --show-progress --continue --tries=5 \
      "https://github.com/danijar/minerl/releases/download/v0.4.4-patched/${MINERL_WHL}"

# ─────────────────────────────────────────────────────────────
# 3. Python deps
# ─────────────────────────────────────────────────────────────
COPY rl_worker/requirements.txt ./requirements.txt
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir --no-deps "./${MINERL_WHL}" && \
    rm "${MINERL_WHL}"

# ─────────────────────────────────────────────────────────────
# 4. App source & entrypoint
# ─────────────────────────────────────────────────────────────
COPY rl_worker/ ./rl_worker
COPY rl_worker/docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

CMD ["bash"]
