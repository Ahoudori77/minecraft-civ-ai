# ------------------------------------------------------------
# rl_worker/Dockerfile   (CPU 版・Python 3.11)
# ------------------------------------------------------------
    FROM python:3.11-slim

    # ---- OS 依存ライブラリ -------------------------------------
    RUN apt-get update && \
        apt-get install -y --no-install-recommends \
            openjdk-17-jdk-headless \
            xvfb x11-utils xauth \
            libgl1-mesa-glx ffmpeg \
            curl wget unzip git && \
        rm -rf /var/lib/apt/lists/*
    
    ENV DISPLAY=:99
    ENV LIBGL_ALWAYS_SOFTWARE=1
    
    WORKDIR /app
    
    # ---- MineRL の巨大 whl を先にキャッシュ --------------------
    ARG MINERL_WHL=https://github.com/danijar/minerl/releases/download/v0.4.4-patched/minerl_mirror-0.4.4-cp311-cp311-linux_x86_64.whl
    RUN wget -q --show-progress --continue --tries=5 "$MINERL_WHL" -O minerl_mirror.whl
    
    # ---- Python 依存関係 ---------------------------------------
    COPY rl_worker/requirements.txt ./requirements.txt
    RUN python -m pip install --upgrade pip && \
        pip install --no-cache-dir -r requirements.txt && \
        pip install --no-cache-dir --no-deps minerl_mirror.whl && \
        rm minerl_mirror.whl
    
    # ---- ソースコード & エントリポイント -----------------------
    COPY rl_worker/ ./rl_worker
    COPY rl_worker/docker-entrypoint.sh /usr/local/bin/
    RUN chmod +x /usr/local/bin/docker-entrypoint.sh
    
    CMD ["bash"]
    