# ------------------------------------------------------------------
# rl_worker: CPU-only, headless MineRL 学習環境
# ------------------------------------------------------------------
    FROM python:3.11-slim

    ENV DEBIAN_FRONTEND=noninteractive \
        DISPLAY=:99
    
    # ---- OS パッケージ ------------------------------------------------
    RUN apt-get update && apt-get install -y --no-install-recommends \
            openjdk-8-jdk-headless \
            xvfb x11-utils xauth \
            libgl1-mesa-glx ffmpeg \
            curl unzip git && \
        rm -rf /var/lib/apt/lists/*
    
    WORKDIR /app
    
    # ========== ① MineRL ホイールを先にだけ取得 ======================
    RUN curl -L -o minerl_mirror.whl \
        https://github.com/danijar/minerl/releases/download/v0.4.4-patched/minerl_mirror-0.4.4-cp311-cp311-linux_x86_64.whl
    
    # ========== ② 依存ライブラリをインストール =========================
    COPY rl_worker/requirements.txt ./requirements.txt
    RUN pip install --no-cache-dir --use-deprecated=legacy-resolver -r requirements.txt
    
    # ========== ③ MineRL を依存解決なしでインストール ================
    RUN pip install --no-cache-dir --no-deps ./minerl_mirror.whl && \
        rm minerl_mirror.whl
    
    # ---- アプリケーションコード --------------------------------------
    COPY rl_worker/ ./rl_worker
    COPY rl_worker/docker-entrypoint.sh /usr/local/bin/
    RUN chmod +x /usr/local/bin/docker-entrypoint.sh
    
    ENTRYPOINT ["docker-entrypoint.sh"]
    CMD ["python", "-m", "rl_worker.train"]
    