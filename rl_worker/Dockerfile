# ── CPU only 版ベースイメージに切り替え
FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    DISPLAY=:99 \
    LIBGL_ALWAYS_SOFTWARE=1

# Xvfb＋Java＋ビルドツール
RUN apt-get update && apt-get install -y --no-install-recommends \
      openjdk-8-jdk-headless \
      xvfb x11-utils x11-xserver-utils xserver-xorg-video-dummy \
      xfonts-base xfonts-75dpi xfonts-100dpi \
      libgl1-mesa-dri libglx-mesa0 libosmesa6 mesa-utils \
      libxi6 libxrender1 libxrandr2 libxfixes3 libxcursor1 libxinerama1 \
      git build-essential cmake unzip zip curl \
    && rm -rf /var/lib/apt/lists/*

# Python 依存は requirements.txt で一括インストール
COPY rl_worker/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# ソースをコピー
WORKDIR /workspace/minecraft-civ-ai/rl_worker

# 依存ファイルをそこから参照
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# ソース一式
COPY . .

# entrypoint
COPY docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["bash"]