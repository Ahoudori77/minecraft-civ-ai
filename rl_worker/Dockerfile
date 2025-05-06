# rl_worker/Dockerfile
FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    DISPLAY=:99 \
    LIBGL_ALWAYS_SOFTWARE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# 1) システム依存ライブラリ
RUN apt-get update && apt-get install -y --no-install-recommends \
      xvfb libgl1-mesa-dri libx11-6 libxrandr2 \
      build-essential curl unzip zip git \
    && rm -rf /var/lib/apt/lists/*

# 2) Python パッケージをまとめたファイルを先に COPY→pip install
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# 3) ソース一式を COPY
COPY . /app

# 4) Xvfb 起動スクリプトをエントリポイント化
COPY docker-entrypoint.sh /app/docker-entrypoint.sh
RUN chmod +x /app/docker-entrypoint.sh

ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["bash"]
