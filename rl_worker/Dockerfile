# ------------------------------------------------------------
# rl_worker/Dockerfile   (CPU • Python 3.11 • Oracle/OpenJDK 8)
# ------------------------------------------------------------
FROM python:3.11-slim

# ── 1. OS-level deps ────────────────────────────────────────
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        xvfb x11-utils xauth \
        libgl1-mesa-glx libxi6 libxrender1 libxrandr2 libxtst6 \
        libasound2 libfontconfig1 ffmpeg \
        build-essential python3-dev procps curl wget unzip git \
        xserver-xorg-core xserver-xorg-video-dummy x11-xserver-utils \
    && rm -rf /var/lib/apt/lists/*

# ── 2. JDK 8 (Temurin 8 u402) ───────────────────────────────
ENV JAVA_HOME=/opt/java8
RUN wget -q https://github.com/adoptium/temurin8-binaries/releases/download/jdk8u402-b06/OpenJDK8U-jre_x64_linux_hotspot_8u402b06.tar.gz \
    -O /tmp/jre8.tgz && \
    mkdir -p "$JAVA_HOME" && \
    tar -xzf /tmp/jre8.tgz --strip-components=1 -C "$JAVA_HOME" && \
    rm /tmp/jre8.tgz
ENV PATH="${JAVA_HOME}/bin:${PATH}"

ENV PYTHONPATH="/app/rl_worker:${PYTHONPATH}"
ENV DISPLAY=:99
ENV LIBGL_ALWAYS_SOFTWARE=1
WORKDIR /app

# ── 3. MineRL Patched wheel (キャッシュ層) ──────────────────
ARG MINERL_WHL=minerl_mirror-0.4.4-cp311-cp311-linux_x86_64.whl
RUN wget -q --continue --tries=5 \
      "https://github.com/danijar/minerl/releases/download/v0.4.4-patched/${MINERL_WHL}"

# ── 4. Python deps ──────────────────────────────────────────
COPY rl_worker/requirements.txt .
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir --no-deps "./${MINERL_WHL}" && \
    rm "${MINERL_WHL}"

# ── 5. App source & entrypoint ──────────────────────────────
COPY rl_worker/ ./rl_worker
COPY rl_worker/docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

CMD ["bash"]
