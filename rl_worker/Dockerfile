# --------------------------------------------------------------------
# Base image : CUDA 11.8 + Ubuntu 22.04 (headless)
# --------------------------------------------------------------------
    FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

    ENV DEBIAN_FRONTEND=noninteractive \
        MAVEN_VERSION=3.6.3 \
        DISPLAY=:99 \
        LIBGL_ALWAYS_SOFTWARE=1
    
    # ---- OS & X11 -------------------------------------------------------
    RUN apt-get update && apt-get install -y --no-install-recommends \
        openjdk-8-jdk-headless \
        xvfb x11-utils x11-xserver-utils xserver-xorg-video-dummy \
        xfonts-base xfonts-75dpi xfonts-100dpi \
        libgl1-mesa-dri libglx-mesa0 libosmesa6 mesa-utils \
        libxi6 libxrender1 libxrandr2 libxfixes3 libxcursor1 libxinerama1 \
        software-properties-common curl unzip zip git build-essential \
        python3-dev libncurses5-dev cmake ffmpeg && \
        rm -rf /var/lib/apt/lists/*
    
    # ---- Maven ----------------------------------------------------------
    RUN mkdir -p /opt/maven && \
        curl -fsSL https://archive.apache.org/dist/maven/maven-3/${MAVEN_VERSION}/binaries/apache-maven-${MAVEN_VERSION}-bin.tar.gz \
          | tar -xz -C /opt/maven && \
        ln -s /opt/maven/apache-maven-${MAVEN_VERSION}/bin/mvn /usr/bin/mvn
    
    # --------------------------------------------------------------------
    # Workdir & Copy
    # --------------------------------------------------------------------
    WORKDIR /workspace/minecraft-civ-ai/rl_worker
    
    # ❶ Python の依存をまとめたファイルを先にコピー
    COPY requirements.txt /tmp/requirements.txt
    
    # ❷ Xvfb 起動スクリプトコピー
    COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
    RUN chmod +x /usr/local/bin/docker-entrypoint.sh
    
    # ❸ ソース一式コピー
    COPY . .
    
    # ---- Python 3.11 + pip/setuptools 固定 -----------------------------
    RUN apt-get update && apt-get install -y --no-install-recommends \
            gnupg curl ca-certificates \
        && curl -fsSL "https://keyserver.ubuntu.com/pks/lookup?op=get&search=0x6A755776" \
           | gpg --dearmor -o /usr/share/keyrings/deadsnakes-archive.gpg \
        && echo "deb [signed-by=/usr/share/keyrings/deadsnakes-archive.gpg] http://ppa.launchpad.net/deadsnakes/ppa/ubuntu focal main" \
           > /etc/apt/sources.list.d/deadsnakes.list \
        && apt-get update \
        && apt-get install -y python3.11 python3.11-dev python3.11-distutils \
        && ln -sf /usr/bin/python3.11 /usr/bin/python \
        && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
        && rm -rf /var/lib/apt/lists/*
    
    # ---- Python ライブラリ一括インストール -----------------------------
    RUN pip install --no-cache-dir -r /tmp/requirements.txt
    
    # ---- Entrypoint: start Xvfb ----------------------------------------
    ENV DISPLAY=:99
    ENTRYPOINT ["docker-entrypoint.sh"]
    CMD ["python", "train.py"]
    