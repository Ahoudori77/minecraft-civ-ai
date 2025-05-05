# --------------------------------------------------------------------
# Base image : CUDA 11.8 + Ubuntu 22.04 (headless)
# --------------------------------------------------------------------
    FROM nvidia/cuda:11.8.0-base-ubuntu22.04

    ENV DEBIAN_FRONTEND=noninteractive \
        DISPLAY=:99 \
        LIBGL_ALWAYS_SOFTWARE=1 \
        PYTHONDONTWRITEBYTECODE=1 \
        PYTHONUNBUFFERED=1
    
    # ---- OS & X11 -------------------------------------------------------
    RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-dev python3-pip \
        openjdk-8-jdk-headless \
        xvfb x11-utils xserver-xorg-video-dummy \
        libgl1-mesa-dri libosmesa6 mesa-utils \
        libxi6 libxrender1 libxrandr2 libxfixes3 libxcursor1 libxinerama1 \
        curl unzip zip git build-essential ffmpeg && \
        ln -sf /usr/bin/python3.11 /usr/bin/python && \
        ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
        python -m pip install --upgrade pip && \
        rm -rf /var/lib/apt/lists/*
    
    # ---- Python deps ----------------------------------------------------
    COPY rl_worker/requirements.txt /tmp/requirements.txt
    RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu118 \
            -r /tmp/requirements.txt
    
    # ---- Maven (for Malmo build helper) ---------------------------------
    ARG MAVEN_VERSION=3.6.3
    RUN curl -fsSL https://archive.apache.org/dist/maven/maven-3/${MAVEN_VERSION}/binaries/apache-maven-${MAVEN_VERSION}-bin.tar.gz \
            | tar -xz -C /opt && \
        ln -s /opt/apache-maven-${MAVEN_VERSION}/bin/mvn /usr/bin/mvn
    
    # ---- Project code ---------------------------------------------------
    WORKDIR /workspace/minecraft-civ-ai
    COPY . .
    
    # ---- Entrypoint：Xvfb を起動して RL ワーカー実行 -------------------
    COPY rl_worker/docker-entrypoint.sh /usr/local/bin/
    RUN chmod +x /usr/local/bin/docker-entrypoint.sh
    ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
    