# --------------------------------------------------------------------
# Base image : CUDA 11.8 + Ubuntu 20.04 (headless)
# --------------------------------------------------------------------
FROM nvidia/cuda:11.8.0-base-ubuntu20.04

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
    python3-dev libncurses5-dev cmake && \
    rm -rf /var/lib/apt/lists/*

# ---- Maven ----------------------------------------------------------
RUN mkdir -p /opt/maven && \
    curl -fsSL https://archive.apache.org/dist/maven/maven-3/${MAVEN_VERSION}/binaries/apache-maven-${MAVEN_VERSION}-bin.tar.gz \
      | tar -xz -C /opt/maven && \
    ln -s /opt/maven/apache-maven-${MAVEN_VERSION}/bin/mvn /usr/bin/mvn

# --------------------------------------------------------------------
# Workspace
# --------------------------------------------------------------------
WORKDIR /workspace/minecraft-civ-ai/rl_worker
COPY . .

# ---- Python 3.11 ----------------------------------------------------
RUN add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update && apt-get install -y python3.11 python3.11-dev python3.11-distutils && \
    ln -sf /usr/bin/python3.11 /usr/bin/python && \
    ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python

# ---- Build‑tool versions (must precede cleanrl / gym install) -------
RUN pip install --no-cache-dir \
        pip==20.3.4 \
        setuptools==52.0.0 \
        wheel==0.36.2 \
        "packaging<21"

# ---- Common Python libs --------------------------------------------
RUN pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cu118 \
        'torch==2.2.2+cu118' \
        'networkx<3' \
        'stable-baselines3==1.6.2' \
        --no-build-isolation && \
    pip install --no-cache-dir 'cleanrl==0.4.8' --no-build-isolation

# ---- Utilities: ffmpeg for VideoRecorder --------------------------------
RUN apt-get update && apt-get install -y ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# ---- MineRL (Gym 0.19 固定) ----------------------------------------
RUN pip install --no-cache-dir \
        pip==20.3.4 \
        setuptools==52.0.0 \
        wheel==0.36.2 \
        'packaging<21' && \
    pip install --no-cache-dir --no-build-isolation \
        'gym==0.19.0' 'numpy==1.24.4' && \
    pip install --no-cache-dir \
        https://github.com/danijar/minerl/releases/download/v0.4.4-patched/minerl_mirror-0.4.4-cp311-cp311-linux_x86_64.whl

# ---- Entrypoint: start Xvfb ----------------------------------------
COPY docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh
ENV DISPLAY=:99
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["bash"]
