# —————————————————————————————
# 1. ベースイメージ
#   NVIDIA の CUDA 11.8 ＋ Ubuntu20.04
FROM nvidia/cuda:11.8.0-base-ubuntu20.04

# —————————————————————————————
# 2. 非対話モードにして、不要なプロンプトを抑制
ENV DEBIAN_FRONTEND=noninteractive

# —————————————————————————————
# 3. システム依存ライブラリをインストール
#    Java8 やビルドツール、X11 関連のライブラリなど
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      binfmt-support \        # マルチアーキ使うなら
      openjdk-8-jdk-headless \# Java 8（headless）
      maven unzip zip git \   # Java プロジェクト用
      build-essential cmake \ # C/C++ ビルドツール
      libgl1-mesa-glx libxi6 libxrender1 libxrandr2 libxfixes3 \
      libxcursor1 libxinerama1 xvfb && \
    rm -rf /var/lib/apt/lists/*

# —————————————————————————————
# 4. 作業ディレクトリを設定
WORKDIR /workspace/minecraft-civ-ai/rl_worker

# —————————————————————————————
# 5. 自分のコードをイメージ内にコピー
COPY rl_worker/ .

# —————————————————————————————
# 6. Python と pip の準備＋依存パッケージをインストール
RUN apt-get update && \
    apt-get install -y python3-pip python3-dev && \
    pip3 install --no-cache-dir \
      'networkx<3' \
      gym==0.21.0 \
      torch==2.2.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118 && \
    rm -rf /root/.cache

# —————————————————————————————
# 7. コンテナ起動時のデフォルトコマンド
CMD ["bash"]
